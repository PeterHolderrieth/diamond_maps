import json
import os
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)

from diffusers.image_processor import PipelineImageInput
from diffusers.utils import is_torch_xla_available, logging
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


def freeze_params(params: Iterable[torch.nn.Parameter]) -> None:
    for param in params:
        param.requires_grad = False


def clip_img_transform(size: int = 224):
    return Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def compute_b_coefficient_from_t(
    t_norm: torch.Tensor, eps: float = 1e-6, max_abs_b: float = 20.0
) -> torch.Tensor:
    """Compute FLUX guidance coefficient with clipped magnitude.

    Using rectified-flow schedule `alpha_t = 1 - t`, `sigma_t = t`, the
    implementation applies guidance as `u_guided = u - b_t * grad`, where:
    `b_t = t / (1 - t)`.

    Args:
        t_norm: Normalized timestep tensor in `[0, 1]`.
        eps: Numerical clamp for denominator.
        max_abs_b: Maximum allowed magnitude for `b_t`.

    Returns:
        Clipped guidance coefficient tensor.
    """
    alpha = (1 - t_norm).clamp(min=eps)
    b = t_norm / alpha
    return b.clamp(min=0.0, max=max_abs_b)


def transformer_param_dtype(module: torch.nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return torch.float32


def vae_decode_dtype(module: torch.nn.Module) -> torch.dtype:
    if hasattr(module, "decoder"):
        for param in module.decoder.parameters():
            return param.dtype
    for param in module.parameters():
        return param.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return torch.float32


def prompt_slug(text: str) -> str:
    return (
        "".join(c if c.isalnum() else "-" for c in text).strip("-").lower()[:80]
        or "prompt"
    )


def cuda_memory_stats(device: torch.device) -> str:
    """Return human-readable CUDA memory stats for step-level debugging."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return "mem=cpu"
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    allocated_gb = torch.cuda.memory_allocated(device_index) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(device_index) / (1024**3)
    peak_gb = torch.cuda.max_memory_allocated(device_index) / (1024**3)
    return (
        f"mem_alloc={allocated_gb:.2f}GB "
        f"mem_reserved={reserved_gb:.2f}GB "
        f"mem_peak={peak_gb:.2f}GB"
    )


class FluxFlowMapFMTTPipeline(FluxPipeline):
    def compute_model_prediction(self, latents, t, t_next, cond_kwargs):
        # Query model twice:
        # 1) t -> t_next for scheduler update (noise/velocity prediction).
        # 2) t -> 0 for direct x0 prediction in flow-map mode.
        model_dtype = transformer_param_dtype(self.transformer)
        model_latents = latents.to(model_dtype)
        timestep = t.expand(latents.shape[0]).to(model_dtype)
        timestep2 = t_next.expand(latents.shape[0]).to(model_dtype)
        guidance = cond_kwargs["guidance"]
        if guidance is not None:
            guidance = guidance.to(model_dtype)

        def run_model(
            target_t2: torch.Tensor,
            pooled_prompt_embeds: torch.Tensor,
            prompt_embeds: torch.Tensor,
            text_ids: torch.Tensor,
            image_embeds: Optional[List[torch.Tensor]],
            cache_key: str,
        ) -> torch.Tensor:
            joint_attention_kwargs = (
                cond_kwargs["joint_attention_kwargs"].copy()
                if cond_kwargs["joint_attention_kwargs"] is not None
                else {}
            )
            if image_embeds is not None:
                joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

            with self.transformer.cache_context(cache_key):
                return self.transformer(
                    hidden_states=model_latents,
                    timestep=torch.stack([timestep / 1000, target_t2 / 1000], dim=-1),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds.to(model_dtype),
                    encoder_hidden_states=prompt_embeds.to(model_dtype),
                    txt_ids=text_ids,
                    img_ids=cond_kwargs["latent_image_ids"],
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

        noise_pred = run_model(
            timestep2,
            cond_kwargs["pooled_prompt_embeds"],
            cond_kwargs["prompt_embeds"],
            cond_kwargs["text_ids"],
            cond_kwargs["image_embeds"],
            "cond",
        )
        final_timestep = torch.zeros_like(timestep2)
        v0_pred = run_model(
            final_timestep,
            cond_kwargs["pooled_prompt_embeds"],
            cond_kwargs["prompt_embeds"],
            cond_kwargs["text_ids"],
            cond_kwargs["image_embeds"],
            "cond",
        )

        if cond_kwargs["do_true_cfg"]:
            neg_noise_pred = run_model(
                timestep2,
                cond_kwargs["negative_pooled_prompt_embeds"],
                cond_kwargs["negative_prompt_embeds"],
                cond_kwargs["negative_text_ids"],
                cond_kwargs["negative_image_embeds"],
                "uncond",
            )
            noise_pred = neg_noise_pred + cond_kwargs["true_cfg_scale"] * (
                noise_pred - neg_noise_pred
            )

            neg_v0_pred = run_model(
                final_timestep,
                cond_kwargs["negative_pooled_prompt_embeds"],
                cond_kwargs["negative_prompt_embeds"],
                cond_kwargs["negative_text_ids"],
                cond_kwargs["negative_image_embeds"],
                "uncond",
            )
            v0_pred = neg_v0_pred + cond_kwargs["true_cfg_scale"] * (
                v0_pred - neg_v0_pred
            )

        x0_hat = latents - (t / 1000) * v0_pred
        noise_pred = noise_pred.to(latents.dtype)
        return x0_hat, noise_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # REWARD KWARGS
        reward_scale: float = 0.0,
        reward_fns: Callable = None,
        gradient_norm_scale: float = 10.0,
        num_guidance_steps: int = 5,
        guidance_start_step: int = 1,
        save_intermediate_imgs: bool = True,
        save_path: Optional[str] = None,
        select_best_scored_image: bool = True,
        save_top_reward_images: int = 0,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        if reward_fns and not hasattr(self, "img_transform"):
            self.img_transform = clip_img_transform()

        if reward_scale > 0:
            freeze_params(self.vae.parameters())
            freeze_params(self.transformer.parameters())
            if self.text_encoder is not None:
                freeze_params(self.text_encoder.parameters())
            if self.text_encoder_2 is not None:
                freeze_params(self.text_encoder_2.parameters())
            self.vae.eval()
            self.transformer.eval()

        if num_guidance_steps < 0:
            raise ValueError(
                f"num_guidance_steps must be >= 0, got {num_guidance_steps}."
            )
        if guidance_start_step < 0:
            raise ValueError(
                f"guidance_start_step must be >= 0, got {guidance_start_step}."
            )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # STRICT REQUIREMENT: No .get() allowed. Will throw KeyError if "scale" is missing.
        lora_scale = (
            self.joint_attention_kwargs["scale"]
            if self.joint_attention_kwargs is not None
            else None
        )

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        negative_text_ids = None
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        if (
            hasattr(self.scheduler.config, "use_flow_sigmas")
            and self.scheduler.config.use_flow_sigmas
        ):
            sigmas = None

        image_seq_len = latents.shape[1]
        # STRICT REQUIREMENT: No .get() allowed. Will throw AttributeError if missing.
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)
        guidance_start_idx = int(guidance_start_step)
        guidance_end_step = min(
            self._num_timesteps,
            guidance_start_idx + int(num_guidance_steps),
        )
        if (
            reward_scale > 0
            and num_guidance_steps > 0
            and guidance_start_idx >= self._num_timesteps
        ):
            logger.warning(
                "guidance_start_step=%s is >= number of timesteps=%s. "
                "Run will be unguided.",
                guidance_start_step,
                self._num_timesteps,
            )

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            ).expand(latents.shape[0])
        else:
            guidance = None

        def save_latent_frame(latents_tensor: torch.Tensor, out_file: str) -> None:
            decode_dtype = vae_decode_dtype(self.vae)
            unpacked = self._unpack_latents(
                latents_tensor.to(decode_dtype),
                height,
                width,
                self.vae_scale_factor,
            )
            unpacked = (
                unpacked / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            decoded = self.vae.decode(unpacked, return_dict=False)[0]
            image = self.image_processor.postprocess(decoded, output_type="pil")[0]
            image.save(out_file)

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None
            and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [
                negative_ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [
                ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        cond_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
            "latent_image_ids": latent_image_ids,
            "guidance": guidance,
            "joint_attention_kwargs": self._joint_attention_kwargs,
            "do_true_cfg": do_true_cfg,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "negative_text_ids": negative_text_ids,
            "image_embeds": image_embeds,
            "negative_image_embeds": negative_image_embeds,
        }

        if save_intermediate_imgs:
            prompt_text = prompt[0] if isinstance(prompt, list) else prompt
            if prompt_text is None:
                prompt_text = "prompt"
            if save_path is None:
                save_path = f"flux_fmtt/{prompt_slug(prompt_text)}"
            xt_dir = os.path.join(save_path, "intermediates", "x_t")
            x0_dir = os.path.join(save_path, "intermediates", "x0_hat")
            os.makedirs(xt_dir, exist_ok=True)
            os.makedirs(x0_dir, exist_ok=True)

        best_reward = float("-inf")
        best_latent_output: torch.Tensor | None = None
        best_image_pred: torch.Tensor | None = None
        top_reward_limit = max(int(save_top_reward_images), 0)
        top_reward_candidates: list[dict[str, Any]] = []

        decode_dtype = vae_decode_dtype(self.vae)

        def decode_packed_latents(latent_tensor: torch.Tensor) -> torch.Tensor:
            """Decode packed FLUX latents into an image tensor in `[0, 1]`."""
            unpacked_latents = self._unpack_latents(
                latent_tensor,
                height,
                width,
                self.vae_scale_factor,
            )
            unpacked_latents = (
                unpacked_latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image_pred = self.vae.decode(
                unpacked_latents.to(decode_dtype),
                return_dict=False,
            )[0]
            return (image_pred / 2 + 0.5).clamp(0, 1)

        def score_decoded_image(image_pred: torch.Tensor) -> float:
            """Score one decoded image tensor with the configured reward model."""
            preprocessed_img = self.img_transform(image_pred)
            raw_reward = 0.0
            for reward_fn in reward_fns:
                raw_reward = (
                    raw_reward
                    + reward_fn(preprocessed_img, prompt, image_pred)
                    * reward_fn.weighting
                )
            return float(raw_reward.detach().mean().item())

        def maybe_update_best_candidate(
            candidate_reward: float,
            candidate_latent_output: torch.Tensor,
            candidate_image_pred: torch.Tensor,
            candidate_source: str,
        ) -> None:
            """Track the best reward-scored FLUX candidate seen so far."""
            nonlocal best_reward, best_latent_output, best_image_pred
            if candidate_reward > best_reward:
                best_reward = candidate_reward
                best_latent_output = candidate_latent_output.detach().clone()
                best_image_pred = candidate_image_pred.detach().clone()
                print(
                    "[FluxFMTT] New best reward-scored candidate: "
                    f"{best_reward:.4f} [source={candidate_source}]"
                )

        def maybe_update_top_candidates(
            candidate_reward: float,
            candidate_image_pred: torch.Tensor,
            candidate_source: str,
        ) -> None:
            """Keep the top-K reward-scored image predictions for inspection."""
            nonlocal top_reward_candidates
            if top_reward_limit <= 0:
                return
            top_reward_candidates.append(
                {
                    "reward": float(candidate_reward),
                    "source": str(candidate_source),
                    "image_pred": candidate_image_pred.detach().cpu().clone(),
                }
            )
            top_reward_candidates = sorted(
                top_reward_candidates,
                key=lambda candidate: float(candidate["reward"]),
                reverse=True,
            )[:top_reward_limit]

        def maybe_track_reward_candidate(
            candidate_reward: float,
            candidate_latent_output: torch.Tensor,
            candidate_image_pred: torch.Tensor,
            candidate_source: str,
        ) -> None:
            """Record both the best candidate and the top-K reward images."""
            maybe_update_best_candidate(
                candidate_reward,
                candidate_latent_output,
                candidate_image_pred,
                candidate_source,
            )
            maybe_update_top_candidates(
                candidate_reward,
                candidate_image_pred,
                candidate_source,
            )

        def format_best_image_output(
            image_pred: torch.Tensor,
            requested_output_type: str,
        ) -> Any:
            """Convert a tracked best image in `[0, 1]`` to the requested output type."""
            image_pred = image_pred.detach().cpu()
            if requested_output_type == "pt":
                return image_pred
            image_np = self.image_processor.pt_to_numpy(image_pred)
            if requested_output_type == "np":
                return image_np
            if requested_output_type == "pil":
                return self.image_processor.numpy_to_pil(image_np)
            raise ValueError(
                f"Unsupported output_type for best image: {requested_output_type}"
            )

        def write_top_reward_candidates(save_root: str | None) -> None:
            """Write the top reward-ranked decoded images and metadata to disk."""
            if (
                save_root is None
                or top_reward_limit <= 0
                or len(top_reward_candidates) == 0
            ):
                return

            top_dir = os.path.join(save_root, "top_reward_images")
            os.makedirs(top_dir, exist_ok=True)
            metadata_rows: list[dict[str, Any]] = []
            for rank, candidate in enumerate(top_reward_candidates, start=1):
                source_fragment = (
                    str(candidate["source"]).replace("/", "_").replace(" ", "_")
                )
                image_np = self.image_processor.pt_to_numpy(candidate["image_pred"])
                image = self.image_processor.numpy_to_pil(image_np)[0]
                output_path = os.path.join(
                    top_dir,
                    f"rank_{rank:02d}_{source_fragment}.png",
                )
                image.save(output_path)
                metadata_rows.append(
                    {
                        "rank": rank,
                        "reward": float(candidate["reward"]),
                        "source": str(candidate["source"]),
                        "image_path": output_path,
                    }
                )

            metadata_path = os.path.join(top_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(metadata_rows, handle, indent=2)

        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            self._current_timestep = t

            if i == len(timesteps) - 1:
                timestep2 = torch.zeros_like(t)
            else:
                timestep2 = timesteps[i + 1]

            is_guided_step = (
                reward_scale > 0 and guidance_start_idx <= i < guidance_end_step
            )
            if is_guided_step:
                latents = latents.detach().requires_grad_(True)
                with torch.enable_grad():
                    x0_hat, noise_pred = self.compute_model_prediction(
                        latents, t, timestep2, cond_kwargs
                    )

                    # Differentiable decoding for FMTT guidance updates.
                    image_pred = decode_packed_latents(x0_hat)
                    preprocessed_img = self.img_transform(image_pred)

                    raw_reward = 0.0
                    for reward_fn in reward_fns:
                        raw_reward = (
                            raw_reward
                            + reward_fn(preprocessed_img, prompt, image_pred)
                            * reward_fn.weighting
                        )

                    reward_value = float(raw_reward.detach().item())
                    maybe_track_reward_candidate(
                        reward_value,
                        x0_hat,
                        image_pred,
                        "guided_main_path",
                    )

                    grad = torch.autograd.grad(
                        outputs=(raw_reward * reward_scale).sum(),
                        inputs=latents,
                        retain_graph=False,
                        create_graph=False,
                    )[0]

                grad_norm = grad.norm()
                # if gradient_norm_scale < grad_norm:
                grad = grad / grad_norm * gradient_norm_scale

                mem_stats = cuda_memory_stats(latents.device)
                print(
                    f"Step {i + 1} (FMTT): Reward={raw_reward.item():.4f}, "
                    f"||∇R||={grad_norm:.4f}, ||velocity||={noise_pred.norm().item():.4f}, "
                    f"{mem_stats}"
                )

                t_norm = (t / 1000.0).view(-1, 1, 1)
                b_3d = compute_b_coefficient_from_t(t_norm).view(-1, 1, 1)
                noise_pred = noise_pred - b_3d * grad

                del preprocessed_img, raw_reward, image_pred, grad
            else:
                with torch.no_grad():
                    x0_hat, noise_pred = self.compute_model_prediction(
                        latents, t, timestep2, cond_kwargs
                    )

            if save_intermediate_imgs:
                save_latent_frame(latents.detach(), f"{xt_dir}/step_{i:03d}.png")
                save_latent_frame(x0_hat.detach(), f"{x0_dir}/step_{i:03d}.png")

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {
                    k: locals()[k] for k in callback_on_step_end_tensor_inputs
                }
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if XLA_AVAILABLE:
                xm.mark_step()

        if reward_fns:
            final_image_pred = decode_packed_latents(latents)
            final_reward = score_decoded_image(final_image_pred)
            maybe_track_reward_candidate(
                final_reward,
                latents,
                final_image_pred,
                "final_sample",
            )

        write_top_reward_candidates(save_path)

        self._current_timestep = None

        if output_type == "latent":
            if select_best_scored_image and best_latent_output is not None:
                image = best_latent_output
            else:
                image = latents
        elif select_best_scored_image and best_image_pred is not None:
            image = format_best_image_output(best_image_pred, output_type)
        else:
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents.to(decode_dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)
