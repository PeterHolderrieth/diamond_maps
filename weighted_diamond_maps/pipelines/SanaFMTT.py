# Copyright 2024 PixArt-Sigma Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from typing import Union, List, Optional, Dict, Any, Callable

import torch
from PIL import Image
from transformers import Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)

from diffusers.loaders import SanaLoraLoaderMixin
from diffusers.models import AutoencoderDC, SanaTransformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers import SanaSprintPipeline

logger = logging.get_logger(__name__)


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def freeze_params(params):
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


class FMTTSana(SanaSprintPipeline):
    r"""
    Pipeline for text-to-image generation using [SANA-Sprint] with FMTT (Flow Matching Time-Travel / Test-Time).
    This uses the naive denoiser approximation: \nabla_{x_t} r(\hat{x}_0(x_t)).
    """

    def __init__(
        self,
        tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
        text_encoder: Gemma2PreTrainedModel,
        vae: AutoencoderDC,
        transformer: SanaTransformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        freeze_params(vae.parameters())
        freeze_params(transformer.parameters())
        freeze_params(text_encoder.parameters())
        vae.eval()
        transformer.eval()
        text_encoder.eval()
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        self.img_transform = clip_img_transform()

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
        lora_scale: Optional[float] = None,
    ):
        if device is None:
            device = self._execution_device

        dtype = (
            self.transformer.dtype
            if self.transformer is not None
            else (self.text_encoder.dtype if self.text_encoder is not None else None)
        )

        if lora_scale is not None and isinstance(self, SanaLoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        max_length = max_sequence_length
        select_index = [0] + list(range(-max_length + 1, 0))

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                device=device,
                dtype=dtype,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=complex_human_instruction,
            )
            prompt_embeds = prompt_embeds[:, select_index]
            prompt_attention_mask = prompt_attention_mask[:, select_index]

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        if self.text_encoder is not None:
            if isinstance(self, SanaLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, prompt_attention_mask

    def apply(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 1,
        timesteps: List[int] = None,
        max_timesteps: float = 1.57080,
        intermediate_timesteps: float = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: int = 1024,
        width: int = 1024,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 300,
        reward_scale: float = 1.0,
        reward_fns: Optional[List[Callable]] = None,
        gradient_norm_scale: float = 10.0,
        save_path: str | None = None,
        save_outputs: bool = False,
        save_intermediate_imgs: bool = True,
        guidance_start_step: int = 1,
        num_guidance_steps: int = 5,
        jump_to_end_after_guidance: bool = False,
        complex_human_instruction: List[str] = [
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ],
    ):
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            max_timesteps=max_timesteps,
            intermediate_timesteps=intermediate_timesteps,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if guidance_start_step < 0:
            raise ValueError(
                f"guidance_start_step must be >= 0, got {guidance_start_step}."
            )
        if num_guidance_steps < 0:
            raise ValueError(
                f"num_guidance_steps must be >= 0, got {num_guidance_steps}."
            )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = (
            self.attention_kwargs.get("scale", None)
            if self.attention_kwargs is not None
            else None
        )

        prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
            lora_scale=lora_scale,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=None,
            max_timesteps=max_timesteps,
            intermediate_timesteps=intermediate_timesteps,
        )
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        latents = latents * self.scheduler.config.sigma_data
        sigma_d = self.scheduler.config.sigma_data

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0]).to(prompt_embeds.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale

        transformer_dtype = self.transformer.dtype
        self._num_timesteps = len(timesteps)

        x0_hat_history, latents_history, reward_history = [], [], []
        best_reward = float("-inf")
        best_image_tensor: torch.Tensor | None = None

        def maybe_update_best_candidate(
            candidate_reward: float,
            candidate_image_tensor: torch.Tensor,
            candidate_source: str,
        ) -> None:
            """Track the best reward-scored SANA FMTT candidate seen so far."""
            nonlocal best_reward, best_image_tensor
            if candidate_reward > best_reward:
                best_reward = candidate_reward
                best_image_tensor = candidate_image_tensor.detach().clone()
                print(
                    "[SanaFMTT] New best reward-scored candidate: "
                    f"{best_reward:.4f} [source={candidate_source}]"
                )

        if save_path is None:
            save_path = f"sanabase/fmtt_aesth/{prompt}/gradnorm{gradient_norm_scale}_reward_scale{reward_scale}_guidance_steps{num_guidance_steps}"
        if save_outputs or save_intermediate_imgs:
            os.makedirs(save_path, exist_ok=True)

        def tau_from_s(s):
            return torch.sin(s) / (torch.sin(s) + torch.cos(s))

        def r_from_tau(tau):
            return torch.sqrt(tau * tau + (1.0 - tau) * (1.0 - tau))

        def compute_b_coefficient_from_s(
            s: torch.Tensor, sigma_d: float, eps: float = 1e-6, max_abs_b: float = 3.0
        ) -> torch.Tensor:
            sin_s = torch.sin(s)
            cos_s = torch.cos(s).clamp(min=eps)
            b = (sigma_d**2) * (sin_s**3 / cos_s - sin_s * cos_s)
            return b.clamp(max=max_abs_b)

        def predict_x0_hat(
            latents_in: torch.Tensor,
            s_in: torch.Tensor,
        ) -> torch.Tensor:
            """Predict the clean-image estimate for the given latent state."""
            tau_in = tau_from_s(s_in).clamp(1e-6, 1 - 1e-6)
            tau_in_4d = tau_in.view(-1, 1, 1, 1)
            r_in = r_from_tau(tau_in)
            cos_in = torch.cos(s_in).view(-1, 1, 1, 1)
            sin_in = torch.sin(s_in).view(-1, 1, 1, 1)
            latents_model_input = latents_in / sigma_d
            x_tau = latents_model_input * r_in
            model_pred = self.transformer(
                x_tau.to(dtype=transformer_dtype),
                encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                encoder_attention_mask=prompt_attention_mask,
                guidance=guidance,
                timestep=tau_in.to(dtype=transformer_dtype),
                return_dict=False,
                attention_kwargs=self.attention_kwargs,
            )[0]
            return (cos_in - sin_in * (1 - 2 * tau_in_4d)) * latents_in - sin_in * (
                1 - 2 * tau_in_4d + 2 * tau_in_4d**2
            ) * sigma_d * model_pred / r_in

        guidance_end_step = min(
            num_inference_steps,
            guidance_start_step + num_guidance_steps,
        )

        for i in range(num_inference_steps):
            s_curr = timesteps[i]
            s_next = timesteps[i + 1]

            s_b = s_curr.expand(latents.shape[0]).to(torch.float32)
            s_next_b = s_next.expand(latents.shape[0]).to(torch.float32)

            cos_s = torch.cos(s_b).view(-1, 1, 1, 1)
            sin_s = torch.sin(s_b).view(-1, 1, 1, 1)

            tau = tau_from_s(s_b).clamp(1e-6, 1 - 1e-6)
            tau_4d = tau.view(-1, 1, 1, 1)
            r = r_from_tau(tau)

            # Only do guidance if we are within the specified steps and have a reward function
            do_guidance = (
                reward_scale > 0
                and reward_fns is not None
                and guidance_start_step <= i < guidance_start_step + num_guidance_steps
            )

            if do_guidance:
                latents = latents.detach().requires_grad_(True)
                context_mngr = torch.enable_grad()
            else:
                context_mngr = torch.no_grad()

            with context_mngr:
                latents_model_input = latents / sigma_d
                x_tau = latents_model_input * r

                model_pred = self.transformer(
                    x_tau.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=guidance,
                    timestep=tau.to(dtype=transformer_dtype),
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                )[0]

                # Predict clean image \hat{x}_0
                x0_hat = (cos_s - sin_s * (1 - 2 * tau_4d)) * latents - sin_s * (
                    1 - 2 * tau_4d + 2 * tau_4d**2
                ) * sigma_d * model_pred / r

                if do_guidance:
                    # Decode to image space
                    image = self.vae.decode(
                        (x0_hat / sigma_d).to(self.vae.dtype)
                        / self.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                    image = (image / 2 + 0.5).clamp(0, 1)
                    preprocessed_img = self.img_transform(image)

                    # Calculate reward
                    raw_reward = 0.0
                    for reward_fn in reward_fns:
                        reward_val = reward_fn(preprocessed_img, prompt, image)
                        raw_reward = raw_reward + reward_val * reward_fn.weighting

                    reward_value = float(raw_reward.detach().item())
                    maybe_update_best_candidate(
                        reward_value,
                        image,
                        "guided_main_path",
                    )

                    # Backpropagate to get \nabla_{x_t} r(\hat{x}_0)
                    reward_grad = torch.autograd.grad(
                        outputs=(raw_reward * reward_scale).sum(),
                        inputs=latents,
                        retain_graph=False,
                        create_graph=False,
                    )[0]

            # Detach tensors so we can safely update them without PyTorch complaining about leaf variables
            if do_guidance:
                latents = latents.detach()
                model_pred = model_pred.detach()
                x0_hat = x0_hat.detach()

                reward_grad_norm = reward_grad.norm().clamp_min(1e-8)
                grad_norm = reward_grad_norm.item()
                # Previous SANA behavior only clipped large reward gradients:
                # if gradient_norm_scale < grad_norm:
                #     reward_grad = reward_grad / grad_norm * gradient_norm_scale
                reward_grad = reward_grad / reward_grad_norm * gradient_norm_scale

                reward_history.append(raw_reward.item())

            # Calculate standard Flow Matching Velocity
            k = cos_s + sin_s
            half_s = (s_b / 2).view(-1, 1, 1, 1)
            tan_half_s = torch.tan(half_s)

            C1 = k * tan_half_s + cos_s - sin_s
            C2 = k * r

            velocity = C1 * latents + C2 * sigma_d * model_pred

            # Apply Guidance Modification
            if do_guidance:
                b = compute_b_coefficient_from_s(s_b, sigma_d)
                velocity = velocity - b * reward_grad
                print(
                    f"Step {i + 1} (FMTT): Reward={raw_reward.item():.4f}, ||∇R||={grad_norm:.4f}, ||velocity||={velocity.norm().item():.4f}"
                )

            latents_t = latents.detach().clone()

            # Euler step
            s_curr_4d = s_b.view(-1, 1, 1, 1)
            s_next_4d = s_next_b.view(-1, 1, 1, 1)
            ds = s_next_4d - s_curr_4d  # This is negative (going s_max → 0)
            latents = latents + ds * velocity

            jumped_to_end = False
            if (
                jump_to_end_after_guidance
                and do_guidance
                and (i + 1) == guidance_end_step
                and guidance_end_step < num_inference_steps
            ):
                with torch.no_grad():
                    x0_hat = predict_x0_hat(latents, s_next_b).detach()
                latents_t = latents.detach().clone()
                jumped_to_end = True

            if save_intermediate_imgs:
                x0_hat_history.append(x0_hat.detach().clone())
                latents_history.append(latents_t)

            if jumped_to_end:
                break

        # ========== SAVE INTERMEDIATE IMAGES ==========
        if save_intermediate_imgs and len(x0_hat_history) > 0:
            x0_dir = f"{save_path}/intermediates/x0_hat"
            xt_dir = f"{save_path}/intermediates/x_t"
            os.makedirs(x0_dir, exist_ok=True)
            os.makedirs(xt_dir, exist_ok=True)

            print("Saving intermediate images...")
            for step_idx, (x0_h, lat_h) in enumerate(
                zip(x0_hat_history, latents_history)
            ):
                with torch.no_grad():
                    x0_decoded = self.vae.decode(
                        (x0_h / sigma_d).to(self.vae.dtype)
                        / self.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                    x0_img = (
                        (x0_decoded / 2 + 0.5)
                        .clamp(0, 1)[0]
                        .permute(1, 2, 0)
                        .float()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    x0_img = (x0_img * 255).astype(np.uint8)

                    lat_decoded = self.vae.decode(
                        (lat_h / sigma_d).to(self.vae.dtype)
                        / self.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                    lat_img = (
                        (lat_decoded / 2 + 0.5)
                        .clamp(0, 1)[0]
                        .permute(1, 2, 0)
                        .float()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    lat_img = (lat_img * 255).astype(np.uint8)

                Image.fromarray(x0_img).save(f"{x0_dir}/step_{step_idx:03d}.png")
                Image.fromarray(lat_img).save(f"{xt_dir}/step_{step_idx:03d}.png")

            if len(reward_history) > 0:
                try:
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(10, 4))
                    plt.plot(reward_history, "b-o", linewidth=2, markersize=4)
                    plt.xlabel("Step")
                    plt.ylabel("Reward")
                    plt.title("FMTT Reward Evolution")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{save_path}/reward_curve.png", dpi=150)
                    plt.close()
                except ImportError:
                    pass

        latents = x0_hat / self.scheduler.config.sigma_data
        latents = latents.to(self.vae.dtype)
        with torch.no_grad():
            final_image_tensor = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            final_image_tensor = (final_image_tensor / 2 + 0.5).clamp(0, 1)

        if reward_scale > 0 and reward_fns is not None:
            final_preprocessed_img = self.img_transform(final_image_tensor)
            final_raw_reward = 0.0
            for reward_fn in reward_fns:
                reward_val = reward_fn(
                    final_preprocessed_img, prompt, final_image_tensor
                )
                final_raw_reward = final_raw_reward + reward_val * reward_fn.weighting
            final_reward_value = float(final_raw_reward.detach().item())
            maybe_update_best_candidate(
                final_reward_value,
                final_image_tensor,
                "final_sample",
            )

        # Return the best reward-scored clean-image estimate seen during guidance
        # or the final jumped sample if it wins.
        if best_image_tensor is not None:
            image = best_image_tensor
        else:
            image = final_image_tensor

        image = image[0].permute(1, 2, 0).detach().cpu().float().numpy()
        image = (image * 255).clip(0, 255).astype("uint8")

        if save_outputs:
            Image.fromarray(image).save(f"{save_path}/output.png")

        return image
