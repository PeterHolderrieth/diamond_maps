import gc
import json
import math
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

DIAMOND_NOISE_MODES: tuple[str, ...] = ("fresh", "fixed", "antithetic")
SCORE_SCALE_FACTOR_CLIP: float = 100.0
SCORE_VALUE_CLIP: float = 1000.0


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


def maybe_release_cuda_cache(device: torch.device) -> None:
    """Release CUDA cache when allocator pressure is already high.

    This is a conservative safeguard for large reward-model runs where the
    likelihood pass and reward pass happen back-to-back inside one guidance
    step. The thresholds are environment-configurable so we can tune or
    disable the behavior without touching code.

    Args:
        device: CUDA device associated with the current tensors.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    reserved_bytes = torch.cuda.memory_reserved(device_index)
    allocated_bytes = torch.cuda.memory_allocated(device_index)
    reserved_ratio = reserved_bytes / max(1, total_bytes)
    slack_gb = (reserved_bytes - allocated_bytes) / (1024**3)

    reserved_ratio_threshold = float(
        os.environ.get("DIAMONDFLOWS_FLUX_CACHE_RELEASE_RATIO", "0.90")
    )
    slack_gb_threshold = float(
        os.environ.get("DIAMONDFLOWS_FLUX_CACHE_RELEASE_MIN_SLACK_GB", "0.5")
    )

    if reserved_ratio < reserved_ratio_threshold:
        return
    if slack_gb < slack_gb_threshold:
        return

    gc.collect()
    torch.cuda.empty_cache()


class FluxFlowMapWeightedDiamondPipeline(FluxPipeline):
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

        x0_hat = (latents - (t / 1000) * v0_pred).to(latents.dtype)
        noise_pred = noise_pred.to(latents.dtype)
        return x0_hat, noise_pred

    def compute_reward_gradient_multi_particle(
        self,
        latents_input: torch.Tensor,
        t_val: torch.Tensor,
        t_next: torch.Tensor,
        cond_kwargs: Dict[str, Any],
        num_particles: int,
        snr_factor: float,
        reward_scale: float,
        gradient_norm_scale: float,
        height: int,
        width: int,
        reward_fns: list[Callable[..., torch.Tensor]],
        generator: Optional[torch.Generator],
        prompt: Union[str, List[str]],
        diamond_noise_mode: str,
        fixed_particle_noise: Optional[torch.Tensor],
        main_loop_noise_pred: torch.Tensor,
        include_likelihood: bool = True,
        include_score: bool = True,
        include_weights: bool = False,
        weight_temperature: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, float],
        float,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Compute multi-particle weighted Diamond Maps reward guidance.

        This implementation computes each particle sequentially and stores only
        detached per-particle gradients and logits. Aggregation is done once
        at the end via a batched softmax weighting.

        Args:
            main_loop_noise_pred: noise_pred already computed in the main
                denoising loop at (t_val -> t_next). Reused here for
                score_t to avoid a redundant model call.
            include_likelihood: Include log p_t(x_t | z^i) in the gradient
                objective. Should be True for a consistent estimator.
            include_score: Include the delta-score correction in the gradient.
                Should be True for a consistent estimator.
            include_weights: When True, importance-sampling logits include
                likelihood + gamma + eps-norm. When False, logits are
                reward * scale only.
            weight_temperature: Divides the full logit before softmax when
                include_weights is True.
        """
        if diamond_noise_mode not in DIAMOND_NOISE_MODES:
            raise ValueError(
                "Unsupported diamond noise mode "
                f"`{diamond_noise_mode}`. Choose from {DIAMOND_NOISE_MODES}."
            )
        if include_weights and weight_temperature <= 0.0:
            raise ValueError(
                "weight_temperature must be > 0 when include_weights=True."
            )
        if not include_likelihood:
            logger.warning(
                "include_likelihood=False produces a biased gradient "
                "estimator. Recommended: True."
            )
        if not include_score:
            logger.warning(
                "include_score=False produces a biased gradient "
                "estimator. Recommended: True."
            )

        batch_size = latents_input.shape[0]
        t_norm = t_val / 1000.0

        def compute_t_prime(t_n, snr_f):
            sqrt_lambda = math.sqrt(snr_f)
            t_prime = (sqrt_lambda * t_n) / (sqrt_lambda * t_n + 1 - t_n)
            return torch.clamp(t_prime, max=0.9999)

        t_prime_norm = compute_t_prime(t_norm, snr_factor)
        t_prime = t_prime_norm * 1000.0

        view_shape = (batch_size,) + (1,) * (latents_input.ndim - 1)
        t_view = t_norm.view(view_shape)
        t_prime_view = t_prime_norm.view(view_shape)

        alpha_t = 1 - t_view
        var_t = t_view**2
        alpha_prev = 1 - t_prime_view
        var_prev = t_prime_view**2

        scale_factor = alpha_prev / (alpha_t + 1e-8)
        var_q = var_prev - (scale_factor**2) * var_t
        var_q = torch.clamp(var_q, min=1e-8)
        std_q = torch.sqrt(var_q)

        def safe_score(x, alpha, x0, var, min_var=1e-4):
            score = -(x - alpha * x0) / torch.clamp(var, min=min_var)
            score = torch.nan_to_num(
                score,
                nan=0.0,
                posinf=SCORE_VALUE_CLIP,
                neginf=-SCORE_VALUE_CLIP,
            )
            return score.clamp(min=-SCORE_VALUE_CLIP, max=SCORE_VALUE_CLIP)

        def per_sample_l2_norm(tensor: torch.Tensor) -> torch.Tensor:
            """Compute per-sample L2 norm for tensor with shape `[B, ...]`."""
            # shape: [B, D]
            flat_tensor = tensor.reshape(batch_size, -1)
            # shape: [B]
            return torch.linalg.vector_norm(flat_tensor, ord=2, dim=1)

        def normalize_term_grad(
            term_grad: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Normalize one gradient term by its own per-sample norm."""
            # shape: [B]
            term_grad_norm = per_sample_l2_norm(term_grad).clamp_min(1e-8)
            # shape: [B, 1, ..., 1]
            norm_view = term_grad_norm.view(view_shape)
            # shape: [B, ...]
            term_grad_unit = term_grad / norm_view
            return term_grad_unit, term_grad_norm

        def sample_standard_noise_like(reference: torch.Tensor) -> torch.Tensor:
            """Sample Gaussian noise while preserving the provided generator.

            Some torch builds do not support passing ``generator=...`` to
            ``torch.randn_like``. Sampling via ``torch.randn(shape, ...)`` keeps
            seeded behavior intact across those versions.
            """
            return torch.randn(
                reference.shape,
                device=reference.device,
                dtype=reference.dtype,
                layout=reference.layout,
                generator=generator,
            )

        # Score at t from main-loop instantaneous velocity.
        score_t = None
        score_scale_factor = torch.clamp(
            scale_factor,
            min=-SCORE_SCALE_FACTOR_CLIP,
            max=SCORE_SCALE_FACTOR_CLIP,
        )
        if include_score or include_weights:
            with torch.no_grad():
                main_loop_noise_pred_detached = main_loop_noise_pred.detach()
                x0_inst_t = (
                    latents_input - t_view * main_loop_noise_pred_detached
                ).detach()
                score_t = safe_score(latents_input, alpha_t, x0_inst_t, var_t).detach()

        particle_logits: list[torch.Tensor] = []
        particle_reward_grads: list[torch.Tensor] = []
        particle_likelihood_grads: list[torch.Tensor] = []
        particle_scores: list[torch.Tensor] = []

        reward_sum = torch.zeros(
            batch_size,
            device=latents_input.device,
            dtype=torch.float32,
        )
        best_reward_particles = float("-inf")
        best_particle_x0_hat: torch.Tensor | None = None
        best_particle_image_pred: torch.Tensor | None = None
        decode_dtype = vae_decode_dtype(self.vae)
        latents_input_detached = latents_input.detach()

        antithetic_base_eps = None
        for particle_idx in range(num_particles):
            if diamond_noise_mode == "fresh":
                eps = sample_standard_noise_like(latents_input)
            elif diamond_noise_mode == "fixed":
                if fixed_particle_noise is None:
                    raise ValueError(
                        "fixed_particle_noise must be provided "
                        "for `diamond_noise_mode='fixed'`."
                    )
                eps = fixed_particle_noise[particle_idx].to(
                    device=latents_input.device,
                    dtype=latents_input.dtype,
                )
            else:  # antithetic
                if particle_idx % 2 == 0:
                    antithetic_base_eps = sample_standard_noise_like(latents_input)
                if antithetic_base_eps is None:
                    raise ValueError(
                        "Antithetic noise base sample is missing. "
                        "This indicates invalid particle loop state."
                    )
                eps = (
                    antithetic_base_eps
                    if particle_idx % 2 == 0
                    else -antithetic_base_eps
                )

            # Likelihood term (separate pass to avoid retaining reward graph).
            likelihood_grad = torch.zeros_like(latents_input)
            if include_likelihood:
                with torch.enable_grad():
                    latents_input_like = latents_input.detach().requires_grad_(True)
                    mu_q_like = scale_factor * latents_input_like
                    latents_renoised_like = mu_q_like + std_q * eps

                    _, noise_pred_like = self.compute_model_prediction(
                        latents_renoised_like, t_prime, t_next, cond_kwargs
                    )
                    x0_inst_like = (
                        latents_renoised_like - t_prime_view * noise_pred_like
                    )
                    mu_p = alpha_t * x0_inst_like
                    log_p_k = (
                        -0.5
                        * ((latents_input_like - mu_p) ** 2).sum(dim=[1, 2])
                        / var_t.reshape(batch_size)
                    )
                    likelihood_grad = torch.autograd.grad(
                        outputs=log_p_k.sum(),
                        inputs=latents_input_like,
                        retain_graph=False,
                        create_graph=False,
                    )[0]

                del (
                    latents_input_like,
                    mu_q_like,
                    latents_renoised_like,
                    noise_pred_like,
                    x0_inst_like,
                    mu_p,
                    log_p_k,
                )
                maybe_release_cuda_cache(latents_input.device)

            # Reward term pass.
            with torch.enable_grad():
                latents_input_k = latents_input.detach().requires_grad_(True)
                mu_q = scale_factor * latents_input_k
                latents_renoised = mu_q + std_q * eps

                # Model prediction at (x_{t'}, t_prime).
                x0_hat_k, noise_pred_k = self.compute_model_prediction(
                    latents_renoised, t_prime, t_next, cond_kwargs
                )

                # Velocity-derived x0 for score & likelihood.
                # x0_hat_k (from v0_pred) is ONLY used for decoding/reward.
                x0_inst_k = latents_renoised - t_prime_view * noise_pred_k

                # Decode & reward (uses flow-map x0_hat_k, unchanged).
                unpacked_x0 = self._unpack_latents(
                    x0_hat_k, height, width, self.vae_scale_factor
                )
                unpacked_x0 = (
                    unpacked_x0 / self.vae.config.scaling_factor
                ) + self.vae.config.shift_factor
                image_k = self.vae.decode(
                    unpacked_x0.to(decode_dtype),
                    return_dict=False,
                )[0]
                image_k = (image_k / 2 + 0.5).clamp(0, 1)
                preprocessed_img = self.img_transform(image_k)

                raw_reward = 0.0
                for reward_fn in reward_fns:
                    raw_reward = (
                        raw_reward
                        + reward_fn(preprocessed_img, prompt, image_k)
                        * reward_fn.weighting
                    )

                reward_value = float(raw_reward.detach().mean().item())
                if reward_value > best_reward_particles:
                    best_reward_particles = reward_value
                    best_particle_x0_hat = x0_hat_k.detach().clone()
                    best_particle_image_pred = image_k.detach().clone()

                reward_objective = raw_reward * reward_scale
                reward_grad = torch.autograd.grad(
                    outputs=reward_objective.sum(),
                    inputs=latents_input_k,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                reward_flat = raw_reward.detach().flatten().to(torch.float32)
                latents_renoised_detached = latents_renoised.detach()
                x0_inst_detached = x0_inst_k.detach()

            del (
                latents_input_k,
                mu_q,
                latents_renoised,
                x0_hat_k,
                noise_pred_k,
                x0_inst_k,
                unpacked_x0,
                image_k,
                preprocessed_img,
                raw_reward,
                reward_objective,
            )

            # Score gradient term.
            score_grad = torch.zeros_like(latents_input)
            score_tp = None
            if include_score or include_weights:
                with torch.no_grad():
                    score_tp = safe_score(
                        latents_renoised_detached,
                        alpha_prev,
                        x0_inst_detached,
                        var_prev,
                    )
                    if include_score:
                        score_grad = score_tp * score_scale_factor - score_t
                        score_grad = torch.nan_to_num(
                            score_grad,
                            nan=0.0,
                            posinf=SCORE_VALUE_CLIP,
                            neginf=-SCORE_VALUE_CLIP,
                        )
                        score_grad = score_grad.clamp(
                            min=-SCORE_VALUE_CLIP,
                            max=SCORE_VALUE_CLIP,
                        )

            # Logit computation.
            reward_sum = reward_sum + reward_flat
            particle_reward_grads.append(reward_grad)
            particle_likelihood_grads.append(likelihood_grad)
            particle_scores.append(score_grad)

            if include_weights:
                with torch.no_grad():
                    residual = latents_input_detached - alpha_t * x0_inst_detached
                    log_p_val = (
                        -0.5 * (residual**2).sum(dim=[1, 2]) / var_t.reshape(batch_size)
                    )

                    eps_norm = 0.5 * (eps**2).sum(dim=[1, 2]).reshape(batch_size)

                    if score_t is None or score_tp is None:
                        raise ValueError(
                            "score_t/score_tp must be available when include_weights=True."
                        )
                    diff = latents_renoised_detached - latents_input_detached
                    avg_score = 0.5 * (score_t + score_tp)
                    gamma_k = (avg_score * diff).sum(dim=[1, 2]).reshape(batch_size)

                    logit = (
                        reward_flat * reward_scale
                        + log_p_val.reshape(batch_size)
                        + gamma_k
                        + eps_norm
                    ).to(torch.float32) / weight_temperature
            else:
                logit = reward_flat

            particle_logits.append(logit.detach())

            del (
                reward_flat,
                logit,
                latents_renoised_detached,
                x0_inst_detached,
                eps,
            )

        # shape: [K, B]
        stacked_logits = torch.stack(particle_logits, dim=0)
        # shape: [K, B]
        particle_weights = torch.softmax(stacked_logits, dim=0)
        # shape: [K, B, 1, ..., 1]
        particle_weight_view = particle_weights.view(
            (num_particles, batch_size) + (1,) * (latents_input.ndim - 1)
        )
        # shape: [K, B, ...]
        weighted_reward_grads = (
            torch.stack(particle_reward_grads, dim=0) * particle_weight_view
        ).sum(dim=0)
        weighted_likelihood_grads = (
            torch.stack(particle_likelihood_grads, dim=0) * particle_weight_view
        ).sum(dim=0)
        weighted_score_grads = (
            torch.stack(particle_scores, dim=0) * particle_weight_view
        ).sum(dim=0)
        # grad normalization
        unit_reward_grads, reward_grad_norm = normalize_term_grad(weighted_reward_grads)
        unit_likelihood_grads = torch.zeros_like(unit_reward_grads)
        likelihood_grad_norm = torch.zeros_like(reward_grad_norm)
        if include_likelihood:
            unit_likelihood_grads, likelihood_grad_norm = normalize_term_grad(
                weighted_likelihood_grads
            )
        unit_score_grads = torch.zeros_like(unit_reward_grads)
        score_grad_norm = torch.zeros_like(reward_grad_norm)
        if include_score:
            unit_score_grads, score_grad_norm = normalize_term_grad(
                weighted_score_grads
            )

        combined_unit_grads = (
            unit_reward_grads + unit_likelihood_grads + unit_score_grads
        )
        total_grad_norm = per_sample_l2_norm(combined_unit_grads).clamp_min(1e-8)

        total_grad = combined_unit_grads * gradient_norm_scale
        total_grad = total_grad.to(latents_input.dtype)

        grad_term_stats: Dict[str, float] = {
            "reward_norm_mean": float(reward_grad_norm.mean().item()),
            "likelihood_norm_mean": float(likelihood_grad_norm.mean().item()),
            "score_norm_mean": float(score_grad_norm.mean().item()),
            "combined_norm_mean": float(total_grad_norm.mean().item()),
        }

        mean_reward = reward_sum.mean() / float(num_particles)
        return (
            total_grad,
            mean_reward,
            grad_term_stats,
            best_reward_particles,
            best_particle_x0_hat,
            best_particle_image_pred,
        )

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
        num_reward_particles: int = 4,
        include_likelihood: bool = True,
        include_score: bool = True,
        include_weights: bool = False,
        reward_fns: Callable = None,
        gradient_norm_scale: float = 10.0,
        snr_factor: float = 5.0,
        weight_temperature: float = 1.0,
        diamond_noise_mode: str = "fresh",
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

        if reward_scale > 0:
            if not hasattr(self, "img_transform"):
                self.img_transform = clip_img_transform()
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

        if diamond_noise_mode not in DIAMOND_NOISE_MODES:
            raise ValueError(
                "Unsupported diamond noise mode "
                f"`{diamond_noise_mode}`. Choose from {DIAMOND_NOISE_MODES}."
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
        decode_dtype = vae_decode_dtype(self.vae)

        best_reward = float("-inf")
        best_latent_output: torch.Tensor | None = None
        best_image_pred: torch.Tensor | None = None
        top_reward_limit = max(int(save_top_reward_images), 0)
        top_reward_candidates: list[dict[str, Any]] = []

        def decode_packed_latents(latent_tensor: torch.Tensor) -> torch.Tensor:
            """Decode packed FLUX latents into an image tensor in `[0, 1]`."""
            unpacked_x0 = self._unpack_latents(
                latent_tensor.to(decode_dtype),
                height,
                width,
                self.vae_scale_factor,
            )
            unpacked_x0 = (
                unpacked_x0 / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            decoded = self.vae.decode(
                unpacked_x0,
                return_dict=False,
            )[0]
            return (decoded / 2 + 0.5).clamp(0, 1)

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
                    "[FluxDiamondMap] New best reward-scored candidate: "
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
            """Convert a tracked best image in `[0, 1]` to the requested output type."""
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

        def save_latent_frame(latents_tensor: torch.Tensor, out_file: str) -> None:
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

        fixed_particle_noise = None
        if (
            reward_scale > 0
            and guidance_end_step > guidance_start_idx
            and diamond_noise_mode == "fixed"
        ):
            fixed_particle_noise = torch.randn(
                (num_reward_particles,) + tuple(latents.shape),
                device=latents.device,
                dtype=latents.dtype,
                generator=generator,
            )

        if save_intermediate_imgs:
            prompt_text = prompt[0] if isinstance(prompt, list) else prompt
            if prompt_text is None:
                prompt_text = "prompt"
            if save_path is None:
                save_path = f"flux_diamond/{prompt_slug(prompt_text)}"
            xt_dir = os.path.join(save_path, "intermediates", "x_t")
            x0_dir = os.path.join(save_path, "intermediates", "x0_hat")
            os.makedirs(xt_dir, exist_ok=True)
            os.makedirs(x0_dir, exist_ok=True)

        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            self._current_timestep = t

            latents = latents.detach()

            if i == len(timesteps) - 1:
                timestep2 = torch.zeros_like(t)
            else:
                timestep2 = timesteps[i + 1]

            is_guided_step = (
                reward_scale > 0 and guidance_start_idx <= i < guidance_end_step
            )
            # The main denoising prediction is inference-only. Guidance
            # gradients are computed inside `compute_reward_gradient_multi_particle`
            # with their own local autograd scopes.
            with torch.no_grad():
                x0_hat, noise_pred = self.compute_model_prediction(
                    latents, t, timestep2, cond_kwargs
                )

            if is_guided_step:
                main_path_image_pred = decode_packed_latents(x0_hat)
                main_path_reward = score_decoded_image(main_path_image_pred)
                maybe_track_reward_candidate(
                    main_path_reward,
                    x0_hat,
                    main_path_image_pred,
                    "guided_main_path",
                )

                (
                    total_grad,
                    mean_reward,
                    grad_term_stats,
                    best_particle_reward,
                    best_particle_x0_hat,
                    best_particle_image_pred,
                ) = self.compute_reward_gradient_multi_particle(
                    latents,
                    t,
                    timestep2,
                    cond_kwargs,
                    num_reward_particles,
                    snr_factor,
                    reward_scale,
                    gradient_norm_scale,
                    height,
                    width,
                    reward_fns,
                    generator,
                    prompt,
                    diamond_noise_mode,
                    fixed_particle_noise,
                    main_loop_noise_pred=noise_pred,
                    include_likelihood=include_likelihood,
                    include_score=include_score,
                    include_weights=include_weights,
                    weight_temperature=weight_temperature,
                )
                if (
                    best_particle_x0_hat is not None
                    and best_particle_image_pred is not None
                ):
                    maybe_track_reward_candidate(
                        best_particle_reward,
                        best_particle_x0_hat,
                        best_particle_image_pred,
                        "guided_particle",
                    )
                t_norm = (t / 1000.0).view(-1, 1, 1)

                mem_stats = cuda_memory_stats(latents.device)
                print(
                    f"[FluxDiamondMap] Step {i + 1}: "
                    f"Reward={mean_reward.item():.4f}, "
                    f"||∇total||={total_grad.norm().item():.4f}, "
                    f"||∇reward||={grad_term_stats['reward_norm_mean']:.4f}, "
                    f"||∇likelihood||={grad_term_stats['likelihood_norm_mean']:.4f}, "
                    f"||∇score||={grad_term_stats['score_norm_mean']:.4f}, "
                    f"||∇combined_raw||={grad_term_stats['combined_norm_mean']:.4f}, "
                    f"||velocity||={noise_pred.norm().item():.4f}, "
                    f"{mem_stats}"
                )
                b_3d = compute_b_coefficient_from_t(t_norm).view(-1, 1, 1)
                noise_pred = noise_pred - b_3d * total_grad

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
