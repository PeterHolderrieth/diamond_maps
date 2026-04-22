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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

# save image
import os
import math

import numpy as np
from PIL import Image
import torch

from transformers import Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast

from diffusers.loaders import SanaLoraLoaderMixin
from diffusers.models import AutoencoderDC, SanaTransformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
)

from diffusers import SanaSprintPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
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


class DiamondFlowSana(SanaSprintPipeline):
    r"""
    Pipeline for text-to-image generation using [SANA-Sprint](https://huggingface.co/papers/2503.09641).
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
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded

            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 300): Maximum sequence length to use for the prompt.
            complex_human_instruction (`list[str]`, defaults to `complex_human_instruction`):
                If `complex_human_instruction` is not empty, the function will use the complex Human instruction for
                the prompt.
        """

        if device is None:
            device = self._execution_device

        if self.transformer is not None:
            dtype = self.transformer.dtype
        elif self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        else:
            dtype = None

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SanaLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        # See Section 3.1. of the paper.
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
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        if self.text_encoder is not None:
            if isinstance(self, SanaLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
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
        num_reward_particles: int = 4,
        include_likelihood: bool = True,
        include_score: bool = False,
        include_weights: bool = False,
        save_intermediate_imgs: bool = True,
        reward_fns: Callable = None,
        gradient_norm_scale: float = 10.0,
        snr_factor: float = 5.0,
        save_path: str | None = None,
        save_outputs: bool = False,
        weight_temperature: float = 1.0,
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
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            max_timesteps (`float`, *optional*, defaults to 1.57080):
                The maximum timestep value used in the SCM scheduler.
            intermediate_timesteps (`float`, *optional*, defaults to 1.3):
                The intermediate timestep value used in SCM scheduler (only used when num_inference_steps=2).
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            attention_kwargs:
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `300`):
                Maximum sequence length to use with the `prompt`.
            complex_human_instruction (`List[str]`, *optional*):
                Instructions for complex human attention:
                https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

        Examples:

        Returns:
            [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images
        """

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

        # 2. Default height and width to transformer
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

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
        ) = self.encode_prompt(
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

        # 4. Prepare timesteps
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

        # 5. Prepare latents.
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

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0]).to(prompt_embeds.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale

        # 7. Denoising loop

        def tau_from_s(s):
            # s can be scalar tensor or (B,) tensor
            return torch.sin(s) / (torch.sin(s) + torch.cos(s))

        def r_from_tau(tau):
            return torch.sqrt(tau * tau + (1.0 - tau) * (1.0 - tau))

        sigma_d = self.scheduler.config.sigma_data

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0]).to(prompt_embeds.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale

        transformer_dtype = self.transformer.dtype
        self._num_timesteps = len(timesteps)

        x0_hat_history = []
        latents_history = []
        tau_history = []
        mean_reward_history = []
        reward_history = []

        if save_path is None:
            save_path = f"sanabase/diamond_aesth/{prompt}/gradnorm{gradient_norm_scale}_reward_scale{reward_scale}_numparticles{num_reward_particles}_snr{snr_factor}_guidance_steps{num_guidance_steps}{'_likelihood' if include_likelihood else ''}{'_score' if include_score else ''}"
        if save_outputs or save_intermediate_imgs:
            os.makedirs(save_path, exist_ok=True)

        def compute_model_prediction(
            latents_in, tau_in, prompt_embeds, prompt_attention_mask, guidance
        ):
            """
            Run the model at a given tau to get x0_hat prediction.

            The forward process is defined in s-space:
                x_s = cos(s) · x_0 + sin(s) · σ_d · ε

            where s ∈ [0, π/2]:
                - s = 0     → pure clean (x_s = x_0)
                - s = π/2   → pure noise (x_s = σ_d · ε)

            τ is a reparametrization of s via:
                τ = sin(s) / (sin(s) + cos(s))
                s = atan2(τ, 1-τ)

            So:
                cos(s) = (1-τ) / r(τ)
                sin(s) = τ / r(τ)
            where r(τ) = √(τ² + (1-τ)²) is a normalization factor.
            """
            # ─────────────────────────────────────────────────────────────
            # Step 1: Convert τ → s (the fundamental angle parameter)
            # ─────────────────────────────────────────────────────────────
            # τ = sin(s)/(sin(s)+cos(s)), inverting gives:
            s = torch.atan2(tau_in, 1 - tau_in)

            # ─────────────────────────────────────────────────────────────
            # Step 2: Compute the interpolation coefficients
            # ─────────────────────────────────────────────────────────────
            # These are the fundamental coefficients from the forward process:
            #   x_s = α_s · x_0 + σ_s · ε
            # where α_s = cos(s) and σ_s = σ_d · sin(s)
            cos_s = torch.cos(s).view(-1, 1, 1, 1)  # = α_s / 1 (signal coefficient)
            sin_s = torch.sin(s).view(
                -1, 1, 1, 1
            )  # = σ_s / σ_d (noise coefficient, normalized)

            # r(τ) is needed because the model was trained with a specific input scaling
            tau_4d = tau_in.view(-1, 1, 1, 1)
            r = torch.sqrt(tau_4d**2 + (1 - tau_4d) ** 2)
            # Note: r = 1 / (cos(s) + sin(s)), it's the "τ-space normalization"

            # ─────────────────────────────────────────────────────────────
            # Step 3: Prepare model input (model-specific scaling)
            # ─────────────────────────────────────────────────────────────
            # The model expects inputs normalized by σ_d and scaled by r
            # This is a quirk of how the model was trained
            latents_model_input = latents_in / sigma_d
            x_tau = latents_model_input * r

            # ─────────────────────────────────────────────────────────────
            # Step 4: Run the transformer
            # ─────────────────────────────────────────────────────────────
            # Model is conditioned on τ (not s) — this is what it was trained with
            model_pred = self.transformer(
                x_tau.to(dtype=transformer_dtype),
                encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                encoder_attention_mask=prompt_attention_mask,
                guidance=guidance,
                timestep=tau_in.to(dtype=transformer_dtype),  # Model expects τ!
                return_dict=False,
                attention_kwargs=self.attention_kwargs,
            )[0]

            # ─────────────────────────────────────────────────────────────
            # Step 5: Convert model output → x0_hat
            # ─────────────────────────────────────────────────────────────
            # The model predicts something related to velocity/score.
            # This formula inverts that to get the clean image estimate.
            # Derived from the model's training objective.
            x0_hat = (cos_s - sin_s * (1 - 2 * tau_4d)) * latents_in - sin_s * (
                1 - 2 * tau_4d + 2 * tau_4d**2
            ) * sigma_d * model_pred / r

            return x0_hat

        def compute_tau_prime(
            tau: torch.Tensor, snr_factor: float = 5.0
        ) -> torch.Tensor:
            """SNR-based τ' computation from Diamond Flows paper."""
            sqrt_lambda = math.sqrt(snr_factor)
            tau_prime = (sqrt_lambda * tau) / (1 + (sqrt_lambda - 1) * tau)
            return torch.clamp(tau_prime, max=0.9999)

        def compute_gaussian_log_prob(x, mu, sigma_sq, clip_sigma: float = 0.1):
            """
            Computes log N(x; mu, sigma_sq).
            """
            # (x - mu)^2 / (2 * sigma^2)
            diff_sq = (x - mu) ** 2
            # Sum over dimensions (C, H, W) -> (B,)
            log_prob = (
                -0.5
                * diff_sq.sum(dim=[1, 2, 3])
                / torch.clamp(sigma_sq, min=clip_sigma).squeeze()
            )
            return log_prob

        def safe_score(x, alpha, x0_hat, var, min_var=0.1):
            """Score with clamped variance."""
            return -(x - alpha * x0_hat) / torch.clamp(var, min=min_var)

        def compute_reward_gradient_multi_particle(
            latents_input,
            tau,
            prompt_embeds,
            prompt_attention_mask,
            guidance,
            x0_hat_t,
            num_particles=5,
            snr_factor=5.0,
            reward_scale=100.0,  # Needs to be high (e.g. 100-2000) to compete with log-likelihoods
            gradient_norm_scale: float = 10.0,
            include_weights: bool = False,
        ):
            if include_weights and weight_temperature <= 0.0:
                raise ValueError(
                    "weight_temperature must be > 0 when include_weights=True."
                )

            batch_size = latents_input.shape[0]

            tau_prime = compute_tau_prime(tau, snr_factor)

            tau_4d = tau.view(-1, 1, 1, 1)
            r_t = torch.sqrt(tau_4d**2 + (1 - tau_4d) ** 2)

            alpha_t = (1 - tau_4d) / r_t
            sigma_t = sigma_d * tau_4d / r_t
            var_t = sigma_t**2

            tau_prime_4d = tau_prime.view(-1, 1, 1, 1)
            r_p = torch.sqrt(tau_prime_4d**2 + (1 - tau_prime_4d) ** 2)

            alpha_prev = (1 - tau_prime_4d) / r_p
            sigma_prev = sigma_d * tau_prime_4d / r_p
            var_prev = sigma_prev**2

            scale_factor = alpha_prev / (alpha_t + 1e-8)
            var_q = var_prev - (scale_factor**2) * var_t
            var_q = torch.clamp(var_q, min=1e-8)
            std_q = torch.sqrt(var_q)

            print(
                f"tau: {tau.mean().item():.4f}, tau': {tau_prime.mean().item():.4f}, "
                f"alpha_t: {alpha_t.mean().item():.4f}, sigma_t: {sigma_t.mean().item():.4f}, "
                f"alpha_t': {alpha_prev.mean().item():.4f}, sigma_t': {sigma_prev.mean().item():.4f}, "
                f"scale_factor: {scale_factor.mean().item():.4f}, var_t: {var_t.mean().item():.4f}, "
                f"var_t': {var_prev.mean().item():.4f}, var_q: {var_q.mean().item():.4f}"
            )

            score_t = None
            if include_score:
                score_t = safe_score(latents_input, alpha_t, x0_hat_t, var_t).detach()

            particle_reward_grads = []
            particle_likelihood_grads = []
            particle_score_diffs = []
            particle_logits = []
            reward_sum = torch.zeros(
                batch_size,
                device=latents_input.device,
                dtype=torch.float32,
            )
            best_reward_particles = float("-inf")
            best_image_particles = None
            view_shape = (batch_size, 1, 1, 1)

            def per_sample_l2_norm(tensor: torch.Tensor) -> torch.Tensor:
                """Compute per-sample L2 norm for tensor with shape `[B, ...]`."""
                flat_tensor = tensor.reshape(batch_size, -1)
                return torch.linalg.vector_norm(flat_tensor, ord=2, dim=1)

            def normalize_term_grad(
                term_grad: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """Normalize one gradient term by its own per-sample norm."""
                term_grad_norm = per_sample_l2_norm(term_grad).clamp_min(1e-8)
                norm_view = term_grad_norm.view(view_shape)
                term_grad_unit = term_grad / norm_view
                return term_grad_unit, term_grad_norm

            for _ in range(num_particles):
                with torch.enable_grad():
                    latents_input_k = latents_input.detach().requires_grad_(True)
                    mu_q = scale_factor * latents_input_k
                    eps = torch.randn(
                        size=latents_input_k.shape,
                        device=latents_input_k.device,
                        generator=generator,
                    )
                    latents_renoised = mu_q + std_q * eps

                    x0_hat_k = compute_model_prediction(
                        latents_renoised,
                        tau_prime,
                        prompt_embeds,
                        prompt_attention_mask,
                        guidance,
                    )

                    score_diff = torch.zeros_like(latents_input_k)
                    if include_score:
                        if score_t is None:
                            raise ValueError(
                                "score_t must be defined when include_score=True."
                            )
                        score_tp = safe_score(
                            latents_renoised.detach(),
                            alpha_prev,
                            x0_hat_k.detach(),
                            var_prev,
                        )
                        score_diff = score_tp * scale_factor - score_t

                    image_k = self.vae.decode(
                        (x0_hat_k / sigma_d).to(self.vae.dtype)
                        / self.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                    image_k = (image_k / 2 + 0.5).clamp(0, 1)
                    preprocessed_img = self.img_transform(image_k)

                    raw_reward = 0.0
                    for reward_fn in reward_fns:
                        raw_reward = (
                            raw_reward
                            + reward_fn(preprocessed_img, prompt) * reward_fn.weighting
                        )

                    reward_value = float(raw_reward.detach().mean().item())
                    if reward_value > best_reward_particles:
                        best_reward_particles = reward_value
                        best_image_particles = (
                            image_k[0]
                            .permute(1, 2, 0)
                            .clone()
                            .float()
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    reward_objective = raw_reward * reward_scale
                    reward_grad = torch.autograd.grad(
                        outputs=reward_objective.sum(),
                        inputs=latents_input_k,
                        retain_graph=include_likelihood,
                        create_graph=False,
                    )[0]

                    likelihood_grad = torch.zeros_like(latents_input_k)
                    if include_likelihood:
                        mu_p = alpha_t * x0_hat_k
                        dist = torch.distributions.Normal(mu_p, torch.sqrt(var_t))
                        log_p_k = dist.log_prob(latents_input_k).sum(dim=(1, 2, 3))
                        likelihood_grad = torch.autograd.grad(
                            outputs=log_p_k.sum(),
                            inputs=latents_input_k,
                            retain_graph=False,
                            create_graph=False,
                        )[0]

                reward_flat = raw_reward.detach().flatten().to(torch.float32)
                reward_sum = reward_sum + reward_flat
                if include_weights:
                    # Previous weighting multiplied raw rewards by reward_scale:
                    # logit = reward_flat * (reward_scale / weight_temperature)
                    logit = reward_flat / weight_temperature
                else:
                    logit = reward_flat

                reward_grad_detached = reward_grad.detach()
                likelihood_grad_detached = likelihood_grad.detach()
                score_diff_detached = score_diff.detach()

                particle_reward_grads.append(reward_grad_detached)
                particle_likelihood_grads.append(likelihood_grad_detached)
                particle_score_diffs.append(score_diff_detached)
                particle_logits.append(logit.detach())

                del x0_hat_k, image_k, latents_renoised
                torch.cuda.empty_cache()

            if not particle_reward_grads:
                raise ValueError("No particles were processed for Diamond guidance.")

            if include_weights:
                stacked_logits = torch.stack(particle_logits, dim=0)
                particle_weights = torch.softmax(stacked_logits, dim=0)
                particle_weight_view = particle_weights.view(
                    (num_particles, batch_size) + (1,) * (latents_input.ndim - 1)
                )
                reward_grads = (
                    torch.stack(particle_reward_grads, dim=0) * particle_weight_view
                ).sum(dim=0)
                likelihood_grads = (
                    torch.stack(particle_likelihood_grads, dim=0) * particle_weight_view
                ).sum(dim=0)
                score_diffs = (
                    torch.stack(particle_score_diffs, dim=0) * particle_weight_view
                ).sum(dim=0)
            else:
                reward_grads = torch.stack(particle_reward_grads, dim=0).mean(dim=0)
                likelihood_grads = torch.stack(particle_likelihood_grads, dim=0).mean(
                    dim=0
                )
                score_diffs = torch.stack(particle_score_diffs, dim=0).mean(dim=0)

            print(
                f"Reward grad norm: {reward_grads.norm().item():.4f}, "
                f"likelihood_grad norm: {likelihood_grads.norm().item():.4f}, "
                f"score_diff norm: {score_diffs.norm().item():.4f}, "
            )

            unit_reward_grads, reward_grad_norm = normalize_term_grad(reward_grads)
            unit_likelihood_grads = torch.zeros_like(unit_reward_grads)
            likelihood_grad_norm = torch.zeros_like(reward_grad_norm)
            if include_likelihood:
                unit_likelihood_grads, likelihood_grad_norm = normalize_term_grad(
                    likelihood_grads
                )
            unit_score_diffs = torch.zeros_like(unit_reward_grads)
            score_grad_norm = torch.zeros_like(reward_grad_norm)
            if include_score:
                unit_score_diffs, score_grad_norm = normalize_term_grad(score_diffs)

            combined_unit_grads = (
                unit_reward_grads + unit_likelihood_grads + unit_score_diffs
            )
            total_grad_norm = per_sample_l2_norm(combined_unit_grads).clamp_min(1e-8)

            total_grad = combined_unit_grads * gradient_norm_scale
            total_grad = total_grad.to(latents_input.dtype)

            print(
                f"Normalized reward grad: {reward_grad_norm.mean().item():.4f}, "
                f"normalized likelihood grad: {likelihood_grad_norm.mean().item():.4f}, "
                f"normalized score grad: {score_grad_norm.mean().item():.4f}, "
                f"combined unit grad: {total_grad_norm.mean().item():.4f}"
            )

            if save_outputs and best_image_particles is not None:
                os.makedirs(f"{save_path}/reward_outputs", exist_ok=True)
                img_np = (best_image_particles * 255).round().astype("uint8")
                Image.fromarray(img_np).save(
                    f"{save_path}/reward_outputs/step_{tau.mean().item():.4f}.png"
                )

            mean_reward = reward_sum.mean() / float(num_particles)
            return (
                total_grad,
                best_reward_particles,
                best_image_particles,
                mean_reward,
            )

        def compute_b_coefficient_from_s(
            s: torch.Tensor, sigma_d: float, eps: float = 1e-6, max_abs_b: float = 3.0
        ) -> torch.Tensor:
            """
            b_s = -σ_s^2 (α̇_s / α_s) - σ̇_s σ_s  with α_s=cos s, σ_s=sigma_d sin s, derivatives w.r.t s.
            s: (B,) tensor in radians, typically in (0, pi/2).
            Returns: (B,) tensor.
            """
            sin_s = torch.sin(s)
            cos_s = torch.cos(s).clamp(min=eps)  # avoid blow-up as s -> pi/2
            b = (sigma_d**2) * (sin_s**3 / cos_s - sin_s * cos_s)
            return b.clamp(max=max_abs_b)

        best_reward = float("-inf")
        best_image = None

        def maybe_update_best_candidate(
            candidate_reward: float,
            candidate_image: torch.Tensor | np.ndarray,
            candidate_source: str,
        ) -> None:
            """Track the best reward-scored SANA Diamond candidate seen so far."""
            nonlocal best_reward, best_image
            if candidate_reward <= best_reward:
                return

            if isinstance(candidate_image, torch.Tensor):
                candidate_image_np = (
                    candidate_image[0]
                    .permute(1, 2, 0)
                    .clone()
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                candidate_image_np = candidate_image.copy()

            best_reward = candidate_reward
            best_image = candidate_image_np
            print(
                "[SanaDiamondMap] New best reward-scored candidate: "
                f"{best_reward:.4f} [source={candidate_source}]"
            )

        guidance_end_step = min(
            num_inference_steps,
            guidance_start_step + num_guidance_steps,
        )

        for i in range(num_inference_steps):
            torch.cuda.empty_cache()
            s_curr = timesteps[i]
            s_next = timesteps[i + 1]

            s_b = s_curr.expand(latents.shape[0]).to(torch.float32)
            s_next_b = s_next.expand(latents.shape[0]).to(torch.float32)

            cos_s = torch.cos(s_b).view(-1, 1, 1, 1)
            sin_s = torch.sin(s_b).view(-1, 1, 1, 1)

            tau = tau_from_s(s_b).clamp(1e-6, 1 - 1e-6)

            tau_4d = tau.view(-1, 1, 1, 1)
            r = r_from_tau(tau)

            latents_model_input = latents / sigma_d  # x_norm
            x_tau = latents_model_input * r

            if reward_scale > 0 and i == 0:
                x_tau = x_tau.detach().requires_grad_(True)
                context_mngr = torch.enable_grad()
            else:
                context_mngr = torch.no_grad()

            with context_mngr:
                model_pred = self.transformer(
                    x_tau.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=guidance,
                    timestep=tau.to(dtype=transformer_dtype),
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                )[0]
                x0_hat = (cos_s - sin_s * (1 - 2 * tau_4d)) * latents - sin_s * (
                    1 - 2 * tau_4d + 2 * tau_4d**2
                ) * sigma_d * model_pred / r

                image = self.vae.decode(
                    (x0_hat / sigma_d).to(self.vae.dtype)
                    / self.vae.config.scaling_factor,
                    return_dict=False,
                )[0]
                image = (image / 2 + 0.5).clamp(0, 1)

                preprocessed_img = self.img_transform(image)

                # Reward
                raw_reward = 0.0
                for reward_fn in reward_fns:
                    reward = reward_fn(preprocessed_img, prompt, image)
                    print(f"Reward {reward_fn.name}: {reward.item():.4f}")
                    raw_reward = raw_reward + reward * reward_fn.weighting
                print(f"Reward total iter{i}: {raw_reward.item():.4f}")

            maybe_update_best_candidate(
                float(raw_reward.item()),
                image,
                "guided_main_path",
            )

            k = cos_s + sin_s  # (B,1,1,1)
            half_s = (s_b / 2).view(-1, 1, 1, 1)
            tan_half_s = torch.tan(half_s)

            C1 = k * tan_half_s + cos_s - sin_s  # = (1-A)/tau, computed stably
            C2 = k * r  # = B/tau, computed stably

            velocity = C1 * latents + C2 * sigma_d * model_pred

            max_reward = float("-inf")
            do_reno_step = False

            if (
                reward_scale > 0
                and tau.mean().item() < 0.99
                and guidance_start_step <= i < guidance_start_step + num_guidance_steps
            ):
                torch.cuda.empty_cache()
                total_grad, max_reward, max_image, mean_reward = (
                    compute_reward_gradient_multi_particle(
                        latents_input=latents,
                        tau=tau,
                        prompt_embeds=prompt_embeds,
                        x0_hat_t=x0_hat,
                        prompt_attention_mask=prompt_attention_mask,
                        num_particles=num_reward_particles,
                        reward_scale=reward_scale,
                        guidance=guidance,
                        snr_factor=snr_factor,
                        gradient_norm_scale=gradient_norm_scale,
                        include_weights=include_weights,
                    )
                )

                b = compute_b_coefficient_from_s(s_b, sigma_d)

                grad_norm_reward = total_grad.norm().item()
                vel_norm = velocity.norm().item()

                print(
                    f"Step {i + 1}: mean_R={mean_reward.item():.4f}, max_R={max_reward}, ||∇R||={grad_norm_reward:.4f}, ||velocity||={vel_norm:.4f}, "
                    f" τ={tau.mean().item():.4f}, b={b.mean().item():.4f}, s={s_b.mean().item():.4f}"
                )

                velocity = velocity - b * total_grad

                mean_reward_history.append(mean_reward.item())
                reward_history.append(raw_reward.item())
                if max_reward > best_reward:
                    print(f"Reward particles generated highes reward {max_reward}")
                    maybe_update_best_candidate(
                        float(max_reward),
                        max_image,
                        "guided_particle",
                    )
            elif reward_scale > 0 and i == 0 and do_reno_step:
                reward_grad = torch.autograd.grad(
                    outputs=(raw_reward * reward_scale).sum(),
                    inputs=x_tau,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                b = compute_b_coefficient_from_s(s_b, sigma_d)

                reward_grad_norm = reward_grad.norm().clamp_min(1e-8)
                grad_norm_reward = reward_grad_norm.item()
                reward_grad = reward_grad / reward_grad_norm * gradient_norm_scale
                vel_norm = velocity.norm().item()

                print(
                    f"Step {i + 1}: ||∇R||={grad_norm_reward:.4f}, ||velocity||={vel_norm:.4f}, "
                    f" τ={tau.mean().item():.4f}, b={b.mean().item():.4f}, s={s_b.mean().item():.4f}"
                )

                velocity = velocity - b * reward_grad

            latents_t = latents.detach().clone()

            # Euler step
            s_curr_4d = s_b.view(-1, 1, 1, 1)
            s_next_4d = s_next_b.view(-1, 1, 1, 1)
            ds = s_next_4d - s_curr_4d  # This is negative (going s_max → 0)
            latents = latents + ds * velocity

            jumped_to_end = False
            if (
                jump_to_end_after_guidance
                and reward_scale > 0
                and guidance_start_step <= i < guidance_end_step
                and (i + 1) == guidance_end_step
                and guidance_end_step < num_inference_steps
            ):
                tau_next = tau_from_s(s_next_b).clamp(1e-6, 1 - 1e-6)
                with torch.no_grad():
                    x0_hat = compute_model_prediction(
                        latents,
                        tau_next,
                        prompt_embeds,
                        prompt_attention_mask,
                        guidance,
                    ).detach()
                latents_t = latents.detach().clone()
                jumped_to_end = True

            if save_intermediate_imgs:
                x0_hat_history.append(x0_hat.detach().clone())
                latents_history.append(latents_t)
                tau_history.append(tau.mean().item())

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
                # Decode x0_hat
                with torch.no_grad():
                    x0_decoded = self.vae.decode(
                        (x0_h / sigma_d).to(self.vae.dtype)
                        / self.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                    x0_img = (x0_decoded / 2 + 0.5).clamp(0, 1)
                    x0_img = x0_img[0].permute(1, 2, 0).float().detach().cpu().numpy()
                    x0_img = (x0_img * 255).astype(np.uint8)

                    # Decode latent
                    lat_decoded = self.vae.decode(
                        (lat_h / sigma_d).to(self.vae.dtype)
                        / self.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                    lat_img = (lat_decoded / 2 + 0.5).clamp(0, 1)
                    lat_img = lat_img[0].permute(1, 2, 0).float().detach().cpu().numpy()
                    lat_img = (lat_img * 255).astype(np.uint8)

                Image.fromarray(x0_img).save(f"{x0_dir}/step_{step_idx:03d}.png")
                Image.fromarray(lat_img).save(f"{xt_dir}/step_{step_idx:03d}.png")

        latents = x0_hat / self.scheduler.config.sigma_data

        latents = latents.to(self.vae.dtype)
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        if reward_scale > 0 and reward_fns is not None:
            final_preprocessed_img = self.img_transform(image)
            final_raw_reward = 0.0
            for reward_fn in reward_fns:
                reward = reward_fn(final_preprocessed_img, prompt, image)
                final_raw_reward = final_raw_reward + reward * reward_fn.weighting
            maybe_update_best_candidate(
                float(final_raw_reward.detach().item()),
                image,
                "final_sample",
            )

        # save output image
        image = image[0].permute(1, 2, 0).detach().cpu().float().numpy()
        image = (image * 255).clip(0, 255).astype("uint8")
        if best_image is None:
            best_image = image
        else:
            best_image = (best_image * 255).clip(0, 255).astype("uint8")
        if save_outputs:
            Image.fromarray(image).save(f"{save_path}/output.png")
            Image.fromarray(best_image).save(
                f"{save_path}/best_image_reward{best_reward}.png"
            )
        return best_image
