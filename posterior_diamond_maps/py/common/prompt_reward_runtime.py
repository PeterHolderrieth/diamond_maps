"""Runtime loaders and adapters for prompt-based reward models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp


PromptRewardScoreFn = Callable[[jnp.ndarray, Any, Any], jnp.ndarray]


@dataclass(frozen=True)
class PromptRewardRuntime:
    score_fn: PromptRewardScoreFn
    prompt_data_by_index: dict[int, Any]
    reward_params: Any


_SUPPORTED_PROMPT_REWARDS = (
    "imagereward",
    "clip",
    "hpsv2",
    "pickscore",
    "composite",
)


def supported_prompt_rewards() -> tuple[str, ...]:
    return _SUPPORTED_PROMPT_REWARDS


def _check_reward_allowed(
    prompt_reward: str,
    allowed_rewards: Optional[Sequence[str]],
) -> None:
    supported_rewards = supported_prompt_rewards()
    if prompt_reward not in supported_rewards:
        raise ValueError(f"Unsupported prompt reward: {prompt_reward}")
    if allowed_rewards is not None and prompt_reward not in allowed_rewards:
        raise ValueError(
            f"Prompt reward must be one of {tuple(allowed_rewards)}; "
            f"got {prompt_reward!r}"
        )


def build_prompt_reward_runtime(
    prompt_reward: str,
    prompt_entries,
    *,
    allowed_rewards: Optional[Sequence[str]] = None,
) -> PromptRewardRuntime:
    prompt_reward = str(prompt_reward).lower()
    _check_reward_allowed(prompt_reward, allowed_rewards)

    if prompt_reward == "clip":
        from .rewards import clip_score, openclip

        model, reward_params = openclip.get_base_flax_clip()
        prompt_data = {
            prompt_index: clip_score.get_clip_text_embed_with_params(
                prompt, reward_params, model=model
            )
            for prompt_index, (prompt, _class_label) in prompt_entries
        }

        def score_fn(pixel_images, runtime_params, text_embed):
            return clip_score.clip_score_diff_with_params(
                pixel_images,
                text_embed,
                runtime_params,
                model=model,
            )

        return PromptRewardRuntime(score_fn, prompt_data, reward_params)

    if prompt_reward == "hpsv2":
        from .rewards import hpsv2, openclip

        model, reward_params = openclip.get_hpsv2_flax_clip(version="v2.1")
        prompt_data = {
            prompt_index: hpsv2.get_hpsv2_text_embed_with_params(
                prompt,
                reward_params,
                version="v2.1",
                model=model,
            )
            for prompt_index, (prompt, _class_label) in prompt_entries
        }

        def score_fn(pixel_images, runtime_params, text_embed):
            return hpsv2.hpsv2_score_diff_with_params(
                pixel_images,
                text_embed,
                runtime_params,
                version="v2.1",
                model=model,
            )

        return PromptRewardRuntime(score_fn, prompt_data, reward_params)

    if prompt_reward == "pickscore":
        from .rewards import openclip, pickscore

        model, reward_params = openclip.get_pickscore_flax_clip()
        prompt_data = {
            prompt_index: pickscore.get_pickscore_prompt_data_with_params(
                prompt,
                reward_params,
                model=model,
            )
            for prompt_index, (prompt, _class_label) in prompt_entries
        }

        def score_fn(pixel_images, runtime_params, current_prompt_data):
            return pickscore.pickscore_score_diff_with_params(
                pixel_images,
                current_prompt_data,
                runtime_params,
                model=model,
            )

        return PromptRewardRuntime(score_fn, prompt_data, reward_params)

    if prompt_reward == "composite":
        from .rewards import clip_score, composite as composite_reward
        from .rewards import hpsv2, imagereward, openclip, pickscore

        imagereward_model, imagereward_params = (
            imagereward.load_imagereward_model_and_params()
        )
        hpsv2_model, hpsv2_params = openclip.get_hpsv2_flax_clip(version="v2.1")
        pickscore_model, pickscore_params = openclip.get_pickscore_flax_clip()
        clip_model, clip_params = openclip.get_base_flax_clip()
        prompt_data = {
            prompt_index: {
                "imagereward": imagereward.tokenize_imagereward_prompt(prompt),
                "hpsv2": hpsv2.get_hpsv2_text_embed_with_params(
                    prompt,
                    hpsv2_params,
                    version="v2.1",
                    model=hpsv2_model,
                ),
                "pickscore": pickscore.get_pickscore_prompt_data_with_params(
                    prompt,
                    pickscore_params,
                    model=pickscore_model,
                ),
                "clip": clip_score.get_clip_text_embed_with_params(
                    prompt,
                    clip_params,
                    model=clip_model,
                ),
            }
            for prompt_index, (prompt, _class_label) in prompt_entries
        }
        reward_params = {
            "imagereward": imagereward_params,
            "hpsv2": hpsv2_params,
            "pickscore": pickscore_params,
            "clip": clip_params,
        }

        def score_fn(pixel_images, runtime_params, current_prompt_data):
            input_ids, attention_mask = current_prompt_data["imagereward"]
            imagereward_scores = imagereward.imagereward_score_diff_with_params(
                pixel_images,
                input_ids,
                attention_mask,
                runtime_params["imagereward"],
                model=imagereward_model,
            )
            hpsv2_scores = hpsv2.hpsv2_score_diff_with_params(
                pixel_images,
                current_prompt_data["hpsv2"],
                runtime_params["hpsv2"],
                version="v2.1",
                model=hpsv2_model,
            )
            pickscore_scores = pickscore.pickscore_score_diff_with_params(
                pixel_images,
                current_prompt_data["pickscore"],
                runtime_params["pickscore"],
                model=pickscore_model,
            )
            clip_scores = clip_score.clip_score_diff_with_params(
                pixel_images,
                current_prompt_data["clip"],
                runtime_params["clip"],
                model=clip_model,
            )
            return (
                composite_reward.IMAGE_REWARD_WEIGHT * imagereward_scores
                + composite_reward.HPSV2_WEIGHT * hpsv2_scores
                + composite_reward.PICKSCORE_WEIGHT * pickscore_scores
                + composite_reward.CLIP_SCORE_WEIGHT * clip_scores
            )

        return PromptRewardRuntime(score_fn, prompt_data, reward_params)

    if prompt_reward == "imagereward":
        from .rewards import imagereward

        model, reward_params = imagereward.load_imagereward_model_and_params()
        prompt_data = {
            prompt_index: imagereward.tokenize_imagereward_prompt(prompt)
            for prompt_index, (prompt, _class_label) in prompt_entries
        }

        def score_fn(pixel_images, runtime_params, current_prompt_data):
            input_ids, attention_mask = current_prompt_data
            return imagereward.imagereward_score_diff_with_params(
                pixel_images,
                input_ids,
                attention_mask,
                runtime_params,
                model=model,
            )

        return PromptRewardRuntime(score_fn, prompt_data, reward_params)

    raise ValueError(f"Unsupported prompt reward: {prompt_reward}")


def broadcast_prompt_data(prompt_data, bs: int):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (bs, *jnp.asarray(x).shape)),
        prompt_data,
    )


def make_prompt_measurement(
    prompt_data,
    bs: int,
    *,
    cfg=None,
    replicate: bool = False,
):
    measurement = {"prompt_data": broadcast_prompt_data(prompt_data, bs)}
    if not replicate:
        return measurement
    if cfg is None:
        raise ValueError("cfg is required when replicate=True.")
    from . import dist_utils

    return jax.tree_util.tree_map(
        lambda x: dist_utils.replicate_batch(cfg, x),
        measurement,
    )


def make_latent_prompt_reward_fn(
    cfg,
    decode_fn,
    prompt_reward_score_fn: PromptRewardScoreFn,
):
    from . import latent_utils

    def reward_fn(variables, x: jnp.ndarray, measurement):
        if measurement is None or "prompt_data" not in measurement:
            raise ValueError("Prompt reward_fn requires prompt_data in measurement.")
        if x.ndim != 3:
            raise ValueError(
                f"Prompt reward_fn expects rank-3 CHW input; got shape {x.shape}."
            )
        pixel_images = latent_utils.decode_batch_to_nhwc(
            cfg,
            x[jnp.newaxis, ...],
            decode_fn,
            unnormalize=True,
        )
        rewards = prompt_reward_score_fn(
            pixel_images,
            variables["reward"],
            measurement["prompt_data"],
        )
        return rewards[0]

    return reward_fn
