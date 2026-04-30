"""Composite prompt reward that combines multiple reward models."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax.numpy as jnp

from . import clip_score, hpsv2, imagereward, pickscore


IMAGE_REWARD_WEIGHT = 1.0
HPSV2_WEIGHT = 5.0
PICKSCORE_WEIGHT = 0.05
CLIP_SCORE_WEIGHT = 1.0


class CompositePromptData(NamedTuple):
    imagereward_input_ids: jnp.ndarray
    imagereward_attention_mask: jnp.ndarray
    hpsv2_text_embed: jnp.ndarray
    pickscore_prompt_data: pickscore.PickScorePromptData
    clip_text_embed: jnp.ndarray


def get_composite_prompt_data(prompt: str) -> CompositePromptData:
    input_ids, attention_mask = imagereward.tokenize_imagereward_prompt(prompt)
    return CompositePromptData(
        imagereward_input_ids=input_ids,
        imagereward_attention_mask=attention_mask,
        hpsv2_text_embed=hpsv2.get_hpsv2_text_embed(prompt, version="v2.1"),
        pickscore_prompt_data=pickscore.get_pickscore_prompt_data(prompt),
        clip_text_embed=clip_score.get_clip_text_embed(prompt),
    )


def composite_score_diff(
    pixel_images: jnp.ndarray,
    prompt_data: CompositePromptData,
) -> jnp.ndarray:
    image_reward_vals = imagereward.imagereward_score_diff(
        pixel_images,
        prompt_data.imagereward_input_ids,
        prompt_data.imagereward_attention_mask,
    )
    hpsv2_vals = hpsv2.hpsv2_score_diff(
        pixel_images, prompt_data.hpsv2_text_embed, version="v2.1"
    )
    pickscore_vals = pickscore.pickscore_score_diff(
        pixel_images, prompt_data.pickscore_prompt_data
    )
    clip_vals = clip_score.clip_score_diff(pixel_images, prompt_data.clip_text_embed)
    return (
        IMAGE_REWARD_WEIGHT * image_reward_vals
        + HPSV2_WEIGHT * hpsv2_vals
        + PICKSCORE_WEIGHT * pickscore_vals
        + CLIP_SCORE_WEIGHT * clip_vals
    )


def build_composite_reward(
    prompt: str,
) -> tuple[Callable[[jnp.ndarray, Any], jnp.ndarray], Any]:
    imagereward_score_fn, imagereward_params = imagereward.build_imagereward_reward(
        prompt
    )
    hpsv2_score_fn, hpsv2_params = hpsv2.build_hpsv2_reward(prompt, version="v2.1")
    pickscore_score_fn, pickscore_params = pickscore.build_pickscore_reward(prompt)
    clip_score_fn, clip_params = clip_score.build_clip_reward(prompt)

    def score_fn(pixel_images: jnp.ndarray, runtime_params) -> jnp.ndarray:
        return (
            IMAGE_REWARD_WEIGHT
            * imagereward_score_fn(pixel_images, runtime_params["imagereward"])
            + HPSV2_WEIGHT * hpsv2_score_fn(pixel_images, runtime_params["hpsv2"])
            + PICKSCORE_WEIGHT
            * pickscore_score_fn(pixel_images, runtime_params["pickscore"])
            + CLIP_SCORE_WEIGHT * clip_score_fn(pixel_images, runtime_params["clip"])
        )

    return score_fn, {
        "imagereward": imagereward_params,
        "hpsv2": hpsv2_params,
        "pickscore": pickscore_params,
        "clip": clip_params,
    }
