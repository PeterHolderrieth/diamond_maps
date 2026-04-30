"""CLIP-score prompt reward helpers."""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax.numpy as jnp

from . import openclip


def get_clip_text_embed(prompt: str) -> jnp.ndarray:
    model, params = openclip.get_base_flax_clip()
    return get_clip_text_embed_with_params(prompt, params, model=model)


def clip_score_diff(pixel_images: jnp.ndarray, text_embed: jnp.ndarray) -> jnp.ndarray:
    model, params = openclip.get_base_flax_clip()
    return clip_score_diff_with_params(
        pixel_images,
        text_embed,
        params,
        model=model,
    )


def get_clip_text_embed_with_params(
    prompt: str,
    params,
    *,
    model: Optional[Any] = None,
) -> jnp.ndarray:
    if model is None:
        model, _ = openclip.get_base_flax_clip()
    input_ids, attention_mask = openclip.tokenize_openclip_prompt(prompt)
    text_features = openclip.get_text_features(model, params, input_ids, attention_mask)
    return openclip.normalize_l2(text_features, axis=-1)[0]


def clip_score_diff_with_params(
    pixel_images: jnp.ndarray,
    text_embed: jnp.ndarray,
    params,
    *,
    model: Optional[Any] = None,
) -> jnp.ndarray:
    if model is None:
        model, _ = openclip.get_base_flax_clip()
    image_embeds = openclip.get_image_features(model, params, pixel_images)
    image_embeds = openclip.normalize_l2(image_embeds, axis=-1)
    return image_embeds @ jnp.reshape(text_embed, (-1,))


def build_clip_reward(
    prompt: str,
) -> tuple[Callable[[jnp.ndarray, Any], jnp.ndarray], Any]:
    model, params = openclip.get_base_flax_clip()
    text_embed = get_clip_text_embed_with_params(prompt, params, model=model)

    def score_fn(pixel_images: jnp.ndarray, runtime_params) -> jnp.ndarray:
        return clip_score_diff_with_params(
            pixel_images,
            text_embed,
            runtime_params,
            model=model,
        )

    return score_fn, params
