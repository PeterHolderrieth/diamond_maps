"""HPSv2 prompt reward helpers backed by OpenCLIP features."""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax.numpy as jnp

from . import openclip


def get_hpsv2_text_embed(prompt: str, version: str = "v2.1") -> jnp.ndarray:
    model, params = openclip.get_hpsv2_flax_clip(version=version)
    return get_hpsv2_text_embed_with_params(
        prompt,
        params,
        version=version,
        model=model,
    )


def hpsv2_score_diff(
    pixel_images: jnp.ndarray,
    text_embed: jnp.ndarray,
    version: str = "v2.1",
) -> jnp.ndarray:
    model, params = openclip.get_hpsv2_flax_clip(version=version)
    return hpsv2_score_diff_with_params(
        pixel_images,
        text_embed,
        params,
        version=version,
        model=model,
    )


def get_hpsv2_text_embed_with_params(
    prompt: str,
    params,
    *,
    version: str = "v2.1",
    model: Optional[Any] = None,
) -> jnp.ndarray:
    if model is None:
        model, _ = openclip.get_hpsv2_flax_clip(version=version)
    input_ids, attention_mask = openclip.tokenize_openclip_prompt(prompt)
    text_features = openclip.get_text_features(model, params, input_ids, attention_mask)
    return openclip.normalize_l2(text_features, axis=-1)[0]


def hpsv2_score_diff_with_params(
    pixel_images: jnp.ndarray,
    text_embed: jnp.ndarray,
    params,
    *,
    version: str = "v2.1",
    model: Optional[Any] = None,
) -> jnp.ndarray:
    if model is None:
        model, _ = openclip.get_hpsv2_flax_clip(version=version)
    image_embeds = openclip.get_image_features(model, params, pixel_images)
    image_embeds = openclip.normalize_l2(image_embeds, axis=-1)
    return image_embeds @ jnp.reshape(text_embed, (-1,))


def build_hpsv2_reward(
    prompt: str,
    *,
    version: str = "v2.1",
) -> tuple[Callable[[jnp.ndarray, Any], jnp.ndarray], Any]:
    model, params = openclip.get_hpsv2_flax_clip(version=version)
    text_embed = get_hpsv2_text_embed_with_params(
        prompt,
        params,
        version=version,
        model=model,
    )

    def score_fn(pixel_images: jnp.ndarray, runtime_params) -> jnp.ndarray:
        return hpsv2_score_diff_with_params(
            pixel_images,
            text_embed,
            runtime_params,
            version=version,
            model=model,
        )

    return score_fn, params
