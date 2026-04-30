"""PickScore prompt reward helpers backed by CLIP embeddings."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import jax.numpy as jnp

from . import openclip


class PickScorePromptData(NamedTuple):
    text_embed: jnp.ndarray
    logit_scale: jnp.ndarray


def get_pickscore_prompt_data(prompt: str) -> PickScorePromptData:
    model, params = openclip.get_pickscore_flax_clip()
    return get_pickscore_prompt_data_with_params(prompt, params, model=model)


def pickscore_score_diff(
    pixel_images: jnp.ndarray,
    prompt_data: PickScorePromptData,
) -> jnp.ndarray:
    model, params = openclip.get_pickscore_flax_clip()
    return pickscore_score_diff_with_params(
        pixel_images,
        prompt_data,
        params,
        model=model,
    )


def get_pickscore_prompt_data_with_params(
    prompt: str,
    params,
    *,
    model: Optional[Any] = None,
) -> PickScorePromptData:
    if model is None:
        model, _ = openclip.get_pickscore_flax_clip()
    input_ids, attention_mask = openclip.tokenize_openclip_prompt(prompt)
    text_features = openclip.get_text_features(model, params, input_ids, attention_mask)
    text_features = openclip.normalize_l2(text_features, axis=-1)[0]
    logit_scale = jnp.exp(jnp.asarray(params["logit_scale"]))
    return PickScorePromptData(text_embed=text_features, logit_scale=logit_scale)


def pickscore_score_diff_with_params(
    pixel_images: jnp.ndarray,
    prompt_data: PickScorePromptData,
    params,
    *,
    model: Optional[Any] = None,
) -> jnp.ndarray:
    if model is None:
        model, _ = openclip.get_pickscore_flax_clip()
    image_embeds = openclip.get_image_features(model, params, pixel_images)
    image_embeds = openclip.normalize_l2(image_embeds, axis=-1)
    scores = image_embeds @ jnp.reshape(prompt_data.text_embed, (-1,))
    return prompt_data.logit_scale * scores


def build_pickscore_reward(
    prompt: str,
) -> tuple[Callable[[jnp.ndarray, Any], jnp.ndarray], Any]:
    model, params = openclip.get_pickscore_flax_clip()
    prompt_data = get_pickscore_prompt_data_with_params(prompt, params, model=model)

    def score_fn(pixel_images: jnp.ndarray, runtime_params) -> jnp.ndarray:
        return pickscore_score_diff_with_params(
            pixel_images,
            prompt_data,
            runtime_params,
            model=model,
        )

    return score_fn, params
