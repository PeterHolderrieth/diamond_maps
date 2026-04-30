"""ImageReward prompt reward helpers and preprocessing."""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

IR_IMAGE_SIZE = 224
IR_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IR_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def load_imagereward_model_and_params():
    from common import image_reward_jax as image_reward

    model, params, _, _ = image_reward.load()
    return model, params


def _get_ir_tokenizer():
    from common import image_reward_jax as image_reward

    return image_reward._get_tokenizer()


def tokenize_imagereward_prompt(prompt: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    tokenizer = _get_ir_tokenizer()
    encoded = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="np",
    )
    input_ids = jnp.asarray(encoded["input_ids"][0], dtype=jnp.int32)
    attention_mask = jnp.asarray(encoded["attention_mask"][0], dtype=jnp.int32)
    return input_ids, attention_mask


def _preprocess_ir_diff(pixel_images: jnp.ndarray) -> jnp.ndarray:
    batch_size, height, width, _ = pixel_images.shape
    pixel_images = jnp.clip(pixel_images, 0.0, 1.0)
    if height != IR_IMAGE_SIZE or width != IR_IMAGE_SIZE:
        scale = IR_IMAGE_SIZE / min(height, width)
        resized_height = max(int(round(height * scale)), IR_IMAGE_SIZE)
        resized_width = max(int(round(width * scale)), IR_IMAGE_SIZE)
        pixel_images = jax.image.resize(
            pixel_images,
            shape=(batch_size, resized_height, resized_width, 3),
            method="bicubic",
            antialias=True,
        )
        top = max((resized_height - IR_IMAGE_SIZE) // 2, 0)
        left = max((resized_width - IR_IMAGE_SIZE) // 2, 0)
        pixel_images = pixel_images[
            :, top : top + IR_IMAGE_SIZE, left : left + IR_IMAGE_SIZE, :
        ]
    pixel_images = (pixel_images - IR_MEAN) / IR_STD
    return jnp.transpose(pixel_images, (0, 3, 1, 2))


def imagereward_score_diff(
    pixel_images: jnp.ndarray,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
) -> jnp.ndarray:
    model, params = load_imagereward_model_and_params()
    return imagereward_score_diff_with_params(
        pixel_images,
        input_ids,
        attention_mask,
        params,
        model=model,
    )


def imagereward_score_diff_with_params(
    pixel_images: jnp.ndarray,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
    params,
    *,
    model: Optional[Any] = None,
) -> jnp.ndarray:
    if model is None:
        model, _ = load_imagereward_model_and_params()
    pixel_values = _preprocess_ir_diff(pixel_images)
    batch_size = pixel_values.shape[0]
    batch_input_ids = jnp.broadcast_to(
        jnp.asarray(input_ids, dtype=jnp.int32),
        (batch_size, input_ids.shape[-1]),
    )
    batch_attention_mask = jnp.broadcast_to(
        jnp.asarray(attention_mask, dtype=jnp.int32),
        (batch_size, attention_mask.shape[-1]),
    )
    return model.apply(
        params, pixel_values, batch_input_ids, batch_attention_mask
    ).astype(jnp.float32)


def build_imagereward_reward(
    prompt: str,
) -> tuple[Callable[[jnp.ndarray, Any], jnp.ndarray], Any]:
    model, params = load_imagereward_model_and_params()
    input_ids, attention_mask = tokenize_imagereward_prompt(prompt)

    def score_fn(pixel_images: jnp.ndarray, runtime_params) -> jnp.ndarray:
        return imagereward_score_diff_with_params(
            pixel_images,
            input_ids,
            attention_mask,
            runtime_params,
            model=model,
        )

    return score_fn, params
