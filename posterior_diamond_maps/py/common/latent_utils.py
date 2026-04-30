"""
Utilities for latent-space datasets that require VAE decode for evaluation.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple

from ml_collections import config_dict

LATENT_DECODE_CHUNK = 32


def force_xla_gpu_deterministic_ops() -> None:
    flag = "--xla_gpu_deterministic_ops=true"
    flags = [
        x
        for x in os.environ.get("XLA_FLAGS", "").split()
        if x not in {flag, "--xla_gpu_deterministic_ops=false"}
    ]
    os.environ["XLA_FLAGS"] = " ".join([*flags, flag])


def is_latent_target(cfg: config_dict.ConfigDict) -> bool:
    from . import datasets

    return datasets.is_imagenet_latent_target(cfg.problem.target)


def get_pixel_image_dims(cfg: config_dict.ConfigDict) -> Tuple[int, int, int]:
    if not is_latent_target(cfg):
        return tuple(cfg.problem.image_dims)

    resolution = int(cfg.problem.image_dims[1]) * 8
    return (3, resolution, resolution)


def _get_vae_type(cfg: config_dict.ConfigDict) -> str:
    return str(cfg.problem.latent_vae_type)


def _get_latent_scale(cfg: config_dict.ConfigDict) -> float:
    return float(cfg.problem.latent_scale)


def get_decode_fn(
    cfg: config_dict.ConfigDict,
) -> Optional[Callable[[Any], Any]]:
    if not is_latent_target(cfg):
        return None

    import jax
    vae_type = _get_vae_type(cfg)
    print(f"Loading VAE decoder: {vae_type}")
    from diffusers.models import FlaxAutoencoderKL
    from diffusers.utils import logging as diffusers_logging

    diffusers_logging.set_verbosity_error()
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{vae_type}", from_pt=True
    )

    @jax.jit
    def decode_fn(latents_nchw):
        out = vae.apply(
            {"params": vae_params},
            latents_nchw,
            method=FlaxAutoencoderKL.decode,
        )
        return out.sample

    return decode_fn


def maybe_decode_latents_chunked(
    cfg: config_dict.ConfigDict,
    xs: Any,
    chunk_size: int,
    decode_fn: Optional[Callable[[Any], Any]],
) -> Any:
    import jax.numpy as jnp

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not is_latent_target(cfg):
        return xs
    if decode_fn is None:
        raise ValueError("decode_fn must be provided for latent targets.")

    xs = jnp.asarray(xs, dtype=jnp.float32)
    if xs.shape[-3] != int(cfg.problem.image_dims[0]):
        return xs

    xs, lead_shape = jnp.reshape(xs, (-1, *xs.shape[-3:])), xs.shape[:-3]
    latent_scale = _get_latent_scale(cfg)
    if xs.shape[0] <= chunk_size:
        decoded = decode_fn(xs / latent_scale)
        return jnp.reshape(decoded, (*lead_shape, *decoded.shape[-3:]))

    decoded_chunks = []
    for start in range(0, xs.shape[0], chunk_size):
        decoded_chunks.append(decode_fn(xs[start : start + chunk_size] / latent_scale))
    decoded = jnp.concatenate(decoded_chunks, axis=0)
    return jnp.reshape(decoded, (*lead_shape, *decoded.shape[-3:]))


def decode_batch_to_nhwc(
    cfg: config_dict.ConfigDict,
    batch: Any,
    decode_fn: Optional[Callable[[Any], Any]],
    *,
    unnormalize: bool = False,
    chunk_size: int = LATENT_DECODE_CHUNK,
) -> Any:
    import jax.numpy as jnp

    leading_shape = batch.shape[:-3]
    flat_batch = jnp.reshape(batch, (-1, *batch.shape[-3:]))
    flat_decoded = maybe_decode_latents_chunked(
        cfg,
        flat_batch,
        chunk_size=chunk_size,
        decode_fn=decode_fn,
    )
    decoded = jnp.reshape(flat_decoded, (*leading_shape, *flat_decoded.shape[1:]))
    channel_axis = len(leading_shape)
    axes = tuple(range(channel_axis)) + (
        channel_axis + 1,
        channel_axis + 2,
        channel_axis,
    )
    decoded = jnp.transpose(decoded, axes)
    if unnormalize:
        from . import datasets

        decoded = datasets.unnormalize_image(decoded)
    return decoded
