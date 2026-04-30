"""Inverse-problem operators and measurements for guided sampling."""

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp

Measurement = Dict[str, jnp.ndarray]


def _downsample_target_shape(x: jnp.ndarray, factor: int) -> tuple[int, int, int]:
    channels, height, width = x.shape
    if height % factor != 0 or width % factor != 0:
        raise ValueError(
            f"Downsample factor {factor} must divide image size {height}x{width}."
        )
    return channels, height // factor, width // factor


def _downsample_bicubic(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    channels, new_h, new_w = _downsample_target_shape(x, factor)
    if factor == 1:
        return x
    x_hwc = jnp.transpose(x, (1, 2, 0))
    resized = jax.image.resize(
        x_hwc,
        (new_h, new_w, channels),
        method="cubic",
        antialias=True,
    )
    return jnp.transpose(resized, (2, 0, 1))


def _upsample_nn(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    if factor == 1:
        return x
    x = jnp.repeat(x, factor, axis=1)
    x = jnp.repeat(x, factor, axis=2)
    return x


class InverseProblem:
    def make_measurement(self, x: jnp.ndarray, key: jnp.ndarray) -> Measurement:
        raise NotImplementedError

    def residual(self, x: jnp.ndarray, measurement: Measurement) -> jnp.ndarray:
        raise NotImplementedError

    def residual_sq(self, x: jnp.ndarray, measurement: Measurement) -> jnp.ndarray:
        diff = self.residual(x, measurement)
        return jnp.sum(diff**2)

    def reward(self, x: jnp.ndarray, measurement: Measurement) -> jnp.ndarray:
        raise NotImplementedError

    def batch_make_measurement(self, xs: jnp.ndarray, key: jnp.ndarray) -> Measurement:
        keys = jax.random.split(key, xs.shape[0])
        return jax.vmap(self.make_measurement, in_axes=(0, 0))(xs, keys)


@dataclass(frozen=True)
class InpaintingProblem(InverseProblem):
    mask_prob: float
    noise_std: float

    def _sample_mask(
        self, key: jnp.ndarray, shape: tuple[int, int, int]
    ) -> jnp.ndarray:
        mask = jax.random.bernoulli(key, p=1.0 - self.mask_prob, shape=shape)
        return mask.astype(jnp.float32)

    def make_measurement(self, x: jnp.ndarray, key: jnp.ndarray) -> Measurement:
        key_mask, key_noise = jax.random.split(key)
        mask_shape = (1, x.shape[-2], x.shape[-1])
        mask = self._sample_mask(key_mask, mask_shape)
        mask = jnp.broadcast_to(mask, x.shape)
        noise = self.noise_std * jax.random.normal(key_noise, x.shape)
        y = mask * x + mask * noise
        return {"y": y, "mask": mask, "y_vis": y}

    def residual(self, x: jnp.ndarray, measurement: Measurement) -> jnp.ndarray:
        return (x - measurement["y"]) * measurement["mask"]

    def reward(self, x: jnp.ndarray, measurement: Measurement) -> jnp.ndarray:
        sigma2 = jnp.maximum(self.noise_std**2, 1e-8)
        return -0.5 * self.residual_sq(x, measurement) / sigma2


@dataclass(frozen=True)
class SuperResolutionProblem(InverseProblem):
    scale_factor: int
    noise_std: float

    def make_measurement(self, x: jnp.ndarray, key: jnp.ndarray) -> Measurement:
        y = _downsample_bicubic(x, self.scale_factor)
        noise = self.noise_std * jax.random.normal(key, y.shape)
        y = y + noise
        y_vis = _upsample_nn(y, self.scale_factor)
        return {"y": y, "y_vis": y_vis}

    def residual(self, x: jnp.ndarray, measurement: Measurement) -> jnp.ndarray:
        return _downsample_bicubic(x, self.scale_factor) - measurement["y"]

    def reward(self, x: jnp.ndarray, measurement: Measurement) -> jnp.ndarray:
        sigma2 = jnp.maximum(self.noise_std**2, 1e-8)
        return -0.5 * self.residual_sq(x, measurement) / sigma2


def make_inverse_problem(
    problem_type: str,
    mask_prob: float,
    noise_std: float,
    sr_factor: int,
) -> InverseProblem:
    if problem_type == "inpainting":
        return InpaintingProblem(mask_prob=mask_prob, noise_std=noise_std)
    if problem_type == "super_resolution":
        return SuperResolutionProblem(scale_factor=sr_factor, noise_std=noise_std)
    raise ValueError(f"Unknown inverse problem type: {problem_type}")
