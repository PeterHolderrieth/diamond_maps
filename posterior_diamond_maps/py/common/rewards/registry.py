"""Factory registry for supported prompt reward functions."""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

from . import clip_score, composite, hpsv2, imagereward, pickscore


_SUPPORTED_PROMPT_REWARDS = (
    "imagereward",
    "clip",
    "hpsv2",
    "pickscore",
    "composite",
)


def supported_prompt_rewards() -> tuple[str, ...]:
    return _SUPPORTED_PROMPT_REWARDS


def build_prompt_reward(
    prompt_reward: str,
    prompt: str,
) -> tuple[Callable[[jnp.ndarray, Any], jnp.ndarray], Any]:
    prompt_reward = str(prompt_reward).lower()
    if prompt_reward == "imagereward":
        return imagereward.build_imagereward_reward(prompt)
    if prompt_reward == "clip":
        return clip_score.build_clip_reward(prompt)
    if prompt_reward == "hpsv2":
        return hpsv2.build_hpsv2_reward(prompt, version="v2.1")
    if prompt_reward == "pickscore":
        return pickscore.build_pickscore_reward(prompt)
    if prompt_reward == "composite":
        return composite.build_composite_reward(prompt)
    raise ValueError(f"Unsupported prompt reward: {prompt_reward}")
