"""Guidance, prompt-alignment, FID, KID, and LPIPS metric helpers."""

from typing import Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from . import dist_utils, fid_utils
from .rewards import registry as reward_registry


SUPPORTED_PROMPT_METRIC_REWARDS = reward_registry.supported_prompt_rewards()
_PROMPT_METRIC_KEY_PREFIXES = {
    "imagereward": "image_reward",
    "clip": "clip",
    "hpsv2": "hpsv2",
    "pickscore": "pickscore",
    "composite": "composite",
}


def normalize_prompt_metric_rewards(
    raw_rewards: Optional[str],
    *,
    default_reward: str,
) -> tuple[str, ...]:
    if raw_rewards is None:
        raw_rewards = default_reward
    raw_rewards = raw_rewards.strip()
    if not raw_rewards:
        raw_rewards = default_reward
    rewards = [reward_name.strip().lower() for reward_name in raw_rewards.split(",")]
    assert all(
        [reward in SUPPORTED_PROMPT_METRIC_REWARDS for reward in rewards]
    ), f"Reward must be in list of supported rewards {SUPPORTED_PROMPT_METRIC_REWARDS}"
    return tuple(sorted(set(rewards)))  # canonicalize order


def _build_prompt_metric_outputs(
    prompt_metric_rewards: Sequence[str],
    totals: Dict[str, float],
    maxes: Dict[str, float],
    denom: int,
) -> Dict[str, float]:
    metric_vals = {}
    for reward_name in prompt_metric_rewards:
        key_prefix = _PROMPT_METRIC_KEY_PREFIXES[reward_name]
        metric_vals[f"{key_prefix}_mean"] = totals[reward_name] / denom
        metric_vals[f"{key_prefix}_max"] = maxes[reward_name]
    return metric_vals


def _build_prompt_metric_runtimes(
    prompt_metric_rewards: Sequence[str],
    prompt: str,
):
    return {
        reward_name: reward_registry.build_prompt_reward(reward_name, prompt)
        for reward_name in prompt_metric_rewards
    }


def load_lpips_j_model():
    import lpips_j.lpips as lpips_j

    model = lpips_j.LPIPS()
    dummy = jnp.zeros((1, 64, 64, 3), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), dummy, dummy)
    return model, params


def _polynomial_kernel_np(
    x: np.ndarray, y: np.ndarray, degree: int = 3, coef: float = 1.0
) -> np.ndarray:
    gamma = 1.0 / x.shape[1]
    return (gamma * (x @ y.T) + coef) ** degree


def _kernel_sum_blocks(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    symmetric: bool,
) -> float:
    total = 0.0
    if symmetric:
        for i in range(0, x.shape[0], block_size):
            x_i = x[i : i + block_size]
            for j in range(0, i + 1, block_size):
                y_j = y[j : j + block_size]
                k_block = _polynomial_kernel_np(x_i, y_j)
                if i == j:
                    total += float(np.sum(k_block) - np.trace(k_block))
                else:
                    total += float(np.sum(k_block)) * 2.0
    else:
        for i in range(0, x.shape[0], block_size):
            x_i = x[i : i + block_size]
            for j in range(0, y.shape[0], block_size):
                y_j = y[j : j + block_size]
                total += float(np.sum(_polynomial_kernel_np(x_i, y_j)))
    return total


def _kid_from_activations_streaming(
    act_real: np.ndarray, act_fake: np.ndarray, block_size: int
) -> float:
    if act_real.shape[0] < 2 or act_fake.shape[0] < 2:
        raise ValueError("KID requires at least 2 samples per set.")
    act_real = np.asarray(act_real, dtype=np.float64)
    act_fake = np.asarray(act_fake, dtype=np.float64)
    sum_xx = _kernel_sum_blocks(act_real, act_real, block_size, symmetric=True)
    sum_yy = _kernel_sum_blocks(act_fake, act_fake, block_size, symmetric=True)
    sum_xy = _kernel_sum_blocks(act_real, act_fake, block_size, symmetric=False)
    m = act_real.shape[0]
    n = act_fake.shape[0]
    return sum_xx / (m * (m - 1)) + sum_yy / (n * (n - 1)) - 2.0 * sum_xy / (m * n)


def calc_guidance_metrics(
    cfg,
    lpips_j_model,
    lpips_j_params: Optional[Dict],
    inception_fn: Callable,
    sample_pair_fn: Callable,
    prng_key: jnp.ndarray,
    variables: Dict,
    n_samples: int,
    bs: int,
) -> Tuple[Optional[float], Optional[float]]:
    compute_lpips: bool = lpips_j_model is not None and lpips_j_params is not None
    compute_kid: bool = inception_fn is not None

    if not compute_lpips and not compute_kid:
        return None, None

    lpips_j_params_rep = (
        dist_utils.safe_replicate(cfg, lpips_j_params) if compute_lpips else None
    )
    desc = "Computing LPIPS-j/KID"
    if compute_lpips and not compute_kid:
        desc = "Computing LPIPS-j"
    elif compute_kid and not compute_lpips:
        desc = "Computing KID"

    n_batches = (n_samples + bs - 1) // bs
    lpips_j_total = 0.0
    act_0_chunks = []
    act_1_chunks = []
    count = 0
    for _ in tqdm(range(n_batches), desc=desc):
        prng_key, step_key = jax.random.split(prng_key)
        images_0, images_1 = sample_pair_fn(step_key, variables, bs, sharded=True)
        images_0 = jnp.clip(images_0, 0.0, 1.0)
        images_1 = jnp.clip(images_1, 0.0, 1.0)

        if compute_lpips:
            images_0_lpips_j = images_0 * 2.0 - 1.0
            images_1_lpips_j = images_1 * 2.0 - 1.0
            lpips_j_vals = lpips_j_model.apply(
                lpips_j_params_rep, images_0_lpips_j, images_1_lpips_j
            )
            lpips_j_vals = jnp.reshape(lpips_j_vals, (lpips_j_vals.shape[0], -1))
            lpips_j_vals = jnp.mean(lpips_j_vals, axis=1)
            lpips_j_total += float(jnp.sum(lpips_j_vals))

        if compute_kid:
            images_0_fid = images_0 * 2.0 - 1.0
            images_1_fid = images_1 * 2.0 - 1.0
            act_0 = fid_utils.resize_and_incept(images_0_fid, inception_fn)
            act_1 = fid_utils.resize_and_incept(images_1_fid, inception_fn)
            act_0 = np.asarray(jnp.reshape(act_0, (act_0.shape[0], -1)))
            act_1 = np.asarray(jnp.reshape(act_1, (act_1.shape[0], -1)))
            act_0_chunks.append(act_0)
            act_1_chunks.append(act_1)

        count += int(images_0.shape[0])

    denom = max(count, 1)
    lpips_j_mean = lpips_j_total / denom if compute_lpips else None
    kid_mean = None
    if compute_kid:
        act_0_all = np.concatenate(act_0_chunks, axis=0)
        act_1_all = np.concatenate(act_1_chunks, axis=0)
        kid_block_size = bs
        kid_mean = _kid_from_activations_streaming(
            act_0_all, act_1_all, block_size=kid_block_size
        )
    return lpips_j_mean, kid_mean


def calc_prompt_alignment_metrics(
    sample_fn: Callable,
    prng_key: jnp.ndarray,
    variables: Dict,
    prompt: str,
    n_samples: int,
    bs: int,
    prompt_metric_rewards: Sequence[str],
) -> Dict[str, float]:
    reward_runtimes = _build_prompt_metric_runtimes(prompt_metric_rewards, prompt)

    n_batches = (n_samples + bs - 1) // bs
    totals = {reward_name: 0.0 for reward_name in prompt_metric_rewards}
    maxes = {reward_name: float("-inf") for reward_name in prompt_metric_rewards}
    count = 0
    for _ in tqdm(range(n_batches), desc="Computing prompt rewards"):
        prng_key, step_key = jax.random.split(prng_key)
        images = sample_fn(step_key, variables, bs, sharded=True)
        images = jnp.clip(images, 0.0, 1.0)
        for reward_name in prompt_metric_rewards:
            score_fn, reward_params = reward_runtimes[reward_name]
            reward_vals = score_fn(images, reward_params)
            totals[reward_name] += float(jnp.sum(reward_vals))
            maxes[reward_name] = max(maxes[reward_name], float(jnp.max(reward_vals)))
        count += int(images.shape[0])

    denom = max(count, 1)
    return _build_prompt_metric_outputs(prompt_metric_rewards, totals, maxes, denom)


def calc_prompt_alignment_metrics_from_images(
    images,
    prompt: str,
    bs: int,
    prompt_metric_rewards: Sequence[str],
) -> Dict[str, float]:
    reward_runtimes = _build_prompt_metric_runtimes(prompt_metric_rewards, prompt)

    n_samples = len(images)
    totals = {reward_name: 0.0 for reward_name in prompt_metric_rewards}
    maxes = {reward_name: float("-inf") for reward_name in prompt_metric_rewards}
    count = 0

    for start in tqdm(range(0, n_samples, bs), desc="Computing prompt rewards"):
        batch_images = jnp.asarray(images[start : start + bs])
        batch_images = jnp.clip(batch_images, 0.0, 1.0)
        for reward_name in prompt_metric_rewards:
            score_fn, reward_params = reward_runtimes[reward_name]
            reward_vals = score_fn(batch_images, reward_params)
            totals[reward_name] += float(jnp.sum(reward_vals))
            maxes[reward_name] = max(maxes[reward_name], float(jnp.max(reward_vals)))
        count += int(batch_images.shape[0])

    denom = max(count, 1)
    return _build_prompt_metric_outputs(prompt_metric_rewards, totals, maxes, denom)
