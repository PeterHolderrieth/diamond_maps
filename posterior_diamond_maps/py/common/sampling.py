"""Sampler construction, guidance, SMC, FID, and visualization utilities."""

import functools
import gc
from enum import Enum
from typing import Callable, Dict, Optional, NamedTuple, Tuple
from matplotlib import pyplot as plt
import jax
import optax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
from ml_collections import config_dict
from tqdm.auto import tqdm

from . import datasets, dist_utils, fid_utils, latent_utils, state_utils

Parameters = Dict[str, Dict]


class SampleType(Enum):
    """Enumeration of sampling methods."""

    FLOW = 0  # flow sampling
    FLOW_MAP = 1  # flow map sampling
    GLASS = 2  # GLASS sampling
    GLASS_EARLY_STOP = 3  # GLASS sampling using early stopping
    DIAMOND_DIAG = 4  # flow sampling using the diagonal of the diamond map
    DIAMOND_EARLY_STOP = 5  # diamond map sampling using early stopping
    FULL_DIAMOND = 6  # full diamond map (t_prime not fixed to 1)
    DIAMOND_RENOISE = 7  # diamond map sampling using renoising

DIAMOND_SAMPLE_TYPES = {
    SampleType.DIAMOND_DIAG,
    SampleType.DIAMOND_EARLY_STOP,
    SampleType.FULL_DIAMOND,
    SampleType.DIAMOND_RENOISE,
}


def _has_sup_network(cfg: config_dict.ConfigDict) -> bool:
    return "sup_network" in cfg


def _uses_glass(network_cfg: config_dict.ConfigDict) -> bool:
    return network_cfg.use_glass


def _is_exact_sample_match(
    network_cfg: config_dict.ConfigDict,
    sample_type: SampleType,
) -> bool:
    matching = network_cfg.matching
    if sample_type in {SampleType.GLASS, SampleType.GLASS_EARLY_STOP}:
        return network_cfg.use_glass
    if sample_type == SampleType.FLOW:
        return matching == "flow"
    if sample_type == SampleType.FLOW_MAP:
        return matching == "flow_map"
    if sample_type in DIAMOND_SAMPLE_TYPES:
        return matching == "diamond_map"
    return False


def _is_fallback_sample_match(
    network_cfg: config_dict.ConfigDict,
    sample_type: SampleType,
) -> bool:
    return sample_type == SampleType.FLOW and network_cfg.matching == "flow_map"


class SampleBatch(NamedTuple):
    batch_x_init: jnp.ndarray
    batch_x_final: jnp.ndarray
    batch_full_traj: Optional[jnp.ndarray]
    batch_labels: Optional[jnp.ndarray]
    batch_x0s: Optional[jnp.ndarray] = None
    batch_x1s: Optional[jnp.ndarray] = None
    batch_measurement: Optional[Dict[str, jnp.ndarray]] = None
    batch_x_t_trace: Optional[jnp.ndarray] = None
    batch_posterior_trace: Optional[jnp.ndarray] = None


class GroundtruthInit(NamedTuple):
    batch_x_init: jnp.ndarray
    batch_x0s: Optional[jnp.ndarray]
    batch_x1s: Optional[jnp.ndarray]
    batch_measurement: Optional[Dict[str, jnp.ndarray]]
    batch_labels: Optional[jnp.ndarray]
    batch_prng_key: jnp.ndarray


def make_groundtruth_init(
    cfg: config_dict.ConfigDict,
    *,
    sharded: bool,
    batch_x_init: jnp.ndarray,
    batch_prng_key: jnp.ndarray,
    batch_x0s: Optional[jnp.ndarray] = None,
    batch_x1s: Optional[jnp.ndarray] = None,
    batch_measurement: Optional[Dict[str, jnp.ndarray]] = None,
    batch_labels: Optional[jnp.ndarray] = None,
) -> GroundtruthInit:
    init = GroundtruthInit(
        batch_x_init=batch_x_init,
        batch_x0s=batch_x0s,
        batch_x1s=batch_x1s,
        batch_measurement=batch_measurement,
        batch_labels=batch_labels,
        batch_prng_key=batch_prng_key,
    )
    if not sharded:
        return init
    return jax.tree_util.tree_map(lambda x: dist_utils.replicate_batch(cfg, x), init)


def expand_schedule(
    values: jnp.ndarray,
    count: int,
) -> jnp.ndarray:
    if len(values) == 1:
        return jnp.full((count,), values[0], dtype=values.dtype)
    if len(values) != count:
        raise ValueError(f"schedule must have length 1 or {count}, got {len(values)}.")
    return jnp.asarray(values, dtype=values.dtype)


def unique_schedule(schedule_np: np.ndarray) -> Tuple[np.ndarray, jnp.ndarray]:
    schedule_np = np.asarray(schedule_np)
    unique_combos, inverse = np.unique(schedule_np, axis=0, return_inverse=True)
    return unique_combos, jnp.asarray(inverse, dtype=jnp.int32)


def build_schedule_branches(
    schedule_np: np.ndarray,
    make_branch_fn: Callable[[np.ndarray], Callable],
) -> Tuple[Tuple[Callable, ...], jnp.ndarray]:
    unique_combos, inverse = unique_schedule(schedule_np)
    branch_fns = tuple(make_branch_fn(combo) for combo in unique_combos)
    return branch_fns, inverse


def _pad_batch(batch: Optional[jnp.ndarray], pad_size: int) -> Optional[jnp.ndarray]:
    if batch is None or pad_size == 0:
        return batch
    pad_width = [(0, pad_size)] + [(0, 0)] * (batch.ndim - 1)
    return jnp.pad(batch, pad_width)


def _pad_measurement(
    measurement: Optional[Dict[str, jnp.ndarray]], pad_size: int
) -> Optional[Dict[str, jnp.ndarray]]:
    if measurement is None or pad_size == 0:
        return measurement
    return {key: _pad_batch(value, pad_size) for key, value in measurement.items()}


def batch_split_keys(batch_prng_key: jnp.ndarray) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    # split each prng_key in the batch into 6 keys
    keys = jax.vmap(jax.random.split, in_axes=(0, None))(batch_prng_key, 6)
    return keys[:, 0], keys[:, 1], keys[:, 2], keys[:, 3], keys[:, 4], keys[:, 5]


def default_batch_guidance_fn(
    variables: Dict,
    batch_x_t: jnp.ndarray,
    t: float,
    batch_label: int,
    batch_key: jnp.ndarray,
    batch_measurement: Optional[jnp.ndarray],
    batch_drift: jnp.ndarray,
    step_idx: int,
    return_posterior: bool,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Default: no guidance, returns input x_t unmodified."""
    del (
        variables,
        t,
        batch_label,
        batch_key,
        batch_measurement,
        batch_drift,
        step_idx,
        return_posterior,
    )
    return jnp.zeros_like(batch_x_t), None


def default_batch_calc_xbar_s0(
    t: float, t_prime: float, batch_x_t: jnp.ndarray, batch_xbar_s0_key: jnp.ndarray
) -> jnp.ndarray:
    """Default: flow/flow map, don't need to add noise."""
    return batch_x_t


def default_batch_calc_x_t_prime(
    batch_x_s: jnp.ndarray,
    batch_x_t: jnp.ndarray,
    t: float,
    t_prime: float,
    s: float,
    batch_keys: jnp.ndarray,
) -> jnp.ndarray:
    """Default: no early stopping, don't need to recalculate x_t_prime."""
    return batch_x_s


def default_batch_resample_x_t_prime(
    variables: Dict,
    batch_x_t_prime: jnp.ndarray,
    step_idx: int,
    t_prime: float,
    batch_label: jnp.ndarray,
    batch_resample_key: jnp.ndarray,
    batch_value_key: jnp.ndarray,
    batch_measurement: Optional[jnp.ndarray],
    batch_potential: jnp.ndarray,
    batch_prev_log_weights: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Default: no SMC, identity function."""
    batch_size = batch_x_t_prime.shape[0]
    return batch_x_t_prime, batch_potential, batch_prev_log_weights, jnp.array(
        float(batch_size)
    )


class SamplerSpec(NamedTuple):
    step_fn: Callable
    s_min: float
    s_max: float
    ts: jnp.ndarray
    inner_steps: int

    # Needs to be passed at initialization as it depends on instance-specific s_max.
    calc_s_max: Callable[[float, float], float]

    batch_guidance_fn: Callable[
        [
            Dict,
            jnp.ndarray,
            float,
            int,
            jnp.ndarray,
            Optional[jnp.ndarray],
            jnp.ndarray,
            int,
            bool,
        ],
        Tuple[jnp.ndarray, Optional[jnp.ndarray]],
    ] = default_batch_guidance_fn

    batch_calc_xbar_s0: Callable[
        [float, float, jnp.ndarray, jnp.ndarray], jnp.ndarray
    ] = default_batch_calc_xbar_s0

    batch_calc_x_t_prime: Callable[
        [jnp.ndarray, jnp.ndarray, float, float, float, jnp.ndarray], jnp.ndarray
    ] = default_batch_calc_x_t_prime

    batch_resample_x_t_prime: Callable[
        [
            Dict,
            jnp.ndarray,
            int,
            float,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            Optional[jnp.ndarray],
            jnp.ndarray,
            jnp.ndarray,
        ],
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ] = default_batch_resample_x_t_prime

    batch_split_keys: Callable[
        [jnp.ndarray],
        Tuple[
            jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
        ],
    ] = batch_split_keys


class SamplerStepResult(NamedTuple):
    batch_prng_key: jnp.ndarray
    batch_xbar_s0_key: jnp.ndarray
    batch_resample_key: jnp.ndarray
    batch_value_key: jnp.ndarray
    batch_guidance_key: jnp.ndarray
    batch_renoise_key: jnp.ndarray
    batch_xbar_s0: jnp.ndarray
    curr_s_max: float
    batch_x_s: jnp.ndarray
    batch_x_t_prime: jnp.ndarray
    batch_resampled_x_t_prime: jnp.ndarray
    batch_guidance: jnp.ndarray
    batch_x_final: jnp.ndarray
    batch_full_traj: Optional[jnp.ndarray]
    batch_potential: jnp.ndarray
    batch_prev_log_weights: jnp.ndarray
    ess: jnp.ndarray


def inner_sample(
    step_fn: Callable,
    variables: Dict,
    t: float,
    t_prime: float,
    s_min: float,
    s_max: float,
    inner_steps: int,
    return_traj: bool,
    xbar_s0: jnp.ndarray,
    label: int,
    x_t: jnp.ndarray,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    s_grid = jnp.linspace(s_min, s_max, inner_steps + 1)
    num_steps = len(s_grid) - 1

    def step(x_s, idx):
        x_s_prime = step_fn(
            variables=variables,
            t=t,
            t_prime=t_prime,
            s=s_grid[idx],
            s_prime=s_grid[idx + 1],
            x_s=x_s,
            x_t=x_t,
            label=label,
            train=False,
        )
        return x_s_prime, (x_s_prime if return_traj else None)

    x_final, full_traj = jax.lax.scan(step, xbar_s0, jnp.arange(num_steps))
    return x_final, full_traj


def run_batched_inner_sample(
    spec: SamplerSpec,
    variables: Dict,
    t: float,
    t_prime: float,
    s_max: float,
    return_traj: bool,
    batch_xbar_s0: jnp.ndarray,
    batch_label: Optional[jnp.ndarray],
    batch_x_t: jnp.ndarray,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    def single_inner(xbar_s0, label, x_t):
        return inner_sample(
            step_fn=spec.step_fn,
            variables=variables,
            t=t,
            t_prime=t_prime,
            s_min=spec.s_min,
            s_max=s_max,
            inner_steps=spec.inner_steps,
            return_traj=return_traj,
            xbar_s0=xbar_s0,
            label=label,
            x_t=x_t,
        )
    return jax.vmap(single_inner)(batch_xbar_s0, batch_label, batch_x_t)


def run_sampler_step(
    spec: SamplerSpec,
    variables: Dict,
    step_idx: int,
    batch_x_t: jnp.ndarray,
    batch_prng_key: jnp.ndarray,
    batch_label: Optional[jnp.ndarray],
    batch_measurement: Optional[Dict[str, jnp.ndarray]],
    return_traj: bool,
    return_posterior: bool = False,
    batch_potential: Optional[jnp.ndarray] = None,
    batch_prev_log_weights: Optional[jnp.ndarray] = None,
) -> Tuple[SamplerStepResult, Optional[jnp.ndarray]]:
    t: float = spec.ts[step_idx]
    t_prime: float = spec.ts[step_idx + 1]
    num_steps = len(spec.ts) - 1
    batch_size = batch_x_t.shape[0]
    if batch_potential is None:
        batch_potential = jnp.zeros(batch_x_t.shape[0])
    if batch_prev_log_weights is None:
        batch_prev_log_weights = jnp.zeros(batch_x_t.shape[0])
    (
        batch_prng_key,
        batch_xbar_s0_key,
        batch_resample_key,
        batch_value_key,
        batch_guidance_key,
        batch_renoise_key,
    ) = spec.batch_split_keys(batch_prng_key)
    batch_xbar_s0: jnp.ndarray = spec.batch_calc_xbar_s0(t, 1.0, batch_x_t, batch_xbar_s0_key)
    curr_s_max: float = spec.calc_s_max(t, t_prime)
    batch_x_s, batch_full_traj = run_batched_inner_sample(
        spec=spec,
        variables=variables,
        t=t,
        t_prime=t_prime,
        s_max=curr_s_max,
        return_traj=return_traj,
        batch_xbar_s0=batch_xbar_s0,
        batch_label=batch_label,
        batch_x_t=batch_x_t,
    )
    batch_x_t_prime = spec.batch_calc_x_t_prime(
        batch_x_s, batch_x_t, t, t_prime, curr_s_max, batch_renoise_key
    )
    batch_drift = (batch_x_t_prime - batch_x_t) / (t_prime - t)
    batch_resampled_x_t_prime, batch_potential, batch_prev_log_weights, ess = jax.lax.cond(
        step_idx == num_steps - 1,
        lambda: (
            batch_x_t_prime,
            batch_potential,
            batch_prev_log_weights,
            jnp.array(float(batch_size)),
        ),
        lambda: spec.batch_resample_x_t_prime(
            variables,
            batch_x_t_prime,
            step_idx,
            t_prime,
            batch_label,
            batch_resample_key,
            batch_value_key,
            batch_measurement,
            batch_potential,
            batch_prev_log_weights,
        ),
    )
    batch_guidance, batch_posterior_samples = spec.batch_guidance_fn(
        variables,
        batch_x_t,
        t,
        batch_label,
        batch_guidance_key,
        batch_measurement,
        batch_drift,
        step_idx,
        return_posterior,
    )
    batch_x_final = batch_resampled_x_t_prime + (t_prime - t) * batch_guidance
    return (
        SamplerStepResult(
            batch_prng_key=batch_prng_key,
            batch_xbar_s0_key=batch_xbar_s0_key,
            batch_resample_key=batch_resample_key,
            batch_value_key=batch_value_key,
            batch_guidance_key=batch_guidance_key,
            batch_renoise_key=batch_renoise_key,
            batch_xbar_s0=batch_xbar_s0,
            curr_s_max=curr_s_max,
            batch_x_s=batch_x_s,
            batch_x_t_prime=batch_x_t_prime,
            batch_resampled_x_t_prime=batch_resampled_x_t_prime,
            batch_guidance=batch_guidance,
            batch_x_final=batch_x_final,
            batch_full_traj=batch_full_traj,
            batch_potential=batch_potential,
            batch_prev_log_weights=batch_prev_log_weights,
            ess=ess,
        ),
        batch_posterior_samples,
    )


def run_batched_sampler(
    spec: SamplerSpec,
    variables: Dict,
    return_traj: bool,
    batch_init_data: jnp.ndarray,
    batch_prng_key: jnp.ndarray,
    batch_label: Optional[jnp.ndarray],
    batch_measurement: Optional[Dict[str, jnp.ndarray]] = None,
    return_posterior: bool = False,
    return_ess: bool = False,
) -> Tuple[
    jnp.ndarray,
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
]:
    num_steps = len(spec.ts) - 1
    batch_size = batch_init_data.shape[0]
    batch_potential_init = jnp.zeros(batch_size)
    batch_prev_log_weights_init = jnp.zeros(batch_size)

    def batch_step(carry, idx):
        batch_x_t, batch_prng_key, batch_potential, batch_prev_log_weights = carry
        step_result, batch_posterior_samples = run_sampler_step(
            spec=spec,
            variables=variables,
            step_idx=idx,
            batch_x_t=batch_x_t,
            batch_prng_key=batch_prng_key,
            batch_label=batch_label,
            batch_measurement=batch_measurement,
            return_traj=return_traj,
            return_posterior=return_posterior,
            batch_potential=batch_potential,
            batch_prev_log_weights=batch_prev_log_weights,
        )
        step_outputs = (
            step_result.batch_full_traj if return_traj else None,
            batch_x_t if return_posterior else None,
            batch_posterior_samples if return_posterior else None,
            step_result.ess if return_ess else None,
        )
        return (
            step_result.batch_x_final,
            step_result.batch_prng_key,
            step_result.batch_potential,
            step_result.batch_prev_log_weights,
        ), step_outputs

    (batch_x_final, _, _, _), step_outputs = jax.lax.scan(
        batch_step,
        (
            batch_init_data,
            batch_prng_key,
            batch_potential_init,
            batch_prev_log_weights_init,
        ),
        jnp.arange(num_steps),
    )
    batch_full_traj, batch_x_t_trace, batch_posterior_samples, ess_per_step = (
        step_outputs
    )
    return (
        batch_x_final,
        batch_full_traj,
        batch_x_t_trace,
        batch_posterior_samples,
        ess_per_step,
    )


def create_batched_sampler(
    spec: SamplerSpec,
) -> Callable:

    @functools.partial(
        jax.jit, static_argnames=("return_traj", "return_posterior", "return_ess")
    )
    def sampler(
        variables: Dict,
        return_traj: bool,
        batch_init_data: jnp.ndarray,
        batch_prng_key: jnp.ndarray,
        batch_label: jnp.ndarray,
        batch_measurement: Optional[jnp.ndarray] = None,
        return_posterior: bool = False,
        return_ess: bool = False,
    ):
        return run_batched_sampler(
            spec=spec,
            variables=variables,
            return_traj=return_traj,
            return_posterior=return_posterior,
            return_ess=return_ess,
            batch_init_data=batch_init_data,
            batch_prng_key=batch_prng_key,
            batch_label=batch_label,
            batch_measurement=batch_measurement,
        )

    return sampler


def make_sample_x1_fn(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    inner_steps: int,
    posterior_sample_type: SampleType,
    network_slot: Optional[str] = None,
) -> Callable:
    if posterior_sample_type in {SampleType.FLOW, SampleType.FLOW_MAP}:
        base_spec = build_sampler_spec(
            cfg,
            statics,
            sample_type=posterior_sample_type,
            outer_steps=inner_steps,
            inner_steps=1,
            network_slot=network_slot,
        )

        def build_ts(t: float) -> jnp.ndarray:
            return jnp.linspace(t, 1.0, inner_steps + 1)

    elif posterior_sample_type in {SampleType.GLASS, SampleType.DIAMOND_EARLY_STOP}:
        base_spec = build_sampler_spec(
            cfg,
            statics,
            sample_type=posterior_sample_type,
            outer_steps=1,
            inner_steps=inner_steps,
            network_slot=network_slot,
        )

        def build_ts(t: float) -> jnp.ndarray:
            return jnp.stack([t, jnp.ones_like(t)])

    else:
        raise NotImplementedError(
            "Unsupported posterior sample type "
            f"{posterior_sample_type.name}. "
            "Expected one of FLOW, FLOW_MAP, GLASS, DIAMOND_EARLY_STOP."
        )

    def sample_x1(
        variables: Dict,
        x_t: jnp.ndarray,
        t: float,
        label: int,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        runtime_spec = base_spec._replace(ts=build_ts(t))
        batch_x_t = jnp.expand_dims(x_t, axis=0)
        batch_key = jnp.expand_dims(key, axis=0)
        batch_label = None
        if label is not None:
            batch_label = jnp.expand_dims(jnp.asarray(label), axis=0)
        batch_x1, _, _, _, _ = run_batched_sampler(
            spec=runtime_spec,
            variables=variables,
            return_traj=False,
            return_posterior=False,
            batch_init_data=batch_x_t,
            batch_prng_key=batch_key,
            batch_label=batch_label,
            batch_measurement=None,
        )
        return batch_x1[0]

    return sample_x1


def make_guidance_value_and_grad_fn(
    sample_x1_fn: Callable,
    reward_fn: Callable,
    mc_samples: int,
    return_posterior: bool = False,
) -> Callable:
    def value_and_grad_fn(
        sample_variables: Dict,
        reward_variables: Dict,
        x_t: jnp.ndarray,
        t: float,
        label: int,
        key: jnp.ndarray,
        measurement: Optional[jnp.ndarray] = None,
    ):
        keys = jax.random.split(key, mc_samples)

        if return_posterior:

            def reward_from_x_t(curr_x_t, curr_key, curr_measurement):
                x1 = sample_x1_fn(sample_variables, curr_x_t, t, label, curr_key)
                return reward_fn(reward_variables, x1, curr_measurement), x1

            value_and_grad = jax.value_and_grad(reward_from_x_t, has_aux=True)

            def scan_step(carry, curr_key):
                logsumexp_prev, grad_prev = carry
                (reward, x1), grad = value_and_grad(x_t, curr_key, measurement)
                logsumexp_next = jnp.logaddexp(logsumexp_prev, reward)
                weight_prev = jnp.exp(logsumexp_prev - logsumexp_next)
                weight_curr = jnp.exp(reward - logsumexp_next)
                grad_next = grad_prev * weight_prev + grad * weight_curr
                return (logsumexp_next, grad_next), x1

            init_carry = (
                jnp.array(-jnp.inf, dtype=x_t.dtype),
                jnp.zeros_like(x_t),
            )
            (logsumexp_val, grad), x1s = jax.lax.scan(scan_step, init_carry, keys)
            return logsumexp_val, grad, x1s

        def reward_from_x_t(curr_x_t, curr_key, curr_measurement):
            x1 = sample_x1_fn(sample_variables, curr_x_t, t, label, curr_key)
            return reward_fn(reward_variables, x1, curr_measurement)

        value_and_grad = jax.value_and_grad(reward_from_x_t)

        def scan_step(carry, curr_key):
            logsumexp_prev, grad_prev = carry
            reward, grad = value_and_grad(x_t, curr_key, measurement)
            logsumexp_next = jnp.logaddexp(logsumexp_prev, reward)
            weight_prev = jnp.exp(logsumexp_prev - logsumexp_next)
            weight_curr = jnp.exp(reward - logsumexp_next)
            grad_next = grad_prev * weight_prev + grad * weight_curr
            return (logsumexp_next, grad_next), None

        init_carry = (
            jnp.array(-jnp.inf, dtype=x_t.dtype),
            jnp.zeros_like(x_t),
        )
        (logsumexp_val, grad), _ = jax.lax.scan(scan_step, init_carry, keys)
        return logsumexp_val, grad

    return value_and_grad_fn


def make_smc_value_fn(
    sample_x1_fn: Callable,
    reward_fn: Callable,
    mc_samples: int,
) -> Callable:
    batch_sample_x1 = jax.vmap(sample_x1_fn, in_axes=(None, None, None, None, 0))
    batch_reward_fn = jax.vmap(reward_fn, in_axes=(None, 0, None))

    def value_fn(
        sample_variables: Dict,
        reward_variables: Dict,
        x_t: jnp.ndarray,
        t_prime: float,
        label: int,
        key: jnp.ndarray,
        measurement: Optional[jnp.ndarray] = None,
    ):
        keys = jax.random.split(key, mc_samples)
        x1s = batch_sample_x1(sample_variables, x_t, t_prime, label, keys)
        rewards = batch_reward_fn(reward_variables, x1s, measurement)
        return logsumexp(rewards) - jnp.log(mc_samples)

    return value_fn


def make_batch_guidance_fn(
    cfg,
    statics,
    reward_fn: Callable,
    guidance_scales: jnp.ndarray,
    guidance_scale_schedule: str,
    mc_samples: jnp.ndarray,
    mc_inner_steps: jnp.ndarray,
    posterior_sample_type: SampleType,
    posterior_network_slot: Optional[str] = None,
) -> Callable:

    schedule_np = np.stack(
        [
            np.asarray(mc_samples),
            np.asarray(mc_inner_steps),
        ],
        axis=1,
    )

    def make_branch_fn(combo: np.ndarray, return_posterior: bool) -> Callable:
        combo_mc_samples = int(combo[0])
        combo_mc_inner_steps = int(combo[1])
        sample_x1_fn = make_sample_x1_fn(
            cfg,
            statics,
            inner_steps=combo_mc_inner_steps,
            posterior_sample_type=posterior_sample_type,
            network_slot=posterior_network_slot,
        )
        value_and_grad_fn = make_guidance_value_and_grad_fn(
            sample_x1_fn,
            reward_fn,
            combo_mc_samples,
        )
        value_and_grad_with_posterior_fn = make_guidance_value_and_grad_fn(
            sample_x1_fn,
            reward_fn,
            combo_mc_samples,
            return_posterior=True,
        )

        def branch_fn(operand):
            (
                variables,
                batch_x_t,
                t,
                batch_label,
                batch_key,
                batch_measurement,
                batch_drift,
                guidance_scale,
            ) = operand

            sample_variables = variables["posterior"]

            def scale_grad(grad, drift):
                curr_guidance_scale = guidance_scale
                if guidance_scale_schedule == "auto":
                    grad_norm = optax.global_norm(grad)
                    drift_norm = optax.global_norm(drift)
                    curr_guidance_scale = (
                        curr_guidance_scale
                        * drift_norm
                        / jnp.maximum(grad_norm, 1e-12)
                    )
                return curr_guidance_scale * grad

            guidance_is_zero = jnp.equal(guidance_scale, 0.0)

            if return_posterior:

                def per_sample_fn(x_t, label, key, measurement, drift):
                    _, grad, posterior_samples = value_and_grad_with_posterior_fn(
                        sample_variables, variables, x_t, t, label, key, measurement
                    )
                    return scale_grad(grad, drift), posterior_samples

                batch_guidance, batch_posterior_samples = jax.vmap(
                    per_sample_fn, in_axes=(0, 0, 0, 0, 0)
                )(batch_x_t, batch_label, batch_key, batch_measurement, batch_drift)
                return batch_guidance, batch_posterior_samples

            def calc_guidance():
                def per_sample_fn(x_t, label, key, measurement, drift):
                    _, grad = value_and_grad_fn(
                        sample_variables, variables, x_t, t, label, key, measurement
                    )
                    return scale_grad(grad, drift)

                batched_fn = jax.vmap(per_sample_fn, in_axes=(0, 0, 0, 0, 0))
                return batched_fn(
                    batch_x_t, batch_label, batch_key, batch_measurement, batch_drift
                )

            batch_guidance = jax.lax.cond(
                guidance_is_zero,
                lambda: jnp.zeros_like(batch_x_t),
                calc_guidance,
            )
            return batch_guidance, None

        return branch_fn

    unique_combos, t_step_to_branch = unique_schedule(schedule_np)
    branch_fns = tuple(make_branch_fn(combo, False) for combo in unique_combos)
    posterior_branch_fns = tuple(make_branch_fn(combo, True) for combo in unique_combos)

    def batch_guidance_fn(
        variables,
        x_t,
        t,
        label,
        key,
        measurement,
        drift,
        step_idx,
        return_posterior,
    ):
        branch_idx = t_step_to_branch[step_idx]
        operand = (
            variables,
            x_t,
            t,
            label,
            key,
            measurement,
            drift,
            guidance_scales[step_idx],
        )
        if return_posterior:
            return jax.lax.switch(branch_idx, posterior_branch_fns, operand)
        return jax.lax.switch(branch_idx, branch_fns, operand)

    return batch_guidance_fn


def compute_ess(log_weights: jnp.ndarray) -> jnp.ndarray:
    log_w_norm = log_weights - jax.nn.logsumexp(log_weights)
    return jnp.exp(-jax.nn.logsumexp(2.0 * log_w_norm))


def systematic_resample_indices(
    log_weights: jnp.ndarray,
    key: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    probs = jax.nn.softmax(log_weights / temperature)
    cdf = jnp.cumsum(probs)
    batch_size = log_weights.shape[0]
    u = jax.random.uniform(key, (), minval=0.0, maxval=1.0 / batch_size)
    points = u + jnp.arange(batch_size) / batch_size
    indices = jnp.searchsorted(cdf, points)
    indices = jnp.clip(indices, 0, batch_size - 1)
    return indices


def make_batch_resample_fn(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    batch_reward_fn: Callable,
    mc_samples: jnp.ndarray,
    mc_inner_steps: jnp.ndarray,
    temperatures: jnp.ndarray,
    resample_steps: int,
    posterior_sample_type: SampleType,
    posterior_network_slot: Optional[str] = None,
    ess_threshold: float = 1.0,
    use_potential: bool = False,
) -> Callable:
    expanded_mc_samples = expand_schedule(mc_samples, resample_steps)
    expanded_mc_inner_steps = expand_schedule(mc_inner_steps, resample_steps)
    expanded_temperatures = expand_schedule(temperatures, resample_steps)
    schedule_np = np.stack(
        [
            np.asarray(expanded_mc_samples),
            np.asarray(expanded_mc_inner_steps),
            np.asarray(expanded_temperatures),
        ],
        axis=1,
    )

    def make_branch_fn(combo: np.ndarray) -> Callable:
        combo_mc_samples = int(combo[0])
        combo_mc_inner_steps = int(combo[1])
        combo_temperature = float(combo[2])
        sample_x1_fn = make_sample_x1_fn(
            cfg,
            statics,
            inner_steps=combo_mc_inner_steps,
            posterior_sample_type=posterior_sample_type,
            network_slot=posterior_network_slot,
        )
        value_fn = make_smc_value_fn(
            sample_x1_fn,
            batch_reward_fn,
            combo_mc_samples,
        )

        def branch_fn(operand):
            (
                variables,
                batch_x_t_prime,
                t_prime,
                batch_label,
                batch_resample_key,
                batch_value_key,
                batch_measurement,
                batch_potential,
                batch_prev_log_weights,
            ) = operand
            resample_key = batch_resample_key[0]
            batch_size = batch_x_t_prime.shape[0]
            sample_variables = variables.get("posterior", variables)

            batch_value_fn = jax.vmap(
                lambda x_t, label, key, measurement: value_fn(
                    sample_variables, variables, x_t, t_prime, label, key, measurement
                )
            )
            log_weights = batch_value_fn(
                batch_x_t_prime, batch_label, batch_value_key, batch_measurement
            )

            batch_potential = batch_potential + log_weights - batch_prev_log_weights
            resample_weights = batch_potential if use_potential else log_weights
            ess = compute_ess(resample_weights / combo_temperature)
            should_resample = ess < ess_threshold * batch_size

            def do_resample(_):
                indices = systematic_resample_indices(
                    resample_weights, resample_key, temperature=combo_temperature
                )
                return (
                    batch_x_t_prime[indices],
                    jnp.zeros_like(batch_potential),
                    log_weights[indices],
                    ess,
                )

            def skip_resample(_):
                return batch_x_t_prime, batch_potential, log_weights, ess

            return jax.lax.cond(should_resample, do_resample, skip_resample, None)

        return branch_fn

    branch_fns, step_to_branch = build_schedule_branches(schedule_np, make_branch_fn)

    def batch_resample_x_t_prime(
        variables: Dict,
        batch_x_t_prime: jnp.ndarray,
        step_idx: int,
        t_prime: float,
        batch_label: Optional[jnp.ndarray],
        batch_resample_key: jnp.ndarray,
        batch_value_key: jnp.ndarray,
        batch_measurement: Optional[jnp.ndarray],
        batch_potential: jnp.ndarray,
        batch_prev_log_weights: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        branch_idx = step_to_branch[step_idx]
        operand = (
            variables,
            batch_x_t_prime,
            t_prime,
            batch_label,
            batch_resample_key,
            batch_value_key,
            batch_measurement,
            batch_potential,
            batch_prev_log_weights,
        )
        return jax.lax.switch(branch_idx, branch_fns, operand)

    return batch_resample_x_t_prime


def build_sampler_spec(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    sample_type: SampleType,
    outer_steps: int,
    inner_steps: int,
    ts_override: Optional[jnp.ndarray] = None,
    network_slot: Optional[str] = None,
) -> SamplerSpec:
    total_steps = outer_steps * inner_steps
    use_main = use_main_network(cfg, sample_type, network_slot=network_slot)
    net = statics.net if use_main else statics.sup_net
    net_cfg = cfg.network if use_main else cfg.sup_network
    uses_glass = _uses_glass(net_cfg)

    if sample_type in {SampleType.GLASS, SampleType.GLASS_EARLY_STOP}:
        assert uses_glass, "GLASS sampling requires the selected network to be a GLASS network."
    elif sample_type in DIAMOND_SAMPLE_TYPES:
        assert (
            net_cfg.matching == "diamond_map"
        ), "Diamond sampling requires the selected network to be a diamond map."

    calc_xbar_s0, calc_s, calc_x_t_prime, calc_x_t_prime_renoise = None, None, None, None
    if sample_type in {SampleType.GLASS, SampleType.GLASS_EARLY_STOP} | DIAMOND_SAMPLE_TYPES:

        def calc_xbar_s0(t: float, t_prime: float, x_t: jnp.ndarray, prng_key: jnp.ndarray):
            return net.calc_xbar_s(prng_key, t, t_prime, 0.0, x_t)

        calc_s = net.calc_s
        calc_x_t_prime = net.calc_x_t_prime
        calc_x_t_prime_renoise = net.calc_x_t_prime_renoise

    if sample_type == SampleType.FLOW:
        def step_fn(
            variables: Dict,
            t: float,
            t_prime: float,
            s: float,
            s_prime: float,
            x_s: jnp.ndarray,
            x_t: jnp.ndarray,
            label: int,
            train: bool = False,
        ):
            dt = t_prime - t
            if uses_glass:
                v = net.apply(
                    variables,
                    t,
                    x_t,
                    label=label,
                    train=train,
                    method="_calc_base_b",
                )
            else:
                v = net.apply(
                    variables,
                    t,
                    x_t,
                    label=label,
                    train=train,
                    method="calc_b",
                )
            return x_t + dt * v

        ts = (
            jnp.linspace(cfg.training.t_min, cfg.training.t_max, total_steps + 1)
            if ts_override is None
            else ts_override
        )
        spec = SamplerSpec(
            step_fn=step_fn,
            ts=ts,
            s_min=0.0,
            s_max=1.0,
            inner_steps=1,
            calc_s_max=lambda t, t_prime: 1.0,
        )
    elif sample_type == SampleType.FLOW_MAP:
        def step_fn(
            variables: Dict,
            t: float,
            t_prime: float,
            s: float,
            s_prime: float,
            x_s: jnp.ndarray,
            x_t: jnp.ndarray,
            label: int,
            train: bool = False,
        ):
            if uses_glass:
                return net.apply(
                    variables,
                    t,
                    t_prime,
                    x_t,
                    label=label,
                    train=train,
                    return_X_and_phi=False,
                    method="_apply_base_map",
                )
            return net.apply(
                variables,
                t,
                t_prime,
                x_t,
                label=label,
                train=train,
                return_X_and_phi=False,
            )

        ts = (
            jnp.linspace(cfg.training.t_min, cfg.training.t_max, total_steps + 1)
            if ts_override is None
            else ts_override
        )
        spec = SamplerSpec(
            step_fn=step_fn,
            ts=ts,
            s_min=0.0,
            s_max=1.0,
            inner_steps=1,
            calc_s_max=lambda t, t_prime: 1.0,
        )
    elif sample_type == SampleType.GLASS:
        assert calc_xbar_s0 is not None
        batch_calc_xbar_s0 = jax.vmap(calc_xbar_s0, in_axes=(None, None, 0, 0))

        def step_fn(
            variables: Dict,
            t: float,
            t_prime: float,
            s: float,
            s_prime: float,
            x_s: jnp.ndarray,
            x_t: jnp.ndarray,
            label: int,
            train: bool = False,
        ):
            ds = s_prime - s
            v = net.apply(
                variables,
                t,
                t_prime,
                s,
                x_t,
                x_s,
                label=label,
                train=train,
            )
            return x_s + ds * v

        ts = (
            jnp.linspace(cfg.training.t_min, cfg.training.t_max, outer_steps + 1)
            if ts_override is None
            else ts_override
        )
        spec = SamplerSpec(
            step_fn=step_fn,
            ts=ts,
            s_min=cfg.training.s_min,
            s_max=cfg.training.s_max,
            inner_steps=inner_steps,
            batch_calc_xbar_s0=batch_calc_xbar_s0,
            calc_s_max=lambda t, t_prime: cfg.training.s_max,
        )
    elif sample_type == SampleType.GLASS_EARLY_STOP:
        assert calc_xbar_s0 is not None and calc_s is not None and calc_x_t_prime is not None
        batch_calc_xbar_s0 = jax.vmap(calc_xbar_s0, in_axes=(None, None, 0, 0))

        def calc_x_t_prime_early_stop(x_s, x_t, t, t_prime, s, key):
            del key
            return calc_x_t_prime(t, 1.0, s, x_t, x_s, target_t_prime=t_prime)

        batch_calc_x_t_prime = jax.vmap(calc_x_t_prime_early_stop, in_axes=(0, 0, None, None, None, 0))

        def step_fn(
            variables: Dict,
            t: float,
            t_prime: float,
            s: float,
            s_prime: float,
            x_s: jnp.ndarray,
            x_t: jnp.ndarray,
            label: int,
            train: bool = False,
        ):
            ds = s_prime - s
            v = net.apply(
                variables,
                t,
                cfg.training.t_max,
                s,
                x_t,
                x_s,
                label=label,
                train=train,
            )
            return x_s + ds * v

        ts = (
            jnp.linspace(cfg.training.t_min, cfg.training.t_max, outer_steps + 1)
            if ts_override is None
            else ts_override
        )
        spec = SamplerSpec(
            step_fn=step_fn,
            ts=ts,
            s_min=cfg.training.s_min,
            s_max=cfg.training.s_max,
            inner_steps=inner_steps,
            batch_calc_xbar_s0=batch_calc_xbar_s0,
            batch_calc_x_t_prime=batch_calc_x_t_prime,
            calc_s_max=lambda t, t_prime: jnp.minimum(cfg.training.s_max, calc_s(t, t_prime)),
        )
    elif sample_type == SampleType.DIAMOND_DIAG:
        assert calc_xbar_s0 is not None
        batch_calc_xbar_s0 = jax.vmap(calc_xbar_s0, in_axes=(None, None, 0, 0))

        def step_fn(
            variables: Dict,
            t: float,
            t_prime: float,
            s: float,
            s_prime: float,
            x_s: jnp.ndarray,
            x_t: jnp.ndarray,
            label: int,
            train: bool = False,
        ):
            ds = s_prime - s
            v = net.apply(
                variables,
                t,
                t_prime,
                s,
                x_t,
                x_s,
                label=label,
                train=train,
                method="calc_b",
            )
            return x_s + ds * v

        ts = (
            jnp.linspace(cfg.training.t_min, cfg.training.t_max, outer_steps + 1)
            if ts_override is None
            else ts_override
        )
        spec = SamplerSpec(
            step_fn=step_fn,
            ts=ts,
            s_min=cfg.training.s_min,
            s_max=cfg.training.s_max,
            inner_steps=inner_steps,
            batch_calc_xbar_s0=batch_calc_xbar_s0,
            calc_s_max=lambda t, t_prime: 1.0,
        )
    elif sample_type == SampleType.DIAMOND_EARLY_STOP:
        assert calc_xbar_s0 is not None and calc_s is not None and calc_x_t_prime is not None
        batch_calc_xbar_s0 = jax.vmap(calc_xbar_s0, in_axes=(None, None, 0, 0))

        def calc_x_t_prime_early_stop(x_s, x_t, t, t_prime, s, key):
            del key
            return calc_x_t_prime(t, 1.0, s, x_t, x_s, target_t_prime=t_prime)

        batch_calc_x_t_prime = jax.vmap(calc_x_t_prime_early_stop, in_axes=(0, 0, None, None, None, 0))

        def step_fn(
            variables: Dict,
            t: float,
            t_prime: float,
            s: float,
            s_prime: float,
            x_s: jnp.ndarray,
            x_t: jnp.ndarray,
            label: int,
            train: bool = False,
        ):
            return net.apply(
                variables,
                t,
                cfg.training.t_max,
                s,
                s_prime,
                x_t,
                x_s,
                label=label,
                train=train,
            )

        ts = (
            jnp.linspace(cfg.training.t_min, cfg.training.t_max, outer_steps + 1)
            if ts_override is None
            else ts_override
        )
        spec = SamplerSpec(
            step_fn=step_fn,
            ts=ts,
            s_min=cfg.training.s_min,
            s_max=cfg.training.s_max,
            inner_steps=inner_steps,
            batch_calc_xbar_s0=batch_calc_xbar_s0,
            batch_calc_x_t_prime=batch_calc_x_t_prime,
            calc_s_max=lambda t, t_prime: jnp.minimum(cfg.training.s_max, calc_s(t, t_prime)),
        )
    elif sample_type == SampleType.FULL_DIAMOND:
        assert calc_xbar_s0 is not None
        batch_calc_xbar_s0 = jax.vmap(calc_xbar_s0, in_axes=(None, None, 0, 0))
        ts = (
            jnp.linspace(cfg.training.t_min, cfg.training.t_max, outer_steps + 1)
            if ts_override is None
            else ts_override
        )
        spec = SamplerSpec(
            step_fn=net.apply,
            ts=ts,
            s_min=cfg.training.s_min,
            s_max=cfg.training.s_max,
            inner_steps=inner_steps,
            batch_calc_xbar_s0=batch_calc_xbar_s0,
            calc_s_max=lambda t, t_prime: cfg.training.s_max,
        )
    elif sample_type == SampleType.DIAMOND_RENOISE:
        assert calc_xbar_s0 is not None and calc_x_t_prime_renoise is not None
        batch_calc_xbar_s0 = jax.vmap(calc_xbar_s0, in_axes=(None, None, 0, 0))
        batch_calc_x_t_prime = jax.vmap(calc_x_t_prime_renoise, in_axes=(0, 0, None, None, None, 0))

        # always sample x1
        def step_fn(
            variables: Dict,
            t: float,
            t_prime: float,
            s: float,
            s_prime: float,
            x_s: jnp.ndarray,
            x_t: jnp.ndarray,
            label: int,
            train: bool = False,
        ):
            return net.apply(
                variables,
                t,
                1.0,
                s,
                s_prime,
                x_t,
                x_s,
                label=label,
                train=train,
            )

        ts = (
            jnp.linspace(cfg.training.t_min, cfg.training.t_max, outer_steps + 1)
            if ts_override is None
            else ts_override
        )

        spec = SamplerSpec(
            step_fn=step_fn,
            ts=ts,
            s_min=cfg.training.s_min,
            s_max=cfg.training.s_max,
            inner_steps=inner_steps,
            batch_calc_xbar_s0=batch_calc_xbar_s0,
            batch_calc_x_t_prime=batch_calc_x_t_prime,
            calc_s_max=lambda t, t_prime: cfg.training.s_max,
        )
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    return spec


def build_guided_sampler_spec(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    base_sample_type: SampleType,
    outer_steps: int,
    inner_steps: int,
    batch_guidance_fn: Callable,
    ts_override: Optional[jnp.ndarray] = None,
    base_network_slot: Optional[str] = None,
) -> SamplerSpec:
    base_spec = build_sampler_spec(
        cfg=cfg,
        statics=statics,
        sample_type=base_sample_type,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        ts_override=ts_override,
        network_slot=base_network_slot,
    )

    def guided_step_fn(
        variables: Dict,
        t: float,
        t_prime: float,
        s: float,
        s_prime: float,
        x_s: jnp.ndarray,
        x_t: jnp.ndarray,
        label: int,
        train: bool = False,
    ):
        return base_spec.step_fn(
            variables=variables["base"],
            t=t,
            t_prime=t_prime,
            s=s,
            s_prime=s_prime,
            x_s=x_s,
            x_t=x_t,
            label=label,
            train=train,
        )

    return base_spec._replace(
        step_fn=guided_step_fn,
        batch_guidance_fn=batch_guidance_fn,
    )


def build_smc_sampler_spec(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    base_sample_type: SampleType,
    posterior_sample_type: SampleType,
    outer_steps: int,
    inner_steps: int,
    batch_reward_fn: Callable,
    mc_samples: jnp.ndarray,
    mc_inner_steps: jnp.ndarray,
    temperatures: jnp.ndarray,
    ts_override: Optional[jnp.ndarray] = None,
    ess_threshold: float = 1.0,
    base_network_slot: Optional[str] = None,
    posterior_network_slot: Optional[str] = None,
    use_potential: bool = False,
) -> SamplerSpec:
    spec = build_guided_sampler_spec(
        cfg=cfg,
        statics=statics,
        base_sample_type=base_sample_type,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        batch_guidance_fn=default_batch_guidance_fn,
        ts_override=ts_override,
        base_network_slot=base_network_slot,
    )

    num_transitions = len(spec.ts) - 1
    resample_steps = num_transitions
    assert resample_steps > 0
    batch_resample_x_t_prime = make_batch_resample_fn(
        cfg=cfg,
        statics=statics,
        batch_reward_fn=batch_reward_fn,
        mc_samples=mc_samples,
        mc_inner_steps=mc_inner_steps,
        temperatures=temperatures,
        resample_steps=resample_steps,
        posterior_sample_type=posterior_sample_type,
        posterior_network_slot=posterior_network_slot,
        ess_threshold=ess_threshold,
        use_potential=use_potential,
    )

    return spec._replace(batch_resample_x_t_prime=batch_resample_x_t_prime)


def create_sampler(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    sample_type: SampleType,
    outer_steps: int,
    inner_steps: int,
    ts_override: Optional[jnp.ndarray] = None,
    network_slot: Optional[str] = None,
) -> Callable:
    spec = build_sampler_spec(
        cfg,
        statics,
        sample_type,
        outer_steps,
        inner_steps,
        ts_override=ts_override,
        network_slot=network_slot,
    )
    return create_batched_sampler(spec)


def create_sample_fn(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    sample_type: SampleType,
    outer_steps: int,
    inner_steps: int,
    batch_reward_fn: Optional[Callable] = None,
    mc_samples: Optional[jnp.ndarray] = None,
    mc_inner_steps: Optional[jnp.ndarray] = None,
    temperatures: Optional[jnp.ndarray] = None,
    ts_override: Optional[jnp.ndarray] = None,
    posterior_sample_type: Optional[SampleType] = None,
    network_slot: Optional[str] = None,
    posterior_network_slot: Optional[str] = None,
    ess_threshold: float = 1.0,
    use_potential: bool = False,
) -> Callable:
    batch_resample_x_t_prime = None

    optional_args = [
        batch_reward_fn,
        mc_samples,
        mc_inner_steps,
        temperatures,
        ts_override,
        posterior_sample_type,
    ]
    all_none = all(arg is None for arg in optional_args)
    all_set = all(arg is not None for arg in optional_args)
    assert (
        all_none or all_set
    ), "Optional arguments must either ALL be None or ALL be set."

    spec = build_sampler_spec(
        cfg,
        statics,
        sample_type,
        outer_steps,
        inner_steps,
        ts_override=ts_override,
        network_slot=network_slot,
    )

    if all_set:
        num_transitions = len(spec.ts) - 1
        resample_steps = num_transitions
        assert resample_steps > 0
        batch_resample_x_t_prime = make_batch_resample_fn(
            cfg=cfg,
            statics=statics,
            batch_reward_fn=batch_reward_fn,
            mc_samples=mc_samples,
            mc_inner_steps=mc_inner_steps,
            temperatures=temperatures,
            resample_steps=resample_steps,
            posterior_sample_type=posterior_sample_type,
            posterior_network_slot=posterior_network_slot,
            ess_threshold=ess_threshold,
            use_potential=use_potential,
        )

    if batch_resample_x_t_prime is not None:
        spec = spec._replace(batch_resample_x_t_prime=batch_resample_x_t_prime)
    sampler = create_batched_sampler(spec)

    @functools.partial(jax.jit, static_argnums=(2, 3, 4))
    def sample_fn(
        prng_key: jnp.ndarray,
        variables: Dict,
        bs: int,
        return_traj: bool,
        sharded: bool,
    ):
        rng_keys = jax.random.split(prng_key, num=bs + 1)
        prng_key, subkeys = rng_keys[0], rng_keys[1:]
        prng_key, rho0_key = jax.random.split(prng_key)

        batch_x0s = statics.sample_rho0(bs, rho0_key)

        # Handle labels
        prng_key, label_key = jax.random.split(prng_key)
        if cfg.training.conditional:
            if cfg.training.class_dropout > 0:
                batch_labels = jax.random.choice(
                    label_key, cfg.problem.num_classes + 1, (bs,)
                )
            else:
                batch_labels = jax.random.choice(
                    label_key, cfg.problem.num_classes, (bs,)
                )
        else:
            batch_labels = None

        # if params are sharded, we need to replicate data
        if sharded:
            batch_x0s = dist_utils.replicate_batch(cfg, batch_x0s)
            batch_labels = dist_utils.replicate_batch(cfg, batch_labels)

        batch_x_final, batch_full_traj, _, _, _ = sampler(
            variables=variables,
            return_traj=return_traj,
            return_posterior=False,
            batch_init_data=batch_x0s,
            batch_prng_key=subkeys,
            batch_label=batch_labels,
        )
        # make sure if data is replicated, output is also replicated
        if sharded:
            batch_x0s = dist_utils.constrain_batch_sharding(cfg, batch_x0s)
            batch_x_final = dist_utils.constrain_batch_sharding(cfg, batch_x_final)
            batch_labels = dist_utils.constrain_batch_sharding(cfg, batch_labels)
        return SampleBatch(
            batch_x_init=batch_x0s,
            batch_x_final=batch_x_final,
            batch_full_traj=batch_full_traj,
            batch_labels=batch_labels,
            batch_x0s=batch_x0s,
        )

    return sample_fn


def create_groundtruth_sample_fn(
    sampler: Callable,
    init_fn: Callable[[jnp.ndarray, int, bool], GroundtruthInit],
) -> Callable:
    def sample_fn(
        prng_key: jnp.ndarray,
        variables: Dict,
        bs: int,
        return_traj: bool,
        sharded: bool,
        return_posterior: bool = False,
    ) -> SampleBatch:
        init = init_fn(prng_key, bs, sharded)
        (
            batch_x_final,
            batch_full_traj,
            batch_x_t_trace,
            batch_posterior_trace,
            _,
        ) = sampler(
            variables=variables,
            return_traj=return_traj,
            return_posterior=return_posterior,
            batch_init_data=init.batch_x_init,
            batch_prng_key=init.batch_prng_key,
            batch_label=init.batch_labels,
            batch_measurement=init.batch_measurement,
        )
        return SampleBatch(
            batch_x_init=init.batch_x_init,
            batch_x_final=batch_x_final,
            batch_full_traj=batch_full_traj,
            batch_labels=init.batch_labels,
            batch_x0s=init.batch_x0s,
            batch_x1s=init.batch_x1s,
            batch_measurement=init.batch_measurement,
            batch_x_t_trace=batch_x_t_trace,
            batch_posterior_trace=batch_posterior_trace,
        )

    return sample_fn


def calc_fid(
    cfg: config_dict.ConfigDict,
    inception_fn: Callable,
    decode_fn: Optional[Callable],
    sample_fn: Callable,
    prng_key: jnp.ndarray,
    variables: Dict,
    n_samples: int,
    bs: int,
) -> float:
    # Load reference statistics
    stats = np.load(cfg.logging.fid_stats_path)
    mu_real, sigma_real = stats["mu"], stats["sigma"]

    # Running mean / covariance online (Welford's algorithm)
    n_seen, mu, M2 = 0, None, None

    # Calculate number of batches
    n_batches = (n_samples + bs - 1) // bs

    for _ in tqdm(range(n_batches), desc="Computing FID"):
        prng_key, step_key = jax.random.split(prng_key)

        # Sample images
        sample_batch = sample_fn(
            step_key, variables, bs, return_traj=False, sharded=True
        )
        imgs = sample_batch.batch_x_final
        imgs = latent_utils.maybe_decode_latents_chunked(
            cfg, imgs, chunk_size=latent_utils.LATENT_DECODE_CHUNK, decode_fn=decode_fn
        )

        # Get Inception features.
        imgs = jnp.clip(imgs, -1, 1)
        imgs = imgs.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        acts = fid_utils.resize_and_incept(imgs, inception_fn)

        acts = np.asarray(acts)
        acts = acts.reshape(acts.shape[0], -1)
        batch_n = acts.shape[0]
        batch_mu = acts.mean(0)
        centered = acts - batch_mu
        batch_M2 = centered.T @ centered

        if mu is None:
            mu = batch_mu
            M2 = batch_M2
            n_seen = batch_n
        else:
            n_new = n_seen + batch_n
            delta = batch_mu - mu
            mu += delta * batch_n / n_new
            M2 += batch_M2 + np.outer(delta, delta) * n_seen * batch_n / n_new
            n_seen = n_new

    sigma_gen = M2 / (n_seen - 1)
    fid = fid_utils.fid_from_stats(mu, sigma_gen, mu_real, sigma_real)

    del imgs, acts
    gc.collect()
    return float(fid)


def use_main_network(
    cfg: config_dict.ConfigDict,
    sample_type: SampleType,
    network_slot: Optional[str] = None,
):
    "Should we use weights of the main or supervisor network?"

    if network_slot is not None:
        network_slot = str(network_slot).lower()

    if network_slot not in {None, "auto", "main", "sup"}:
        raise ValueError(
            f"Unknown network slot {network_slot!r}. Expected one of auto, main, sup."
        )

    if network_slot == "main":
        if _is_exact_sample_match(cfg.network, sample_type):
            return True
        if _is_fallback_sample_match(cfg.network, sample_type):
            return True
        raise ValueError(
            f"Main network cannot serve sample type {sample_type.name}. "
            f"Main matching={cfg.network.matching}, "
            f"main use_glass={_uses_glass(cfg.network)}."
        )

    if network_slot == "sup":
        if not _has_sup_network(cfg):
            raise ValueError(
                f"Supervisor network slot requested for {sample_type.name}, but cfg has no sup_network."
            )
        if _is_exact_sample_match(cfg.sup_network, sample_type):
            return False
        if _is_fallback_sample_match(cfg.sup_network, sample_type):
            return False
        raise ValueError(
            f"Supervisor network cannot serve sample type {sample_type.name}. "
            f"Sup matching={cfg.sup_network.matching}, "
            f"sup use_glass={_uses_glass(cfg.sup_network)}."
        )

    if _is_exact_sample_match(cfg.network, sample_type):
        return True
    if _has_sup_network(cfg) and _is_exact_sample_match(cfg.sup_network, sample_type):
        return False
    if _is_fallback_sample_match(cfg.network, sample_type):
        return True
    if _has_sup_network(cfg) and _is_fallback_sample_match(cfg.sup_network, sample_type):
        return False

    raise ValueError(
        f"Could not find a network matching sample type {sample_type.name}. "
        f"Main matching={cfg.network.matching}, "
        f"main use_glass={_uses_glass(cfg.network)}, "
        f"has_sup={_has_sup_network(cfg)}."
    )


def get_params(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    sample_type: SampleType,
    ema_factor_override: Optional[float] = None,
    network_slot: Optional[str] = None,
) -> Dict:
    if use_main_network(cfg, sample_type, network_slot=network_slot):
        print(f"using main network for {sample_type}")
        return train_state.ema_params[
            ema_factor_override
            if ema_factor_override is not None
            else cfg.logging.ema_factor
        ]
    print(f"using sup network for {sample_type}")
    return statics.sup_params


def make_traj_plot(nrows, ncols, xhats, titles, plot_idxs):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.0 * ncols + 1, 2.5 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Plot trajectory
    for i in range(nrows):
        for j, idx in enumerate(plot_idxs):
            ax = axes[i, j]
            if j == 0:
                ax.set_ylabel(titles[i])

            ax.imshow(datasets.unnormalize_image(xhats[i, idx]))
    return fig


def make_sample_plot(nrows, ncols, xfinals):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.0 * ncols, 2.0 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Plot final samples
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            ax.imshow(datasets.unnormalize_image(xfinals[i, j]))

    return fig


def sample_from_ckpt(
    cfg,
    statics,
    train_state,
    prng_key,
    outer_steps,
    inner_steps,
    sample_type,
    comp_fid,
    fid_samples,
    fid_bs,
    ema_factor: Optional[float] = None,
    network_slot: Optional[str] = None,
    compute_visuals: bool = True,
):
    """Sample from a checkpoint with specified configuration."""
    path = (
        cfg.network.load_path
        if use_main_network(cfg, sample_type, network_slot=network_slot)
        else cfg.sup_network.load_path
    )
    print(
        f"Sampling from {path} with {outer_steps},{inner_steps} and {sample_type.name} and {ema_factor}"
    )

    params = get_params(
        cfg,
        statics,
        train_state,
        sample_type,
        ema_factor_override=ema_factor,
        network_slot=network_slot,
    )

    sample_fn = create_sample_fn(
        cfg,
        statics,
        sample_type,
        outer_steps,
        inner_steps,
        network_slot=network_slot,
    )
    decode_fn = statics.decode_fn

    batch_x0s = None
    batch_full_traj = None
    batch_x_final = None
    if compute_visuals:
        plot_dims = latent_utils.get_pixel_image_dims(cfg)

        prng_key, subkey = jax.random.split(prng_key)

        # calculate full trajectory
        sample_batch = sample_fn(
            subkey, params, bs=1, return_traj=True, sharded=False
        )  # only need 1 sample for trajectory
        batch_x0s = sample_batch.batch_x0s
        batch_full_traj = sample_batch.batch_full_traj

        # calculate final samples
        prng_key, subkey = jax.random.split(prng_key)
        sample_batch = sample_fn(
            subkey, params, bs=cfg.logging.plot_bs, return_traj=False, sharded=False
        )
        batch_x_final = sample_batch.batch_x_final

    # calculate fid
    prng_key, subkey = jax.random.split(prng_key)
    fid = None
    if comp_fid:
        fid = calc_fid(
            cfg,
            statics.inception_fn,
            decode_fn,
            sample_fn,
            subkey,
            params,
            n_samples=fid_samples,
            bs=fid_bs,
        )
        print(f"FID={fid}")

    if not compute_visuals:
        return batch_x_final, batch_full_traj, fid

    batch_x0s = latent_utils.maybe_decode_latents_chunked(
        cfg, batch_x0s, chunk_size=latent_utils.LATENT_DECODE_CHUNK, decode_fn=decode_fn
    )
    batch_full_traj = latent_utils.maybe_decode_latents_chunked(
        cfg, batch_full_traj, chunk_size=latent_utils.LATENT_DECODE_CHUNK, decode_fn=decode_fn
    )
    batch_x_final = latent_utils.maybe_decode_latents_chunked(
        cfg, batch_x_final, chunk_size=latent_utils.LATENT_DECODE_CHUNK, decode_fn=decode_fn
    )

    batch_x_final = jnp.transpose(batch_x_final, (0, 2, 3, 1))

    batch_full_traj = jnp.reshape(
        batch_full_traj,
        (1, outer_steps * inner_steps, *plot_dims),
    )
    batch_full_traj = jnp.concatenate(
        [jnp.expand_dims(batch_x0s, 1), batch_full_traj], axis=1
    )
    batch_full_traj = jnp.transpose(batch_full_traj, (0, 1, 3, 4, 2))
    return batch_x_final, batch_full_traj, fid


def collect_x1_batch(ds, batch_size):
    x1_chunks = []
    label_chunks = []
    total = 0
    while total < batch_size:
        batch = next(ds)
        if isinstance(batch, dict):
            x1_batch = batch["image"]
            label_batch = batch.get("label")
        else:
            x1_batch = batch
            label_batch = None
        x1_batch = np.asarray(x1_batch)
        x1_chunks.append(x1_batch)
        if label_batch is not None:
            label_chunks.append(np.asarray(label_batch))
        total += x1_batch.shape[0]

    x1s = np.concatenate(x1_chunks, axis=0)[:batch_size]
    labels = None
    if label_chunks:
        labels = np.concatenate(label_chunks, axis=0)[:batch_size]
    return x1s, labels


def make_posterior_sample_fn(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    sample_type: SampleType,
    outer_steps: int,
    inner_steps: int,
    t: float,
    init_override: Optional[Callable[[jnp.ndarray, int, bool], GroundtruthInit]] = None,
    network_slot: Optional[str] = None,
):
    ts_override = jnp.linspace(t, 1.0, outer_steps + 1)
    sampler = create_sampler(
        cfg,
        statics,
        sample_type,
        outer_steps,
        inner_steps,
        ts_override=ts_override,
        network_slot=network_slot,
    )

    if init_override is not None:
        return create_groundtruth_sample_fn(sampler, init_override)

    def init_fn(prng_key, bs, sharded):
        prng_key, x0_key = jax.random.split(prng_key)
        batch_x1s, batch_labels = collect_x1_batch(statics.ds, bs)
        batch_x1s = jnp.asarray(batch_x1s, dtype=jnp.float32)
        batch_x0s = statics.sample_rho0(bs, x0_key)
        t_batch = jnp.full((bs,), t)
        batch_x_init = statics.interp.batch_calc_It(t_batch, batch_x0s, batch_x1s)

        if cfg.training.conditional:
            if batch_labels is None:
                raise ValueError("Dataset labels required for conditional sampling.")
            batch_labels = jnp.asarray(batch_labels, dtype=jnp.int32)
        else:
            batch_labels = None

        prng_key, step_key = jax.random.split(prng_key)
        batch_subkeys = jax.random.split(step_key, num=bs)

        return make_groundtruth_init(
            cfg,
            sharded=sharded,
            batch_x_init=batch_x_init,
            batch_prng_key=batch_subkeys,
            batch_x0s=batch_x0s,
            batch_x1s=batch_x1s,
            batch_measurement=None,
            batch_labels=batch_labels,
        )

    return create_groundtruth_sample_fn(sampler, init_fn)


def make_posterior_sample_plot(
    cfg,
    statics,
    train_state,
    outer_step,
    inner_step,
    t,
    ema_factor,
    prng_key,
    posterior_sample_types,
    network_slot: Optional[str] = None,
):
    bs = cfg.logging.plot_bs
    decode_fn = statics.decode_fn
    prng_key, x0_key = jax.random.split(prng_key)
    batch_x1s, batch_labels = collect_x1_batch(statics.ds, bs)
    batch_x1s = jnp.asarray(batch_x1s, dtype=jnp.float32)
    batch_x0s = statics.sample_rho0(bs, x0_key)
    t_batch = jnp.full((bs,), t)
    batch_x_init = statics.interp.batch_calc_It(t_batch, batch_x0s, batch_x1s)

    if cfg.training.conditional:
        if batch_labels is None:
            raise ValueError("Dataset labels required for conditional sampling.")
        batch_labels = jnp.asarray(batch_labels, dtype=jnp.int32)
    else:
        batch_labels = None

    # make plot based on init data
    def build_plot(
        prng_key_inner, batch_x1s_inner, batch_x0s_inner, batch_x_init_inner, labels
    ):
        plot_dims = latent_utils.get_pixel_image_dims(cfg)
        nrows = len(posterior_sample_types) + 2
        xfinals = np.zeros(
            (
                nrows,
                bs,
                plot_dims[1],
                plot_dims[2],
                plot_dims[0],
            )
        )
        batch_x1s_plot = latent_utils.maybe_decode_latents_chunked(
            cfg,
            batch_x1s_inner,
            chunk_size=latent_utils.LATENT_DECODE_CHUNK,
            decode_fn=decode_fn,
        )
        batch_x_init_plot = latent_utils.maybe_decode_latents_chunked(
            cfg,
            batch_x_init_inner,
            chunk_size=latent_utils.LATENT_DECODE_CHUNK,
            decode_fn=decode_fn,
        )
        xfinals[0] = np.asarray(jnp.transpose(batch_x1s_plot, (0, 2, 3, 1)))
        xfinals[1] = np.asarray(jnp.transpose(batch_x_init_plot, (0, 2, 3, 1)))

        # override with the same batch_x0s, x1s, etc for a correct comparison
        def init_override(prng_key_override, bs_inner, sharded):
            if bs_inner != bs:
                raise ValueError("Plot batch size mismatch.")
            prng_key_override, step_key = jax.random.split(prng_key_override)
            batch_subkeys = jax.random.split(step_key, num=bs_inner)
            return make_groundtruth_init(
                cfg,
                sharded=sharded,
                batch_x_init=batch_x_init_inner,
                batch_prng_key=batch_subkeys,
                batch_x0s=batch_x0s_inner,
                batch_x1s=batch_x1s_inner,
                batch_measurement=None,
                batch_labels=labels,
            )

        for idx, sample_type in enumerate(posterior_sample_types):
            params = get_params(
                cfg,
                statics,
                train_state,
                sample_type,
                ema_factor_override=ema_factor,
                network_slot=network_slot,
            )
            sample_fn = make_posterior_sample_fn(
                cfg,
                statics,
                sample_type,
                outer_step,
                inner_step,
                t,
                init_override=init_override,
                network_slot=network_slot,
            )

            prng_key_inner, step_key = jax.random.split(prng_key_inner)
            sample_batch = sample_fn(
                step_key, params, bs, return_traj=False, sharded=False
            )
            batch_x_final = latent_utils.maybe_decode_latents_chunked(
                cfg,
                sample_batch.batch_x_final,
                chunk_size=latent_utils.LATENT_DECODE_CHUNK,
                decode_fn=decode_fn,
            )
            xfinals[idx + 2] = np.asarray(jnp.transpose(batch_x_final, (0, 2, 3, 1)))

        fig = make_sample_plot(nrows, bs, xfinals)
        return prng_key_inner, fig

    prng_key, sample_fig = build_plot(
        prng_key, batch_x1s, batch_x0s, batch_x_init, batch_labels
    )

    sample_idx = 0
    batch_x1s_single = batch_x1s[sample_idx : sample_idx + 1]
    batch_x0s_single = batch_x0s[sample_idx : sample_idx + 1]
    batch_x_init_single = batch_x_init[sample_idx : sample_idx + 1]
    if batch_labels is None:
        batch_labels_single = None
    else:
        batch_labels_single = batch_labels[sample_idx : sample_idx + 1]

    batch_x1s_repeated = jnp.repeat(batch_x1s_single, bs, axis=0)
    batch_x0s_repeated = jnp.repeat(batch_x0s_single, bs, axis=0)
    batch_x_init_repeated = jnp.repeat(batch_x_init_single, bs, axis=0)
    if batch_labels_single is None:
        batch_labels_repeated = None
    else:
        batch_labels_repeated = jnp.repeat(batch_labels_single, bs, axis=0)

    prng_key, multiseed_fig = build_plot(
        prng_key,
        batch_x1s_repeated,
        batch_x0s_repeated,
        batch_x_init_repeated,
        batch_labels_repeated,
    )
    return prng_key, sample_fig, multiseed_fig
