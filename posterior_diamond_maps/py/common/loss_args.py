"""Sampling utilities for per-step loss-function arguments."""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import state_utils
from . import dist_utils


def _sample_diagonal(
    key: jnp.ndarray, bs: int, t_min: float, t_max: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample points on the diagonal (t=t_prime)."""
    t = jax.random.uniform(key, shape=(bs,), minval=t_min, maxval=t_max)
    return t, t


def _sample_triangle(
    key1: jnp.ndarray,
    key2: jnp.ndarray,
    bs: int,
    t_min: float,
    t_max: float,
    gap: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample uniformly from upper triangle with t_prime - t >= gap."""
    eff_t_max = t_max - gap
    temp1 = jax.random.uniform(key1, shape=(bs,), minval=t_min, maxval=eff_t_max)
    temp2 = jax.random.uniform(key2, shape=(bs,), minval=t_min, maxval=eff_t_max)
    t = jnp.minimum(temp1, temp2)
    t_prime = jnp.maximum(temp1, temp2)
    t_prime += gap
    return t, t_prime


def _get_diag_offdiag_bs(cfg: config_dict.ConfigDict, bs: int) -> Tuple[int, int]:
    """Get diagonal and off-diagonal batch sizes."""
    diag_bs = max(1, int(bs * cfg.optimization.diag_fraction))
    offdiag_bs = bs - diag_bs

    return diag_bs, offdiag_bs


def _concat_diag_offdiag(
    t_diag: jnp.ndarray,
    t_prime_diag: jnp.ndarray,
    t_offdiag: jnp.ndarray,
    t_prime_offdiag: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Concatenate diagonal and off-diagonal samples."""
    t_batch = jnp.concatenate([t_diag, t_offdiag])
    t_prime_batch = jnp.concatenate([t_prime_diag, t_prime_offdiag])
    return t_batch, t_prime_batch


@functools.partial(jax.jit, static_argnums=(1, 2))
def get_loss_fn_args_randomness_diamond(
    prng_key: jnp.ndarray,
    cfg: config_dict.ConfigDict,
    sample_rho0: Callable,
) -> Tuple:
    """Draw random values needed for each loss function iteration."""
    (
        t_prime_key,
        t_key,
        x0key,
        s_diag_key,
        s_key,
        s_prime_key,
        dropout_key,
        xbar_s_key,
        prng_key,
    ) = jax.random.split(prng_key, num=9)
    x0batch = sample_rho0(cfg.optimization.bs, x0key)

    bs = cfg.optimization.bs
    t_min, t_max = cfg.training.t_min, cfg.training.t_max
    s_min, s_max = cfg.training.s_min, cfg.training.s_max

    t_batch, t_prime_batch = _sample_triangle(t_key, t_prime_key, bs, t_min, t_max, 0.0)
    if cfg.training.fixed_t_prime is not None:
        t_batch, _ = _sample_diagonal(t_key, bs, t_min, t_max)
        t_prime_batch = jnp.ones((bs,)) * cfg.training.fixed_t_prime
    if cfg.training.fixed_t is not None:
        t_batch = jnp.ones((bs,)) * cfg.training.fixed_t

    s_diag_batch, _ = _sample_diagonal(s_diag_key, bs, s_min, s_max)
    s_batch, s_prime_batch = _sample_triangle(
        s_key, s_prime_key, bs, s_min, s_max, 0.0
    )

    dropout_keys = jax.random.split(dropout_key, num=bs).reshape((bs, -1))
    xbar_s_keys = jax.random.split(xbar_s_key, num=bs).reshape((bs, -1))
    return (
        t_batch,
        t_prime_batch,
        x0batch,
        s_diag_batch,
        s_batch,
        s_prime_batch,
        dropout_keys,
        xbar_s_keys,
        prng_key,
    )


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def get_loss_fn_args_randomness(
    prng_key: jnp.ndarray,
    cfg: config_dict.ConfigDict,
    sample_rho0: Callable,
    diag_bs: int,
    offdiag_bs: int,
) -> Tuple:
    """Draw random values needed for each loss function iteration."""
    (
        t_prime_key,
        t_key,
        x0key,
        tkey2,
        dropout_key,
        prng_key,
    ) = jax.random.split(prng_key, num=6)
    x0batch = sample_rho0(cfg.optimization.bs, x0key)

    bs = cfg.optimization.bs
    t_min = cfg.training.t_min
    t_max = cfg.training.t_max

    # If offdiag_bs is 0, use full batch on diagonal
    if offdiag_bs == 0:
        t_batch, t_prime_batch = _sample_diagonal(t_key, bs, t_min, t_max)
    else:
        # sample diagonal and off-diagonal points
        t_diag, t_prime_diag = (
            _sample_diagonal(t_key, diag_bs, t_min, t_max)
            if diag_bs > 0
            else (jnp.array([]), jnp.array([]))
        )
        t_offdiag, t_prime_offdiag = (
            _sample_triangle(t_prime_key, tkey2, offdiag_bs, t_min, t_max, 0.0)
            if offdiag_bs > 0
            else (jnp.array([]), jnp.array([]))
        )

        t_batch, t_prime_batch = _concat_diag_offdiag(
            t_diag, t_prime_diag, t_offdiag, t_prime_offdiag
        )

    dropout_keys = jax.random.split(dropout_key, num=cfg.optimization.bs).reshape(
        (cfg.optimization.bs, -1)
    )
    return (
        t_batch,
        t_prime_batch,
        x0batch,
        dropout_keys,
        prng_key,
    )


def get_batch(
    cfg: config_dict.ConfigDict, statics: state_utils.StaticArgs, prng_key: jnp.ndarray
) -> int:
    """Extract a batch based on the structure expected for image
    or non-image datasets."""
    is_image_dataset = (cfg.problem.target in ["cifar10", "celeb_a"]) or (
        "imagenet_latent" in cfg.problem.target
    )

    batch = next(statics.ds)
    if is_image_dataset:
        x1batch = batch["image"]
        label_batch = jnp.array(batch["label"])
    else:
        x1batch = batch
        label_batch = None

    x1batch = jnp.array(x1batch)

    # add droput to randomly replace fraction cfg.class_dropout of labels by num_classes
    # if not conditional, we don't need the labels
    if not cfg.training.conditional:
        label_batch = None

    elif cfg.training.class_dropout > 0:
        assert cfg.network.use_cfg_token
        mask = jax.random.bernoulli(
            prng_key, cfg.training.class_dropout, shape=(cfg.optimization.bs,)
        )
        mask = mask > 0
        label_batch = label_batch.at[mask].set(cfg.problem.num_classes)
        prng_key = jax.random.split(prng_key)[0]

    return x1batch, label_batch, prng_key


def get_loss_fn_args(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> Tuple:

    # Determine batch sizes based on splitting configuration
    bs = cfg.optimization.bs

    # Normal batch splitting
    diag_bs, offdiag_bs = _get_diag_offdiag_bs(cfg, bs)

    # drew randomness needed for the objective
    (
        t_batch,
        t_prime_batch,
        x0batch,
        dropout_keys,
        prng_key,
    ) = get_loss_fn_args_randomness(
        prng_key,
        cfg,
        statics.sample_rho0,
        diag_bs,
        offdiag_bs,
    )

    # grab next batch of samples and labels
    x1batch, label_batch, prng_key = get_batch(cfg, statics, prng_key)

    # set up the teacher (uses current params for self-distillation)
    teacher_params = train_state.params

    # for training flow map
    loss_fn_args = (
        x0batch,
        x1batch,
        label_batch,
        t_batch,
        t_prime_batch,
        dropout_keys,
    )
    loss_fn_args = dist_utils.replicate_loss_fn_args(cfg, loss_fn_args)
    loss_fn_args = (teacher_params, *loss_fn_args)

    return loss_fn_args, prng_key


def get_loss_fn_args_diamond(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> Tuple:
    # grab next batch of samples and labels
    x1batch, label_batch, prng_key = get_batch(cfg, statics, prng_key)

    # drew randomness needed for the objective
    (
        t_batch,
        t_prime_batch,
        x0batch,
        s_diag_batch,
        s_batch,
        s_prime_batch,
        dropout_keys,
        xbar_s_keys,
        prng_key,
    ) = get_loss_fn_args_randomness_diamond(
        prng_key,
        cfg,
        statics.sample_rho0,
    )

    # set up the teacher (uses current params for self-distillation)
    teacher_params = train_state.params
    supervise_type = cfg.training.supervise_type
    if supervise_type not in {"none", "glass"}:
        raise ValueError(f"Unsupported training.supervise_type: {supervise_type}")
    uses_glass_supervision = supervise_type == "glass"

    # for training flow map
    loss_fn_args = (
        x0batch,
        x1batch,
        label_batch,
        t_batch,
        t_prime_batch,
        s_diag_batch,
        s_batch,
        s_prime_batch,
        dropout_keys,
        xbar_s_keys,
    )
    loss_fn_args = dist_utils.replicate_loss_fn_args(cfg, loss_fn_args)
    if uses_glass_supervision:
        assert statics.sup_params is not None, "Expected supervising params to be set."
        loss_fn_args = (teacher_params, statics.sup_params, *loss_fn_args)
    else:
        loss_fn_args = (teacher_params, *loss_fn_args)

    return loss_fn_args, prng_key


def setup_loss_fn_args(cfg):
    if cfg.network.matching == "diamond_map":
        return get_loss_fn_args_diamond
    return get_loss_fn_args
