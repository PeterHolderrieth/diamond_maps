"""Training loss builders for flow-map and diamond-map matching."""

import functools
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import flow_map as flow_map
from . import interpolant as interpolant
from . import loss_args

Parameters = Dict[str, Dict]


def mean_reduce(func):
    """
    A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return jnp.mean(batched_outputs)

    return wrapper


def adaptive_norm_weight(
    loss: jnp.ndarray, norm_eps: float, norm_p: float
) -> jnp.ndarray:
    """Apply meanflow-style adaptive loss normalization."""
    norm_weight = (loss + norm_eps) ** norm_p
    return loss / jax.lax.stop_gradient(norm_weight)


def make_loss_rescale_decorator(norm_eps: float, norm_p: float):
    """Create a decorator that applies adaptive normalization to loss outputs."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return adaptive_norm_weight(
                func(*args, **kwargs), norm_eps=norm_eps, norm_p=norm_p
            )

        return wrapper

    return decorator


def _apply_cfg_velocity(
    label: Optional[jnp.ndarray],
    cfg_omega: float,
    cfg_kappa: float,
    cfg_unconditional_label: Optional[int],
    velocity_fn: Callable[[Optional[jnp.ndarray]], jnp.ndarray],
    *,
    sup_cfg_batch: bool = False,
    v_t: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """CFG velocity field"""
    if cfg_unconditional_label is None:
        return velocity_fn(label)

    if sup_cfg_batch:
        # Pack cond/uncond labels into one local micro-batch.
        packed_labels = jnp.stack(
            [
                jnp.asarray(label, dtype=jnp.int32),
                jnp.asarray(cfg_unconditional_label, dtype=jnp.int32),
            ]
        )
        v_cond, v_uncond = jax.vmap(velocity_fn)(packed_labels)
    else:
        v_cond, v_uncond = velocity_fn(label), velocity_fn(cfg_unconditional_label)

    if v_t is None:
        v_t = v_cond
    return (
        cfg_omega * v_t + (1.0 - cfg_omega - cfg_kappa) * v_uncond + cfg_kappa * v_cond
    )


def diagonal_term(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t_prime: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    cfg_omega: float = 1.0,
    cfg_kappa: float = 0.0,
    cfg_unconditional_label: Optional[int] = None,
) -> float:
    """Compute the diagonal (interpolant) term of the loss."""
    # compute interpolant and the target
    I_t = interp.calc_It(t_prime, x0, x1)
    I_t_dot = interp.calc_It_dot(t_prime, x0, x1)

    # compute the velocity matching loss
    bt = _apply_cfg_velocity(
        label,
        cfg_omega,
        cfg_kappa,
        cfg_unconditional_label,
        lambda curr_label: X.apply(
            params, t_prime, I_t, curr_label, train=True, method="calc_b", rngs=rng
        ),
        v_t=I_t_dot,
    )
    velocity_loss = jnp.sum((bt - I_t_dot) ** 2)

    weight_tt = X.apply(params, t_prime, t_prime, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


def diagonal_term_supervised(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    t_prime: float,
    s: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.DiamondMap,
    sup_net,
    sup_params: Parameters,
    xbar_rng: jnp.ndarray,
    sup_cfg_omega: float = 1.0,
    sup_cfg_kappa: float = 0.0,
    sup_cfg_unconditional_label: Optional[int] = None,
    sup_cfg_batch: bool = False,
) -> float:
    """Compute the diagonal (interpolant) term of the loss supervised by another network."""

    # compute interpolant and the target
    x_t = interp.calc_It(t, x0, x1)
    xbar_s = X.calc_xbar_s(xbar_rng, t, t_prime, s, x_t, x1)
    target = jax.lax.stop_gradient(
        _apply_cfg_velocity(
            label,
            sup_cfg_omega,
            sup_cfg_kappa,
            sup_cfg_unconditional_label,
            lambda curr_label: sup_net.apply(
                sup_params,
                t,
                t_prime,
                s,
                x_t,
                xbar_s,
                label=curr_label,
                train=False,
            ),
            sup_cfg_batch=sup_cfg_batch,
        )
    )

    bt = X.apply(
        params,
        t,
        t_prime,
        s,
        x_t,
        xbar_s,
        label=label,
        train=True,
        method="calc_b",
        rngs=rng,
    )

    velocity_loss = jnp.sum((bt - target) ** 2)

    weight = X.apply(params, t, t_prime, s, s, method="calc_weight")
    return jnp.exp(-weight) * velocity_loss + weight


def lsd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    t_prime: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
    cfg_omega: float = 1.0,
    cfg_kappa: float = 0.0,
    cfg_unconditional_label: Optional[int] = None,
) -> float:
    """Compute the LSD term of the loss."""
    I_t = interp.calc_It(t, x0, x1)

    # Compute the distillation loss
    x_t_prime, dt_prime_x_t_prime = X.apply(
        params, t, t_prime, I_t, label, train=False, method="partial_t_prime", rngs=rng
    )

    if stopgrad_type == "convex":
        x_t_prime = jax.lax.stop_gradient(x_t_prime)
        b_eval = jax.lax.stop_gradient(
            _apply_cfg_velocity(
                label,
                cfg_omega,
                cfg_kappa,
                cfg_unconditional_label,
                lambda curr_label: X.apply(
                    teacher_params,
                    t_prime,
                    x_t_prime,
                    curr_label,
                    train=False,
                    method="calc_b",
                    rngs=rng,
                ),
            )
        )
    elif stopgrad_type == "none":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    error = b_eval - dt_prime_x_t_prime
    lsd_loss = jnp.sum(error**2)

    weight_t_t_prime = X.apply(params, t, t_prime, method="calc_weight")
    return jnp.exp(-weight_t_t_prime) * lsd_loss + weight_t_t_prime


def lsd_term_supervised(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    t_prime: float,
    s: float,
    s_prime: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.DiamondMap,
    stopgrad_type: str,
    sup_net,
    sup_params: Parameters,
    xbar_rng: jnp.ndarray,
    sup_cfg_omega: float = 1.0,
    sup_cfg_kappa: float = 0.0,
    sup_cfg_unconditional_label: Optional[int] = None,
    sup_cfg_batch: bool = False,
) -> float:
    """Compute the LSD term of the loss."""
    x_t = interp.calc_It(t, x0, x1)
    xbar_s = X.calc_xbar_s(xbar_rng, t, t_prime, s, x_t, x1)
    # Compute the distillation loss
    xbar_s_prime, ds_prime_xbar = X.apply(
        params,
        t,
        t_prime,
        s,
        s_prime,
        x_t,
        xbar_s,
        label=label,
        train=False,
        method="partial_s_prime",
        rngs=rng,
    )

    if stopgrad_type == "convex":
        xbar_s_prime = jax.lax.stop_gradient(xbar_s_prime)
        b_eval = jax.lax.stop_gradient(
            _apply_cfg_velocity(
                label,
                sup_cfg_omega,
                sup_cfg_kappa,
                sup_cfg_unconditional_label,
                lambda curr_label: sup_net.apply(
                    sup_params,
                    t,
                    t_prime,
                    s_prime,
                    x_t,
                    xbar_s_prime,
                    label=curr_label,
                    train=False,
                ),
                sup_cfg_batch=sup_cfg_batch,
            )
        )
    elif stopgrad_type == "none":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    lsd_loss = jnp.sum((b_eval - ds_prime_xbar) ** 2)

    weight = X.apply(params, t, t_prime, s, s_prime, method="calc_weight")
    return jnp.exp(-weight) * lsd_loss + weight


def setup_loss(
    cfg: config_dict.ConfigDict,
    net: flow_map.FlowMap,
    interp: interpolant.Interpolant,
    sup_net=None,
) -> Callable:
    """Setup the loss function."""

    print(f"Setting up loss: {cfg.training.loss_type}")
    print(f"Stopgrad type: {cfg.training.stopgrad_type}")
    norm_eps = cfg.training.norm_eps
    norm_p = cfg.training.norm_p
    print(f"Adaptive norm weighting: eps={norm_eps}, p={norm_p}")
    rescale_loss = make_loss_rescale_decorator(norm_eps=norm_eps, norm_p=norm_p)

    cfg_omega = float(cfg.training.cfg_omega)
    cfg_kappa = float(cfg.training.cfg_kappa)
    sup_cfg_velocity = bool(cfg.training.sup_cfg_velocity)
    supervise_type = cfg.training.supervise_type
    uses_glass_supervision = supervise_type == "glass"

    cfg_unconditional_label = None
    sup_cfg_unconditional_label = None
    if sup_cfg_velocity:
        if uses_glass_supervision:
            sup_cfg_unconditional_label = int(cfg.problem.num_classes)
        else:
            cfg_unconditional_label = int(cfg.problem.num_classes)

    sup_cfg_batch = bool(cfg.training.sup_cfg_batch)

    if uses_glass_supervision:
        assert sup_net is not None

        # Supervised diagonal loss
        @mean_reduce
        @rescale_loss
        @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))
        def diagonal_only_loss(
            params, sup_params, x0, x1, label, t, t_prime, s, rng, xbar_rng
        ):

            rng = {"dropout": rng}
            return diagonal_term_supervised(
                params,
                x0,
                x1,
                label,
                t,
                t_prime,
                s,
                rng,
                interp=interp,
                X=net,
                sup_net=sup_net,
                sup_params=sup_params,
                xbar_rng=xbar_rng,
                sup_cfg_omega=cfg_omega,
                sup_cfg_kappa=cfg_kappa,
                sup_cfg_unconditional_label=sup_cfg_unconditional_label,
                sup_cfg_batch=sup_cfg_batch,
            )

    else:
        # Pure diagonal loss
        @mean_reduce
        @rescale_loss
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
        def diagonal_only_loss(params, x0, x1, label, t_prime, rng):

            rng = {"dropout": rng}
            return diagonal_term(
                params,
                x0,
                x1,
                label,
                t_prime,
                rng,
                interp=interp,
                X=net,
                cfg_omega=cfg_omega,
                cfg_kappa=cfg_kappa,
                cfg_unconditional_label=cfg_unconditional_label,
            )

    if uses_glass_supervision:
        # Supervised off-diagonal loss
        @mean_reduce
        @rescale_loss
        @functools.partial(
            jax.vmap,
            in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        def offdiagonal_only_loss(
            params,
            teacher_params,
            sup_params,
            x0,
            x1,
            label,
            t,
            t_prime,
            s,
            s_prime,
            dropout_keys,
            xbar_s_keys,
        ):
            rng = {"dropout": dropout_keys}
            return lsd_term_supervised(
                params,
                teacher_params,
                x0,
                x1,
                label,
                t,
                t_prime,
                s,
                s_prime,
                rng,
                interp=interp,
                X=net,
                stopgrad_type=cfg.training.stopgrad_type,
                sup_net=sup_net,
                sup_params=sup_params,
                xbar_rng=xbar_s_keys,
                sup_cfg_omega=cfg_omega,
                sup_cfg_kappa=cfg_kappa,
                sup_cfg_unconditional_label=sup_cfg_unconditional_label,
                sup_cfg_batch=sup_cfg_batch,
            )

    else:
        # Pure off-diagonal loss
        @mean_reduce
        @rescale_loss
        @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
        def offdiagonal_only_loss(
            params, teacher_params, x0, x1, label, t, t_prime, dropout_keys
        ):
            rng = {"dropout": dropout_keys}
            return lsd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                t,
                t_prime,
                rng,
                interp=interp,
                X=net,
                stopgrad_type=cfg.training.stopgrad_type,
                cfg_omega=cfg_omega,
                cfg_kappa=cfg_kappa,
                cfg_unconditional_label=cfg_unconditional_label,
            )

    if uses_glass_supervision:

        def loss(
            params,
            teacher_params,
            sup_params,
            x0,
            x1,
            label,
            t,
            t_prime,
            s_diag,
            s,
            s_prime,
            dropout_keys,
            xbar_s_keys,
        ):
            """Split batch into diagonal and off-diagonal portions."""
            total_bs = x0.shape[0]
            diag_bs, offdiag_bs = loss_args._get_diag_offdiag_bs(cfg, total_bs)

            total_loss = 0.0

            # Compute diagonal loss on first portion
            if diag_bs > 0:
                label_diag = None if label is None else label[:diag_bs]
                diag_loss = diagonal_only_loss(
                    params,
                    sup_params,
                    x0[:diag_bs],
                    x1[:diag_bs],
                    label_diag,
                    t[:diag_bs],
                    t_prime[:diag_bs],
                    s_diag[:diag_bs],
                    dropout_keys[:diag_bs],
                    xbar_s_keys[:diag_bs],
                )
                total_loss += diag_loss * diag_bs

            # Compute off-diagonal loss on second portion
            if offdiag_bs > 0:
                label_offdiag = None if label is None else label[diag_bs:]

                offdiag_loss = offdiagonal_only_loss(
                    params,
                    teacher_params,
                    sup_params,
                    x0[diag_bs:],
                    x1[diag_bs:],
                    label_offdiag,
                    t[diag_bs:],
                    t_prime[diag_bs:],
                    s[diag_bs:],
                    s_prime[diag_bs:],
                    dropout_keys[diag_bs:],
                    xbar_s_keys[diag_bs:],
                )
                total_loss += offdiag_loss * offdiag_bs

            # Normalize by total batch size
            return total_loss / total_bs

    else:

        def loss(params, teacher_params, x0, x1, label, t, t_prime, dropout_keys):
            """Split batch into diagonal and off-diagonal portions."""
            total_bs = x0.shape[0]
            diag_bs, offdiag_bs = loss_args._get_diag_offdiag_bs(cfg, total_bs)

            total_loss = 0.0

            # Compute diagonal loss on first portion
            if diag_bs > 0:
                label_diag = None if label is None else label[:diag_bs]
                diag_loss = diagonal_only_loss(
                    params,
                    x0[:diag_bs],
                    x1[:diag_bs],
                    label_diag,
                    t_prime[:diag_bs],
                    dropout_keys[:diag_bs],
                )
                total_loss += diag_loss * diag_bs

            # Compute off-diagonal loss on second portion
            if offdiag_bs > 0:
                label_offdiag = None if label is None else label[diag_bs:]

                offdiag_loss = offdiagonal_only_loss(
                    params,
                    teacher_params,
                    x0[diag_bs:],
                    x1[diag_bs:],
                    label_offdiag,
                    t[diag_bs:],
                    t_prime[diag_bs:],
                    dropout_keys[diag_bs:],
                )
                total_loss += offdiag_loss * offdiag_bs

            # Normalize by total batch size
            return total_loss / total_bs

    return loss
