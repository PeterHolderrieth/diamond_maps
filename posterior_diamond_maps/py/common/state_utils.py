"""Training-state, optimizer, checkpoint, and model setup helpers."""

from copy import deepcopy
from typing import Any, Callable, Dict, NamedTuple, Tuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pickle
import tensorflow as tf
from flax import struct, traverse_util
from flax.serialization import from_state_dict
from flax.training import train_state
from ml_collections import config_dict

from . import flow_map, interpolant, flow

Parameters = Dict[str, Dict]


class EMATrainState(train_state.TrainState):
    """Train state including EMA parameters."""

    ema_params: Dict[float, Any] = struct.field(default_factory=dict)


class StaticArgs(NamedTuple):
    net: nn.Module
    schedule: optax.Schedule
    loss: Callable
    get_loss_fn_args: Callable
    train_step: Callable
    ds: tf.data.Dataset
    interp: interpolant.Interpolant
    sample_rho0: Callable
    inception_fn: Optional[Callable] = None  # For FID computation
    decode_fn: Optional[Callable] = None
    sup_net: Optional[nn.Module] = None
    sup_params: Optional[Parameters] = None


def _load_checkpoint_payload(load_path: str) -> Any:
    """Load a checkpoint payload onto CPU memory."""
    cpu_device = jax.devices("cpu")[0]
    with open(load_path, "rb") as f:
        with jax.default_device(cpu_device):
            return pickle.load(f)


def load_params_from_checkpoint(
    load_path: str,
    ema_factor: float,
) -> Parameters:
    """Load a params tree directly from checkpoint without building train state."""
    print(f"Loading params from checkpoint {load_path}")
    state_dict = _load_checkpoint_payload(load_path)
    return state_dict["ema_params"][str(ema_factor)] # train state loads keys as string


def load_checkpoint(
    load_path: str,
    train_state: EMATrainState,
) -> EMATrainState:
    """Load a training checkpoint."""
    print(f"Loading checkpoint from {load_path}")
    state_dict = _load_checkpoint_payload(load_path)
    train_state = from_state_dict(train_state, state_dict)
    return train_state


def setup_schedule(
    cfg: config_dict.ConfigDict,
) -> optax.Schedule:
    """Set up the learning rate schedule."""
    if cfg.optimization.schedule_type == "cosine":
        return optax.cosine_decay_schedule(
            init_value=cfg.optimization.learning_rate,
            decay_steps=cfg.optimization.decay_steps,
            alpha=0.0,
        )
    elif cfg.optimization.schedule_type == "sqrt":
        return lambda step: cfg.optimization.learning_rate / jnp.sqrt(
            jnp.maximum(step / cfg.optimization.decay_steps, 1.0)
        )
    elif cfg.optimization.schedule_type == "constant":
        return lambda step: cfg.optimization.learning_rate
    else:
        raise ValueError(f"Unknown schedule type: {cfg.schedule_type}")


def setup_optimizer(cfg: config_dict.ConfigDict):
    """Set up the optimizer."""
    schedule = setup_schedule(cfg)

    # optimizer mask for positional embeddings (which do not have a constants key)
    def mask_fn(variables):
        masks = {
            "params": jax.tree_util.tree_map(lambda _: True, variables["params"]),
        }
        if "constants" in variables:  # network has Fourier tables
            masks["constants"] = jax.tree_util.tree_map(
                lambda _: False, variables["constants"]
            )
        return masks

    # define optimizer
    tx = optax.masked(
        optax.chain(
            optax.clip_by_global_norm(cfg.optimization.clip),
            optax.radam(
                learning_rate=schedule,
                b1=cfg.optimization.b1,
                b2=cfg.optimization.b2,
                eps=cfg.optimization.eps,
            ),
        ),
        mask_fn,
    )

    return tx, schedule


def _copy_matching_values(src_tree: Dict, dst_tree: Dict) -> Tuple[Dict, int, int]:
    """Copy parameters from a converted flow-map checkpoint to diamond map"""
    src_flat = traverse_util.flatten_dict(src_tree)
    dst_flat = traverse_util.flatten_dict(dst_tree)

    reused = 0
    total = len(dst_flat)
    merged = {}

    for key, dst_val in dst_flat.items():
        merged[key] = dst_val

        # replace diamond_map with flow_map in the dst key
        remapped_key = tuple(
            "flow_map" if part == "diamond_map" else part for part in key
        )
        src_val = src_flat.get(remapped_key, None)
        if src_val is not None and src_val.shape == dst_val.shape:
            merged[key] = src_val
            reused += 1

    return traverse_util.unflatten_dict(merged), reused, total


def _build_top_level_model(
    network_cfg: config_dict.ConfigDict,
    interp_type: str,
) -> nn.Module:
    if network_cfg.use_glass:
        assert network_cfg.matching in {"flow", "flow_map"}, "GLASS requires matching to be 'flow' or 'flow_map'."
        return flow.GlassFlow(config=network_cfg, interp_type=interp_type)
    if network_cfg.matching == "diamond_map":
        return flow_map.DiamondMap(config=network_cfg, interp_type=interp_type)
    return flow_map.FlowMap(network_cfg)


def _initialize_main_model(
    cfg: config_dict.ConfigDict,
    ex_input: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> Tuple[nn.Module, Parameters, jnp.ndarray]:
    if cfg.network.use_glass:
        assert cfg.network.load_path != "", "GLASS requires network.load_path to point to a base checkpoint"
        net = _build_top_level_model(cfg.network, cfg.problem.interp_type)
        params = load_params_from_checkpoint(
            cfg.network.load_path,
            ema_factor=cfg.logging.ema_factor,
        )
        return net, params, prng_key

    if cfg.network.matching == "diamond_map":
        net, params, prng_key = flow_map.initialize_diamond_map(
            cfg.network, cfg.problem.interp_type, ex_input, prng_key
        )
        init_from = cfg.network.init_from

        if (
            cfg.network.network_type == "sit"
            and init_from != ""
            and cfg.network.load_path == ""
        ):
            print(
                f"Initializing diamond {cfg.network.network_type} from flow-map checkpoint: {init_from}"
            )
            flow_map_ckpt_params = load_params_from_checkpoint(
                init_from,
                ema_factor=cfg.logging.ema_factor,
            )

            params, reused, total = _copy_matching_values(
                flow_map_ckpt_params, params
            )
            print(
                f"Copied {reused}/{total} parameter leaves from flow-map checkpoint."
            )
        return net, params, prng_key

    return flow_map.initialize_flow_map(cfg.network, ex_input, prng_key)


def _load_supervisor_model(
    cfg: config_dict.ConfigDict,
) -> Tuple[Optional[nn.Module], Optional[Parameters]]:
    if "sup_network" not in cfg:
        return None, None

    assert cfg.sup_network.load_path != "", "No checkpoint found for supervising network"

    sup_net = _build_top_level_model(cfg.sup_network, cfg.problem.interp_type)
    sup_params = load_params_from_checkpoint(
        cfg.sup_network.load_path,
        ema_factor=cfg.logging.ema_factor,
    )
    return sup_net, sup_params


def setup_training_state(
    cfg: config_dict.ConfigDict,
    ex_input: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> Tuple[
    EMATrainState,
    nn.Module,
    optax.Schedule,
    jnp.ndarray,
    nn.Module,
    flow.Parameters,
]:
    """Load flax training state."""
    tx, schedule = setup_optimizer(cfg)

    net, params, prng_key = _initialize_main_model(cfg, ex_input, prng_key)

    ema_params = {ema_fac: deepcopy(params) for ema_fac in cfg.training.ema_facs}

    # define training state
    train_state = EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params=ema_params,
        tx=tx,
    )

    # load pretrained supervising network
    sup_net, sup_params = _load_supervisor_model(cfg)

    # load training state from checkpoint, if desired
    if cfg.network.load_path != "":
        print("Loading full training state checkpoint.")
        train_state = load_checkpoint(cfg.network.load_path, train_state)
        print("Loaded training state checkpoint.")

        if cfg.network.reset_optimizer:
            print("Resetting optimizer state.")
            train_state = train_state.replace(
                opt_state=tx.init(train_state.params),
                step=0,
            )

    return train_state, net, schedule, prng_key, sup_net, sup_params
