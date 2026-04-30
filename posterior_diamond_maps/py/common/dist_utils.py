"""JAX sharding helpers for single-node data parallelism."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ml_collections import config_dict

# Global mesh state
MESH = None


# TODO: Pass in MESH instead of using global state 
def setup_mesh(ndevices: int):
    """Initialize the global device mesh."""
    global MESH
    devices = jax.devices()
    if len(devices) < ndevices:
        raise ValueError(f"Requested {ndevices} devices but only found {len(devices)}")
    # Use all available devices up to ndevices
    mesh_devices = devices[:ndevices]
    MESH = Mesh(mesh_devices, axis_names=("data",))


def safe_replicate(cfg: config_dict.ConfigDict, x: Any) -> Any:
    """
    Replicate data across devices using NamedSharding.
    Unlike pmap, this does not add a leading dimension.
    """
    if cfg.training.ndevices <= 1:
        return x

    # Replicate on all devices (empty PartitionSpec means full replication)
    return jax.device_put(x, NamedSharding(MESH, P()))


def replicate_batch(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
    """
    Shard batch across local devices for data parallelism.
    Splits the first dimension (batch) across the 'data' axis of the mesh.
    """
    if cfg.training.ndevices <= 1 or x is None:
        return x

    # Shard along the first dimension (batch dim)
    return jax.device_put(x, NamedSharding(MESH, P("data")))


def constrain_batch_sharding(cfg: config_dict.ConfigDict, x: Any) -> Any:
    """
    Constrain an in-jit value to stay sharded along the leading batch dimension.

    This is needed when a larger jitted wrapper builds sharded intermediates
    internally but would otherwise return them with replicated output layout.
    """
    if cfg.training.ndevices <= 1 or x is None:
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(MESH, P("data")))


# def unreplicate_batch(cfg: config_dict.ConfigDict, x: Any) -> jnp.ndarray:
#     """
#     Merge batch from local devices.
#     With sharding, this is an identity operation.
#     """
#     return x


def replicate_loss_fn_args(cfg: config_dict.ConfigDict, loss_fn_args: Tuple) -> Tuple:
    """Replicate all loss function arguments for data parallelism."""
    return tuple(replicate_batch(cfg, arg) for arg in loss_fn_args)


# def unreplicate_loss_fn_args(cfg: config_dict.ConfigDict, loss_fn_args: Tuple) -> Tuple:
#     """Unreplicate all loss function arguments."""
#     return tuple(unreplicate_batch(cfg, arg) for arg in loss_fn_args)
