"""JAX training-step construction with optimizer and EMA updates."""

import functools
from typing import Callable, Dict, Tuple

import jax
import ml_collections.config_dict as config_dict
import optax
from jax import value_and_grad

from . import state_utils, edm2_net

Parameters = Dict[str, Dict]


def setup_train_step(cfg: config_dict.ConfigDict) -> Callable:
    """Setup the training step function for single or multi-device training."""

    decorator = functools.partial(jax.jit, static_argnums=(1,))

    @decorator
    def train_step(
        state: state_utils.EMATrainState,
        loss_func: Callable[[Parameters], float],
        loss_func_args=tuple(),
    ) -> Tuple[state_utils.EMATrainState, float, float]:
        """Single training step for the neural network.

        Args:
            state: Training state.
            loss_func: Loss function for the parameters.
            loss_func_args: Argument other than the parameters for the loss function.
        """
        loss_value, grads = value_and_grad(loss_func)(state.params, *loss_func_args)
        grad_norm = optax.global_norm(grads)

        state = state.apply_gradients(grads=grads)

        # project for the edm2 network
        state = state.replace(params=edm2_net.safe_project_to_sphere(cfg, state.params))

        new_ema_params = {}
        for ema_fac, ema_params in state.ema_params.items():
            new_ema_params[ema_fac] = jax.tree_util.tree_map(
                lambda param, ema_param: ema_fac * ema_param + (1 - ema_fac) * param,
                state.params,
                ema_params,
            )
        state = state.replace(ema_params=new_ema_params)

        return state, loss_value, grad_norm

    return train_step
