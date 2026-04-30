"""Flow-map and diamond-map Flax wrappers for matching objectives."""

from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from ml_collections import config_dict

from . import edm2_net, network_utils, stoch

Parameters = Dict[str, Dict]


class DiamondMap(stoch.Stoch):
    """Basic class for a diamond map."""

    def setup(self):
        """Set up the diamond map."""
        self.diamond_map = network_utils.setup_network(self.config)

    def __call__(
        self,
        t: float,
        t_prime: float,
        s: float,
        s_prime: float,
        x_t: jnp.ndarray,
        x_s: jnp.ndarray,
        label: float = None,
        train: bool = True,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
    ) -> jnp.ndarray:
        """Apply the diamond map."""
        return self.diamond_map(
            t,
            t_prime,
            x_s,
            label,
            train,
            return_X_and_phi,
            init_weights,
            s=s,
            s_prime=s_prime,
            x_t=x_t,
        )

    def partial_s_prime(
        self,
        t: float,
        t_prime: float,
        s: float,
        s_prime: float,
        x_t: jnp.ndarray,
        x_s: jnp.ndarray,
        label: float = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """Compute the partial derivative with respect to the end of the inner map."""
        x_s_prime, ds_prime_x_s_prime = jax.jvp(
            lambda s_prime: self.diamond_map(
                t,
                t_prime,
                x_s,
                label,
                train,
                s=s,
                s_prime=s_prime,
                x_t=x_t,
            ),
            primals=(s_prime,),
            tangents=(jnp.ones_like(s_prime),),
        )

        return x_s_prime, ds_prime_x_s_prime

    def calc_weight(self, t: float, t_prime: float, s: float, s_prime: float) -> jnp.ndarray:
        """Compute the weights for the diamond map."""
        return self.diamond_map.calc_weight(t, t_prime, s, s_prime)

    def calc_phi(
        self,
        t: float,
        t_prime: float,
        s: float,
        s_prime: float,
        x_t: jnp.ndarray,
        x_s: jnp.ndarray,
        label: float = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """Compute the velocity from s to s_prime."""
        return self.diamond_map.calc_phi(
            t,
            t_prime,
            x_s,
            label=label,
            train=train,
            s=s,
            s_prime=s_prime,
            x_t=x_t,
        )

    def calc_b(
        self,
        t: float,
        t_prime: float,
        s: float,
        x_t: jnp.ndarray,
        x_s: jnp.ndarray,
        label: Optional[float] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """Compute the instantaneous velocity at s."""
        return self.diamond_map.calc_b(
            s,
            x_s,
            label=label,
            train=train,
            t=t,
            t_prime=t_prime,
            x_t=x_t,
        )


class FlowMap(nn.Module):
    """Basic class for a flow map."""

    config: config_dict.ConfigDict

    def setup(self):
        """Set up the flow map."""
        self.flow_map = network_utils.setup_network(self.config)

    def __call__(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
    ) -> jnp.ndarray:
        """Apply the flow map."""
        return self.flow_map(t, t_prime, x, label, train, return_X_and_phi, init_weights)

    def partial_t_prime(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """Compute the partial derivative with respect to t_prime."""
        x_t_prime, dt_prime_x_t_prime = jax.jvp(
            lambda t_prime: self.flow_map(t, t_prime, x, label, train),
            primals=(t_prime,),
            tangents=(jnp.ones_like(t_prime),),
        )

        return x_t_prime, dt_prime_x_t_prime

    def calc_weight(self, t: float, t_prime: float) -> jnp.ndarray:
        """Compute the weights for the flow map."""
        return self.flow_map.calc_weight(t, t_prime, None, None)

    def calc_phi(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """Compute the flow map."""
        return self.flow_map.calc_phi(t, t_prime, x, label=label, train=train)

    def calc_b(
        self,
        t_prime: float,
        x: jnp.ndarray,
        label: Optional[float] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """Apply the flow map."""
        return self.flow_map.calc_b(t_prime, x, label=label, train=train)

def initialize_flow_map(
    network_config: config_dict.ConfigDict, ex_input: jnp.ndarray, prng_key: jnp.ndarray
) -> Tuple[nn.Module, Parameters, jnp.ndarray]:
    # define the network
    net = FlowMap(network_config)

    # initialize the parameters
    ex_t = ex_t_prime = 0.0
    ex_label = 0
    prng_key, t_key = jax.random.split(prng_key)

    params = net.init(
        {"params": prng_key, "constants": t_key},
        ex_t,
        ex_t_prime,
        ex_input,
        ex_label,
        train=False,
        init_weights=True,  # This triggers initialization of all weight params
    )

    prng_key = jax.random.split(prng_key)[0]

    print(f"Number of parameters: {ravel_pytree(params)[0].size}")

    if network_config.network_type == "edm2":
        params = edm2_net.project_to_sphere(params)

    return net, params, prng_key

def initialize_diamond_map(
    network_config: config_dict.ConfigDict,
    interp_type: str,
    ex_input: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> Tuple[nn.Module, Parameters, jnp.ndarray]:
    # define the network
    net = DiamondMap(config=network_config, interp_type=interp_type)

    # initialize the parameters
    ex_t = ex_t_prime = ex_s = ex_s_prime = 0.0
    ex_label = 0
    prng_key, t_key = jax.random.split(prng_key)

    params = net.init(
        {"params": prng_key, "constants": t_key},
        ex_t,
        ex_t_prime,
        ex_s,
        ex_s_prime,
        ex_input,
        ex_input,
        label=ex_label,
        train=False,
        init_weights=True,  # This triggers initialization of all weight params
    )

    prng_key = jax.random.split(prng_key)[0]

    print(f"Number of parameters: {ravel_pytree(params)[0].size}")

    if network_config.network_type == "edm2":
        params = edm2_net.project_to_sphere(params)

    return net, params, prng_key
