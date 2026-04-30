"""Network factory and Flax modules for different architectures"""

from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import edm2_net as edm2_net
from . import sit_net as sit_net


class MLP(nn.Module):
    """Simple MLP network with square weight pattern."""

    n_hidden: int
    n_neurons: int
    output_dim: int
    act: Callable
    use_residual: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.n_neurons)(x)
        x = self.act(x)

        for _ in range(self.n_hidden):
            if self.use_residual:
                x = x + nn.Dense(self.n_neurons)(x)
            else:
                x = nn.Dense(self.n_neurons)(x)
            x = self.act(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class MapMLP(nn.Module):
    """Generic MLP for Flow and Diamond Maps.

    Handles both:
    1. Flow Map: v_{t, t_prime}(x)
    2. Diamond Map: v_{t, t_prime, s, s_prime}(x | x_t)
    """

    config: config_dict.ConfigDict

    def setup(self):
        self.is_diamond = (
            "matching" in self.config and self.config.matching == "diamond_map"
        )
        self.phi_mlp = MLP(
            n_hidden=self.config.n_hidden,
            n_neurons=self.config.n_neurons,
            output_dim=self.config.output_dim,
            act=get_act(self.config),
            use_residual=self.config.use_residual,
        )
        self.weight_mlp = MLP(
            n_hidden=1,
            n_neurons=self.config.n_neurons,
            output_dim=1,
            act=jax.nn.gelu,
            use_residual=False,
        )

    def _get_conditioning(self, t, t_prime, s, s_prime, x_t):
        cond = [jnp.atleast_1d(t), jnp.atleast_1d(t_prime)]

        if self.is_diamond:
            if s is None or s_prime is None or x_t is None:
                raise ValueError("Diamond Map requires s, s_prime, and x_t.")

            cond.extend(
                [jnp.atleast_1d(s), jnp.atleast_1d(s_prime), x_t / self.config.rescale]
            )

        return jnp.concatenate(cond, axis=-1)

    def calc_phi(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        init_weights: bool = False,
        s: Optional[float] = None,
        s_prime: Optional[float] = None,
        x_t: Optional[float] = None,
    ) -> jnp.ndarray:
        del label
        del train
        del init_weights  # MLP doesn't have dual weights to initialize

        cond = self._get_conditioning(t, t_prime, s, s_prime, x_t)
        inp = jnp.concatenate((cond, x / self.config.rescale), axis=-1)
        return self.config.rescale * self.phi_mlp(inp)

    def calc_weight(self, t, t_prime, s=None, s_prime=None):
        """Placeholder for weight calculation."""
        del t
        del t_prime
        del s
        del s_prime
        return 1.0

    def calc_b(
        self,
        s: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        t: Optional[float] = None,
        t_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if self.is_diamond:
            return self.calc_phi(t, t_prime, x, label, train, s=s, s_prime=s, x_t=x_t)
        else:
            return self.calc_phi(s, s, x, label, train)

    def __call__(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
        s: Optional[float] = None,
        s_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        del label
        phi = self.calc_phi(
            t,
            t_prime,
            x,
            label=None,
            train=train,
            init_weights=init_weights,
            s=s,
            s_prime=s_prime,
            x_t=x_t,
        )

        dt = (s_prime - s) if self.is_diamond else (t_prime - t)
        x_out = x + dt * phi

        if return_X_and_phi:
            return x_out, phi
        else:
            return x_out


class EDM2FlowMap(nn.Module):
    """UNet architecture based on EDM2.
    Note: assumes that there is no batch dimension, to interface with the rest of the code.
    Adds a padded batch dimension to handle this.
    """

    config: config_dict.ConfigDict

    def setup(self):
        self.is_diamond = self.config.matching == "diamond_map"

        use_cfg_token = self.config.use_cfg_token
        self.one_hot_dim = (
            self.config.label_dim + 1 if use_cfg_token else self.config.label_dim
        )
        self.net = edm2_net.PrecondFlowMap(
            img_resolution=self.config.img_resolution,
            img_channels=self.config.img_channels,
            label_dim=self.one_hot_dim,
            sigma_data=self.config.rescale,
            logvar_channels=self.config.logvar_channels,
            use_bfloat16=self.config.use_bfloat16,
            use_weight=self.config.use_weight,
            is_diamond=self.is_diamond,
            unet_kwargs=self.config.unet_kwargs,
        )

    def process_inputs(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: Optional[float] = None,
        s: Optional[float] = None,
        s_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ):
        # add batch dimensions
        t = jnp.asarray(t, dtype=jnp.float32)
        t_prime = jnp.asarray(t_prime, dtype=jnp.float32)
        x = x.reshape((1, *x.shape))

        # one-hot encode
        if label is not None:
            label = jax.nn.one_hot(label, num_classes=self.one_hot_dim).reshape((1, -1))

        if self.is_diamond:
            if s is None or s_prime is None or x_t is None:
                raise ValueError("Diamond Map requires s, s_prime, and x_t.")

            s = jnp.asarray(s, dtype=jnp.float32)
            s_prime = jnp.asarray(s_prime, dtype=jnp.float32)
            x_t = x_t.reshape((1, *x_t.shape))
        return t, t_prime, x, label, s, s_prime, x_t

    def calc_weight(
        self, t: float, t_prime: float, s: Optional[float], s_prime: Optional[float]
    ) -> jnp.ndarray:
        # add batch dimension
        t = jnp.asarray(t, dtype=jnp.float32)
        t_prime = jnp.asarray(t_prime, dtype=jnp.float32)
        if self.is_diamond:
            s = jnp.asarray(s, dtype=jnp.float32)
            s_prime = jnp.asarray(s_prime, dtype=jnp.float32)
        return self.net.calc_weight(t, t_prime, s, s_prime)

    def calc_phi(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        init_weights: bool = False,
        s: Optional[float] = None,
        s_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        t, t_prime, x, label, s, s_prime, x_t = self.process_inputs(
            t, t_prime, x, label, s, s_prime, x_t
        )

        rslt = self.net.calc_phi(
            t,
            t_prime,
            x,
            label,
            train,
            init_weights,
            s=s,
            s_prime=s_prime,
            x_t=x_t,
        )

        return rslt[0]

    def calc_b(
        self,
        s: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        t: Optional[float] = None,
        t_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if self.is_diamond:
            return self.calc_phi(t, t_prime, x, label, train, s=s, s_prime=s, x_t=x_t)
        else:
            return self.calc_phi(s, s, x, label, train)

    def __call__(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
        s: Optional[float] = None,
        s_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ):
        t, t_prime, x, label, s, s_prime, x_t = self.process_inputs(
            t, t_prime, x, label, s, s_prime, x_t
        )

        rslt = self.net(
            t,
            t_prime,
            x,
            label,
            train,
            return_X_and_phi,
            init_weights,
            s=s,
            s_prime=s_prime,
            x_t=x_t,
        )

        if return_X_and_phi:
            x_out, phi = rslt
            return x_out[0], phi[0]
        else:
            x_out = rslt
            return x_out[0]


class SiTFlow(nn.Module):
    """Unified SiT architecture for flow, flow-map, and diamond-map matching."""

    config: config_dict.ConfigDict

    def setup(self):
        self.matching = self.config.matching
        self.is_diamond = self.matching == "diamond_map"
        self.is_flow = self.matching == "flow"

        sit_cfg_scale = self.config.sit_cfg_scale
        sit_cfg_channels = self.config.sit_cfg_channels
        sit_model_name = self.config.sit_model_name
        sit_kwargs = self.config.sit_kwargs.to_dict()
        use_cfg_token = bool(self.config.use_cfg_token)
        sit_kwargs["use_cfg_token"] = use_cfg_token
        compute_dtype = jnp.dtype(self.config.compute_dtype).type
        param_dtype = jnp.dtype(self.config.param_dtype).type

        self.net = sit_net.PrecondSiT(
            img_resolution=self.config.img_resolution,
            img_channels=self.config.img_channels,
            label_dim=self.config.label_dim,
            compute_dtype=compute_dtype,
            param_dtype=param_dtype,
            sigma_data=self.config.rescale,
            matching=self.matching,
            cfg_scale=sit_cfg_scale,
            cfg_channels=sit_cfg_channels,
            reverse_time=self.config.reverse_time,
            sit_model_name=sit_model_name,
            sit_kwargs=sit_kwargs,
        )

    def process_inputs(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: Optional[float] = None,
        s: Optional[float] = None,
        s_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ):
        t = jnp.asarray(t, dtype=jnp.float32)
        t_prime = jnp.asarray(t_prime, dtype=jnp.float32)
        x = x.reshape((1, *x.shape))
        if label is not None:
            label = jnp.asarray(label, dtype=jnp.int32).reshape((1,))
        if self.is_diamond:
            if s is None or s_prime is None or x_t is None:
                raise ValueError("Diamond SiT requires s, s_prime, and x_t.")
            s = jnp.asarray(s, dtype=jnp.float32).reshape((1,))
            s_prime = jnp.asarray(s_prime, dtype=jnp.float32).reshape((1,))
            x_t = x_t.reshape((1, *x_t.shape))
        return t, t_prime, x, label, s, s_prime, x_t

    def calc_weight(
        self, t: float, t_prime: float, s: Optional[float], s_prime: Optional[float]
    ) -> jnp.ndarray:
        t_prime = jnp.asarray(t_prime, dtype=jnp.float32)
        if self.is_flow:
            return self.net.calc_weight(t_prime.reshape(1))
        else:
            t = jnp.asarray(t, dtype=jnp.float32)
            if s is not None:
                s = jnp.asarray(s, dtype=jnp.float32)
            if s_prime is not None:
                s_prime = jnp.asarray(s_prime, dtype=jnp.float32)
            return self.net.calc_weight(
                t.reshape(1), t_prime.reshape(1), s=s, s_prime=s_prime
            )

    def calc_phi(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        init_weights: bool = False,
        s: Optional[float] = None,
        s_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        del init_weights
        if self.is_flow:
            raise NotImplementedError("Flow-mode SiT only supports calc_b.")

        t, t_prime, x, label, s, s_prime, x_t = self.process_inputs(
            t, t_prime, x, label, s, s_prime, x_t
        )
        rslt = self.net.calc_phi(
            t,
            t_prime,
            x,
            class_labels=label,
            train=train,
            s=s,
            s_prime=s_prime,
            x_t=x_t,
        )
        return rslt[0]

    def calc_b(
        self,
        s: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        t: Optional[float] = None,
        t_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if self.is_flow:
            s, _, x, label, _, _, _ = self.process_inputs(s, s, x, label)
            rslt = self.net.calc_b(
                s,
                x,
                class_labels=label,
                train=train,
            )
            return rslt[0]

        if self.is_diamond:
            return self.calc_phi(t, t_prime, x, label, train, s=s, s_prime=s, x_t=x_t)

        s, _, x, label, _, _, _ = self.process_inputs(s, s, x, label)
        rslt = self.net.calc_b(
            s,
            x,
            class_labels=label,
            train=train,
        )
        return rslt[0]

    def __call__(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
        s: Optional[float] = None,
        s_prime: Optional[float] = None,
        x_t: Optional[jnp.ndarray] = None,
    ):
        del init_weights
        if self.is_flow:
            del t
            del return_X_and_phi
            del s
            del s_prime
            del x_t
            return self.calc_b(t_prime, x, label=label, train=train)

        t, t_prime, x, label, s, s_prime, x_t = self.process_inputs(
            t, t_prime, x, label, s, s_prime, x_t
        )
        rslt = self.net(
            t,
            t_prime,
            x,
            class_labels=label,
            train=train,
            return_X_and_phi=return_X_and_phi,
            s=s,
            s_prime=s_prime,
            x_t=x_t,
        )
        if return_X_and_phi:
            x_out, phi = rslt
            return x_out[0], phi[0]
        else:
            return rslt[0]


def get_act(
    config: config_dict.ConfigDict,
) -> Callable:
    """Get the activation function for the network.

    Args:
        config: Configuration dictionary.
    """
    if config.act == "gelu":
        return jax.nn.gelu
    elif config.act == "swish" or config.act == "silu":
        return jax.nn.silu
    else:
        raise ValueError(f"Activation function {config.activation} not recognized.")


def setup_network(
    network_config: config_dict.ConfigDict,
) -> nn.Module:
    """Setup the neural network for the system.

    Args:
        config: Configuration dictionary.
    """
    if "mlp" in network_config.network_type:
        return MapMLP(config=network_config)
    elif network_config.network_type == "edm2":
        return EDM2FlowMap(config=network_config)
    elif network_config.network_type == "sit":
        return SiTFlow(config=network_config)
    else:
        raise ValueError(f"Network type {network_config.network_type} not recognized.")
