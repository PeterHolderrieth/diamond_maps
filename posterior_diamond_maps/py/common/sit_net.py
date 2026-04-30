"""
SiT network adapters for flow, flow-map, and diamond-map matching.
"""

from dataclasses import field
from typing import Dict, Optional

import jax.numpy as jnp
from flax import linen as nn
from jax.typing import DTypeLike

from . import sit_model

Parameters = Dict[str, Dict]


class PrecondSiT(nn.Module):
    img_resolution: int
    img_channels: int
    label_dim: int
    compute_dtype: DTypeLike
    param_dtype: DTypeLike
    sigma_data: float = 1.0
    matching: str = "flow"
    sit_model_name: str = "SiT_XL_2"
    sit_kwargs: dict = field(default_factory=dict)
    cfg_scale: float = 4.0
    cfg_channels: int = 3
    reverse_time: bool = False

    def setup(self):
        if self.sit_model_name not in sit_model.SiT_MODELS:
            raise ValueError(f"Unknown SiT model {self.sit_model_name}")
        if self.matching not in {"flow", "flow_map", "diamond_map"}:
            raise ValueError(f"Unknown SiT matching mode {self.matching}")

        model_cls = sit_model.SiT_MODELS[self.sit_model_name]
        self.is_flow = self.matching == "flow"
        self.is_diamond = self.matching == "diamond_map"
        self.net = model_cls(
            input_size=self.img_resolution,
            in_channels=self.img_channels,
            num_classes=self.label_dim,
            matching=self.matching,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            **self.sit_kwargs,
        )

    def calc_weight(
        self,
        t: jnp.ndarray,
        t_prime: Optional[jnp.ndarray] = None,
        s: Optional[jnp.ndarray] = None,
        s_prime: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        del t_prime
        del s
        del s_prime
        return jnp.zeros_like(t, dtype=jnp.float32).reshape(-1, 1, 1, 1)

    def _process_input(self, t: jnp.ndarray, t_prime: jnp.ndarray, x: jnp.ndarray):
        t = jnp.asarray(t, dtype=jnp.float32).reshape(-1)
        t_prime = jnp.asarray(t_prime, dtype=jnp.float32).reshape(-1)
        x = x.astype(jnp.float32)
        x = jnp.transpose(x, (0, 2, 3, 1))
        return t, t_prime, x

    def _process_label(self, class_labels: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
        if self.label_dim == 0 or class_labels is None:
            return None
        return class_labels.astype(jnp.int32).reshape(-1)

    def _forward_flow(
        self,
        x_in: jnp.ndarray,
        t_model: jnp.ndarray,
        class_labels: Optional[jnp.ndarray],
        train: bool,
    ) -> jnp.ndarray:
        if self.cfg_scale == 1.0 or class_labels is None or self.label_dim == 0:
            return self.net(
                x_in,
                t_model,
                class_labels,
                train=train,
            ).astype(jnp.float32)

        uncond_labels = jnp.full_like(class_labels, self.label_dim)
        cond = self.net(x_in, t_model, class_labels, train=train).astype(jnp.float32)
        uncond = self.net(x_in, t_model, uncond_labels, train=train).astype(jnp.float32)

        n_cfg_channels = min(self.cfg_channels, cond.shape[-1])
        guided_head = uncond[..., :n_cfg_channels] + self.cfg_scale * (
            cond[..., :n_cfg_channels] - uncond[..., :n_cfg_channels]
        )
        guided_tail = cond[..., n_cfg_channels:]
        return jnp.concatenate([guided_head, guided_tail], axis=-1)

    def calc_phi(
        self,
        t: jnp.ndarray,
        t_prime: jnp.ndarray,
        x: jnp.ndarray,
        class_labels: jnp.ndarray = None,
        train: bool = False,
        init_weights: bool = False,
        s: Optional[jnp.ndarray] = None,
        s_prime: Optional[jnp.ndarray] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        del init_weights

        if self.is_flow:
            raise NotImplementedError("Flow-mode SiT only supports calc_b.")

        t, t_prime, x = self._process_input(t, t_prime, x)
        class_labels = self._process_label(class_labels)
        s_model = s_prime_model = x_t_in = None
        if self.is_diamond:
            if s is None or s_prime is None or x_t is None:
                raise ValueError("Diamond SiT requires s, s_prime, and x_t.")
            s_model = jnp.asarray(s, dtype=jnp.float32).reshape(-1)
            s_prime_model = jnp.asarray(s_prime, dtype=jnp.float32).reshape(-1)
            if self.reverse_time:
                s_model, s_prime_model = 1.0 - s_prime_model, 1.0 - s_model
            x_t_in = x_t.astype(jnp.float32)
            x_t_in = jnp.transpose(x_t_in, (0, 2, 3, 1))

        if self.reverse_time:
            t_model = 1.0 - t_prime
            t_prime_model = 1.0 - t
            phi_scale = -1.0
        else:
            t_model = t
            t_prime_model = t_prime
            phi_scale = 1.0

        c_out = self.sigma_data
        c_in = 1.0 / self.sigma_data
        # Do the wrapper-side preconditioning in fp32, then cast down for the model forward.
        x_in = (c_in * x).astype(self.compute_dtype)
        if x_t_in is not None:
            x_t_in = (c_in * x_t_in).astype(self.compute_dtype)

        # Cast back at the wrapper boundary so losses and solver math stay in fp32.
        phi = c_out * self.net(
            x_in,
            t_prime_model,
            class_labels,
            train=train,
            t=t_model,
            s=s_model,
            s_prime=s_prime_model,
            x_t=x_t_in,
        ).astype(jnp.float32)
        phi = jnp.transpose(phi, (0, 3, 1, 2))
        phi = phi_scale * phi

        return phi

    def calc_b(
        self,
        s: float,
        x: jnp.ndarray,
        class_labels: jnp.ndarray = None,
        train: bool = False,
        t: Optional[jnp.ndarray] = None,
        t_prime: Optional[jnp.ndarray] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if self.is_flow:
            t_model, _, x = self._process_input(s, s, x)
            class_labels = self._process_label(class_labels)

            c_out = self.sigma_data
            c_in = 1.0 / self.sigma_data
            # Do the wrapper-side preconditioning in fp32, then cast down for the model forward.
            x_in = (c_in * x).astype(self.compute_dtype)

            b_t = c_out * self._forward_flow(x_in, t_model, class_labels, train=train)
            b_t = jnp.transpose(b_t, (0, 3, 1, 2))
            return b_t

        if self.is_diamond:
            return self.calc_phi(
                t,
                t_prime,
                x,
                class_labels=class_labels,
                train=train,
                s=s,
                s_prime=s,
                x_t=x_t,
            )

        return self.calc_phi(
            s,
            s,
            x,
            class_labels=class_labels,
            train=train,
        )

    def __call__(
        self,
        t: jnp.ndarray,
        t_prime: jnp.ndarray,
        x: jnp.ndarray,
        class_labels: jnp.ndarray = None,
        train: bool = False,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
        s: Optional[jnp.ndarray] = None,
        s_prime: Optional[jnp.ndarray] = None,
        x_t: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        del init_weights

        if self.is_flow:
            del return_X_and_phi
            del s
            del s_prime
            del x_t
            return self.calc_b(
                s=t_prime,
                x=x,
                class_labels=class_labels,
                train=train,
            )

        phi = self.calc_phi(
            t,
            t_prime,
            x,
            class_labels=class_labels,
            train=train,
            s=s,
            s_prime=s_prime,
            x_t=x_t,
        )

        if self.is_diamond:
            dt = (
                jnp.asarray(s_prime, dtype=jnp.float32) - jnp.asarray(s, dtype=jnp.float32)
            ).reshape(-1, 1, 1, 1)
        else:
            dt = (jnp.asarray(t_prime, dtype=jnp.float32) - jnp.asarray(t, dtype=jnp.float32)).reshape(
                -1,
                1,
                1,
                1,
            )
        x_out = x + dt * phi

        if return_X_and_phi:
            return x_out, phi
        return x_out
