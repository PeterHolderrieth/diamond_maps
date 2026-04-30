"""GLASS flow wrapper and covariance schedules for inner dynamics."""

from typing import Dict, Optional

import jax.numpy as jnp

from . import network_utils, stoch
from .interpolant import clip_zero

Parameters = Dict[str, Dict]


class GlassFlow(stoch.Stoch):
    """sigma and alpha are from https://arxiv.org/pdf/2505.18825
    where x_t = sigma_t x_0 + alpha_t x_1
    with x_0 noise and x_1 data

    this is slightly different from the GLASS paper
    (t, t_prime) and s denote the outer/inner steps respectively
    """

    def setup(self):
        self.flow_map = network_utils.setup_network(self.config)

    def _denoiser(
        self,
        t_prime: float,
        x: jnp.ndarray,
        label: Optional[float] = None,
        train: bool = False,
    ):
        out_interp = self.out_interp()
        b_t = self.flow_map.calc_b(t_prime, x, label=label, train=train)
        scale = 1.0 / clip_zero(
            out_interp.alpha_dot(t_prime) * out_interp.sigma(t_prime)
            - out_interp.alpha(t_prime) * out_interp.sigma_dot(t_prime)
        )
        return scale * (out_interp.sigma(t_prime) * b_t - out_interp.sigma_dot(t_prime) * x)

    def _glass_denoiser(
        self,
        t: float,
        t_prime: float,
        s: float,
        x_t: jnp.ndarray,
        x_s: jnp.ndarray,
        label: Optional[float] = None,
        train: bool = False,
    ):
        out_interp = self.out_interp()
        suff_stat, denom = self.calc_suff_stat(t, t_prime, s, x_t, x_s)
        reparam_t = out_interp.g_inv(1.0 / denom)
        reparam_x = out_interp.alpha(reparam_t) * suff_stat
        return self._denoiser(reparam_t, reparam_x, label=label, train=train)

    def _calc_base_b(
        self,
        t_prime: float,
        x: jnp.ndarray,
        label: Optional[float] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        return self.flow_map.calc_b(t_prime, x, label=label, train=train)

    def _apply_base_map(
        self,
        t: float,
        t_prime: float,
        x: jnp.ndarray,
        label: Optional[float] = None,
        train: bool = False,
        return_X_and_phi: bool = False,
    ):
        return self.flow_map(
            t,
            t_prime,
            x,
            label=label,
            train=train,
            return_X_and_phi=return_X_and_phi,
        )

    def calc_b(
        self,
        t: float,
        t_prime: float,
        s: float,
        x_t: jnp.ndarray,
        x_s: jnp.ndarray,
        label: Optional[float] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        in_interp = self.in_interp()
        z = self._glass_denoiser(t, t_prime, s, x_t, x_s, label=label, train=train)
        sigma_bar_dot, sigma_bar = in_interp.sigma_dot(t, t_prime, s), in_interp.sigma(
            t, t_prime, s
        )
        alpha_bar_dot, alpha_bar = in_interp.alpha_dot(t, t_prime, s), in_interp.alpha(
            t, t_prime, s
        )
        w_1 = sigma_bar_dot / clip_zero(sigma_bar)
        w_2 = alpha_bar_dot - alpha_bar * w_1
        w_3 = -in_interp.gamma(t, t_prime) * w_1
        return w_1 * x_s + w_2 * z + w_3 * x_t

    def __call__(
        self,
        t: float,
        t_prime: float,
        s: float,
        x_t: jnp.ndarray,
        x_s: jnp.ndarray,
        label: Optional[float] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        return self.calc_b(t, t_prime, s, x_t, x_s, label=label, train=train)
