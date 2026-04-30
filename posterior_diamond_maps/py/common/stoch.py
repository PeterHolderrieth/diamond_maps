"""Shared stochastic interpolant base class for flow modules."""

from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from ml_collections import config_dict

from . import interpolant


class Stoch(nn.Module):
    """Shared stochastic interpolant helpers for GLASS-like models."""

    config: config_dict.ConfigDict
    interp_type: str

    @nn.nowrap
    def out_interp(self) -> interpolant.Interpolant:
        return interpolant.setup_interpolant(self.interp_type)

    @nn.nowrap
    def in_interp(self) -> interpolant.InnerInterpolant:
        return interpolant.InnerInterpolant.from_interpolant(
            self.out_interp(),
            rescale=self.config.rescale,
            sigma_0=1.0,
        )

    @nn.nowrap
    def _stable_inv(self, x: jnp.ndarray):
        """Stable inverse for covariance matrix."""
        return jnp.linalg.inv(x + 1e-4 * jnp.eye(x.shape[0], dtype=x.dtype))

    @nn.nowrap
    def _mu_cov(self, t: float, t_prime: float, s: float):
        out_interp = self.out_interp()
        in_interp = self.in_interp()
        gamma = in_interp.gamma(t, t_prime)
        mu = jnp.array(
            [
                out_interp.alpha(t),
                in_interp.alpha(t, t_prime, s) + gamma * out_interp.alpha(t),
            ]
        )
        cross_term = out_interp.sigma(t) ** 2 * gamma
        cov = jnp.array(
            [
                [out_interp.sigma(t) ** 2, cross_term],
                [
                    cross_term,
                    in_interp.sigma(t, t_prime, s) ** 2
                    + gamma**2 * out_interp.sigma(t) ** 2,
                ],
            ],
        )
        return mu, cov

    @nn.nowrap
    def calc_s(self, t: float, t_prime: float) -> float:
        return self.in_interp().calc_s(t, t_prime)

    @nn.nowrap
    def calc_xbar_s(
        self,
        prng_key: jnp.ndarray,
        t: float,
        t_prime: float,
        s: float,
        x_t: jnp.ndarray,
        x1: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self.in_interp().calc_xbar_s(prng_key, t, t_prime, s, x_t, x1)

    @nn.nowrap
    def calc_x_t_prime_renoise(
        self,
        x1: jnp.ndarray,
        x_t: jnp.ndarray,
        t: float,
        t_prime: float,
        s: float,
        prng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.in_interp().calc_x_t_prime_renoise(x1, x_t, t, t_prime, s, prng_key)

    @nn.nowrap
    def calc_suff_stat(
        self, t: float, t_prime: float, s: float, x_t: jnp.ndarray, x_s: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float]:
        mu, cov = self._mu_cov(t, t_prime, s)
        cov_inv = self._stable_inv(cov)
        curr = jnp.array([x_t, x_s])
        denom = interpolant.clip_zero(mu.T @ cov_inv @ mu)
        suff_stat = jnp.einsum("i,ij,j...->...", mu, cov_inv, curr) / denom
        return suff_stat, denom

    @nn.nowrap
    def calc_x_t_prime(
        self,
        t: float,
        t_prime: float,
        s: float,
        x_t: jnp.ndarray,
        x_s: jnp.ndarray,
        target_t_prime: Optional[float] = None,
    ) -> jnp.ndarray:
        suff_stat, _ = self.calc_suff_stat(t, t_prime, s, x_t, x_s)
        return (
            self.out_interp().alpha(t_prime if target_t_prime is None else target_t_prime)
            * suff_stat
        )
