"""Stochastic interpolant schedules and GLASS inner-flow utilities."""

import dataclasses
import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp


def clip_zero(x: float):
    return jnp.maximum(1e-8, x)


@dataclasses.dataclass
class Interpolant:
    """Basic class for a stochastic interpolant"""

    sigma: Callable[[float], float]
    alpha: Callable[[float], float]
    sigma_dot: Callable[[float], float]
    alpha_dot: Callable[[float], float]
    g: Callable[[float], float]
    g_inv: Optional[Callable[[float], float]] = None

    def calc_It(self, t_prime: float, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
        return self.sigma(t_prime) * x0 + self.alpha(t_prime) * x1

    def calc_It_dot(self, t_prime: float, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
        return self.sigma_dot(t_prime) * x0 + self.alpha_dot(t_prime) * x1

    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def batch_calc_It(
        self, t_prime: jnp.ndarray, x0: jnp.ndarray, x1: jnp.ndarray
    ) -> jnp.ndarray:
        return self.calc_It(t_prime, x0, x1)

    def __hash__(self):
        return hash((self.sigma, self.alpha))

    def __eq__(self, other):
        return self.sigma == other.sigma and self.alpha == other.alpha


@dataclasses.dataclass
class InnerInterpolant:
    """Basic class for inner flow in GLASS"""

    calc_x_t_prime: Callable[
        [jnp.ndarray, jnp.ndarray, float, float, float, jnp.ndarray], jnp.ndarray
    ]
    calc_x_t_prime_renoise: Callable[
        [jnp.ndarray, jnp.ndarray, float, float, float, jnp.ndarray], jnp.ndarray
    ]
    # functions of (t, t_prime, s)
    sigma: Callable[[float, float, float], float]
    alpha: Callable[[float, float, float], float]
    sigma_dot: Callable[[float, float, float], float]
    alpha_dot: Callable[[float, float, float], float]
    # functions of (t, t_prime)
    rho: Callable[[float, float], float]
    gamma: Callable[[float, float], float]
    # functions of (t, t_prime)
    calc_s: Callable[[float, float], float]
    # constants
    sigma_0: float
    rescale: float

    @classmethod
    def from_interpolant(
        cls,
        interp: Interpolant,
        rho: Optional[Callable[[float, float], float]] = None,
        rescale: float = 1.0,
        sigma_0: float = 1.0,
    ) -> "InnerInterpolant":

        # default DDPM rho if not provided
        if rho is None:

            def rho(t: float, t_prime: float) -> float:
                return (
                    interp.alpha(t)
                    * interp.sigma(t_prime)
                    / clip_zero(interp.alpha(t_prime) * interp.sigma(t))
                )

        def gamma(t: float, t_prime: float) -> float:
            return rho(t, t_prime) * interp.sigma(t_prime) / clip_zero(interp.sigma(t))

        def sigma_bar(t: float, t_prime: float, s: float) -> float:
            return (1.0 - s) * jnp.sqrt(sigma_0) + s * jnp.sqrt(
                jnp.maximum(0.0, interp.sigma(t_prime) ** 2 * (1.0 - rho(t, t_prime) ** 2))
            )

        def sigma_bar_dot(t: float, t_prime: float, s: float) -> float:
            return -jnp.sqrt(sigma_0) + jnp.sqrt(
                jnp.maximum(0.0, interp.sigma(t_prime) ** 2 * (1.0 - rho(t, t_prime) ** 2))
            )

        def alpha_bar(t: float, t_prime: float, s: float) -> float:
            return s * (interp.alpha(t_prime) - gamma(t, t_prime) * interp.alpha(t))

        def alpha_bar_dot(t: float, t_prime: float, s: float) -> float:
            return interp.alpha(t_prime) - gamma(t, t_prime) * interp.alpha(t)

        def _calc_s(t: float, t_prime: float) -> float:
            diff = interp.g(t) - interp.g(t_prime)
            fraction = interp.g(t_prime) * interp.g(t) / diff
            s = interp.g_inv(fraction)
            return s

        def calc_s(t: float, t_prime: float) -> float:
            return jax.lax.cond(t_prime == 1.0, lambda: 1.0, lambda: _calc_s(t, t_prime))

        # DO NOT USE, use calc_suff_stat instead
        def calc_x_t_prime(
            x_s: jnp.ndarray,
            x_t: jnp.ndarray,
            t: float,
            t_prime: float,
            s: float,
            key: jnp.ndarray,
        ) -> jnp.ndarray:
            del key
            return (
                interp.alpha(t_prime)
                * (
                    x_s * interp.alpha(s) * interp.sigma(t) ** 2
                    + x_t * interp.alpha(t) * interp.sigma(s) ** 2
                )
                / (
                    interp.alpha(t) ** 2 * interp.sigma(s) ** 2
                    + interp.alpha(s) ** 2 * interp.sigma(t) ** 2
                )
            )

        def calc_x_t_prime_renoise(
            x1: jnp.ndarray,
            x_t: jnp.ndarray,
            t: float,
            t_prime: float,
            s: float,
            key: jnp.ndarray,
        ) -> jnp.ndarray:
            del x_t, t, s
            return interp.calc_It(t_prime, jax.random.normal(key, x1.shape), x1)

        return cls(
            sigma=sigma_bar,
            alpha=alpha_bar,
            sigma_dot=sigma_bar_dot,
            alpha_dot=alpha_bar_dot,
            rho=rho,
            gamma=gamma,
            rescale=rescale,
            sigma_0=sigma_0,
            calc_s=calc_s,
            calc_x_t_prime=calc_x_t_prime,
            calc_x_t_prime_renoise=calc_x_t_prime_renoise,
        )

    def calc_xbar_s(
        self,
        prng_key: jnp.ndarray,
        t: float,
        t_prime: float,
        s: float,
        x_t: jnp.ndarray,
        x1: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return (
            (jnp.zeros_like(x_t) if x1 is None else self.alpha(t, t_prime, s) * x1)
            + self.gamma(t, t_prime) * x_t
            + self.sigma(t, t_prime, s)
            * self.rescale
            * jax.random.normal(prng_key, x_t.shape)
        )


def setup_interpolant(interp_type: str) -> Interpolant:
    g_inv = None

    if interp_type == "linear":
        sigma = lambda t_prime: 1.0 - t_prime
        alpha = lambda t_prime: t_prime
        sigma_dot = lambda _: -1.0
        alpha_dot = lambda _: 1.0
        g_inv = lambda t_prime: 1.0 / (1.0 + jnp.sqrt(jnp.maximum(0.0, t_prime)))
    elif interp_type == "trig":
        sigma = lambda t_prime: jnp.cos(jnp.pi * t_prime / 2)
        alpha = lambda t_prime: jnp.sin(jnp.pi * t_prime / 2)
        sigma_dot = lambda t_prime: -0.5 * jnp.pi * jnp.sin(jnp.pi * t_prime / 2)
        alpha_dot = lambda t_prime: 0.5 * jnp.pi * jnp.cos(jnp.pi * t_prime / 2)
    else:
        raise NotImplementedError("Interpolant type not implemented.")

    g = lambda t_prime: (sigma(t_prime) / clip_zero(alpha(t_prime))) ** 2

    interp = Interpolant(
        sigma=sigma, alpha=alpha, sigma_dot=sigma_dot, alpha_dot=alpha_dot, g=g, g_inv=g_inv
    )

    return interp
