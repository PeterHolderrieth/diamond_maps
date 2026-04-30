"""Torch-compatible layer primitives for converted checkpoints."""

from math import sqrt
import jax.random as jr
from flax import linen as nn
from jax.typing import DTypeLike


class TorchLinear(nn.Module):
    in_features: int
    out_features: int
    dtype: DTypeLike
    param_dtype: DTypeLike
    bias: bool = True
    weight_init: str = "torch"
    bias_init: str = "torch"

    def setup(self):
        if self.weight_init == "torch":
            weight_initializer = nn.initializers.variance_scaling(
                scale=1 / 3.0,
                mode="fan_in",
                distribution="uniform",
            )
        elif self.weight_init == "xavier_uniform":
            weight_initializer = nn.initializers.xavier_uniform()
        elif self.weight_init == "0.02":
            weight_initializer = lambda key, shape, dtype: jr.normal(key, shape) * 0.02
        elif self.weight_init == "zeros":
            weight_initializer = nn.initializers.zeros
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        if self.bias_init == "torch":
            bias_initializer = lambda key, shape, dtype: jr.uniform(
                key,
                shape,
                minval=-sqrt(1 / self.in_features),
                maxval=sqrt(1 / self.in_features),
            )
        elif self.bias_init == "zeros":
            bias_initializer = nn.initializers.zeros
        else:
            raise ValueError(f"Invalid bias_init: {self.bias_init}")

        self._flax_linear = nn.Dense(
            features=self.out_features,
            use_bias=self.bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=weight_initializer,
            bias_init=bias_initializer,
        )

    def __call__(self, x):
        return self._flax_linear(x)


class TorchEmbedding(nn.Module):
    num_embeddings: int
    embedding_dim: int
    dtype: DTypeLike
    param_dtype: DTypeLike

    def setup(self):
        self._flax_embedding = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.embedding_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=lambda key, shape, dtype: jr.normal(key, shape) * 0.02,
        )

    def __call__(self, x):
        return self._flax_embedding(x)


class TorchLayerNorm(nn.Module):
    hidden_size: int
    dtype: DTypeLike
    param_dtype: DTypeLike
    elementwise_affine: bool = False
    eps: float = 1e-6

    def setup(self):
        assert not self.elementwise_affine, NotImplementedError()
        self._flax_layernorm = nn.LayerNorm(
            epsilon=self.eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=self.elementwise_affine,
            use_scale=self.elementwise_affine,
        )

    def __call__(self, x):
        y = self._flax_layernorm(x)
        # Keep norm reductions in fp32, but convert back to the caller's stream dtype.
        y = y.astype(x.dtype)
        return y
