"""Transformer layer components for JAX/Flax SiT models."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.typing import DTypeLike


class PatchEmbed(nn.Module):
    input_size: int
    initial_patch_size: int
    in_channels: int
    hidden_size: int
    dtype: DTypeLike
    param_dtype: DTypeLike
    bias: bool = True

    def setup(self):
        self.patch_size = (self.initial_patch_size, self.initial_patch_size)
        self.img_size = self.input_size
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(
            self.img_size
        )
        self.proj = nn.Conv(
            self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=self.bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(in_axis=(0, 1, 2), out_axis=-1),
            bias_init=nn.initializers.zeros,
        )

    def _init_img_size(self, img_size: int):
        img_size = (img_size, img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def __call__(self, x):
        batch_size, height, width, channels = x.shape
        assert (
            channels == self.in_channels
        ), f"Expected {self.in_channels} channels but got {channels}"
        assert (
            height,
            width,
        ) == self.img_size, (
            f"Input size mismatch. Got {(height, width)} but expected {self.img_size}"
        )

        x = self.proj(x)
        x = x.reshape(batch_size, -1, x.shape[3])
        return x


class Attention(nn.Module):
    dim: int
    dtype: DTypeLike
    param_dtype: DTypeLike
    num_heads: int = 8
    qkv_bias: bool = True
    norm_layer: nn.Module = None
    linear_layer: nn.Module = None

    def setup(self):
        if self.linear_layer is None:
            raise ValueError("linear_layer must be provided to Attention")
        if self.norm_layer is None:
            raise ValueError("norm_layer must be provided to Attention")

        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = self.linear_layer(
            self.dim,
            self.dim * 3,
            bias=self.qkv_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.q_norm = lambda x: x
        self.k_norm = lambda x: x
        self.attn_drop = lambda x: x
        self.proj = self.linear_layer(
            self.dim,
            self.dim,
            bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.proj_drop = lambda x: x

    def __call__(self, x):
        batch_size, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = jnp.split(qkv, 3, axis=0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * jnp.asarray(self.scale, dtype=q.dtype)
        attn = q @ k.transpose(0, 1, 2, 4, 3)
        # Softmax is more numerically sensitive than the surrounding attention matmuls.
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(v.dtype)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x[0].transpose(0, 2, 1, 3).reshape(batch_size, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    in_features: int
    hidden_features: int
    dtype: DTypeLike
    param_dtype: DTypeLike
    act_layer: nn.Module = None
    drop: float = 0.0
    linear_layer: nn.Module = None

    def setup(self):
        if self.linear_layer is None:
            raise ValueError("linear_layer must be provided to Mlp")
        if self.act_layer is None:
            raise ValueError("act_layer must be provided to Mlp")

        out_features = self.in_features
        hidden_features = self.hidden_features or self.in_features
        bias = (True, True)
        assert self.drop < 1e-3, "Dropout in Mlp is not implemented."

        self.fc1 = self.linear_layer(
            self.in_features,
            hidden_features,
            bias=bias[0],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.act = self.act_layer()
        self.drop1 = lambda x: x
        self.norm = lambda x: x
        self.fc2 = self.linear_layer(
            hidden_features,
            out_features,
            bias=bias[1],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.drop2 = lambda x: x

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
