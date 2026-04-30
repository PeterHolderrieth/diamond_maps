"""SiT model definitions and positional embedding helpers."""

import math
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.typing import DTypeLike

from .transformer_layers import Attention, Mlp, PatchEmbed
from .torch_compat_layers import TorchEmbedding, TorchLayerNorm, TorchLinear

SiTLinear = partial(TorchLinear, weight_init="xavier_uniform", bias_init="zeros")
SiTAttention = partial(Attention, linear_layer=SiTLinear, norm_layer=TorchLayerNorm)
SiTMlp = partial(Mlp, linear_layer=SiTLinear)


def unsqueeze(t, dim):
    return jnp.expand_dims(t, axis=dim)


def modulate(x, shift, scale):
    return x * (1 + unsqueeze(scale, 1)) + unsqueeze(shift, 1)


class TimestepEmbedder(nn.Module):
    hidden_size: int
    compute_dtype: DTypeLike
    param_dtype: DTypeLike
    frequency_embedding_size: int = 256

    def setup(self):
        self.mlp = nn.Sequential(
            [
                TorchLinear(
                    self.frequency_embedding_size,
                    self.hidden_size,
                    bias=True,
                    weight_init="0.02",
                    bias_init="zeros",
                    dtype=self.compute_dtype,
                    param_dtype=self.param_dtype,
                ),
                nn.silu,
                TorchLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    weight_init="0.02",
                    bias_init="zeros",
                    dtype=self.compute_dtype,
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class EmbeddingMixer(nn.Module):
    input_size: int
    hidden_size: int
    compute_dtype: DTypeLike
    param_dtype: DTypeLike

    def setup(self):
        self.in_proj = TorchLinear(
            self.input_size,
            self.hidden_size,
            bias=True,
            weight_init="xavier_uniform",
            bias_init="zeros",
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )
        self.out_proj = TorchLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, residual, features):
        delta = self.in_proj(features)
        delta = nn.silu(delta)
        delta = self.out_proj(delta)
        return residual + delta


class LabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int
    compute_dtype: DTypeLike
    param_dtype: DTypeLike
    use_cfg_embedding: bool = False

    def setup(self):
        self.embedding_table = TorchEmbedding(
            self.num_classes + int(self.use_cfg_embedding),
            self.hidden_size,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, labels, train, force_drop_ids=None):
        del train
        del force_drop_ids
        return self.embedding_table(labels)


class SiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    compute_dtype: DTypeLike
    param_dtype: DTypeLike
    mlp_ratio: float = 4.0

    def setup(self):
        # LayerNorm stays in fp32 even when the transformer compute path is bf16.
        self.norm1 = TorchLayerNorm(
            self.hidden_size,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attn = SiTAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            qkv_bias=True,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )
        self.norm2 = TorchLayerNorm(
            self.hidden_size,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            elementwise_affine=False,
            eps=1e-6,
        )
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        approx_gelu = lambda: partial(nn.gelu, approximate=True)
        self.mlp = SiTMlp(
            in_features=self.hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )
        self.adaLN_modulation = nn.Sequential(
            [
                nn.silu,
                TorchLinear(
                    self.hidden_size,
                    6 * self.hidden_size,
                    bias=True,
                    weight_init="zeros",
                    bias_init="zeros",
                    dtype=self.compute_dtype,
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    def __call__(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.adaLN_modulation(c),
            6,
            axis=1,
        )
        x = x + unsqueeze(gate_msa, 1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + unsqueeze(gate_mlp, 1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int
    compute_dtype: DTypeLike
    param_dtype: DTypeLike

    def setup(self):
        # Keep the final normalization in fp32 before projecting back to image space.
        self.norm_final = TorchLayerNorm(
            self.hidden_size,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = TorchLinear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )
        self.adaLN_modulation = nn.Sequential(
            [
                nn.silu,
                TorchLinear(
                    self.hidden_size,
                    2 * self.hidden_size,
                    bias=True,
                    weight_init="zeros",
                    bias_init="zeros",
                    dtype=self.compute_dtype,
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    compute_dtype: DTypeLike
    param_dtype: DTypeLike
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    decoder_hidden_size: Optional[int] = None
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    use_cfg_token: bool = False
    learn_sigma: bool = True
    matching: str = "flow"
    mix_time_embed: bool = False
    mix_image_embed: bool = False

    def setup(self):
        if self.matching not in {"flow", "flow_map", "diamond_map"}:
            raise ValueError(f"Unknown SiT matching mode {self.matching}")

        del self.decoder_hidden_size
        self.is_flow = self.matching == "flow"
        self.is_diamond = self.matching == "diamond_map"
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.x_embedder = PatchEmbed(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )
        if self.is_diamond:
            self.x_t_embedder = PatchEmbed(
                self.input_size,
                self.patch_size,
                self.in_channels,
                self.hidden_size,
                bias=True,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.x_t_embedder = None
        self.t_embedder = (
            TimestepEmbedder(
                self.hidden_size,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
            if self.is_flow
            else None
        )
        self.t_prime_embedder = (
            TimestepEmbedder(
                self.hidden_size,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
            if not self.is_flow
            else None
        )
        self.dt_embedder = (
            TimestepEmbedder(
                self.hidden_size,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
            if not self.is_flow
            else None
        )
        if self.is_diamond:
            self.s_embedder = TimestepEmbedder(
                self.hidden_size,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
            self.s_prime_embedder = TimestepEmbedder(
                self.hidden_size,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.s_embedder = None
            self.s_prime_embedder = None
        if self.is_diamond and self.mix_time_embed:
            self.time_embed_mixer = EmbeddingMixer(
                input_size=4 * self.hidden_size,
                hidden_size=self.hidden_size,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.time_embed_mixer = None
        if self.is_diamond and self.mix_image_embed:
            self.image_embed_mixer = EmbeddingMixer(
                input_size=2 * self.hidden_size,
                hidden_size=self.hidden_size,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.image_embed_mixer = None
        self.y_embedder = (
            LabelEmbedder(
                self.num_classes,
                self.hidden_size,
                use_cfg_embedding=self.use_cfg_token,
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )
            if self.num_classes > 0
            else None
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed_func = lambda: jnp.array(
            get_2d_sincos_pos_embed(self.hidden_size, int(num_patches**0.5))
        ).astype(self.compute_dtype)
        self.blocks = nn.Sequential(
            [
                SiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    compute_dtype=self.compute_dtype,
                    param_dtype=self.param_dtype,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )

    def unpatchify(self, x):
        channels = self.out_channels
        patch = self.x_embedder.patch_size[0]
        height = width = int(x.shape[1] ** 0.5)
        assert height * width == x.shape[1]

        x = x.reshape((x.shape[0], height, width, patch, patch, channels))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        imgs = x.reshape((x.shape[0], height * patch, width * patch, channels))
        return imgs

    def __call__(
        self,
        x,
        t_prime,
        y=None,
        train: bool = False,
        force_drop_ids: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        s: Optional[jnp.ndarray] = None,
        s_prime: Optional[jnp.ndarray] = None,
        x_t: Optional[jnp.ndarray] = None,
    ):
        x_tokens = self.x_embedder(x)
        if self.is_flow:
            t_embed = self.t_embedder(t_prime)
            c = t_embed
        else:
            if t is None:
                raise ValueError("SiT flow_map/diamond_map matching requires t.")
            t_prime_embed = self.t_prime_embedder(t_prime)
            dt_embed = self.dt_embedder(t_prime - t)
            c = t_prime_embed + dt_embed

        if self.is_diamond:
            if s is None or s_prime is None or x_t is None:
                raise ValueError("Diamond-map SiT requires s, s_prime, and x_t.")
            x_t_tokens = self.x_t_embedder(x_t)
            s_embed = self.s_embedder(s)
            s_prime_embed = self.s_prime_embedder(s_prime - s)
            if self.image_embed_mixer is not None:
                x_tokens = self.image_embed_mixer(
                    x_tokens,
                    jnp.concatenate([x_tokens, x_t_tokens], axis=-1),
                )
            else:
                x_tokens = x_tokens + x_t_tokens
            if self.time_embed_mixer is not None:
                c = self.time_embed_mixer(
                    t_prime_embed + dt_embed,
                    jnp.concatenate(
                        [t_prime_embed, dt_embed, s_embed, s_prime_embed], axis=-1
                    ),
                )
            else:
                c = c + s_embed + s_prime_embed

        x = x_tokens + self.pos_embed_func()
        if self.y_embedder is not None:
            assert y is not None, "Labels (y) must be provided when num_classes > 0"
            y = self.y_embedder(y, train=train, force_drop_ids=force_drop_ids)
            c = c + y

        for block in self.blocks.layers:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        if self.learn_sigma:
            x, _ = jnp.split(x, 2, axis=-1)
        return x


SiT_XL_2 = partial(SiT, depth=28, hidden_size=1152, patch_size=2, num_heads=16)
SiT_XL_4 = partial(SiT, depth=28, hidden_size=1152, patch_size=4, num_heads=16)
SiT_XL_8 = partial(SiT, depth=28, hidden_size=1152, patch_size=8, num_heads=16)
SiT_L_2 = partial(SiT, depth=24, hidden_size=1024, patch_size=2, num_heads=16)
SiT_L_4 = partial(SiT, depth=24, hidden_size=1024, patch_size=4, num_heads=16)
SiT_L_8 = partial(SiT, depth=24, hidden_size=1024, patch_size=8, num_heads=16)
SiT_B_2 = partial(SiT, depth=12, hidden_size=768, patch_size=2, num_heads=12)
SiT_B_4 = partial(SiT, depth=12, hidden_size=768, patch_size=4, num_heads=12)
SiT_B_8 = partial(SiT, depth=12, hidden_size=768, patch_size=8, num_heads=12)
SiT_S_2 = partial(SiT, depth=12, hidden_size=384, patch_size=2, num_heads=6)
SiT_S_4 = partial(SiT, depth=12, hidden_size=384, patch_size=4, num_heads=6)
SiT_S_8 = partial(SiT, depth=12, hidden_size=384, patch_size=8, num_heads=6)

SiT_MODELS = {
    "SiT_XL_2": SiT_XL_2,
    "SiT_XL_4": SiT_XL_4,
    "SiT_XL_8": SiT_XL_8,
    "SiT_L_2": SiT_L_2,
    "SiT_L_4": SiT_L_4,
    "SiT_L_8": SiT_L_8,
    "SiT_B_2": SiT_B_2,
    "SiT_B_4": SiT_B_4,
    "SiT_B_8": SiT_B_8,
    "SiT_S_2": SiT_S_2,
    "SiT_S_4": SiT_S_4,
    "SiT_S_8": SiT_S_8,
}

_SIT_MODEL_ALIASES = {
    "SiT-XL/2": "SiT_XL_2",
    "SiT-XL/4": "SiT_XL_4",
    "SiT-XL/8": "SiT_XL_8",
    "SiT-L/2": "SiT_L_2",
    "SiT-L/4": "SiT_L_4",
    "SiT-L/8": "SiT_L_8",
    "SiT-B/2": "SiT_B_2",
    "SiT-B/4": "SiT_B_4",
    "SiT-B/8": "SiT_B_8",
    "SiT-S/2": "SiT_S_2",
    "SiT-S/4": "SiT_S_4",
    "SiT-S/8": "SiT_S_8",
}
for alias, canonical in _SIT_MODEL_ALIASES.items():
    SiT_MODELS[alias] = SiT_MODELS[canonical]


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
