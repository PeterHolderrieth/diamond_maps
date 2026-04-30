"""JAX ImageReward BLIP model components and preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.core import freeze, unfreeze
from flax.serialization import from_bytes, to_bytes
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer

from . import repo_paths


HF_REPO_ID = "THUDM/ImageReward"


def _resolve_reward_param_dtype() -> tuple[jnp.dtype, str]:
    raw_dtype = os.environ.get("REWARD_PARAM_DTYPE", "bf16").strip().lower()
    if raw_dtype in ("bf16", "bfloat16"):
        return jnp.bfloat16, "v2_bf16"
    if raw_dtype in ("fp32", "float32"):
        return jnp.float32, "v3_fp32"
    raise ValueError(
        "Unsupported REWARD_PARAM_DTYPE. Expected one of: bf16, bfloat16, fp32, float32."
    )


IMAGE_REWARD_PARAM_DTYPE, IMAGE_REWARD_FLAX_CACHE_VERSION = (
    _resolve_reward_param_dtype()
)

_MODELS = {"ImageReward-v1.0": "ImageReward.pt"}
_TOKENIZER = None


def _hf_download(filename: str) -> str:
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)


def _resolve_paths(name: str) -> tuple[str, str]:
    if name in _MODELS:
        model_path = _hf_download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found.")
    med_config = _hf_download("med_config.json")
    return model_path, med_config


def _read_med_config(med_config_path: str) -> dict:
    with open(med_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["encoder_width"] = 1024
    return config


def _flax_cache_path(name: str) -> str:
    converted_dir = repo_paths.metric_models_path("converted")
    os.makedirs(converted_dir, exist_ok=True)
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return os.path.join(
        converted_dir,
        f"{IMAGE_REWARD_FLAX_CACHE_VERSION}_imagereward_{digest}.msgpack",
    )


def _torch_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _linear_kernel(value: torch.Tensor) -> np.ndarray:
    return _torch_array(value).T


def _conv_kernel(value: torch.Tensor) -> np.ndarray:
    return np.transpose(_torch_array(value), (2, 3, 1, 0))


def _attention_bias(attention_mask: jnp.ndarray) -> jnp.ndarray:
    return (1.0 - attention_mask.astype(jnp.float32))[:, None, None, :] * -10000.0


def _cast_floating_params(params):
    return freeze(
        jax.tree_util.tree_map(
            lambda x: x.astype(IMAGE_REWARD_PARAM_DTYPE)
            if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating)
            else x,
            params,
        )
    )


class VisionPatchEmbed(nn.Module):
    embed_dim: int
    patch_size: int

    @nn.compact
    def __call__(self, pixel_values: jnp.ndarray) -> jnp.ndarray:
        x = jnp.transpose(pixel_values, (0, 2, 3, 1))
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=True,
            name="proj",
        )(x)
        return jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))


class VisionAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        head_dim = self.embed_dim // self.num_heads
        qkv = nn.Dense(3 * self.embed_dim, use_bias=True, name="qkv")(hidden_states)
        qkv = qkv.reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            3,
            self.num_heads,
            head_dim,
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = jnp.einsum("bhtd,bhsd->bhts", query, key) * (head_dim ** -0.5)
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", weights, value)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(
            hidden_states.shape[0], hidden_states.shape[1], self.embed_dim
        )
        return nn.Dense(self.embed_dim, use_bias=True, name="proj")(out)


class VisionMlp(nn.Module):
    embed_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = nn.Dense(self.mlp_dim, use_bias=True, name="fc1")(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states, approximate=False)
        hidden_states = nn.Dense(self.embed_dim, use_bias=True, name="fc2")(hidden_states)
        return hidden_states


class VisionBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm(epsilon=1e-6, name="norm1")(hidden_states)
        hidden_states = hidden_states + VisionAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            name="attn",
        )(x)
        x = nn.LayerNorm(epsilon=1e-6, name="norm2")(hidden_states)
        hidden_states = hidden_states + VisionMlp(
            embed_dim=self.embed_dim,
            mlp_dim=self.mlp_dim,
            name="mlp",
        )(x)
        return hidden_states


class VisionTransformer(nn.Module):
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    patch_size: int = 16
    image_size: int = 224

    @nn.compact
    def __call__(self, pixel_values: jnp.ndarray) -> jnp.ndarray:
        x = VisionPatchEmbed(self.embed_dim, self.patch_size, name="patch_embed")(pixel_values)
        num_patches = (self.image_size // self.patch_size) ** 2
        cls_token = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.embed_dim)
        )
        pos_embed = self.param(
            "pos_embed", nn.initializers.zeros, (1, num_patches + 1, self.embed_dim)
        )
        cls_tokens = jnp.broadcast_to(cls_token, (x.shape[0], 1, self.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x = x + pos_embed[:, : x.shape[1], :]
        for idx in range(self.depth):
            x = VisionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=4 * self.embed_dim,
                name=f"block_{idx}",
            )(x)
        return nn.LayerNorm(epsilon=1e-6, name="norm")(x)


class TextEmbeddings(nn.Module):
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        seq_len = input_ids.shape[1]
        position_ids = jnp.arange(seq_len)[None, :]
        word_embeddings = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            name="word_embeddings",
        )(input_ids)
        position_embeddings = nn.Embed(
            num_embeddings=self.max_position_embeddings,
            features=self.hidden_size,
            name="position_embeddings",
        )(position_ids)
        hidden_states = word_embeddings + position_embeddings
        return nn.LayerNorm(epsilon=1e-12, name="LayerNorm")(hidden_states)


class TextAttentionCore(nn.Module):
    hidden_size: int
    num_heads: int
    kv_size: int

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_bias: Optional[jnp.ndarray],
        key_value_states: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        head_dim = self.hidden_size // self.num_heads
        source_states = hidden_states if key_value_states is None else key_value_states
        query = nn.Dense(self.hidden_size, use_bias=True, name="query")(hidden_states)
        key = nn.Dense(self.hidden_size, use_bias=True, name="key")(source_states)
        value = nn.Dense(self.hidden_size, use_bias=True, name="value")(source_states)
        query = query.reshape(query.shape[0], query.shape[1], self.num_heads, head_dim)
        key = key.reshape(key.shape[0], key.shape[1], self.num_heads, head_dim)
        value = value.reshape(value.shape[0], value.shape[1], self.num_heads, head_dim)
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.transpose(value, (0, 2, 1, 3))
        scores = jnp.einsum("bhtd,bhsd->bhts", query, key) * (head_dim ** -0.5)
        if attention_bias is not None:
            scores = scores + attention_bias
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", weights, value)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(
            hidden_states.shape[0], hidden_states.shape[1], self.hidden_size
        )
        return out


class TextAttentionOutput(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
        hidden_states = nn.Dense(self.hidden_size, use_bias=True, name="dense")(hidden_states)
        return nn.LayerNorm(epsilon=1e-12, name="LayerNorm")(hidden_states + residual)


class TextAttention(nn.Module):
    hidden_size: int
    num_heads: int
    kv_size: int

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_bias: Optional[jnp.ndarray],
        key_value_states: Optional[jnp.ndarray] = None,
        key_value_bias: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        attn_output = TextAttentionCore(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            kv_size=self.kv_size,
            name="self",
        )(hidden_states, key_value_bias if key_value_states is not None else attention_bias, key_value_states)
        return TextAttentionOutput(self.hidden_size, name="output")(attn_output, hidden_states)


class TextIntermediate(nn.Module):
    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = nn.Dense(self.intermediate_size, use_bias=True, name="dense")(hidden_states)
        return jax.nn.gelu(hidden_states, approximate=False)


class TextOutput(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
        hidden_states = nn.Dense(self.hidden_size, use_bias=True, name="dense")(hidden_states)
        return nn.LayerNorm(epsilon=1e-12, name="LayerNorm")(hidden_states + residual)


class TextLayer(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_heads: int
    encoder_width: int

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_bias: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_bias: jnp.ndarray,
    ) -> jnp.ndarray:
        hidden_states = TextAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            kv_size=self.hidden_size,
            name="attention",
        )(hidden_states, attention_bias)
        hidden_states = TextAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            kv_size=self.encoder_width,
            name="crossattention",
        )(hidden_states, attention_bias, encoder_hidden_states, encoder_attention_bias)
        intermediate = TextIntermediate(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            name="intermediate",
        )(hidden_states)
        return TextOutput(self.hidden_size, name="output")(intermediate, hidden_states)


class TextEncoder(nn.Module):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_layers: int
    max_position_embeddings: int
    encoder_width: int

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        encoder_attention_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        hidden_states = TextEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_position_embeddings,
            name="embeddings",
        )(input_ids)
        attention_bias = _attention_bias(attention_mask)
        encoder_attention_bias = _attention_bias(encoder_attention_mask)
        for idx in range(self.num_layers):
            hidden_states = TextLayer(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_heads=self.num_heads,
                encoder_width=self.encoder_width,
                name=f"layer_{idx}",
            )(hidden_states, attention_bias, encoder_hidden_states, encoder_attention_bias)
        return hidden_states


class RewardMlp(nn.Module):
    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        hidden_states = nn.Dense(1024, use_bias=True, name="dense_0")(hidden_states)
        hidden_states = nn.Dense(128, use_bias=True, name="dense_1")(hidden_states)
        hidden_states = nn.Dense(64, use_bias=True, name="dense_2")(hidden_states)
        hidden_states = nn.Dense(16, use_bias=True, name="dense_3")(hidden_states)
        hidden_states = nn.Dense(1, use_bias=True, name="dense_4")(hidden_states)
        return hidden_states


class _ImageRewardModule(nn.Module):
    vocab_size: int
    max_position_embeddings: int
    text_hidden_size: int
    text_intermediate_size: int
    text_num_heads: int
    text_num_layers: int
    encoder_width: int

    @nn.compact
    def __call__(
        self,
        pixel_values: jnp.ndarray,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        image_embeds = VisionTransformer(name="visual_encoder")(pixel_values)
        encoder_attention_mask = jnp.ones(
            image_embeds.shape[:2], dtype=attention_mask.dtype
        )
        hidden_states = TextEncoder(
            vocab_size=self.vocab_size,
            hidden_size=self.text_hidden_size,
            intermediate_size=self.text_intermediate_size,
            num_heads=self.text_num_heads,
            num_layers=self.text_num_layers,
            max_position_embeddings=self.max_position_embeddings,
            encoder_width=self.encoder_width,
            name="text_encoder",
        )(input_ids, attention_mask, image_embeds, encoder_attention_mask)
        rewards = RewardMlp(name="mlp")(hidden_states[:, 0, :])
        rewards = (rewards - 0.16717362830052426) / 1.0333394966054072
        return jnp.squeeze(rewards, axis=-1)


class ImageRewardModel:
    def __init__(self, module: _ImageRewardModule):
        self.module = module

    def apply(
        self,
        params,
        pixel_values: jnp.ndarray,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        return self.module.apply(
            {"params": params},
            pixel_values,
            input_ids,
            attention_mask,
        )


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        _TOKENIZER = tokenizer
    return _TOKENIZER


def _set_param(params: dict, path: tuple[str, ...], value: np.ndarray) -> None:
    curr = params
    for key in path[:-1]:
        curr = curr[key]
    curr[path[-1]] = value


def _assign_vision_params(params: dict, state_dict: dict) -> None:
    prefix = "blip.visual_encoder"
    _set_param(params, ("visual_encoder", "cls_token"), _torch_array(state_dict[f"{prefix}.cls_token"]))
    _set_param(params, ("visual_encoder", "pos_embed"), _torch_array(state_dict[f"{prefix}.pos_embed"]))
    _set_param(
        params,
        ("visual_encoder", "patch_embed", "proj", "kernel"),
        _conv_kernel(state_dict[f"{prefix}.patch_embed.proj.weight"]),
    )
    _set_param(
        params,
        ("visual_encoder", "patch_embed", "proj", "bias"),
        _torch_array(state_dict[f"{prefix}.patch_embed.proj.bias"]),
    )
    _set_param(
        params,
        ("visual_encoder", "norm", "scale"),
        _torch_array(state_dict[f"{prefix}.norm.weight"]),
    )
    _set_param(
        params,
        ("visual_encoder", "norm", "bias"),
        _torch_array(state_dict[f"{prefix}.norm.bias"]),
    )
    for idx in range(24):
        block = f"block_{idx}"
        state_prefix = f"{prefix}.blocks.{idx}"
        _set_param(params, ("visual_encoder", block, "norm1", "scale"), _torch_array(state_dict[f"{state_prefix}.norm1.weight"]))
        _set_param(params, ("visual_encoder", block, "norm1", "bias"), _torch_array(state_dict[f"{state_prefix}.norm1.bias"]))
        _set_param(params, ("visual_encoder", block, "attn", "qkv", "kernel"), _linear_kernel(state_dict[f"{state_prefix}.attn.qkv.weight"]))
        _set_param(params, ("visual_encoder", block, "attn", "qkv", "bias"), _torch_array(state_dict[f"{state_prefix}.attn.qkv.bias"]))
        _set_param(params, ("visual_encoder", block, "attn", "proj", "kernel"), _linear_kernel(state_dict[f"{state_prefix}.attn.proj.weight"]))
        _set_param(params, ("visual_encoder", block, "attn", "proj", "bias"), _torch_array(state_dict[f"{state_prefix}.attn.proj.bias"]))
        _set_param(params, ("visual_encoder", block, "norm2", "scale"), _torch_array(state_dict[f"{state_prefix}.norm2.weight"]))
        _set_param(params, ("visual_encoder", block, "norm2", "bias"), _torch_array(state_dict[f"{state_prefix}.norm2.bias"]))
        _set_param(params, ("visual_encoder", block, "mlp", "fc1", "kernel"), _linear_kernel(state_dict[f"{state_prefix}.mlp.fc1.weight"]))
        _set_param(params, ("visual_encoder", block, "mlp", "fc1", "bias"), _torch_array(state_dict[f"{state_prefix}.mlp.fc1.bias"]))
        _set_param(params, ("visual_encoder", block, "mlp", "fc2", "kernel"), _linear_kernel(state_dict[f"{state_prefix}.mlp.fc2.weight"]))
        _set_param(params, ("visual_encoder", block, "mlp", "fc2", "bias"), _torch_array(state_dict[f"{state_prefix}.mlp.fc2.bias"]))


def _assign_text_params(params: dict, state_dict: dict, num_layers: int) -> None:
    prefix = "blip.text_encoder"
    _set_param(
        params,
        ("text_encoder", "embeddings", "word_embeddings", "embedding"),
        _torch_array(state_dict[f"{prefix}.embeddings.word_embeddings.weight"]),
    )
    _set_param(
        params,
        ("text_encoder", "embeddings", "position_embeddings", "embedding"),
        _torch_array(state_dict[f"{prefix}.embeddings.position_embeddings.weight"]),
    )
    _set_param(
        params,
        ("text_encoder", "embeddings", "LayerNorm", "scale"),
        _torch_array(state_dict[f"{prefix}.embeddings.LayerNorm.weight"]),
    )
    _set_param(
        params,
        ("text_encoder", "embeddings", "LayerNorm", "bias"),
        _torch_array(state_dict[f"{prefix}.embeddings.LayerNorm.bias"]),
    )
    for idx in range(num_layers):
        layer = f"layer_{idx}"
        state_prefix = f"{prefix}.encoder.layer.{idx}"
        for attn_name in ("attention", "crossattention"):
            for proj_name in ("query", "key", "value"):
                _set_param(
                    params,
                    ("text_encoder", layer, attn_name, "self", proj_name, "kernel"),
                    _linear_kernel(state_dict[f"{state_prefix}.{attn_name}.self.{proj_name}.weight"]),
                )
                _set_param(
                    params,
                    ("text_encoder", layer, attn_name, "self", proj_name, "bias"),
                    _torch_array(state_dict[f"{state_prefix}.{attn_name}.self.{proj_name}.bias"]),
                )
            _set_param(
                params,
                ("text_encoder", layer, attn_name, "output", "dense", "kernel"),
                _linear_kernel(state_dict[f"{state_prefix}.{attn_name}.output.dense.weight"]),
            )
            _set_param(
                params,
                ("text_encoder", layer, attn_name, "output", "dense", "bias"),
                _torch_array(state_dict[f"{state_prefix}.{attn_name}.output.dense.bias"]),
            )
            _set_param(
                params,
                ("text_encoder", layer, attn_name, "output", "LayerNorm", "scale"),
                _torch_array(state_dict[f"{state_prefix}.{attn_name}.output.LayerNorm.weight"]),
            )
            _set_param(
                params,
                ("text_encoder", layer, attn_name, "output", "LayerNorm", "bias"),
                _torch_array(state_dict[f"{state_prefix}.{attn_name}.output.LayerNorm.bias"]),
            )
        _set_param(
            params,
            ("text_encoder", layer, "intermediate", "dense", "kernel"),
            _linear_kernel(state_dict[f"{state_prefix}.intermediate.dense.weight"]),
        )
        _set_param(
            params,
            ("text_encoder", layer, "intermediate", "dense", "bias"),
            _torch_array(state_dict[f"{state_prefix}.intermediate.dense.bias"]),
        )
        _set_param(
            params,
            ("text_encoder", layer, "output", "dense", "kernel"),
            _linear_kernel(state_dict[f"{state_prefix}.output.dense.weight"]),
        )
        _set_param(
            params,
            ("text_encoder", layer, "output", "dense", "bias"),
            _torch_array(state_dict[f"{state_prefix}.output.dense.bias"]),
        )
        _set_param(
            params,
            ("text_encoder", layer, "output", "LayerNorm", "scale"),
            _torch_array(state_dict[f"{state_prefix}.output.LayerNorm.weight"]),
        )
        _set_param(
            params,
            ("text_encoder", layer, "output", "LayerNorm", "bias"),
            _torch_array(state_dict[f"{state_prefix}.output.LayerNorm.bias"]),
        )


def _assign_mlp_params(params: dict, state_dict: dict) -> None:
    for idx, pt_idx in enumerate((0, 2, 4, 6, 7)):
        _set_param(
            params,
            ("mlp", f"dense_{idx}", "kernel"),
            _linear_kernel(state_dict[f"mlp.layers.{pt_idx}.weight"]),
        )
        _set_param(
            params,
            ("mlp", f"dense_{idx}", "bias"),
            _torch_array(state_dict[f"mlp.layers.{pt_idx}.bias"]),
        )


def load(name: str = "ImageReward-v1.0"):
    model_path, med_config_path = _resolve_paths(name)
    config = _read_med_config(med_config_path)
    module = _ImageRewardModule(
        vocab_size=config["vocab_size"],
        max_position_embeddings=config["max_position_embeddings"],
        text_hidden_size=config["hidden_size"],
        text_intermediate_size=config["intermediate_size"],
        text_num_heads=config["num_attention_heads"],
        text_num_layers=config["num_hidden_layers"],
        encoder_width=config["encoder_width"],
    )
    dummy_pixel_values = jnp.zeros((1, 3, 224, 224), dtype=jnp.float32)
    dummy_input_ids = jnp.zeros((1, 35), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones((1, 35), dtype=jnp.int32)
    params_template = module.init(
        jax.random.PRNGKey(0),
        dummy_pixel_values,
        dummy_input_ids,
        dummy_attention_mask,
    )["params"]
    cache_path = _flax_cache_path(name)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            params = freeze(from_bytes(params_template, f.read()))
        return ImageRewardModel(module), params, model_path, med_config_path

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    params = unfreeze(params_template)
    _assign_vision_params(params, state_dict)
    _assign_text_params(params, state_dict, config["num_hidden_layers"])
    _assign_mlp_params(params, state_dict)
    params = _cast_floating_params(params)
    with open(cache_path, "wb") as f:
        f.write(to_bytes(params))
    return ImageRewardModel(module), params, model_path, med_config_path
