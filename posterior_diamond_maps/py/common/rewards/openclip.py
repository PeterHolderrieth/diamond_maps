"""OpenCLIP/CLIP model loading, conversion, tokenization, and features."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.core import freeze
from flax.serialization import from_bytes, to_bytes
from huggingface_hub import hf_hub_download

from .. import repo_paths

os.environ.setdefault("USE_TF", "0")


def _resolve_reward_param_dtype() -> tuple[jnp.dtype, str, str]:
    raw_dtype = os.environ.get("REWARD_PARAM_DTYPE", "bf16").strip().lower()
    if raw_dtype in ("bf16", "bfloat16"):
        return jnp.bfloat16, "bf16", "v2_bf16"
    if raw_dtype in ("fp32", "float32"):
        return jnp.float32, "fp32", "v3_fp32"
    raise ValueError(
        "Unsupported REWARD_PARAM_DTYPE. Expected one of: bf16, bfloat16, fp32, float32."
    )


FLAX_MODEL_DTYPE, FLAX_MODEL_DTYPE_NAME, FLAX_CACHE_FORMAT_VERSION = (
    _resolve_reward_param_dtype()
)

from transformers import AutoConfig, AutoProcessor, CLIPModel, FlaxCLIPModel
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax


OPENCLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
OPENCLIP_IMAGE_SIZE = 224
OPENCLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
OPENCLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

HPSV2_REPO_ID = "xswu/HPSv2"
HPSV2_CHECKPOINTS = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}

_OPENCLIP_PROCESSOR = None


def normalize_l2(x: jnp.ndarray, axis: int = -1, eps: float = 1e-12) -> jnp.ndarray:
    x = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.square(jnp.linalg.norm(x, axis=axis, keepdims=True)) + eps)
    return x / norm


def preprocess_openclip_pixels(pixel_images: jnp.ndarray) -> jnp.ndarray:
    """
    Ported from: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transform.py
    """
    batch_size, height, width, _ = pixel_images.shape
    pixel_images = jnp.clip(pixel_images, 0.0, 1.0)

    if height != OPENCLIP_IMAGE_SIZE or width != OPENCLIP_IMAGE_SIZE:
        scale = OPENCLIP_IMAGE_SIZE / min(height, width)
        resized_height = max(int(round(height * scale)), OPENCLIP_IMAGE_SIZE)
        resized_width = max(int(round(width * scale)), OPENCLIP_IMAGE_SIZE)
        pixel_images = jax.image.resize(
            pixel_images,
            shape=(batch_size, resized_height, resized_width, 3),
            method="bicubic",
            antialias=True,
        )

        top = max((resized_height - OPENCLIP_IMAGE_SIZE) // 2, 0)
        left = max((resized_width - OPENCLIP_IMAGE_SIZE) // 2, 0)
        pixel_images = pixel_images[
            :, top : top + OPENCLIP_IMAGE_SIZE, left : left + OPENCLIP_IMAGE_SIZE, :
        ]
    pixel_images = (pixel_images - OPENCLIP_MEAN) / OPENCLIP_STD
    return jnp.transpose(pixel_images, (0, 3, 1, 2))


def _get_openclip_processor():
    global _OPENCLIP_PROCESSOR
    if _OPENCLIP_PROCESSOR is None:
        _OPENCLIP_PROCESSOR = AutoProcessor.from_pretrained(OPENCLIP_MODEL_ID)
    return _OPENCLIP_PROCESSOR


def tokenize_openclip_prompt(prompt: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    processor = _get_openclip_processor()
    encoded = processor(
        text=[prompt],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="np",
    )
    input_ids = jnp.asarray(encoded["input_ids"][0], dtype=jnp.int32)
    attention_mask = jnp.asarray(encoded["attention_mask"][0], dtype=jnp.int32)
    return input_ids, attention_mask


def _load_flax_clip_from_hf(model_id: str) -> tuple[FlaxCLIPModel, Dict]:
    cached = _load_cached_flax_clip(model_id, cache_tag="hf")
    if cached is not None:
        return cached

    pt_model = CLIPModel.from_pretrained(model_id, use_safetensors=True).eval()
    flax_model = FlaxCLIPModel(pt_model.config, dtype=FLAX_MODEL_DTYPE)
    params = _cast_floating_params(
        freeze(convert_pytorch_state_dict_to_flax(pt_model.state_dict(), flax_model))
    )
    _write_cached_flax_params(_get_flax_cache_path("hf", model_id), params)
    return flax_model, params


def _slugify_model_id(model_id: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", model_id).strip("_").lower()
    if not slug:
        slug = "model"
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:80]}_{digest}"


def _get_flax_cache_path(cache_tag: str, model_id: str) -> str:
    converted_dir = repo_paths.metric_models_path("converted")
    os.makedirs(converted_dir, exist_ok=True)
    filename = (
        f"{FLAX_CACHE_FORMAT_VERSION}_{cache_tag}_{_slugify_model_id(model_id)}.msgpack"
    )
    return os.path.join(converted_dir, filename)


def _cast_floating_params(params: Dict) -> Dict:
    return freeze(
        jax.tree_util.tree_map(
            lambda x: x.astype(FLAX_MODEL_DTYPE)
            if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating)
            else x,
            params,
        )
    )


def _load_cached_flax_clip(
    model_id: str,
    *,
    cache_tag: str,
) -> Optional[Tuple[FlaxCLIPModel, Dict]]:
    cache_path = _get_flax_cache_path(cache_tag, model_id)
    if not os.path.exists(cache_path):
        return None

    config = AutoConfig.from_pretrained(model_id)
    flax_model = FlaxCLIPModel(config, dtype=FLAX_MODEL_DTYPE, _do_init=False)
    with open(cache_path, "rb") as f:
        params = freeze(from_bytes(flax_model.params_shape_tree, f.read()))
    return flax_model, params


def _write_cached_flax_params(cache_path: str, params: Dict) -> None:
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        f.write(to_bytes(params))
    os.replace(tmp_path, cache_path)


def get_base_flax_clip() -> tuple[FlaxCLIPModel, Dict]:
    return _load_flax_clip_from_hf(OPENCLIP_MODEL_ID)


def get_pickscore_flax_clip() -> tuple[FlaxCLIPModel, Dict]:
    return _load_flax_clip_from_hf("yuvalkirstain/PickScore_v1")


def _split_qkv(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_weight, k_weight, v_weight = torch.chunk(
        state_dict[f"{prefix}.attn.in_proj_weight"], chunks=3, dim=0
    )
    q_bias, k_bias, v_bias = torch.chunk(
        state_dict[f"{prefix}.attn.in_proj_bias"], chunks=3, dim=0
    )
    return q_weight, k_weight, v_weight, q_bias, k_bias, v_bias


def _layer_indices(state_dict: Dict[str, torch.Tensor], prefix: str) -> Iterable[int]:
    prefix = f"{prefix}.resblocks."
    return sorted(
        {
            int(key[len(prefix) :].split(".")[0])
            for key in state_dict
            if key.startswith(prefix)
        }
    )


def _convert_openclip_state_dict_to_hf_clip(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Ported from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/convert_clip_original_pytorch_to_hf.py
    """
    converted = {
        "text_model.embeddings.token_embedding.weight": state_dict["token_embedding.weight"],
        "text_model.embeddings.position_embedding.weight": state_dict[
            "positional_embedding"
        ],
        "text_model.final_layer_norm.weight": state_dict["ln_final.weight"],
        "text_model.final_layer_norm.bias": state_dict["ln_final.bias"],
        "text_projection.weight": state_dict["text_projection"].T,
        "vision_model.embeddings.class_embedding": state_dict["visual.class_embedding"],
        "vision_model.embeddings.position_embedding.weight": state_dict[
            "visual.positional_embedding"
        ],
        "vision_model.embeddings.patch_embedding.weight": state_dict[
            "visual.conv1.weight"
        ],
        "vision_model.pre_layrnorm.weight": state_dict["visual.ln_pre.weight"],
        "vision_model.pre_layrnorm.bias": state_dict["visual.ln_pre.bias"],
        "vision_model.post_layernorm.weight": state_dict["visual.ln_post.weight"],
        "vision_model.post_layernorm.bias": state_dict["visual.ln_post.bias"],
        "visual_projection.weight": state_dict["visual.proj"].T,
        "logit_scale": state_dict["logit_scale"],
    }

    for idx in _layer_indices(state_dict, "transformer"):
        src = f"transformer.resblocks.{idx}"
        dst = f"text_model.encoder.layers.{idx}"
        (
            q_weight,
            k_weight,
            v_weight,
            q_bias,
            k_bias,
            v_bias,
        ) = _split_qkv(state_dict, src)
        converted[f"{dst}.self_attn.q_proj.weight"] = q_weight
        converted[f"{dst}.self_attn.k_proj.weight"] = k_weight
        converted[f"{dst}.self_attn.v_proj.weight"] = v_weight
        converted[f"{dst}.self_attn.q_proj.bias"] = q_bias
        converted[f"{dst}.self_attn.k_proj.bias"] = k_bias
        converted[f"{dst}.self_attn.v_proj.bias"] = v_bias
        converted[f"{dst}.self_attn.out_proj.weight"] = state_dict[
            f"{src}.attn.out_proj.weight"
        ]
        converted[f"{dst}.self_attn.out_proj.bias"] = state_dict[
            f"{src}.attn.out_proj.bias"
        ]
        converted[f"{dst}.layer_norm1.weight"] = state_dict[f"{src}.ln_1.weight"]
        converted[f"{dst}.layer_norm1.bias"] = state_dict[f"{src}.ln_1.bias"]
        converted[f"{dst}.layer_norm2.weight"] = state_dict[f"{src}.ln_2.weight"]
        converted[f"{dst}.layer_norm2.bias"] = state_dict[f"{src}.ln_2.bias"]
        converted[f"{dst}.mlp.fc1.weight"] = state_dict[f"{src}.mlp.c_fc.weight"]
        converted[f"{dst}.mlp.fc1.bias"] = state_dict[f"{src}.mlp.c_fc.bias"]
        converted[f"{dst}.mlp.fc2.weight"] = state_dict[f"{src}.mlp.c_proj.weight"]
        converted[f"{dst}.mlp.fc2.bias"] = state_dict[f"{src}.mlp.c_proj.bias"]

    for idx in _layer_indices(state_dict, "visual.transformer"):
        src = f"visual.transformer.resblocks.{idx}"
        dst = f"vision_model.encoder.layers.{idx}"
        (
            q_weight,
            k_weight,
            v_weight,
            q_bias,
            k_bias,
            v_bias,
        ) = _split_qkv(state_dict, src)
        converted[f"{dst}.self_attn.q_proj.weight"] = q_weight
        converted[f"{dst}.self_attn.k_proj.weight"] = k_weight
        converted[f"{dst}.self_attn.v_proj.weight"] = v_weight
        converted[f"{dst}.self_attn.q_proj.bias"] = q_bias
        converted[f"{dst}.self_attn.k_proj.bias"] = k_bias
        converted[f"{dst}.self_attn.v_proj.bias"] = v_bias
        converted[f"{dst}.self_attn.out_proj.weight"] = state_dict[
            f"{src}.attn.out_proj.weight"
        ]
        converted[f"{dst}.self_attn.out_proj.bias"] = state_dict[
            f"{src}.attn.out_proj.bias"
        ]
        converted[f"{dst}.layer_norm1.weight"] = state_dict[f"{src}.ln_1.weight"]
        converted[f"{dst}.layer_norm1.bias"] = state_dict[f"{src}.ln_1.bias"]
        converted[f"{dst}.layer_norm2.weight"] = state_dict[f"{src}.ln_2.weight"]
        converted[f"{dst}.layer_norm2.bias"] = state_dict[f"{src}.ln_2.bias"]
        converted[f"{dst}.mlp.fc1.weight"] = state_dict[f"{src}.mlp.c_fc.weight"]
        converted[f"{dst}.mlp.fc1.bias"] = state_dict[f"{src}.mlp.c_fc.bias"]
        converted[f"{dst}.mlp.fc2.weight"] = state_dict[f"{src}.mlp.c_proj.weight"]
        converted[f"{dst}.mlp.fc2.bias"] = state_dict[f"{src}.mlp.c_proj.bias"]

    return converted


def _load_hpsv2_flax_clip_full(version: str = "v2.1") -> tuple[FlaxCLIPModel, Dict]:
    if version not in HPSV2_CHECKPOINTS:
        raise ValueError(f"Unsupported HPS version: {version}")
    cache_key = f"{HPSV2_REPO_ID}:{HPSV2_CHECKPOINTS[version]}:{FLAX_MODEL_DTYPE_NAME}"
    cached = _load_cached_flax_clip(
        OPENCLIP_MODEL_ID, cache_tag=_slugify_model_id(cache_key)
    )
    if cached is not None:
        return cached

    checkpoint_path = hf_hub_download(
        repo_id=HPSV2_REPO_ID,
        filename=HPSV2_CHECKPOINTS[version],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    config = AutoConfig.from_pretrained(OPENCLIP_MODEL_ID)
    flax_model = FlaxCLIPModel(config, dtype=FLAX_MODEL_DTYPE)
    hf_state_dict = _convert_openclip_state_dict_to_hf_clip(state_dict)
    params = _cast_floating_params(
        freeze(convert_pytorch_state_dict_to_flax(hf_state_dict, flax_model))
    )
    _write_cached_flax_params(
        _get_flax_cache_path(_slugify_model_id(cache_key), OPENCLIP_MODEL_ID),
        params,
    )
    return flax_model, params


def get_hpsv2_flax_clip(version: str = "v2.1") -> tuple[FlaxCLIPModel, Dict]:
    return _load_hpsv2_flax_clip_full(version)


def get_text_features(
    model: FlaxCLIPModel,
    params: Dict,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
) -> jnp.ndarray:
    batch_input_ids = input_ids[jnp.newaxis, ...] if input_ids.ndim == 1 else input_ids
    batch_attention_mask = (
        attention_mask[jnp.newaxis, ...]
        if attention_mask.ndim == 1
        else attention_mask
    )
    return model.get_text_features(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
        params=params,
    ).astype(jnp.float32)


def get_image_features(
    model: FlaxCLIPModel,
    params: Dict,
    pixel_images: jnp.ndarray,
) -> jnp.ndarray:
    pixel_values = preprocess_openclip_pixels(pixel_images)
    return model.get_image_features(pixel_values=pixel_values, params=params).astype(
        jnp.float32
    )
