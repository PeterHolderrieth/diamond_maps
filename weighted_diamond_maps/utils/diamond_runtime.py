"""Shared runtime helpers for Diamond Maps scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = "./model_cache"

DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_FLUX_LORA_SOURCE = "gabeguofanclub/flux-1-dev-flowmap-lsd"
DEFAULT_FLUX_LORA_WEIGHT_NAME = (
    "01-12-26/runs/res_512_steps_50k_rank_64_lr_1e-4/checkpoint-43000/"
    "pytorch_lora_weights.safetensors"
)
DEFAULT_FLUX_SIZE = 512

DEFAULT_SANA_MODEL = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
DEFAULT_SANA_SIZE = 1024

REWARD_MODEL_CHOICES = ("composite", "imagereward", "blueness")

INFERENCE_METHODS = ("base", "flow_map_guidance", "weighted_diamond")
BENCHMARK_METHODS = (*INFERENCE_METHODS, "best_of_n")
METHOD_ALIASES = {
    "base": "base",
    "fmtt": "flow_map_guidance",
    "flow_map_guidance": "flow_map_guidance",
    "diamond_full": "weighted_diamond",
    "weighted_diamond": "weighted_diamond",
    "best_of_n": "best_of_n",
    "diamond_base": "diamond_base",
}


def normalize_method_name(name: str) -> str:
    """Normalize paper-facing method names and historical aliases."""
    normalized = str(name).strip().lower().replace("-", "_")
    try:
        return METHOD_ALIASES[normalized]
    except KeyError as exc:
        valid = ", ".join(BENCHMARK_METHODS)
        raise ValueError(f"Unsupported method: {name}. Choose from {valid}.") from exc


def dtype_from_name(name: str):
    """Resolve a dtype name without importing torch at module import time."""
    import torch

    normalized = str(name).lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_generator(device: str, seed: int):
    """Create a deterministic torch generator for the requested device."""
    import torch

    generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slugify(text: str, max_len: int = 72) -> str:
    """Convert text to a compact filesystem-safe slug."""
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return (slug or "prompt")[:max_len]


def load_flux_pipeline(
    cls: type,
    *,
    model_name: str,
    cache_dir: str,
    lora_source: str,
    weight_name: str,
    device: str,
    dtype: Any,
):
    """Load a FLUX flow-map pipeline with dual-time embedding and LoRA."""
    from utils.dual_time_embedder import add_dual_time_embedder

    pipeline = cls.from_pretrained(
        model_name,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    pipeline.transformer = add_dual_time_embedder(pipeline.transformer)
    pipeline = pipeline.to(device, dtype)
    pipeline.load_lora_weights(
        lora_source,
        weight_name=weight_name,
        cache_dir=cache_dir,
    )
    pipeline = pipeline.to(device, dtype)
    return pipeline


def load_sana_pipeline(
    cls: type,
    *,
    model_name: str,
    cache_dir: str,
    device: str,
    dtype: Any,
):
    """Load a SANA pipeline."""
    pipeline = cls.from_pretrained(
        model_name,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
    return pipeline.to(device, dtype)


def build_reward_model(
    *,
    reward_model: str,
    dtype: Any,
    device: str,
    cache_dir: str,
):
    """Build the reward models used by inference scripts."""
    if reward_model == "imagereward":
        from rewards.imagereward import ImageRewardLoss

        return ImageRewardLoss(
            weighting=1.0,
            dtype=dtype,
            device=device,
            cache_dir=cache_dir,
            memsave=False,
        )

    if reward_model == "blueness":
        from rewards.blueness import BlueRewardLoss

        return BlueRewardLoss(
            weighting=1.0,
            dtype=dtype,
            device=device,
        )

    if reward_model == "composite":
        from rewards.composite import CompositeRewardLoss

        return CompositeRewardLoss(
            weighting=1.0,
            dtype=dtype,
            device=device,
            cache_dir=cache_dir,
            imagereward_weight=1.0,
            hps_weight=5.0,
            clip_weight=0.01,
            pickscore_weight=0.05,
        )

    raise ValueError(
        f"Unsupported reward model: {reward_model}. "
        f"Choose from {', '.join(REWARD_MODEL_CHOICES)}."
    )
