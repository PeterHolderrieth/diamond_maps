"""Runtime helpers for benchmark sample generation."""

from __future__ import annotations

import argparse
from typing import Any

from utils.diamond_runtime import dtype_from_name, load_flux_pipeline, load_sana_pipeline


def clear_cuda_cache(device: str) -> None:
    """Clear CUDA cache when running on CUDA."""
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


def build_reward_model(args: argparse.Namespace, dtype: Any):
    """Build the reward model selected by the benchmark CLI."""
    if args.reward_model == "imagereward":
        from rewards.imagereward import ImageRewardLoss

        return ImageRewardLoss(
            weighting=1.0,
            dtype=dtype,
            device=args.device,
            cache_dir=args.cache_dir,
            memsave=False,
        )

    if args.reward_model == "blueness":
        from rewards.blueness import BlueRewardLoss

        return BlueRewardLoss(
            weighting=1.0,
            dtype=dtype,
            device=args.device,
        )

    if args.reward_model == "composite":
        from rewards.composite import CompositeRewardLoss

        return CompositeRewardLoss(
            weighting=1.0,
            dtype=dtype,
            device=args.device,
            cache_dir=args.cache_dir,
            imagereward_weight=1.0,
            hps_weight=5.0,
            clip_weight=0.01,
            pickscore_weight=0.05,
        )


def score_pil_image(
    reward_model: Any,
    prompt: str,
    image: Any,
    device: str,
) -> float:
    """Compute a scalar reward score for one PIL image."""
    import numpy as np
    import torch

    image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    image_tensor = (
        torch.from_numpy(image_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=reward_model.dtype)
    )
    score = reward_model.score_raw_images(prompt, image_tensor)
    return float(score.mean().item())
