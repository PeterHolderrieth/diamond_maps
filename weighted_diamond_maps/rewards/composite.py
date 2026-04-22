"""Composite reward model wrapper used for benchmark runs."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from rewards.clip import CLIPLoss
from rewards.hps import HPSLoss
from rewards.imagereward import ImageRewardLoss, clip_img_transform
from rewards.pickscore import PickScoreLoss


class CompositeRewardLoss:
    """Composite reward: 1*ImageReward + 5*HPS + 0.01*CLIP + 0.05*PickScore.

    This wrapper keeps the same reward interface used by the pipelines:
    `reward(image, prompt, raw_image) -> Tensor`.
    """

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device | str,
        cache_dir: str,
        imagereward_weight: float = 1.0,
        hps_weight: float = 5.0,
        clip_weight: float = 0.01,
        pickscore_weight: float = 0.05,
    ) -> None:
        """Initialize composite reward components.

        Args:
            weighting: External weighting used by optimization pipelines.
            dtype: Target dtype for reward models.
            device: Target device.
            cache_dir: Cache path for model weights/tokenizer.
            imagereward_weight: ImageReward component weight.
            hps_weight: HPSv2 component weight.
            clip_weight: CLIP component weight.
            pickscore_weight: PickScore component weight.
        """
        self.name = "CompositeReward"
        self.weighting = float(weighting)
        self.dtype = dtype
        self.device = torch.device(device)
        self.img_transform = clip_img_transform()

        tokenizer = AutoTokenizer.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.image_reward = ImageRewardLoss(
            weighting=imagereward_weight,
            dtype=dtype,
            device=self.device,
            cache_dir=cache_dir,
            memsave=False,
        )
        self.hps_reward = HPSLoss(
            weighting=hps_weight,
            dtype=dtype,
            device=self.device,
            cache_dir=cache_dir,
            memsave=False,
        )
        self.clip_reward = CLIPLoss(
            weighting=clip_weight,
            dtype=dtype,
            device=self.device,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            memsave=False,
        )
        self.pickscore_reward = PickScoreLoss(
            weighting=pickscore_weight,
            dtype=dtype,
            device=self.device,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            memsave=False,
        )

        self.components: tuple[Any, ...] = (
            self.image_reward,
            self.hps_reward,
            self.clip_reward,
            self.pickscore_reward,
        )

    def _weighted_sum(
        self,
        image: Tensor,
        prompt: str | list[str],
        raw_image: Tensor | None,
    ) -> Tensor:
        """Compute weighted sum over all component rewards.

        Args:
            image: Preprocessed image tensor.
            prompt: Prompt text.
            raw_image: Raw image tensor in `[0, 1]` when available.

        Returns:
            Weighted composite score tensor.
        """
        total: Tensor | None = None
        for component in self.components:
            component_score = component(
                image=image, prompt=prompt, raw_image=raw_image
            ) * float(component.weighting)
            if total is None:
                total = component_score
            else:
                total = total + component_score
        if total is None:
            raise ValueError("Composite reward has no components.")
        return total

    def __call__(
        self,
        image: Tensor,
        prompt: str | list[str],
        raw_image: Tensor | None = None,
    ) -> Tensor:
        """Score images with the weighted composite reward.

        Args:
            image: Preprocessed image tensor.
            prompt: Prompt text.
            raw_image: Optional raw image tensor in `[0, 1]`.

        Returns:
            Composite score tensor.
        """
        image_in = image.to(device=self.device, dtype=self.dtype)
        raw_in = (
            raw_image.to(device=self.device, dtype=self.dtype)
            if raw_image is not None
            else None
        )
        return self._weighted_sum(image=image_in, prompt=prompt, raw_image=raw_in)

    def score_raw_images(self, prompt: str | list[str], image: Tensor) -> Tensor:
        """Score raw `[0, 1]` images.

        Args:
            prompt: Prompt text(s).
            image: Raw image tensor with shape `[B, 3, H, W]`.

        Returns:
            Composite score tensor.
        """
        raw_image = image.to(device=self.device, dtype=self.dtype)
        preprocessed_image = self.img_transform(raw_image)
        return self._weighted_sum(
            image=preprocessed_image,
            prompt=prompt,
            raw_image=raw_image,
        )
