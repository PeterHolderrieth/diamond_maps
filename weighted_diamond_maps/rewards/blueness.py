"""Differentiable blueness reward."""

from __future__ import annotations

import torch
from torch import Tensor


class BlueRewardLoss:
    """Simple differentiable reward that measures how blue an image is.

    The score is computed from raw RGB tensors in `[0, 1]` as:

    `mean(blue - 0.5 * (red + green))`

    Higher values indicate stronger blue dominance.
    """

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        """Initialize blueness reward.

        Args:
            weighting: Relative weighting used by guidance pipelines.
            dtype: Preferred tensor dtype for benchmark image scoring paths.
            device: Target device string or torch.device.
        """
        self.name = "BluenessReward"
        self.weighting = float(weighting)
        self.dtype = dtype
        self.device = torch.device(device)

    def _validate_raw_image(self, image: Tensor) -> None:
        """Validate raw image tensor contract.

        Args:
            image: Raw image tensor expected in shape `[B, 3, H, W]` and range `[0, 1]`.

        Raises:
            ValueError: If shape/range constraints are violated.
        """
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"Expected raw image shape [B, 3, H, W], got {tuple(image.shape)}."
            )
        if not image.is_floating_point():
            raise ValueError("Blueness reward expects floating-point image tensors.")
        image_min = float(image.detach().amin().item())
        image_max = float(image.detach().amax().item())
        if image_min < 0.0 or image_max > 1.0:
            raise ValueError(
                "Blueness reward expects raw image range [0, 1], "
                f"got min={image_min:.6f}, max={image_max:.6f}."
            )
        if torch.isnan(image).any():
            raise ValueError("Raw image contains NaN values.")

    def score_raw_images(self, prompt: str | list[str], image: Tensor) -> Tensor:
        """Score a batch of raw images.

        Args:
            prompt: Prompt text(s), unused by this reward.
            image: Raw image tensor with shape `[B, 3, H, W]` in `[0, 1]`.

        Returns:
            Tensor of shape `[B]` containing blueness scores.
        """
        del prompt
        self._validate_raw_image(image)

        red_channel = image[:, 0, :, :]
        green_channel = image[:, 1, :, :]
        blue_channel = image[:, 2, :, :]
        blueness_map = blue_channel - 0.5 * (red_channel + green_channel)
        return blueness_map.mean(dim=(1, 2))

    def __call__(
        self,
        image: Tensor,
        prompt: str | list[str],
        raw_image: Tensor | None = None,
    ) -> Tensor:
        """Compute reward value.

        Args:
            image: Preprocessed image tensor, unused.
            prompt: Prompt text(s), unused.
            raw_image: Raw image tensor with shape `[B, 3, H, W]`.

        Returns:
            Tensor of shape `[B]` containing reward values.

        Raises:
            ValueError: If `raw_image` is missing.
        """
        del image
        if raw_image is None:
            raise ValueError(
                "Blueness reward requires `raw_image` input in shape [B, 3, H, W]."
            )
        return self.score_raw_images(prompt, raw_image)
