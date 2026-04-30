"""Shared small-image EDM2 network config helpers."""

from __future__ import annotations

from typing import Any, Mapping

import ml_collections


CIFAR_UNET_KWARGS = {
    "model_channels": 128,
    "channel_mult": [2, 2, 2],
    "num_blocks": 4,
    "attn_resolutions": [16],
    "block_kwargs": {
        "dropout": 0.13,
    },
}

CELEBA_UNET_KWARGS = {
    "model_channels": 128,
    "channel_mult": [1, 2, 3, 4],
    "num_blocks": 3,
    "attn_resolutions": [16, 8],
    "block_kwargs": {
        "dropout": 0.0,
    },
}


def build_edm2_network(
    *,
    image_dims: tuple[int, int, int],
    num_classes: int,
    conditional: bool,
    unet_kwargs: Mapping[str, Any],
    matching: str | None = None,
    load_path: str = "",
    rescale: float,
    use_bfloat16: bool,
    reset_optimizer: bool = True,
    use_weight: bool = True,
    logvar_channels: int = 128,
    use_glass: bool = False,
    use_cfg_token: bool = False,
) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    if matching is not None:
        config.matching = matching
    config.network_type = "edm2"
    config.load_path = load_path
    config.init_from = ""
    config.img_resolution = image_dims[1]
    config.img_channels = image_dims[0]
    config.input_dims = image_dims
    config.label_dim = num_classes if conditional else 0
    config.use_glass = use_glass
    config.use_cfg_token = use_cfg_token
    config.reset_optimizer = reset_optimizer
    config.logvar_channels = logvar_channels
    config.use_bfloat16 = use_bfloat16
    config.use_weight = use_weight
    config.rescale = rescale
    config.unet_kwargs = dict(unet_kwargs)
    return config
