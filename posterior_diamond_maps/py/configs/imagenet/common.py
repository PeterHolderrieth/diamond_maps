"""Shared ImageNet latent config helpers."""

from __future__ import annotations

import ml_collections
from common.repo_paths import ckpt_path, datasets_path


FLOW = "flow"
FLOW_MAP = "flow_map"
GLASS = "glass"
DIAMOND_EARLY_STOP = "diamond_early_stop"

IMAGENET_LATENT_DATASET = datasets_path("imagenet_latent_256_ema")
IMAGENET_FID_STATS = datasets_path(
    "imagenet_latent_256_ema",
    "imagenet_stats.npz",
)

MEANFLOW_SIT_B2_FLOW_MAP_CKPT = ckpt_path(
    "meanflow_sit_assets",
    "MeanFlow-B-2.pkl",
)
MEANFLOW_SIT_XL2_FLOW_MAP_CKPT = ckpt_path(
    "meanflow_sit_assets",
    "MeanFlow-XL-2.pkl",
)
SIT_B2_FLOW_CKPT = ckpt_path(
    "sit_assets",
    "SiT-B-2.pkl",
)
SIT_L2_FLOW_CKPT = ckpt_path(
    "sit_assets",
    "SiT-L-2.pkl",
)
SIT_XL2_FLOW_CKPT = ckpt_path(
    "sit_assets",
    "SiT-XL-2.pkl",
)
DIAMOND_MAP_SIT_B2_CKPT = ckpt_path(
    "ImageNet-DiamondMap-B2.pkl",
)


def build_sit_network(
    *,
    image_dims: tuple[int, int, int],
    num_classes: int,
    matching: str,
    model_name: str,
    load_path: str = "",
    init_from: str = "",
    reverse_time: bool,
    use_cfg_token: bool,
    sit_cfg_scale: float = 4.0,
    sit_cfg_channels: int = 3,
    compute_dtype: str = "bfloat16",
    param_dtype: str = "float32",
    reset_optimizer: bool = False,
    use_weight: bool = False,
    rescale: float = 1.0,
    learn_sigma: bool = False,
    mix_time_embed: bool | None = None,
    mix_image_embed: bool | None = None,
    use_glass: bool = False,
) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.matching = matching
    config.network_type = "sit"
    config.load_path = load_path
    config.use_glass = use_glass
    config.init_from = init_from
    config.img_resolution = image_dims[1]
    config.img_channels = image_dims[0]
    config.input_dims = image_dims
    config.label_dim = num_classes
    config.compute_dtype = compute_dtype
    config.param_dtype = param_dtype
    config.reverse_time = reverse_time
    config.use_cfg_token = use_cfg_token
    config.sit_cfg_scale = sit_cfg_scale
    config.sit_cfg_channels = sit_cfg_channels
    config.reset_optimizer = reset_optimizer
    config.use_weight = use_weight
    config.rescale = rescale
    config.sit_model_name = model_name
    sit_kwargs = {"learn_sigma": learn_sigma}
    if mix_time_embed is not None:
        sit_kwargs["mix_time_embed"] = mix_time_embed
    if mix_image_embed is not None:
        sit_kwargs["mix_image_embed"] = mix_image_embed
    config.sit_kwargs = ml_collections.ConfigDict(sit_kwargs)
    return config
