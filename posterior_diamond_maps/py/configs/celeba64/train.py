"""CelebA-64 training config variants."""

from __future__ import annotations

import ml_collections

from configs.celeba64 import common
from configs.small_image import CELEBA_UNET_KWARGS, build_edm2_network


def _build_flow_map_lsd(
    output_folder: str,
) -> ml_collections.ConfigDict:
    config = common.make_base_config(output_folder, shuffle=True)
    config.logging.sample_types = ["FLOW_MAP"]
    config.logging.outer_steps = [1]
    config.logging.inner_steps = [1, 2, 4, 8]
    config.network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CELEBA_UNET_KWARGS,
        matching="flow_map",
        rescale=1.0,
        use_bfloat16=True,
        use_glass=False,
    )
    config.logging.wandb_name = "celeba_no_rescale_lsd"
    config.logging.output_name = config.logging.wandb_name
    return config


def _build_glass_diamond_early_stop_lsd(
    output_folder: str,
) -> ml_collections.ConfigDict:
    config = common.make_base_config(output_folder, shuffle=True)
    common.configure_fixed_early_stop_diamond_training(config, supervise_type="glass")
    config.logging.sample_types = ["DIAMOND_EARLY_STOP"]
    config.logging.outer_steps = [1, 2]
    config.logging.inner_steps = [1, 2, 4, 8]
    config.network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CELEBA_UNET_KWARGS,
        matching="diamond_map",
        rescale=1.0,
        use_bfloat16=True,
        use_glass=False,
    )
    config.sup_network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CELEBA_UNET_KWARGS,
        matching="flow_map",
        load_path=common.FLOW_MAP_CKPT,
        rescale=1.0,
        use_bfloat16=True,
        use_glass=True,
    )
    config.logging.wandb_name = "celeba_glass_no_rescale_diamond_early_stop_lsd"
    config.logging.output_name = config.logging.wandb_name
    return config


VARIANTS = (
    {
        "name": "flow_map_lsd",
        "builder": _build_flow_map_lsd,
    },
    {
        "name": "glass_diamond_early_stop_lsd",
        "builder": _build_glass_diamond_early_stop_lsd,
    },
)


def get_config(
    slurm_id: int, output_folder: str
) -> ml_collections.ConfigDict:
    variant = VARIANTS[slurm_id]
    return variant["builder"](output_folder)
