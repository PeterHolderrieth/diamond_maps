"""CelebA-64 inverse-guidance config variants."""

from __future__ import annotations

import ml_collections

from configs.celeba64 import common
from configs.small_image import CELEBA_UNET_KWARGS, build_edm2_network


def _build_guidance_config(
    *,
    posterior_sample_type: str,
) -> ml_collections.ConfigDict:
    guidance = ml_collections.ConfigDict()
    guidance.problem_type = "super_resolution"
    guidance.mask_prob = None
    guidance.noise_std = 0.05
    guidance.sr_factor = 4
    guidance.latent_split = None

    guidance.base_sample_type = "FLOW"
    guidance.posterior_sample_type = posterior_sample_type

    guidance.guidance_scale_schedule = "constant"
    guidance.ema_factor = 0.9999
    guidance.metric_samples = 1000
    guidance.metric_bs = 100
    guidance.comp_lpips = True
    guidance.comp_kid = True
    guidance.plot_bs = 5
    guidance.seed_override = None
    return guidance


def _build_sr4_flow_posterior_flow(
    output_folder: str,
) -> ml_collections.ConfigDict:
    config = common.make_base_config(output_folder, shuffle=False)
    config.guidance = _build_guidance_config(
        posterior_sample_type="FLOW",
    )
    config.base_network_slot = "main"
    config.posterior_network_slot = "main"
    config.network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CELEBA_UNET_KWARGS,
        matching="flow_map",
        load_path=common.FLOW_MAP_CKPT,
        rescale=1.0,
        use_bfloat16=True,
        use_glass=False,
    )
    config.logging.wandb_name = "celeba_sr4_flow_base_posterior_flow"
    config.logging.output_name = config.logging.wandb_name
    return config


def _build_sr4_flow_posterior_flow_map(
    output_folder: str,
) -> ml_collections.ConfigDict:
    config = common.make_base_config(output_folder, shuffle=False)
    config.guidance = _build_guidance_config(
        posterior_sample_type="FLOW_MAP",
    )
    config.base_network_slot = "main"
    config.posterior_network_slot = "main"
    config.network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CELEBA_UNET_KWARGS,
        matching="flow_map",
        load_path=common.FLOW_MAP_CKPT,
        rescale=1.0,
        use_bfloat16=True,
        use_glass=False,
    )
    config.logging.wandb_name = "celeba_sr4_flow_base_posterior_flow_map"
    config.logging.output_name = config.logging.wandb_name
    return config


def _build_sr4_flow_posterior_diamond_early_stop(
    output_folder: str,
) -> ml_collections.ConfigDict:
    config = common.make_base_config(output_folder, shuffle=False)
    config.guidance = _build_guidance_config(
        posterior_sample_type="DIAMOND_EARLY_STOP",
    )
    config.base_network_slot = "main"
    config.posterior_network_slot = "sup"
    config.network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CELEBA_UNET_KWARGS,
        matching="flow_map",
        load_path=common.FLOW_MAP_CKPT,
        rescale=1.0,
        use_bfloat16=True,
        use_glass=False,
    )
    config.logging.wandb_name = "celeba_sr4_flow_base_posterior_diamond_early_stop"
    config.logging.output_name = config.logging.wandb_name
    common.configure_fixed_early_stop_diamond_training(config, supervise_type="none")
    config.sup_network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CELEBA_UNET_KWARGS,
        matching="diamond_map",
        load_path=common.DIAMOND_EARLY_STOP_CKPT,
        rescale=1.0,
        use_bfloat16=True,
        use_glass=False,
    )
    return config


VARIANTS = (
    {
        "name": "sr4_flow_posterior_flow",
        "builder": _build_sr4_flow_posterior_flow,
    },
    {
        "name": "sr4_flow_posterior_flow_map",
        "builder": _build_sr4_flow_posterior_flow_map,
    },
    {
        "name": "sr4_flow_posterior_diamond_early_stop",
        "builder": _build_sr4_flow_posterior_diamond_early_stop,
    },
)


def get_config(
    slurm_id: int, output_folder: str
) -> ml_collections.ConfigDict:
    variant = VARIANTS[slurm_id]
    return variant["builder"](output_folder)
