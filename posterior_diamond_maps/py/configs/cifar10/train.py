"""CIFAR-10 training config variants."""

from __future__ import annotations

import os

import ml_collections
from common.repo_paths import ckpt_path, datasets_path

from configs.small_image import CIFAR_UNET_KWARGS, build_edm2_network


def _make_base_config(output_folder: str) -> ml_collections.ConfigDict:
    import jax

    config = ml_collections.ConfigDict()
    dataset_location = datasets_path()

    config.training = ml_collections.ConfigDict()
    config.training.supervise_type = "none"
    config.training.shuffle = True
    config.training.conditional = False
    config.training.class_dropout = 0.0
    config.training.cfg_omega = 1.0
    config.training.cfg_kappa = 0.0
    config.training.sup_cfg_velocity = False
    config.training.sup_cfg_batch = False
    config.training.stopgrad_type = "convex"
    config.training.loss_type = "lsd"
    config.training.norm_eps = 0.01
    config.training.norm_p = 0.0
    config.training.t_min = 0.0
    config.training.t_max = 1.0
    config.training.s_min = 0.0
    config.training.s_max = 1.0
    config.training.fixed_t = None
    config.training.fixed_t_prime = None
    config.training.seed = 42
    config.training.ema_facs = [0.999, 0.9999]
    config.training.ndevices = jax.device_count()

    config.base_network_slot = "auto"
    config.posterior_network_slot = "auto"

    config.problem = ml_collections.ConfigDict()
    config.problem.image_dims = (3, 32, 32)
    config.problem.num_classes = 10
    config.problem.target = "cifar10"
    config.problem.dataset_location = dataset_location
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"

    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 512
    config.optimization.diag_fraction = 0.75
    config.optimization.b1 = 0.9
    config.optimization.b2 = 0.999
    config.optimization.eps = 1e-8
    config.optimization.learning_rate = 1e-2
    config.optimization.clip = 1.0
    config.optimization.total_samples = 204_800_000
    config.optimization.total_steps = int(
        config.optimization.total_samples // config.optimization.bs
    )
    config.optimization.decay_steps = 35000
    config.optimization.schedule_type = "sqrt"

    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 9
    config.logging.visual_freq = 5000
    config.logging.save_freq = 5000
    config.logging.wandb_project = "self-distill-diamond-maps"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "diamond_flows")
    config.logging.output_folder = output_folder
    config.logging.fid_freq = 10000
    config.logging.fid_stats_path = datasets_path("cifar10", "cifar_stats.npz")
    config.logging.fid_n_samples = 10000
    config.logging.fid_batch_size = 256
    config.logging.ema_factor = 0.9999

    return config


def _build_flow_map_lsd(
    output_folder: str,
) -> ml_collections.ConfigDict:
    config = _make_base_config(output_folder)
    config.logging.sample_types = ["FLOW_MAP"]
    config.logging.outer_steps = [1]
    config.logging.inner_steps = [1, 2, 4]
    config.network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CIFAR_UNET_KWARGS,
        matching="flow_map",
        rescale=1.0,
        use_bfloat16=False,
    )
    config.logging.wandb_name = "cifar10_flow_map_lsd"
    config.logging.output_name = config.logging.wandb_name
    return config


def _build_glass_diamond_early_stop_lsd(
    output_folder: str,
) -> ml_collections.ConfigDict:
    config = _make_base_config(output_folder)
    config.training.supervise_type = "glass"
    config.training.fixed_t_prime = 1.0
    config.training.s_min = 0.01
    config.training.s_max = 0.99
    config.logging.sample_types = ["DIAMOND_EARLY_STOP"]
    config.logging.outer_steps = [1, 2]
    config.logging.inner_steps = [1, 2, 4]
    config.network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CIFAR_UNET_KWARGS,
        matching="diamond_map",
        rescale=1.0,
        use_bfloat16=False,
    )
    config.sup_network = build_edm2_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        conditional=config.training.conditional,
        unet_kwargs=CIFAR_UNET_KWARGS,
        matching="flow_map",
        load_path=ckpt_path("cifar10_flow_map_lsd.pkl"),
        rescale=1.0,
        use_bfloat16=False,
        use_glass=True,
    )
    config.logging.wandb_name = "cifar10_glass_diamond_early_stop_lsd"
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
