"""Base CelebA-64 training config helpers."""

from __future__ import annotations

import os

import ml_collections
from common.repo_paths import ckpt_path, datasets_path


FLOW_MAP_CKPT = ckpt_path("CelebA-FlowMap.pkl")
DIAMOND_EARLY_STOP_CKPT = ckpt_path("CelebA-DiamondMap.pkl")


def make_base_config(
    output_folder: str,
    *,
    shuffle: bool,
) -> ml_collections.ConfigDict:
    import jax

    config = ml_collections.ConfigDict()
    dataset_location = datasets_path()

    config.training = ml_collections.ConfigDict()
    config.training.supervise_type = "none"
    config.training.shuffle = shuffle
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
    config.problem.image_dims = (3, 64, 64)
    config.problem.num_classes = 0
    config.problem.target = "celeb_a"
    config.problem.dataset_location = dataset_location
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"

    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 256
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
    config.logging.plot_bs = 10
    config.logging.visual_freq = 20000
    config.logging.save_freq = 20000
    config.logging.wandb_project = "self-distill-diamond-maps"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "diamond_flows")
    config.logging.output_folder = output_folder
    config.logging.fid_freq = config.logging.visual_freq
    config.logging.fid_stats_path = datasets_path("celeb_a", "celeba_stats.npz")
    config.logging.fid_n_samples = 10000
    config.logging.fid_batch_size = 256
    config.logging.ema_factor = 0.9999

    return config


def configure_fixed_early_stop_diamond_training(
    config: ml_collections.ConfigDict,
    *,
    supervise_type: str,
) -> None:
    config.training.supervise_type = supervise_type
    config.training.fixed_t_prime = 1.0
    config.training.s_min = 0.01
    config.training.s_max = 0.99
