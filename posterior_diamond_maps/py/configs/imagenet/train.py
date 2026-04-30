"""ImageNet latent training config variants."""

from __future__ import annotations

import os

import ml_collections

from configs.imagenet import common


def _build_diamond_early_stop(
    output_folder: str,
) -> ml_collections.ConfigDict:
    import jax

    config = ml_collections.ConfigDict()

    config.training = ml_collections.ConfigDict()
    config.training.supervise_type = "glass"
    config.training.shuffle = True
    config.training.conditional = True
    config.training.class_dropout = 0.1
    config.training.cfg_omega = 3.0
    config.training.cfg_kappa = 0.0
    config.training.sup_cfg_velocity = True
    config.training.sup_cfg_batch = True
    config.training.stopgrad_type = "convex"
    config.training.loss_type = "lsd"
    config.training.norm_eps = 0.01
    config.training.norm_p = 0.0
    config.training.t_min = 0.0
    config.training.t_max = 1.0
    config.training.s_min = 0.01
    config.training.s_max = 0.99
    config.training.seed = 42
    config.training.fixed_t = None
    config.training.fixed_t_prime = 1.0
    config.training.ema_facs = [0.9999]
    config.training.ndevices = jax.device_count()

    config.base_network_slot = "auto"
    config.posterior_network_slot = "auto"

    config.problem = ml_collections.ConfigDict()
    config.problem.image_dims = (4, 32, 32)
    config.problem.num_classes = 1000
    config.problem.target = "imagenet_latent_256"
    config.problem.dataset_location = common.IMAGENET_LATENT_DATASET
    config.problem.interp_type = "linear"
    config.problem.base = "gaussian"
    config.problem.latent_scale = 0.18215
    config.problem.latent_vae_type = "ema"
    config.problem.latent_hflip = False
    config.problem.latent_split = "train"
    config.problem.latent_sample = True
    config.problem.latent_spatial_size = 32

    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 128
    config.optimization.diag_fraction = 0.75
    config.optimization.b1 = 0.9
    config.optimization.b2 = 0.999
    config.optimization.eps = 1e-8
    config.optimization.learning_rate = 1e-4
    config.optimization.clip = 1.0
    config.optimization.total_steps = 1_000_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = config.optimization.total_steps // 2
    config.optimization.schedule_type = "constant"

    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 4
    config.logging.visual_freq = 10_000_000
    config.logging.save_freq = 10_000
    config.logging.sample_types = ["DIAMOND_EARLY_STOP"]
    config.logging.outer_steps = [1]
    config.logging.inner_steps = [1]
    config.logging.wandb_project = "self-distill-diamond-maps"
    config.logging.wandb_name = (
        "imagenet256_latent32_glass_diamond_early_stop_sit_b2_l2_cfg3_lsd"
    )
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "diamond_flows")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name
    config.logging.fid_freq = config.logging.visual_freq
    config.logging.fid_stats_path = common.IMAGENET_FID_STATS
    config.logging.fid_n_samples = 10000
    config.logging.fid_batch_size = 64
    config.logging.ema_factor = 0.9999

    config.network = common.build_sit_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        matching="diamond_map",
        model_name="SiT_B_2",
        init_from=common.MEANFLOW_SIT_B2_FLOW_MAP_CKPT,
        reverse_time=True,
        use_cfg_token=True,
        compute_dtype="bfloat16",
        param_dtype="float32",
        reset_optimizer=False,
        use_weight=False,
        rescale=1.0,
        learn_sigma=False,
        mix_time_embed=True,
        mix_image_embed=True,
        use_glass=False,
    )
    config.sup_network = common.build_sit_network(
        image_dims=config.problem.image_dims,
        num_classes=config.problem.num_classes,
        matching="flow",
        model_name="SiT_L_2",
        load_path=common.SIT_L2_FLOW_CKPT,
        reverse_time=False,
        use_cfg_token=True,
        sit_cfg_scale=3.0,
        sit_cfg_channels=3,
        compute_dtype="bfloat16",
        param_dtype="float32",
        reset_optimizer=False,
        use_weight=False,
        rescale=1.0,
        learn_sigma=True,
        use_glass=True,
    )

    return config


VARIANTS = (
    {
        "name": "glass_diamond_early_stop_b2_from_meanflow_b2_l2_cfg3",
        "builder": _build_diamond_early_stop,
    },
)


def get_config(
    slurm_id: int, output_folder: str
) -> ml_collections.ConfigDict:
    variant = VARIANTS[slurm_id]
    return variant["builder"](output_folder)
