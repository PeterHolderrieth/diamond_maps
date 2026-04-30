"""ImageNet latent guidance config variants."""

from __future__ import annotations

import os

import ml_collections

from configs.imagenet import common


def _build_guidance_config(
    *,
    problem_type: str,
    posterior_sample_type: str,
    metric_samples: int,
    metric_bs: int,
    comp_lpips: bool,
    comp_kid: bool,
    plot_bs: int,
    noise_std: float | None = None,
    sr_factor: int | None = None,
) -> ml_collections.ConfigDict:
    guidance = ml_collections.ConfigDict()
    guidance.problem_type = problem_type
    guidance.mask_prob = None
    guidance.noise_std = noise_std
    guidance.sr_factor = sr_factor
    guidance.latent_split = "validation"

    guidance.base_sample_type = "FLOW"
    guidance.posterior_sample_type = posterior_sample_type.upper()

    guidance.guidance_scale_schedule = "auto"
    guidance.ema_factor = 0.9999
    guidance.metric_samples = metric_samples
    guidance.metric_bs = metric_bs
    guidance.comp_lpips = comp_lpips
    guidance.comp_kid = comp_kid
    guidance.plot_bs = plot_bs
    guidance.seed_override = 42
    return guidance


def _replace_config(
    target: ml_collections.ConfigDict, source: ml_collections.ConfigDict
) -> None:
    for key in list(target.keys()):
        del target[key]
    target.update(source)


def configure_guidance_base_network(
    network_cfg: ml_collections.ConfigDict,
    sample_type: str,
) -> None:
    if sample_type in {common.FLOW, common.GLASS}:
        configured = common.build_sit_network(
            image_dims=tuple(network_cfg.input_dims),
            num_classes=network_cfg.label_dim,
            matching="flow",
            model_name="SiT_B_2",
            load_path=common.SIT_B2_FLOW_CKPT,
            init_from="",
            reverse_time=False,
            use_cfg_token=True,
            sit_cfg_scale=4.0,
            sit_cfg_channels=3,
            compute_dtype=network_cfg.compute_dtype,
            param_dtype=network_cfg.param_dtype,
            reset_optimizer=network_cfg.reset_optimizer,
            use_weight=network_cfg.use_weight,
            rescale=1.0,
            learn_sigma=True,
            use_glass=True,
        )
    elif sample_type == common.FLOW_MAP:
        configured = common.build_sit_network(
            image_dims=tuple(network_cfg.input_dims),
            num_classes=network_cfg.label_dim,
            matching="flow_map",
            model_name="SiT_B_2",
            load_path=common.MEANFLOW_SIT_B2_FLOW_MAP_CKPT,
            init_from="",
            reverse_time=True,
            use_cfg_token=True,
            sit_cfg_scale=1.0,
            sit_cfg_channels=3,
            compute_dtype=network_cfg.compute_dtype,
            param_dtype=network_cfg.param_dtype,
            reset_optimizer=network_cfg.reset_optimizer,
            use_weight=network_cfg.use_weight,
            rescale=1.0,
            learn_sigma=False,
            use_glass=False,
        )
    elif sample_type == common.DIAMOND_EARLY_STOP:
        configured = common.build_sit_network(
            image_dims=tuple(network_cfg.input_dims),
            num_classes=network_cfg.label_dim,
            matching="diamond_map",
            model_name="SiT_B_2",
            load_path=common.DIAMOND_MAP_SIT_B2_CKPT,
            init_from="",
            reverse_time=True,
            use_cfg_token=True,
            sit_cfg_scale=1.0,
            sit_cfg_channels=3,
            compute_dtype=network_cfg.compute_dtype,
            param_dtype=network_cfg.param_dtype,
            reset_optimizer=network_cfg.reset_optimizer,
            use_weight=network_cfg.use_weight,
            rescale=1.0,
            learn_sigma=False,
            mix_time_embed=True,
            mix_image_embed=True,
            use_glass=False,
        )
    else:
        raise ValueError(f"Unsupported guidance base sample type: {sample_type}")
    _replace_config(network_cfg, configured)


def configure_guidance_xl2_network(
    network_cfg: ml_collections.ConfigDict,
    sample_type: str,
) -> None:
    if sample_type in {common.FLOW, common.GLASS}:
        configured = common.build_sit_network(
            image_dims=tuple(network_cfg.input_dims),
            num_classes=network_cfg.label_dim,
            matching="flow",
            model_name="SiT_XL_2",
            load_path=common.SIT_XL2_FLOW_CKPT,
            init_from="",
            reverse_time=False,
            use_cfg_token=True,
            sit_cfg_scale=4.0,
            sit_cfg_channels=3,
            compute_dtype=network_cfg.compute_dtype,
            param_dtype=network_cfg.param_dtype,
            reset_optimizer=network_cfg.reset_optimizer,
            use_weight=network_cfg.use_weight,
            rescale=1.0,
            learn_sigma=True,
            use_glass=True,
        )
    elif sample_type == common.FLOW_MAP:
        configured = common.build_sit_network(
            image_dims=tuple(network_cfg.input_dims),
            num_classes=network_cfg.label_dim,
            matching="flow_map",
            model_name="SiT_XL_2",
            load_path=common.MEANFLOW_SIT_XL2_FLOW_MAP_CKPT,
            init_from="",
            reverse_time=True,
            use_cfg_token=True,
            sit_cfg_scale=1.0,
            sit_cfg_channels=3,
            compute_dtype=network_cfg.compute_dtype,
            param_dtype=network_cfg.param_dtype,
            reset_optimizer=network_cfg.reset_optimizer,
            use_weight=network_cfg.use_weight,
            rescale=1.0,
            learn_sigma=True,
            use_glass=False,
        )
    else:
        raise ValueError(f"Unsupported guidance XL2 sample type: {sample_type}")
    _replace_config(network_cfg, configured)


def _build_guidance_variant(
    output_folder: str,
    *,
    suffix: str,
    guidance: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
    import jax

    config = ml_collections.ConfigDict()

    config.training = ml_collections.ConfigDict()
    config.training.supervise_type = "glass"
    config.training.shuffle = False
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

    config.base_network_slot = "main"
    config.posterior_network_slot = "sup"

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
    config.problem.latent_split = guidance.latent_split
    config.problem.latent_sample = False
    config.problem.latent_spatial_size = 32

    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 256
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
        "imagenet256_latent32_glass_diamond_early_stop_sit_b2_l2_cfg3_eval_lsd"
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
        init_from="",
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

    config.training.supervise_type = "none"
    config.training.cfg_omega = 1.0
    config.training.cfg_kappa = 0.0
    config.training.sup_cfg_velocity = False
    config.training.sup_cfg_batch = False

    configure_guidance_xl2_network(config.network, common.FLOW)
    configure_guidance_base_network(
        config.sup_network, guidance.posterior_sample_type.lower()
    )
    config.guidance = guidance
    config.logging.wandb_name = f"{config.logging.wandb_name}_{suffix}"
    config.logging.output_name = config.logging.wandb_name
    return config


def _sr32_builder(posterior_sample_type: str):
    def _builder(output_folder: str) -> ml_collections.ConfigDict:
        return _build_guidance_variant(
            output_folder,
            suffix=f"sr32_xl2_flow_posterior_b2_{posterior_sample_type}",
            guidance=_build_guidance_config(
                problem_type="super_resolution",
                posterior_sample_type=posterior_sample_type,
                noise_std=0.05,
                sr_factor=32,
                metric_samples=64,
                metric_bs=8,
                comp_lpips=True,
                comp_kid=True,
                plot_bs=5,
            ),
        )

    return _builder


def _prompt_metrics_builder(posterior_sample_type: str):
    def _builder(output_folder: str) -> ml_collections.ConfigDict:
        return _build_guidance_variant(
            output_folder,
            suffix=f"prompt_metrics_xl2_flow_posterior_b2_{posterior_sample_type}",
            guidance=_build_guidance_config(
                problem_type="prompt_alignment",
                posterior_sample_type=posterior_sample_type,
                metric_samples=8,
                metric_bs=2,
                comp_lpips=True,
                comp_kid=False,
                plot_bs=5,
            ),
        )

    return _builder


def _prompt_visual_diamond(
    output_folder: str,
) -> ml_collections.ConfigDict:
    return _build_guidance_variant(
        output_folder,
        suffix="prompt_visual_xl2_flow_posterior_b2_diamond_early_stop",
        guidance=_build_guidance_config(
            problem_type="prompt_alignment",
            posterior_sample_type=common.DIAMOND_EARLY_STOP,
            metric_samples=5,
            metric_bs=5,
            comp_lpips=False,
            comp_kid=False,
            plot_bs=5,
        ),
    )


VARIANTS = (
    {
        "name": "sr32_xl2_flow_posterior_b2_flow",
        "builder": _sr32_builder(common.FLOW),
    },
    {
        "name": "sr32_xl2_flow_posterior_b2_flow_map",
        "builder": _sr32_builder(common.FLOW_MAP),
    },
    {
        "name": "sr32_xl2_flow_posterior_b2_diamond_early_stop",
        "builder": _sr32_builder(common.DIAMOND_EARLY_STOP),
    },
    {
        "name": "prompt_metrics_xl2_flow_posterior_b2_flow",
        "builder": _prompt_metrics_builder(common.FLOW),
    },
    {
        "name": "prompt_metrics_xl2_flow_posterior_b2_flow_map",
        "builder": _prompt_metrics_builder(common.FLOW_MAP),
    },
    {
        "name": "prompt_metrics_xl2_flow_posterior_b2_diamond_early_stop",
        "builder": _prompt_metrics_builder(common.DIAMOND_EARLY_STOP),
    },
    {
        "name": "prompt_visual_xl2_flow_posterior_b2_diamond_early_stop",
        "builder": _prompt_visual_diamond,
    },
)


def get_config(
    slurm_id: int, output_folder: str
) -> ml_collections.ConfigDict:
    variant = VARIANTS[slurm_id]
    return variant["builder"](output_folder)
