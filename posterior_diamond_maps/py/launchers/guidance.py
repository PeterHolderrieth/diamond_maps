"""Run inverse-problem and prompt-alignment guidance experiments."""

# isort: off
import os
import sys

# Set up path for imports FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

from common import latent_utils

latent_utils.force_xla_gpu_deterministic_ops()
# isort: on

import importlib
import json
import gc
from typing import Any, Dict, Optional

import click
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from common import (
    datasets,
    metric_utils,
    prompt_reward_runtime,
    repo_paths,
    sampling,
)
from common.inverse_problems import make_inverse_problem
from common.sampling import SampleType, make_sample_plot
from common import cli_common
from launchers.learn import setup_state

SUPPORTED_PROMPT_REWARDS = prompt_reward_runtime.supported_prompt_rewards()


def _build_run_metadata(
    *,
    cfg_path: str,
    guidance_config,
    base_network_slot: str,
    posterior_network_slot: str,
    base_outer_steps: int,
    base_inner_steps: int,
    mc_samples_schedule: str,
    mc_inner_steps_schedule: str,
    guidance_scales_arg: str,
) -> dict[str, object]:
    return {
        "base_inner_steps": int(base_inner_steps),
        "base_network_slot": base_network_slot,
        "base_outer_steps": int(base_outer_steps),
        "cfg_path": cfg_path,
        "guidance": _jsonify_config_value(guidance_config),
        "guidance_scale": cli_common.normalize_csv_arg(guidance_scales_arg, sep=","),
        "mc_inner_steps_schedule": cli_common.normalize_csv_arg(
            mc_inner_steps_schedule, sep=","
        ),
        "mc_samples_schedule": cli_common.normalize_csv_arg(
            mc_samples_schedule, sep=","
        ),
        "posterior_network_slot": posterior_network_slot,
    }


def _jsonify_config_value(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return {key: _jsonify_config_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_config_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_guidance_settings(cfg) -> dict[str, Any]:
    guidance = cfg.guidance
    problem = str(guidance.problem_type).lower()
    base_sample_type = SampleType[str(guidance.base_sample_type).upper()]
    posterior_sample_type = SampleType[str(guidance.posterior_sample_type).upper()]
    base_network_slot = str(cfg.base_network_slot).lower()
    posterior_network_slot = str(cfg.posterior_network_slot).lower()
    guidance_scale_schedule = str(guidance.guidance_scale_schedule).lower()

    ema_factor = float(guidance.ema_factor)
    metric_samples = int(guidance.metric_samples)
    metric_bs = int(guidance.metric_bs)
    plot_bs = int(guidance.plot_bs)
    comp_lpips = bool(guidance.comp_lpips)
    comp_kid = bool(guidance.comp_kid)
    seed_override = guidance.seed_override
    if seed_override is not None:
        seed_override = int(seed_override)

    mask_prob = guidance.mask_prob
    if mask_prob is not None:
        mask_prob = float(mask_prob)
    noise_std = guidance.noise_std
    if noise_std is not None:
        noise_std = float(noise_std)
    sr_factor = guidance.sr_factor
    if sr_factor is not None:
        sr_factor = int(sr_factor)

    return {
        "guidance_config": guidance,
        "problem": problem,
        "mask_prob": mask_prob,
        "noise_std": noise_std,
        "sr_factor": sr_factor,
        "latent_split": guidance.latent_split,
        "base_sample_type": base_sample_type,
        "posterior_sample_type": posterior_sample_type,
        "base_network_slot": base_network_slot,
        "posterior_network_slot": posterior_network_slot,
        "guidance_scale_schedule": guidance_scale_schedule,
        "ema_factor": ema_factor,
        "metric_samples": metric_samples,
        "metric_bs": metric_bs,
        "plot_bs": plot_bs,
        "comp_lpips": comp_lpips,
        "comp_kid": comp_kid,
        "seed_override": seed_override,
    }


def _build_model_params(
    cfg,
    statics,
    train_state,
    base_sample_type: SampleType,
    posterior_sample_type: SampleType,
    ema_factor: float,
    *,
    base_network_slot: Optional[str] = None,
    posterior_network_slot: Optional[str] = None,
    reward_params: Optional[Dict] = None,
) -> Dict[str, Dict]:
    main_params = train_state.ema_params[ema_factor]
    sup_params = statics.sup_params

    model_params = {
        "base": (
            main_params
            if sampling.use_main_network(
                cfg, base_sample_type, network_slot=base_network_slot
            )
            else sup_params
        ),
        "posterior": (
            main_params
            if sampling.use_main_network(
                cfg, posterior_sample_type, network_slot=posterior_network_slot
            )
            else sup_params
        ),
    }
    if reward_params is not None:
        model_params["reward"] = reward_params
    return model_params


def _make_inverse_init_fn(cfg, statics, problem):
    decode_fn = statics.decode_fn

    def init_fn(prng_key, bs, sharded):
        prng_key, x0_key, meas_key, step_key = jax.random.split(prng_key, 4)
        batch_x0s = statics.sample_rho0(bs, x0_key)
        batch_x1s, batch_labels = sampling.collect_x1_batch(statics.ds, bs)
        batch_x1s = jnp.asarray(batch_x1s, dtype=jnp.float32)
        batch_x1s = latent_utils.maybe_decode_latents_chunked(
            cfg,
            batch_x1s,
            chunk_size=latent_utils.LATENT_DECODE_CHUNK,
            decode_fn=decode_fn,
        )
        batch_measurement = problem.batch_make_measurement(batch_x1s, meas_key)

        if cfg.training.conditional:
            if batch_labels is None:
                raise ValueError("Dataset labels required for conditional sampling.")
            batch_labels = jnp.asarray(batch_labels, dtype=jnp.int32)
        else:
            batch_labels = None

        batch_subkeys = jax.random.split(step_key, num=bs)

        return sampling.make_groundtruth_init(
            cfg,
            sharded=sharded,
            batch_x_init=batch_x0s,
            batch_prng_key=batch_subkeys,
            batch_x0s=batch_x0s,
            batch_x1s=batch_x1s,
            batch_measurement=batch_measurement,
            batch_labels=batch_labels,
        )

    return init_fn


def _make_prompt_init_fn(cfg, statics, class_label: Optional[int], prompt_data):
    def init_fn(prng_key, bs, sharded):
        prng_key, x0_key, step_key = jax.random.split(prng_key, 3)
        batch_x0s = statics.sample_rho0(bs, x0_key)

        if cfg.training.conditional:
            if class_label is None:
                raise ValueError(
                    "Prompt alignment requires class labels from the prompt set."
                )
            batch_labels = jnp.full((bs,), class_label, dtype=jnp.int32)
        else:
            batch_labels = None

        batch_subkeys = jax.random.split(step_key, num=bs)
        batch_measurement = prompt_reward_runtime.make_prompt_measurement(
            prompt_data,
            bs,
            cfg=cfg,
            replicate=sharded,
        )

        return sampling.make_groundtruth_init(
            cfg,
            sharded=sharded,
            batch_x_init=batch_x0s,
            batch_prng_key=batch_subkeys,
            batch_x0s=batch_x0s,
            batch_x1s=None,
            batch_measurement=batch_measurement,
            batch_labels=batch_labels,
        )

    return init_fn


def _make_inverse_reward_fn(cfg, inverse_problem, decode_fn):
    def reward_fn(_variables, x: jnp.ndarray, measurement):
        x_decoded = latent_utils.maybe_decode_latents_chunked(
            cfg, x, chunk_size=latent_utils.LATENT_DECODE_CHUNK, decode_fn=decode_fn
        )
        return inverse_problem.reward(x_decoded, measurement)

    return reward_fn


def _sample_visual_batch(
    cfg,
    sample_fn,
    prng_key,
    variables,
    plot_bs: int,
    *,
    problem: str,
    return_posterior: bool,
):
    use_sharded = (
        problem == "prompt_alignment"
        and cfg.training.ndevices > 1
        and plot_bs >= cfg.training.ndevices
    )
    return sample_fn(
        prng_key,
        variables,
        plot_bs,
        return_traj=False,
        sharded=use_sharded,
        return_posterior=return_posterior,
    )


def save_final_visuals(
    cfg,
    decode_fn,
    sample_batch,
    out_path: str,
):
    visual_rows = []
    batch_x1s = sample_batch.batch_x1s
    batch_measurement = sample_batch.batch_measurement
    if batch_x1s is not None:
        if batch_measurement is None or "y_vis" not in batch_measurement:
            raise ValueError("Expected measurement alongside groundtruth samples.")
        visual_rows.append(
            np.asarray(
                latent_utils.decode_batch_to_nhwc(cfg, batch_x1s, decode_fn),
                dtype=np.float32,
            )
        )
        visual_rows.append(
            np.asarray(
                jnp.transpose(batch_measurement["y_vis"], (0, 2, 3, 1)),
                dtype=np.float32,
            )
        )
    elif batch_measurement is not None and "y_vis" in batch_measurement:
        raise ValueError("Expected groundtruth samples alongside visual measurements.")

    batch_x_final_vis = np.asarray(
        latent_utils.decode_batch_to_nhwc(cfg, sample_batch.batch_x_final, decode_fn),
        dtype=np.float32,
    )
    visual_rows.append(batch_x_final_vis)

    xfinals = np.stack(visual_rows, axis=0)
    fig = make_sample_plot(
        nrows=xfinals.shape[0],
        ncols=xfinals.shape[1],
        xfinals=xfinals,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return batch_x_final_vis


def save_posterior_visuals(
    cfg,
    decode_fn,
    batch_x_final_vis: np.ndarray,
    batch_x_t_trace: jnp.ndarray,
    batch_posterior_trace: jnp.ndarray,
    step_ts: jnp.ndarray,
    posterior_out_dir: str,
):
    posterior_dir = os.path.join(posterior_out_dir, "posterior")
    os.makedirs(posterior_dir, exist_ok=True)

    batch_x_t_vis = np.asarray(
        latent_utils.decode_batch_to_nhwc(cfg, batch_x_t_trace, decode_fn),
        dtype=np.float32,
    )
    posterior_vis = np.asarray(
        latent_utils.decode_batch_to_nhwc(cfg, batch_posterior_trace, decode_fn),
        dtype=np.float32,
    )

    num_outer_steps = len(step_ts) - 1
    if batch_x_t_vis.shape[0] != num_outer_steps:
        raise ValueError("batch_x_t_trace does not match the timestep schedule.")
    if posterior_vis.shape[0] != num_outer_steps:
        raise ValueError("batch_posterior_trace does not match the timestep schedule.")

    plot_bs = batch_x_final_vis.shape[0]
    mc_samples = posterior_vis.shape[2]
    row_labels = ["x_t"] + [f"posterior_{idx:02d}" for idx in range(mc_samples)]
    ncols = len(step_ts)

    for sample_idx in range(plot_bs):
        fig, axes = plt.subplots(
            len(row_labels),
            ncols,
            figsize=(2.0 * ncols, 2.0 * len(row_labels)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for col_idx in range(ncols):
            is_final_col = col_idx == ncols - 1
            for row_idx, row_label in enumerate(row_labels):
                ax = axes[row_idx, col_idx]
                if row_idx == 0:
                    image = (
                        batch_x_final_vis[sample_idx]
                        if is_final_col
                        else batch_x_t_vis[col_idx, sample_idx]
                    )
                    ax.imshow(datasets.unnormalize_image(image))
                elif is_final_col:
                    ax.axis("off")
                    continue
                else:
                    image = posterior_vis[col_idx, sample_idx, row_idx - 1]
                    ax.imshow(datasets.unnormalize_image(image))
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(row_label)

        out_path = os.path.join(
            posterior_dir,
            f"sample_{sample_idx:02d}.png",
        )
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved posterior visualization to {out_path}")


def _make_metric_sample_fn(cfg, sample_fn, decode_fn):
    def metric_sample_fn(prng_key, variables, bs, sharded):
        sample_batch = sample_fn(
            prng_key, variables, bs, return_traj=False, sharded=sharded
        )
        batch_x1s = sample_batch.batch_x1s
        if batch_x1s is None:
            raise ValueError("Groundtruth samples required for pair metrics.")
        batch_x1s = latent_utils.decode_batch_to_nhwc(
            cfg,
            batch_x1s,
            decode_fn,
            unnormalize=True,
        )
        batch_x_final = latent_utils.decode_batch_to_nhwc(
            cfg,
            sample_batch.batch_x_final,
            decode_fn,
            unnormalize=True,
        )
        return batch_x1s, batch_x_final

    return metric_sample_fn


def _make_prompt_metric_sample_fn(cfg, sample_fn, decode_fn):
    def sample_image_fn(prng_key, variables, bs, sharded):
        sample_batch = sample_fn(
            prng_key, variables, bs, return_traj=False, sharded=sharded
        )
        return latent_utils.decode_batch_to_nhwc(
            cfg,
            sample_batch.batch_x_final,
            decode_fn,
            unnormalize=True,
        )

    return sample_image_fn


def _parse_schedule_arrays(
    mc_samples_schedule: str,
    mc_inner_steps_schedule: str,
    guidance_scales: str,
):
    mc_samples = jnp.array(
        cli_common.parse_csv_ints(
            mc_samples_schedule, option_name="mc_samples_schedule"
        ),
        dtype=jnp.int32,
    )
    mc_inner_steps = jnp.array(
        cli_common.parse_csv_ints(
            mc_inner_steps_schedule, option_name="mc_inner_steps_schedule"
        ),
        dtype=jnp.int32,
    )
    guidance_scales = jnp.array(
        cli_common.parse_csv_floats(guidance_scales, option_name="guidance_scales"),
        dtype=jnp.float32,
    )
    return mc_samples, mc_inner_steps, guidance_scales


def _prepare_guidance_config(cfg, guidance_settings: dict[str, Any]) -> None:
    problem = guidance_settings["problem"]
    if problem == "prompt_alignment" or not guidance_settings["comp_kid"]:
        cfg.logging.fid_freq = 0
    if guidance_settings["seed_override"] is not None:
        cfg.training.seed = guidance_settings["seed_override"]


def _setup_guidance_state(cfg_path: str, slurm_id, run_dir: str):
    cfg = importlib.import_module(cfg_path).get_config(slurm_id, run_dir)
    guidance_settings = _load_guidance_settings(cfg)
    _prepare_guidance_config(cfg, guidance_settings)
    prng_key = jax.random.PRNGKey(cfg.training.seed)
    cfg, statics, train_state, prng_key = setup_state(cfg, prng_key)
    return cfg, guidance_settings, statics, train_state, prng_key


def _write_run_metadata(
    out_dir: str,
    *,
    cfg_path: str,
    guidance_settings: dict[str, Any],
    base_outer_steps: int,
    base_inner_steps: int,
    mc_samples_schedule: str,
    mc_inner_steps_schedule: str,
    guidance_scales_arg: str,
    indent=None,
    sort_keys: bool = False,
    final_newline: bool = False,
) -> None:
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            _build_run_metadata(
                cfg_path=cfg_path,
                guidance_config=guidance_settings["guidance_config"],
                base_network_slot=guidance_settings["base_network_slot"],
                posterior_network_slot=guidance_settings["posterior_network_slot"],
                base_outer_steps=base_outer_steps,
                base_inner_steps=base_inner_steps,
                mc_samples_schedule=mc_samples_schedule,
                mc_inner_steps_schedule=mc_inner_steps_schedule,
                guidance_scales_arg=guidance_scales_arg,
            ),
            f,
            indent=indent,
            sort_keys=sort_keys,
        )
        if final_newline:
            f.write("\n")


def _create_guided_sampler(
    *,
    cfg,
    statics,
    reward_fn,
    guidance_settings: dict[str, Any],
    base_outer_steps: int,
    base_inner_steps: int,
    mc_samples,
    mc_inner_steps,
    guidance_scales,
):
    mc_samples = sampling.expand_schedule(mc_samples, base_outer_steps)
    mc_inner_steps = sampling.expand_schedule(mc_inner_steps, base_outer_steps)
    guidance_scales = sampling.expand_schedule(guidance_scales, base_outer_steps)
    batch_guidance_fn = sampling.make_batch_guidance_fn(
        cfg=cfg,
        statics=statics,
        reward_fn=reward_fn,
        guidance_scales=guidance_scales,
        guidance_scale_schedule=guidance_settings["guidance_scale_schedule"],
        mc_samples=mc_samples,
        mc_inner_steps=mc_inner_steps,
        posterior_sample_type=guidance_settings["posterior_sample_type"],
        posterior_network_slot=guidance_settings["posterior_network_slot"],
    )
    spec = sampling.build_guided_sampler_spec(
        cfg=cfg,
        statics=statics,
        base_sample_type=guidance_settings["base_sample_type"],
        outer_steps=base_outer_steps,
        inner_steps=base_inner_steps,
        batch_guidance_fn=batch_guidance_fn,
        base_network_slot=guidance_settings["base_network_slot"],
    )
    return spec, sampling.create_batched_sampler(spec)


def _build_guidance_model_params(
    cfg,
    statics,
    train_state,
    guidance_settings: dict[str, Any],
    *,
    reward_params=None,
):
    return _build_model_params(
        cfg,
        statics,
        train_state,
        guidance_settings["base_sample_type"],
        guidance_settings["posterior_sample_type"],
        guidance_settings["ema_factor"],
        base_network_slot=guidance_settings["base_network_slot"],
        posterior_network_slot=guidance_settings["posterior_network_slot"],
        reward_params=reward_params,
    )


def _require_posterior_visualization_data(sample_batch) -> None:
    if (
        sample_batch.batch_x_t_trace is None
        or sample_batch.batch_posterior_trace is None
    ):
        raise ValueError(
            "Expected posterior visualization data when visualize_posterior=True."
        )


def _maybe_save_visualization(
    *,
    cfg,
    statics,
    sample_fn,
    prng_key,
    model_params,
    plot_bs: int,
    problem: str,
    visualize: bool,
    visualize_posterior: bool,
    spec,
    out_dir: str,
):
    if not visualize:
        return prng_key

    vis_path = os.path.join(out_dir, "visual.png")
    prng_key, vis_key = jax.random.split(prng_key)
    sample_batch = _sample_visual_batch(
        cfg,
        sample_fn,
        vis_key,
        model_params,
        plot_bs,
        problem=problem,
        return_posterior=visualize_posterior,
    )
    batch_x_final_vis = save_final_visuals(
        cfg,
        statics.decode_fn,
        sample_batch,
        vis_path,
    )
    if visualize_posterior:
        _require_posterior_visualization_data(sample_batch)
        save_posterior_visuals(
            cfg=cfg,
            decode_fn=statics.decode_fn,
            batch_x_final_vis=batch_x_final_vis,
            batch_x_t_trace=sample_batch.batch_x_t_trace,
            batch_posterior_trace=sample_batch.batch_posterior_trace,
            step_ts=spec.ts,
            posterior_out_dir=out_dir,
        )
    print(f"Saved visualization to {vis_path}")
    return prng_key


def _run_prompt_alignment(
    *,
    cfg,
    statics,
    train_state,
    prng_key,
    run_dir: str,
    cfg_path: str,
    guidance_settings: dict[str, Any],
    prompt_set: str,
    prompt_index: Optional[int],
    prompt_reward: str,
    prompt_metric_rewards,
    base_outer_steps: int,
    base_inner_steps: int,
    mc_samples_schedule: str,
    mc_inner_steps_schedule: str,
    guidance_scales_arg: str,
    mc_samples,
    mc_inner_steps,
    guidance_scales,
    visualize: bool,
    visualize_posterior: bool,
) -> None:
    prompt_entries = cli_common.selected_prompt_entries(prompt_set, prompt_index)
    reward_runtime = prompt_reward_runtime.build_prompt_reward_runtime(
        prompt_reward, prompt_entries
    )
    reward_fn = prompt_reward_runtime.make_latent_prompt_reward_fn(
        cfg, statics.decode_fn, reward_runtime.score_fn
    )
    spec, sampler = _create_guided_sampler(
        cfg=cfg,
        statics=statics,
        reward_fn=reward_fn,
        guidance_settings=guidance_settings,
        base_outer_steps=base_outer_steps,
        base_inner_steps=base_inner_steps,
        mc_samples=mc_samples,
        mc_inner_steps=mc_inner_steps,
        guidance_scales=guidance_scales,
    )
    model_params = _build_guidance_model_params(
        cfg,
        statics,
        train_state,
        guidance_settings,
        reward_params=reward_runtime.reward_params,
    )
    del train_state
    gc.collect()
    needs_metrics = guidance_settings["metric_samples"] > 0

    _write_run_metadata(
        run_dir,
        cfg_path=cfg_path,
        guidance_settings=guidance_settings,
        base_outer_steps=base_outer_steps,
        base_inner_steps=base_inner_steps,
        mc_samples_schedule=mc_samples_schedule,
        mc_inner_steps_schedule=mc_inner_steps_schedule,
        guidance_scales_arg=guidance_scales_arg,
    )

    for current_prompt_index, (prompt, class_label) in prompt_entries:
        prompt_dir = os.path.join(run_dir, f"{prompt_set}_{current_prompt_index}")
        os.makedirs(prompt_dir, exist_ok=True)
        init_fn = _make_prompt_init_fn(
            cfg,
            statics,
            class_label,
            reward_runtime.prompt_data_by_index[current_prompt_index],
        )
        sample_fn = sampling.create_groundtruth_sample_fn(sampler, init_fn)
        _write_run_metadata(
            prompt_dir,
            cfg_path=cfg_path,
            guidance_settings=guidance_settings,
            base_outer_steps=base_outer_steps,
            base_inner_steps=base_inner_steps,
            mc_samples_schedule=mc_samples_schedule,
            mc_inner_steps_schedule=mc_inner_steps_schedule,
            guidance_scales_arg=guidance_scales_arg,
        )
        prng_key = _maybe_save_visualization(
            cfg=cfg,
            statics=statics,
            sample_fn=sample_fn,
            prng_key=prng_key,
            model_params=model_params,
            plot_bs=guidance_settings["plot_bs"],
            problem=guidance_settings["problem"],
            visualize=visualize,
            visualize_posterior=visualize_posterior,
            spec=spec,
            out_dir=prompt_dir,
        )

        metric_vals = {}
        if needs_metrics:
            prng_key, metric_key = jax.random.split(prng_key)
            sample_image_fn = _make_prompt_metric_sample_fn(
                cfg, sample_fn, statics.decode_fn
            )
            print("[prompt_metric_phase] inline_metrics_start", flush=True)
            metric_vals = metric_utils.calc_prompt_alignment_metrics(
                sample_image_fn,
                metric_key,
                model_params,
                prompt,
                n_samples=guidance_settings["metric_samples"],
                bs=guidance_settings["metric_bs"],
                prompt_metric_rewards=prompt_metric_rewards,
            )
            for metric_name, metric_value in metric_vals.items():
                print(f"{metric_name}={metric_value}")
        with open(os.path.join(prompt_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metric_vals, f, indent=2, sort_keys=True)
            f.write("\n")


def _run_inverse_problem(
    *,
    cfg,
    statics,
    train_state,
    prng_key,
    run_dir: str,
    cfg_path: str,
    guidance_settings: dict[str, Any],
    base_outer_steps: int,
    base_inner_steps: int,
    mc_samples_schedule: str,
    mc_inner_steps_schedule: str,
    guidance_scales_arg: str,
    mc_samples,
    mc_inner_steps,
    guidance_scales,
    visualize: bool,
    visualize_posterior: bool,
    **_unused,
) -> None:
    inverse_problem = make_inverse_problem(
        problem_type=guidance_settings["problem"],
        mask_prob=guidance_settings["mask_prob"],
        noise_std=guidance_settings["noise_std"],
        sr_factor=guidance_settings["sr_factor"],
    )
    reward_fn = _make_inverse_reward_fn(cfg, inverse_problem, statics.decode_fn)
    init_fn = _make_inverse_init_fn(cfg, statics, inverse_problem)
    model_params = _build_guidance_model_params(
        cfg,
        statics,
        train_state,
        guidance_settings,
    )
    del train_state
    gc.collect()
    spec, sampler = _create_guided_sampler(
        cfg=cfg,
        statics=statics,
        reward_fn=reward_fn,
        guidance_settings=guidance_settings,
        base_outer_steps=base_outer_steps,
        base_inner_steps=base_inner_steps,
        mc_samples=mc_samples,
        mc_inner_steps=mc_inner_steps,
        guidance_scales=guidance_scales,
    )
    sample_fn = sampling.create_groundtruth_sample_fn(sampler, init_fn)
    _write_run_metadata(
        run_dir,
        cfg_path=cfg_path,
        guidance_settings=guidance_settings,
        base_outer_steps=base_outer_steps,
        base_inner_steps=base_inner_steps,
        mc_samples_schedule=mc_samples_schedule,
        mc_inner_steps_schedule=mc_inner_steps_schedule,
        guidance_scales_arg=guidance_scales_arg,
        indent=2,
        sort_keys=True,
        final_newline=True,
    )

    prng_key = _maybe_save_visualization(
        cfg=cfg,
        statics=statics,
        sample_fn=sample_fn,
        prng_key=prng_key,
        model_params=model_params,
        plot_bs=guidance_settings["plot_bs"],
        problem=guidance_settings["problem"],
        visualize=visualize,
        visualize_posterior=visualize_posterior,
        spec=spec,
        out_dir=run_dir,
    )

    needs_metrics = guidance_settings["comp_lpips"] or guidance_settings["comp_kid"]
    if needs_metrics:
        lpips_j_model = None
        lpips_j_params = None
        if guidance_settings["comp_lpips"]:
            lpips_j_model, lpips_j_params = metric_utils.load_lpips_j_model()
        prng_key, metric_key = jax.random.split(prng_key)
        sample_pair_fn = _make_metric_sample_fn(cfg, sample_fn, statics.decode_fn)
        lpips_j_val, kid_val = metric_utils.calc_guidance_metrics(
            cfg,
            lpips_j_model,
            lpips_j_params,
            statics.inception_fn,
            sample_pair_fn,
            metric_key,
            model_params,
            n_samples=guidance_settings["metric_samples"],
            bs=guidance_settings["metric_bs"],
        )
        if lpips_j_val is not None:
            print(f"LPIPS_j={lpips_j_val}")
        if kid_val is not None:
            print(f"KID={kid_val}")


_PROBLEM_RUNNERS = {
    "inpainting": _run_inverse_problem,
    "super_resolution": _run_inverse_problem,
    "prompt_alignment": _run_prompt_alignment,
}


@click.command()
@click.option("--cfg_path", required=True)
@click.option("--output_folder", default=repo_paths.repo_path("outputs"))
@click.option("--slurm_id", type=int, default=0)
@click.option("--prompt_set", type=str, default=None)
@click.option("--prompt_index", type=int, default=None)
@click.option(
    "--prompt_reward",
    type=click.Choice(list(SUPPORTED_PROMPT_REWARDS), case_sensitive=False),
    default="imagereward",
)
@click.option(
    "--prompt_metric_rewards",
    type=str,
    default=None,
    help="Comma-separated prompt rewards to evaluate for prompt metrics. Defaults to --prompt_reward.",
)
@click.option("--base_outer_steps", type=int, default=50)
@click.option("--base_inner_steps", type=int, default=1)
@click.option("--mc_samples_schedule", type=str, default="10")
@click.option("--mc_inner_steps_schedule", type=str, default="1")
@click.option("--guidance_scales", type=str, default="0.1")
@click.option("--visualize_posterior", is_flag=True)
@click.option("--visualize", is_flag=True)
def main(
    cfg_path,
    output_folder,
    slurm_id,
    prompt_set,
    prompt_index,
    prompt_reward,
    prompt_metric_rewards,
    base_outer_steps,
    base_inner_steps,
    mc_samples_schedule,
    mc_inner_steps_schedule,
    guidance_scales,
    visualize_posterior,
    visualize,
):
    guidance_scales_arg = guidance_scales
    prompt_reward = prompt_reward.lower()
    prompt_metric_rewards = metric_utils.normalize_prompt_metric_rewards(
        prompt_metric_rewards, default_reward=prompt_reward
    )
    mc_samples, mc_inner_steps, guidance_scales = _parse_schedule_arrays(
        mc_samples_schedule,
        mc_inner_steps_schedule,
        guidance_scales,
    )
    os.makedirs(output_folder, exist_ok=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = os.path.join(output_folder, f"{slurm_job_id}_guidance")
    os.makedirs(run_dir, exist_ok=False)

    cfg, guidance_settings, statics, train_state, prng_key = _setup_guidance_state(
        cfg_path,
        slurm_id,
        run_dir,
    )
    _PROBLEM_RUNNERS[guidance_settings["problem"]](
        cfg=cfg,
        statics=statics,
        train_state=train_state,
        prng_key=prng_key,
        run_dir=run_dir,
        cfg_path=cfg_path,
        guidance_settings=guidance_settings,
        prompt_set=prompt_set,
        prompt_index=prompt_index,
        prompt_reward=prompt_reward,
        prompt_metric_rewards=prompt_metric_rewards,
        base_outer_steps=base_outer_steps,
        base_inner_steps=base_inner_steps,
        mc_samples_schedule=mc_samples_schedule,
        mc_inner_steps_schedule=mc_inner_steps_schedule,
        guidance_scales_arg=guidance_scales_arg,
        mc_samples=mc_samples,
        mc_inner_steps=mc_inner_steps,
        guidance_scales=guidance_scales,
        visualize=visualize,
        visualize_posterior=visualize_posterior,
    )


if __name__ == "__main__":
    main()
