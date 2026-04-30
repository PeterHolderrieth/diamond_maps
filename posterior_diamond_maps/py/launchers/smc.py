"""Run SMC prompt-alignment sampling and reward evaluation."""

# isort: off
import os
import sys
import csv
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.insert(0, py_dir)

# Suppress TensorFlow logging before any TF imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from common import latent_utils

latent_utils.force_xla_gpu_deterministic_ops()
# isort: on

import importlib

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from common import sampling
from common import prompt_reward_runtime
from common import repo_paths
from common.sampling import SampleType, get_params
import common.dist_utils as dist_utils
from common import cli_common
from launchers.learn import setup_state

DEFAULT_CFG_PATH = "configs.imagenet.smc"

SUPPORTED_BASE_TYPES = (
    "glass",
    "diamond",
    "diamond_early_stop",
    "diamond_renoise",
    "flow_map",
    "flow",
)
SUPPORTED_POSTERIOR_TYPES = ("glass", "flow_map", "diamond", "diamond_early_stop", "flow")
SUPPORTED_PROMPT_REWARDS = ("clip", "imagereward")

_BASE_TYPE_TO_SAMPLE_TYPE = {
    "glass": SampleType.GLASS,
    "diamond": SampleType.DIAMOND_EARLY_STOP,
    "diamond_early_stop": SampleType.DIAMOND_EARLY_STOP,
    "diamond_renoise": SampleType.DIAMOND_RENOISE,
    "flow_map": SampleType.FLOW_MAP,
    "flow": SampleType.FLOW,
}
_POSTERIOR_TYPE_TO_SAMPLE_TYPE = {
    "glass": SampleType.GLASS,
    "flow_map": SampleType.FLOW_MAP,
    "diamond": SampleType.DIAMOND_EARLY_STOP,
    "diamond_early_stop": SampleType.DIAMOND_EARLY_STOP,
    "flow": SampleType.FLOW,
}

def decode_latents_to_pixels(cfg, latents_nchw, decode_fn):
    """Decode latent-space samples to pixel-space images (B,H,W,3) in [0,1]."""
    return latent_utils.decode_batch_to_nhwc(
        cfg,
        latents_nchw,
        decode_fn,
        unnormalize=True,
        chunk_size=latent_utils.LATENT_DECODE_CHUNK,
    )


def _make_batch_prompt_measurement(cfg, prompt_data, bs: int):
    return prompt_reward_runtime.make_prompt_measurement(
        prompt_data,
        bs,
        cfg=cfg,
        replicate=True,
    )


def save_single_image_png(img01_hwc: np.ndarray, out_path: str) -> None:
    img01_hwc = np.clip(img01_hwc, 0.0, 1.0)
    img_u8 = (img01_hwc * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img_u8).save(out_path)


CSV_FIELDS = [
    "slurm_job_id",
    "dataset",
    "mode",
    "reward",
    "prompt_set",
    "prompt_idx",
    "prompt_name",
    "class_label",
    "best_clip",
    "mean_clip",
    "std_clip",
    "topk_mean_clip",
    "smc_best_clip",
    "best_ir",
    "mean_ir",
    "std_ir",
    "smc_best_ir",
    "best_image_filename",
    "smc_best_image_filename",
    "grid_filename",
    "ess_per_step",
    "metric_samples",
]


def append_csv_row(csv_path: str, row: dict) -> None:
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _parse_smc_schedules(
    mc_samples_schedule: str,
    mc_inner_steps_schedule: str,
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
    return mc_samples, mc_inner_steps


def _load_smc_settings(cfg):
    smc = cfg.smc
    base_sample_type = cli_common.normalize_choice(
        str(smc.base_sample_type).lower(),
        SUPPORTED_BASE_TYPES,
        option_name="smc.base_sample_type",
    )
    posterior_sample_type = cli_common.normalize_choice(
        str(smc.posterior_sample_type).lower(),
        SUPPORTED_POSTERIOR_TYPES,
        option_name="smc.posterior_sample_type",
    )

    seed_override = smc.seed_override
    if seed_override is not None:
        seed_override = int(seed_override)

    return {
        "dataset": str(cfg.problem.target),
        "output_folder": str(cfg.logging.output_folder),
        "temp": float(smc.temp),
        "ess_threshold": float(smc.ess_threshold),
        "use_potential": bool(smc.use_potential),
        "base_sample_type": base_sample_type,
        "posterior_sample_type": posterior_sample_type,
        "base_network_slot": str(cfg.base_network_slot).lower(),
        "posterior_network_slot": str(cfg.posterior_network_slot).lower(),
        "seed_override": seed_override,
        "ndevices": int(cfg.training.ndevices),
    }


def load_and_configure(args):
    """Load the model config. Returns (cfg, statics, train_state, prng_key)."""
    cfg_path = args.cfg_path or DEFAULT_CFG_PATH
    print(f"{cfg_path=}")
    output_folder = args.output_folder or repo_paths.repo_path("outputs")
    cfg = importlib.import_module(cfg_path).get_config(
        args.slurm_id,
        output_folder,
    )
    smc_settings = _load_smc_settings(cfg)

    if smc_settings["seed_override"] is not None:
        with cfg.unlocked():
            cfg.training.seed = smc_settings["seed_override"]

    ndevices = smc_settings["ndevices"]
    if ndevices > 1:
        dist_utils.setup_mesh(ndevices)
        print(f"Multi-GPU: sharding across {ndevices} devices")

    args.cfg_path = cfg_path
    args.dataset = smc_settings["dataset"]
    args.output_folder = smc_settings["output_folder"]
    args.temp = smc_settings["temp"]
    args.ess_threshold = smc_settings["ess_threshold"]
    args.use_potential = smc_settings["use_potential"]
    args.base_sample_type = smc_settings["base_sample_type"]
    args.posterior_sample_type = smc_settings["posterior_sample_type"]
    args.base_network_slot = smc_settings["base_network_slot"]
    args.posterior_network_slot = smc_settings["posterior_network_slot"]

    prng_key = jax.random.PRNGKey(cfg.training.seed)
    cfg, statics, train_state, prng_key = setup_state(cfg, prng_key)

    # Eagerly load VAE decoder
    decode_fn = statics.decode_fn
    if latent_utils.is_latent_target(cfg):
        dummy = jnp.zeros((1, *cfg.problem.image_dims))
        _ = latent_utils.maybe_decode_latents_chunked(
            cfg,
            dummy,
            chunk_size=latent_utils.LATENT_DECODE_CHUNK,
            decode_fn=decode_fn,
        )
        print("VAE decoder pre-loaded.")

    return cfg, statics, train_state, prng_key, decode_fn


# ---------------------------------------------------------------------------
# Sampler construction
# ---------------------------------------------------------------------------

def build_sampler(args, cfg, statics, base_sample_type, posterior_sample_type,
                  reward_fn, mc_samples, mc_inner_steps, temperatures, ts_override):
    """Build the SMC sampler from args. Returns the sampler callable."""
    n_steps = len(ts_override) - 1

    spec = sampling.build_smc_sampler_spec(
        cfg=cfg,
        statics=statics,
        base_sample_type=base_sample_type,
        posterior_sample_type=posterior_sample_type,
        outer_steps=n_steps,
        inner_steps=args.base_inner_steps,
        batch_reward_fn=reward_fn,
        mc_samples=mc_samples,
        mc_inner_steps=mc_inner_steps,
        temperatures=temperatures,
        ts_override=ts_override,
        ess_threshold=args.ess_threshold,
        base_network_slot=args.base_network_slot,
        posterior_network_slot=args.posterior_network_slot,
        use_potential=args.use_potential,
    )

    return sampling.create_batched_sampler(spec)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_batch(args, cfg, statics, subkey, fixed_label):
    """Create initial noise, labels, and RNG keys for a batch. Returns (batch_x0s, batch_keys, batch_labels)."""
    bs = args.batch_size
    rng_keys = jax.random.split(subkey, num=bs + 1)
    subkey, batch_keys = rng_keys[0], rng_keys[1:]
    subkey, rho0_key, label_key = jax.random.split(subkey, 3)
    batch_x0s = statics.sample_rho0(bs, rho0_key)

    if fixed_label is not None:
        batch_labels = jnp.full((bs,), fixed_label, dtype=jnp.int32)
    elif cfg.training.conditional:
        num_classes = cfg.problem.num_classes
        if cfg.training.class_dropout > 0:
            num_classes += 1
        batch_labels = jax.random.choice(label_key, num_classes, (bs,))
    else:
        batch_labels = None

    batch_x0s = dist_utils.replicate_batch(cfg, batch_x0s)
    batch_keys = dist_utils.replicate_batch(cfg, batch_keys)
    batch_labels = dist_utils.replicate_batch(cfg, batch_labels)
    return batch_x0s, batch_keys, batch_labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_samples(
    args,
    cfg,
    decode_fn,
    samples_latent,
    reward_fn,
    reward_score_fn,
    reward_params,
    prompt_name,
    prompt_data,
):
    """Decode samples, compute CLIP and optional ImageReward scores.

    Returns (unorm_samples, smc_best_idx, smc_best_reward, clip_rewards,
             ir_best, ir_mean, ir_std, ir_smc_best).
    """
    unorm_samples = decode_latents_to_pixels(cfg, samples_latent, decode_fn)
    metric_samples = (
        unorm_samples.shape[0] if args.metric_samples is None else args.metric_samples
    )
    if metric_samples <= 0:
        raise ValueError("metric_samples must be positive.")
    if metric_samples > unorm_samples.shape[0]:
        raise ValueError(
            f"metric_samples={metric_samples} exceeds batch_size={unorm_samples.shape[0]}."
        )
    metric_images = unorm_samples[:metric_samples]

    reward_variables = {"reward": reward_params}
    prompt_measurement = {"prompt_data": prompt_data}
    batch_reward_fn = jax.vmap(reward_fn, in_axes=(None, 0, None))
    smc_rewards = np.array(
        batch_reward_fn(reward_variables, samples_latent, prompt_measurement)
    )
    smc_best_idx = int(np.argmax(smc_rewards))
    smc_best_reward = float(smc_rewards[smc_best_idx])

    if args.prompt_reward == "clip":
        clip_rewards = np.array(
            reward_score_fn(jnp.array(metric_images), reward_params, prompt_data)
        )
    else:
        from common.rewards import registry as rewards

        clip_score_fn, clip_params = rewards.build_prompt_reward("clip", prompt_name)
        clip_rewards = np.array(clip_score_fn(jnp.array(metric_images), clip_params))
    best_clip = float(np.max(clip_rewards))
    print(f"Best CLIP reward (final samples): {best_clip:.4f}")

    ir_best = ir_mean = ir_std = ir_smc_best = float("nan")
    if args.prompt_reward == "imagereward":
        ir_scores = np.array(
            reward_score_fn(jnp.array(metric_images), reward_params, prompt_data)
        )
        ir_best = float(np.max(ir_scores))
        ir_mean = float(np.mean(ir_scores))
        ir_std = float(np.std(ir_scores))
        ir_smc_best = smc_best_reward
        print(f"Best ImageReward (final samples): {ir_best:.4f}")
        print(f"Mean ImageReward (final samples): {ir_mean:.4f}")
        print(f"SMC best ImageReward:             {ir_smc_best:.4f}")

    return (
        unorm_samples,
        smc_best_idx,
        smc_best_reward,
        clip_rewards,
        ir_best,
        ir_mean,
        ir_std,
        ir_smc_best,
    )


# ---------------------------------------------------------------------------
# Saving results (images, grid, CSV)
# ---------------------------------------------------------------------------

def save_results(args, unorm_samples, smc_best_idx, smc_best_reward, clip_rewards,
                 ir_best, ir_mean, ir_std, ir_smc_best,
                 prompt_dir, run_meta,
                 prompt_idx, prompt_name, class_label):
    """Save best images, top-k grid, and append a CSV row."""

    # Save SMC best image
    smc_best_img = np.clip(np.array(unorm_samples[smc_best_idx]), 0.0, 1.0)
    smc_best_filename = "smc_best.png"
    save_single_image_png(smc_best_img, os.path.join(prompt_dir, smc_best_filename))
    print(
        f"Saved SMC best x1 (reward={smc_best_reward:.4f}) "
        f"to {prompt_dir}/{smc_best_filename}"
    )

    # Save CLIP-best image at original decoded resolution.
    sorted_indices = np.argsort(clip_rewards)[::-1]
    best_idx = int(sorted_indices[0])
    best_img_01 = np.array(unorm_samples[best_idx])

    best_image_filename = "best.png"
    save_single_image_png(best_img_01, os.path.join(prompt_dir, best_image_filename))
    print(f"Saved best image to {prompt_dir}/{best_image_filename}")

    # Save top-k grid at original decoded resolution.
    top_k = min(10, len(clip_rewards))
    top_indices = sorted_indices[:top_k]
    top_samples = unorm_samples[top_indices]

    n_cols = min(5, top_k)
    n_rows = (top_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i in range(top_k):
        axes[i].imshow(np.array(top_samples[i]))
        axes[i].axis("off")
    for i in range(top_k, len(axes)):
        axes[i].axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    grid_filename = "smc_grid.png"
    plt.savefig(os.path.join(prompt_dir, grid_filename), dpi=150)
    plt.close(fig)
    print(f"Saved grid plot to {prompt_dir}/{grid_filename}")

    row = {
        **run_meta,
        "prompt_idx": prompt_idx,
        "prompt_name": prompt_name,
        "class_label": class_label,
        "best_clip": float(np.max(clip_rewards)),
        "mean_clip": float(np.mean(clip_rewards)),
        "std_clip": float(np.std(clip_rewards)),
        "topk_mean_clip": float(np.mean(clip_rewards[top_indices])),
        "smc_best_clip": smc_best_reward if args.prompt_reward != "imagereward" else float("nan"),
        "best_ir": ir_best,
        "mean_ir": ir_mean,
        "std_ir": ir_std,
        "smc_best_ir": ir_smc_best,
        "best_image_filename": best_image_filename,
        "smc_best_image_filename": smc_best_filename,
        "grid_filename": grid_filename,
        "metric_samples": len(clip_rewards),
    }
    prompt_metadata = {
        "base_inner_steps": int(args.base_inner_steps),
        "base_network_slot": args.base_network_slot,
        "base_outer_steps": int(args.base_outer_steps),
        "base_sample_type": _BASE_TYPE_TO_SAMPLE_TYPE[args.base_sample_type].name,
        "batch_size": int(args.batch_size),
        "cfg_path": args.cfg_path,
        "slurm_id": int(args.slurm_id),
        "ess_threshold": args.ess_threshold,
        "mc_inner_steps_schedule": cli_common.normalize_csv_arg(
            args.mc_inner_steps_schedule, sep=","
        ),
        "mc_samples_schedule": cli_common.normalize_csv_arg(
            args.mc_samples_schedule, sep=","
        ),
        "metric_samples": len(clip_rewards),
        "posterior_sample_type": _POSTERIOR_TYPE_TO_SAMPLE_TYPE[
            args.posterior_sample_type
        ].name,
        "posterior_network_slot": args.posterior_network_slot,
        "problem": "prompt_alignment",
        "prompt_reward": args.prompt_reward,
        "temp": args.temp,
        "use_potential": bool(args.use_potential),
    }
    metrics = {
        "best_clip": row["best_clip"],
        "mean_clip": row["mean_clip"],
        "std_clip": row["std_clip"],
        "topk_mean_clip": row["topk_mean_clip"],
        "smc_best_clip": row["smc_best_clip"],
        "best_ir": row["best_ir"],
        "mean_ir": row["mean_ir"],
        "std_ir": row["std_ir"],
        "smc_best_ir": row["smc_best_ir"],
        "metric_samples": len(clip_rewards),
    }
    with open(os.path.join(prompt_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(prompt_metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(os.path.join(prompt_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    csv_path = os.path.join(prompt_dir, "results.csv")
    append_csv_row(csv_path, row)
    print(f"Appended CSV row -> {csv_path}")


# ---------------------------------------------------------------------------
# Per-prompt sampling pipeline
# ---------------------------------------------------------------------------

def run_prompt(args, cfg, statics, decode_fn, prng_key,
               prompt_idx, prompt_name, class_label,
               prompt_data,
               reward_fn, reward_score_fn, reward_params,
               sampler, sampling_vars,
               base_sample_type, posterior_sample_type,
               run_meta, run_dir):
    """Run the full SMC pipeline for a single prompt. Returns updated prng_key."""
    fixed_label = class_label

    # Build schedules
    mc_samples = args.mc_samples
    mc_inner_steps = args.mc_inner_steps
    ts_override = jnp.linspace(0.0, 1.0, args.base_outer_steps + 1)
    temperatures = jnp.array([args.temp], dtype=jnp.float32)

    print(
        f"""
{args.batch_size=}
{mc_samples=}
{mc_inner_steps=}
{temperatures=}
{ts_override=}
base_sample_type={base_sample_type.name}
posterior_sample_type={posterior_sample_type.name}
{fixed_label=}
ess_threshold={args.ess_threshold}
"""
    )

    prng_key, subkey = jax.random.split(prng_key)

    # Generate batch and run sampler
    batch_x0s, batch_keys, batch_labels = generate_batch(args, cfg, statics, subkey, fixed_label)
    batch_measurement = _make_batch_prompt_measurement(
        cfg,
        prompt_data,
        args.batch_size,
    )

    samples_latent, _, _, _, ess_per_step = sampler(
        variables=sampling_vars,
        return_traj=False,
        return_ess=True,
        batch_init_data=batch_x0s,
        batch_prng_key=batch_keys,
        batch_label=batch_labels,
        batch_measurement=batch_measurement,
    )
    ess_str = ";".join(f"{e:.2f}" for e in np.array(ess_per_step))
    print(f"  ESS per step: {ess_str}")

    # Evaluate
    (unorm_samples, smc_best_idx, smc_best_reward, clip_rewards,
     ir_best, ir_mean, ir_std, ir_smc_best) = evaluate_samples(
        args, cfg, decode_fn, samples_latent, reward_fn, reward_score_fn,
        reward_params, prompt_name, prompt_data,
    )

    # Save
    prompt_dir = os.path.join(run_dir, f"{args.prompt_set}_{prompt_idx}")
    os.makedirs(prompt_dir, exist_ok=True)
    save_results(
        args, unorm_samples, smc_best_idx, smc_best_reward, clip_rewards,
        ir_best, ir_mean, ir_std, ir_smc_best,
        prompt_dir, {**run_meta, "ess_per_step": ess_str},
        prompt_idx, prompt_name, class_label,
    )

    return prng_key


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--prompt_reward",
        type=str,
        default="imagereward",
        choices=SUPPORTED_PROMPT_REWARDS,
        help="Prompt reward function.",
    )
    parser.add_argument("--cfg_path", type=str, default=None)
    parser.add_argument("--slurm_id", type=int, default=0)
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        type=str,
        default=None,
        help="Output folder for this SMC run. Defaults to repo outputs.",
    )

    cli_common.add_prompt_selection_args(parser, prompt_set_required=True)

    parser.add_argument("--differentiable", action="store_true", default=False)
    parser.add_argument("--no-differentiable", dest="differentiable", action="store_false")

    parser.add_argument("--base_outer_steps", type=int, default=5)
    parser.add_argument("--base_inner_steps", type=int, default=1)
    parser.add_argument("--mc_samples_schedule", type=str, default="6")
    parser.add_argument("--mc_inner_steps_schedule", type=str, default="1")
    parser.add_argument(
        "--metric_samples",
        type=int,
        default=None,
        help="Number of final samples to use for saved prompt metrics. Defaults to batch_size.",
    )

    args = parser.parse_args()

    args.prompt_reward = cli_common.normalize_choice(
        args.prompt_reward,
        SUPPORTED_PROMPT_REWARDS,
        option_name="prompt_reward",
    )
    args.mc_samples_schedule = cli_common.normalize_csv_arg(
        args.mc_samples_schedule, sep=","
    )
    args.mc_inner_steps_schedule = cli_common.normalize_csv_arg(
        args.mc_inner_steps_schedule, sep=","
    )
    args.mc_samples, args.mc_inner_steps = _parse_smc_schedules(
        args.mc_samples_schedule,
        args.mc_inner_steps_schedule,
    )
    args._prompt_entries = cli_common.selected_prompt_entries(
        args.prompt_set,
        args.prompt_index,
    )

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_single_config(
    args,
    cfg,
    statics,
    train_state,
    prng_key,
    decode_fn,
    slurm_job_id,
    task_metadata=None,
):
    """Run all prompts for the current args config. Returns updated prng_key."""
    mode = f"smc_base_{args.base_sample_type}_post_{args.posterior_sample_type}"

    run_dir = os.path.join(args.output_folder, f"{slurm_job_id}_smc")
    os.makedirs(run_dir, exist_ok=False)
    print(f"Run output directory: {run_dir}")

    run_meta = {
        "slurm_job_id": slurm_job_id,
        "dataset": args.dataset,
        "mode": mode,
        "reward": args.prompt_reward,
        "prompt_set": args.prompt_set,
    }
    prompt_entries = args._prompt_entries
    reward_runtime = prompt_reward_runtime.build_prompt_reward_runtime(
        args.prompt_reward,
        prompt_entries,
        allowed_rewards=SUPPORTED_PROMPT_REWARDS,
    )
    reward_fn = prompt_reward_runtime.make_latent_prompt_reward_fn(
        cfg,
        decode_fn,
        reward_runtime.score_fn,
    )
    print(f"Running {len(prompt_entries)} prompt(s):"
          + "".join(
              f"\n  {idx}. {prompt}" for idx, (prompt, _label) in prompt_entries
          ))

    base_sample_type = _BASE_TYPE_TO_SAMPLE_TYPE[args.base_sample_type]
    posterior_sample_type = _POSTERIOR_TYPE_TO_SAMPLE_TYPE[args.posterior_sample_type]
    mc_samples = args.mc_samples
    mc_inner_steps = args.mc_inner_steps
    ts_override = jnp.linspace(0.0, 1.0, args.base_outer_steps + 1)
    temperatures = jnp.array([args.temp], dtype=jnp.float32)

    sampler = build_sampler(
        args,
        cfg,
        statics,
        base_sample_type,
        posterior_sample_type,
        reward_fn,
        mc_samples,
        mc_inner_steps,
        temperatures,
        ts_override,
    )
    base_params = get_params(
        cfg,
        statics,
        train_state,
        base_sample_type,
        network_slot=args.base_network_slot,
    )
    posterior_params = get_params(
        cfg,
        statics,
        train_state,
        posterior_sample_type,
        network_slot=args.posterior_network_slot,
    )
    sampling_vars = {
        "base": base_params,
        "posterior": posterior_params,
        "reward": reward_runtime.reward_params,
    }
    print(
        f"Using base={args.base_sample_type} params, "
        f"posterior={args.posterior_sample_type} params"
    )

    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_inner_steps": int(args.base_inner_steps),
                "base_network_slot": args.base_network_slot,
                "base_outer_steps": int(args.base_outer_steps),
                "base_sample_type": base_sample_type.name,
                "batch_size": int(args.batch_size),
                "cfg_path": args.cfg_path,
                "slurm_id": int(args.slurm_id),
                "ess_threshold": args.ess_threshold,
                "mc_inner_steps_schedule": cli_common.normalize_csv_arg(
                    args.mc_inner_steps_schedule, sep=","
                ),
                "mc_samples_schedule": cli_common.normalize_csv_arg(
                    args.mc_samples_schedule, sep=","
                ),
                "metric_samples": args.metric_samples,
                "posterior_sample_type": posterior_sample_type.name,
                "posterior_network_slot": args.posterior_network_slot,
                "problem": "prompt_alignment",
                "prompt_reward": args.prompt_reward,
                "temp": args.temp,
                "use_potential": bool(args.use_potential),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    for prompt_number, (prompt_idx, (prompt_name, class_label)) in enumerate(
        prompt_entries, start=1
    ):
        print(f"\n{'='*60}")
        print(f"Running prompt {prompt_number}/{len(prompt_entries)}: {prompt_name}")
        print(f"{'='*60}")

        prng_key = run_prompt(
            args, cfg, statics, decode_fn, prng_key,
            prompt_idx, prompt_name, class_label,
            reward_runtime.prompt_data_by_index[prompt_idx],
            reward_fn,
            reward_runtime.score_fn,
            reward_runtime.reward_params,
            sampler, sampling_vars,
            base_sample_type, posterior_sample_type,
            run_meta, run_dir,
        )

    print(f"\n{'='*60}")
    print(f"Done: {run_dir}")
    print(f"{'='*60}")
    return prng_key


def main():
    args = parse_args()

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")

    # Load model once
    cfg, statics, train_state, prng_key, decode_fn = load_and_configure(args)

    os.makedirs(args.output_folder, exist_ok=True)
    _run_single_config(
        args, cfg, statics, train_state, prng_key, decode_fn,
        slurm_job_id,
    )


if __name__ == "__main__":
    main()
