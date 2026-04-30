"""Train models from configured datasets and checkpoints."""

# isort: off
import os
import sys
import signal

# Set up path for imports FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)
import common.repo_paths as repo_paths

# Suppress TensorFlow logging before any TF imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

# Force TensorFlow to use CPU only for data loading - no GPU ops
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # Hide all GPUs from TensorFlow
# isort: on

# set compilation cache for jax, helps with startup costs on repeated runs with the same config
# memory usage blowsup if doing a lot of different configurations
# _default_cache_dir = repo_paths.repo_path(".jax_cache")
# _cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", _default_cache_dir)
# os.makedirs(_cache_dir, exist_ok=True)
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"

import argparse
import importlib
import time
from typing import Dict, Tuple

import common.datasets as datasets
import common.dist_utils as dist_utils
import common.fid_utils as fid_utils
import common.interpolant as interpolant
import common.latent_utils as latent_utils
import common.logging as logging
import common.loss_args as loss_args
import common.losses as losses
import common.state_utils as state_utils
import common.updates as updates
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import wandb
from ml_collections import config_dict  # type: ignore
from tqdm.auto import tqdm as tqdm

Parameters = Dict[str, Dict]
mpl.rc_file(repo_paths.repo_path("py", "launchers", "matplotlibrc"))


def _warn_unstable_latent_eval(cfg: config_dict.ConfigDict) -> None:
    if not any(device.platform == "gpu" for device in jax.devices()):
        return

    if not datasets.is_imagenet_latent_target(cfg.problem.target):
        return

    wants_fid = cfg.logging.fid_freq > 0
    wants_visual = cfg.logging.visual_freq > 0
    if not (wants_fid or wants_visual):
        return

    if "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", ""):
        return

    print(
        "Warning: ImageNet latent visualization/FID is running without "
        "--xla_gpu_deterministic_ops=true."
    )


def train_loop(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: np.ndarray,
) -> None:
    """Carry out the training loop."""

    # Get starting step (for resuming from checkpoint)
    # start_step = int(dist_utils.safe_index(cfg, train_state.step))
    start_step = int(train_state.step)

    # guard against sigterm/sigint
    stop = False

    def handler(signum, frame):
        nonlocal stop
        if not stop:
            print(
                f"\nSignal {signum} received. Stopping training after current step..."
            )
        stop = True

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    pbar = tqdm(
        range(start_step, cfg.optimization.total_steps),
        initial=start_step,
        total=cfg.optimization.total_steps,
    )
    for i in pbar:

        # break out of loop if signal
        if stop:
            logging.save_recent_state(train_state, cfg)
            print("Training stopped safely by signal.")
            break

        # construct loss function arguments
        start_time = time.time()
        loss_fn_args, prng_key = statics.get_loss_fn_args(
            cfg, statics, train_state, prng_key
        )

        # take a step on the loss
        train_state, loss_value, grad_norm = statics.train_step(
            train_state, statics.loss, loss_fn_args
        )
        end_time = time.time()

        # log to wandb
        prng_key = logging.log_metrics(
            cfg,
            statics,
            train_state,
            grad_norm,
            loss_value,
            loss_fn_args,
            prng_key,
            end_time - start_time,
        )

        pbar.set_postfix(loss=loss_value)

    # if we naturally ended, save one last time
    if not stop:
        logging.save_recent_state(train_state, cfg)

    return


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Direct flow map learning.")
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--slurm_id", type=int)
    parser.add_argument("--output_folder", type=str)
    return parser.parse_args()


def setup_config_dict():
    args = parse_command_line_arguments()
    cfg_module = importlib.import_module(args.cfg_path)
    return cfg_module.get_config(args.slurm_id, args.output_folder)


def setup_state(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray) -> Tuple[
    config_dict.ConfigDict,
    state_utils.StaticArgs,
    state_utils.EMATrainState,
    jnp.ndarray,
]:
    """Construct static arguments and training state objects."""
    # setup sharding mesh
    dist_utils.setup_mesh(cfg.training.ndevices)
    print(f"Setup sharding mesh {dist_utils.MESH}")

    # define dataset
    cfg, ds, prng_key = datasets.setup_target(cfg, prng_key)
    ex_input = next(ds)
    if isinstance(ex_input, dict):  # handle image datasets
        ex_input = ex_input["image"][0]
    else:
        ex_input = ex_input[0]

    supervise_type = cfg.training.supervise_type
    if supervise_type == "glass":
        assert cfg.sup_network.matching in {"flow", "flow_map"}
        assert cfg.network.matching == "diamond_map"
        assert cfg.sup_network.use_glass
        assert cfg.network.rescale == cfg.sup_network.rescale
        if cfg.training.conditional:
            assert cfg.network.label_dim != 0
            assert cfg.sup_network.label_dim != 0
            assert cfg.network.label_dim == cfg.sup_network.label_dim

    interp = interpolant.setup_interpolant(cfg.problem.interp_type)

    cfg = config_dict.FrozenConfigDict(cfg)

    # define training state
    train_state, net, schedule, prng_key, sup_net, sup_params = (
        state_utils.setup_training_state(
            cfg,
            ex_input,
            prng_key,
        )
    )

    if sup_params is not None:
        sup_params = dist_utils.safe_replicate(cfg, sup_params)

    # define the loss

    loss = losses.setup_loss(cfg, net, interp, sup_net=sup_net)

    get_loss_fn_args = loss_args.setup_loss_fn_args(cfg)

    decode_fn = latent_utils.get_decode_fn(cfg)

    # initialize FID network if FID computation is enabled
    inception_fn = None
    if cfg.logging.fid_freq > 0:
        print("Initializing Inception network for FID computation...")
        inception_fn = fid_utils.get_fid_network()
        print("Inception network initialized.")

    # define static object
    statics = state_utils.StaticArgs(
        net=net,
        schedule=schedule,
        loss=loss,
        get_loss_fn_args=get_loss_fn_args,
        train_step=updates.setup_train_step(cfg),
        ds=ds,
        interp=interp,
        sample_rho0=datasets.setup_base(cfg, ex_input),
        inception_fn=inception_fn,
        decode_fn=decode_fn,
        sup_net=sup_net,
        sup_params=sup_params,
    )

    train_state = dist_utils.safe_replicate(cfg, train_state)

    return cfg, statics, train_state, prng_key


if __name__ == "__main__":
    print("Entering main. Setting up config dict and PRNG key.")
    cfg = setup_config_dict()
    if cfg.network.use_glass:
        raise ValueError("GLASS cannot be used as the main network for training.")
    _warn_unstable_latent_eval(cfg)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "manual")

    # Auto-resume: find latest checkpoint if load_path not specified
    os.makedirs(cfg.logging.output_folder, exist_ok=True)
    run_id = slurm_job_id

    if os.environ.get("SLURM_JOB_PARTITION", "general") == "preempt":
        if cfg.network.load_path == "":
            latest_ckpts = logging.find_latest_checkpoints(
                cfg.logging.output_folder, cfg.logging.output_name
            )
            if latest_ckpts:
                latest_ckpt = latest_ckpts[0]
                print(f"Auto-resume: Found checkpoint {latest_ckpt}", flush=True)
                cfg.network.load_path = latest_ckpt
            else:
                print("No existing checkpoints found. Starting fresh.", flush=True)
            cfg.network.reset_optimizer = False
            cfg.logging.save_freq //= 10  # save much more frequently with preempt

    # Populate JAX device information for single-node multi-GPU training
    cfg.training.ndevices = jax.device_count()
    print(f"Initialized with {cfg.training.ndevices} local GPUs")

    prng_key = jax.random.PRNGKey(cfg.training.seed)

    # Set up weights and biases tracking
    print("Setting up wandb.")
    wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        name=cfg.logging.wandb_name,
        resume="allow",
        config=cfg.to_dict() | {"slurm_job_id": slurm_job_id},
        id=run_id,
    )

    print("Config dict set up. Setting up static arguments and training state.")
    cfg, statics, train_state, prng_key = setup_state(cfg, prng_key)

    train_loop(cfg, statics, train_state, prng_key)
