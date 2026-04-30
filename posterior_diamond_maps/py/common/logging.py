"""Checkpointing, metric logging, and evaluation utilities."""

import os
import glob
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import wandb
import pickle
from flax.serialization import to_state_dict
from matplotlib import pyplot as plt
from ml_collections import config_dict
from . import latent_utils, state_utils
from .sampling import (
    SampleType,
    make_sample_plot,
    make_traj_plot,
    sample_from_ckpt,
)

Parameters = Dict[str, Dict]


def find_latest_checkpoints(output_folder: str, output_name: str) -> Optional[str]:
    """Find the latest checkpoint using the {name}_step_{step}.pkl format."""

    pattern = os.path.join(output_folder, f"{output_name}_step_*.pkl")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    def get_step(path):
        # split by the specific delimiter used in save_state
        # ie. /path/to/model_step_1000.pkl -> 1000
        step_str = path.split("_step_")[-1].replace(".pkl", "")
        return int(step_str)

    # return sorted checkpoints with most recent first
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints


def atomic_save_state(train_state, save_path):
    tmp_path = save_path + ".tmp"

    print(f"Saving state to {save_path}")

    with open(
        tmp_path,
        "wb",
    ) as f:
        pickle.dump(
            to_state_dict(train_state),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        f.flush()
        os.fsync(f.fileno())
        # can evict checkpoint from cache after flushing to file
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    os.replace(tmp_path, save_path)
    return


def save_recent_state(
    train_state: state_utils.EMATrainState,
    cfg: config_dict.ConfigDict,
    max_keep: int = 2,
) -> None:
    """Save flax training state."""

    step = int(train_state.step)

    # save file with format: name_step_123.pkl
    filename = f"{cfg.logging.output_name}_step_{step}.pkl"
    save_path = os.path.join(cfg.logging.output_folder, filename)
    atomic_save_state(train_state, save_path)

    checkpoints = find_latest_checkpoints(
        cfg.logging.output_folder, cfg.logging.output_name
    )
    for old_ckpt in checkpoints[max_keep:]:
        os.remove(old_ckpt)
    return


def save_best_state(
    train_state: state_utils.EMATrainState, cfg: config_dict.ConfigDict, curr_fids: Dict
) -> None:
    best_fids_path = os.path.join(cfg.logging.output_folder, "best_fids.pkl")
    best_fids = {k: float("inf") for k in curr_fids}  # init to inf fid
    if os.path.exists(best_fids_path):
        with open(best_fids_path, "rb") as f:
            best_fids = pickle.load(f)
    for key in curr_fids:
        sample_type, outer_step, inner_step = key
        if best_fids[key] > curr_fids[key]:
            best_fids[key] = curr_fids[key]
            save_path = os.path.join(
                cfg.logging.output_folder,
                f"{cfg.logging.output_name}_{sample_type}_{outer_step}_{inner_step}.pkl",
            )
            atomic_save_state(train_state, save_path)
        # log best fids
        wandb.log(
            {
                f"best_fid/{sample_type}_{outer_step}_{inner_step}": best_fids[key],
            }
        )
    with open(best_fids_path, "wb") as f:
        pickle.dump(best_fids, f)
    return


def log_metrics(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    grad_norm: jnp.ndarray,
    loss_value: float,
    loss_fn_args: Tuple,
    prng_key: jnp.ndarray,
    step_time: float,
) -> jnp.ndarray:
    """Log some metrics to wandb, make a figure, and checkpoint the parameters."""

    step = train_state.step
    learning_rate = statics.schedule(step)

    metrics = {
        f"loss": loss_value,
        f"grad": grad_norm,
        f"learning_rate": learning_rate,
        f"step_time": step_time,
        f"step": step,
    }
    wandb.log(metrics)

    fid_freq = cfg.logging.fid_freq
    visual_freq = int(cfg.logging.visual_freq)
    save_freq = int(cfg.logging.save_freq)
    comp_fid = fid_freq > 0 and step % fid_freq == 0

    if visual_freq > 0 and (step % visual_freq) == 0:
        sample_types = [
            SampleType[sample_type] for sample_type in cfg.logging.sample_types
        ]
        prng_key = evaluation(
            cfg,
            statics,
            train_state,
            prng_key,
            sample_types,
            cfg.logging.outer_steps,
            cfg.logging.inner_steps,
            comp_fid,
            cfg.logging.fid_n_samples,
            cfg.logging.fid_batch_size,
        )

        make_loss_fn_args_plot(cfg, statics, train_state, loss_fn_args)

    if save_freq > 0 and step % save_freq == 0:
        save_recent_state(train_state, cfg)

    return prng_key


def evaluation(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    sample_types: list[SampleType],
    outer_steps: list[int],
    inner_steps: list[int],
    comp_fid: bool,
    fid_samples: int,
    fid_bs: int,
) -> jnp.ndarray:
    plot_dims = latent_utils.get_pixel_image_dims(cfg)
    titles = [sample_type.name for sample_type in sample_types]
    nrows = len(titles)
    curr_fids = {}
    step = int(train_state.step)
    for outer_step in outer_steps:
        for inner_step in inner_steps:
            plot_idxs = [0] + [
                i * inner_step for i in range(1, outer_step + 1)
            ]  # marginals
            xhats, xfinals = np.zeros(
                (
                    nrows,
                    outer_step * inner_step + 1,
                    plot_dims[1],
                    plot_dims[2],
                    plot_dims[0],
                )
            ), np.zeros(
                (
                    nrows,
                    cfg.logging.plot_bs,
                    plot_dims[1],
                    plot_dims[2],
                    plot_dims[0],
                )
            )
            for i, sample_type in enumerate(sample_types):
                prng_key, sub_key = jax.random.split(prng_key)
                xfinals[i], xhats[i], fid = sample_from_ckpt(
                    cfg,
                    statics,
                    train_state,
                    sub_key,
                    outer_step,
                    inner_step,
                    sample_type,
                    comp_fid,
                    fid_samples,
                    fid_bs,
                    ema_factor=cfg.logging.ema_factor,
                    network_slot=cfg.base_network_slot,
                )
                if comp_fid:
                    curr_fids[(sample_type.name, outer_step, inner_step)] = fid

                    wandb.log(
                        {
                            f"fid/{sample_type.name}_{outer_step}_{inner_step}": fid,
                        }
                    )
            traj_fig = make_traj_plot(
                nrows,
                len(plot_idxs),
                xhats,
                titles,
                plot_idxs,
            )
            sample_fig = make_sample_plot(
                nrows, cfg.logging.plot_bs, xfinals, titles
            )

            wandb.log(
                {
                    f"plots/trajs_{outer_step}_{inner_step}": wandb.Image(traj_fig),
                    f"plots/samples_{outer_step}_{inner_step}": wandb.Image(sample_fig),
                }
            )

            plt.close(traj_fig)
            plt.close(sample_fig)
    # save best ckpts by FID
    if comp_fid:
        save_best_state(train_state, cfg, curr_fids)

    return prng_key


def make_loss_fn_args_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
) -> None:
    """Make a plot of the loss function arguments."""
    is_diamond = cfg.network.matching == "diamond_map"
    step = int(train_state.step)
    # unpack the full loss arguments
    supervise_type = cfg.training.supervise_type
    uses_glass_supervision = supervise_type == "glass"
    data_args = (
        loss_fn_args[2:]
        if is_diamond and uses_glass_supervision
        else loss_fn_args[1:]
    )
    s_diag_batch, s_batch, s_prime_batch = None, None, None
    if is_diamond:
        (
            _,
            _,
            _,
            t_batch,
            t_prime_batch,
            s_diag_batch,
            s_batch,
            s_prime_batch,
            _,
            _,
        ) = data_args
    else:
        (_, _, _, t_batch, t_prime_batch, _) = data_args

    # remove pmap reshaping
    t_batch = jnp.squeeze(t_batch)
    t_prime_batch = jnp.squeeze(t_prime_batch)
    if is_diamond:
        s_diag_batch = jnp.squeeze(s_diag_batch)
        s_batch = jnp.squeeze(s_batch)
        s_prime_batch = jnp.squeeze(s_prime_batch)

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    ## set up plot array
    titles = [r"$(t, t')$"]
    times = [r"$(t, t')$"]

    if is_diamond:
        titles.extend([r"$(s, s)$", r"$(s, s')$"])
        times.extend([r"$(s, s)$", r"$(s, s')$"])

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )

    for kk, ax in enumerate(axs.ravel()):
        if titles[kk] in times:
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])

        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[0, jj]
        ax.set_title(title, fontsize=fontsize)

        if jj == 0:
            ax.scatter(t_batch, t_prime_batch, s=0.1, alpha=0.5, marker="o")
        elif jj == 1:
            ax.scatter(s_diag_batch, s_diag_batch, s=0.1, alpha=0.5, marker="o")
        elif jj == 2:
            ax.scatter(s_batch, s_prime_batch, s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)})

    plt.close(fig)
    return
