"""Compute and save dataset FID reference statistics for small image datasets."""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

# Prevent TensorFlow from pre-allocating all GPU memory
import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

import click
import common.fid_utils as fid_utils
from common.repo_paths import datasets_path
import numpy as np
import jax.numpy as jnp
import tensorflow_datasets as tfds


@click.command()
@click.option(
    "--batch",
    default=1024,
    show_default=True,
    help="How many real images to stream through Inception at once.",
)
@click.option(
    "--dataset",
    help="Name of dataset",
)
@click.option(
    "--out",
    default="fid_stats.npz",
    show_default=True,
    help="Path to save (mu,sigma).npz",
)
def main(batch: int, dataset: str, out: str):
    if dataset == "cifar10":
        preprocess = fid_utils.process_image_for_fid
    elif dataset == "celeb_a":
        preprocess = fid_utils.process_celeba_for_fid
    else:
        raise ValueError(f"Dataset name: {dataset} not implemented for FID computation.")

    ds = (
        tfds.load(
            dataset,
            split="train",
            shuffle_files=False,
            data_dir=datasets_path(),
        )
        .map(
            preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch)
        .as_numpy_iterator()
    )

    # set up inception model
    inception = fid_utils.get_fid_network()

    # compute stats using online algorithm to avoid memory issues
    n_seen = 0
    mu = None
    M2 = None  # Welford vars

    for batch_np in ds:
        batch_acts = fid_utils.resize_and_incept(jnp.asarray(batch_np), inception)

        # online mean and covariance
        batch_acts = np.asarray(np.squeeze(batch_acts))
        curr_bs = batch_acts.shape[0]
        n_seen += curr_bs

        if mu is None:
            mu = batch_acts.mean(0)
            M2 = np.cov(batch_acts, rowvar=False) * (curr_bs - 1)
        else:
            delta = batch_acts.mean(0) - mu
            mu += delta * curr_bs / n_seen
            M2 += (
                np.cov(batch_acts, rowvar=False) * (curr_bs - 1)
                + np.outer(delta, delta) * (n_seen - curr_bs) * curr_bs / n_seen
            )

        print(f"\rProcessed {n_seen}", end="", flush=True)

    sigma = M2 / (n_seen - 1)

    print("\nComputing mean and covariance...")
    print(f"Total samples: {n_seen}, Feature dim: {mu.shape[0]}")

    # save statistics
    np.savez(out, mu=np.asarray(mu), sigma=np.asarray(sigma))
    print(f"\nSaved stats to {out}")


if __name__ == "__main__":
    main()
