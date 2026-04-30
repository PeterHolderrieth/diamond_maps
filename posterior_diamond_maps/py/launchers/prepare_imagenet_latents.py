"""Prepare HuggingFace ImageNet images as latent TFRecords and FID stats."""

# isort: off
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

from common import latent_utils

latent_utils.force_xla_gpu_deterministic_ops()
# isort: on

import argparse
import glob
import re
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from diffusers.models import FlaxAutoencoderKL
from diffusers.utils import logging as diffusers_logging
from jax.sharding import NamedSharding, PartitionSpec as P
from torch.utils.data import DataLoader, Subset
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

import common.dist_utils as dist_utils
import common.fid_utils as fid_utils

diffusers_logging.set_verbosity_error()


# --- TFRecord Serialization Helpers ---

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(latent_array: np.ndarray, label: int) -> bytes:
    """Serialize a numpy array and label into a tf.train.Example."""
    feature = {
        'image': _bytes_feature(latent_array.astype(np.float32).tobytes()),
        'label': _int64_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# --------------------------------------

def center_crop_arr(pil_image, image_size):
    """
    Ported from: https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX,
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def _latent_transform(image_size: int):
    def transform(pil_image):
        arr = center_crop_arr(pil_image, image_size)
        arr = (arr.astype(np.float32) / 127.5) - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    return transform


def _fid_transform(image_size: int):
    def transform(pil_image):
        arr = center_crop_arr(pil_image, image_size)
        return arr.copy()

    return transform


def _build_hf_transform(image_size: int, for_fid: bool):
    image_transform = (
        _fid_transform(image_size) if for_fid else _latent_transform(image_size)
    )

    def transform_batch(batch):
        images = []
        for image in batch["image"]:
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            images.append(image_transform(image))

        labels = [int(label) for label in batch["label"]]
        return {"image": images, "label": labels}

    return transform_batch


def build_split_dataset(
    split: str,
    image_size: int,
    for_fid: bool,
):
    ds = load_dataset("ILSVRC/imagenet-1k", split=split)
    ds = ds.with_transform(
        _build_hf_transform(image_size=image_size, for_fid=for_fid),
        columns=["image", "label"],
    )
    return ds, f"hf_dataset:ILSVRC/imagenet-1k:{split}"


def build_dataloader(ds: Any, batch_size: int, num_workers: int):
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **loader_kwargs)


def subset_from_start(
    ds: Any, start_index: int, max_samples: Optional[int]
) -> Tuple[Subset, int, int, int]:
    total = len(ds)
    end_index = total
    if max_samples is not None:
        end_index = min(total, start_index + max_samples)
    subset = Subset(ds, range(start_index, end_index))
    return subset, total, start_index, end_index


def _resolve_start_index(train_dir: str, samples_per_shard: int) -> Tuple[int, int]:
    """Finds the last written TFRecord shard and determines resume indices."""
    shard_files = glob.glob(os.path.join(train_dir, "latent_shard_*.tfrecord"))
    if not shard_files:
        return 0, 0
        
    shard_indices = []
    for f in shard_files:
        match = re.search(r"latent_shard_(\d+)\.tfrecord", f)
        if match:
            shard_indices.append(int(match.group(1)))
            
    if not shard_indices:
        return 0, 0
        
    max_shard = max(shard_indices)
    
    # To prevent corrupted partial shards on resume, we overwrite the last written shard.
    start_idx = max_shard * samples_per_shard
    return start_idx, max_shard


def _pad_first_dim_to_multiple(x: np.ndarray, multiple: int) -> Tuple[np.ndarray, int]:
    n = x.shape[0]
    if n % multiple == 0:
        return x, n
    pad = multiple - (n % multiple)
    pad_width = [(0, pad)] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, pad_width, mode="constant"), n


def build_cached_encoder(
    vae: FlaxAutoencoderKL,
    vae_params,
):
    n_devices = jax.local_device_count()

    dist_utils.setup_mesh(n_devices)
    data_sharding = NamedSharding(dist_utils.MESH, P("data"))
    replicated_sharding = NamedSharding(dist_utils.MESH, P())
    sharded_vae_params = jax.device_put(vae_params, replicated_sharding)

    def _encode_sharded(params, images_bchw):
        latent_dist = vae.apply(
            {"params": params},
            images_bchw,
            method=FlaxAutoencoderKL.encode,
        ).latent_dist
        # BHWC -> BCHW
        latent = jnp.concatenate([latent_dist.mean, latent_dist.std], axis=-1)
        latent = jnp.transpose(latent, (0, 3, 1, 2))
        return latent

    _encode_sharded = jax.jit(
        _encode_sharded,
        in_shardings=(replicated_sharding, data_sharding),
        out_shardings=data_sharding,
    )

    def _encode(images_bchw):
        images_bchw = np.asarray(images_bchw)
        padded_images_bchw, n_valid = _pad_first_dim_to_multiple(images_bchw, n_devices)
        sharded_batch = jax.device_put(padded_images_bchw, data_sharding)
        cached = _encode_sharded(sharded_vae_params, sharded_batch)
        return np.asarray(cached)[:n_valid]

    print(f"Latent encoding uses across {n_devices} local devices")
    return _encode


def build_fid_feature_extractor(inception: Callable):
    n_devices = jax.local_device_count()
    dist_utils.setup_mesh(n_devices)
    data_sharding = NamedSharding(dist_utils.MESH, P("data"))

    def _extract_sharded(imgs_nhwc):
        imgs299 = jax.image.resize(
            imgs_nhwc,
            (imgs_nhwc.shape[0], 299, 299, imgs_nhwc.shape[-1]),
            method="bilinear",
        )
        return inception(imgs299)

    _extract_sharded = jax.jit(
        _extract_sharded,
        in_shardings=data_sharding,
        out_shardings=data_sharding,
    )

    def _extract(imgs_nhwc):
        imgs_nhwc = np.asarray(imgs_nhwc)
        padded_imgs_nhwc, n_valid = _pad_first_dim_to_multiple(imgs_nhwc, n_devices)
        sharded_batch = jax.device_put(padded_imgs_nhwc, data_sharding)
        acts = _extract_sharded(sharded_batch)
        acts = np.asarray(acts)[:n_valid]
        return acts.reshape(acts.shape[0], -1)

    print(f"FID features across {n_devices} local devices")
    return _extract


def _compute_and_save_latents_split(
    output_dir: str,
    split: str,
    encode_cached: Callable,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_samples: Optional[int],
    samples_per_shard: int = 10000,  # Added shard size
):
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    start_index, current_shard_idx = _resolve_start_index(split_dir, samples_per_shard)
    print(
        f"Resuming {split} latent generation from index {start_index} "
        f"(shard {current_shard_idx})"
    )

    ds_full, source_name = build_split_dataset(
        split=split,
        image_size=image_size,
        for_fid=False,
    )
    ds, total, begin, end = subset_from_start(ds_full, start_index, max_samples)
    if begin >= total:
        print(f"{split} latent dataset already complete: {total} samples.")
        return

    loader = build_dataloader(ds, batch_size=batch_size, num_workers=num_workers)

    print(
        f"Encoding ImageNet {split} latents from {source_name}: "
        f"indices [{begin}, {end}) out of total {total}"
    )

    writer = None
    samples_in_current_shard = 0
    index = begin

    for batch in tqdm(loader, desc="Encoding latents"):
        image = batch["image"]
        label = batch["label"]
        cached = np.asarray(encode_cached(image.numpy()))
        labels = label.numpy()

        for i in range(cached.shape[0]):
            # Open a new shard if necessary
            if writer is None or samples_in_current_shard >= samples_per_shard:
                if writer is not None:
                    writer.close()
                shard_path = os.path.join(
                    split_dir, f"latent_shard_{current_shard_idx:05d}.tfrecord"
                )
                writer = tf.io.TFRecordWriter(shard_path)
                current_shard_idx += 1
                samples_in_current_shard = 0

            # Write serialized example to current shard
            ex = serialize_example(cached[i], int(labels[i]))
            writer.write(ex)

            samples_in_current_shard += 1
            index += 1

    if writer is not None:
        writer.close()

    print(
        f"Saved/updated TFRecord shards in {split_dir}. "
        f"Last written index: {index - 1} (target end: {end - 1})."
    )


def compute_and_save_latents(
    output_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    vae_type: str,
    max_samples: Optional[int],
    split: str,
    samples_per_shard: int = 10000,
):
    print(f"Loading VAE stabilityai/sd-vae-ft-{vae_type}")
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{vae_type}", from_pt=True
    )
    encode_cached = build_cached_encoder(vae, vae_params)

    splits = ("train", "validation") if split == "both" else (split,)
    for split in splits:
        _compute_and_save_latents_split(
            output_dir=output_dir,
            split=split,
            encode_cached=encode_cached,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            max_samples=max_samples,
            samples_per_shard=samples_per_shard,
        )


def update_online_stats(
    batch_acts: np.ndarray, n_seen: int, mu: np.ndarray, M2: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    batch_acts = np.asarray(batch_acts, dtype=np.float64)

    curr_bs = batch_acts.shape[0]
    n_new = n_seen + curr_bs
    batch_mu = batch_acts.mean(0)
    centered = batch_acts - batch_mu
    batch_M2 = centered.T @ centered

    if mu is None:
        mu = batch_mu
        M2 = batch_M2
    else:
        delta = batch_mu - mu
        mu = mu + delta * curr_bs / n_new
        M2 = M2 + batch_M2 + np.outer(delta, delta) * n_seen * curr_bs / n_new

    return n_new, mu, M2


def compute_and_save_fid_stats(
    output_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    fid_max_samples: Optional[int],
):
    out_path = os.path.join(output_dir, "imagenet_stats.npz")

    ds_full, source_name = build_split_dataset(
        split="train",
        image_size=image_size,
        for_fid=True,
    )
    ds, total, begin, end = subset_from_start(ds_full, 0, fid_max_samples)
    loader = build_dataloader(ds, batch_size=batch_size, num_workers=num_workers)
    inception = fid_utils.get_fid_network()
    extract_features = build_fid_feature_extractor(inception)

    n_seen = 0
    mu = None
    M2 = None

    print(
        f"Computing ImageNet FID reference stats from {source_name}: "
        f"indices [{begin}, {end}) out of total {total}"
    )
    for batch in tqdm(loader, desc="FID features"):
        image = batch["image"]
        imgs = image.numpy().astype(np.float32)
        imgs = imgs / 127.5 - 1.0
        acts = extract_features(imgs)
        n_seen, mu, M2 = update_online_stats(acts, n_seen, mu, M2)

    sigma = M2 / max(n_seen - 1, 1)
    np.savez(out_path, mu=np.asarray(mu), sigma=np.asarray(sigma))
    print(f"Saved FID stats ({n_seen} samples) to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ImageNet latent dataset")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vae_type", type=str, default="ema", choices=["mse", "ema"])
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "validation", "both"],
        help="Which split to prepare when --compute_latent is enabled.",
    )
    parser.add_argument("--compute_latent", action="store_true")
    parser.add_argument("--compute_fid", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--fid_max_samples", type=int, default=None)
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=10000,
        help="Latents per TFRecord",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.image_size % 8 != 0:
        raise ValueError("image_size must be divisible by 8")
    if not args.compute_latent and not args.compute_fid:
        raise ValueError("Specify at least one of --compute_latent or --compute_fid")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX local_device_count: {jax.local_device_count()}")
    print("Using Hugging Face dataset source: ILSVRC/imagenet-1k")

    if args.compute_latent:
        compute_and_save_latents(
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            vae_type=args.vae_type,
            max_samples=args.max_samples,
            split=args.split,
            samples_per_shard=args.samples_per_shard,
        )

    if args.compute_fid:
        compute_and_save_fid_stats(
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fid_max_samples=args.fid_max_samples,
        )


if __name__ == "__main__":
    main()
