"""Dataset loading and preprocessing helpers for image and latent targets."""

import functools
import os
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_collections import config_dict


def is_imagenet_latent_target(target: str) -> bool:
    return "imagenet_latent" in target


def unnormalize_image(image: jnp.ndarray):
    """Unnormalize an image from [-1, 1] to [0, 1] by scaling and clipping."""
    image = (image + 1) / 2
    image = jnp.clip(image, 0.0, 1.0)
    return image


def normalize_image_tf(image: tf.Tensor):
    """Normalize an image to have pixel values in the range [-1, 1]."""
    return (2 * (image / 255)) - 1


def preprocess_celeb_a(image: tf.Tensor) -> tf.Tensor:
    """Crop an image to 140x140, then resize to 64x64 pixels."""
    image = normalize_image_tf(image)
    crop = tf.image.resize_with_crop_or_pad(image, 140, 140)
    crop64 = tf.image.resize(crop, [64, 64], method="area", antialias=True)
    return crop64


def preprocess_image(cfg, x: Dict) -> Dict:
    """Preprocess the image for TensorFlow datasets."""
    image = x["image"]

    if cfg.problem.target == "celeb_a":
        # celeb_a doesn't have labels; artificially pad them all to 1
        label = 1.0
    else:
        label = x["label"]

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)

    if cfg.problem.target == "cifar10":
        image = normalize_image_tf(image)
    elif cfg.problem.target == "celeb_a":
        image = preprocess_celeb_a(image)
    else:
        raise ValueError("Unknown dataset type.")

    # ensure (C, H, W)
    image = tf.transpose(image, [2, 0, 1])

    return {"image": image, "label": label}


def get_image_dataset(cfg: config_dict.ConfigDict):
    """Assemble a TensorFlow dataset for the specified problem target."""
    read_config = tfds.ReadConfig(try_autocache=False)
    shuffle = bool(cfg.training.shuffle)

    if cfg.problem.target == "cifar10":
        ds = tfds.load(
            "cifar10",
            split="train",
            shuffle_files=shuffle,
            data_dir=cfg.problem.dataset_location,
            read_config=read_config,
        )
    elif cfg.problem.target == "celeb_a":
        ds = tfds.load(
            "celeb_a",
            split="train",
            shuffle_files=shuffle,
            data_dir=cfg.problem.dataset_location,
            read_config=read_config,
        )
    else:
        raise ValueError(f"Unsupported image target: {cfg.problem.target}")

    ds = ds.map(
        lambda x: preprocess_image(cfg, x),
        num_parallel_calls=tf.data.AUTOTUNE if shuffle else 1,
    )
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = (
        ds.repeat()
        .batch(cfg.optimization.bs)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    return ds


def get_imagenet_latent_dataset(cfg: config_dict.ConfigDict):
    """Load precomputed ImageNet latent dataset from TFRecords."""
    shuffle = bool(cfg.training.shuffle)
    ds = build_imagenet_latent_tf_dataset(
        cfg,
        shuffle_files=shuffle,
        shuffle_examples=shuffle,
    )

    ds = ds.repeat()
    ds = ds.batch(cfg.optimization.bs, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds.as_numpy_iterator()


def build_imagenet_latent_tf_dataset(
    cfg: config_dict.ConfigDict,
    *,
    shuffle_files: bool,
    shuffle_examples: bool,
):
    """Build the parsed ImageNet latent TF dataset before repeat/batch."""

    # Configuration
    latent_scale = cfg.problem.latent_scale
    latent_sample = cfg.problem.latent_sample
    latent_hflip = cfg.problem.latent_hflip
    latent_split = cfg.problem.latent_split
    latent_spatial_size = cfg.problem.latent_spatial_size

    split_root = os.path.join(cfg.problem.dataset_location, latent_split)

    # 1. Locate all TFRecord shards
    filenames = sorted(tf.io.gfile.glob(os.path.join(split_root, "*.tfrecord")))
    if not filenames:
        raise ValueError(f"No TFRecord files found in {split_root}")

    # 2. Interleave reads from multiple files for maximum disk throughput
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle_files:
        ds = ds.shuffle(len(filenames), reshuffle_each_iteration=True)
        cycle_length = tf.data.AUTOTUNE
        num_parallel_calls = tf.data.AUTOTUNE
        deterministic = False
    else:
        cycle_length = 1
        num_parallel_calls = 1
        deterministic = True
    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
    )

    # 3. Parse and augment on the fly
    def parse_and_process(example_proto):
        # Define the expected format of the binary data
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Decode the raw bytes back into a float32 tensor
        image = tf.io.decode_raw(parsed['image'], tf.float32)
        
        # Reshape flat array back to [8, H, W]
        image = tf.reshape(image, [8, latent_spatial_size, latent_spatial_size])
        
        # Reparameterization trick (Sample from mean and std)
        if latent_sample:
            # Split the 8 channels into 4 mean and 4 std channels
            mean, std = tf.split(image, num_or_size_splits=2, axis=0)
            noise = tf.random.normal(shape=tf.shape(std), dtype=tf.float32)
            image = mean + std * noise
        else:
            # Just take the mean
            image = image[:4, :, :]
            
        # Random Horizontal Flip
        if latent_hflip:
            # image shape is [C, H, W], so width is axis 2
            do_flip = tf.random.uniform([]) < 0.5
            image = tf.cond(
                do_flip,
                lambda: tf.reverse(image, axis=[2]),
                lambda: image
            )
            
        # Apply scaling factor
        image = image * latent_scale
        
        # Cast label to int32 to match original JAX expectations
        label = tf.cast(parsed['label'], tf.int32)
        
        return {"image": image, "label": label}

    # 4. Apply mapping, shuffle, batch, and prefetch
    map_parallelism = tf.data.AUTOTUNE if shuffle_files else 1
    ds = ds.map(parse_and_process, num_parallel_calls=map_parallelism)

    # 10,000 is a good shuffle buffer for TFRecords. The interleave already
    # provides macro-shuffling across shards.
    if shuffle_examples:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)

    return ds


def setup_base(cfg: config_dict.ConfigDict, ex_input: jnp.ndarray) -> Callable:
    """Set up the base density for the system."""
    if cfg.problem.base == "gaussian":

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho0(bs: int, key: jnp.ndarray):
            return cfg.network.rescale * jax.random.normal(
                key, shape=(bs, *ex_input.shape)
            )

    else:
        raise ValueError("Specified base density is not implemented.")

    return sample_rho0


def setup_target(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray):
    """Set up the target density for the system."""
    if cfg.problem.target == "cifar10" or cfg.problem.target == "celeb_a":
        ds = get_image_dataset(cfg)
        print("Loaded image dataset.")

    elif is_imagenet_latent_target(cfg.problem.target):
        ds = get_imagenet_latent_dataset(cfg)
        print("Loaded ImageNet latent dataset.")

    else:
        raise ValueError("Specified target density is not implemented.")

    return cfg, ds, prng_key
