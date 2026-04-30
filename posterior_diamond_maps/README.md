Official implementation of **Posterior Diamond Maps** experiments from *Diamond Maps*. The code includes training, evaluation, inverse-problem guidance, prompt-alignment guidance, and
SMC for flow-matching, flow map and diamond map image models on CIFAR10, CelebA64, and latent ImageNet1k 256x256. A portion of this codebase is based directly on https://github.com/nmboffi/flow-maps.

Run commands from the repo root. The main launchers are:

- `py/launchers/learn.py`
- `py/launchers/eval.py`
- `py/launchers/guidance.py`
- `py/launchers/smc.py`

## Setup

Use the project conda environment and install the runtime dependencies:

```bash
conda activate diamond_maps
python -m pip install -r requirements.txt
python -c "import jax; print(jax.devices())"
```

The default requirements install CUDA 12 JAX. If you need a different CUDA or
CPU-only JAX build, install the matching JAX package for your machine.

## Data And Checkpoints

The repo resolves paths relative to its root:

- datasets: `./datasets`
- checkpoints: `./ckpt`
- metric/reward assets: `./metric_models`
- outputs: `./outputs`

Model asset behavior:

Stable Diffusion VAE, Inception weights, and HuggingFace prompt-reward
models/tokenizers are downloaded lazily as needed. Pretrained checkpoints under
`./ckpt/sit_assets`, `./ckpt/meanflow_sit_assets`, and Diamond Map checkpoints
such as `./ckpt/ImageNet-DiamondMap-B2.pkl` must be staged before running the
release configs. The teacher and initialization checkpoints originate in
PyTorch from https://huggingface.co/nyu-visionx/SiT-collections/tree/main and
https://github.com/zhuyu-cs/MeanFlow, then are converted to fit the JAX SiT
model defined in this codebase. To download all release assets, run
`hf download MonkeyDoug/diamond-maps --local-dir .` in the **repository root**
(important to preserve the repo structure). This creates `ckpt/` and `datasets/` paths directly under the current directory.

Dataset behavior:

Small-image datasets are handled automatically: CIFAR-10 and CelebA64 are
downloaded by TFDS on first use into `./datasets`. Training and evaluation do
not prepare ImageNet latents automatically; build cached latent TFRecords from
the HuggingFace `ILSVRC/imagenet-1k` dataset with the command below (requires
accepting the dataset terms and being logged in with the `hf` CLI):

```bash
python ./py/launchers/prepare_imagenet_latents.py \
  --output_dir ./datasets/imagenet_latent_256_ema \
  --image_size 256 \
  --batch_size 32 \
  --num_workers 6 \
  --vae_type ema \
  --compute_latent
```

`--compute_latent` prepares both `train` and `validation` splits by default;
use `--split train` or `--split validation` to prepare only one split.

We provide the statistics we use for our results in the same HuggingFace
repository as the model checkpoints. However, we also provide scripts to
calculate FID stats manually.

Small-image FID stats:

```bash
mkdir -p ./datasets/cifar10 ./datasets/celeb_a
python ./py/launchers/calc_dataset_fid_stats.py --dataset cifar10 --out ./datasets/cifar10/cifar_stats.npz
python ./py/launchers/calc_dataset_fid_stats.py --dataset celeb_a --out ./datasets/celeb_a/celeba_stats.npz
```

ImageNet FID stats:

```bash
python ./py/launchers/prepare_imagenet_latents.py \
  --output_dir ./datasets/imagenet_latent_256_ema \
  --image_size 256 \
  --batch_size 32 \
  --num_workers 6 \
  --compute_fid
```

The ImageNet FID command writes
`./datasets/imagenet_latent_256_ema/imagenet_stats.npz`, matching the path used
by the ImageNet configs.

## Train

Small image example, CIFAR-10 flow-map training:

```bash
python ./py/launchers/learn.py \
  --cfg_path configs.cifar10.train \
  --slurm_id 0 \
  --output_folder ./outputs/cifar10_flow_map
```

ImageNet example, B/2 diamond-map training. Prepare ImageNet latents and stage
the SiT L/2 plus MeanFlow B/2 assets first.

```bash
python ./py/launchers/learn.py \
  --cfg_path configs.imagenet.train \
  --slurm_id 0 \
  --output_folder ./outputs/imagenet_diamond
```

The same commands can be submitted through Slurm:

```bash
sbatch --partition=general --gres=gpu:L40S:1 \
  ./slurm_scripts/run.sh \
  --py_dir="$PWD/py/launchers" \
  --launcher_file=learn.py \
  --slurm_id=0 \
  configs.cifar10.train
```

## Evaluate

Use `eval.py` with a trained checkpoint. This example evaluates a CelebA64
flow-map checkpoint.

```bash
python ./py/launchers/eval.py \
  --cfg_path configs.celeba64.eval \
  --slurm_id 0 \
  --ckpt_path /path/to/checkpoint.pkl \
  --output_folder ./outputs/celeba_eval \
  --sample_types FLOW,FLOW_MAP \
  --outer_steps 1 \
  --inner_steps 1,2,4 \
  --ema_factors 0.9999 \
  --comp_fid
```

For ImageNet evaluation, use `configs.imagenet.eval`, `--slurm_id 0`, an
ImageNet diamond-map checkpoint, and prepared ImageNet latents.

## Guidance Examples

ImageNet inverse problem with guidance, SR32:

```bash
python ./py/launchers/guidance.py \
  --cfg_path configs.imagenet.guidance \
  --slurm_id 2 \
  --output_folder ./outputs/imagenet_sr32 \
  --base_outer_steps 12 \
  --base_inner_steps 1 \
  --mc_samples_schedule 1 \
  --mc_inner_steps_schedule 1 \
  --guidance_scales 1.0
```

ImageNet prompt alignment with guidance:

```bash
python ./py/launchers/guidance.py \
  --cfg_path configs.imagenet.guidance \
  --slurm_id 5 \
  --output_folder ./outputs/imagenet_prompt_guidance \
  --prompt_set guidance_imagenet_eval \
  --prompt_index 0 \
  --prompt_reward clip \
  --prompt_metric_rewards clip \
  --base_outer_steps 48 \
  --base_inner_steps 1 \
  --mc_samples_schedule 1 \
  --mc_inner_steps_schedule 1 \
  --guidance_scales 3.0
```

ImageNet SMC with prompt guidance:

```bash
python ./py/launchers/smc.py \
  --cfg_path configs.imagenet.smc \
  --slurm_id 1 \
  --output_folder ./outputs/imagenet_smc \
  --prompt_set smc_imagenet_eval \
  --prompt_index 0 \
  --prompt_reward imagereward \
  --base_outer_steps 6 \
  --base_inner_steps 8 \
  --mc_samples_schedule 1 \
  --mc_inner_steps_schedule 1 \
  --batch_size 10 \
  --metric_samples 8
```

## Reward Parameter Dtype

`REWARD_PARAM_DTYPE` controls the dtype and cache format for converted prompt
reward parameters:

```bash
export REWARD_PARAM_DTYPE=bf16   # default; aliases: bf16, bfloat16
export REWARD_PARAM_DTYPE=fp32   # aliases: fp32, float32
```

Paper results are using bf16.
