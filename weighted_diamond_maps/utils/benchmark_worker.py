#!/usr/bin/env python
"""Internal worker for GenEval or UniGenBench sample generation.

This entrypoint targets large prompt lists and supports two output layouts:

- GenEval:
  - `<run_root>/<prompt_index:05d>/metadata.jsonl`
  - `<run_root>/<prompt_index:05d>/samples/<seed:05d>.png`
- UniGenBench:
  - `<run_root>/<prompt_id>_<image_id>.png`

The script supports prompt sharding across multiple GPUs so large runs can be
resumed and distributed safely. Sampling is deterministic per
`(seed, prompt_index, sample_index)` tuple, which makes sharding and resume
behavior stable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image
import torch

from utils.diamond_runtime import (
    DEFAULT_FLUX_LORA_SOURCE,
    DEFAULT_FLUX_LORA_WEIGHT_NAME,
    DEFAULT_FLUX_MODEL,
    DEFAULT_FLUX_SIZE,
    DEFAULT_SANA_MODEL,
    DEFAULT_SANA_SIZE,
    normalize_method_name,
)

ModelFamilyName = Literal["flux", "sana"]
MethodName = Literal[
    "base",
    "flow_map_guidance",
    "weighted_diamond",
    "diamond_base",
    "best_of_n",
]
BenchmarkName = Literal["geneval", "unigenbench"]


def benchmark_worker_method_arg(value: str) -> str:
    """Parse worker methods while accepting historical aliases."""
    try:
        return normalize_method_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def format_float_for_name(value: float) -> str:
    """Format a float for filesystem-safe run names.

    Args:
        value: Floating-point value to encode.

    Returns:
        Compact string representation such as `3` or `7p5`.
    """
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace("-", "m").replace(".", "p")


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it.

    Args:
        path: Directory path.

    Returns:
        The created directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_metadata(path: Path, max_prompts: int | None) -> list[dict[str, Any]]:
    """Load GenEval JSONL metadata entries.

    Args:
        path: JSONL metadata file.
        max_prompts: Optional prefix length cap.

    Returns:
        Parsed metadata rows.
    """
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_prompts is not None and len(rows) >= max_prompts:
                break
    return rows


def load_unigenbench_prompts(
    path: Path, max_prompts: int | None
) -> list[dict[str, Any]]:
    """Load UniGenBench CSV prompt rows.

    Args:
        path: CSV prompt file.
        max_prompts: Optional prefix length cap.

    Returns:
        Parsed CSV rows as dictionaries.
    """
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            if max_prompts is not None and len(rows) >= max_prompts:
                break
    return rows


def select_shard(
    metadata_rows: list[dict[str, Any]],
    num_shards: int,
    shard_index: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Select one modulo-based prompt shard.

    Args:
        metadata_rows: Full metadata list.
        num_shards: Total number of prompt shards.
        shard_index: Zero-based shard index.

    Returns:
        List of `(global_prompt_index, metadata)` pairs for this shard.
    """
    selected: list[tuple[int, dict[str, Any]]] = []
    for prompt_index, metadata in enumerate(metadata_rows):
        if prompt_index % num_shards == shard_index:
            selected.append((prompt_index, metadata))
    return selected


def make_prompt_generator(
    device: str,
    seed: int,
    prompt_index: int,
    sample_index: int = 0,
) -> torch.Generator:
    """Create a deterministic generator for one prompt.

    This avoids dependence on processing order, which is important for sharded
    or resumed runs.

    Args:
        device: Target device string.
        seed: Base seed for the run.
        prompt_index: Global prompt index within the metadata file.
        sample_index: Optional within-prompt sample index.

    Returns:
        Seeded torch generator.
    """
    generator_device = "cuda" if device.startswith("cuda") else "cpu"
    combined_seed = (
        int(seed) * 1_000_003 + int(prompt_index) * 1_009 + int(sample_index)
    )
    return torch.Generator(device=generator_device).manual_seed(combined_seed)


def build_run_name(args: argparse.Namespace) -> str:
    """Build the output run name from generation settings.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Descriptive run directory name.
    """
    family = str(args.model_family)
    method = str(args.method)
    if method == "best_of_n":
        return (
            f"{family}_best_of_n_{args.reward_model}"
            f"_n{args.num_inference_steps}"
            f"_N{args.num_samples}"
        )
    if method in ("base", "flow_map_guidance"):
        return (
            f"{family}_{method}_{args.reward_model}"
            f"_reward{format_float_for_name(args.reward_scale)}"
            f"_norm{format_float_for_name(args.gradient_norm_scale)}"
            f"_n{args.num_inference_steps}"
            f"_g{args.num_guidance_steps}"
            f"_start{args.guidance_start_step}"
        )

    weight_tag = "theoremweights" if args.include_weights else "rewardsoftmax"
    run_name = (
        f"{family}_{method}_{args.reward_model}"
        f"_reward{format_float_for_name(args.reward_scale)}"
        f"_norm{format_float_for_name(args.gradient_norm_scale)}"
        f"_n{args.num_inference_steps}"
        f"_g{args.num_guidance_steps}"
        f"_start{args.guidance_start_step}"
        f"_p{args.num_reward_particles}"
        f"_snr{format_float_for_name(args.snr_factor)}"
        f"_{args.diamond_noise_mode}"
        f"_{weight_tag}"
    )
    if args.include_weights:
        run_name = f"{run_name}_temp{format_float_for_name(args.weight_temperature)}"
    if args.model_family == "sana":
        jump_tag = "jump" if args.sana_jump_to_end_after_guidance else "nojump"
        run_name = f"{run_name}_{jump_tag}"
    return run_name


def write_metadata_file(path: Path, metadata: dict[str, Any]) -> None:
    """Write one GenEval metadata file.

    Args:
        path: Destination metadata path.
        metadata: Prompt metadata entry.
    """
    with path.open("w") as handle:
        handle.write(json.dumps(metadata))
        handle.write("\n")


def resolve_unigenbench_prompt_column(args: argparse.Namespace) -> str:
    """Choose the UniGenBench prompt column for the requested category."""
    if args.unigenbench_prompt_column is not None:
        return str(args.unigenbench_prompt_column)
    if args.unigenbench_category in {"en", "en_long"}:
        return "prompt_en"
    if args.unigenbench_category in {"zh", "zh_long"}:
        return "prompt_zh"
    raise ValueError(
        f"Unsupported UniGenBench category: {args.unigenbench_category}. "
        "Choose from en, en_long, zh, zh_long."
    )


def resolve_prompt_text(args: argparse.Namespace, metadata: dict[str, Any]) -> str:
    """Extract the prompt text for the active benchmark."""
    if args.benchmark == "geneval":
        return str(metadata["prompt"])
    prompt_column = resolve_unigenbench_prompt_column(args)
    return str(metadata[prompt_column])


def resolve_unigenbench_prompt_id(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    prompt_index: int,
) -> str:
    """Resolve the prompt id used in UniGenBench flat filenames."""
    raw_value = metadata.get(args.unigenbench_index_column)
    if raw_value is None:
        return str(prompt_index)
    try:
        return str(int(raw_value))
    except (TypeError, ValueError):
        return str(raw_value)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate GenEval or UniGenBench samples with the current SANA or "
            "FLUX pipelines "
            "and optional prompt sharding."
        )
    )
    parser.add_argument(
        "--benchmark",
        choices=["geneval", "unigenbench"],
        default="geneval",
        help="Benchmark output/layout mode.",
    )
    parser.add_argument(
        "--model-family",
        choices=["flux", "sana"],
        default="sana",
        help="Model family to generate with.",
    )
    parser.add_argument(
        "--method",
        type=benchmark_worker_method_arg,
        default="weighted_diamond",
        metavar="{base,flow_map_guidance,weighted_diamond,best_of_n}",
        help="Generation method for the selected model family.",
    )
    parser.add_argument(
        "--prompt-list-file",
        "--prompt_list_file",
        default="assets/evaluation_metadata.jsonl",
        help="Prompt metadata file (GenEval JSONL or UniGenBench CSV).",
    )
    parser.add_argument(
        "--output-base-dir",
        "--output_base_dir",
        default="geneval",
        help="Base directory containing run folders.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run directory name.",
    )
    parser.add_argument("--cache-dir", "--cache_dir", default="./model_cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--sana-model",
        default=DEFAULT_SANA_MODEL,
    )
    parser.add_argument("--sana-size", type=int, default=DEFAULT_SANA_SIZE)
    parser.add_argument("--flux-model", default=DEFAULT_FLUX_MODEL)
    parser.add_argument(
        "--lora-dir",
        default=DEFAULT_FLUX_LORA_SOURCE,
        help=(
            "LoRA source passed to diffusers. Defaults to the Hugging Face "
            "FLUX Flow Map LoRA repo; local directories are still supported."
        ),
    )
    parser.add_argument("--weight-name", default=DEFAULT_FLUX_LORA_WEIGHT_NAME)
    parser.add_argument("--flux-size", type=int, default=DEFAULT_FLUX_SIZE)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of prompt shards processed in parallel.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based prompt shard index for this worker.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional prefix cap on the metadata list for debugging.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip prompts whose sample file already exists.",
    )
    parser.add_argument(
        "--images-per-prompt",
        type=int,
        default=1,
        help="Number of output images to generate per prompt in UniGenBench mode.",
    )
    parser.add_argument(
        "--unigenbench-category",
        choices=["en", "en_long", "zh", "zh_long"],
        default="en",
        help="UniGenBench split/category.",
    )
    parser.add_argument(
        "--unigenbench-index-column",
        default="index",
        help="CSV column used for UniGenBench prompt ids.",
    )
    parser.add_argument(
        "--unigenbench-prompt-column",
        default=None,
        help="Optional CSV column override for UniGenBench prompt text.",
    )

    parser.add_argument(
        "--reward-model",
        choices=["imagereward", "blueness", "composite"],
        default="composite",
    )
    parser.add_argument(
        "--reward-scale",
        "--reward_scale",
        type=float,
        default=250.0,
    )
    parser.add_argument(
        "--gradient-norm-scale",
        "--grad_norm_scale",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--cfg-guidance-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num-inference-steps",
        "--num_inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--num-guidance-steps",
        "--num_guidance_steps",
        type=int,
        default=10,
    )
    parser.add_argument("--guidance-start-step", type=int, default=1)
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate for best-of-N selection.",
    )
    parser.add_argument(
        "--num-reward-particles",
        "--num_reward_particles",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--snr-factor",
        "--snr_factor",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--weight-temperature",
        "--weight_temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--diamond-noise-mode",
        choices=["fresh", "fixed", "antithetic"],
        default="fresh",
    )
    parser.add_argument(
        "--include-likelihood",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Diamond likelihood term.",
    )
    parser.add_argument(
        "--include-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Diamond score-correction term.",
    )
    parser.add_argument(
        "--include-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable weighted particle aggregation for FLUX Diamond.",
    )
    parser.add_argument(
        "--sana-jump-to-end-after-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after the last guided step and decode a clean prediction.",
    )
    
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"shard_index must satisfy 0 <= shard_index < num_shards, got "
            f"{args.shard_index} for num_shards={args.num_shards}."
        )
    if args.images_per_prompt <= 0:
        raise ValueError(
            f"images_per_prompt must be >= 1, got {args.images_per_prompt}."
        )
    return args


def main() -> None:
    """Run one sharded GenEval generation worker."""
    args = parse_args()

    from utils.benchmark_runtime import (
        build_reward_model,
        clear_cuda_cache,
        dtype_from_name,
        load_flux_pipeline,
        load_sana_pipeline,
        score_pil_image,
    )
    from pipelines.FluxDiamondMap import FluxFlowMapWeightedDiamondPipeline
    from pipelines.FluxFMTT import FluxFlowMapFMTTPipeline
    from pipelines.SanaDiamondMap import DiamondFlowSana
    from pipelines.SanaFMTT import FMTTSana

    dtype = dtype_from_name(args.dtype)
    metadata_path = Path(args.prompt_list_file)
    if args.benchmark == "geneval":
        metadata_rows = load_metadata(metadata_path, args.max_prompts)
    else:
        metadata_rows = load_unigenbench_prompts(metadata_path, args.max_prompts)
    shard_rows = select_shard(metadata_rows, args.num_shards, args.shard_index)

    run_name = args.run_name or build_run_name(args)
    run_root = ensure_dir(Path(args.output_base_dir) / run_name)

    config_payload = {
        "run_name": run_name,
        "benchmark": args.benchmark,
        "seed": int(args.seed),
        "prompt_list_file": str(metadata_path),
        "num_total_prompts": len(metadata_rows),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "num_prompts_in_shard": len(shard_rows),
        "model_family": args.model_family,
        "method": args.method,
        "sana_model": args.sana_model,
        "flux_model": args.flux_model,
        "reward_model": args.reward_model,
        "reward_scale": float(args.reward_scale),
        "gradient_norm_scale": float(args.gradient_norm_scale),
        "num_inference_steps": int(args.num_inference_steps),
        "cfg_guidance_scale": float(args.cfg_guidance_scale),
        "num_guidance_steps": int(args.num_guidance_steps),
        "guidance_start_step": int(args.guidance_start_step),
        "num_samples": int(args.num_samples),
        "num_reward_particles": int(args.num_reward_particles),
        "snr_factor": float(args.snr_factor),
        "diamond_noise_mode": args.diamond_noise_mode,
        "weight_temperature": float(args.weight_temperature),
        "include_likelihood": bool(args.include_likelihood),
        "include_score": bool(args.include_score),
        "include_weights": bool(args.include_weights),
        "sana_jump_to_end_after_guidance": bool(args.sana_jump_to_end_after_guidance),
        "images_per_prompt": int(args.images_per_prompt),
        "unigenbench_category": args.unigenbench_category,
        "unigenbench_index_column": args.unigenbench_index_column,
        "unigenbench_prompt_column": args.unigenbench_prompt_column,
        "device": args.device,
        "dtype": args.dtype,
    }
    with (run_root / "config.json").open("w") as handle:
        json.dump(config_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    reward_model = build_reward_model(args, dtype)
    if args.model_family == "sana":
        if args.method in ("base", "flow_map_guidance", "best_of_n"):
            pipeline = load_sana_pipeline(
                FMTTSana,
                model_name=args.sana_model,
                cache_dir=args.cache_dir,
                device=args.device,
                dtype=dtype,
            )
        else:
            pipeline = load_sana_pipeline(
                DiamondFlowSana,
                model_name=args.sana_model,
                cache_dir=args.cache_dir,
                device=args.device,
                dtype=dtype,
            )
    else:
        if args.method in ("base", "flow_map_guidance", "best_of_n"):
            pipeline = load_flux_pipeline(
                FluxFlowMapFMTTPipeline,
                model_name=args.flux_model,
                cache_dir=args.cache_dir,
                lora_source=args.lora_dir,
                weight_name=args.weight_name,
                device=args.device,
                dtype=dtype,
            )
        else:
            pipeline = load_flux_pipeline(
                FluxFlowMapWeightedDiamondPipeline,
                model_name=args.flux_model,
                cache_dir=args.cache_dir,
                lora_source=args.lora_dir,
                weight_name=args.weight_name,
                device=args.device,
                dtype=dtype,
            )

    print(
        f"Run root: {run_root}\n"
        f"Seed: {args.seed}\n"
        f"Shard: {args.shard_index}/{args.num_shards}\n"
        f"Prompts in shard: {len(shard_rows)} / {len(metadata_rows)}",
        flush=True,
    )

    def generate_single_base_image(
        prompt_text: str,
        prompt_dir: Path,
        prompt_generator: torch.Generator,
    ) -> Image.Image:
        """Generate one unguided image for the active model family."""
        if args.model_family == "sana":
            image_size = int(args.sana_size)
            output_img = pipeline.apply(
                prompt=prompt_text,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.cfg_guidance_scale),
                width=image_size,
                height=image_size,
                generator=prompt_generator,
                reward_scale=0.0,
                reward_fns=[reward_model],
                gradient_norm_scale=0.0,
                guidance_start_step=int(args.guidance_start_step),
                num_guidance_steps=0,
                jump_to_end_after_guidance=bool(args.sana_jump_to_end_after_guidance),
                save_path=str(prompt_dir),
                save_intermediate_imgs=False,
                save_outputs=False,
            )
            return Image.fromarray(output_img)

        image_size = int(args.flux_size)
        output = pipeline(
            prompt=prompt_text,
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=float(args.cfg_guidance_scale),
            width=image_size,
            height=image_size,
            generator=prompt_generator,
            reward_scale=0.0,
            reward_fns=[reward_model],
            gradient_norm_scale=0.0,
            num_guidance_steps=0,
            guidance_start_step=int(args.guidance_start_step),
            save_path=str(prompt_dir),
            save_intermediate_imgs=False,
        )
        return output.images[0]

    def generate_image_for_sample(
        prompt_text: str,
        prompt_index: int,
        sample_index: int,
        prompt_save_path: Path,
    ) -> Image.Image:
        """Generate one final image for one prompt/image slot."""
        if args.method == "best_of_n":
            best_score = float("-inf")
            best_image: Image.Image | None = None
            for bon_index in range(int(args.num_samples)):
                sample_generator = make_prompt_generator(
                    args.device,
                    args.seed,
                    prompt_index,
                    sample_index * int(args.num_samples) + bon_index,
                )
                candidate_image = generate_single_base_image(
                    prompt_text=prompt_text,
                    prompt_dir=prompt_save_path,
                    prompt_generator=sample_generator,
                )
                candidate_score = score_pil_image(
                    reward_model=reward_model,
                    prompt=prompt_text,
                    image=candidate_image,
                    device=args.device,
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_image = candidate_image.copy()
            if best_image is None:
                raise RuntimeError("Best-of-N generation did not produce any image.")
            return best_image

        generator = make_prompt_generator(
            args.device,
            args.seed,
            prompt_index,
            sample_index,
        )

        if args.model_family == "sana":
            image_size = int(args.sana_size)
            if args.method == "base":
                output_img = pipeline.apply(
                    prompt=prompt_text,
                    num_inference_steps=int(args.num_inference_steps),
                    guidance_scale=float(args.cfg_guidance_scale),
                    width=image_size,
                    height=image_size,
                    generator=generator,
                    reward_scale=0.0,
                    reward_fns=[reward_model],
                    gradient_norm_scale=0.0,
                    guidance_start_step=int(args.guidance_start_step),
                    num_guidance_steps=0,
                    jump_to_end_after_guidance=bool(
                        args.sana_jump_to_end_after_guidance
                    ),
                    save_path=str(prompt_save_path),
                    save_intermediate_imgs=False,
                    save_outputs=False,
                )
                return Image.fromarray(output_img)
            if args.method == "flow_map_guidance":
                output_img = pipeline.apply(
                    prompt=prompt_text,
                    num_inference_steps=int(args.num_inference_steps),
                    guidance_scale=float(args.cfg_guidance_scale),
                    width=image_size,
                    height=image_size,
                    generator=generator,
                    reward_scale=float(args.reward_scale),
                    reward_fns=[reward_model],
                    gradient_norm_scale=float(args.gradient_norm_scale),
                    guidance_start_step=int(args.guidance_start_step),
                    num_guidance_steps=int(args.num_guidance_steps),
                    jump_to_end_after_guidance=bool(
                        args.sana_jump_to_end_after_guidance
                    ),
                    save_path=str(prompt_save_path),
                    save_intermediate_imgs=False,
                    save_outputs=False,
                )
                return Image.fromarray(output_img)

            include_likelihood = args.method == "weighted_diamond" and bool(
                args.include_likelihood
            )
            include_score = args.method == "weighted_diamond" and bool(args.include_score)
            output_img = pipeline.apply(
                prompt=prompt_text,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.cfg_guidance_scale),
                width=image_size,
                height=image_size,
                generator=generator,
                reward_scale=float(args.reward_scale),
                num_reward_particles=int(args.num_reward_particles),
                include_likelihood=include_likelihood,
                include_score=include_score,
                include_weights=bool(args.include_weights),
                reward_fns=[reward_model],
                gradient_norm_scale=float(args.gradient_norm_scale),
                snr_factor=float(args.snr_factor),
                weight_temperature=float(args.weight_temperature),
                guidance_start_step=int(args.guidance_start_step),
                num_guidance_steps=int(args.num_guidance_steps),
                jump_to_end_after_guidance=bool(args.sana_jump_to_end_after_guidance),
                save_path=str(prompt_save_path),
                save_intermediate_imgs=False,
                save_outputs=False,
            )
            return Image.fromarray(output_img)

        image_size = int(args.flux_size)
        if args.method == "base":
            output = pipeline(
                prompt=prompt_text,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.cfg_guidance_scale),
                width=image_size,
                height=image_size,
                generator=generator,
                reward_scale=0.0,
                reward_fns=[reward_model],
                gradient_norm_scale=0.0,
                num_guidance_steps=0,
                guidance_start_step=int(args.guidance_start_step),
                save_path=str(prompt_save_path),
                save_intermediate_imgs=False,
            )
            return output.images[0]
        if args.method == "flow_map_guidance":
            output = pipeline(
                prompt=prompt_text,
                num_inference_steps=int(args.num_inference_steps),
                guidance_scale=float(args.cfg_guidance_scale),
                width=image_size,
                height=image_size,
                generator=generator,
                reward_scale=float(args.reward_scale),
                reward_fns=[reward_model],
                gradient_norm_scale=float(args.gradient_norm_scale),
                num_guidance_steps=int(args.num_guidance_steps),
                guidance_start_step=int(args.guidance_start_step),
                save_path=str(prompt_save_path),
                save_intermediate_imgs=False,
            )
            return output.images[0]

        include_likelihood = args.method == "weighted_diamond" and bool(
            args.include_likelihood
        )
        include_score = args.method == "weighted_diamond" and bool(args.include_score)
        output = pipeline(
            prompt=prompt_text,
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=float(args.cfg_guidance_scale),
            width=image_size,
            height=image_size,
            generator=generator,
            reward_scale=float(args.reward_scale),
            num_reward_particles=int(args.num_reward_particles),
            include_likelihood=include_likelihood,
            include_score=include_score,
            include_weights=bool(args.include_weights),
            reward_fns=[reward_model],
            gradient_norm_scale=float(args.gradient_norm_scale),
            snr_factor=float(args.snr_factor),
            weight_temperature=float(args.weight_temperature),
            diamond_noise_mode=str(args.diamond_noise_mode),
            num_guidance_steps=int(args.num_guidance_steps),
            guidance_start_step=int(args.guidance_start_step),
            save_path=str(prompt_save_path),
            save_intermediate_imgs=False,
        )
        return output.images[0]

    for shard_position, (prompt_index, metadata) in enumerate(shard_rows, start=1):
        prompt = resolve_prompt_text(args, metadata)

        if args.benchmark == "geneval":
            prompt_dir = ensure_dir(run_root / f"{prompt_index:05d}")
            samples_dir = ensure_dir(prompt_dir / "samples")
            metadata_file = prompt_dir / "metadata.jsonl"
            sample_file = samples_dir / f"{args.seed:05d}.png"

            if not metadata_file.exists():
                write_metadata_file(metadata_file, metadata)

            if args.skip_existing and sample_file.exists():
                print(
                    f"[{shard_position:03d}/{len(shard_rows):03d}] "
                    f"prompt={prompt_index:05d} skip existing",
                    flush=True,
                )
                continue

            print(
                f"[{shard_position:03d}/{len(shard_rows):03d}] prompt={prompt_index:05d}",
                flush=True,
            )
            generated_image = generate_image_for_sample(
                prompt_text=prompt,
                prompt_index=prompt_index,
                sample_index=0,
                prompt_save_path=prompt_dir,
            )
            generated_image.save(sample_file)
            clear_cuda_cache(args.device)
            print(f"Saved {sample_file}", flush=True)
            continue

        prompt_id = resolve_unigenbench_prompt_id(args, metadata, prompt_index)
        sample_files = [
            run_root / f"{prompt_id}_{image_id}.png"
            for image_id in range(int(args.images_per_prompt))
        ]
        if args.skip_existing and all(path.exists() for path in sample_files):
            print(
                f"[{shard_position:03d}/{len(shard_rows):03d}] "
                f"prompt={prompt_id} skip existing",
                flush=True,
            )
            continue

        print(
            f"[{shard_position:03d}/{len(shard_rows):03d}] prompt={prompt_id}",
            flush=True,
        )
        for image_id, sample_file in enumerate(sample_files):
            if args.skip_existing and sample_file.exists():
                print(f"Skip existing {sample_file}", flush=True)
                continue
            prompt_save_path = ensure_dir(
                run_root / ".prompt_cache" / f"{prompt_id}_{image_id}"
            )
            generated_image = generate_image_for_sample(
                prompt_text=prompt,
                prompt_index=prompt_index,
                sample_index=image_id,
                prompt_save_path=prompt_save_path,
            )
            generated_image.save(sample_file)
            clear_cuda_cache(args.device)
            print(f"Saved {sample_file}", flush=True)


if __name__ == "__main__":
    main()
