#!/usr/bin/env python
"""Benchmark generation wrapper for GenEval and UniGenBench."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from utils.diamond_runtime import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FLUX_LORA_SOURCE,
    DEFAULT_FLUX_LORA_WEIGHT_NAME,
    DEFAULT_FLUX_MODEL,
    DEFAULT_FLUX_SIZE,
    DEFAULT_SANA_MODEL,
    DEFAULT_SANA_SIZE,
    BENCHMARK_METHODS,
    normalize_method_name,
)

DEFAULT_FLUX_PARTICLE_COUNTS = (1, 2, 4, 8, 16, 32)


def csv_int_tuple(value: str) -> tuple[int, ...]:
    """Parse comma-separated integer values."""
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def benchmark_method_arg(value: str) -> str:
    """Parse a benchmark method while accepting historical aliases."""
    try:
        method = normalize_method_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if method not in BENCHMARK_METHODS:
        valid = ", ".join(BENCHMARK_METHODS)
        raise argparse.ArgumentTypeError(
            f"Unsupported benchmark method: {value}. Choose from {valid}."
        )
    return method


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate GenEval or UniGenBench samples for Diamond Maps methods. "
            "FLUX Weighted Diamond Maps defaults to the paper particle grid."
        )
    )
    parser.add_argument("--benchmark", choices=["geneval", "unigenbench"], default="geneval")
    parser.add_argument("--model-family", choices=["flux", "sana"], default="flux")
    parser.add_argument(
        "--method",
        type=benchmark_method_arg,
        default="weighted_diamond",
        metavar="{base,flow_map_guidance,weighted_diamond,best_of_n}",
    )
    parser.add_argument(
        "--prompt-list-file",
        default=None,
        help="GenEval JSONL or UniGenBench CSV. Defaults to the existing benchmark assets.",
    )
    parser.add_argument("--output-base-dir", default=None)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--images-per-prompt", type=int, default=1)
    parser.add_argument(
        "--unigenbench-category",
        choices=["en", "en_long", "zh", "zh_long"],
        default="en",
    )

    parser.add_argument("--flux-model", default=DEFAULT_FLUX_MODEL)
    parser.add_argument("--flux-lora-source", default=DEFAULT_FLUX_LORA_SOURCE)
    parser.add_argument("--flux-lora-weight-name", default=DEFAULT_FLUX_LORA_WEIGHT_NAME)
    parser.add_argument("--flux-size", type=int, default=DEFAULT_FLUX_SIZE)

    parser.add_argument("--sana-model", default=DEFAULT_SANA_MODEL)
    parser.add_argument("--sana-size", type=int, default=DEFAULT_SANA_SIZE)

    parser.add_argument("--reward-model", default="composite")
    parser.add_argument("--reward-scale", type=float, default=250.0)
    parser.add_argument("--gradient-norm-scale", type=float, default=20.0)
    parser.add_argument("--cfg-guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--num-guidance-steps", type=int, default=None)
    parser.add_argument("--guidance-start-step", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-reward-particles", type=int, default=None)
    parser.add_argument(
        "--particle-counts",
        type=csv_int_tuple,
        default=None,
        help=(
            "Comma-separated Weighted Diamond Maps particle grid. "
            "FLUX defaults to 1,2,4,8,16,32."
        ),
    )
    parser.add_argument("--snr-factor", type=float, default=None)
    parser.add_argument(
        "--particle-weighting",
        choices=["reward_softmax", "theorem"],
        default="reward_softmax",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded benchmark worker commands without running them.",
    )
    return parser.parse_args(argv)


def apply_defaults(args: argparse.Namespace) -> None:
    """Apply paper defaults for the selected model family."""
    if args.prompt_list_file is None:
        args.prompt_list_file = (
            "UniGenBench/data/test_prompts_en.csv"
            if args.benchmark == "unigenbench"
            else "assets/evaluation_metadata.jsonl"
        )
    if args.output_base_dir is None:
        args.output_base_dir = (
            "UniGenBench/eval_data/en"
            if args.benchmark == "unigenbench"
            else "geneval"
        )

    if args.model_family == "flux":
        if args.num_inference_steps is None:
            args.num_inference_steps = 25
        if args.num_guidance_steps is None:
            args.num_guidance_steps = 10 if args.method != "base" else 0
        if args.snr_factor is None:
            args.snr_factor = 1.5
        if args.method == "weighted_diamond" and args.particle_counts is None:
            if args.num_reward_particles is None:
                args.particle_counts = DEFAULT_FLUX_PARTICLE_COUNTS
            else:
                args.particle_counts = (int(args.num_reward_particles),)
        return

    if args.num_inference_steps is None:
        args.num_inference_steps = 20
    if args.num_guidance_steps is None:
        args.num_guidance_steps = 5 if args.method != "base" else 0
    if args.snr_factor is None:
        args.snr_factor = 20.0
    if args.method == "weighted_diamond" and args.particle_counts is None:
        args.particle_counts = (int(args.num_reward_particles or 4),)


def add_bool_flag(command: list[str], flag_name: str, enabled: bool) -> None:
    """Append a BooleanOptionalAction style flag."""
    command.append(flag_name if enabled else f"--no-{flag_name[2:]}")


def build_command(args: argparse.Namespace, particle_count: int | None) -> list[str]:
    """Build one internal benchmark worker command."""
    command = [
        sys.executable,
        "-m",
        "utils.benchmark_worker",
        "--benchmark",
        args.benchmark,
        "--model-family",
        args.model_family,
        "--method",
        args.method,
        "--prompt-list-file",
        str(args.prompt_list_file),
        "--output-base-dir",
        str(args.output_base_dir),
        "--cache-dir",
        args.cache_dir,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(args.shard_index),
        "--reward-model",
        args.reward_model,
        "--reward-scale",
        str(args.reward_scale),
        "--gradient-norm-scale",
        str(args.gradient_norm_scale),
        "--cfg-guidance-scale",
        str(args.cfg_guidance_scale),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--num-guidance-steps",
        str(args.num_guidance_steps),
        "--guidance-start-step",
        str(args.guidance_start_step),
    ]

    if args.max_prompts is not None:
        command.extend(["--max-prompts", str(args.max_prompts)])
    add_bool_flag(command, "--skip-existing", bool(args.skip_existing))

    if args.benchmark == "unigenbench":
        command.extend(
            [
                "--images-per-prompt",
                str(args.images_per_prompt),
                "--unigenbench-category",
                args.unigenbench_category,
            ]
        )

    if args.model_family == "flux":
        command.extend(
            [
                "--flux-model",
                args.flux_model,
                "--lora-dir",
                args.flux_lora_source,
                "--weight-name",
                args.flux_lora_weight_name,
                "--flux-size",
                str(args.flux_size),
            ]
        )
    else:
        command.extend(
            [
                "--sana-model",
                args.sana_model,
                "--sana-size",
                str(args.sana_size),
                "--sana-jump-to-end-after-guidance",
            ]
        )

    if args.method == "best_of_n":
        command.extend(["--num-samples", str(args.num_samples)])
    elif args.method == "weighted_diamond":
        command.extend(
            [
                "--num-reward-particles",
                str(particle_count),
                "--snr-factor",
                str(args.snr_factor),
                "--diamond-noise-mode",
                "fresh",
                "--include-likelihood",
                "--include-score",
            ]
        )
        add_bool_flag(
            command,
            "--include-weights",
            args.particle_weighting == "theorem",
        )

    return command


def main() -> None:
    """Run benchmark generation commands."""
    args = parse_args()
    apply_defaults(args)

    particle_counts: tuple[int | None, ...]
    if args.method == "weighted_diamond":
        particle_counts = tuple(int(value) for value in args.particle_counts)
    else:
        particle_counts = (None,)

    for particle_count in particle_counts:
        command = build_command(args, particle_count)
        if args.dry_run:
            print(shlex.join(command), flush=True)
            continue
        print("Executing:", shlex.join(command), flush=True)
        subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)


if __name__ == "__main__":
    main()
