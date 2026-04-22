"""Minimal Diamond Maps inference for one prompt or a short prompt file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.diamond_runtime import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FLUX_LORA_SOURCE,
    DEFAULT_FLUX_LORA_WEIGHT_NAME,
    DEFAULT_FLUX_MODEL,
    DEFAULT_FLUX_SIZE,
    DEFAULT_SANA_MODEL,
    DEFAULT_SANA_SIZE,
    REWARD_MODEL_CHOICES,
    INFERENCE_METHODS,
    build_reward_model,
    dtype_from_name,
    ensure_dir,
    load_flux_pipeline,
    load_sana_pipeline,
    make_generator,
    normalize_method_name,
    slugify,
)


def inference_method_arg(value: str) -> str:
    """Parse an inference method while accepting historical aliases."""
    try:
        method = normalize_method_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if method not in INFERENCE_METHODS:
        valid = ", ".join(INFERENCE_METHODS)
        raise argparse.ArgumentTypeError(
            f"Unsupported inference method: {value}. Choose from {valid}."
        )
    return method


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Diamond Maps inference and save trajectory images."
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Single prompt to generate.")
    prompt_group.add_argument(
        "--prompt-file",
        help="Newline-delimited prompt file. Empty lines and # comments are skipped.",
    )

    parser.add_argument("--output-dir", default="outputs_infer")
    parser.add_argument("--model-family", choices=["flux", "sana"], default="flux")
    parser.add_argument(
        "--method",
        type=inference_method_arg,
        default="weighted_diamond",
        metavar="{base,flow_map_guidance,weighted_diamond}",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)

    parser.add_argument("--flux-model", default=DEFAULT_FLUX_MODEL)
    parser.add_argument("--flux-lora-source", default=DEFAULT_FLUX_LORA_SOURCE)
    parser.add_argument("--flux-lora-weight-name", default=DEFAULT_FLUX_LORA_WEIGHT_NAME)
    parser.add_argument("--flux-size", type=int, default=DEFAULT_FLUX_SIZE)

    parser.add_argument("--sana-model", default=DEFAULT_SANA_MODEL)
    parser.add_argument("--sana-size", type=int, default=DEFAULT_SANA_SIZE)

    parser.add_argument("--reward-model", choices=REWARD_MODEL_CHOICES, default="composite")
    parser.add_argument("--reward-scale", type=float, default=250.0)
    parser.add_argument("--gradient-norm-scale", type=float, default=20.0)
    parser.add_argument("--cfg-guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--num-guidance-steps", type=int, default=None)
    parser.add_argument("--guidance-start-step", type=int, default=1)
    parser.add_argument("--num-reward-particles", type=int, default=None)
    parser.add_argument("--snr-factor", type=float, default=None)
    parser.add_argument(
        "--particle-weighting",
        choices=["reward_softmax", "theorem"],
        default="reward_softmax",
        help="FLUX theorem weighting is experimental; reward_softmax matches the paper recipe.",
    )
    parser.add_argument(
        "--include-likelihood",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-score",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-intermediate-imgs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save decoded x_t and x_0 trajectory images.",
    )
    parser.add_argument(
        "--select-best-scored-image",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Return the best reward-scored candidate instead of the final trajectory sample.",
    )
    parser.add_argument(
        "--sana-jump-to-end-after-guidance",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def apply_family_defaults(args: argparse.Namespace) -> None:
    """Fill model-family-specific defaults after parsing."""
    if args.model_family == "flux":
        if args.num_inference_steps is None:
            args.num_inference_steps = 25
        if args.num_guidance_steps is None:
            args.num_guidance_steps = 24
        if args.snr_factor is None:
            args.snr_factor = 1.1
        if args.select_best_scored_image is None:
            args.select_best_scored_image = False
        if args.sana_jump_to_end_after_guidance is None:
            args.sana_jump_to_end_after_guidance = False
        if args.num_reward_particles is None:
            args.num_reward_particles = 16
        return

    if args.num_inference_steps is None:
        args.num_inference_steps = 20
    if args.num_guidance_steps is None:
        args.num_guidance_steps = 5
    if args.snr_factor is None:
        args.snr_factor = 20.0
    if args.select_best_scored_image is None:
        args.select_best_scored_image = True
    if args.sana_jump_to_end_after_guidance is None:
        args.sana_jump_to_end_after_guidance = True
    if args.num_reward_particles is None:
        args.num_reward_particles = 4


def load_prompts(args: argparse.Namespace) -> list[str]:
    """Load prompts from CLI arguments."""
    if args.prompt is not None:
        return [args.prompt]

    prompt_path = Path(args.prompt_file)
    prompts: list[str] = []
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            prompts.append(text)
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}.")
    return prompts


def format_float_for_name(value: float) -> str:
    """Format a float for filesystem-safe output names."""
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace("-", "m").replace(".", "p")


def build_output_run_name(args: argparse.Namespace) -> str:
    """Build the hyperparameter-tagged output folder name."""
    if args.method == "base":
        return build_base_run_name(args)

    parts = [
        str(args.model_family),
        str(args.method),
        str(args.reward_model),
        f"reward{format_float_for_name(args.reward_scale)}",
        f"norm{format_float_for_name(args.gradient_norm_scale)}",
        f"n{int(args.num_inference_steps)}",
        f"cfg{format_float_for_name(args.cfg_guidance_scale)}",
        f"g{int(args.num_guidance_steps)}",
        f"start{int(args.guidance_start_step)}",
    ]
    if args.method == "weighted_diamond":
        weight_tag = (
            "theoremweights"
            if args.particle_weighting == "theorem"
            else "rewardsoftmax"
        )
        parts.extend(
            [
                f"p{int(args.num_reward_particles)}",
                f"snr{format_float_for_name(args.snr_factor)}",
                "fresh",
                weight_tag,
            ]
        )
    if args.model_family == "sana":
        jump_tag = "jump" if args.sana_jump_to_end_after_guidance else "nojump"
        parts.append(jump_tag)
    return "_".join(parts)


def build_base_run_name(args: argparse.Namespace) -> str:
    """Build the output folder name for same-seed base comparisons."""
    parts = [
        str(args.model_family),
        "base",
        f"n{int(args.num_inference_steps)}",
        f"cfg{format_float_for_name(args.cfg_guidance_scale)}",
    ]
    if args.model_family == "sana":
        jump_tag = "jump" if args.sana_jump_to_end_after_guidance else "nojump"
        parts.append(jump_tag)
    return "_".join(parts)


def build_prompt_output_dir(
    args: argparse.Namespace, prompt_index: int, prompt: str
) -> Path:
    """Build the prompt-level output directory."""
    return Path(args.output_dir) / f"{prompt_index:03d}_{slugify(prompt)}"


def save_image(image: Any, path: Path) -> None:
    """Save a PIL image or uint8 numpy image."""
    if hasattr(image, "save"):
        image.save(path)
        return
    from PIL import Image

    Image.fromarray(image).save(path)


def load_pipeline(args: argparse.Namespace, dtype: Any):
    """Load the requested pipeline."""
    if args.model_family == "flux":
        if args.method == "base":
            from pipelines.FluxFlowMap import FluxPipelineTwoTimestep

            cls = FluxPipelineTwoTimestep
        elif args.method == "flow_map_guidance":
            from pipelines.FluxFMTT import FluxFlowMapFMTTPipeline

            cls = FluxFlowMapFMTTPipeline
        else:
            from pipelines.FluxDiamondMap import FluxFlowMapWeightedDiamondPipeline

            cls = FluxFlowMapWeightedDiamondPipeline

        return load_flux_pipeline(
            cls,
            model_name=args.flux_model,
            cache_dir=args.cache_dir,
            lora_source=args.flux_lora_source,
            weight_name=args.flux_lora_weight_name,
            device=args.device,
            dtype=dtype,
        )

    if args.method in {"base", "flow_map_guidance"}:
        from pipelines.SanaFMTT import FMTTSana

        cls = FMTTSana
    else:
        from pipelines.SanaDiamondMap import DiamondFlowSana

        cls = DiamondFlowSana

    return load_sana_pipeline(
        cls,
        model_name=args.sana_model,
        cache_dir=args.cache_dir,
        device=args.device,
        dtype=dtype,
    )


def prompt_kwargs(
    args: argparse.Namespace,
    *,
    prompt: str,
    prompt_dir: Path,
    seed: int,
    size: int,
) -> dict[str, Any]:
    """Build arguments shared by every prompt-level pipeline call."""
    return {
        "prompt": prompt,
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.cfg_guidance_scale),
        "width": size,
        "height": size,
        "generator": make_generator(args.device, seed),
        "save_path": str(prompt_dir),
        "save_intermediate_imgs": bool(args.save_intermediate_imgs),
    }


def reward_guidance_kwargs(
    args: argparse.Namespace, reward_fns: list[Any]
) -> dict[str, Any]:
    """Build arguments shared by reward-guided methods."""
    return {
        "reward_scale": float(args.reward_scale),
        "reward_fns": reward_fns,
        "gradient_norm_scale": float(args.gradient_norm_scale),
        "guidance_start_step": int(args.guidance_start_step),
        "num_guidance_steps": int(args.num_guidance_steps),
    }


def diamond_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build arguments specific to weighted Diamond guidance."""
    return {
        "num_reward_particles": int(args.num_reward_particles),
        "include_likelihood": bool(args.include_likelihood),
        "include_score": bool(args.include_score),
        "include_weights": args.particle_weighting == "theorem",
        "snr_factor": float(args.snr_factor),
    }


def run_flux_prompt(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    reward_fns: list[Any],
    prompt: str,
    prompt_dir: Path,
    seed: int,
):
    """Run one FLUX prompt."""
    kwargs = prompt_kwargs(
        args,
        prompt=prompt,
        prompt_dir=prompt_dir,
        seed=seed,
        size=int(args.flux_size),
    )

    if args.method == "base":
        return pipeline(**kwargs).images[0]

    kwargs.update(
        reward_guidance_kwargs(args, reward_fns),
        select_best_scored_image=bool(args.select_best_scored_image),
    )
    if args.method == "weighted_diamond":
        kwargs.update(diamond_kwargs(args), diamond_noise_mode="fresh")
    return pipeline(**kwargs).images[0]


def run_sana_prompt(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    reward_fns: list[Any],
    prompt: str,
    prompt_dir: Path,
    seed: int,
):
    """Run one SANA prompt."""
    kwargs = prompt_kwargs(
        args,
        prompt=prompt,
        prompt_dir=prompt_dir,
        seed=seed,
        size=int(args.sana_size),
    )
    kwargs.update(
        jump_to_end_after_guidance=bool(args.sana_jump_to_end_after_guidance),
        save_outputs=False,
    )

    if args.method == "base":
        kwargs.update(
            reward_scale=0.0,
            reward_fns=None,
            gradient_norm_scale=0.0,
            guidance_start_step=int(args.guidance_start_step),
            num_guidance_steps=0,
        )
        return pipeline.apply(**kwargs)

    kwargs.update(reward_guidance_kwargs(args, reward_fns))
    if args.method == "weighted_diamond":
        kwargs.update(diamond_kwargs(args), weight_temperature=1.0)
    return pipeline.apply(**kwargs)


def run_base_prompt(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    prompt: str,
    prompt_dir: Path,
    seed: int,
):
    """Run same-seed base inference with the already-loaded pipeline."""
    size = int(args.flux_size) if args.model_family == "flux" else int(args.sana_size)
    kwargs = prompt_kwargs(
        args,
        prompt=prompt,
        prompt_dir=prompt_dir,
        seed=seed,
        size=size,
    )
    if args.model_family == "flux":
        return pipeline(**kwargs).images[0]

    kwargs.update(
        reward_scale=0.0,
        reward_fns=None,
        gradient_norm_scale=0.0,
        guidance_start_step=int(args.guidance_start_step),
        num_guidance_steps=0,
        jump_to_end_after_guidance=bool(args.sana_jump_to_end_after_guidance),
        save_outputs=False,
    )
    return pipeline.apply(**kwargs)


def run_prompt(
    *,
    args: argparse.Namespace,
    pipeline: Any,
    reward_fns: list[Any],
    prompt: str,
    prompt_dir: Path,
    seed: int,
):
    """Run one prompt with the selected model family."""
    runner = run_flux_prompt if args.model_family == "flux" else run_sana_prompt
    return runner(
        args=args,
        pipeline=pipeline,
        reward_fns=reward_fns,
        prompt=prompt,
        prompt_dir=prompt_dir,
        seed=seed,
    )


def load_reward_fns(args: argparse.Namespace, dtype: Any) -> list[Any]:
    """Load reward models only for guided methods."""
    if args.method == "base":
        return []
    return [
        build_reward_model(
            reward_model=args.reward_model,
            dtype=dtype,
            device=args.device,
            cache_dir=args.cache_dir,
        )
    ]


def build_metadata_record(
    *,
    args: argparse.Namespace,
    method: str,
    prompt_index: int,
    prompt: str,
    seed: int,
    run_name: str,
    run_dir: Path,
    image_path: Path,
) -> dict[str, Any]:
    """Build one output metadata record."""
    return {
        "prompt_index": prompt_index,
        "prompt": prompt,
        "seed": seed,
        "seed_dir": str(run_dir.parent),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model_family": args.model_family,
        "method": method,
        "num_inference_steps": int(args.num_inference_steps),
        "cfg_guidance_scale": float(args.cfg_guidance_scale),
        "num_guidance_steps": 0 if method == "base" else int(args.num_guidance_steps),
        "guidance_start_step": int(args.guidance_start_step),
        "num_reward_particles": (
            int(args.num_reward_particles) if method == "weighted_diamond" else None
        ),
        "snr_factor": float(args.snr_factor) if method == "weighted_diamond" else None,
        "reward_model": None if method == "base" else args.reward_model,
        "select_best_scored_image": bool(args.select_best_scored_image),
        "sana_jump_to_end_after_guidance": bool(args.sana_jump_to_end_after_guidance),
        "save_intermediate_imgs": bool(args.save_intermediate_imgs),
        "output_path": str(image_path),
    }


def main() -> None:
    """Run inference."""
    args = parse_args()
    apply_family_defaults(args)
    prompts = load_prompts(args)

    base_run_name = build_base_run_name(args)
    current_run_name = build_output_run_name(args)
    dtype = dtype_from_name(args.dtype)
    pipeline = load_pipeline(args, dtype)
    reward_fns = load_reward_fns(args, dtype)

    for prompt_index, prompt in enumerate(prompts):
        seed = int(args.seed) + prompt_index
        prompt_dir = ensure_dir(build_prompt_output_dir(args, prompt_index, prompt))
        seed_dir = ensure_dir(prompt_dir / f"seed_{seed}")

        runs_to_save = [("base", base_run_name)]
        if args.method != "base":
            runs_to_save.append((args.method, current_run_name))

        for method, run_name in runs_to_save:
            run_dir = ensure_dir(seed_dir / run_name)
            if method == "base":
                image = run_base_prompt(
                    args=args,
                    pipeline=pipeline,
                    prompt=prompt,
                    prompt_dir=run_dir,
                    seed=seed,
                )
            else:
                image = run_prompt(
                    args=args,
                    pipeline=pipeline,
                    reward_fns=reward_fns,
                    prompt=prompt,
                    prompt_dir=run_dir,
                    seed=seed,
                )
            image_path = run_dir / "final.png"
            save_image(image, image_path)

            record = build_metadata_record(
                args=args,
                method=method,
                prompt_index=prompt_index,
                prompt=prompt,
                seed=seed,
                run_name=run_name,
                run_dir=run_dir,
                image_path=image_path,
            )
            metadata_path = run_dir / "metadata.jsonl"
            with metadata_path.open("a", encoding="utf-8") as metadata_file:
                metadata_file.write(json.dumps(record, sort_keys=True) + "\n")
            print(f"Saved {image_path}", flush=True)


if __name__ == "__main__":
    main()
