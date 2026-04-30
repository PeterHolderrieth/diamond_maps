"""Evaluate checkpoints with sampling, posterior visualization, and FID metrics."""

# isort: off
import gc
import os
import sys

# Set up path for imports FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

from common import latent_utils

latent_utils.force_xla_gpu_deterministic_ops()
# isort: on

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import click
import jax
import numpy as np
import seaborn as sns
import wandb
from common.sampling import (
    SampleType,
    calc_fid,
    get_params,
    make_posterior_sample_fn,
    make_posterior_sample_plot,
    make_sample_plot,
    make_traj_plot,
    sample_from_ckpt,
)
from launchers.learn import setup_state
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class EvalCase:
    sample_types: Tuple[SampleType, ...]
    outer_step: int
    inner_step: int
    ema_factor: float
    network_slot: str
    posterior_t: Optional[float] = None

    @property
    def is_posterior(self) -> bool:
        return self.posterior_t is not None

    @property
    def total_steps(self) -> int:
        return self.outer_step * self.inner_step


@dataclass(frozen=True)
class FidRecord:
    sample_type: str
    ema_factor: float
    outer_step: int
    inner_step: int
    fid: Optional[float]
    network_slot: str
    posterior_t: Optional[float] = None

    @property
    def is_posterior(self) -> bool:
        return self.posterior_t is not None

    @property
    def total_steps(self) -> int:
        return self.outer_step * self.inner_step


@dataclass(frozen=True)
class FigureArtifact:
    figure: Any
    output_name: str
    wandb_key: str


@dataclass(frozen=True)
class EvalRuntime:
    cfg: Any
    statics: Any
    train_state: Any
    comp_fid: bool
    fid_samples: int
    fid_bs: int
    metrics_only: bool
    output_folder: Optional[str]
    wb: Any
    plot_dims: Tuple[int, int, int]
    vis_mode: str


@dataclass
class EvalCaseResult:
    fid_records: List[FidRecord]
    figure_artifacts: List[FigureArtifact]
    next_prng_key: Any


def _parse_int_list(value: Optional[str]) -> Tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(part.strip()) for part in value.split(","))


def _parse_sample_types(value: Optional[str]) -> Tuple[SampleType, ...]:
    if value is None:
        return ()
    return tuple(SampleType[part.strip()] for part in value.split(","))


def _parse_ema_factors(
    value: Optional[str], default_values: Sequence[float]
) -> Tuple[float, ...]:
    if value is None:
        return tuple(float(factor) for factor in default_values)
    return tuple(float(part.strip()) for part in value.split(","))


def _parse_posterior_t_list(value: Optional[str]) -> Tuple[float, ...]:
    if value is None:
        return ()

    posterior_t_values = tuple(float(part.strip()) for part in value.split(","))
    for tval in posterior_t_values:
        if not (0.0 <= tval <= 1.0):
            raise ValueError("posterior_t values must be in [0, 1].")
    return posterior_t_values


def _build_eval_cases(
    sample_types: Sequence[SampleType],
    outer_steps: Sequence[int],
    inner_steps: Sequence[int],
    ema_factors: Sequence[float],
    network_slot: str,
    posterior_t_values: Optional[Sequence[float]] = None,
) -> List[EvalCase]:
    if not sample_types or not outer_steps or not inner_steps or not ema_factors:
        return []

    cases: List[EvalCase] = []
    sample_types_tuple = tuple(sample_types)
    t_values = tuple(posterior_t_values) if posterior_t_values is not None else (None,)

    for posterior_t in t_values:
        for outer_step in outer_steps:
            for inner_step in inner_steps:
                for ema_factor in ema_factors:
                    cases.append(
                        EvalCase(
                            sample_types=sample_types_tuple,
                            outer_step=int(outer_step),
                            inner_step=int(inner_step),
                            ema_factor=float(ema_factor),
                            network_slot=network_slot,
                            posterior_t=None if posterior_t is None else float(posterior_t),
                        )
                    )
    return cases


def _compute_plot_idxs(vis_mode: str, outer_step: int, inner_step: int) -> List[int]:
    if vis_mode == "marginal":
        return [0] + [i * inner_step for i in range(1, outer_step + 1)]
    if vis_mode == "full":
        return [0] + [
            i * inner_step + j + 1
            for i in range(outer_step)
            for j in range(inner_step)
        ]
    raise ValueError(f"Unknown vis_mode={vis_mode!r}")


def _build_fid_records(
    case: EvalCase, curr_fids: Dict[str, Optional[float]]
) -> List[FidRecord]:
    fid_records: List[FidRecord] = []
    for sample_type in case.sample_types:
        sample_name = sample_type.name
        if sample_name not in curr_fids:
            continue
        fid_records.append(
            FidRecord(
                sample_type=sample_name,
                ema_factor=case.ema_factor,
                outer_step=case.outer_step,
                inner_step=case.inner_step,
                fid=curr_fids[sample_name],
                network_slot=case.network_slot,
                posterior_t=case.posterior_t,
            )
        )
    return fid_records


def _format_fid_message(record: FidRecord) -> str:
    if record.is_posterior:
        return (
            "Posterior FID="
            f"{record.fid} ({record.sample_type}, outer={record.outer_step}, "
            f"inner={record.inner_step}, ema={record.ema_factor}, t={record.posterior_t})"
        )
    return (
        f"fid/{record.sample_type}_{record.outer_step}_{record.inner_step}_"
        f"ema={record.ema_factor} = {record.fid}"
    )


def _persist_figure_artifacts(
    figure_artifacts: Sequence[FigureArtifact],
    output_folder: Optional[str],
    wb,
) -> None:
    for artifact in figure_artifacts:
        if output_folder is None:
            raise ValueError("output_folder is required when figure artifacts are produced.")
        artifact.figure.savefig(os.path.join(output_folder, artifact.output_name))
        if wb is not None:
            wb.log({artifact.wandb_key: wandb.Image(artifact.figure)})
        plt.close(artifact.figure)


def _log_fid_table_to_wandb(fid_records: Sequence[FidRecord], wb) -> None:
    if wb is None or not fid_records:
        return

    table = wandb.Table(
        columns=[
            "kind",
            "sample_type",
            "ema_factor",
            "network_slot",
            "outer_steps",
            "inner_steps",
            "total_steps",
            "posterior_t",
            "fid",
        ]
    )
    for record in fid_records:
        table.add_data(
            "posterior" if record.is_posterior else "marginal",
            record.sample_type,
            record.ema_factor,
            record.network_slot,
            record.outer_step,
            record.inner_step,
            record.total_steps,
            record.posterior_t,
            record.fid,
        )
    wb.log({"fid_table/all_records": table})


def _run_eval_cases(
    cases: Sequence[EvalCase],
    runtime: EvalRuntime,
    prng_key,
    executor: Callable[[EvalCase, EvalRuntime, Any], EvalCaseResult],
) -> Tuple[List[FidRecord], Any]:
    fid_records: List[FidRecord] = []
    curr_key = prng_key

    for case in cases:
        case_result = executor(case, runtime, curr_key)
        fid_records.extend(case_result.fid_records)
        _persist_figure_artifacts(
            case_result.figure_artifacts,
            runtime.output_folder,
            runtime.wb,
        )
        curr_key = case_result.next_prng_key
        gc.collect()

    return fid_records, curr_key


def _run_marginal_case(case: EvalCase, runtime: EvalRuntime, prng_key) -> EvalCaseResult:
    plot_idxs = _compute_plot_idxs(runtime.vis_mode, case.outer_step, case.inner_step)
    titles = [sample_type.name for sample_type in case.sample_types]
    nrows = len(case.sample_types)
    ncols = len(plot_idxs)

    curr_fids: Dict[str, Optional[float]] = {}
    xhats = None
    xfinals = None
    if not runtime.metrics_only:
        xhats = np.zeros(
            (
                nrows,
                case.total_steps + 1,
                runtime.plot_dims[1],
                runtime.plot_dims[2],
                runtime.plot_dims[0],
            ),
            dtype=np.float32,
        )
        xfinals = np.zeros(
            (
                nrows,
                runtime.cfg.logging.plot_bs,
                runtime.plot_dims[1],
                runtime.plot_dims[2],
                runtime.plot_dims[0],
            ),
            dtype=np.float32,
        )

    for idx, sample_type in enumerate(case.sample_types):
        batch_xfinals, batch_xhats, curr_fid = sample_from_ckpt(
            runtime.cfg,
            runtime.statics,
            runtime.train_state,
            prng_key,
            case.outer_step,
            case.inner_step,
            sample_type,
            runtime.comp_fid,
            runtime.fid_samples,
            runtime.fid_bs,
            ema_factor=case.ema_factor,
            network_slot=case.network_slot,
            compute_visuals=not runtime.metrics_only,
        )
        curr_fids[sample_type.name] = curr_fid
        if not runtime.metrics_only:
            xfinals[idx] = batch_xfinals
            xhats[idx] = batch_xhats

    fid_records = _build_fid_records(case, curr_fids)
    for record in fid_records:
        print(_format_fid_message(record))

    figure_artifacts: List[FigureArtifact] = []
    if not runtime.metrics_only:
        traj_fig = make_traj_plot(nrows, ncols, xhats, titles, plot_idxs)
        samples_fig = make_sample_plot(
            nrows,
            runtime.cfg.logging.plot_bs,
            xfinals,
            titles,
        )
        figure_artifacts = [
            FigureArtifact(
                figure=samples_fig,
                output_name=(
                    f"samples_ema={case.ema_factor}_{case.outer_step}_{case.inner_step}.pdf"
                ),
                wandb_key=(
                    f"plots/samples_ema={case.ema_factor}_{case.outer_step}_{case.inner_step}"
                ),
            ),
            FigureArtifact(
                figure=traj_fig,
                output_name=(
                    f"trajs_ema={case.ema_factor}_{case.outer_step}_{case.inner_step}.pdf"
                ),
                wandb_key=(
                    f"plots/trajs_ema={case.ema_factor}_{case.outer_step}_{case.inner_step}"
                ),
            ),
        ]
        del xhats, xfinals

    return EvalCaseResult(
        fid_records=fid_records,
        figure_artifacts=figure_artifacts,
        next_prng_key=prng_key,
    )


def _run_posterior_case(case: EvalCase, runtime: EvalRuntime, prng_key) -> EvalCaseResult:
    if case.posterior_t is None:
        raise ValueError("Posterior eval case requires posterior_t.")

    curr_fids: Dict[str, Optional[float]] = {}
    curr_key = prng_key
    if runtime.comp_fid:
        for sample_type in case.sample_types:
            sample_fn = make_posterior_sample_fn(
                runtime.cfg,
                runtime.statics,
                sample_type,
                case.outer_step,
                case.inner_step,
                case.posterior_t,
                network_slot=case.network_slot,
            )
            params = get_params(
                runtime.cfg,
                runtime.statics,
                runtime.train_state,
                sample_type,
                ema_factor_override=case.ema_factor,
                network_slot=case.network_slot,
            )
            curr_key, subkey = jax.random.split(curr_key)
            curr_fids[sample_type.name] = calc_fid(
                runtime.cfg,
                runtime.statics.inception_fn,
                runtime.statics.decode_fn,
                sample_fn,
                subkey,
                params,
                n_samples=runtime.fid_samples,
                bs=runtime.fid_bs,
            )

    fid_records = _build_fid_records(case, curr_fids)
    for record in fid_records:
        print(_format_fid_message(record))

    figure_artifacts: List[FigureArtifact] = []
    if not runtime.metrics_only:
        curr_key, sample_fig, multiseed_fig = make_posterior_sample_plot(
            runtime.cfg,
            runtime.statics,
            runtime.train_state,
            case.outer_step,
            case.inner_step,
            case.posterior_t,
            case.ema_factor,
            curr_key,
            case.sample_types,
            network_slot=case.network_slot,
        )
        figure_artifacts = [
            FigureArtifact(
                figure=sample_fig,
                output_name=(
                    "posterior_recovery_"
                    f"{case.posterior_t}_{case.outer_step}_{case.inner_step}_{case.ema_factor}.pdf"
                ),
                wandb_key=(
                    "posterior_samples/"
                    f"t={case.posterior_t}_outer={case.outer_step}_inner={case.inner_step}_"
                    f"ema={case.ema_factor}"
                ),
            ),
            FigureArtifact(
                figure=multiseed_fig,
                output_name=(
                    "posterior_single_"
                    f"{case.posterior_t}_{case.outer_step}_{case.inner_step}_{case.ema_factor}.pdf"
                ),
                wandb_key=(
                    "posterior_single/"
                    f"t={case.posterior_t}_outer={case.outer_step}_inner={case.inner_step}_"
                    f"ema={case.ema_factor}"
                ),
            ),
        ]

    return EvalCaseResult(
        fid_records=fid_records,
        figure_artifacts=figure_artifacts,
        next_prng_key=curr_key,
    )


@click.command()
@click.option("--cfg_path", required=True)
@click.option("--output_folder")
@click.option("--slurm_id", type=int, default=0)
@click.option(
    "--outer_steps", type=str, help="comma separated list of outer steps", default=None
)
@click.option(
    "--inner_steps", type=str, help="comma separated list of inner steps", default=None
)
@click.option("--ckpt_path", type=str, help="checkpoint path", default="")
@click.option(
    "--vis_mode",
    type=click.Choice(["full", "marginal"], case_sensitive=False),
    default="marginal",
)
@click.option("--comp_fid", is_flag=True)
@click.option("--fid_samples", type=int, default=50000)
@click.option("--fid_bs", type=int, default=128)
@click.option("--use_wandb", is_flag=True)
@click.option("--ema_factors", type=str, default=None)
@click.option(
    "--sample_types",
    default=None,
    type=str,
    help="comma separated list of sample types to evaluate",
)
@click.option(
    "--posterior_t",
    type=str,
    default=None,
    help=(
        "Comma separated list of t values for posterior sampling from t to t_prime=1."
    ),
)
@click.option(
    "--posterior_outer_steps",
    type=str,
    default=None,
    help="Comma separated list of outer steps for posterior sampling.",
)
@click.option(
    "--posterior_inner_steps",
    type=str,
    default=None,
    help="Comma separated list of inner steps for posterior sampling.",
)
@click.option(
    "--posterior_sample_types",
    type=str,
    default=None,
    help="Comma separated list of posterior sample types to evaluate.",
)
@click.option(
    "--base_network_slot",
    type=click.Choice(["auto", "main", "sup"], case_sensitive=False),
    default="auto",
)
@click.option(
    "--posterior_network_slot",
    type=click.Choice(["auto", "main", "sup"], case_sensitive=False),
    default="auto",
)
@click.option("--metrics_only", is_flag=True)
def main(
    cfg_path,
    output_folder,
    slurm_id,
    outer_steps,
    inner_steps,
    ckpt_path,
    vis_mode,
    comp_fid,
    fid_samples,
    fid_bs,
    use_wandb,
    ema_factors,
    sample_types,
    posterior_t,
    posterior_outer_steps,
    posterior_inner_steps,
    posterior_sample_types,
    base_network_slot,
    posterior_network_slot,
    metrics_only,
):
    # argument validation
    if posterior_t is not None:
        if posterior_outer_steps is None or posterior_inner_steps is None:
            raise ValueError(
                "posterior_outer_steps and posterior_inner_steps are required when "
                "posterior_t is set."
            )
        if posterior_sample_types is None:
            raise ValueError(
                "posterior_sample_types is required when posterior_t is set."
            )

    cfg = importlib.import_module(cfg_path).get_config(slurm_id, None)
    if ckpt_path:
        cfg.network.load_path = ckpt_path
    if base_network_slot.lower() != "auto":
        cfg.base_network_slot = base_network_slot.lower()
    if posterior_network_slot.lower() != "auto":
        cfg.posterior_network_slot = posterior_network_slot.lower()
    prng_key = jax.random.PRNGKey(cfg.training.seed)
    cfg, statics, train_state, prng_key = setup_state(cfg, prng_key)

    plot_dims = latent_utils.get_pixel_image_dims(cfg)

    # Init wandb
    wb = None
    if use_wandb and jax.process_index() == 0:
        wb = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            group="eval",
            config={
                "cfg_path": cfg_path,
                "dataset_location": cfg.problem.dataset_location,
                "slurm_id": slurm_id,
                "vis_mode": vis_mode,
                "outer_steps": outer_steps,
                "inner_steps": inner_steps,
                "ckpt_path": ckpt_path,
                "posterior_sample_types": posterior_sample_types,
                "base_network_slot": cfg.base_network_slot,
                "posterior_network_slot": cfg.posterior_network_slot,
                "fid": comp_fid,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
                "cfg": cfg.to_dict(),
            },
        )

    print("Configuration:")
    print(f"{vis_mode=}")
    print(f"{outer_steps=}")
    print(f"{inner_steps=}")
    print(f"{sample_types=}")
    print(f"{posterior_sample_types=}")
    print(f"base_network_slot={cfg.base_network_slot}")
    print(f"posterior_network_slot={cfg.posterior_network_slot}")

    outer_steps_list = _parse_int_list(outer_steps)
    inner_steps_list = _parse_int_list(inner_steps)
    sample_types_list = _parse_sample_types(sample_types)
    posterior_t_list = _parse_posterior_t_list(posterior_t)
    posterior_outer_steps_list = _parse_int_list(posterior_outer_steps)
    posterior_inner_steps_list = _parse_int_list(posterior_inner_steps)
    posterior_sample_types_list = _parse_sample_types(posterior_sample_types)
    ema_factors_list = _parse_ema_factors(ema_factors, cfg.training.ema_facs)

    if not output_folder and not metrics_only:
        raise ValueError("Need output_folder to store evaluation artifacts.")
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    plt.close("all")
    sns.set_palette("deep")

    runtime = EvalRuntime(
        cfg=cfg,
        statics=statics,
        train_state=train_state,
        comp_fid=comp_fid,
        fid_samples=fid_samples,
        fid_bs=fid_bs,
        metrics_only=metrics_only,
        output_folder=output_folder,
        wb=wb,
        plot_dims=plot_dims,
        vis_mode=vis_mode,
    )

    all_fid_records: List[FidRecord] = []
    posterior_cases = _build_eval_cases(
        posterior_sample_types_list,
        posterior_outer_steps_list,
        posterior_inner_steps_list,
        ema_factors_list,
        cfg.posterior_network_slot,
        posterior_t_values=posterior_t_list,
    )
    marginal_cases = _build_eval_cases(
        sample_types_list,
        outer_steps_list,
        inner_steps_list,
        ema_factors_list,
        cfg.base_network_slot,
    )

    posterior_fid_records, prng_key = _run_eval_cases(
        posterior_cases,
        runtime,
        prng_key,
        _run_posterior_case,
    )
    all_fid_records.extend(posterior_fid_records)

    marginal_fid_records, prng_key = _run_eval_cases(
        marginal_cases,
        runtime,
        prng_key,
        _run_marginal_case,
    )
    all_fid_records.extend(marginal_fid_records)

    if wb is not None:
        if comp_fid and all_fid_records:
            _log_fid_table_to_wandb(all_fid_records, wb)

        wb.finish()
    return
if __name__ == "__main__":
    main()
