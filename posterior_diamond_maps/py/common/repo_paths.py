"""Repository-relative path helpers for datasets, checkpoints, and metrics."""

import os
from pathlib import Path


_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> str:
    return str(_DEFAULT_REPO_ROOT)


def repo_path(*parts: str) -> str:
    return os.path.join(repo_root(), *parts)


def datasets_path(*parts: str) -> str:
    return repo_path("datasets", *parts)


def ckpt_path(*parts: str) -> str:
    return repo_path("ckpt", *parts)


def metric_models_path(*parts: str) -> str:
    return repo_path("metric_models", *parts)
