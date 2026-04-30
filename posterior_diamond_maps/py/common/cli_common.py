"""Shared CLI parsing helpers for launcher prompt selection."""

from __future__ import annotations

from typing import Optional, Sequence

from common import prompt_sets


def normalize_choice(value: str, choices: Sequence[str], *, option_name: str) -> str:
    normalized = str(value).lower()
    if normalized not in choices:
        raise ValueError(
            f"{option_name} must be one of {tuple(choices)}; got {value!r}"
        )
    return normalized


def normalize_csv_arg(raw_value: str, *, sep: str = ",") -> str:
    return sep.join(piece.strip() for piece in raw_value.split(",") if piece.strip())


def parse_csv_ints(raw_value: str, *, option_name: str) -> list[int]:
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        raise ValueError(f"{option_name} must contain at least one integer.")
    return [int(value) for value in values]


def parse_csv_floats(raw_value: str, *, option_name: str) -> list[float]:
    values = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not values:
        raise ValueError(f"{option_name} must contain at least one float.")
    return [float(value) for value in values]


def add_prompt_selection_args(parser, *, prompt_set_required: bool) -> None:
    parser.add_argument("--prompt_set", type=str, required=prompt_set_required)
    parser.add_argument("--prompt_index", type=int, default=None)


def selected_prompt_entries(prompt_set: str, prompt_index: Optional[int] = None):
    if prompt_set not in prompt_sets.PROMPT_SETS:
        raise ValueError(
            f"Unknown prompt_set {prompt_set!r}. "
            f"Expected one of {tuple(prompt_sets.PROMPT_SETS)}."
        )
    prompts = prompt_sets.PROMPT_SETS[prompt_set]
    if prompt_index is None:
        return list(enumerate(prompts))
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise ValueError(
            f"prompt_index={prompt_index} is out of range for "
            f"{prompt_set!r} with {len(prompts)} prompts."
        )
    return [(prompt_index, prompts[prompt_index])]
