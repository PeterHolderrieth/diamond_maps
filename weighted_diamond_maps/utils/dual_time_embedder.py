"""Dual-timestep embedding patch for FLUX Flow Map LoRA inference."""

from __future__ import annotations

import copy
from typing import Any

import torch


class DualTimeEmbedder(torch.nn.Module):
    """Wrap a FLUX time-text embedder so it can accept `[t, t_next]` inputs."""

    def __init__(self, original_embedder: Any):
        super().__init__()
        self.original_embedder = original_embedder
        self.second_embedder = copy.deepcopy(original_embedder)

    def forward(self, timestep, guidance, pooled_projections):
        if len(timestep.shape) >= 2 and timestep.shape[-1] == 2:
            t1, t2 = timestep.unbind(dim=-1)
            emb1 = self.original_embedder(t1, guidance, pooled_projections)
            emb2 = self.second_embedder(t2, guidance, pooled_projections)
            return (emb1 + emb2) / 2
        return self.original_embedder(timestep, guidance, pooled_projections)


def add_dual_time_embedder(single_time_flux_transformer):
    """Patch a FLUX transformer in-place for two-timestep Flow Map inputs."""
    single_time_flux_transformer.time_text_embed = DualTimeEmbedder(
        single_time_flux_transformer.time_text_embed
    )
    return single_time_flux_transformer
