"""Spatial selection utilities for the adaptive multiscale prototype."""

from .tile_ranking import (
    TileGrid,
    attach_error_evaluation,
    paired_bootstrap_interval,
    random_rank_matrix,
    selection_metrics,
    uncertainty_tile_rankings,
)

__all__ = [
    "TileGrid",
    "attach_error_evaluation",
    "paired_bootstrap_interval",
    "random_rank_matrix",
    "selection_metrics",
    "uncertainty_tile_rankings",
]
