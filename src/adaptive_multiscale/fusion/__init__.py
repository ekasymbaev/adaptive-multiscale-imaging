"""Probability-level fusion primitives for selective native inspection."""

from adaptive_multiscale.fusion.adaptive import (
    descending_ranks,
    feather_tile_weight,
    fuse_probability_tiles,
    fuse_selected_tiles,
    oracle_gain_rankings,
)
from adaptive_multiscale.fusion.online import (
    SelectedTilePrediction,
    infer_selected_tile_probabilities,
    synchronize_device,
    timed_call,
    validate_selected_indices,
    warm_up_segmentation_models,
)

__all__ = [
    "descending_ranks",
    "feather_tile_weight",
    "fuse_probability_tiles",
    "fuse_selected_tiles",
    "oracle_gain_rankings",
    "SelectedTilePrediction",
    "infer_selected_tile_probabilities",
    "synchronize_device",
    "timed_call",
    "validate_selected_indices",
    "warm_up_segmentation_models",
]
