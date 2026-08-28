"""Online native-tile inference helpers with device-synchronized timing."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
from torch import nn


T = TypeVar("T")


@dataclass(frozen=True)
class SelectedTilePrediction:
    """Probabilities produced for exactly the requested native tiles."""

    probabilities: dict[int, np.ndarray]
    tiles_processed: int
    batches_executed: int


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work before reading a wall-clock timer."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def timed_call(function: Callable[[], T], device: torch.device) -> tuple[T, float]:
    """Run one callable and return its output with synchronized elapsed seconds."""

    synchronize_device(device)
    start = time.perf_counter()
    result = function()
    synchronize_device(device)
    return result, time.perf_counter() - start


def validate_selected_indices(
    selected_tile_indices: Iterable[int], tile_count: int
) -> list[int]:
    """Return a validated ordered list of unique tile indices."""

    selected = [int(value) for value in selected_tile_indices]
    if not selected:
        raise ValueError("At least one fine tile must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError("Selected tile indices must be unique")
    if any(value < 0 or value >= tile_count for value in selected):
        raise ValueError("Selected tile index is outside the native grid")
    return selected


@torch.inference_mode()
def infer_selected_tile_probabilities(
    model: nn.Module,
    native_image: np.ndarray,
    selected_tile_indices: Iterable[int],
    normalization_mean: float,
    normalization_std: float,
    tile_size: int,
    batch_size: int,
    device: torch.device,
) -> SelectedTilePrediction:
    """Run the fine model on selected tiles only and return no unselected output."""

    image = np.asarray(native_image)
    if image.ndim != 2 or image.shape[0] % tile_size or image.shape[1] % tile_size:
        raise ValueError("Native image must be one 2D array divisible by tile size")
    if normalization_std <= 0.0 or batch_size <= 0:
        raise ValueError("Normalization standard deviation and batch size must be positive")
    grid_rows = image.shape[0] // tile_size
    grid_columns = image.shape[1] // tile_size
    selected = validate_selected_indices(
        selected_tile_indices, grid_rows * grid_columns
    )

    normalized_tiles: list[np.ndarray] = []
    for tile_index in selected:
        tile_row, tile_column = divmod(tile_index, grid_columns)
        y0, x0 = tile_row * tile_size, tile_column * tile_size
        tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
        values = tile.astype(np.float32) / 255.0
        normalized_tiles.append(((values - normalization_mean) / normalization_std)[None])

    model.eval()
    probabilities: dict[int, np.ndarray] = {}
    batches = 0
    for start in range(0, len(selected), batch_size):
        indices = selected[start : start + batch_size]
        tensor = torch.from_numpy(
            np.stack(normalized_tiles[start : start + batch_size])
        ).to(device)
        output = torch.sigmoid(model(tensor))[:, 0].detach().cpu().numpy()
        for tile_index, probability in zip(indices, output, strict=True):
            probabilities[tile_index] = probability.astype(np.float32, copy=False)
        batches += 1

    if set(probabilities) != set(selected):
        raise RuntimeError("Fine inference did not return exactly the selected tiles")
    return SelectedTilePrediction(
        probabilities=probabilities,
        tiles_processed=len(selected),
        batches_executed=batches,
    )


@torch.inference_mode()
def warm_up_segmentation_models(
    coarse_model: nn.Module,
    fine_model: nn.Module,
    coarse_shape: tuple[int, int],
    tile_size: int,
    fine_batch_shapes: Iterable[int],
    device: torch.device,
) -> None:
    """Compile common accelerator shapes before any reported timings."""

    coarse_model.eval()
    fine_model.eval()
    coarse_model(torch.zeros((1, 1, *coarse_shape), device=device))
    for batch_size in sorted(set(int(value) for value in fine_batch_shapes)):
        if batch_size <= 0:
            raise ValueError("Warm-up batch sizes must be positive")
        fine_model(
            torch.zeros((batch_size, 1, tile_size, tile_size), device=device)
        )
    synchronize_device(device)
