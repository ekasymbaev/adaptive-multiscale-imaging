"""Seam-aware probability fusion for non-overlapping native image tiles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def descending_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic one-based descending ranks with index tie-breaking."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Ranking scores must be a non-empty vector")
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores contain non-finite values")
    order = np.lexsort((np.arange(scores.size), -scores))
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def feather_tile_weight(
    tile_size: int,
    blend_width: int,
    tile_row: int,
    tile_column: int,
    grid_rows: int,
    grid_columns: int,
) -> np.ndarray:
    """Create a cosine-squared blend window that tapers only at internal seams."""

    values = (tile_size, blend_width, grid_rows, grid_columns)
    if any(int(value) <= 0 for value in values):
        raise ValueError("Tile, blend, and grid dimensions must be positive")
    if 2 * blend_width >= tile_size:
        raise ValueError("Blend width must be less than half the tile size")
    if not 0 <= tile_row < grid_rows or not 0 <= tile_column < grid_columns:
        raise ValueError("Tile coordinates are outside the grid")

    ramp = np.sin(
        0.5 * np.pi * np.clip(np.arange(tile_size) / blend_width, 0.0, 1.0)
    ) ** 2
    horizontal = np.ones(tile_size, dtype=np.float32)
    vertical = np.ones(tile_size, dtype=np.float32)
    if tile_column > 0:
        horizontal = np.minimum(horizontal, ramp)
    if tile_column < grid_columns - 1:
        horizontal = np.minimum(horizontal, ramp[::-1])
    if tile_row > 0:
        vertical = np.minimum(vertical, ramp)
    if tile_row < grid_rows - 1:
        vertical = np.minimum(vertical, ramp[::-1])
    return np.outer(vertical, horizontal).astype(np.float32)


def fuse_selected_tiles(
    coarse_probability: np.ndarray,
    fine_probability: np.ndarray,
    selected_tile_indices: Iterable[int],
    tile_size: int,
    blend_width: int,
    rule: str = "feathered",
) -> np.ndarray:
    """Replace selected coarse regions with hard or feathered fine probabilities."""

    coarse = np.asarray(coarse_probability, dtype=np.float32)
    fine = np.asarray(fine_probability, dtype=np.float32)
    if coarse.ndim != 2 or coarse.shape != fine.shape:
        raise ValueError("Coarse and fine probabilities must share one 2D shape")
    if not np.isfinite(coarse).all() or not np.isfinite(fine).all():
        raise ValueError("Probability maps contain non-finite values")
    if coarse.shape[0] % tile_size or coarse.shape[1] % tile_size:
        raise ValueError("Probability-map dimensions must be divisible by tile size")
    if rule not in {"feathered", "hard"}:
        raise ValueError(f"Unsupported fusion rule: {rule}")

    grid_rows = coarse.shape[0] // tile_size
    grid_columns = coarse.shape[1] // tile_size
    tile_count = grid_rows * grid_columns
    selected = [int(value) for value in selected_tile_indices]
    if len(selected) != len(set(selected)):
        raise ValueError("Selected tile indices must be unique")
    if any(value < 0 or value >= tile_count for value in selected):
        raise ValueError("Selected tile index is outside the image grid")

    result = coarse.copy()
    for tile_index in selected:
        tile_row, tile_column = divmod(tile_index, grid_columns)
        y0, x0 = tile_row * tile_size, tile_column * tile_size
        y1, x1 = y0 + tile_size, x0 + tile_size
        if rule == "hard":
            weight = 1.0
        else:
            weight = feather_tile_weight(
                tile_size,
                blend_width,
                tile_row,
                tile_column,
                grid_rows,
                grid_columns,
            )
        result[y0:y1, x0:x1] = (
            weight * fine[y0:y1, x0:x1]
            + (1.0 - weight) * coarse[y0:y1, x0:x1]
        )
    return result


def fuse_probability_tiles(
    coarse_probability: np.ndarray,
    fine_tiles: Mapping[int, np.ndarray],
    tile_size: int,
    blend_width: int,
    rule: str = "feathered",
) -> np.ndarray:
    """Fuse only the supplied fine-tile probabilities into a coarse map.

    Unlike :func:`fuse_selected_tiles`, this online variant never receives a
    full-resolution fine probability map. This makes it impossible for an
    unselected tile to influence the adaptive result.
    """

    coarse = np.asarray(coarse_probability, dtype=np.float32)
    if coarse.ndim != 2 or not np.isfinite(coarse).all():
        raise ValueError("Coarse probability must be one finite 2D array")
    if coarse.shape[0] % tile_size or coarse.shape[1] % tile_size:
        raise ValueError("Probability-map dimensions must be divisible by tile size")
    if rule not in {"feathered", "hard"}:
        raise ValueError(f"Unsupported fusion rule: {rule}")

    grid_rows = coarse.shape[0] // tile_size
    grid_columns = coarse.shape[1] // tile_size
    tile_count = grid_rows * grid_columns
    result = coarse.copy()
    for raw_index, raw_probability in fine_tiles.items():
        tile_index = int(raw_index)
        if tile_index < 0 or tile_index >= tile_count:
            raise ValueError("Selected tile index is outside the image grid")
        probability = np.asarray(raw_probability, dtype=np.float32)
        if probability.shape != (tile_size, tile_size):
            raise ValueError(
                f"Fine tile {tile_index} has shape {probability.shape}, "
                f"expected {(tile_size, tile_size)}"
            )
        if not np.isfinite(probability).all():
            raise ValueError(f"Fine tile {tile_index} contains non-finite values")
        tile_row, tile_column = divmod(tile_index, grid_columns)
        y0, x0 = tile_row * tile_size, tile_column * tile_size
        y1, x1 = y0 + tile_size, x0 + tile_size
        if rule == "hard":
            weight = 1.0
        else:
            weight = feather_tile_weight(
                tile_size,
                blend_width,
                tile_row,
                tile_column,
                grid_rows,
                grid_columns,
            )
        result[y0:y1, x0:x1] = (
            weight * probability
            + (1.0 - weight) * coarse[y0:y1, x0:x1]
        )
    return result


def oracle_gain_rankings(
    coarse_probability: np.ndarray,
    fine_probability: np.ndarray,
    target: np.ndarray,
    tile_size: int,
    blend_width: int,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Rank tiles by true feathered error reduction for evaluation only."""

    coarse = np.asarray(coarse_probability, dtype=np.float32)
    fine = np.asarray(fine_probability, dtype=np.float32)
    truth = np.asarray(target, dtype=bool)
    if coarse.shape != fine.shape or coarse.shape != truth.shape or coarse.ndim != 2:
        raise ValueError("Coarse, fine, and target arrays must share one 2D shape")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be within [0, 1]")
    if coarse.shape[0] % tile_size or coarse.shape[1] % tile_size:
        raise ValueError("Array dimensions must be divisible by tile size")

    grid_rows = coarse.shape[0] // tile_size
    grid_columns = coarse.shape[1] // tile_size
    records: list[dict[str, int]] = []
    for tile_row in range(grid_rows):
        for tile_column in range(grid_columns):
            tile_index = tile_row * grid_columns + tile_column
            y0, x0 = tile_row * tile_size, tile_column * tile_size
            y1, x1 = y0 + tile_size, x0 + tile_size
            weight = feather_tile_weight(
                tile_size,
                blend_width,
                tile_row,
                tile_column,
                grid_rows,
                grid_columns,
            )
            coarse_tile = coarse[y0:y1, x0:x1]
            candidate = weight * fine[y0:y1, x0:x1] + (1.0 - weight) * coarse_tile
            target_tile = truth[y0:y1, x0:x1]
            coarse_errors = int(np.count_nonzero((coarse_tile >= threshold) != target_tile))
            candidate_errors = int(np.count_nonzero((candidate >= threshold) != target_tile))
            records.append(
                {
                    "tile_index": tile_index,
                    "tile_row": tile_row,
                    "tile_column": tile_column,
                    "coarse_error_pixels": coarse_errors,
                    "feathered_error_pixels": candidate_errors,
                    "oracle_gain_pixels": coarse_errors - candidate_errors,
                }
            )
    result = pd.DataFrame(records)
    result["oracle_gain_rank"] = descending_ranks(
        result["oracle_gain_pixels"].to_numpy(dtype=np.float64)
    )
    return result
