"""Tile-level uncertainty ranking and evaluation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from adaptive_multiscale.uncertainty.evaluation import safe_pearson


@dataclass(frozen=True)
class TileGrid:
    """Exact correspondence between native and coarse non-overlapping grids."""

    native_height: int
    native_width: int
    coarse_height: int
    coarse_width: int
    native_tile_size: int

    def __post_init__(self) -> None:
        values = (
            self.native_height,
            self.native_width,
            self.coarse_height,
            self.coarse_width,
            self.native_tile_size,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Grid dimensions and tile size must be positive")
        if self.native_height % self.native_tile_size:
            raise ValueError("Native height is not divisible by tile size")
        if self.native_width % self.native_tile_size:
            raise ValueError("Native width is not divisible by tile size")
        if self.native_height % self.coarse_height:
            raise ValueError("Native/coarse height ratio must be integral")
        if self.native_width % self.coarse_width:
            raise ValueError("Native/coarse width ratio must be integral")
        if self.native_height // self.coarse_height != self.native_width // self.coarse_width:
            raise ValueError("Native-to-coarse scale must be isotropic")
        if self.native_tile_size % self.scale_factor:
            raise ValueError("Native tile size must map to an integral coarse size")
        if self.coarse_height % self.coarse_tile_size:
            raise ValueError("Coarse height is not divisible by mapped tile size")
        if self.coarse_width % self.coarse_tile_size:
            raise ValueError("Coarse width is not divisible by mapped tile size")

    @property
    def scale_factor(self) -> int:
        return self.native_height // self.coarse_height

    @property
    def coarse_tile_size(self) -> int:
        return self.native_tile_size // self.scale_factor

    @property
    def rows(self) -> int:
        return self.native_height // self.native_tile_size

    @property
    def columns(self) -> int:
        return self.native_width // self.native_tile_size

    @property
    def tile_count(self) -> int:
        return self.rows * self.columns


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Ranking scores must be a non-empty vector")
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores contain non-finite values")
    order = np.lexsort((np.arange(scores.size), -scores))
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[order] = np.arange(1, scores.size + 1)
    return ranks


def uncertainty_tile_rankings(
    entropy: np.ndarray,
    variance: np.ndarray,
    grid: TileGrid,
    percentile: float = 90.0,
) -> pd.DataFrame:
    """Create uncertainty rankings without accepting labels or predictions."""

    entropy_values = np.asarray(entropy, dtype=np.float32)
    variance_values = np.asarray(variance, dtype=np.float32)
    expected_shape = (grid.coarse_height, grid.coarse_width)
    if entropy_values.shape != expected_shape or variance_values.shape != expected_shape:
        raise ValueError(
            f"Expected uncertainty maps with shape {expected_shape}, received "
            f"{entropy_values.shape} and {variance_values.shape}"
        )
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("Percentile must be between zero and 100")
    if not np.isfinite(entropy_values).all() or not np.isfinite(variance_values).all():
        raise ValueError("Uncertainty maps contain non-finite values")

    records: list[dict[str, object]] = []
    for tile_row in range(grid.rows):
        for tile_column in range(grid.columns):
            tile_index = tile_row * grid.columns + tile_column
            coarse_y0 = tile_row * grid.coarse_tile_size
            coarse_x0 = tile_column * grid.coarse_tile_size
            coarse_y1 = coarse_y0 + grid.coarse_tile_size
            coarse_x1 = coarse_x0 + grid.coarse_tile_size
            native_y0 = tile_row * grid.native_tile_size
            native_x0 = tile_column * grid.native_tile_size
            entropy_block = entropy_values[coarse_y0:coarse_y1, coarse_x0:coarse_x1]
            variance_block = variance_values[coarse_y0:coarse_y1, coarse_x0:coarse_x1]
            records.append(
                {
                    "tile_index": tile_index,
                    "tile_row": tile_row,
                    "tile_column": tile_column,
                    "native_x0": native_x0,
                    "native_y0": native_y0,
                    "native_x1": native_x0 + grid.native_tile_size,
                    "native_y1": native_y0 + grid.native_tile_size,
                    "coarse_x0": coarse_x0,
                    "coarse_y0": coarse_y0,
                    "coarse_x1": coarse_x1,
                    "coarse_y1": coarse_y1,
                    "entropy_p90_bits": float(np.percentile(entropy_block, percentile)),
                    "variance_p90": float(np.percentile(variance_block, percentile)),
                }
            )
    result = pd.DataFrame(records)
    result["entropy_rank"] = _descending_ranks(result["entropy_p90_bits"].to_numpy())
    result["variance_rank"] = _descending_ranks(result["variance_p90"].to_numpy())
    return result


def attach_error_evaluation(
    rankings: pd.DataFrame,
    error_map: np.ndarray,
    grid: TileGrid,
) -> pd.DataFrame:
    """Attach label-derived error measurements after uncertainty ranking."""

    errors = np.asarray(error_map, dtype=bool)
    expected_shape = (grid.coarse_height, grid.coarse_width)
    if errors.shape != expected_shape:
        raise ValueError(f"Expected error map shape {expected_shape}, received {errors.shape}")
    if len(rankings) != grid.tile_count:
        raise ValueError(f"Expected {grid.tile_count} ranking rows, found {len(rankings)}")
    result = rankings.copy()
    error_pixels = []
    for row in result.itertuples(index=False):
        block = errors[row.coarse_y0 : row.coarse_y1, row.coarse_x0 : row.coarse_x1]
        error_pixels.append(int(block.sum()))
    result["error_pixels"] = error_pixels
    result["coarse_pixels"] = grid.coarse_tile_size**2
    result["error_fraction"] = result["error_pixels"] / result["coarse_pixels"]
    result["oracle_error_rank"] = _descending_ranks(result["error_pixels"].to_numpy())
    return result


def selection_metrics(
    tiles: pd.DataFrame,
    rank_column: str,
    budgets: Iterable[int],
) -> pd.DataFrame:
    """Measure the fraction of a fixed coarse error map captured by a ranking."""

    if rank_column not in tiles:
        raise ValueError(f"Missing ranking column: {rank_column}")
    total_errors = int(tiles["error_pixels"].sum())
    total_pixels = int(tiles["coarse_pixels"].sum())
    if total_errors <= 0:
        raise ValueError("At least one coarse segmentation error is required")
    rows = []
    for budget in budgets:
        if not 1 <= int(budget) <= len(tiles):
            raise ValueError(f"Budget {budget} is outside [1, {len(tiles)}]")
        selected = tiles[tiles[rank_column] <= int(budget)]
        if len(selected) != int(budget):
            raise ValueError(f"Ranking {rank_column} did not select exactly {budget} tiles")
        selected_errors = int(selected["error_pixels"].sum())
        selected_pixels = int(selected["coarse_pixels"].sum())
        coverage = selected_pixels / total_pixels
        capture = selected_errors / total_errors
        selected_error_rate = selected_errors / selected_pixels
        overall_error_rate = total_errors / total_pixels
        rows.append(
            {
                "budget_k": int(budget),
                "coverage_fraction": coverage,
                "selected_error_pixels": selected_errors,
                "total_error_pixels": total_errors,
                "error_capture": capture,
                "selected_error_rate": selected_error_rate,
                "error_density_enrichment": selected_error_rate / overall_error_rate,
            }
        )
    return pd.DataFrame(rows)


def random_rank_matrix(
    tile_count: int,
    trial_count: int,
    seed_base: int,
    sample_number: int,
) -> np.ndarray:
    """Return reproducible independent random rankings for one image."""

    if tile_count <= 0 or trial_count <= 0:
        raise ValueError("Tile and trial counts must be positive")
    rows = []
    for trial_index in range(trial_count):
        generator = np.random.default_rng(
            np.random.SeedSequence([seed_base, trial_index, sample_number])
        )
        order = generator.permutation(tile_count)
        ranks = np.empty(tile_count, dtype=np.int64)
        ranks[order] = np.arange(1, tile_count + 1)
        rows.append(ranks)
    return np.stack(rows, axis=0)


def tile_error_correlations(tiles: pd.DataFrame) -> dict[str, float]:
    errors = tiles["error_fraction"].to_numpy(dtype=np.float64)
    entropy = tiles["entropy_p90_bits"].to_numpy(dtype=np.float64)
    variance = tiles["variance_p90"].to_numpy(dtype=np.float64)
    return {
        "entropy_error_pearson": safe_pearson(entropy, errors),
        "entropy_error_spearman": float(spearmanr(entropy, errors).statistic),
        "variance_error_pearson": safe_pearson(variance, errors),
        "variance_error_spearman": float(spearmanr(variance, errors).statistic),
    }


def paired_bootstrap_interval(
    differences: np.ndarray,
    confidence_level: float,
    resamples: int,
    random_seed: int,
) -> tuple[float, float, float]:
    """Return mean and percentile CI from paired image-level differences."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must contain at least two finite differences")
    if not 0.0 < confidence_level < 1.0 or resamples <= 0:
        raise ValueError("Invalid bootstrap configuration")
    generator = np.random.default_rng(random_seed)
    indices = generator.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(values.mean()), float(low), float(high)
