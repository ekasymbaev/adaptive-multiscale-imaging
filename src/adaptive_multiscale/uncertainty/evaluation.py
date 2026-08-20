"""Evaluation-only comparisons between uncertainty and segmentation errors."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    first = np.asarray(x, dtype=np.float64).ravel()
    second = np.asarray(y, dtype=np.float64).ravel()
    if first.size != second.size or first.size < 2:
        raise ValueError("Correlation arrays must have equal length of at least two")
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def uncertainty_error_statistics(
    uncertainty: np.ndarray, error: np.ndarray
) -> dict[str, float]:
    """Score uncertainty as a pixel-wise detector of a fixed error map."""

    values = np.asarray(uncertainty, dtype=np.float32).ravel()
    errors = np.asarray(error, dtype=bool).ravel()
    if values.shape != errors.shape:
        raise ValueError("Uncertainty and error maps must have matching shapes")
    if not np.isfinite(values).all():
        raise ValueError("Uncertainty contains non-finite values")
    correct = ~errors
    if not errors.any() or not correct.any():
        raise ValueError("Both correct and incorrect pixels are required")
    correct_mean = float(values[correct].mean())
    incorrect_mean = float(values[errors].mean())
    return {
        "error_rate": float(errors.mean()),
        "correct_mean": correct_mean,
        "incorrect_mean": incorrect_mean,
        "incorrect_minus_correct": incorrect_mean - correct_mean,
        "incorrect_to_correct_ratio": incorrect_mean / max(correct_mean, 1e-12),
        "error_pearson": safe_pearson(values, errors.astype(np.float32)),
        "error_roc_auc": float(roc_auc_score(errors, values)),
        "error_average_precision": float(average_precision_score(errors, values)),
    }


def top_uncertainty_concentration(
    uncertainty: np.ndarray,
    error: np.ndarray,
    fractions: Iterable[float],
) -> dict[str, float]:
    """Measure how much error lies in exact top-uncertainty pixel fractions."""

    values = np.asarray(uncertainty, dtype=np.float32).ravel()
    errors = np.asarray(error, dtype=bool).ravel()
    if values.shape != errors.shape:
        raise ValueError("Uncertainty and error maps must have matching shapes")
    total_errors = int(errors.sum())
    if total_errors == 0:
        raise ValueError("At least one error pixel is required")
    overall_error_rate = total_errors / errors.size
    result: dict[str, float] = {}
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError("Top fractions must be between zero and one")
        selected_count = max(1, int(np.ceil(values.size * fraction)))
        selected = np.argpartition(values, values.size - selected_count)[
            values.size - selected_count :
        ]
        selected_errors = int(errors[selected].sum())
        selected_error_rate = selected_errors / selected_count
        label = f"top_{int(round(fraction * 100)):02d}pct"
        result[f"{label}_selected_fraction"] = selected_count / values.size
        result[f"{label}_error_capture"] = selected_errors / total_errors
        result[f"{label}_error_enrichment"] = (
            selected_error_rate / overall_error_rate
        )
        result[f"{label}_selected_error_rate"] = selected_error_rate
    return result


def local_region_table(
    entropy: np.ndarray,
    variance: np.ndarray,
    error: np.ndarray,
    region_size: int,
    metadata: dict[str, object],
) -> pd.DataFrame:
    """Aggregate maps into a fixed non-overlapping analysis grid."""

    entropy_values = np.asarray(entropy, dtype=np.float32)
    variance_values = np.asarray(variance, dtype=np.float32)
    error_values = np.asarray(error, dtype=bool)
    if entropy_values.shape != variance_values.shape or entropy_values.shape != error_values.shape:
        raise ValueError("Entropy, variance, and error maps must match")
    height, width = entropy_values.shape
    if height % region_size or width % region_size:
        raise ValueError(
            f"Map shape {(height, width)} is not divisible by region size {region_size}"
        )
    rows = []
    region_index = 0
    for region_row, y0 in enumerate(range(0, height, region_size)):
        for region_column, x0 in enumerate(range(0, width, region_size)):
            region_entropy = entropy_values[y0 : y0 + region_size, x0 : x0 + region_size]
            region_variance = variance_values[y0 : y0 + region_size, x0 : x0 + region_size]
            region_error = error_values[y0 : y0 + region_size, x0 : x0 + region_size]
            rows.append(
                {
                    **metadata,
                    "region_index": region_index,
                    "region_row": region_row,
                    "region_column": region_column,
                    "coarse_y0": y0,
                    "coarse_x0": x0,
                    "region_size": region_size,
                    "pixels": int(region_error.size),
                    "error_pixels": int(region_error.sum()),
                    "error_fraction": float(region_error.mean()),
                    "mean_entropy_bits": float(region_entropy.mean()),
                    "mean_predictive_variance": float(region_variance.mean()),
                    "max_entropy_bits": float(region_entropy.max()),
                }
            )
            region_index += 1
    return pd.DataFrame(rows)


def local_region_correlations(regions: pd.DataFrame) -> dict[str, float]:
    entropy = regions["mean_entropy_bits"].to_numpy()
    variance = regions["mean_predictive_variance"].to_numpy()
    errors = regions["error_fraction"].to_numpy()
    return {
        "region_entropy_error_pearson": safe_pearson(entropy, errors),
        "region_entropy_error_spearman": float(spearmanr(entropy, errors).statistic),
        "region_variance_error_pearson": safe_pearson(variance, errors),
        "region_variance_error_spearman": float(spearmanr(variance, errors).statistic),
    }


def top_region_concentration(
    regions: pd.DataFrame,
    uncertainty_column: str,
    fractions: Iterable[float],
) -> dict[str, float]:
    """Measure error concentration in highest-mean-uncertainty regions."""

    values = regions[uncertainty_column].to_numpy(dtype=np.float64)
    error_pixels = regions["error_pixels"].to_numpy(dtype=np.int64)
    pixels = regions["pixels"].to_numpy(dtype=np.int64)
    total_errors = int(error_pixels.sum())
    overall_error_rate = total_errors / int(pixels.sum())
    if total_errors == 0:
        raise ValueError("At least one region error is required")
    result: dict[str, float] = {}
    for fraction in fractions:
        selected_count = max(1, int(np.ceil(len(regions) * fraction)))
        order = np.argsort(values, kind="stable")[::-1][:selected_count]
        selected_errors = int(error_pixels[order].sum())
        selected_pixels = int(pixels[order].sum())
        selected_error_rate = selected_errors / selected_pixels
        label = f"top_{int(round(fraction * 100)):02d}pct_regions"
        result[f"{label}_selected_fraction"] = selected_count / len(regions)
        result[f"{label}_error_capture"] = selected_errors / total_errors
        result[f"{label}_error_enrichment"] = selected_error_rate / overall_error_rate
        result[f"{label}_selected_error_rate"] = selected_error_rate
    return result
