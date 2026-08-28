#!/usr/bin/env python3
"""Evaluate frozen uncertainty-guided native-tile fusion over five held-out folds."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multiscale-imaging-matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from adaptive_multiscale.data.ma_islands import load_grayscale_image
from adaptive_multiscale.fusion import (
    feather_tile_weight,
    fuse_selected_tiles,
    oracle_gain_rankings,
)
from adaptive_multiscale.models import CompactUNet
from adaptive_multiscale.selection.tile_ranking import (
    paired_bootstrap_interval,
    random_rank_matrix,
)
from adaptive_multiscale.training.reproducibility import seed_everything, select_device


METRICS = ("dice", "iou", "precision", "recall")
COUNT_NAMES = ("true_positive", "false_positive", "false_negative", "true_negative")
POLICY_LABELS = {
    "coarse_only": "Coarse only",
    "full_fine": "Full fine",
    "entropy_feathered": "Entropy + feathered",
    "variance_feathered": "Variance + feathered",
    "entropy_hard": "Entropy + hard replace",
    "oracle_gain_feathered": "Oracle gain + feathered",
    "random_feathered": "Random + feathered",
}
POLICY_COLORS = {
    "coarse_only": "#94A3B8",
    "full_fine": "#16324F",
    "entropy_feathered": "#7C3AED",
    "variance_feathered": "#0891B2",
    "entropy_hard": "#F59E0B",
    "oracle_gain_feathered": "#16A34A",
    "random_feathered": "#64748B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/adaptive_fusion.json")
    )
    parser.add_argument("--folds", type=int, nargs="*", help="Optional fold subset")
    parser.add_argument(
        "--limit-images", type=int, help="Optional per-fold image limit for smoke tests"
    )
    parser.add_argument("--output-dir", type=Path, help="Optional output override")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def build_model(checkpoint: dict[str, Any], device: torch.device) -> CompactUNet:
    model_config = checkpoint["model_config"]
    model = CompactUNet(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        encoder_channels=tuple(model_config["encoder_channels"]),
        bottleneck_dropout=float(model_config["bottleneck_dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def infer_coarse_probability(
    model: torch.nn.Module,
    image_path: Path,
    normalization_mean: float,
    normalization_std: float,
    device: torch.device,
) -> np.ndarray:
    image = load_grayscale(image_path).astype(np.float32) / 255.0
    values = (image - normalization_mean) / normalization_std
    tensor = torch.from_numpy(values[None, None]).to(device)
    return torch.sigmoid(model(tensor))[0, 0].cpu().numpy().astype(np.float32)


@torch.inference_mode()
def infer_fine_probability(
    model: torch.nn.Module,
    native_image: np.ndarray,
    normalization_mean: float,
    normalization_std: float,
    tile_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    height, width = native_image.shape
    if height % tile_size or width % tile_size:
        raise ValueError("Native image is not divisible by the fine tile size")
    rows, columns = height // tile_size, width // tile_size
    tiles: list[np.ndarray] = []
    coordinates: list[tuple[int, int]] = []
    for tile_row in range(rows):
        for tile_column in range(columns):
            y0, x0 = tile_row * tile_size, tile_column * tile_size
            tile = native_image[y0 : y0 + tile_size, x0 : x0 + tile_size]
            values = tile.astype(np.float32) / 255.0
            values = (values - normalization_mean) / normalization_std
            tiles.append(values[None])
            coordinates.append((y0, x0))

    result = np.zeros((height, width), dtype=np.float32)
    for start in range(0, len(tiles), batch_size):
        batch = torch.from_numpy(np.stack(tiles[start : start + batch_size])).to(device)
        probabilities = torch.sigmoid(model(batch))[:, 0].cpu().numpy()
        for offset, probability in enumerate(probabilities):
            y0, x0 = coordinates[start + offset]
            result[y0 : y0 + tile_size, x0 : x0 + tile_size] = probability
    return result


def confusion_vector(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    return np.asarray(
        [
            np.count_nonzero(predicted & truth),
            np.count_nonzero(predicted & ~truth),
            np.count_nonzero(~predicted & truth),
            np.count_nonzero(~predicted & ~truth),
        ],
        dtype=np.float64,
    )


def fields_from_counts(counts: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = np.asarray(counts, dtype=np.float64)
    epsilon = 1e-12
    total = tp + fp + fn + tn
    predicted_fraction = (tp + fp) / total
    target_fraction = (tp + fn) / total
    return {
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_negative": float(tn),
        "dice": float((2.0 * tp) / (2.0 * tp + fp + fn + epsilon)),
        "iou": float(tp / (tp + fp + fn + epsilon)),
        "precision": float(tp / (tp + fp + epsilon)),
        "recall": float(tp / (tp + fn + epsilon)),
        "predicted_foreground_fraction": float(predicted_fraction),
        "target_foreground_fraction": float(target_fraction),
        "area_fraction_error_signed": float(predicted_fraction - target_fraction),
        "area_fraction_error_absolute": float(abs(predicted_fraction - target_fraction)),
    }


def metric_row(
    record: Any,
    policy: str,
    fusion_rule: str,
    ranking_source: str,
    uses_ground_truth_for_ranking: bool,
    budget: int,
    tile_count: int,
    tile_size: int,
    counts: np.ndarray,
    random_trials: int = 0,
) -> dict[str, object]:
    return {
        "outer_fold": int(record.outer_fold),
        "sample_id": str(record.sample_id),
        "file_name": str(record.file_name),
        "source_regime": str(record.source_regime),
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "fusion_rule": fusion_rule,
        "ranking_source": ranking_source,
        "uses_ground_truth_for_ranking": uses_ground_truth_for_ranking,
        "budget_k": int(budget),
        "coverage_fraction": budget / tile_count,
        "fine_tiles_inspected": int(budget),
        "fine_pixels_inspected": int(budget * tile_size * tile_size),
        "random_trials": int(random_trials),
        **fields_from_counts(counts),
    }


def counts_for_selection(
    base_counts: np.ndarray,
    coarse_tile_counts: np.ndarray,
    candidate_tile_counts: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    if selected.size == 0:
        return base_counts.copy()
    return base_counts + (
        candidate_tile_counts[selected] - coarse_tile_counts[selected]
    ).sum(axis=0)


def tile_effects(
    record: Any,
    rankings: pd.DataFrame,
    oracle: pd.DataFrame,
    coarse_probability: np.ndarray,
    fine_probability: np.ndarray,
    target: np.ndarray,
    tile_size: int,
    blend_width: int,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    height, width = target.shape
    grid_rows, grid_columns = height // tile_size, width // tile_size
    ranking_index = rankings.set_index("tile_index")
    oracle_index = oracle.set_index("tile_index")
    rows: list[dict[str, object]] = []
    coarse_counts: list[np.ndarray] = []
    feather_counts: list[np.ndarray] = []
    hard_counts: list[np.ndarray] = []
    for tile_index in range(grid_rows * grid_columns):
        tile_row, tile_column = divmod(tile_index, grid_columns)
        y0, x0 = tile_row * tile_size, tile_column * tile_size
        y1, x1 = y0 + tile_size, x0 + tile_size
        coarse_tile = coarse_probability[y0:y1, x0:x1]
        fine_tile = fine_probability[y0:y1, x0:x1]
        target_tile = target[y0:y1, x0:x1]
        weight = feather_tile_weight(
            tile_size,
            blend_width,
            tile_row,
            tile_column,
            grid_rows,
            grid_columns,
        )
        feather_probability = weight * fine_tile + (1.0 - weight) * coarse_tile
        coarse_vector = confusion_vector(coarse_tile >= threshold, target_tile)
        feather_vector = confusion_vector(feather_probability >= threshold, target_tile)
        hard_vector = confusion_vector(fine_tile >= threshold, target_tile)
        coarse_counts.append(coarse_vector)
        feather_counts.append(feather_vector)
        hard_counts.append(hard_vector)
        rank = ranking_index.loc[tile_index]
        oracle_row = oracle_index.loc[tile_index]
        rows.append(
            {
                "outer_fold": int(record.outer_fold),
                "sample_id": record.sample_id,
                "file_name": record.file_name,
                "source_regime": record.source_regime,
                "tile_index": tile_index,
                "tile_row": tile_row,
                "tile_column": tile_column,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "entropy_p90_bits": float(rank["entropy_p90_bits"]),
                "variance_p90": float(rank["variance_p90"]),
                "entropy_rank": int(rank["entropy_rank"]),
                "variance_rank": int(rank["variance_rank"]),
                "oracle_gain_pixels": int(oracle_row["oracle_gain_pixels"]),
                "oracle_gain_rank": int(oracle_row["oracle_gain_rank"]),
                "coarse_error_pixels": int(coarse_vector[1] + coarse_vector[2]),
                "feathered_error_pixels": int(feather_vector[1] + feather_vector[2]),
                "hard_fine_error_pixels": int(hard_vector[1] + hard_vector[2]),
                "feathered_error_reduction_pixels": int(
                    coarse_vector[1] + coarse_vector[2] - feather_vector[1] - feather_vector[2]
                ),
                "hard_error_reduction_pixels": int(
                    coarse_vector[1] + coarse_vector[2] - hard_vector[1] - hard_vector[2]
                ),
            }
        )
    return (
        pd.DataFrame(rows),
        np.stack(coarse_counts),
        np.stack(feather_counts),
        np.stack(hard_counts),
    )


def random_mean_rows(
    trials: pd.DataFrame,
    record: Any,
    tile_count: int,
    tile_size: int,
    random_trials: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    numeric = list(COUNT_NAMES) + list(METRICS) + [
        "predicted_foreground_fraction",
        "target_foreground_fraction",
        "area_fraction_error_signed",
        "area_fraction_error_absolute",
    ]
    for budget, group in trials.groupby("budget_k", sort=True):
        row = {
            "outer_fold": int(record.outer_fold),
            "sample_id": record.sample_id,
            "file_name": record.file_name,
            "source_regime": record.source_regime,
            "policy": "random_feathered",
            "policy_label": POLICY_LABELS["random_feathered"],
            "fusion_rule": "feathered",
            "ranking_source": "reproducible_random",
            "uses_ground_truth_for_ranking": False,
            "budget_k": int(budget),
            "coverage_fraction": int(budget) / tile_count,
            "fine_tiles_inspected": int(budget),
            "fine_pixels_inspected": int(budget * tile_size * tile_size),
            "random_trials": random_trials,
        }
        row.update({name: float(group[name].mean()) for name in numeric})
        rows.append(row)
    return rows


def bootstrap_mean(
    values: np.ndarray,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float(array.mean()), float("nan"), float("nan")
    return paired_bootstrap_interval(array, confidence_level, resamples, seed)


def summarize_performance(
    per_image: pd.DataFrame,
    confidence_level: float,
    resamples: int,
    seed_base: int,
) -> pd.DataFrame:
    scoped = [("all_images", "all", per_image)]
    scoped.extend(
        (f"regime_{regime}", regime, group)
        for regime, group in per_image.groupby("source_regime", sort=True)
    )
    rows: list[dict[str, object]] = []
    for scope_index, (scope, regime, frame) in enumerate(scoped):
        coarse = frame[frame["policy"] == "coarse_only"].set_index("sample_id")
        full_fine = frame[frame["policy"] == "full_fine"].set_index("sample_id")
        full_gain = float((full_fine["dice"] - coarse["dice"]).mean())
        for group_index, ((policy, budget), group) in enumerate(
            frame.groupby(["policy", "budget_k"], sort=True)
        ):
            group = group.sort_values("sample_id")
            sample_ids = group["sample_id"].tolist()
            dice_gain = group["dice"].to_numpy() - coarse.loc[sample_ids, "dice"].to_numpy()
            fine_gap = group["dice"].to_numpy() - full_fine.loc[sample_ids, "dice"].to_numpy()
            dice_mean, dice_low, dice_high = bootstrap_mean(
                group["dice"].to_numpy(),
                confidence_level,
                resamples,
                seed_base + 1000 * scope_index + group_index,
            )
            area_mean, area_low, area_high = bootstrap_mean(
                group["area_fraction_error_absolute"].to_numpy(),
                confidence_level,
                resamples,
                seed_base + 5000 + 1000 * scope_index + group_index,
            )
            _, gain_low, gain_high = bootstrap_mean(
                dice_gain,
                confidence_level,
                resamples,
                seed_base + 9000 + 1000 * scope_index + group_index,
            )
            total_counts = group[list(COUNT_NAMES)].sum().to_numpy(dtype=np.float64)
            micro = fields_from_counts(total_counts)
            rows.append(
                {
                    "scope": scope,
                    "source_regime": regime,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "budget_k": int(budget),
                    "images": int(group["sample_id"].nunique()),
                    "coverage_fraction": float(group["coverage_fraction"].mean()),
                    "fine_pixels_inspected": float(group["fine_pixels_inspected"].mean()),
                    "macro_dice": dice_mean,
                    "macro_dice_ci_low": dice_low,
                    "macro_dice_ci_high": dice_high,
                    "macro_iou": float(group["iou"].mean()),
                    "macro_precision": float(group["precision"].mean()),
                    "macro_recall": float(group["recall"].mean()),
                    "micro_dice": micro["dice"],
                    "micro_iou": micro["iou"],
                    "micro_precision": micro["precision"],
                    "micro_recall": micro["recall"],
                    "area_fraction_error_absolute_mean": area_mean,
                    "area_fraction_error_absolute_ci_low": area_low,
                    "area_fraction_error_absolute_ci_high": area_high,
                    "dice_gain_over_coarse_mean": float(dice_gain.mean()),
                    "dice_gain_over_coarse_ci_low": gain_low,
                    "dice_gain_over_coarse_ci_high": gain_high,
                    "dice_difference_from_full_fine_mean": float(fine_gap.mean()),
                    "full_fine_dice_gain_recovery_fraction": (
                        float(dice_gain.mean() / full_gain)
                        if abs(full_gain) > 1e-12
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "policy", "budget_k"])


def paired_comparisons(
    per_image: pd.DataFrame,
    budgets: list[int],
    confidence_level: float,
    resamples: int,
    seed_base: int,
) -> pd.DataFrame:
    comparisons = [
        ("entropy_feathered", "random_feathered", "entropy_vs_random"),
        ("variance_feathered", "random_feathered", "variance_vs_random"),
        ("entropy_feathered", "coarse_only", "entropy_vs_coarse"),
        ("entropy_feathered", "full_fine", "entropy_vs_full_fine"),
        ("entropy_feathered", "entropy_hard", "feathered_vs_hard"),
    ]
    metric_columns = list(METRICS) + ["area_fraction_error_absolute"]
    scoped = [("all_images", "all", per_image)]
    scoped.extend(
        (f"regime_{regime}", regime, group)
        for regime, group in per_image.groupby("source_regime", sort=True)
    )
    rows: list[dict[str, object]] = []
    row_index = 0
    for scope, regime, frame in scoped:
        for left_policy, comparator_policy, comparison_name in comparisons:
            for budget in budgets:
                left = frame[
                    (frame["policy"] == left_policy) & (frame["budget_k"] == budget)
                ]
                if comparator_policy in {"coarse_only", "full_fine"}:
                    comparator = frame[frame["policy"] == comparator_policy]
                else:
                    comparator = frame[
                        (frame["policy"] == comparator_policy)
                        & (frame["budget_k"] == budget)
                    ]
                merged = left.merge(
                    comparator,
                    on="sample_id",
                    suffixes=("_left", "_comparator"),
                    validate="one_to_one",
                )
                if merged.empty:
                    continue
                for metric in metric_columns:
                    if metric == "area_fraction_error_absolute":
                        differences = (
                            merged[f"{metric}_comparator"] - merged[f"{metric}_left"]
                        ).to_numpy()
                        direction = "comparator_minus_left_lower_is_better"
                    else:
                        differences = (
                            merged[f"{metric}_left"] - merged[f"{metric}_comparator"]
                        ).to_numpy()
                        direction = "left_minus_comparator_higher_is_better"
                    mean, low, high = bootstrap_mean(
                        differences,
                        confidence_level,
                        resamples,
                        seed_base + row_index,
                    )
                    rows.append(
                        {
                            "scope": scope,
                            "source_regime": regime,
                            "comparison": comparison_name,
                            "left_policy": left_policy,
                            "comparator_policy": comparator_policy,
                            "budget_k": budget,
                            "coverage_fraction": budget / 48,
                            "metric": metric,
                            "images": int(len(merged)),
                            "left_mean": float(merged[f"{metric}_left"].mean()),
                            "comparator_mean": float(
                                merged[f"{metric}_comparator"].mean()
                            ),
                            "improvement_direction": direction,
                            "improvement_mean": mean,
                            "improvement_ci_low": low,
                            "improvement_ci_high": high,
                            "improved_image_fraction": float(
                                np.mean(differences > 0.0)
                            ),
                            "confidence_level": confidence_level,
                        }
                    )
                    row_index += 1
    return pd.DataFrame(rows)


def plot_performance_tradeoff(summary: pd.DataFrame, output_path: Path) -> None:
    overall = summary[summary["scope"] == "all_images"]
    coarse_dice = float(overall.loc[overall["policy"] == "coarse_only", "macro_dice"].iloc[0])
    fine_dice = float(overall.loc[overall["policy"] == "full_fine", "macro_dice"].iloc[0])
    coarse_area = float(
        overall.loc[
            overall["policy"] == "coarse_only", "area_fraction_error_absolute_mean"
        ].iloc[0]
    )
    fine_area = float(
        overall.loc[
            overall["policy"] == "full_fine", "area_fraction_error_absolute_mean"
        ].iloc[0]
    )
    policies = [
        "entropy_feathered",
        "variance_feathered",
        "random_feathered",
        "oracle_gain_feathered",
        "entropy_hard",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for policy in policies:
        frame = overall[overall["policy"] == policy].sort_values("coverage_fraction")
        linestyle = "--" if policy == "entropy_hard" else "-"
        axes[0].plot(
            100 * frame["coverage_fraction"],
            frame["macro_dice"],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
        axes[1].plot(
            100 * frame["coverage_fraction"],
            frame["area_fraction_error_absolute_mean"],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            color=POLICY_COLORS[policy],
        )
        axes[2].plot(
            100 * frame["coverage_fraction"],
            100 * frame["full_fine_dice_gain_recovery_fraction"],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            color=POLICY_COLORS[policy],
        )
    axes[0].axhline(coarse_dice, color=POLICY_COLORS["coarse_only"], linestyle=":")
    axes[0].axhline(fine_dice, color=POLICY_COLORS["full_fine"], linestyle=":")
    axes[1].axhline(coarse_area, color=POLICY_COLORS["coarse_only"], linestyle=":")
    axes[1].axhline(fine_area, color=POLICY_COLORS["full_fine"], linestyle=":")
    axes[2].axhline(0.0, color=POLICY_COLORS["coarse_only"], linestyle=":")
    axes[2].axhline(100.0, color=POLICY_COLORS["full_fine"], linestyle=":")
    titles = (
        "Segmentation performance",
        "M-A area-fraction error",
        "Recovered coarse-to-fine Dice gain",
    )
    ylabels = ("Macro Dice", "Mean absolute area-fraction error", "Gain recovered (%)")
    for axis, title, ylabel in zip(axes, titles, ylabels, strict=True):
        axis.set_title(title)
        axis.set_xlabel("Native high-resolution coverage (%)")
        axis.set_ylabel(ylabel)
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Adaptive fusion performance versus inspection cost", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_regime_tradeoff(summary: pd.DataFrame, output_path: Path) -> None:
    regimes = sorted(summary.loc[summary["scope"] != "all_images", "source_regime"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for axis in axes.flat:
        axis.set_visible(False)
    for axis, regime in zip(axes.flat, regimes, strict=False):
        axis.set_visible(True)
        frame = summary[summary["source_regime"] == regime]
        for policy in ("entropy_feathered", "random_feathered", "oracle_gain_feathered"):
            policy_frame = frame[frame["policy"] == policy].sort_values("coverage_fraction")
            axis.plot(
                100 * policy_frame["coverage_fraction"],
                policy_frame["macro_dice"],
                marker="o",
                linewidth=2,
                color=POLICY_COLORS[policy],
                label=POLICY_LABELS[policy],
            )
        coarse = float(frame.loc[frame["policy"] == "coarse_only", "macro_dice"].iloc[0])
        full_fine = float(frame.loc[frame["policy"] == "full_fine", "macro_dice"].iloc[0])
        axis.axhline(coarse, color=POLICY_COLORS["coarse_only"], linestyle=":", label="Coarse")
        axis.axhline(full_fine, color=POLICY_COLORS["full_fine"], linestyle=":", label="Full fine")
        axis.set_title(f"Source Regime {regime}")
        axis.set_xlabel("Native high-resolution coverage (%)")
        axis.set_ylabel("Macro Dice")
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Adaptive fusion performance by source regime", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def select_representatives(per_image: pd.DataFrame, example_budget: int) -> pd.DataFrame:
    adaptive = per_image[
        (per_image["policy"] == "entropy_feathered")
        & (per_image["budget_k"] == example_budget)
    ].copy()
    coarse = per_image[per_image["policy"] == "coarse_only"][
        ["sample_id", "dice"]
    ].rename(columns={"dice": "coarse_dice"})
    adaptive = adaptive.merge(coarse, on="sample_id", validate="one_to_one")
    adaptive["adaptive_minus_coarse_dice"] = adaptive["dice"] - adaptive["coarse_dice"]
    selected: list[pd.Series] = []
    largest = adaptive.sort_values(
        ["adaptive_minus_coarse_dice", "sample_id"], ascending=[False, True]
    ).iloc[0].copy()
    largest["example_role"] = "largest adaptive gain"
    selected.append(largest)
    smallest = adaptive.sort_values(
        ["adaptive_minus_coarse_dice", "sample_id"], ascending=[True, True]
    ).iloc[0].copy()
    smallest["example_role"] = "smallest adaptive gain"
    selected.append(smallest)
    for regime, group in adaptive.groupby("source_regime", sort=True):
        median = group["adaptive_minus_coarse_dice"].median()
        group = group.assign(
            distance=(group["adaptive_minus_coarse_dice"] - median).abs()
        )
        row = group.sort_values(["distance", "sample_id"]).iloc[0].copy()
        row["example_role"] = f"Regime {regime} median gain"
        selected.append(row)
    return pd.DataFrame(selected).drop_duplicates("sample_id").reset_index(drop=True)


def error_code_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    codes = np.zeros(target.shape, dtype=np.uint8)
    codes[prediction & ~target] = 1
    codes[~prediction & target] = 2
    return codes


def plot_representatives(
    representatives: pd.DataFrame,
    manifest: pd.DataFrame,
    rankings: pd.DataFrame,
    project_root: Path,
    config: dict[str, Any],
    output_dir: Path,
    output_path: Path,
) -> None:
    rows = len(representatives)
    figure, axes = plt.subplots(
        rows, 7, figsize=(20, 3.0 * rows), constrained_layout=True, squeeze=False
    )
    indexed_manifest = manifest.set_index("sample_id")
    error_cmap = ListedColormap(["#111827", "#EF4444", "#22D3EE"])
    tile_size = int(config["dataset"]["tile_size"])
    example_budget = int(config["visualization"]["example_budget"])
    for row_index, row in representatives.iterrows():
        record = indexed_manifest.loc[row["sample_id"]]
        image, _, _ = load_grayscale_image(project_root / record["source_image_path"])
        target = load_grayscale(project_root / record["source_mask_path"]) > 0
        fold = int(row["outer_fold"])
        coarse_small = load_grayscale(
            project_root
            / config["models"]["coarse_prediction_pattern"].format(
                fold=fold, sample_id=row["sample_id"]
            )
        )
        coarse = cv2.resize(
            coarse_small,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        fine = load_grayscale(
            project_root
            / config["models"]["fine_prediction_pattern"].format(
                fold=fold, sample_id=row["sample_id"]
            )
        ) > 0
        adaptive = load_grayscale(
            output_dir
            / "predicted_masks"
            / "entropy_feathered"
            / f"k{example_budget}"
            / f"fold_{fold}"
            / f"{row['sample_id']}.png"
        ) > 0
        panels = (
            (image, "gray", 0, 255),
            (target, "gray", 0, 1),
            (coarse, "gray", 0, 1),
            (adaptive, "gray", 0, 1),
            (fine, "gray", 0, 1),
            (error_code_map(adaptive, target), error_cmap, 0, 2),
            (image, "gray", 0, 255),
        )
        for column, (values, cmap, minimum, maximum) in enumerate(panels):
            axes[row_index, column].imshow(values, cmap=cmap, vmin=minimum, vmax=maximum)
            axes[row_index, column].axis("off")
        selected = rankings[
            (rankings["sample_id"] == row["sample_id"])
            & (rankings["entropy_rank"] <= example_budget)
        ]
        for selected_tile in selected.itertuples(index=False):
            axes[row_index, 6].add_patch(
                Rectangle(
                    (selected_tile.native_x0, selected_tile.native_y0),
                    tile_size,
                    tile_size,
                    fill=False,
                    edgecolor="#FDE047",
                    linewidth=2,
                )
            )
        axes[row_index, 0].text(
            -0.03,
            0.5,
            (
                f"{row['sample_id']} | Regime {row['source_regime']}\n"
                f"{row['example_role']}\n"
                f"ΔDice {row['adaptive_minus_coarse_dice']:+.3f}"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
            clip_on=False,
        )
        for column, value in (
            (2, row["coarse_dice"]),
            (3, row["dice"]),
        ):
            axes[row_index, column].text(
                0.02,
                0.98,
                f"Dice {value:.3f}",
                transform=axes[row_index, column].transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox={
                    "facecolor": "#111827",
                    "alpha": 0.75,
                    "pad": 2,
                    "edgecolor": "none",
                },
            )
    for column, title in enumerate(
        (
            "Native SEM",
            "Ground truth",
            "Coarse",
            f"Adaptive K={example_budget}",
            "Full fine",
            "Adaptive error",
            "Selected tiles",
        )
    ):
        axes[0, column].set_title(title, fontsize=10)
    figure.legend(
        handles=[
            Patch(facecolor="#111827", label="Correct"),
            Patch(facecolor="#EF4444", label="False positive"),
            Patch(facecolor="#22D3EE", label="False negative"),
            Patch(facecolor="#FDE047", label="Selected tile outline"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"Held-out uncertainty-guided adaptive fusion at K={example_budget}", fontsize=15
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = copy.deepcopy(load_json(config_path))
    output_dir = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else project_root / (args.output_dir or config["output_dir"])
    )
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    seed_everything(int(config["reproducibility"]["random_seed"]))
    device = select_device(config["models"]["device"])
    if device.type == "cpu":
        torch.set_num_threads(8)
    folds = args.folds if args.folds is not None else [0, 1, 2, 3, 4]
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("Folds must be selected from 0 through 4")
    print(f"device={device} folds={folds}", flush=True)

    dataset_config = config["dataset"]
    selection_config = config["selection"]
    fusion_config = config["fusion"]
    statistics_config = config["statistics"]
    tile_size = int(dataset_config["tile_size"])
    tile_count = int(dataset_config["tiles_per_image"])
    native_height = int(dataset_config["native_height"])
    native_width = int(dataset_config["native_width"])
    blend_width = int(fusion_config["blend_width_pixels"])
    threshold = float(config["models"]["prediction_threshold"])
    budgets = [int(value) for value in selection_config["budgets"]]
    if budgets != sorted(set(budgets)) or budgets[-1] != tile_count:
        raise ValueError("Budgets must be unique, sorted, and end at full coverage")
    random_trials_count = int(selection_config["random_trials"])
    if random_trials_count < 20:
        raise ValueError("At least 20 random trials are required")

    manifest = pd.read_csv(project_root / dataset_config["coarse_manifest"])
    cv_manifest = pd.read_csv(project_root / dataset_config["cv_manifest"])
    rankings = pd.read_csv(project_root / selection_config["rankings"])
    heldout = cv_manifest[
        (cv_manifest["split"] == "test") & (cv_manifest["outer_fold"].isin(folds))
    ].merge(
        manifest,
        on=["sample_id", "file_name", "source_regime"],
        validate="one_to_one",
    )
    heldout = heldout.sort_values(["outer_fold", "sample_id"]).reset_index(drop=True)
    if args.limit_images is not None:
        if args.limit_images <= 0:
            raise ValueError("--limit-images must be positive")
        heldout = heldout.groupby("outer_fold", group_keys=False).head(args.limit_images)
    expected_rank_rows = len(heldout) * tile_count
    rankings = rankings[rankings["sample_id"].isin(heldout["sample_id"])].copy()
    if len(rankings) != expected_rank_rows:
        raise ValueError(
            f"Expected {expected_rank_rows} ranking rows, found {len(rankings)}"
        )

    per_image_rows: list[dict[str, object]] = []
    random_trial_frames: list[pd.DataFrame] = []
    tile_frames: list[pd.DataFrame] = []
    verification_rows: list[dict[str, object]] = []
    example_budget = int(config["visualization"]["example_budget"])

    for fold in folds:
        fold_records = heldout[heldout["outer_fold"] == fold]
        if fold_records.empty:
            continue
        coarse_checkpoint = torch.load(
            project_root
            / config["models"]["coarse_checkpoint_pattern"].format(fold=fold),
            map_location=device,
            weights_only=False,
        )
        fine_checkpoint = torch.load(
            project_root
            / config["models"]["fine_checkpoint_pattern"].format(fold=fold),
            map_location=device,
            weights_only=False,
        )
        coarse_model = build_model(coarse_checkpoint, device)
        fine_model = build_model(fine_checkpoint, device)
        for record in fold_records.itertuples(index=False):
            sample_rankings = rankings[
                rankings["sample_id"] == record.sample_id
            ].sort_values("tile_index")

            # No label is loaded before the label-free rankings and both model
            # probability maps have been generated.
            coarse_small = infer_coarse_probability(
                coarse_model,
                project_root / record.coarse_image_path,
                float(coarse_checkpoint["normalization_mean"]),
                float(coarse_checkpoint["normalization_std"]),
                device,
            )
            native_image, _, _ = load_grayscale_image(
                project_root / record.source_image_path
            )
            fine_probability = infer_fine_probability(
                fine_model,
                native_image,
                float(fine_checkpoint["normalization_mean"]),
                float(fine_checkpoint["normalization_std"]),
                tile_size,
                int(config["models"]["fine_inference_batch_size"]),
                device,
            )
            coarse_probability = cv2.resize(
                coarse_small,
                (native_width, native_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)

            saved_coarse = load_grayscale(
                project_root
                / config["models"]["coarse_prediction_pattern"].format(
                    fold=fold, sample_id=record.sample_id
                )
            ) > 0
            saved_fine = load_grayscale(
                project_root
                / config["models"]["fine_prediction_pattern"].format(
                    fold=fold, sample_id=record.sample_id
                )
            ) > 0
            coarse_mismatch = int(
                np.count_nonzero((coarse_small >= threshold) != saved_coarse)
            )
            fine_mismatch = int(
                np.count_nonzero((fine_probability >= threshold) != saved_fine)
            )
            if coarse_mismatch or fine_mismatch:
                raise ValueError(
                    f"Frozen inference mismatch for {record.sample_id}: "
                    f"coarse={coarse_mismatch}, fine={fine_mismatch}"
                )
            verification_rows.append(
                {
                    "outer_fold": fold,
                    "sample_id": record.sample_id,
                    "coarse_mismatch_pixels": coarse_mismatch,
                    "fine_mismatch_pixels": fine_mismatch,
                    "status": "passed",
                }
            )

            # Labels enter only now, for evaluation and the explicit oracle.
            target = load_grayscale(project_root / record.source_mask_path) > 0
            if target.shape != (native_height, native_width):
                raise ValueError(f"Unexpected target shape for {record.sample_id}")
            oracle = oracle_gain_rankings(
                coarse_probability,
                fine_probability,
                target,
                tile_size,
                blend_width,
                threshold,
            )
            effects, coarse_tiles, feather_tiles, hard_tiles = tile_effects(
                record,
                sample_rankings,
                oracle,
                coarse_probability,
                fine_probability,
                target,
                tile_size,
                blend_width,
                threshold,
            )
            tile_frames.append(effects)
            coarse_total = coarse_tiles.sum(axis=0)
            fine_total = hard_tiles.sum(axis=0)
            per_image_rows.append(
                metric_row(
                    record,
                    "coarse_only",
                    "none",
                    "none",
                    False,
                    0,
                    tile_count,
                    tile_size,
                    coarse_total,
                )
            )
            per_image_rows.append(
                metric_row(
                    record,
                    "full_fine",
                    "full_native_fine",
                    "all_tiles",
                    False,
                    tile_count,
                    tile_count,
                    tile_size,
                    fine_total,
                )
            )

            policy_specs = (
                ("entropy_feathered", "entropy_rank", feather_tiles, "feathered", False),
                ("variance_feathered", "variance_rank", feather_tiles, "feathered", False),
                ("entropy_hard", "entropy_rank", hard_tiles, "hard", False),
                (
                    "oracle_gain_feathered",
                    "oracle_gain_rank",
                    feather_tiles,
                    "feathered",
                    True,
                ),
            )
            for policy, rank_column, candidate_counts, rule, uses_truth in policy_specs:
                for budget in budgets:
                    selected = effects.loc[
                        effects[rank_column] <= budget, "tile_index"
                    ].to_numpy(dtype=np.int64)
                    if len(selected) != budget:
                        raise ValueError(
                            f"{policy} did not select exactly {budget} tiles"
                        )
                    counts = counts_for_selection(
                        coarse_total, coarse_tiles, candidate_counts, selected
                    )
                    per_image_rows.append(
                        metric_row(
                            record,
                            policy,
                            rule,
                            rank_column,
                            uses_truth,
                            budget,
                            tile_count,
                            tile_size,
                            counts,
                        )
                    )
                    if policy == "entropy_feathered" and budget == example_budget:
                        probability = fuse_selected_tiles(
                            coarse_probability,
                            fine_probability,
                            selected,
                            tile_size,
                            blend_width,
                            rule="feathered",
                        )
                        mask_dir = (
                            output_dir
                            / "predicted_masks"
                            / "entropy_feathered"
                            / f"k{budget}"
                            / f"fold_{fold}"
                        )
                        mask_dir.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(
                            str(mask_dir / f"{record.sample_id}.png"),
                            (probability >= threshold).astype(np.uint8) * 255,
                        ):
                            raise OSError(f"Failed to save adaptive mask for {record.sample_id}")

            sample_number = int(str(record.sample_id).split("_")[-1])
            random_ranks = random_rank_matrix(
                tile_count,
                random_trials_count,
                int(selection_config["random_seed_base"]),
                sample_number,
            )
            trial_rows: list[dict[str, object]] = []
            for trial_index in range(random_trials_count):
                for budget in budgets:
                    selected = np.flatnonzero(random_ranks[trial_index] <= budget)
                    counts = counts_for_selection(
                        coarse_total, coarse_tiles, feather_tiles, selected
                    )
                    trial_rows.append(
                        {
                            "outer_fold": fold,
                            "sample_id": record.sample_id,
                            "file_name": record.file_name,
                            "source_regime": record.source_regime,
                            "random_trial_index": trial_index,
                            "random_seed_base": int(selection_config["random_seed_base"]),
                            "budget_k": budget,
                            "coverage_fraction": budget / tile_count,
                            **fields_from_counts(counts),
                        }
                    )
            trial_frame = pd.DataFrame(trial_rows)
            random_trial_frames.append(trial_frame)
            per_image_rows.extend(
                random_mean_rows(
                    trial_frame,
                    record,
                    tile_count,
                    tile_size,
                    random_trials_count,
                )
            )
            print(
                f"fold={fold} sample={record.sample_id} probabilities=verified "
                f"policies=evaluated",
                flush=True,
            )
        del coarse_model, fine_model, coarse_checkpoint, fine_checkpoint
        if device.type == "mps":
            torch.mps.empty_cache()

    per_image = pd.DataFrame(per_image_rows).sort_values(
        ["outer_fold", "sample_id", "policy", "budget_k"]
    )
    random_trials = pd.concat(random_trial_frames, ignore_index=True).sort_values(
        ["outer_fold", "sample_id", "random_trial_index", "budget_k"]
    )
    tile_frame = pd.concat(tile_frames, ignore_index=True).sort_values(
        ["outer_fold", "sample_id", "tile_index"]
    )
    verification = pd.DataFrame(verification_rows).sort_values(
        ["outer_fold", "sample_id"]
    )
    confidence_level = float(statistics_config["confidence_level"])
    resamples = int(statistics_config["bootstrap_resamples"])
    bootstrap_seed = int(statistics_config["bootstrap_seed"])
    performance = summarize_performance(
        per_image, confidence_level, resamples, bootstrap_seed
    )
    paired = paired_comparisons(
        per_image,
        budgets,
        confidence_level,
        resamples,
        bootstrap_seed + 30000,
    )
    representatives = select_representatives(per_image, example_budget)

    per_image.to_csv(metrics_dir / "per_image_policy_metrics.csv", index=False)
    random_trials.to_csv(metrics_dir / "random_trial_metrics.csv", index=False)
    tile_frame.to_csv(metrics_dir / "tile_fusion_effects.csv", index=False)
    verification.to_csv(metrics_dir / "mask_verification.csv", index=False)
    performance.to_csv(metrics_dir / "performance_cost_summary.csv", index=False)
    paired.to_csv(metrics_dir / "paired_comparisons.csv", index=False)
    representatives.to_csv(metrics_dir / "representative_samples.csv", index=False)

    plot_performance_tradeoff(
        performance, figures_dir / "adaptive_performance_tradeoff.png"
    )
    plot_regime_tradeoff(
        performance, figures_dir / "adaptive_performance_by_regime.png"
    )
    plot_representatives(
        representatives,
        manifest,
        rankings,
        project_root,
        config,
        output_dir,
        figures_dir / "adaptive_fusion_examples.png",
    )

    overall = performance[performance["scope"] == "all_images"]
    primary = overall[
        (overall["policy"] == "entropy_feathered")
        & (overall["budget_k"] == example_budget)
    ].iloc[0]
    coarse = overall[overall["policy"] == "coarse_only"].iloc[0]
    full_fine = overall[overall["policy"] == "full_fine"].iloc[0]
    primary_random = overall[
        (overall["policy"] == "random_feathered")
        & (overall["budget_k"] == example_budget)
    ].iloc[0]
    primary_pair = paired[
        (paired["scope"] == "all_images")
        & (paired["comparison"] == "entropy_vs_random")
        & (paired["budget_k"] == example_budget)
        & (paired["metric"] == "dice")
    ].iloc[0]
    summary: dict[str, object] = {
        "experiment_name": config["experiment_name"],
        "completed_folds": sorted(int(value) for value in folds),
        "held_out_images": int(per_image["sample_id"].nunique()),
        "held_out_source_regimes": {
            str(key): int(value)
            for key, value in heldout.groupby("source_regime").size().to_dict().items()
        },
        "no_training_performed": True,
        "frozen_probability_verification": {
            "images": int(len(verification)),
            "coarse_mismatch_pixels": int(verification["coarse_mismatch_pixels"].sum()),
            "fine_mismatch_pixels": int(verification["fine_mismatch_pixels"].sum()),
            "status": "passed",
        },
        "selection": {
            "budgets": budgets,
            "coverage_fractions": [budget / tile_count for budget in budgets],
            "random_trials_per_image": random_trials_count,
            "primary": "within-image entropy q90 ranking from Step 4",
            "secondary": "within-image predictive-variance q90 ranking from Step 4",
            "oracle": "evaluation-only true feathered pixel-error reduction",
            "ground_truth_use": config["reproducibility"]["ground_truth_use"],
        },
        "fusion": config["fusion"],
        "cost_interpretation": config["reproducibility"]["cost_interpretation"],
        "coarse_baseline": {
            "macro_dice": float(coarse["macro_dice"]),
            "macro_iou": float(coarse["macro_iou"]),
            "area_fraction_error_absolute_mean": float(
                coarse["area_fraction_error_absolute_mean"]
            ),
        },
        "full_fine_baseline": {
            "macro_dice": float(full_fine["macro_dice"]),
            "macro_iou": float(full_fine["macro_iou"]),
            "area_fraction_error_absolute_mean": float(
                full_fine["area_fraction_error_absolute_mean"]
            ),
        },
        f"primary_entropy_k{example_budget}": {
            "coverage_fraction": float(primary["coverage_fraction"]),
            "macro_dice": float(primary["macro_dice"]),
            "macro_iou": float(primary["macro_iou"]),
            "macro_precision": float(primary["macro_precision"]),
            "macro_recall": float(primary["macro_recall"]),
            "area_fraction_error_absolute_mean": float(
                primary["area_fraction_error_absolute_mean"]
            ),
            "dice_gain_over_coarse": float(primary["dice_gain_over_coarse_mean"]),
            "full_fine_dice_gain_recovery_fraction": float(
                primary["full_fine_dice_gain_recovery_fraction"]
            ),
            "random_macro_dice": float(primary_random["macro_dice"]),
            "entropy_minus_random_dice": float(primary_pair["improvement_mean"]),
            "entropy_minus_random_dice_ci": [
                float(primary_pair["improvement_ci_low"]),
                float(primary_pair["improvement_ci_high"]),
            ],
        },
        "statistics": {
            "paired_bootstrap_resamples": resamples,
            "confidence_level": confidence_level,
        },
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "runtime_seconds": time.perf_counter() - start,
        "config_path": str(config_path.relative_to(project_root)),
        "config": config,
    }
    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "run_environment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "command": " ".join(sys.argv),
                "device": str(device),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
                "config": config,
            },
            handle,
            indent=2,
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
