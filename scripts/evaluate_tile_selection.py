#!/usr/bin/env python3
"""Evaluate tile-level rankings from frozen coarse-model uncertainty maps."""

from __future__ import annotations

import argparse
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
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr, wilcoxon

from adaptive_multiscale.selection.tile_ranking import (
    TileGrid,
    attach_error_evaluation,
    paired_bootstrap_interval,
    random_rank_matrix,
    selection_metrics,
    tile_error_correlations,
    uncertainty_tile_rankings,
)
from adaptive_multiscale.uncertainty.evaluation import safe_pearson


POLICY_RANK_COLUMNS = {
    "entropy_q90": "entropy_rank",
    "variance_q90": "variance_rank",
    "oracle_error": "oracle_error_rank",
}
POLICY_LABELS = {
    "entropy_q90": "Entropy q90",
    "variance_q90": "Variance q90",
    "random": "Random (100 trials)",
    "oracle_error": "Oracle error",
}
POLICY_COLORS = {
    "entropy_q90": "#7C3AED",
    "variance_q90": "#0891B2",
    "random": "#64748B",
    "oracle_error": "#16A34A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tile_selection.json")
    )
    return parser.parse_args()


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def bootstrap_mean(
    values: np.ndarray,
    confidence_level: float,
    resamples: int,
    random_seed: int,
) -> tuple[float, float, float]:
    return paired_bootstrap_interval(
        np.asarray(values, dtype=np.float64),
        confidence_level,
        resamples,
        random_seed,
    )


def summarize_policy_budgets(
    per_image: pd.DataFrame,
    confidence_level: float,
    resamples: int,
    seed_base: int,
    group_column: str | None = None,
) -> pd.DataFrame:
    group_columns = ([group_column] if group_column else []) + ["policy", "budget_k"]
    rows: list[dict[str, object]] = []
    for group_index, (keys, group) in enumerate(
        per_image.groupby(group_columns, sort=True, observed=True)
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_columns, keys, strict=True))
        mean, low, high = bootstrap_mean(
            group["error_capture"].to_numpy(),
            confidence_level,
            resamples,
            seed_base + group_index,
        )
        rows.append(
            {
                **key_values,
                "images": int(group["sample_id"].nunique()),
                "coverage_fraction": float(group["coverage_fraction"].mean()),
                "error_capture_mean": mean,
                "error_capture_std": float(group["error_capture"].std(ddof=1)),
                "error_capture_ci_low": low,
                "error_capture_ci_high": high,
                "pooled_error_capture": float(
                    group["selected_error_pixels"].sum()
                    / group["total_error_pixels"].sum()
                ),
                "error_density_enrichment_mean": float(
                    group["error_density_enrichment"].mean()
                ),
            }
        )
    result = pd.DataFrame(rows)
    random_reference_columns = ([group_column] if group_column else []) + [
        "budget_k",
        "error_capture_mean",
    ]
    random_reference = result[result["policy"] == "random"][
        random_reference_columns
    ].rename(columns={"error_capture_mean": "random_error_capture_mean"})
    merge_columns = ([group_column] if group_column else []) + ["budget_k"]
    result = result.merge(random_reference, on=merge_columns, validate="many_to_one")
    oracle_reference_columns = ([group_column] if group_column else []) + [
        "budget_k",
        "error_capture_mean",
    ]
    oracle_reference = result[result["policy"] == "oracle_error"][
        oracle_reference_columns
    ].rename(columns={"error_capture_mean": "oracle_error_capture_mean"})
    result = result.merge(oracle_reference, on=merge_columns, validate="many_to_one")
    result["capture_enrichment_over_random"] = (
        result["error_capture_mean"] / result["random_error_capture_mean"]
    )
    result["capture_minus_random"] = (
        result["error_capture_mean"] - result["random_error_capture_mean"]
    )
    result["capture_fraction_of_oracle"] = (
        result["error_capture_mean"] / result["oracle_error_capture_mean"]
    )
    result["capture_gap_to_oracle"] = (
        result["oracle_error_capture_mean"] - result["error_capture_mean"]
    )
    return result.sort_values(group_columns).reset_index(drop=True)


def paired_policy_comparisons(
    per_image: pd.DataFrame,
    budgets: list[int],
    confidence_level: float,
    resamples: int,
    seed_base: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = [("all_images", per_image)] + [
        (f"regime_{regime}", per_image[per_image["source_regime"] == regime])
        for regime in ("I", "II", "III", "IV")
    ]
    comparison_index = 0
    for scope, scope_frame in scopes:
        for policy in ("entropy_q90", "variance_q90"):
            for budget in budgets:
                selected = scope_frame[
                    (scope_frame["policy"] == policy)
                    & (scope_frame["budget_k"] == budget)
                ][["sample_id", "error_capture"]].rename(
                    columns={"error_capture": "policy_error_capture"}
                )
                random = scope_frame[
                    (scope_frame["policy"] == "random")
                    & (scope_frame["budget_k"] == budget)
                ][["sample_id", "error_capture"]].rename(
                    columns={"error_capture": "random_error_capture"}
                )
                paired = selected.merge(random, on="sample_id", validate="one_to_one")
                differences = (
                    paired["policy_error_capture"]
                    - paired["random_error_capture"]
                ).to_numpy()
                mean, low, high = paired_bootstrap_interval(
                    differences,
                    confidence_level,
                    resamples,
                    seed_base + comparison_index,
                )
                if np.allclose(differences, 0.0):
                    p_value = 1.0
                else:
                    p_value = float(
                        wilcoxon(differences, alternative="greater", zero_method="wilcox").pvalue
                    )
                rows.append(
                    {
                        "scope": scope,
                        "policy": policy,
                        "budget_k": budget,
                        "coverage_fraction": budget / 48,
                        "images": len(paired),
                        "policy_error_capture_mean": float(
                            paired["policy_error_capture"].mean()
                        ),
                        "random_error_capture_mean": float(
                            paired["random_error_capture"].mean()
                        ),
                        "capture_difference_mean": mean,
                        "capture_difference_ci_low": low,
                        "capture_difference_ci_high": high,
                        "capture_difference_median": float(np.median(differences)),
                        "win_fraction": float((differences > 0.0).mean()),
                        "one_sided_wilcoxon_p": p_value,
                        "confidence_level": confidence_level,
                    }
                )
                comparison_index += 1
    return pd.DataFrame(rows)


def correlation_summary(
    per_image: pd.DataFrame,
    evaluated_tiles: pd.DataFrame,
    confidence_level: float,
    resamples: int,
    seed_base: int,
) -> pd.DataFrame:
    definitions = (
        ("entropy_q90", "pearson", "entropy_error_pearson", "entropy_p90_bits"),
        ("entropy_q90", "spearman", "entropy_error_spearman", "entropy_p90_bits"),
        ("variance_q90", "pearson", "variance_error_pearson", "variance_p90"),
        ("variance_q90", "spearman", "variance_error_spearman", "variance_p90"),
    )
    rows: list[dict[str, object]] = []
    scopes = [("all_images", per_image, evaluated_tiles)] + [
        (
            f"regime_{regime}",
            per_image[per_image["source_regime"] == regime],
            evaluated_tiles[evaluated_tiles["source_regime"] == regime],
        )
        for regime in ("I", "II", "III", "IV")
    ]
    summary_index = 0
    for scope, image_frame, tile_frame in scopes:
        for policy, method, image_column, score_column in definitions:
            values = image_frame[image_column].to_numpy(dtype=np.float64)
            mean, low, high = bootstrap_mean(
                values,
                confidence_level,
                resamples,
                seed_base + summary_index,
            )
            if method == "pearson":
                pooled = safe_pearson(
                    tile_frame[score_column].to_numpy(),
                    tile_frame["error_fraction"].to_numpy(),
                )
            else:
                pooled = float(
                    spearmanr(
                        tile_frame[score_column].to_numpy(),
                        tile_frame["error_fraction"].to_numpy(),
                    ).statistic
                )
            rows.append(
                {
                    "scope": scope,
                    "policy": policy,
                    "correlation": method,
                    "images": int(image_frame["sample_id"].nunique()),
                    "per_image_mean": mean,
                    "per_image_std": float(values.std(ddof=1)),
                    "per_image_ci_low": low,
                    "per_image_ci_high": high,
                    "pooled_tiles_correlation": pooled,
                    "tiles": int(len(tile_frame)),
                }
            )
            summary_index += 1
    return pd.DataFrame(rows)


def select_representatives(step3_metrics: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    regime_i = step3_metrics[step3_metrics["source_regime"] == "I"].copy()
    hardest = regime_i.sort_values(["deterministic_dice", "sample_id"]).iloc[0].copy()
    hardest["example_role"] = "Regime I hardest coarse case"
    selected.append(hardest)
    for regime in ("I", "II", "III", "IV"):
        group = step3_metrics[step3_metrics["source_regime"] == regime].copy()
        median = group["deterministic_dice"].median()
        group["distance_to_median"] = (group["deterministic_dice"] - median).abs()
        row = group.sort_values(["distance_to_median", "sample_id"]).iloc[0].copy()
        row["example_role"] = f"Regime {regime} median-Dice example"
        selected.append(row)
    return pd.DataFrame(selected).drop_duplicates("sample_id").reset_index(drop=True)


def plot_overall_performance(
    budget_summary: pd.DataFrame,
    paired: pd.DataFrame,
    correlation: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for policy in ("entropy_q90", "variance_q90", "random", "oracle_error"):
        subset = budget_summary[budget_summary["policy"] == policy].sort_values("budget_k")
        coverage = subset["coverage_fraction"].to_numpy() * 100
        mean = subset["error_capture_mean"].to_numpy() * 100
        low = subset["error_capture_ci_low"].to_numpy() * 100
        high = subset["error_capture_ci_high"].to_numpy() * 100
        axes[0].plot(
            coverage,
            mean,
            marker="o",
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
        axes[0].fill_between(coverage, low, high, color=POLICY_COLORS[policy], alpha=0.12)
    axes[0].plot([0, 100], [0, 100], "--", color="#94A3B8", linewidth=1)
    axes[0].set_xlabel("Native high-resolution coverage (%)")
    axes[0].set_ylabel("Coarse segmentation error captured (%)")
    axes[0].set_title("Selection performance across 40 held-out images")
    axes[0].legend(frameon=False, fontsize=9)

    for policy in ("entropy_q90", "variance_q90"):
        subset = paired[
            (paired["scope"] == "all_images") & (paired["policy"] == policy)
        ].sort_values("budget_k")
        coverage = subset["coverage_fraction"].to_numpy() * 100
        mean = subset["capture_difference_mean"].to_numpy() * 100
        low = subset["capture_difference_ci_low"].to_numpy() * 100
        high = subset["capture_difference_ci_high"].to_numpy() * 100
        axes[1].plot(
            coverage,
            mean,
            marker="o",
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
        axes[1].fill_between(coverage, low, high, color=POLICY_COLORS[policy], alpha=0.15)
    axes[1].axhline(0.0, color="#64748B", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Native high-resolution coverage (%)")
    axes[1].set_ylabel("Error capture minus random (percentage points)")
    axes[1].set_title("Paired image-level improvement over random")
    axes[1].legend(frameon=False)

    correlations = correlation[
        (correlation["policy"] == "entropy_q90")
        & (correlation["correlation"] == "spearman")
        & (correlation["scope"] != "all_images")
    ].copy()
    correlations["regime"] = correlations["scope"].str.replace("regime_", "", regex=False)
    order = ["I", "II", "III", "IV"]
    correlations = correlations.set_index("regime").loc[order]
    values = correlations["per_image_mean"].to_numpy()
    low = correlations["per_image_ci_low"].to_numpy()
    high = correlations["per_image_ci_high"].to_numpy()
    axes[2].bar(
        np.arange(4),
        values,
        yerr=np.vstack([values - low, high - values]),
        color=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"],
        capsize=4,
    )
    axes[2].set_xticks(np.arange(4), [f"Regime {value}" for value in order])
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("Within-image Spearman correlation")
    axes[2].set_title("Entropy q90 versus tile error")

    for axis in axes:
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    figure.suptitle("Step 4: uncertainty-guided tile ranking", fontsize=16)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_regime_performance(regime_summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, regime in zip(axes.flat, ("I", "II", "III", "IV"), strict=True):
        regime_data = regime_summary[regime_summary["source_regime"] == regime]
        for policy in ("entropy_q90", "variance_q90", "random", "oracle_error"):
            subset = regime_data[regime_data["policy"] == policy].sort_values("budget_k")
            axis.plot(
                subset["coverage_fraction"] * 100,
                subset["error_capture_mean"] * 100,
                marker="o",
                color=POLICY_COLORS[policy],
                label=POLICY_LABELS[policy],
            )
        axis.plot([0, 100], [0, 100], "--", color="#CBD5E1", linewidth=1)
        axis.set_title(f"Regime {regime} (n=10 images)")
        axis.set_xlabel("Coverage (%)")
        axis.set_ylabel("Error captured (%)")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Tile-selection performance by source regime", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_examples(
    representatives: pd.DataFrame,
    heldout: pd.DataFrame,
    evaluated_tiles: pd.DataFrame,
    per_image: pd.DataFrame,
    project_root: Path,
    budget: int,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(representatives), 3, figsize=(13, 3.1 * len(representatives)), constrained_layout=True
    )
    entropy_image = None
    error_image = None
    heldout_index = heldout.set_index("sample_id")
    for row_index, representative in representatives.iterrows():
        sample_id = representative["sample_id"]
        record = heldout_index.loc[sample_id]
        native = load_grayscale(project_root / record["source_image_path"])
        tiles = evaluated_tiles[evaluated_tiles["sample_id"] == sample_id].sort_values(
            "tile_index"
        )
        entropy_grid = tiles["entropy_p90_bits"].to_numpy().reshape(6, 8)
        error_grid = tiles["error_fraction"].to_numpy().reshape(6, 8)
        selected = tiles[tiles["entropy_rank"] <= budget]
        capture = per_image[
            (per_image["sample_id"] == sample_id)
            & (per_image["policy"] == "entropy_q90")
            & (per_image["budget_k"] == budget)
        ]["error_capture"].iloc[0]

        axes[row_index, 0].imshow(native, cmap="gray", vmin=0, vmax=255)
        for tile in selected.itertuples(index=False):
            axes[row_index, 0].add_patch(
                Rectangle(
                    (tile.native_x0, tile.native_y0),
                    tile.native_x1 - tile.native_x0,
                    tile.native_y1 - tile.native_y0,
                    fill=False,
                    edgecolor="#22D3EE",
                    linewidth=2.0,
                )
            )
        axes[row_index, 0].set_ylabel(
            f"{sample_id}\n{representative['example_role']}\nK={budget}, capture={capture:.1%}",
            fontsize=8,
        )
        entropy_image = axes[row_index, 1].imshow(
            entropy_grid, cmap="magma", vmin=0.0, vmax=1.0
        )
        error_image = axes[row_index, 2].imshow(
            error_grid, cmap="Reds", vmin=0.0, vmax=max(0.5, float(error_grid.max()))
        )
        for tile in selected.itertuples(index=False):
            axes[row_index, 2].add_patch(
                Rectangle(
                    (tile.tile_column - 0.5, tile.tile_row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#0891B2",
                    linewidth=2.0,
                )
            )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title(f"Native SEM with top-{budget} entropy tiles")
    axes[0, 1].set_title("Tile entropy q90 (ranking input)")
    axes[0, 2].set_title("Tile coarse-error fraction (evaluation only)")
    if entropy_image is not None:
        figure.colorbar(entropy_image, ax=axes[:, 1], fraction=0.025, pad=0.02, label="bits")
    if error_image is not None:
        figure.colorbar(error_image, ax=axes[:, 2], fraction=0.025, pad=0.02, label="fraction")
    figure.suptitle(
        "Uncertainty rankings are generated before the ground-truth error map is loaded",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)

    start = time.perf_counter()
    output_dir = project_root / config["output_dir"]
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    grid = TileGrid(
        native_height=int(config["native_height"]),
        native_width=int(config["native_width"]),
        coarse_height=int(config["coarse_height"]),
        coarse_width=int(config["coarse_width"]),
        native_tile_size=int(config["native_tile_size"]),
    )
    budgets = [int(value) for value in config["budgets"]]
    if budgets != sorted(set(budgets)) or budgets[-1] != grid.tile_count:
        raise ValueError("Budgets must be unique, sorted, and include full coverage")
    random_trials_count = int(config["random_trials"])
    if random_trials_count < 20:
        raise ValueError("At least 20 random trials are required")

    coarse_manifest = pd.read_csv(project_root / config["coarse_manifest"])
    cv_manifest = pd.read_csv(project_root / config["cv_manifest"])
    step3_metrics = pd.read_csv(project_root / config["step3_metrics"])
    heldout = cv_manifest[cv_manifest["split"] == "test"].merge(
        coarse_manifest,
        on=["sample_id", "file_name", "source_regime"],
        how="left",
        validate="one_to_one",
    )
    heldout = heldout.sort_values(["outer_fold", "sample_id"]).reset_index(drop=True)
    if len(heldout) != 40 or heldout["sample_id"].nunique() != 40:
        raise ValueError("Expected 40 unique held-out images across five folds")
    if set(heldout["sample_id"]) != set(step3_metrics["sample_id"]):
        raise ValueError("Step 3 uncertainty metrics do not match the held-out manifest")

    # Pass 1: generate all uncertainty and random rankings without loading labels.
    ranking_frames: list[pd.DataFrame] = []
    random_ranks_by_sample: dict[str, np.ndarray] = {}
    for record in heldout.itertuples(index=False):
        uncertainty_path = project_root / config["uncertainty_pattern"].format(
            sample_id=record.sample_id
        )
        with np.load(uncertainty_path) as uncertainty:
            if any("mask" in key or "target" in key or "error" in key for key in uncertainty.files):
                raise ValueError(f"Label-like data found in {uncertainty_path}")
            rankings = uncertainty_tile_rankings(
                uncertainty["predictive_entropy_bits"],
                uncertainty["predictive_variance"],
                grid,
                percentile=float(config["tile_score_percentile"]),
            )
        rankings.insert(0, "source_regime", record.source_regime)
        rankings.insert(0, "file_name", record.file_name)
        rankings.insert(0, "sample_id", record.sample_id)
        rankings.insert(0, "outer_fold", int(record.outer_fold))
        ranking_frames.append(rankings)
        sample_number = int(str(record.sample_id).split("_")[-1])
        random_ranks_by_sample[record.sample_id] = random_rank_matrix(
            grid.tile_count,
            random_trials_count,
            int(config["random_seed_base"]),
            sample_number,
        )
    uncertainty_rankings = pd.concat(ranking_frames, ignore_index=True)
    uncertainty_rankings.to_csv(metrics_dir / "uncertainty_rankings.csv", index=False)

    # Pass 2: labels and saved predictions enter only for selection-quality evaluation.
    evaluated_frames: list[pd.DataFrame] = []
    correlation_rows: list[dict[str, object]] = []
    policy_frames: list[pd.DataFrame] = []
    random_trial_frames: list[pd.DataFrame] = []
    for record in heldout.itertuples(index=False):
        sample_rankings = uncertainty_rankings[
            uncertainty_rankings["sample_id"] == record.sample_id
        ].copy()
        target = load_grayscale(project_root / record.coarse_mask_path) > 0
        prediction_path = project_root / config["prediction_pattern"].format(
            fold=int(record.outer_fold), sample_id=record.sample_id
        )
        prediction = load_grayscale(prediction_path) > 0
        error_map = prediction != target
        evaluated = attach_error_evaluation(sample_rankings, error_map, grid)
        evaluated_frames.append(evaluated)

        correlations = tile_error_correlations(evaluated)
        correlation_rows.append(
            {
                "outer_fold": int(record.outer_fold),
                "sample_id": record.sample_id,
                "source_regime": record.source_regime,
                "total_error_pixels": int(error_map.sum()),
                **correlations,
            }
        )

        for policy, rank_column in POLICY_RANK_COLUMNS.items():
            metrics = selection_metrics(evaluated, rank_column, budgets)
            metrics.insert(0, "source_regime", record.source_regime)
            metrics.insert(0, "sample_id", record.sample_id)
            metrics.insert(0, "outer_fold", int(record.outer_fold))
            metrics.insert(3, "policy", policy)
            metrics["random_trials"] = 0
            metrics["random_trial_std"] = np.nan
            metrics["random_trial_ci_low"] = np.nan
            metrics["random_trial_ci_high"] = np.nan
            policy_frames.append(metrics)

        ranks = random_ranks_by_sample[record.sample_id]
        trial_rows = []
        for trial_index in range(random_trials_count):
            random_tiles = evaluated.copy()
            random_tiles["random_rank"] = ranks[trial_index]
            trial_metrics = selection_metrics(random_tiles, "random_rank", budgets)
            trial_metrics.insert(0, "source_regime", record.source_regime)
            trial_metrics.insert(0, "sample_id", record.sample_id)
            trial_metrics.insert(0, "outer_fold", int(record.outer_fold))
            trial_metrics.insert(3, "random_trial_index", trial_index)
            trial_metrics.insert(4, "random_seed_base", int(config["random_seed_base"]))
            trial_rows.append(trial_metrics)
        trials = pd.concat(trial_rows, ignore_index=True)
        random_trial_frames.append(trials)
        random_mean = (
            trials.groupby("budget_k", as_index=False)
            .agg(
                coverage_fraction=("coverage_fraction", "mean"),
                selected_error_pixels=("selected_error_pixels", "mean"),
                total_error_pixels=("total_error_pixels", "first"),
                error_capture=("error_capture", "mean"),
                selected_error_rate=("selected_error_rate", "mean"),
                error_density_enrichment=("error_density_enrichment", "mean"),
                random_trial_std=("error_capture", "std"),
                random_trial_ci_low=("error_capture", lambda values: values.quantile(0.025)),
                random_trial_ci_high=("error_capture", lambda values: values.quantile(0.975)),
            )
        )
        random_mean.insert(0, "source_regime", record.source_regime)
        random_mean.insert(0, "sample_id", record.sample_id)
        random_mean.insert(0, "outer_fold", int(record.outer_fold))
        random_mean.insert(3, "policy", "random")
        random_mean["random_trials"] = random_trials_count
        policy_frames.append(random_mean)

    evaluated_tiles = pd.concat(evaluated_frames, ignore_index=True).sort_values(
        ["outer_fold", "sample_id", "tile_index"]
    )
    per_image_correlations = pd.DataFrame(correlation_rows).sort_values(
        ["outer_fold", "sample_id"]
    )
    per_image = pd.concat(policy_frames, ignore_index=True).sort_values(
        ["outer_fold", "sample_id", "policy", "budget_k"]
    )
    random_trials = pd.concat(random_trial_frames, ignore_index=True).sort_values(
        ["outer_fold", "sample_id", "random_trial_index", "budget_k"]
    )

    random_reference = per_image[per_image["policy"] == "random"][
        ["sample_id", "budget_k", "error_capture"]
    ].rename(columns={"error_capture": "random_error_capture"})
    per_image = per_image.merge(
        random_reference, on=["sample_id", "budget_k"], validate="many_to_one"
    )
    per_image["capture_minus_random"] = (
        per_image["error_capture"] - per_image["random_error_capture"]
    )
    per_image["capture_enrichment_over_random"] = (
        per_image["error_capture"] / per_image["random_error_capture"]
    )

    confidence_level = float(config["confidence_level"])
    resamples = int(config["bootstrap_resamples"])
    bootstrap_seed = int(config["bootstrap_seed"])
    budget_summary = summarize_policy_budgets(
        per_image, confidence_level, resamples, bootstrap_seed
    )
    regime_summary = summarize_policy_budgets(
        per_image,
        confidence_level,
        resamples,
        bootstrap_seed + 1000,
        group_column="source_regime",
    )
    paired = paired_policy_comparisons(
        per_image,
        budgets,
        confidence_level,
        resamples,
        bootstrap_seed + 2000,
    )
    correlations = correlation_summary(
        per_image_correlations,
        evaluated_tiles,
        confidence_level,
        resamples,
        bootstrap_seed + 3000,
    )

    representatives = select_representatives(step3_metrics)
    representatives.to_csv(metrics_dir / "representative_samples.csv", index=False)
    evaluated_tiles.to_csv(metrics_dir / "tile_evaluation.csv", index=False)
    per_image_correlations.to_csv(metrics_dir / "per_image_tile_correlations.csv", index=False)
    per_image.to_csv(metrics_dir / "per_image_budget_metrics.csv", index=False)
    random_trials.to_csv(metrics_dir / "random_selection_trials.csv", index=False)
    budget_summary.to_csv(metrics_dir / "budget_summary.csv", index=False)
    regime_summary.to_csv(metrics_dir / "regime_budget_summary.csv", index=False)
    paired.to_csv(metrics_dir / "paired_policy_vs_random.csv", index=False)
    correlations.to_csv(metrics_dir / "correlation_summary.csv", index=False)

    plot_overall_performance(
        budget_summary,
        paired,
        correlations,
        figures_dir / "tile_selection_performance.png",
    )
    plot_regime_performance(
        regime_summary,
        figures_dir / "tile_selection_by_regime.png",
    )
    plot_examples(
        representatives,
        heldout,
        evaluated_tiles,
        per_image,
        project_root,
        int(config["example_budget"]),
        figures_dir / "tile_selection_examples.png",
    )

    entropy_paired = paired[
        (paired["scope"] == "all_images") & (paired["policy"] == "entropy_q90")
    ].sort_values("budget_k")
    entropy_correlations = correlations[
        (correlations["scope"] == "all_images")
        & (correlations["policy"] == "entropy_q90")
    ]
    summary: dict[str, object] = {
        "experiment_name": config["experiment_name"],
        "held_out_images": int(heldout["sample_id"].nunique()),
        "source_regime_images": {
            key: int(value)
            for key, value in heldout.groupby("source_regime").size().to_dict().items()
        },
        "tile_geometry": {
            "native_shape": [grid.native_height, grid.native_width],
            "coarse_shape": [grid.coarse_height, grid.coarse_width],
            "native_tile_size": grid.native_tile_size,
            "coarse_tile_size": grid.coarse_tile_size,
            "scale_factor": grid.scale_factor,
            "rows": grid.rows,
            "columns": grid.columns,
            "tiles_per_image": grid.tile_count,
        },
        "ranking": {
            "primary_policy": "90th percentile predictive entropy within each 64 x 64 coarse block",
            "secondary_policy": "90th percentile predictive variance within each 64 x 64 coarse block",
            "ranking_scope": "within each image",
            "ground_truth_use": "evaluation and oracle only, after uncertainty and random rankings were generated",
        },
        "budgets": budgets,
        "coverage_fractions": [budget / grid.tile_count for budget in budgets],
        "random_trials_per_image": random_trials_count,
        "paired_confidence_intervals": {
            "method": "percentile bootstrap over paired image-level capture differences",
            "confidence_level": confidence_level,
            "resamples": resamples,
            "random_seed": bootstrap_seed,
        },
        "entropy_overall": {
            "budget_results": entropy_paired.to_dict(orient="records"),
            "correlations": entropy_correlations.to_dict(orient="records"),
        },
        "overall_budget_summary": budget_summary.to_dict(orient="records"),
        "regime_budget_summary": regime_summary.to_dict(orient="records"),
        "runtime": {
            "seconds": time.perf_counter() - start,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "opencv_version": cv2.__version__,
            "platform": platform.platform(),
            "command": " ".join(sys.argv),
        },
        "config": config,
    }
    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "run_environment.json").open("w", encoding="utf-8") as handle:
        json.dump(summary["runtime"], handle, indent=2)

    print(
        budget_summary[
            budget_summary["policy"].isin(["entropy_q90", "random", "variance_q90", "oracle_error"])
        ][
            [
                "policy",
                "budget_k",
                "coverage_fraction",
                "error_capture_mean",
                "capture_enrichment_over_random",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
