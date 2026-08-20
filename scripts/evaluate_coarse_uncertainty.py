#!/usr/bin/env python3
"""Evaluate 8-pass MC-dropout uncertainty on held-out coarse-model images."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multiscale-imaging-matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

from adaptive_multiscale.models import CompactUNet
from adaptive_multiscale.training.metrics import binary_segmentation_metrics
from adaptive_multiscale.training.reproducibility import seed_everything, select_device
from adaptive_multiscale.uncertainty.evaluation import (
    local_region_correlations,
    local_region_table,
    top_region_concentration,
    top_uncertainty_concentration,
    uncertainty_error_statistics,
)
from adaptive_multiscale.uncertainty.mc_dropout import mc_dropout_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/coarse_uncertainty.json")
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def make_model(checkpoint: dict[str, Any], device: torch.device) -> CompactUNet:
    model = CompactUNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def normalized_tensor(
    image: np.ndarray,
    mean: float,
    std: float,
    device: torch.device,
) -> torch.Tensor:
    values = image.astype(np.float32) / 255.0
    values = (values - mean) / std
    return torch.from_numpy(values[None, None]).to(device)


def prefixed(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def concentration_rows(
    uncertainty: np.ndarray,
    error: np.ndarray,
    fractions: list[float],
    scope: str,
    level: str,
    metric_name: str,
) -> list[dict[str, object]]:
    values = top_uncertainty_concentration(uncertainty, error, fractions)
    rows = []
    for fraction in fractions:
        label = f"top_{int(round(fraction * 100)):02d}pct"
        rows.append(
            {
                "scope": scope,
                "level": level,
                "uncertainty_metric": metric_name,
                "selected_fraction": values[f"{label}_selected_fraction"],
                "error_capture": values[f"{label}_error_capture"],
                "error_enrichment": values[f"{label}_error_enrichment"],
                "selected_error_rate": values[f"{label}_selected_error_rate"],
                "global_error_rate": float(np.asarray(error, dtype=bool).mean()),
            }
        )
    return rows


def region_concentration_rows(
    regions: pd.DataFrame,
    uncertainty_column: str,
    fractions: list[float],
    scope: str,
    metric_name: str,
) -> list[dict[str, object]]:
    values = top_region_concentration(regions, uncertainty_column, fractions)
    global_error_rate = float(regions["error_pixels"].sum() / regions["pixels"].sum())
    rows = []
    for fraction in fractions:
        label = f"top_{int(round(fraction * 100)):02d}pct_regions"
        rows.append(
            {
                "scope": scope,
                "level": "local_region",
                "uncertainty_metric": metric_name,
                "selected_fraction": values[f"{label}_selected_fraction"],
                "error_capture": values[f"{label}_error_capture"],
                "error_enrichment": values[f"{label}_error_enrichment"],
                "selected_error_rate": values[f"{label}_selected_error_rate"],
                "global_error_rate": global_error_rate,
            }
        )
    return rows


def select_representatives(metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose deterministic, non-cherry-picked examples plus the hardest Regime I case."""

    selected: list[pd.Series] = []
    regime_i = metrics[metrics["source_regime"] == "I"].copy()
    hardest = regime_i.sort_values(["deterministic_dice", "sample_id"]).iloc[0].copy()
    hardest["example_role"] = "Regime I hardest coarse case"
    selected.append(hardest)
    for regime in ("I", "II", "III", "IV"):
        group = metrics[metrics["source_regime"] == regime].copy()
        median = group["deterministic_dice"].median()
        group["distance_to_regime_median_dice"] = (
            group["deterministic_dice"] - median
        ).abs()
        row = group.sort_values(
            ["distance_to_regime_median_dice", "sample_id"]
        ).iloc[0].copy()
        row["example_role"] = f"Regime {regime} median-Dice example"
        selected.append(row)
    result = pd.DataFrame(selected).drop_duplicates("sample_id", keep="first")
    return result.reset_index(drop=True)


def error_code_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    codes = np.zeros(target.shape, dtype=np.uint8)
    codes[prediction & ~target] = 1
    codes[~prediction & target] = 2
    return codes


def plot_one_example(
    row: pd.Series,
    project_root: Path,
    maps_dir: Path,
    prediction_pattern: str,
    output_path: Path,
) -> None:
    image = load_grayscale(project_root / row["coarse_image_path"])
    target = load_grayscale(project_root / row["coarse_mask_path"]) > 0
    prediction = load_grayscale(
        project_root
        / prediction_pattern.format(
            fold=int(row["outer_fold"]), sample_id=row["sample_id"]
        )
    ) > 0
    with np.load(maps_dir / f"{row['sample_id']}.npz") as maps:
        entropy = maps["predictive_entropy_bits"]

    figure, axes = plt.subplots(1, 5, figsize=(18, 3.8), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Coarse SEM")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Step 2 prediction\nDice {row['deterministic_dice']:.3f}")
    axes[3].imshow(
        error_code_map(prediction, target),
        cmap=ListedColormap(["#111827", "#EF4444", "#22D3EE"]),
        vmin=0,
        vmax=2,
    )
    axes[3].set_title("Segmentation error")
    heatmap = axes[4].imshow(entropy, cmap="magma", vmin=0.0, vmax=1.0)
    axes[4].set_title(
        f"Predictive entropy (bits)\nerror AUC {row['entropy_error_roc_auc']:.3f}"
    )
    for axis in axes:
        axis.axis("off")
    figure.colorbar(heatmap, ax=axes[4], fraction=0.047, pad=0.03, label="bits")
    figure.legend(
        handles=[
            Patch(facecolor="#111827", label="Correct"),
            Patch(facecolor="#EF4444", label="False positive"),
            Patch(facecolor="#22D3EE", label="False negative"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        f"{row['sample_id']} — {row['example_role']} ({row['source_regime']})",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_representative_grid(
    representatives: pd.DataFrame,
    project_root: Path,
    maps_dir: Path,
    prediction_pattern: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(representatives), 5, figsize=(18, 3.25 * len(representatives)), constrained_layout=True
    )
    last_heatmap = None
    for row_index, row in representatives.iterrows():
        image = load_grayscale(project_root / row["coarse_image_path"])
        target = load_grayscale(project_root / row["coarse_mask_path"]) > 0
        prediction = load_grayscale(
            project_root
            / prediction_pattern.format(
                fold=int(row["outer_fold"]), sample_id=row["sample_id"]
            )
        ) > 0
        with np.load(maps_dir / f"{row['sample_id']}.npz") as maps:
            entropy = maps["predictive_entropy_bits"]
        axes[row_index, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 3].imshow(
            error_code_map(prediction, target),
            cmap=ListedColormap(["#111827", "#EF4444", "#22D3EE"]),
            vmin=0,
            vmax=2,
        )
        last_heatmap = axes[row_index, 4].imshow(
            entropy, cmap="magma", vmin=0.0, vmax=1.0
        )
        axes[row_index, 0].set_ylabel(
            f"{row['sample_id']}\n{row['example_role']}\nDice {row['deterministic_dice']:.3f}\nAUC {row['entropy_error_roc_auc']:.3f}",
            fontsize=8,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    for column, title in enumerate(
        [
            "Coarse SEM",
            "Ground truth",
            "Step 2 prediction",
            "Error map",
            "MC predictive entropy",
        ]
    ):
        axes[0, column].set_title(title, fontsize=11)
    if last_heatmap is not None:
        figure.colorbar(
            last_heatmap,
            ax=axes[:, 4],
            fraction=0.025,
            pad=0.02,
            label="Predictive entropy (bits)",
        )
    figure.legend(
        handles=[
            Patch(facecolor="#111827", label="Correct"),
            Patch(facecolor="#EF4444", label="False positive"),
            Patch(facecolor="#22D3EE", label="False negative"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Held-out MC-dropout uncertainty: 8 stochastic passes per coarse image",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_quantitative_summary(
    per_image: pd.DataFrame,
    regime: pd.DataFrame,
    concentration: pd.DataFrame,
    output_path: Path,
) -> None:
    order = ["I", "II", "III", "IV"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    positions = np.arange(len(order))
    correct = regime.set_index("source_regime").loc[order, "entropy_correct_mean"]
    incorrect = regime.set_index("source_regime").loc[order, "entropy_incorrect_mean"]
    width = 0.36
    axes[0, 0].bar(positions - width / 2, correct, width, label="Correct pixels", color="#94A3B8")
    axes[0, 0].bar(positions + width / 2, incorrect, width, label="Error pixels", color="#EF4444")
    axes[0, 0].set_xticks(positions, [f"Regime {value}" for value in order])
    axes[0, 0].set_ylabel("Mean predictive entropy (bits)")
    axes[0, 0].set_title("Uncertainty on correct versus error pixels")
    axes[0, 0].legend(frameon=False)

    pooled_pixels = concentration[
        (concentration["scope"] == "all_images_pooled")
        & (concentration["level"] == "pixel")
    ]
    for metric, label, color in (
        ("predictive_entropy_bits", "Entropy", "#7C3AED"),
        ("predictive_variance", "Variance", "#0891B2"),
    ):
        subset = pooled_pixels[pooled_pixels["uncertainty_metric"] == metric]
        axes[0, 1].plot(
            np.r_[0.0, subset["selected_fraction"]],
            np.r_[0.0, subset["error_capture"]],
            marker="o",
            label=label,
            color=color,
        )
    axes[0, 1].plot([0, 0.3], [0, 0.3], "--", color="#64748B", label="Random expectation")
    axes[0, 1].set_xlim(0, 0.31)
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_xlabel("Highest-uncertainty pixel fraction")
    axes[0, 1].set_ylabel("Fraction of all error captured")
    axes[0, 1].set_title("Error concentration in uncertain pixels")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].bar(
        positions,
        regime.set_index("source_regime").loc[
            order, "region_entropy_error_pearson_image_mean"
        ],
        color=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"],
    )
    axes[1, 0].axhline(0.0, color="#334155", linewidth=0.8)
    axes[1, 0].set_xticks(positions, [f"Regime {value}" for value in order])
    axes[1, 0].set_ylim(-0.1, 1.0)
    axes[1, 0].set_ylabel("Mean within-image 64 x 64 region correlation")
    axes[1, 0].set_title("Local uncertainty versus local error")

    axes[1, 1].scatter(
        per_image["deterministic_dice"],
        per_image["mc_mean_dice"],
        c=per_image["source_regime"].map(
            {"I": "#3B82F6", "II": "#10B981", "III": "#F59E0B", "IV": "#8B5CF6"}
        ),
        edgecolor="white",
        linewidth=0.5,
        s=55,
    )
    axes[1, 1].plot([0.5, 1.0], [0.5, 1.0], "--", color="#64748B")
    axes[1, 1].set_xlim(0.5, 1.0)
    axes[1, 1].set_ylim(0.5, 1.0)
    axes[1, 1].set_xlabel("Step 2 deterministic Dice")
    axes[1, 1].set_ylabel("8-pass MC-mean Dice")
    axes[1, 1].set_title("MC mean preserves the coarse prediction")

    for axis in axes.flat:
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    figure.suptitle("MC-dropout uncertainty/error relationship", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def paired_wilcoxon_greater(incorrect: pd.Series, correct: pd.Series) -> float:
    return float(
        wilcoxon(
            incorrect.to_numpy(),
            correct.to_numpy(),
            alternative="greater",
            zero_method="wilcox",
        ).pvalue
    )


def pooled_result(
    entropy: np.ndarray,
    variance: np.ndarray,
    error: np.ndarray,
    regions: pd.DataFrame,
    fractions: list[float],
) -> dict[str, object]:
    entropy_stats = uncertainty_error_statistics(entropy, error)
    variance_stats = uncertainty_error_statistics(variance, error)
    result: dict[str, object] = {
        "pixels": int(error.size),
        "error_pixels": int(error.sum()),
        "error_rate": float(error.mean()),
        "entropy": {
            **entropy_stats,
            "average_precision_lift_over_error_prevalence": (
                entropy_stats["error_average_precision"] / entropy_stats["error_rate"]
            ),
            **top_uncertainty_concentration(entropy, error, fractions),
        },
        "predictive_variance": {
            **variance_stats,
            "average_precision_lift_over_error_prevalence": (
                variance_stats["error_average_precision"] / variance_stats["error_rate"]
            ),
            **top_uncertainty_concentration(variance, error, fractions),
        },
        "local_regions": {
            "count": int(len(regions)),
            **local_region_correlations(regions),
            **top_region_concentration(
                regions, "mean_entropy_bits", fractions
            ),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = read_json(config_path)
    output_dir = project_root / config["output_dir"]
    metrics_dir = output_dir / "metrics"
    maps_dir = output_dir / "uncertainty_maps"
    mc_masks_dir = output_dir / "mc_mean_masks"
    figures_dir = output_dir / "figures"
    examples_dir = figures_dir / "examples"
    for directory in (metrics_dir, maps_dir, mc_masks_dir, figures_dir, examples_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if int(config["cpu_threads"]) > 0:
        torch.set_num_threads(int(config["cpu_threads"]))
    device = select_device(str(config["device"]))
    passes = int(config["mc_passes"])
    threshold = float(config["prediction_threshold"])
    fractions = [float(value) for value in config["top_uncertainty_fractions"]]
    region_size = int(config["local_region_size"])
    base_seed = int(config["random_seed"])
    start = time.perf_counter()

    coarse_manifest = pd.read_csv(project_root / config["coarse_manifest"])
    cv_manifest = pd.read_csv(project_root / config["cv_manifest"])
    coarse_metrics = pd.read_csv(project_root / config["coarse_metrics"])
    if len(coarse_metrics) != 40 or coarse_metrics["sample_id"].nunique() != 40:
        raise ValueError("Expected exactly 40 held-out Step 2 predictions")

    per_image_rows: list[dict[str, object]] = []
    local_frames: list[pd.DataFrame] = []
    arrays_by_sample: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    dropout_counts: set[int] = set()

    print(f"device={device} mc_passes={passes} folds={config['folds']}", flush=True)
    for fold in config["folds"]:
        checkpoint_path = project_root / config["checkpoint_pattern"].format(fold=fold)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if int(checkpoint["outer_fold"]) != int(fold):
            raise ValueError(f"Checkpoint/fold mismatch at {checkpoint_path}")
        if abs(float(checkpoint["prediction_threshold"]) - threshold) > 1e-12:
            raise ValueError(f"Threshold mismatch at fold {fold}")
        model = make_model(checkpoint, device)
        test = cv_manifest[
            (cv_manifest["outer_fold"] == fold) & (cv_manifest["split"] == "test")
        ].merge(
            coarse_manifest,
            on=["sample_id", "file_name", "source_regime"],
            how="left",
            validate="one_to_one",
        )
        if len(test) != 8:
            raise ValueError(f"Expected 8 test images in fold {fold}, found {len(test)}")

        for record in test.sort_values("sample_id").itertuples(index=False):
            image = load_grayscale(project_root / record.coarse_image_path)
            image_tensor = normalized_tensor(
                image,
                float(checkpoint["normalization_mean"]),
                float(checkpoint["normalization_std"]),
                device,
            )
            sample_number = int(str(record.sample_id).split("_")[-1])
            sample_seed = base_seed + int(fold) * 1000 + sample_number
            seed_everything(sample_seed)

            # This inference step has no access to ground truth or the saved prediction.
            mc_prediction = mc_dropout_predict(model, image_tensor, passes=passes)
            dropout_counts.add(mc_prediction.dropout_modules_enabled)
            np.savez_compressed(
                maps_dir / f"{record.sample_id}.npz",
                mean_foreground_probability=mc_prediction.mean_probability,
                predictive_entropy_bits=mc_prediction.predictive_entropy_bits,
                predictive_variance=mc_prediction.predictive_variance,
                mc_passes=np.int32(passes),
                random_seed=np.int32(sample_seed),
                outer_fold=np.int32(fold),
                prediction_threshold=np.float32(threshold),
            )
            mc_mask = mc_prediction.mean_probability >= threshold
            if not cv2.imwrite(
                str(mc_masks_dir / f"{record.sample_id}.png"),
                mc_mask.astype(np.uint8) * 255,
            ):
                raise OSError(f"Could not save MC-mean mask for {record.sample_id}")

            # Labels enter only below, after uncertainty has been generated and saved.
            target = load_grayscale(project_root / record.coarse_mask_path) > 0
            deterministic_path = project_root / config["prediction_pattern"].format(
                fold=fold, sample_id=record.sample_id
            )
            deterministic = load_grayscale(deterministic_path) > 0
            deterministic_error = deterministic != target
            mc_error = mc_mask != target
            deterministic_metrics = binary_segmentation_metrics(deterministic, target)
            mc_metrics = binary_segmentation_metrics(mc_mask, target)
            recorded = coarse_metrics[coarse_metrics["sample_id"] == record.sample_id]
            if len(recorded) != 1 or abs(
                float(recorded.iloc[0]["dice"]) - float(deterministic_metrics["dice"])
            ) > 1e-12:
                raise ValueError(f"Step 2 prediction metric mismatch for {record.sample_id}")

            entropy_stats = uncertainty_error_statistics(
                mc_prediction.predictive_entropy_bits, deterministic_error
            )
            variance_stats = uncertainty_error_statistics(
                mc_prediction.predictive_variance, deterministic_error
            )
            entropy_concentration = top_uncertainty_concentration(
                mc_prediction.predictive_entropy_bits,
                deterministic_error,
                fractions,
            )
            variance_concentration = top_uncertainty_concentration(
                mc_prediction.predictive_variance,
                deterministic_error,
                fractions,
            )
            regions = local_region_table(
                mc_prediction.predictive_entropy_bits,
                mc_prediction.predictive_variance,
                deterministic_error,
                region_size,
                {
                    "outer_fold": int(fold),
                    "sample_id": record.sample_id,
                    "source_regime": record.source_regime,
                },
            )
            region_stats = local_region_correlations(regions)
            region_concentration = top_region_concentration(
                regions, "mean_entropy_bits", fractions
            )
            local_frames.append(regions)
            arrays_by_sample[record.sample_id] = (
                mc_prediction.predictive_entropy_bits.ravel().copy(),
                mc_prediction.predictive_variance.ravel().copy(),
                deterministic_error.ravel().copy(),
            )

            per_image_rows.append(
                {
                    "outer_fold": int(fold),
                    "sample_id": record.sample_id,
                    "file_name": record.file_name,
                    "source_regime": record.source_regime,
                    "mc_passes": passes,
                    "random_seed": sample_seed,
                    "dropout_modules_enabled": mc_prediction.dropout_modules_enabled,
                    "prediction_threshold": threshold,
                    "deterministic_dice": deterministic_metrics["dice"],
                    "mc_mean_dice": mc_metrics["dice"],
                    "mc_minus_deterministic_dice": (
                        float(mc_metrics["dice"]) - float(deterministic_metrics["dice"])
                    ),
                    "deterministic_mc_pixel_agreement": float(
                        (deterministic == mc_mask).mean()
                    ),
                    "deterministic_error_rate": float(deterministic_error.mean()),
                    "mc_mean_error_rate": float(mc_error.mean()),
                    "mean_foreground_probability": float(
                        mc_prediction.mean_probability.mean()
                    ),
                    "mean_predictive_entropy_bits": float(
                        mc_prediction.predictive_entropy_bits.mean()
                    ),
                    "mean_predictive_variance": float(
                        mc_prediction.predictive_variance.mean()
                    ),
                    **prefixed(entropy_stats, "entropy"),
                    **prefixed(variance_stats, "variance"),
                    **prefixed(entropy_concentration, "entropy"),
                    **prefixed(variance_concentration, "variance"),
                    **region_stats,
                    **prefixed(region_concentration, "entropy"),
                    "coarse_image_path": record.coarse_image_path,
                    "coarse_mask_path": record.coarse_mask_path,
                    "deterministic_prediction_path": str(
                        deterministic_path.relative_to(project_root)
                    ),
                    "uncertainty_map_path": str(
                        (maps_dir / f"{record.sample_id}.npz").relative_to(project_root)
                    ),
                    "mc_mean_mask_path": str(
                        (mc_masks_dir / f"{record.sample_id}.png").relative_to(project_root)
                    ),
                }
            )
            print(
                f"fold={fold} sample={record.sample_id} dice={deterministic_metrics['dice']:.3f} "
                f"entropy_auc={entropy_stats['error_roc_auc']:.3f} "
                f"top20_capture={entropy_concentration['top_20pct_error_capture']:.3f}",
                flush=True,
            )

    if dropout_counts != {1}:
        raise ValueError(f"Unexpected dropout module counts: {dropout_counts}")
    per_image = pd.DataFrame(per_image_rows).sort_values(
        ["outer_fold", "sample_id"]
    ).reset_index(drop=True)
    local_regions = pd.concat(local_frames, ignore_index=True).sort_values(
        ["outer_fold", "sample_id", "region_index"]
    )
    if len(per_image) != 40 or per_image["sample_id"].nunique() != 40:
        raise ValueError("MC inference did not produce exactly 40 unique images")
    if len(local_regions) != 40 * 48:
        raise ValueError("Expected 48 local analysis regions for each image")

    all_entropy = np.concatenate([arrays_by_sample[value][0] for value in per_image["sample_id"]])
    all_variance = np.concatenate([arrays_by_sample[value][1] for value in per_image["sample_id"]])
    all_error = np.concatenate([arrays_by_sample[value][2] for value in per_image["sample_id"]])
    overall = pooled_result(
        all_entropy, all_variance, all_error, local_regions, fractions
    )
    overall["per_image_macro"] = {
        "entropy_error_pearson_mean": float(per_image["entropy_error_pearson"].mean()),
        "entropy_error_pearson_std": float(per_image["entropy_error_pearson"].std(ddof=1)),
        "entropy_error_roc_auc_mean": float(per_image["entropy_error_roc_auc"].mean()),
        "entropy_error_roc_auc_std": float(per_image["entropy_error_roc_auc"].std(ddof=1)),
        "entropy_error_average_precision_mean": float(
            per_image["entropy_error_average_precision"].mean()
        ),
        "entropy_incorrect_mean": float(per_image["entropy_incorrect_mean"].mean()),
        "entropy_correct_mean": float(per_image["entropy_correct_mean"].mean()),
        "entropy_incorrect_to_correct_ratio_mean": float(
            per_image["entropy_incorrect_to_correct_ratio"].mean()
        ),
        "variance_error_pearson_mean": float(per_image["variance_error_pearson"].mean()),
        "region_entropy_error_pearson_mean": float(
            per_image["region_entropy_error_pearson"].mean()
        ),
        "deterministic_dice_mean": float(per_image["deterministic_dice"].mean()),
        "mc_mean_dice_mean": float(per_image["mc_mean_dice"].mean()),
        "mc_minus_deterministic_dice_mean": float(
            per_image["mc_minus_deterministic_dice"].mean()
        ),
        "deterministic_mc_pixel_agreement_mean": float(
            per_image["deterministic_mc_pixel_agreement"].mean()
        ),
        "entropy_incorrect_greater_wilcoxon_p": paired_wilcoxon_greater(
            per_image["entropy_incorrect_mean"], per_image["entropy_correct_mean"]
        ),
        "variance_incorrect_greater_wilcoxon_p": paired_wilcoxon_greater(
            per_image["variance_incorrect_mean"], per_image["variance_correct_mean"]
        ),
    }
    for fraction in fractions:
        percent = int(round(fraction * 100))
        overall["per_image_macro"].update(
            {
                f"entropy_top_{percent:02d}pct_error_capture_mean": float(
                    per_image[f"entropy_top_{percent:02d}pct_error_capture"].mean()
                ),
                f"entropy_top_{percent:02d}pct_error_enrichment_mean": float(
                    per_image[f"entropy_top_{percent:02d}pct_error_enrichment"].mean()
                ),
                f"entropy_top_{percent:02d}pct_regions_error_capture_mean": float(
                    per_image[
                        f"entropy_top_{percent:02d}pct_regions_error_capture"
                    ].mean()
                ),
                f"entropy_top_{percent:02d}pct_regions_error_enrichment_mean": float(
                    per_image[
                        f"entropy_top_{percent:02d}pct_regions_error_enrichment"
                    ].mean()
                ),
            }
        )

    regime_rows: list[dict[str, object]] = []
    concentration_records: list[dict[str, object]] = []
    concentration_records.extend(
        concentration_rows(
            all_entropy,
            all_error,
            fractions,
            "all_images_pooled",
            "pixel",
            "predictive_entropy_bits",
        )
    )
    concentration_records.extend(
        concentration_rows(
            all_variance,
            all_error,
            fractions,
            "all_images_pooled",
            "pixel",
            "predictive_variance",
        )
    )
    concentration_records.extend(
        region_concentration_rows(
            local_regions,
            "mean_entropy_bits",
            fractions,
            "all_images_pooled",
            "predictive_entropy_bits",
        )
    )
    concentration_records.extend(
        region_concentration_rows(
            local_regions,
            "mean_predictive_variance",
            fractions,
            "all_images_pooled",
            "predictive_variance",
        )
    )

    regime_summaries: dict[str, object] = {}
    for regime in ("I", "II", "III", "IV"):
        regime_images = per_image[per_image["source_regime"] == regime]
        sample_ids = regime_images["sample_id"].tolist()
        entropy = np.concatenate([arrays_by_sample[value][0] for value in sample_ids])
        variance = np.concatenate([arrays_by_sample[value][1] for value in sample_ids])
        error = np.concatenate([arrays_by_sample[value][2] for value in sample_ids])
        regions = local_regions[local_regions["source_regime"] == regime]
        pooled = pooled_result(entropy, variance, error, regions, fractions)
        regime_summaries[regime] = pooled
        regime_rows.append(
            {
                "source_regime": regime,
                "images": len(regime_images),
                "deterministic_dice_mean": float(regime_images["deterministic_dice"].mean()),
                "deterministic_error_rate": float(error.mean()),
                "entropy_correct_mean": pooled["entropy"]["correct_mean"],
                "entropy_incorrect_mean": pooled["entropy"]["incorrect_mean"],
                "entropy_incorrect_minus_correct": pooled["entropy"]["incorrect_minus_correct"],
                "entropy_incorrect_to_correct_ratio": pooled["entropy"]["incorrect_to_correct_ratio"],
                "entropy_error_pearson": pooled["entropy"]["error_pearson"],
                "entropy_error_roc_auc": pooled["entropy"]["error_roc_auc"],
                "entropy_error_average_precision": pooled["entropy"]["error_average_precision"],
                "entropy_average_precision_lift": pooled["entropy"]["average_precision_lift_over_error_prevalence"],
                "entropy_top_10pct_error_capture": pooled["entropy"]["top_10pct_error_capture"],
                "entropy_top_10pct_error_enrichment": pooled["entropy"]["top_10pct_error_enrichment"],
                "entropy_top_20pct_error_capture": pooled["entropy"]["top_20pct_error_capture"],
                "entropy_top_20pct_error_enrichment": pooled["entropy"]["top_20pct_error_enrichment"],
                "variance_error_pearson": pooled["predictive_variance"]["error_pearson"],
                "variance_error_roc_auc": pooled["predictive_variance"]["error_roc_auc"],
                "region_entropy_error_pearson": pooled["local_regions"]["region_entropy_error_pearson"],
                "region_entropy_error_spearman": pooled["local_regions"]["region_entropy_error_spearman"],
                "region_top_20pct_error_capture": pooled["local_regions"]["top_20pct_regions_error_capture"],
                "region_top_20pct_error_enrichment": pooled["local_regions"]["top_20pct_regions_error_enrichment"],
                "entropy_error_roc_auc_image_mean": float(
                    regime_images["entropy_error_roc_auc"].mean()
                ),
                "entropy_top_20pct_error_capture_image_mean": float(
                    regime_images["entropy_top_20pct_error_capture"].mean()
                ),
                "entropy_top_20pct_error_enrichment_image_mean": float(
                    regime_images["entropy_top_20pct_error_enrichment"].mean()
                ),
                "region_entropy_error_pearson_image_mean": float(
                    regime_images["region_entropy_error_pearson"].mean()
                ),
                "region_top_20pct_error_capture_image_mean": float(
                    regime_images["entropy_top_20pct_regions_error_capture"].mean()
                ),
                "region_top_20pct_error_enrichment_image_mean": float(
                    regime_images["entropy_top_20pct_regions_error_enrichment"].mean()
                ),
                "entropy_incorrect_greater_wilcoxon_p": paired_wilcoxon_greater(
                    regime_images["entropy_incorrect_mean"], regime_images["entropy_correct_mean"]
                ),
            }
        )
        scope = f"regime_{regime}_pooled"
        concentration_records.extend(
            concentration_rows(
                entropy, error, fractions, scope, "pixel", "predictive_entropy_bits"
            )
        )
        concentration_records.extend(
            concentration_rows(
                variance, error, fractions, scope, "pixel", "predictive_variance"
            )
        )
        concentration_records.extend(
            region_concentration_rows(
                regions,
                "mean_entropy_bits",
                fractions,
                scope,
                "predictive_entropy_bits",
            )
        )

    regime_frame = pd.DataFrame(regime_rows)
    concentration_frame = pd.DataFrame(concentration_records)
    representatives = select_representatives(per_image)

    for row in representatives.itertuples(index=False):
        plot_one_example(
            pd.Series(row._asdict()),
            project_root,
            maps_dir,
            config["prediction_pattern"],
            examples_dir / f"{row.sample_id}_uncertainty.png",
        )
    plot_representative_grid(
        representatives,
        project_root,
        maps_dir,
        config["prediction_pattern"],
        figures_dir / "representative_uncertainty.png",
    )
    plot_quantitative_summary(
        per_image,
        regime_frame,
        concentration_frame,
        figures_dir / "uncertainty_error_summary.png",
    )

    per_image.to_csv(metrics_dir / "per_image_uncertainty_metrics.csv", index=False)
    local_regions.to_csv(metrics_dir / "local_region_metrics.csv", index=False)
    regime_frame.to_csv(metrics_dir / "regime_uncertainty_metrics.csv", index=False)
    concentration_frame.to_csv(metrics_dir / "uncertainty_concentration.csv", index=False)
    representatives.to_csv(metrics_dir / "representative_samples.csv", index=False)

    summary: dict[str, object] = {
        "experiment_name": config["experiment_name"],
        "held_out_images": int(len(per_image)),
        "mc_passes": passes,
        "dropout_modules_enabled": sorted(dropout_counts),
        "dropout_probability": 0.1,
        "prediction_under_evaluation": "saved deterministic Step 2 segmentation",
        "ground_truth_use": "evaluation only, after uncertainty generation",
        "local_analysis_grid": {
            "coarse_region_size": region_size,
            "regions_per_image": int(len(local_regions) / len(per_image)),
            "total_regions": int(len(local_regions)),
        },
        "overall": overall,
        "by_source_regime": regime_summaries,
        "runtime": {
            "device": str(device),
            "seconds": time.perf_counter() - start,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
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
    print(json.dumps(summary["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
