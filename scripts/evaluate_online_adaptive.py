#!/usr/bin/env python3
"""Run the frozen adaptive pipeline with fine inference restricted to selected tiles."""

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
import torch

from adaptive_multiscale.data.ma_islands import load_grayscale_image
from adaptive_multiscale.fusion import (
    fuse_probability_tiles,
    infer_selected_tile_probabilities,
    timed_call,
    warm_up_segmentation_models,
)
from adaptive_multiscale.models import CompactUNet
from adaptive_multiscale.selection.tile_ranking import (
    TileGrid,
    paired_bootstrap_interval,
    random_rank_matrix,
    uncertainty_tile_rankings,
)
from adaptive_multiscale.training.reproducibility import seed_everything, select_device
from adaptive_multiscale.uncertainty.mc_dropout import mc_dropout_predict


METRICS = ("dice", "iou", "precision", "recall")
POLICY_LABELS = {
    "coarse_only_online": "Coarse only",
    "entropy_online": "Adaptive entropy",
    "random_online": "Adaptive random",
    "full_fine_online": "Full fine",
}
POLICY_COLORS = {
    "coarse_only_online": "#94A3B8",
    "entropy_online": "#7C3AED",
    "random_online": "#64748B",
    "full_fine_online": "#16324F",
}
STAGE_COLUMNS = (
    "coarse_preprocess_seconds",
    "coarse_inference_seconds",
    "coarse_postprocess_seconds",
    "uncertainty_seconds",
    "selection_seconds",
    "fine_inference_seconds",
    "fusion_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/online_adaptive_inference.json")
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


def timed_host_call(function: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter() - start


def repeated_device_call(
    function: Any,
    device: torch.device,
    repeats: int,
    before_each: Any | None = None,
) -> tuple[Any, float, list[float]]:
    if repeats <= 0:
        raise ValueError("Timing repeats must be positive")
    values: Any = None
    durations: list[float] = []
    for _ in range(repeats):
        if before_each is not None:
            before_each()
        values, seconds = timed_call(function, device)
        durations.append(seconds)
    return values, float(np.median(durations)), durations


def repeated_host_call(
    function: Any, repeats: int
) -> tuple[Any, float, list[float]]:
    if repeats <= 0:
        raise ValueError("Timing repeats must be positive")
    values: Any = None
    durations: list[float] = []
    for _ in range(repeats):
        values, seconds = timed_host_call(function)
        durations.append(seconds)
    return values, float(np.median(durations)), durations


def timing_repeat_records(
    record: Any,
    stage: str,
    policy: str,
    budget: int,
    durations: list[float],
) -> list[dict[str, object]]:
    return [
        {
            "outer_fold": int(record.outer_fold),
            "sample_id": record.sample_id,
            "source_regime": record.source_regime,
            "stage": stage,
            "policy": policy,
            "budget_k": int(budget),
            "repeat_index": repeat_index,
            "seconds": float(seconds),
        }
        for repeat_index, seconds in enumerate(durations)
    ]


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


def normalized_tensor(
    image: np.ndarray, mean: float, std: float, device: torch.device
) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError("Normalization standard deviation must be positive")
    values = image.astype(np.float32) / 255.0
    values = (values - mean) / std
    return torch.from_numpy(values[None, None]).to(device)


@torch.inference_mode()
def deterministic_coarse_probability(
    model: torch.nn.Module, image_tensor: torch.Tensor
) -> np.ndarray:
    model.eval()
    return (
        torch.sigmoid(model(image_tensor))[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def confusion_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    tp = float(np.count_nonzero(predicted & truth))
    fp = float(np.count_nonzero(predicted & ~truth))
    fn = float(np.count_nonzero(~predicted & truth))
    tn = float(np.count_nonzero(~predicted & ~truth))
    epsilon = 1e-12
    total = tp + fp + fn + tn
    predicted_fraction = (tp + fp) / total
    target_fraction = (tp + fn) / total
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "dice": (2.0 * tp) / (2.0 * tp + fp + fn + epsilon),
        "iou": tp / (tp + fp + fn + epsilon),
        "precision": tp / (tp + fp + epsilon),
        "recall": tp / (tp + fn + epsilon),
        "predicted_foreground_fraction": predicted_fraction,
        "target_foreground_fraction": target_fraction,
        "area_fraction_error_signed": predicted_fraction - target_fraction,
        "area_fraction_error_absolute": abs(predicted_fraction - target_fraction),
    }


def timing_record(
    coarse_preprocess: float = 0.0,
    coarse_inference: float = 0.0,
    coarse_postprocess: float = 0.0,
    uncertainty: float = 0.0,
    selection: float = 0.0,
    fine_inference: float = 0.0,
    fusion: float = 0.0,
    fine_tiles: int = 0,
    fine_batches: int = 0,
) -> dict[str, float | int]:
    stages = {
        "coarse_preprocess_seconds": coarse_preprocess,
        "coarse_inference_seconds": coarse_inference,
        "coarse_postprocess_seconds": coarse_postprocess,
        "uncertainty_seconds": uncertainty,
        "selection_seconds": selection,
        "fine_inference_seconds": fine_inference,
        "fusion_seconds": fusion,
    }
    return {
        **stages,
        "total_compute_seconds": float(sum(stages.values())),
        "fine_tiles_processed": int(fine_tiles),
        "fine_batches_executed": int(fine_batches),
    }


def bootstrap_mean(
    values: np.ndarray, confidence_level: float, resamples: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float(array.mean()), float("nan"), float("nan")
    return paired_bootstrap_interval(array, confidence_level, resamples, seed)


def summarize(
    per_image: pd.DataFrame,
    confidence_level: float,
    resamples: int,
    seed_base: int,
) -> pd.DataFrame:
    scopes = [("all_images", "all", per_image)]
    scopes.extend(
        (f"regime_{regime}", regime, group)
        for regime, group in per_image.groupby("source_regime", sort=True)
    )
    rows: list[dict[str, object]] = []
    row_index = 0
    for scope, regime, frame in scopes:
        for (policy, budget), group in frame.groupby(
            ["policy", "budget_k"], sort=True
        ):
            group = group.sort_values("sample_id")
            dice_mean, dice_low, dice_high = bootstrap_mean(
                group["dice"].to_numpy(),
                confidence_level,
                resamples,
                seed_base + row_index,
            )
            time_mean, time_low, time_high = bootstrap_mean(
                group["total_compute_seconds"].to_numpy(),
                confidence_level,
                resamples,
                seed_base + 1000 + row_index,
            )
            rows.append(
                {
                    "scope": scope,
                    "source_regime": regime,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "budget_k": int(budget),
                    "images": int(len(group)),
                    "coverage_fraction": float(group["coverage_fraction"].mean()),
                    "macro_dice": dice_mean,
                    "macro_dice_ci_low": dice_low,
                    "macro_dice_ci_high": dice_high,
                    "macro_iou": float(group["iou"].mean()),
                    "macro_precision": float(group["precision"].mean()),
                    "macro_recall": float(group["recall"].mean()),
                    "area_fraction_error_absolute_mean": float(
                        group["area_fraction_error_absolute"].mean()
                    ),
                    "total_compute_seconds_mean": time_mean,
                    "total_compute_seconds_ci_low": time_low,
                    "total_compute_seconds_ci_high": time_high,
                    "total_compute_seconds_median": float(
                        group["total_compute_seconds"].median()
                    ),
                    "total_compute_seconds_p90": float(
                        group["total_compute_seconds"].quantile(0.9)
                    ),
                    "speedup_vs_full_fine_mean": float(
                        group["speedup_vs_full_fine"].mean()
                    ),
                    "compute_fraction_vs_full_fine_mean": float(
                        group["compute_fraction_vs_full_fine"].mean()
                    ),
                    "fine_time_fraction_vs_full_fine_mean": float(
                        group["fine_time_fraction_vs_full_fine"].mean()
                    ),
                    **{
                        f"{column}_mean": float(group[column].mean())
                        for column in STAGE_COLUMNS
                    },
                }
            )
            row_index += 1
    return pd.DataFrame(rows).sort_values(["scope", "policy", "budget_k"])


def paired_comparisons(
    per_image: pd.DataFrame,
    budgets: list[int],
    confidence_level: float,
    resamples: int,
    seed_base: int,
) -> pd.DataFrame:
    specs = [
        ("entropy_online", "random_online", "entropy_vs_random"),
        ("entropy_online", "coarse_only_online", "entropy_vs_coarse"),
        ("entropy_online", "full_fine_online", "entropy_vs_full_fine"),
    ]
    rows: list[dict[str, object]] = []
    row_index = 0
    for left_policy, comparator_policy, name in specs:
        for budget in budgets:
            left = per_image[
                (per_image["policy"] == left_policy)
                & (per_image["budget_k"] == budget)
            ]
            if comparator_policy in {"coarse_only_online", "full_fine_online"}:
                right = per_image[per_image["policy"] == comparator_policy]
            else:
                right = per_image[
                    (per_image["policy"] == comparator_policy)
                    & (per_image["budget_k"] == budget)
                ]
            merged = left.merge(
                right, on="sample_id", suffixes=("_left", "_comparator"), validate="one_to_one"
            )
            for metric in (
                "dice",
                "area_fraction_error_absolute",
                "total_compute_seconds",
                "fine_inference_seconds",
            ):
                if metric in {
                    "area_fraction_error_absolute",
                    "total_compute_seconds",
                    "fine_inference_seconds",
                }:
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
                        "comparison": name,
                        "left_policy": left_policy,
                        "comparator_policy": comparator_policy,
                        "budget_k": budget,
                        "metric": metric,
                        "images": int(len(merged)),
                        "improvement_direction": direction,
                        "improvement_mean": mean,
                        "improvement_ci_low": low,
                        "improvement_ci_high": high,
                        "improved_image_fraction": float(np.mean(differences > 0.0)),
                        "confidence_level": confidence_level,
                    }
                )
                row_index += 1
    return pd.DataFrame(rows)


def verify_reference_metrics(
    per_image: pd.DataFrame,
    step6_policy_metrics: pd.DataFrame,
    step6_random_trials: pd.DataFrame,
    random_trial_index: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in per_image.itertuples(index=False):
        if record.policy == "random_online":
            reference = step6_random_trials[
                (step6_random_trials["sample_id"] == record.sample_id)
                & (step6_random_trials["budget_k"] == record.budget_k)
                & (step6_random_trials["random_trial_index"] == random_trial_index)
            ]
            reference_policy = f"random_trial_{random_trial_index}"
        else:
            mapping = {
                "coarse_only_online": "coarse_only",
                "entropy_online": "entropy_feathered",
                "full_fine_online": "full_fine",
            }
            reference_policy = mapping[record.policy]
            reference = step6_policy_metrics[
                (step6_policy_metrics["sample_id"] == record.sample_id)
                & (step6_policy_metrics["policy"] == reference_policy)
                & (step6_policy_metrics["budget_k"] == record.budget_k)
            ]
        if len(reference) != 1:
            raise ValueError(
                f"Missing Step 6 reference for {record.sample_id} "
                f"{record.policy} K={record.budget_k}"
            )
        reference_row = reference.iloc[0]
        differences = {
            metric: abs(float(getattr(record, metric)) - float(reference_row[metric]))
            for metric in (*METRICS, "area_fraction_error_absolute")
        }
        rows.append(
            {
                "outer_fold": int(record.outer_fold),
                "sample_id": record.sample_id,
                "online_policy": record.policy,
                "reference_policy": reference_policy,
                "budget_k": int(record.budget_k),
                **{f"{metric}_absolute_difference": value for metric, value in differences.items()},
                "maximum_metric_absolute_difference": max(differences.values()),
            }
        )
    return pd.DataFrame(rows)


def plot_performance_runtime(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary[summary["scope"] == "all_images"].copy()
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for policy, group in frame.groupby("policy", sort=False):
        group = group.sort_values("coverage_fraction")
        label = POLICY_LABELS[policy]
        color = POLICY_COLORS[policy]
        axes[0].plot(
            100 * group["coverage_fraction"], group["macro_dice"],
            marker="o", linewidth=2, color=color, label=label,
        )
        axes[1].plot(
            100 * group["coverage_fraction"],
            1000 * group["total_compute_seconds_mean"],
            marker="o", linewidth=2, color=color,
        )
        axes[2].scatter(
            1000 * group["total_compute_seconds_mean"], group["macro_dice"],
            s=80, color=color, label=label,
        )
        for row in group.itertuples(index=False):
            axes[2].annotate(
                f"K={row.budget_k}",
                (1000 * row.total_compute_seconds_mean, row.macro_dice),
                xytext=(4, 4), textcoords="offset points", fontsize=8,
            )
    axes[0].set_xlabel("Native high-resolution coverage (%)")
    axes[0].set_ylabel("Macro Dice")
    axes[0].set_title("Segmentation benefit")
    axes[1].set_xlabel("Native high-resolution coverage (%)")
    axes[1].set_ylabel("Mean synchronized compute time (ms/image)")
    axes[1].set_title("Realized pipeline compute")
    axes[2].set_xlabel("Mean synchronized compute time (ms/image)")
    axes[2].set_ylabel("Macro Dice")
    axes[2].set_title("Performance versus measured compute")
    for axis in axes:
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    figure.suptitle("Online adaptive inference on 40 held-out SEM images", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_timing_breakdown(summary: pd.DataFrame, output_path: Path) -> None:
    frame = summary[summary["scope"] == "all_images"].copy()
    order = [
        ("coarse_only_online", 0),
        ("entropy_online", 12),
        ("entropy_online", 24),
        ("random_online", 12),
        ("random_online", 24),
        ("full_fine_online", 48),
    ]
    indexed = frame.set_index(["policy", "budget_k"])
    frame = pd.DataFrame([indexed.loc[key] for key in order])
    labels = [f"{POLICY_LABELS[policy]}\nK={budget}" for policy, budget in order]
    stages = {
        "Coarse preparation + inference": [
            "coarse_preprocess_seconds_mean",
            "coarse_inference_seconds_mean",
            "coarse_postprocess_seconds_mean",
        ],
        "MC dropout": ["uncertainty_seconds_mean"],
        "Selection": ["selection_seconds_mean"],
        "Selective fine inference": ["fine_inference_seconds_mean"],
        "Fusion / assembly": ["fusion_seconds_mean"],
    }
    colors = ["#94A3B8", "#7C3AED", "#F59E0B", "#0891B2", "#16A34A"]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bottom = np.zeros(len(frame), dtype=np.float64)
    for (label, columns), color in zip(stages.items(), colors, strict=True):
        values = 1000 * frame[columns].sum(axis=1).to_numpy(dtype=np.float64)
        axis.bar(labels, values, bottom=bottom, color=color, label=label)
        bottom += values
    axis.set_ylabel("Mean synchronized compute time (ms/image)")
    axis.set_title("Online pipeline timing breakdown (dataset file I/O excluded)")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_json(config_path)
    output_dir = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else project_root / (args.output_dir or config["output_dir"])
    )
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    for directory in (metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    experiment_start = time.perf_counter()
    seed_everything(int(config["reproducibility"]["random_seed"]))
    device = select_device(str(config["models"]["device"]))
    if device.type == "cpu":
        torch.set_num_threads(int(config["models"]["cpu_threads"]))
    folds = args.folds if args.folds is not None else [0, 1, 2, 3, 4]
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("Folds must be selected from 0 through 4")
    print(f"device={device} folds={folds} mode=online_selective", flush=True)

    dataset = config["dataset"]
    models = config["models"]
    uncertainty_config = config["uncertainty"]
    selection_config = config["selection"]
    fusion_config = config["fusion"]
    tile_size = int(dataset["tile_size"])
    tile_count = int(dataset["tiles_per_image"])
    native_shape = (int(dataset["native_height"]), int(dataset["native_width"]))
    coarse_shape = (int(dataset["coarse_height"]), int(dataset["coarse_width"]))
    threshold = float(models["prediction_threshold"])
    batch_size = int(models["fine_inference_batch_size"])
    blend_width = int(fusion_config["blend_width_pixels"])
    budgets = [int(value) for value in selection_config["budgets"]]
    if budgets != [12, 24]:
        raise ValueError("Step 7 primary budgets must be exactly K=12 and K=24")
    grid = TileGrid(
        native_height=native_shape[0], native_width=native_shape[1],
        coarse_height=coarse_shape[0], coarse_width=coarse_shape[1],
        native_tile_size=tile_size,
    )
    if grid.tile_count != tile_count:
        raise ValueError("Configured tile count does not match the image grid")

    manifest = pd.read_csv(project_root / dataset["coarse_manifest"])
    cv_manifest = pd.read_csv(project_root / dataset["cv_manifest"])
    reference_rankings = pd.read_csv(
        project_root / uncertainty_config["reference_rankings"]
    )
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

    per_image_rows: list[dict[str, object]] = []
    online_ranking_frames: list[pd.DataFrame] = []
    mask_verification_rows: list[dict[str, object]] = []
    timing_repeat_rows: list[dict[str, object]] = []
    mc_passes = int(uncertainty_config["mc_passes"])
    random_trial_index = int(selection_config["random_trial_index"])
    timing_repeats = int(config["timing"]["timing_repeats_per_image"])
    if timing_repeats < 3:
        raise ValueError("Step 7 requires at least three timing repeats per image")

    for fold in folds:
        fold_records = heldout[heldout["outer_fold"] == fold]
        if fold_records.empty:
            continue
        coarse_checkpoint = torch.load(
            project_root / models["coarse_checkpoint_pattern"].format(fold=fold),
            map_location=device,
            weights_only=False,
        )
        fine_checkpoint = torch.load(
            project_root / models["fine_checkpoint_pattern"].format(fold=fold),
            map_location=device,
            weights_only=False,
        )
        coarse_model = build_model(coarse_checkpoint, device)
        fine_model = build_model(fine_checkpoint, device)
        warm_up_segmentation_models(
            coarse_model,
            fine_model,
            coarse_shape,
            tile_size,
            fine_batch_shapes=(8, 12, batch_size),
            device=device,
        )

        for record in fold_records.itertuples(index=False):
            coarse_image, coarse_read_seconds = timed_host_call(
                lambda: load_grayscale(project_root / record.coarse_image_path)
            )
            native_image, native_read_seconds = timed_host_call(
                lambda: load_grayscale_image(project_root / record.source_image_path)[0]
            )
            if coarse_image.shape != coarse_shape or native_image.shape != native_shape:
                raise ValueError(f"Unexpected image shape for {record.sample_id}")

            image_tensor, preprocess_seconds, durations = repeated_device_call(
                lambda: normalized_tensor(
                    coarse_image,
                    float(coarse_checkpoint["normalization_mean"]),
                    float(coarse_checkpoint["normalization_std"]),
                    device,
                ),
                device,
                timing_repeats,
            )
            timing_repeat_rows.extend(
                timing_repeat_records(
                    record, "coarse_preprocess", "shared_coarse", 0, durations
                )
            )

            def run_coarse() -> np.ndarray:
                coarse_model.eval()
                return deterministic_coarse_probability(coarse_model, image_tensor)

            coarse_small, coarse_inference_seconds, durations = repeated_device_call(
                run_coarse,
                device,
                timing_repeats,
            )
            timing_repeat_rows.extend(
                timing_repeat_records(
                    record, "coarse_inference", "shared_coarse", 0, durations
                )
            )
            coarse_probability, postprocess_seconds, durations = repeated_host_call(
                lambda: cv2.resize(
                    coarse_small,
                    (native_shape[1], native_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.float32),
                timing_repeats,
            )
            timing_repeat_rows.extend(
                timing_repeat_records(
                    record, "coarse_postprocess", "shared_coarse", 0, durations
                )
            )

            sample_number = int(str(record.sample_id).split("_")[-1])
            sample_seed = (
                int(uncertainty_config["random_seed_base"])
                + int(fold) * 1000
                + sample_number
            )
            mc_prediction, uncertainty_seconds, durations = repeated_device_call(
                lambda: mc_dropout_predict(coarse_model, image_tensor, passes=mc_passes),
                device,
                timing_repeats,
                before_each=lambda: seed_everything(sample_seed),
            )
            timing_repeat_rows.extend(
                timing_repeat_records(
                    record, "mc_dropout", "entropy_shared", 0, durations
                )
            )
            online_rankings, entropy_selection_seconds, durations = repeated_host_call(
                lambda: uncertainty_tile_rankings(
                    mc_prediction.predictive_entropy_bits,
                    mc_prediction.predictive_variance,
                    grid,
                    percentile=float(uncertainty_config["tile_score_percentile"]),
                ),
                timing_repeats,
            )
            timing_repeat_rows.extend(
                timing_repeat_records(
                    record, "entropy_selection", "entropy_shared", 0, durations
                )
            )
            online_rankings.insert(0, "outer_fold", int(fold))
            online_rankings.insert(1, "sample_id", record.sample_id)
            online_rankings.insert(2, "source_regime", record.source_regime)
            online_ranking_frames.append(online_rankings)

            reference = reference_rankings[
                reference_rankings["sample_id"] == record.sample_id
            ].sort_values("tile_index")
            current = online_rankings.sort_values("tile_index")
            if len(reference) != tile_count:
                raise ValueError(f"Missing reference rankings for {record.sample_id}")
            entropy_rank_mismatches = int(
                np.count_nonzero(
                    current["entropy_rank"].to_numpy()
                    != reference["entropy_rank"].to_numpy()
                )
            )
            variance_rank_mismatches = int(
                np.count_nonzero(
                    current["variance_rank"].to_numpy()
                    != reference["variance_rank"].to_numpy()
                )
            )
            if entropy_rank_mismatches or variance_rank_mismatches:
                raise ValueError(
                    f"Online uncertainty ranking mismatch for {record.sample_id}: "
                    f"entropy={entropy_rank_mismatches}, variance={variance_rank_mismatches}"
                )

            random_ranks, random_selection_seconds, durations = repeated_host_call(
                lambda: random_rank_matrix(
                    tile_count,
                    random_trial_index + 1,
                    int(selection_config["random_seed_base"]),
                    sample_number,
                )[random_trial_index],
                timing_repeats,
            )
            timing_repeat_rows.extend(
                timing_repeat_records(
                    record, "random_selection", "random_shared", 0, durations
                )
            )

            predictions: dict[tuple[str, int], np.ndarray] = {
                ("coarse_only_online", 0): coarse_probability >= threshold
            }
            timings: dict[tuple[str, int], dict[str, float | int]] = {
                ("coarse_only_online", 0): timing_record(
                    preprocess_seconds,
                    coarse_inference_seconds,
                    postprocess_seconds,
                )
            }

            for policy, rank_column, selection_seconds in (
                ("entropy_online", "entropy_rank", entropy_selection_seconds),
                ("random_online", "random_rank", random_selection_seconds),
            ):
                for budget in budgets:
                    if policy == "entropy_online":
                        selected = (
                            current.sort_values(rank_column)
                            .head(budget)["tile_index"]
                            .to_numpy(dtype=np.int64)
                        )
                    else:
                        selected = np.flatnonzero(random_ranks <= budget).astype(np.int64)
                    if len(selected) != budget:
                        raise ValueError(f"{policy} failed to select K={budget}")
                    fine_result, fine_seconds, durations = repeated_device_call(
                        lambda selected=selected: infer_selected_tile_probabilities(
                            fine_model,
                            native_image,
                            selected,
                            float(fine_checkpoint["normalization_mean"]),
                            float(fine_checkpoint["normalization_std"]),
                            tile_size,
                            batch_size,
                            device,
                        ),
                        device,
                        timing_repeats,
                    )
                    timing_repeat_rows.extend(
                        timing_repeat_records(
                            record, "fine_inference", policy, budget, durations
                        )
                    )
                    fused, fusion_seconds, durations = repeated_host_call(
                        lambda fine_result=fine_result: fuse_probability_tiles(
                            coarse_probability,
                            fine_result.probabilities,
                            tile_size,
                            blend_width,
                            rule=str(fusion_config["rule"]),
                        ),
                        timing_repeats,
                    )
                    timing_repeat_rows.extend(
                        timing_repeat_records(
                            record, "fusion", policy, budget, durations
                        )
                    )
                    predictions[(policy, budget)] = fused >= threshold
                    timings[(policy, budget)] = timing_record(
                        preprocess_seconds,
                        coarse_inference_seconds,
                        postprocess_seconds,
                        uncertainty_seconds if policy == "entropy_online" else 0.0,
                        selection_seconds,
                        fine_seconds,
                        fusion_seconds,
                        fine_result.tiles_processed,
                        fine_result.batches_executed,
                    )

            all_tiles = np.arange(tile_count, dtype=np.int64)
            full_result, full_fine_seconds, durations = repeated_device_call(
                lambda: infer_selected_tile_probabilities(
                    fine_model,
                    native_image,
                    all_tiles,
                    float(fine_checkpoint["normalization_mean"]),
                    float(fine_checkpoint["normalization_std"]),
                    tile_size,
                    batch_size,
                    device,
                ),
                device,
                timing_repeats,
            )
            timing_repeat_rows.extend(
                timing_repeat_records(
                    record, "fine_inference", "full_fine_online", tile_count, durations
                )
            )
            full_probability, full_assembly_seconds, durations = repeated_host_call(
                lambda: fuse_probability_tiles(
                    coarse_probability,
                    full_result.probabilities,
                    tile_size,
                    blend_width,
                    rule="hard",
                ),
                timing_repeats,
            )
            timing_repeat_rows.extend(
                timing_repeat_records(
                    record, "fusion", "full_fine_online", tile_count, durations
                )
            )
            predictions[("full_fine_online", tile_count)] = full_probability >= threshold
            timings[("full_fine_online", tile_count)] = timing_record(
                fine_inference=full_fine_seconds,
                fusion=full_assembly_seconds,
                fine_tiles=full_result.tiles_processed,
                fine_batches=full_result.batches_executed,
            )

            saved_coarse = load_grayscale(
                project_root
                / models["coarse_prediction_pattern"].format(
                    fold=fold, sample_id=record.sample_id
                )
            ) > 0
            saved_fine = load_grayscale(
                project_root
                / models["fine_prediction_pattern"].format(
                    fold=fold, sample_id=record.sample_id
                )
            ) > 0
            online_coarse_small = coarse_small >= threshold
            coarse_mismatches = int(np.count_nonzero(online_coarse_small != saved_coarse))
            full_fine_mismatches = int(
                np.count_nonzero(predictions[("full_fine_online", tile_count)] != saved_fine)
            )
            entropy_k12_reference = load_grayscale(
                project_root
                / config["verification"]["step6_entropy_mask_pattern"].format(
                    fold=fold, sample_id=record.sample_id
                )
            ) > 0
            entropy_k12_mismatches = int(
                np.count_nonzero(
                    predictions[("entropy_online", 12)] != entropy_k12_reference
                )
            )
            mask_verification_rows.append(
                {
                    "outer_fold": int(fold),
                    "sample_id": record.sample_id,
                    "entropy_rank_mismatches": entropy_rank_mismatches,
                    "variance_rank_mismatches": variance_rank_mismatches,
                    "coarse_mask_mismatched_pixels": coarse_mismatches,
                    "full_fine_mask_mismatched_pixels": full_fine_mismatches,
                    "entropy_k12_step6_mask_mismatched_pixels": entropy_k12_mismatches,
                }
            )

            # Ground truth is deliberately loaded only after all online policies finish.
            target = load_grayscale(project_root / record.source_mask_path) > 0
            if target.shape != native_shape:
                raise ValueError(f"Unexpected target shape for {record.sample_id}")
            for (policy, budget), prediction in predictions.items():
                timing = timings[(policy, budget)]
                uses_native = policy != "coarse_only_online"
                row = {
                    "outer_fold": int(fold),
                    "sample_id": record.sample_id,
                    "file_name": record.file_name,
                    "source_regime": record.source_regime,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "budget_k": int(budget),
                    "coverage_fraction": budget / tile_count,
                    "ranking_source": (
                        "online_entropy_q90"
                        if policy == "entropy_online"
                        else "online_random_trial_0"
                        if policy == "random_online"
                        else "none"
                    ),
                    "mc_passes": mc_passes if policy == "entropy_online" else 0,
                    "fine_model_executed_only_on_selected_tiles": policy
                    in {"entropy_online", "random_online"},
                    "coarse_image_read_seconds_observed": coarse_read_seconds
                    if policy != "full_fine_online"
                    else 0.0,
                    "native_image_read_seconds_observed": native_read_seconds
                    if uses_native
                    else 0.0,
                    "dataset_io_excluded_from_compute_total": True,
                    **timing,
                    **confusion_metrics(prediction, target),
                }
                per_image_rows.append(row)

            for budget in budgets:
                mask_dir = (
                    output_dir
                    / "predicted_masks"
                    / "entropy_online"
                    / f"k{budget}"
                    / f"fold_{fold}"
                )
                mask_dir.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(mask_dir / f"{record.sample_id}.png"),
                    predictions[("entropy_online", budget)].astype(np.uint8) * 255,
                ):
                    raise OSError(f"Failed to save online mask for {record.sample_id}")
            print(
                f"fold={fold} sample={record.sample_id} ranks=verified "
                f"fine_tiles=12,24,12,24,48",
                flush=True,
            )

        del coarse_model, fine_model, coarse_checkpoint, fine_checkpoint
        if device.type == "mps":
            torch.mps.empty_cache()

    per_image = pd.DataFrame(per_image_rows).sort_values(
        ["outer_fold", "sample_id", "policy", "budget_k"]
    )
    full_times = per_image[per_image["policy"] == "full_fine_online"].set_index(
        "sample_id"
    )
    per_image["full_fine_compute_seconds"] = per_image["sample_id"].map(
        full_times["total_compute_seconds"]
    )
    per_image["full_fine_inference_seconds"] = per_image["sample_id"].map(
        full_times["fine_inference_seconds"]
    )
    per_image["speedup_vs_full_fine"] = (
        per_image["full_fine_compute_seconds"] / per_image["total_compute_seconds"]
    )
    per_image["compute_fraction_vs_full_fine"] = (
        per_image["total_compute_seconds"] / per_image["full_fine_compute_seconds"]
    )
    per_image["fine_time_fraction_vs_full_fine"] = (
        per_image["fine_inference_seconds"] / per_image["full_fine_inference_seconds"]
    )

    statistics = config["statistics"]
    performance = summarize(
        per_image,
        float(statistics["confidence_level"]),
        int(statistics["bootstrap_resamples"]),
        int(statistics["bootstrap_seed"]),
    )
    paired = paired_comparisons(
        per_image,
        budgets,
        float(statistics["confidence_level"]),
        int(statistics["bootstrap_resamples"]),
        int(statistics["bootstrap_seed"]) + 30000,
    )
    step6_policy_metrics = pd.read_csv(
        project_root / config["verification"]["step6_per_image_metrics"]
    )
    step6_random_trials = pd.read_csv(
        project_root / config["verification"]["step6_random_trial_metrics"]
    )
    metric_verification = verify_reference_metrics(
        per_image,
        step6_policy_metrics,
        step6_random_trials,
        random_trial_index,
    )
    mask_verification = pd.DataFrame(mask_verification_rows).sort_values(
        ["outer_fold", "sample_id"]
    )
    online_rankings = pd.concat(online_ranking_frames, ignore_index=True).sort_values(
        ["outer_fold", "sample_id", "tile_index"]
    )
    timing_repeats_frame = pd.DataFrame(timing_repeat_rows).sort_values(
        ["outer_fold", "sample_id", "stage", "policy", "budget_k", "repeat_index"]
    )

    per_image.to_csv(metrics_dir / "per_image_online_metrics.csv", index=False)
    performance.to_csv(metrics_dir / "online_performance_runtime_summary.csv", index=False)
    paired.to_csv(metrics_dir / "online_paired_comparisons.csv", index=False)
    metric_verification.to_csv(metrics_dir / "step6_metric_verification.csv", index=False)
    mask_verification.to_csv(metrics_dir / "online_mask_verification.csv", index=False)
    online_rankings.to_csv(metrics_dir / "online_uncertainty_rankings.csv", index=False)
    timing_repeats_frame.to_csv(metrics_dir / "timing_repeats.csv", index=False)

    plot_performance_runtime(
        performance, figures_dir / "online_performance_runtime.png"
    )
    plot_timing_breakdown(
        performance, figures_dir / "online_timing_breakdown.png"
    )

    overall = performance[performance["scope"] == "all_images"].set_index(
        ["policy", "budget_k"]
    )
    primary_results: dict[str, object] = {}
    for budget in budgets:
        entropy = overall.loc[("entropy_online", budget)]
        random = overall.loc[("random_online", budget)]
        primary_results[f"k{budget}"] = {
            "coverage_fraction": float(entropy["coverage_fraction"]),
            "macro_dice": float(entropy["macro_dice"]),
            "macro_iou": float(entropy["macro_iou"]),
            "area_fraction_error_absolute_mean": float(
                entropy["area_fraction_error_absolute_mean"]
            ),
            "total_compute_seconds_mean": float(
                entropy["total_compute_seconds_mean"]
            ),
            "fine_inference_seconds_mean": float(
                entropy["fine_inference_seconds_mean"]
            ),
            "compute_fraction_vs_full_fine_mean": float(
                entropy["compute_fraction_vs_full_fine_mean"]
            ),
            "fine_time_fraction_vs_full_fine_mean": float(
                entropy["fine_time_fraction_vs_full_fine_mean"]
            ),
            "random_trial_0_macro_dice": float(random["macro_dice"]),
        }
    summary = {
        "experiment_name": config["experiment_name"],
        "completed_folds": sorted(int(value) for value in folds),
        "held_out_images": int(per_image["sample_id"].nunique()),
        "no_training_performed": True,
        "online_execution": {
            "fine_model_receives_only_selected_tiles": True,
            "entropy_budgets": budgets,
            "random_baseline_trial_index": random_trial_index,
            "full_fine_tiles": tile_count,
            "mc_passes": mc_passes,
            "device_synchronized_timing": True,
            "warm_up_excluded": True,
            "timing_repeats_per_image": timing_repeats,
            "per_image_stage_aggregation": "median",
        },
        "verification": {
            "ranking_mismatches": int(
                mask_verification[
                    ["entropy_rank_mismatches", "variance_rank_mismatches"]
                ].to_numpy().sum()
            ),
            "coarse_mask_mismatched_pixels": int(
                mask_verification["coarse_mask_mismatched_pixels"].sum()
            ),
            "full_fine_mask_mismatched_pixels": int(
                mask_verification["full_fine_mask_mismatched_pixels"].sum()
            ),
            "entropy_k12_step6_mask_mismatched_pixels": int(
                mask_verification[
                    "entropy_k12_step6_mask_mismatched_pixels"
                ].sum()
            ),
            "maximum_step6_metric_absolute_difference": float(
                metric_verification["maximum_metric_absolute_difference"].max()
            ),
        },
        "coarse_only": {
            "macro_dice": float(overall.loc[("coarse_only_online", 0), "macro_dice"]),
            "total_compute_seconds_mean": float(
                overall.loc[
                    ("coarse_only_online", 0), "total_compute_seconds_mean"
                ]
            ),
        },
        "full_fine": {
            "macro_dice": float(overall.loc[("full_fine_online", 48), "macro_dice"]),
            "total_compute_seconds_mean": float(
                overall.loc[
                    ("full_fine_online", 48), "total_compute_seconds_mean"
                ]
            ),
        },
        "adaptive_entropy": primary_results,
        "cost_interpretation": config["reproducibility"]["cost_interpretation"],
        "ground_truth_use": config["reproducibility"]["ground_truth_use"],
        "runtime": {
            "experiment_seconds": time.perf_counter() - experiment_start,
            "device": str(device),
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
    print(
        f"online evaluation complete images={summary['held_out_images']} "
        f"runtime_seconds={summary['runtime']['experiment_seconds']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
