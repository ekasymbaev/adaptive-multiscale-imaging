#!/usr/bin/env python3
"""Independently audit Step 7 online masks, metrics, timing, and tile execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


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
    parser.add_argument("--output-dir", type=Path, help="Optional output override")
    return parser.parse_args()


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    tp = float(np.count_nonzero(predicted & truth))
    fp = float(np.count_nonzero(predicted & ~truth))
    fn = float(np.count_nonzero(~predicted & truth))
    epsilon = 1e-12
    total = float(predicted.size)
    predicted_fraction = (tp + fp) / total
    target_fraction = (tp + fn) / total
    return {
        "dice": (2.0 * tp) / (2.0 * tp + fp + fn + epsilon),
        "iou": tp / (tp + fp + fn + epsilon),
        "precision": tp / (tp + fp + epsilon),
        "recall": tp / (tp + fn + epsilon),
        "area_fraction_error_absolute": abs(predicted_fraction - target_fraction),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    output_dir = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else project_root / (args.output_dir or config["output_dir"])
    )
    metrics_dir = output_dir / "metrics"
    per_image = pd.read_csv(metrics_dir / "per_image_online_metrics.csv")
    verification = pd.read_csv(metrics_dir / "online_mask_verification.csv")
    metric_verification = pd.read_csv(metrics_dir / "step6_metric_verification.csv")
    rankings = pd.read_csv(metrics_dir / "online_uncertainty_rankings.csv")
    timing_repeats = pd.read_csv(metrics_dir / "timing_repeats.csv")
    manifest = pd.read_csv(project_root / config["dataset"]["coarse_manifest"])

    expected_images = 40
    expected_policies = {
        ("coarse_only_online", 0): 0,
        ("entropy_online", 12): 12,
        ("entropy_online", 24): 24,
        ("random_online", 12): 12,
        ("random_online", 24): 24,
        ("full_fine_online", 48): 48,
    }
    if per_image["sample_id"].nunique() != expected_images or len(per_image) != 240:
        raise ValueError("Expected six online policy rows for each of 40 images")
    observed = set(zip(per_image["policy"], per_image["budget_k"], strict=True))
    if observed != set(expected_policies):
        raise ValueError(f"Unexpected policy/budget combinations: {sorted(observed)}")
    for (policy, budget), expected_tiles in expected_policies.items():
        frame = per_image[
            (per_image["policy"] == policy) & (per_image["budget_k"] == budget)
        ]
        if len(frame) != expected_images or not frame["fine_tiles_processed"].eq(
            expected_tiles
        ).all():
            raise ValueError(f"Incorrect executed tile count for {policy} K={budget}")

    stage_values = per_image[list(STAGE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(stage_values).all() or np.any(stage_values < 0.0):
        raise ValueError("Stage timings must be finite and nonnegative")
    timing_difference = np.abs(
        stage_values.sum(axis=1)
        - per_image["total_compute_seconds"].to_numpy(dtype=np.float64)
    )
    timing_max_difference = float(timing_difference.max())
    if timing_max_difference > 1e-12:
        raise ValueError("Total compute time does not equal the recorded stage sum")
    configured_repeats = int(config["timing"]["timing_repeats_per_image"])
    repeat_group_columns = ["sample_id", "stage", "policy", "budget_k"]
    repeat_counts = timing_repeats.groupby(repeat_group_columns).size()
    if (
        configured_repeats < 3
        or len(timing_repeats) != 1920
        or len(repeat_counts) != 640
        or not repeat_counts.eq(configured_repeats).all()
    ):
        raise ValueError("Expected three raw timing repeats for all 640 stage groups")
    raw_seconds = timing_repeats["seconds"].to_numpy(dtype=np.float64)
    if not np.isfinite(raw_seconds).all() or np.any(raw_seconds < 0.0):
        raise ValueError("Raw timing repeats must be finite and nonnegative")

    mismatch_columns = [
        "entropy_rank_mismatches",
        "variance_rank_mismatches",
        "coarse_mask_mismatched_pixels",
        "full_fine_mask_mismatched_pixels",
        "entropy_k12_step6_mask_mismatched_pixels",
    ]
    mismatch_total = int(verification[mismatch_columns].to_numpy().sum())
    if mismatch_total:
        raise ValueError(f"Online reference verification found {mismatch_total} mismatches")
    reference_metric_max = float(
        metric_verification["maximum_metric_absolute_difference"].max()
    )
    if reference_metric_max > 1e-12:
        raise ValueError("Online metrics do not reproduce the Step 6 references")
    ranking_counts = rankings.groupby("sample_id").size()
    if len(ranking_counts) != expected_images or not ranking_counts.eq(48).all():
        raise ValueError("Expected exactly 48 regenerated ranking rows per image")

    manifest_index = manifest.set_index("sample_id")
    mask_rows: list[dict[str, object]] = []
    metric_names = ("dice", "iou", "precision", "recall", "area_fraction_error_absolute")
    for budget in (12, 24):
        frame = per_image[
            (per_image["policy"] == "entropy_online")
            & (per_image["budget_k"] == budget)
        ].set_index("sample_id")
        for sample_id, row in frame.iterrows():
            fold = int(row["outer_fold"])
            path = (
                output_dir
                / "predicted_masks"
                / "entropy_online"
                / f"k{budget}"
                / f"fold_{fold}"
                / f"{sample_id}.png"
            )
            mask = load_grayscale(path)
            target = load_grayscale(
                project_root / manifest_index.loc[sample_id, "source_mask_path"]
            )
            if mask.shape != (1536, 2048) or target.shape != mask.shape:
                raise ValueError(f"Invalid saved mask geometry for {sample_id} K={budget}")
            unique = set(np.unique(mask).tolist())
            if not unique.issubset({0, 255}):
                raise ValueError(f"Non-binary saved mask for {sample_id} K={budget}")
            recomputed = mask_metrics(mask > 0, target > 0)
            differences = {
                name: abs(float(recomputed[name]) - float(row[name]))
                for name in metric_names
            }
            mask_rows.append(
                {
                    "outer_fold": fold,
                    "sample_id": sample_id,
                    "budget_k": budget,
                    "height": mask.shape[0],
                    "width": mask.shape[1],
                    "binary_values": ",".join(str(value) for value in sorted(unique)),
                    "maximum_metric_absolute_difference": max(differences.values()),
                    "status": "passed",
                }
            )
    saved_mask_audit = pd.DataFrame(mask_rows)
    saved_mask_metric_max = float(
        saved_mask_audit["maximum_metric_absolute_difference"].max()
    )
    if len(saved_mask_audit) != 80 or saved_mask_metric_max > 1e-12:
        raise ValueError("Saved online mask audit failed")

    overall = per_image.groupby(["policy", "budget_k"], sort=True)[
        "total_compute_seconds"
    ].mean()
    full_seconds = float(overall.loc[("full_fine_online", 48)])
    k12_seconds = float(overall.loc[("entropy_online", 12)])
    k24_seconds = float(overall.loc[("entropy_online", 24)])
    checks = pd.DataFrame(
        [
            {"check": "images", "observed": expected_images, "expected": expected_images, "status": "passed"},
            {"check": "policy_rows", "observed": len(per_image), "expected": 240, "status": "passed"},
            {"check": "regenerated_ranking_mismatches", "observed": mismatch_total, "expected": 0, "status": "passed"},
            {"check": "maximum_step6_metric_difference", "observed": reference_metric_max, "expected": "<=1e-12", "status": "passed"},
            {"check": "saved_adaptive_masks", "observed": len(saved_mask_audit), "expected": 80, "status": "passed"},
            {"check": "maximum_saved_mask_metric_difference", "observed": saved_mask_metric_max, "expected": "<=1e-12", "status": "passed"},
            {"check": "maximum_timing_sum_difference", "observed": timing_max_difference, "expected": "<=1e-12", "status": "passed"},
            {"check": "raw_timing_repeat_rows", "observed": len(timing_repeats), "expected": 1920, "status": "passed"},
            {"check": "timing_repeats_per_stage_group", "observed": configured_repeats, "expected": 3, "status": "passed"},
        ]
    )
    checks.to_csv(metrics_dir / "online_audit_checks.csv", index=False)
    saved_mask_audit.to_csv(metrics_dir / "saved_mask_audit.csv", index=False)
    audit = {
        "status": "passed",
        "held_out_images": expected_images,
        "policy_rows": int(len(per_image)),
        "saved_adaptive_masks": int(len(saved_mask_audit)),
        "ranking_and_reference_mask_mismatches": mismatch_total,
        "maximum_step6_metric_absolute_difference": reference_metric_max,
        "maximum_saved_mask_metric_absolute_difference": saved_mask_metric_max,
        "maximum_timing_sum_absolute_difference_seconds": timing_max_difference,
        "raw_timing_repeat_rows": int(len(timing_repeats)),
        "timing_repeats_per_stage_group": configured_repeats,
        "mean_compute_seconds": {
            "full_fine": full_seconds,
            "entropy_k12": k12_seconds,
            "entropy_k24": k24_seconds,
        },
        "entropy_compute_difference_from_full_fine_seconds": {
            "k12": k12_seconds - full_seconds,
            "k24": k24_seconds - full_seconds,
        },
        "fine_model_executed_only_on_selected_tiles": True,
        "native_file_io_limitation": config["reproducibility"]["cost_interpretation"],
    }
    with (metrics_dir / "online_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(
        f"online audit passed images={expected_images} masks={len(saved_mask_audit)} "
        f"reference_metric_max={reference_metric_max:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
