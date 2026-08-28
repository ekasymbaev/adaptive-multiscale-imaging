#!/usr/bin/env python3
"""Verify Step 6 masks, reported metrics, tile counts, and seam behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from adaptive_multiscale.training.metrics import binary_segmentation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/adaptive_fusion.json")
    )
    return parser.parse_args()


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    result_root = project_root / config["output_dir"]
    metric_root = result_root / "metrics"
    dataset = config["dataset"]
    example_budget = int(config["visualization"]["example_budget"])
    height = int(dataset["native_height"])
    width = int(dataset["native_width"])
    tile_size = int(dataset["tile_size"])
    manifest = pd.read_csv(project_root / dataset["coarse_manifest"]).set_index(
        "sample_id"
    )
    per_image = pd.read_csv(metric_root / "per_image_policy_metrics.csv")
    rankings = pd.read_csv(project_root / config["selection"]["rankings"])
    heldout_ids = sorted(per_image["sample_id"].unique())
    if len(heldout_ids) != 40:
        raise ValueError(f"Expected 40 held-out images, found {len(heldout_ids)}")
    selected_counts = (
        rankings[
            rankings["sample_id"].isin(heldout_ids)
            & rankings["entropy_rank"].le(example_budget)
        ]
        .groupby("sample_id")
        .size()
    )
    if len(selected_counts) != 40 or not selected_counts.eq(example_budget).all():
        raise ValueError("Entropy ranking did not select the exact example budget")

    seam_half_width = 2
    seam = np.zeros((height, width), dtype=bool)
    for x_coordinate in range(tile_size, width, tile_size):
        seam[:, x_coordinate - seam_half_width : x_coordinate + seam_half_width] = True
    for y_coordinate in range(tile_size, height, tile_size):
        seam[y_coordinate - seam_half_width : y_coordinate + seam_half_width, :] = True

    error_counts = {
        "coarse_only": {"seam": 0, "interior": 0},
        "full_fine": {"seam": 0, "interior": 0},
        f"entropy_feathered_k{example_budget}": {"seam": 0, "interior": 0},
    }
    metric_differences: list[float] = []
    for sample_id in heldout_ids:
        rows = per_image[per_image["sample_id"] == sample_id]
        fold = int(rows["outer_fold"].iloc[0])
        record = manifest.loc[sample_id]
        target = load_grayscale(project_root / record["source_mask_path"]) > 0
        coarse_small = load_grayscale(
            project_root
            / config["models"]["coarse_prediction_pattern"].format(
                fold=fold, sample_id=sample_id
            )
        )
        coarse = cv2.resize(
            coarse_small, (width, height), interpolation=cv2.INTER_NEAREST
        ) > 0
        fine = load_grayscale(
            project_root
            / config["models"]["fine_prediction_pattern"].format(
                fold=fold, sample_id=sample_id
            )
        ) > 0
        adaptive_path = (
            result_root
            / "predicted_masks"
            / "entropy_feathered"
            / f"k{example_budget}"
            / f"fold_{fold}"
            / f"{sample_id}.png"
        )
        adaptive_raw = load_grayscale(adaptive_path)
        if adaptive_raw.shape != (height, width):
            raise ValueError(f"Unexpected adaptive mask shape: {adaptive_path}")
        if not set(np.unique(adaptive_raw)).issubset({0, 255}):
            raise ValueError(f"Adaptive mask is not binary: {adaptive_path}")
        adaptive = adaptive_raw > 0
        predictions = {
            "coarse_only": coarse,
            "full_fine": fine,
            f"entropy_feathered_k{example_budget}": adaptive,
        }
        for policy, prediction in predictions.items():
            error = prediction != target
            error_counts[policy]["seam"] += int(error[seam].sum())
            error_counts[policy]["interior"] += int(error[~seam].sum())
            metrics = binary_segmentation_metrics(prediction, target)
            policy_name = (
                "entropy_feathered"
                if policy.startswith("entropy_feathered")
                else policy
            )
            expected = rows[
                (rows["policy"] == policy_name)
                & (
                    rows["budget_k"].eq(example_budget)
                    if policy_name == "entropy_feathered"
                    else True
                )
            ]["dice"].iloc[0]
            metric_differences.append(abs(float(metrics["dice"]) - float(expected)))

    seam_pixels = int(seam.sum()) * len(heldout_ids)
    interior_pixels = int((~seam).sum()) * len(heldout_ids)
    rows: list[dict[str, object]] = []
    for policy, counts in error_counts.items():
        seam_rate = counts["seam"] / seam_pixels
        interior_rate = counts["interior"] / interior_pixels
        rows.append(
            {
                "policy": policy,
                "images": len(heldout_ids),
                "seam_band_width_pixels": 2 * seam_half_width,
                "seam_error_rate": seam_rate,
                "interior_error_rate": interior_rate,
                "seam_minus_interior_error_rate": seam_rate - interior_rate,
                "seam_to_interior_error_ratio": seam_rate / interior_rate,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(metric_root / "boundary_audit.csv", index=False)
    summary = {
        "status": "passed",
        "adaptive_masks": len(heldout_ids),
        "adaptive_mask_shape": [height, width],
        "adaptive_masks_binary": True,
        "selected_tiles_per_image": example_budget,
        "reported_metric_max_absolute_difference": max(metric_differences),
        "boundary_audit": audit.to_dict(orient="records"),
        "interpretation": (
            "Compare the entropy-feathered seam/interior error ratio with the full-fine "
            "ratio to assess whether Step 6 blending controls the Step 5 tile-boundary penalty."
        ),
    }
    with (metric_root / "adaptive_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
