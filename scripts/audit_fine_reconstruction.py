#!/usr/bin/env python3
"""Verify Step 5 outputs and quantify non-overlapping tile-boundary error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fine_model.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    result_root = project_root / config["output_dir"]
    metric_root = result_root / "metrics"
    comparison = pd.read_csv(metric_root / "fine_vs_coarse_per_image.csv")
    image_manifest = pd.read_csv(
        project_root / config["dataset"]["coarse_manifest"]
    ).set_index("sample_id")
    tile_manifest = pd.read_csv(metric_root / "native_tile_manifest.csv")
    fold_tiles = pd.read_csv(metric_root / "fold_tile_manifest.csv")

    height = int(config["dataset"]["native_height"])
    width = int(config["dataset"]["native_width"])
    tile_size = int(config["dataset"]["tile_size"])
    tiles_per_image = int(config["dataset"]["tiles_per_image"])
    expected_images = len(image_manifest)
    expected_tiles = expected_images * tiles_per_image

    if len(comparison) != expected_images:
        raise ValueError(f"Expected {expected_images} comparison rows, found {len(comparison)}")
    if len(tile_manifest) != expected_tiles:
        raise ValueError(f"Expected {expected_tiles} native tiles, found {len(tile_manifest)}")
    if not tile_manifest.groupby("sample_id").size().eq(tiles_per_image).all():
        raise ValueError("At least one native image does not have exactly 48 tiles")
    if not fold_tiles.groupby(["outer_fold", "split", "sample_id"]).size().eq(
        tiles_per_image
    ).all():
        raise ValueError("At least one fold/split image does not have exactly 48 tiles")

    for fold, frame in fold_tiles.groupby("outer_fold"):
        split_ids = {
            split: set(group["sample_id"])
            for split, group in frame.groupby("split")
        }
        if (
            split_ids["train"] & split_ids["validation"]
            or split_ids["train"] & split_ids["test"]
            or split_ids["validation"] & split_ids["test"]
        ):
            raise ValueError(f"Image leakage detected in fold {fold}")

    best_checkpoints = sorted((result_root / "checkpoints").glob("fold_*_best.pt"))
    last_checkpoints = sorted((result_root / "checkpoints").glob("fold_*_last.pt"))
    if len(best_checkpoints) != 5 or len(last_checkpoints) != 5:
        raise ValueError("Expected five best and five last checkpoints")

    seam_half_width = 2
    seam = np.zeros((height, width), dtype=bool)
    for x_coordinate in range(tile_size, width, tile_size):
        seam[:, x_coordinate - seam_half_width : x_coordinate + seam_half_width] = True
    for y_coordinate in range(tile_size, height, tile_size):
        seam[y_coordinate - seam_half_width : y_coordinate + seam_half_width, :] = True

    error_counts = {
        "fine": {"seam": 0, "interior": 0},
        "coarse": {"seam": 0, "interior": 0},
    }
    prediction_files: list[Path] = []
    for row in comparison.itertuples(index=False):
        target = cv2.imread(
            str(project_root / image_manifest.loc[row.sample_id, "source_mask_path"]),
            cv2.IMREAD_GRAYSCALE,
        )
        fine_path = (
            result_root
            / "predicted_masks"
            / f"fold_{int(row.outer_fold)}"
            / f"{row.sample_id}.png"
        )
        fine = cv2.imread(str(fine_path), cv2.IMREAD_GRAYSCALE)
        coarse_small = cv2.imread(
            str(
                project_root
                / config["coarse_comparison"]["prediction_pattern"].format(
                    fold=int(row.outer_fold), sample_id=row.sample_id
                )
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        if target is None or fine is None or coarse_small is None:
            raise FileNotFoundError(f"Missing target or prediction for {row.sample_id}")
        if target.shape != (height, width) or fine.shape != (height, width):
            raise ValueError(f"Unexpected native shape for {row.sample_id}")
        if not set(np.unique(fine)).issubset({0, 255}):
            raise ValueError(f"Fine prediction is not binary: {fine_path}")
        prediction_files.append(fine_path)

        target_binary = target > 0
        fine_binary = fine > 0
        coarse_binary = cv2.resize(
            coarse_small,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        for model_name, prediction in (
            ("fine", fine_binary),
            ("coarse", coarse_binary),
        ):
            error = prediction != target_binary
            error_counts[model_name]["seam"] += int(error[seam].sum())
            error_counts[model_name]["interior"] += int(error[~seam].sum())

    seam_pixels = int(seam.sum()) * expected_images
    interior_pixels = int((~seam).sum()) * expected_images
    rows: list[dict[str, object]] = []
    for model_name in ("coarse", "fine"):
        seam_rate = error_counts[model_name]["seam"] / seam_pixels
        interior_rate = error_counts[model_name]["interior"] / interior_pixels
        rows.append(
            {
                "model": model_name,
                "images": expected_images,
                "seam_band_width_pixels": 2 * seam_half_width,
                "seam_error_rate": seam_rate,
                "interior_error_rate": interior_rate,
                "seam_minus_interior_error_rate": seam_rate - interior_rate,
                "seam_to_interior_error_ratio": seam_rate / interior_rate,
            }
        )
    audit_frame = pd.DataFrame(rows)
    audit_frame.to_csv(metric_root / "reconstruction_audit.csv", index=False)

    summary = {
        "status": "passed",
        "native_prediction_files": len(prediction_files),
        "native_prediction_shape": [height, width],
        "native_predictions_binary": True,
        "best_checkpoints": len(best_checkpoints),
        "last_checkpoints": len(last_checkpoints),
        "native_tile_rows": len(tile_manifest),
        "fold_tile_rows": len(fold_tiles),
        "tiles_per_image": tiles_per_image,
        "image_level_split_disjointness": True,
        "boundary_audit": audit_frame.to_dict(orient="records"),
        "interpretation": (
            "The fine model has a tile-boundary penalty if its seam-to-interior "
            "error ratio exceeds the coarse control ratio; consider contextual or "
            "overlapping inference before a production adaptive-fusion experiment."
        ),
    }
    with (metric_root / "reconstruction_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
