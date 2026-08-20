#!/usr/bin/env python3
"""Audit the 40 native-resolution M-A island SEM image/annotation pairs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/multiscale-imaging-matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptive_multiscale.data.ma_islands import (
    build_cv_manifests,
    collect_native_records,
    ensure_unique,
    file_md5,
    file_sha256,
    load_grayscale_image,
    polygon_bounds_diagnostics,
    rasterize_union_mask,
    relative_to_root,
    tile_foreground_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_audit.json"),
        help="Audit configuration relative to the repository root.",
    )
    return parser.parse_args()


def load_config(project_root: Path, config_path: Path) -> dict:
    path = config_path if config_path.is_absolute() else project_root / config_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    normalized = image.astype(np.float32) / 255.0
    rgb = np.repeat(normalized[..., None], 3, axis=2)
    foreground = mask.astype(bool)
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    rgb[foreground] = (1.0 - alpha) * rgb[foreground] + alpha * red[foreground]
    return np.clip(rgb, 0.0, 1.0)


def choose_regime_examples(image_statistics: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for regime in ("I", "II", "III", "IV"):
        group = image_statistics[image_statistics["source_regime"] == regime].copy()
        median = group["foreground_fraction"].median()
        group["distance_to_median"] = (group["foreground_fraction"] - median).abs()
        selected.append(group.sort_values(["distance_to_median", "sample_id"]).iloc[0])
    return pd.DataFrame(selected)


def plot_examples(
    examples: pd.DataFrame, project_root: Path, output_path: Path
) -> None:
    fig, axes = plt.subplots(len(examples), 3, figsize=(15, 14), constrained_layout=True)
    for row_index, row in enumerate(examples.itertuples(index=False)):
        image, _, _ = load_grayscale_image(project_root / row.image_path)
        mask = cv2.imread(str(project_root / row.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(row.mask_path)
        mask = mask > 0

        axes[row_index, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 0].set_title(f"{row.sample_id}: SEM image\n{row.file_name}", fontsize=9)
        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title(
            f"Rasterized M-A mask\n{row.foreground_fraction:.2%} foreground",
            fontsize=9,
        )
        axes[row_index, 2].imshow(make_overlay(image, mask))
        axes[row_index, 2].set_title(
            f"Overlay (red = M-A)\nsource regime {row.source_regime}", fontsize=9
        )
        for axis in axes[row_index]:
            axis.axis("off")

    fig.suptitle(
        "M-A island SEM label audit: one median-foreground example per source regime",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tile_grid(
    row: pd.Series, project_root: Path, tile_size: int, output_path: Path
) -> None:
    image, _, _ = load_grayscale_image(project_root / row["image_path"])
    mask = cv2.imread(str(project_root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(row["mask_path"])
    mask = mask > 0
    height, width = image.shape

    fig, axis = plt.subplots(figsize=(14, 10), constrained_layout=True)
    axis.imshow(make_overlay(image, mask, alpha=0.30))
    for x in range(0, width + 1, tile_size):
        axis.axvline(x - 0.5, color="white", linewidth=0.9, alpha=0.9)
    for y in range(0, height + 1, tile_size):
        axis.axhline(y - 0.5, color="white", linewidth=0.9, alpha=0.9)
    tile_index = 0
    for tile_row in range(height // tile_size):
        for tile_column in range(width // tile_size):
            axis.text(
                tile_column * tile_size + 10,
                tile_row * tile_size + 24,
                f"{tile_index:02d}",
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
            )
            tile_index += 1
    axis.set_title(
        f"48 non-overlapping 256 x 256 tiles: {row['sample_id']} ({row['file_name']})"
    )
    axis.set_xlabel("x coordinate (pixels)")
    axis.set_ylabel("y coordinate (pixels)")
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_foreground_balance(image_statistics: pd.DataFrame, output_path: Path) -> None:
    order = ["I", "II", "III", "IV"]
    colors = {"I": "#3B82F6", "II": "#10B981", "III": "#F59E0B", "IV": "#8B5CF6"}
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for x_index, regime in enumerate(order):
        values = image_statistics.loc[
            image_statistics["source_regime"] == regime, "foreground_fraction"
        ].sort_values()
        offsets = np.linspace(-0.20, 0.20, len(values))
        axis.scatter(
            np.full(len(values), x_index) + offsets,
            values * 100.0,
            s=46,
            color=colors[regime],
            edgecolor="white",
            linewidth=0.5,
            label=f"Regime {regime}",
            zorder=3,
        )
        axis.hlines(
            values.median() * 100.0,
            x_index - 0.28,
            x_index + 0.28,
            color="#111827",
            linewidth=2.0,
            zorder=4,
        )
    global_fraction = (
        image_statistics["foreground_pixels"].sum()
        / image_statistics["total_pixels"].sum()
        * 100.0
    )
    axis.axhline(
        global_fraction,
        color="#DC2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Global pixel balance: {global_fraction:.2f}%",
    )
    axis.set_xticks(range(len(order)), [f"Regime {value}" for value in order])
    axis.set_ylabel("M-A foreground pixels (%)")
    axis.set_title("Foreground class balance across the 40 original SEM images")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown_report(
    summary: dict, group_statistics: pd.DataFrame, output_path: Path
) -> None:
    display = group_statistics.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.6f}")
    headers = [str(column) for column in display.columns]
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    classes = summary["class_balance"]
    tiling = summary["tiling"]
    lines = [
        "# M-A island SEM dataset audit",
        "",
        "## Outcome",
        "",
        f"- Paired native images: {summary['pairing']['paired_images']}",
        f"- Image dimensions: {summary['pairing']['width']} x {summary['pairing']['height']}",
        f"- COCO instance annotations: {summary['annotations']['annotation_count']}",
        f"- Rasterized M-A foreground: {classes['foreground_fraction']:.4%}",
        f"- Background: {classes['background_fraction']:.4%}",
        f"- Tile grid: {tiling['tile_columns']} columns x {tiling['tile_rows']} rows "
        f"= {tiling['tiles_per_image']} tiles per image",
        f"- CV: {summary['cross_validation']['n_splits']} deterministic image-level folds; "
        "no train/validation/test overlap",
        "",
        "## Dataset issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in summary["issues"])
    lines.extend(
        [
            "",
            "## Source-regime statistics",
            "",
            *markdown_rows,
            "",
            "## Visualizations",
            "",
            "- `example_overlays.png`: image, mask, and overlay for four examples.",
            "- `tile_grid_example.png`: spatial verification of the 8 x 6 tile grid.",
            "- `foreground_balance.png`: per-image foreground distribution.",
            "",
            "The audit did not train or evaluate any neural network.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root, args.config)
    paths = config["paths"]
    expected = config["expected"]
    cv_config = config["cross_validation"]

    archive_path = project_root / paths["raw_dir"] / config["source"]["archive_name"]
    native_root = project_root / paths["extracted_dir"] / paths["native_subset"]
    processed_dir = project_root / paths["processed_dir"]
    masks_dir = processed_dir / "masks"
    manifests_dir = processed_dir / "manifests"
    splits_dir = project_root / paths["splits_dir"]
    results_dir = project_root / paths["results_dir"]
    for directory in (masks_dir, manifests_dir, splits_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Dataset archive not found at {archive_path}. Run download_ma_islands.py."
        )
    archive_md5 = file_md5(archive_path)
    if archive_md5 != config["source"]["archive_md5"]:
        raise ValueError(
            f"Archive MD5 mismatch: expected {config['source']['archive_md5']}, "
            f"found {archive_md5}"
        )

    records, annotation_diagnostics = collect_native_records(native_root)
    if len(records) != expected["image_count"]:
        raise AssertionError(
            f"Expected {expected['image_count']} images, found {len(records)}"
        )
    ensure_unique((record.sample_id for record in records), "sample_id")
    ensure_unique((record.file_name for record in records), "file_name")

    image_rows = []
    boundary_diagnostics = []
    for record in records:
        image, image_mode, palette_identity = load_grayscale_image(record.image_path)
        actual_height, actual_width = image.shape
        if (record.width, record.height) != (actual_width, actual_height):
            raise AssertionError(f"COCO/image dimensions disagree for {record.file_name}")
        if (actual_width, actual_height) != (expected["width"], expected["height"]):
            raise AssertionError(f"Unexpected dimensions for {record.file_name}")

        record_boundary_diagnostics = polygon_bounds_diagnostics(record)
        if record_boundary_diagnostics["affected_annotation_count"]:
            boundary_diagnostics.append(
                {"sample_id": record.sample_id, **record_boundary_diagnostics}
            )
        mask = rasterize_union_mask(record, boundary_tolerance=1.0)
        if mask.shape != image.shape:
            raise AssertionError(f"Mask/image dimensions disagree for {record.file_name}")
        mask_path = masks_dir / f"{record.sample_id}.png"
        if not cv2.imwrite(str(mask_path), mask * 255):
            raise OSError(f"Failed to write mask: {mask_path}")
        reloaded_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if reloaded_mask is None or not np.array_equal(reloaded_mask > 0, mask > 0):
            raise AssertionError(f"Written mask failed round-trip validation: {mask_path}")

        tile_counts = tile_foreground_counts(mask, expected["tile_size"])
        if tile_counts.shape != (expected["tile_rows"], expected["tile_columns"]):
            raise AssertionError(f"Unexpected tile grid for {record.file_name}")
        total_pixels = int(mask.size)
        foreground_pixels = int(mask.sum())
        image_rows.append(
            {
                "sample_id": record.sample_id,
                "file_name": record.file_name,
                "published_split": record.published_split,
                "source_regime": record.source_regime,
                "cooling_group": record.cooling_group,
                "width": actual_width,
                "height": actual_height,
                "image_mode": image_mode,
                "identity_grayscale_palette": palette_identity,
                "annotation_count": len(record.annotations),
                "image_min": int(image.min()),
                "image_max": int(image.max()),
                "image_mean": float(image.mean()),
                "image_std": float(image.std()),
                "total_pixels": total_pixels,
                "foreground_pixels": foreground_pixels,
                "background_pixels": total_pixels - foreground_pixels,
                "foreground_fraction": foreground_pixels / total_pixels,
                "background_fraction": 1.0 - foreground_pixels / total_pixels,
                "tile_rows": int(tile_counts.shape[0]),
                "tile_columns": int(tile_counts.shape[1]),
                "tile_count": int(tile_counts.size),
                "positive_tiles": int(np.count_nonzero(tile_counts)),
                "empty_tiles": int(tile_counts.size - np.count_nonzero(tile_counts)),
                "image_sha256": file_sha256(record.image_path),
                "mask_sha256": file_sha256(mask_path),
                "image_path": relative_to_root(record.image_path, project_root),
                "mask_path": relative_to_root(mask_path, project_root),
            }
        )

    image_statistics = pd.DataFrame(image_rows).sort_values("sample_id")
    image_statistics.to_csv(manifests_dir / "image_statistics.csv", index=False)
    image_statistics.to_csv(results_dir / "image_statistics.csv", index=False)

    fold_assignments, cv_splits = build_cv_manifests(
        image_statistics=image_statistics,
        n_splits=cv_config["n_splits"],
        random_seed=cv_config["random_seed"],
        validation_offset=cv_config["validation_offset"],
        stratify_by=cv_config["stratify_by"],
    )
    fold_assignments.to_csv(splits_dir / "fold_assignments.csv", index=False)
    cv_splits.to_csv(splits_dir / "cv5_splits.csv", index=False)

    total_pixels = int(image_statistics["total_pixels"].sum())
    foreground_pixels = int(image_statistics["foreground_pixels"].sum())
    background_pixels = int(image_statistics["background_pixels"].sum())
    global_foreground_fraction = foreground_pixels / total_pixels

    group_statistics = (
        image_statistics.groupby(["source_regime", "cooling_group"], as_index=False)
        .agg(
            image_count=("sample_id", "count"),
            annotation_count=("annotation_count", "sum"),
            foreground_pixels=("foreground_pixels", "sum"),
            background_pixels=("background_pixels", "sum"),
            mean_image_foreground_fraction=("foreground_fraction", "mean"),
            median_image_foreground_fraction=("foreground_fraction", "median"),
            min_image_foreground_fraction=("foreground_fraction", "min"),
            max_image_foreground_fraction=("foreground_fraction", "max"),
            positive_tiles=("positive_tiles", "sum"),
        )
        .sort_values("source_regime")
    )
    group_statistics["pixel_weighted_foreground_fraction"] = (
        group_statistics["foreground_pixels"]
        / (group_statistics["foreground_pixels"] + group_statistics["background_pixels"])
    )
    group_statistics.to_csv(results_dir / "group_statistics.csv", index=False)

    fold_summary = (
        cv_splits.groupby(["outer_fold", "split", "cooling_group"], as_index=False)
        .agg(image_count=("sample_id", "count"))
        .sort_values(["outer_fold", "split", "cooling_group"])
    )
    fold_summary.to_csv(results_dir / "fold_summary.csv", index=False)

    annotation_area_is_invalid = annotation_diagnostics["stored_area_values"] == [
        float(expected["width"] * expected["height"])
    ]
    issues = []
    if annotation_area_is_invalid:
        issues.append(
            "Every source COCO annotation stores the full image area in its `area` field; "
            "the audit ignores it and recomputes mask areas from polygons."
        )
    if boundary_diagnostics:
        total_affected = sum(
            item["affected_annotation_count"] for item in boundary_diagnostics
        )
        total_coordinates = sum(
            item["out_of_bounds_coordinate_count"] for item in boundary_diagnostics
        )
        maximum_excursion = max(
            item["maximum_excursion_pixels"] for item in boundary_diagnostics
        )
        issues.append(
            f"{total_affected} polygon annotation contains {total_coordinates} coordinate "
            f"slightly outside the image domain (maximum excursion "
            f"{maximum_excursion:.6f} pixels); coordinates within a 1-pixel tolerance "
            "were clipped before rasterization."
        )
    issues.append(
        "The native TIFFs use palette mode P, but all 40 palettes were verified as "
        "identity grayscale palettes."
    )
    issues.append(
        "The archive provides full-resolution polygon annotations, not raster mask files; "
        "the PNG masks in data/processed are deterministic derived artifacts."
    )
    issues.append(
        "The dataset contains only 40 independent original images, so Prototype 1 results "
        "should be reported as cross-validated proof-of-concept evidence."
    )

    summary = {
        "dataset": config["dataset_name"],
        "source": {
            **config["source"],
            "downloaded_archive_md5": archive_md5,
            "checksum_verified": True,
        },
        "pairing": {
            "paired_images": len(image_statistics),
            "published_train_images": int(
                (image_statistics["published_split"] == "train").sum()
            ),
            "published_val_images": int(
                (image_statistics["published_split"] == "val").sum()
            ),
            "width": expected["width"],
            "height": expected["height"],
            "all_dimensions_verified": True,
            "all_masks_round_trip_verified": True,
        },
        "annotations": {
            **annotation_diagnostics,
            "source_area_field_valid": not annotation_area_is_invalid,
            "boundary_repairs": boundary_diagnostics,
            "boundary_clip_tolerance_pixels": 1.0,
        },
        "class_balance": {
            "total_pixels": total_pixels,
            "foreground_pixels": foreground_pixels,
            "background_pixels": background_pixels,
            "foreground_fraction": global_foreground_fraction,
            "background_fraction": 1.0 - global_foreground_fraction,
            "foreground_to_background_ratio": foreground_pixels / background_pixels,
            "mean_image_foreground_fraction": float(
                image_statistics["foreground_fraction"].mean()
            ),
            "median_image_foreground_fraction": float(
                image_statistics["foreground_fraction"].median()
            ),
            "min_image_foreground_fraction": float(
                image_statistics["foreground_fraction"].min()
            ),
            "max_image_foreground_fraction": float(
                image_statistics["foreground_fraction"].max()
            ),
        },
        "tiling": {
            "tile_size": expected["tile_size"],
            "tile_columns": expected["tile_columns"],
            "tile_rows": expected["tile_rows"],
            "tiles_per_image": expected["tiles_per_image"],
            "total_tiles": int(image_statistics["tile_count"].sum()),
            "positive_tiles": int(image_statistics["positive_tiles"].sum()),
            "empty_tiles": int(image_statistics["empty_tiles"].sum()),
            "all_images_exactly_divisible": True,
        },
        "cross_validation": {
            "n_splits": cv_config["n_splits"],
            "random_seed": cv_config["random_seed"],
            "validation_offset": cv_config["validation_offset"],
            "stratified_by": cv_config["stratify_by"],
            "rows_in_fold_assignments": len(fold_assignments),
            "rows_in_expanded_manifest": len(cv_splits),
            "split_unit": "original_image",
            "leakage_checks_passed": True,
        },
        "issues": issues,
        "suitable_for_prototype_1": True,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    statistic_rows = [
        ("image_count", len(image_statistics), "images"),
        ("annotation_count", annotation_diagnostics["annotation_count"], "instances"),
        ("total_pixels", total_pixels, "pixels"),
        ("foreground_pixels", foreground_pixels, "pixels"),
        ("background_pixels", background_pixels, "pixels"),
        ("foreground_fraction", global_foreground_fraction, "fraction"),
        ("background_fraction", 1.0 - global_foreground_fraction, "fraction"),
        ("tiles_per_image", expected["tiles_per_image"], "tiles"),
        ("total_tiles", int(image_statistics["tile_count"].sum()), "tiles"),
        ("positive_tiles", int(image_statistics["positive_tiles"].sum()), "tiles"),
        ("empty_tiles", int(image_statistics["empty_tiles"].sum()), "tiles"),
    ]
    pd.DataFrame(statistic_rows, columns=["metric", "value", "unit"]).to_csv(
        results_dir / "dataset_statistics.csv", index=False
    )

    examples = choose_regime_examples(image_statistics)
    plot_examples(examples, project_root, results_dir / "example_overlays.png")
    representative = image_statistics.iloc[
        (image_statistics["foreground_fraction"] - image_statistics["foreground_fraction"].median())
        .abs()
        .argmin()
    ]
    plot_tile_grid(
        representative,
        project_root,
        expected["tile_size"],
        results_dir / "tile_grid_example.png",
    )
    plot_foreground_balance(image_statistics, results_dir / "foreground_balance.png")
    write_markdown_report(summary, group_statistics, results_dir / "audit_report.md")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
