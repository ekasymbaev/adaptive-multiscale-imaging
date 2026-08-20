"""Utilities for auditing the full-resolution M-A island SEM dataset.

The published full-resolution labels are COCO polygon annotations. This module
validates the image/annotation pairing, rasterizes union masks, verifies the
256 x 256 tile geometry, and creates deterministic image-level CV splits.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class MAImageRecord:
    """One original SEM image and its associated COCO annotations."""

    sample_id: str
    file_name: str
    image_path: Path
    published_split: str
    coco_image_id: int
    width: int
    height: int
    source_regime: str
    cooling_group: str
    annotations: tuple[dict[str, Any], ...]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_source_regime(file_name: str) -> str:
    """Recover the four source regimes encoded in the published file names."""

    if file_name.startswith("10_400"):
        return "I"
    if file_name.startswith("10_500"):
        return "II"
    if file_name.startswith("10-500-03"):
        return "III"
    if file_name.startswith("10-400-01"):
        return "IV"
    raise ValueError(f"Unrecognized source regime in filename: {file_name}")


def cooling_group_for_regime(source_regime: str) -> str:
    """Map source regimes to the three published modeling datasets."""

    mapping = {"I": "I+IV", "IV": "I+IV", "II": "II", "III": "III"}
    try:
        return mapping[source_regime]
    except KeyError as exc:
        raise ValueError(f"Unexpected source regime: {source_regime}") from exc


def _validate_categories(payload: dict[str, Any], annotation_path: Path) -> None:
    categories = payload.get("categories")
    expected = [{"supercategory": "MA", "id": 1, "name": "MA"}]
    if categories != expected:
        raise ValueError(
            f"Unexpected categories in {annotation_path}: {categories!r}"
        )


def collect_native_records(
    native_root: Path,
) -> tuple[list[MAImageRecord], dict[str, Any]]:
    """Pair all native images with annotations and validate COCO structure."""

    pending: list[dict[str, Any]] = []
    all_file_names: set[str] = set()
    stored_areas: set[float] = set()
    annotation_count = 0
    polygon_count = 0
    min_vertices: int | None = None
    max_vertices = 0

    for published_split, image_dir_name, label_name in (
        ("train", "im_train", "train.json"),
        ("val", "im_val", "val.json"),
    ):
        image_dir = native_root / image_dir_name
        annotation_path = native_root / "label" / label_name
        if not image_dir.is_dir() or not annotation_path.is_file():
            raise FileNotFoundError(
                f"Missing native image directory or annotation file for {published_split}"
            )

        payload = read_json(annotation_path)
        _validate_categories(payload, annotation_path)

        images_by_id: dict[int, dict[str, Any]] = {}
        for image_meta in payload.get("images", []):
            image_id = int(image_meta["id"])
            if image_id in images_by_id:
                raise ValueError(f"Duplicate COCO image id {image_id} in {annotation_path}")
            images_by_id[image_id] = image_meta

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id not in images_by_id:
                raise ValueError(
                    f"Annotation refers to unknown image id {image_id} in {annotation_path}"
                )
            if int(annotation.get("category_id", -1)) != 1:
                raise ValueError(f"Unexpected category in annotation {annotation.get('id')}")
            if int(annotation.get("iscrowd", 0)) != 0:
                raise ValueError(f"Crowd annotation is unsupported: {annotation.get('id')}")

            segmentations = annotation.get("segmentation")
            if not isinstance(segmentations, list) or not segmentations:
                raise ValueError(f"Missing polygon segmentation: {annotation.get('id')}")
            for flat_polygon in segmentations:
                if len(flat_polygon) < 6 or len(flat_polygon) % 2:
                    raise ValueError(
                        f"Invalid polygon coordinate list: {annotation.get('id')}"
                    )
                vertices = len(flat_polygon) // 2
                min_vertices = vertices if min_vertices is None else min(min_vertices, vertices)
                max_vertices = max(max_vertices, vertices)
                polygon_count += 1

            annotations_by_image[image_id].append(annotation)
            stored_areas.add(float(annotation.get("area", np.nan)))
            annotation_count += 1

        declared_names = {str(meta["file_name"]) for meta in images_by_id.values()}
        physical_names = {path.name for path in image_dir.iterdir() if path.is_file()}
        if declared_names != physical_names:
            missing = sorted(declared_names - physical_names)
            extra = sorted(physical_names - declared_names)
            raise ValueError(
                f"Image/COCO filename mismatch in {published_split}; "
                f"missing={missing}, extra={extra}"
            )

        overlap = all_file_names & declared_names
        if overlap:
            raise ValueError(f"Duplicate filenames across published splits: {sorted(overlap)}")
        all_file_names.update(declared_names)

        for image_id, image_meta in images_by_id.items():
            file_name = str(image_meta["file_name"])
            source_regime = infer_source_regime(file_name)
            pending.append(
                {
                    "file_name": file_name,
                    "image_path": image_dir / file_name,
                    "published_split": published_split,
                    "coco_image_id": image_id,
                    "width": int(image_meta["width"]),
                    "height": int(image_meta["height"]),
                    "source_regime": source_regime,
                    "cooling_group": cooling_group_for_regime(source_regime),
                    "annotations": tuple(annotations_by_image.get(image_id, [])),
                }
            )

    records = [
        MAImageRecord(sample_id=f"ma_{index:03d}", **item)
        for index, item in enumerate(
            sorted(pending, key=lambda item: item["file_name"]), start=1
        )
    ]
    diagnostics = {
        "image_count": len(records),
        "annotation_count": annotation_count,
        "polygon_count": polygon_count,
        "min_polygon_vertices": min_vertices,
        "max_polygon_vertices": max_vertices,
        "stored_area_values": sorted(stored_areas),
        "category": "MA",
    }
    return records, diagnostics


def load_grayscale_image(path: Path) -> tuple[np.ndarray, str, bool]:
    """Load a palette TIFF while verifying whether its palette is grayscale."""

    with Image.open(path) as image:
        mode = image.mode
        palette = image.getpalette()
        palette_is_identity_grayscale = bool(
            mode == "P"
            and palette is not None
            and len(palette) >= 768
            and all(palette[3 * i : 3 * i + 3] == [i, i, i] for i in range(256))
        )
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    return grayscale, mode, palette_is_identity_grayscale


def polygon_bounds_diagnostics(record: MAImageRecord) -> dict[str, Any]:
    """Describe polygon vertices outside the declared pixel-center domain."""

    affected_annotations: set[int] = set()
    out_of_bounds_coordinates = 0
    maximum_excursion = 0.0
    for annotation in record.annotations:
        for flat_polygon in annotation["segmentation"]:
            coordinates = np.asarray(flat_polygon, dtype=np.float64).reshape(-1, 2)
            x, y = coordinates[:, 0], coordinates[:, 1]
            excursions = np.maximum.reduce(
                [
                    np.maximum(-x, 0.0),
                    np.maximum(x - (record.width - 1), 0.0),
                    np.maximum(-y, 0.0),
                    np.maximum(y - (record.height - 1), 0.0),
                ]
            )
            count = int(np.count_nonzero(excursions))
            if count:
                affected_annotations.add(int(annotation["id"]))
                out_of_bounds_coordinates += count
                maximum_excursion = max(maximum_excursion, float(excursions.max()))
    return {
        "affected_annotation_ids": sorted(affected_annotations),
        "affected_annotation_count": len(affected_annotations),
        "out_of_bounds_coordinate_count": out_of_bounds_coordinates,
        "maximum_excursion_pixels": maximum_excursion,
    }


def rasterize_union_mask(
    record: MAImageRecord, boundary_tolerance: float = 1.0
) -> np.ndarray:
    """Rasterize all instance polygons into one binary semantic mask."""

    mask = np.zeros((record.height, record.width), dtype=np.uint8)
    for annotation in record.annotations:
        for flat_polygon in annotation["segmentation"]:
            coordinates = np.asarray(flat_polygon, dtype=np.float64).reshape(-1, 2)
            if not np.isfinite(coordinates).all():
                raise ValueError(
                    f"Non-finite coordinates in annotation {annotation.get('id')}"
                )
            x, y = coordinates[:, 0], coordinates[:, 1]
            excursion = max(
                float(max(-x.min(), 0.0)),
                float(max(x.max() - (record.width - 1), 0.0)),
                float(max(-y.min(), 0.0)),
                float(max(y.max() - (record.height - 1), 0.0)),
            )
            if excursion > boundary_tolerance:
                raise ValueError(
                    f"Out-of-bounds polygon in {record.file_name}, "
                    f"annotation {annotation.get('id')}: {excursion:.3f} pixels"
                )
            coordinates[:, 0] = np.clip(coordinates[:, 0], 0, record.width - 1)
            coordinates[:, 1] = np.clip(coordinates[:, 1], 0, record.height - 1)
            points = np.rint(coordinates).astype(np.int32)
            cv2.fillPoly(mask, [points], color=1)
    return mask


def tile_foreground_counts(mask: np.ndarray, tile_size: int) -> np.ndarray:
    """Return foreground pixel counts for a non-overlapping spatial grid."""

    height, width = mask.shape
    if height % tile_size or width % tile_size:
        raise ValueError(
            f"Image shape {(height, width)} is not divisible by tile size {tile_size}"
        )
    rows = height // tile_size
    columns = width // tile_size
    tiled = mask.reshape(rows, tile_size, columns, tile_size).transpose(0, 2, 1, 3)
    return tiled.sum(axis=(2, 3), dtype=np.int64)


def build_cv_manifests(
    image_statistics: pd.DataFrame,
    n_splits: int,
    random_seed: int,
    validation_offset: int = 1,
    stratify_by: str = "source_regime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic image-level fold assignments and expanded splits."""

    if validation_offset % n_splits == 0:
        raise ValueError("Validation fold must differ from the test fold")
    required = {
        "sample_id",
        "file_name",
        "cooling_group",
        "source_regime",
        stratify_by,
    }
    missing = required - set(image_statistics.columns)
    if missing:
        raise ValueError(f"Image statistics are missing columns: {sorted(missing)}")

    ordered = image_statistics.sort_values("sample_id").reset_index(drop=True).copy()
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_seed
    )
    assignments = np.full(len(ordered), -1, dtype=np.int64)
    labels = ordered[stratify_by].to_numpy()
    for fold, (_, test_indices) in enumerate(splitter.split(ordered, labels)):
        assignments[test_indices] = fold
    if np.any(assignments < 0):
        raise AssertionError("Not every image received a CV fold")

    fold_assignments = ordered[
        ["sample_id", "file_name", "source_regime", "cooling_group"]
    ].copy()
    fold_assignments["cv_fold"] = assignments
    fold_assignments["split_unit"] = "original_image"
    fold_assignments["random_seed"] = random_seed

    expanded_rows: list[dict[str, Any]] = []
    for outer_fold in range(n_splits):
        validation_fold = (outer_fold + validation_offset) % n_splits
        for row in fold_assignments.itertuples(index=False):
            if row.cv_fold == outer_fold:
                split = "test"
            elif row.cv_fold == validation_fold:
                split = "validation"
            else:
                split = "train"
            expanded_rows.append(
                {
                    "outer_fold": outer_fold,
                    "sample_id": row.sample_id,
                    "file_name": row.file_name,
                    "source_regime": row.source_regime,
                    "cooling_group": row.cooling_group,
                    "assigned_cv_fold": row.cv_fold,
                    "split": split,
                    "split_unit": "original_image",
                    "random_seed": random_seed,
                }
            )
    cv_splits = pd.DataFrame(expanded_rows)
    validate_cv_manifests(fold_assignments, cv_splits, n_splits)
    return fold_assignments, cv_splits


def validate_cv_manifests(
    fold_assignments: pd.DataFrame,
    cv_splits: pd.DataFrame,
    n_splits: int,
) -> None:
    """Assert that every fold is image-disjoint and covers all source images."""

    expected_ids = set(fold_assignments["sample_id"])
    if fold_assignments["sample_id"].duplicated().any():
        raise AssertionError("Duplicate original images in fold assignments")
    if set(fold_assignments["cv_fold"]) != set(range(n_splits)):
        raise AssertionError("Fold assignments do not cover every fold")

    for outer_fold, frame in cv_splits.groupby("outer_fold"):
        if set(frame["sample_id"]) != expected_ids or len(frame) != len(expected_ids):
            raise AssertionError(f"Fold {outer_fold} does not cover each image once")
        split_sets = {
            name: set(group["sample_id"]) for name, group in frame.groupby("split")
        }
        if set(split_sets) != {"train", "validation", "test"}:
            raise AssertionError(f"Fold {outer_fold} is missing a split")
        if split_sets["train"] & split_sets["validation"]:
            raise AssertionError(f"Train/validation leakage in fold {outer_fold}")
        if split_sets["train"] & split_sets["test"]:
            raise AssertionError(f"Train/test leakage in fold {outer_fold}")
        if split_sets["validation"] & split_sets["test"]:
            raise AssertionError(f"Validation/test leakage in fold {outer_fold}")


def relative_to_root(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def ensure_unique(values: Iterable[str], label: str) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} values detected")
