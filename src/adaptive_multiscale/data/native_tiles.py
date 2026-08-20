"""Native-resolution non-overlapping tile data for fine segmentation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from adaptive_multiscale.data.coarse import NormalizationStatistics
from adaptive_multiscale.data.ma_islands import (
    load_grayscale_image,
    tile_foreground_counts,
)


def _resolve(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def build_native_tile_manifest(
    image_manifest: pd.DataFrame,
    project_root: Path,
    tile_size: int,
    expected_height: int,
    expected_width: int,
) -> pd.DataFrame:
    """Describe all exact native tiles and their target foreground counts."""

    required = {
        "sample_id",
        "file_name",
        "source_regime",
        "source_image_path",
        "source_mask_path",
    }
    missing = required - set(image_manifest.columns)
    if missing:
        raise ValueError(f"Image manifest is missing columns: {sorted(missing)}")
    if expected_height % tile_size or expected_width % tile_size:
        raise ValueError("Native dimensions must be divisible by tile size")

    rows: list[dict[str, object]] = []
    tile_rows = expected_height // tile_size
    tile_columns = expected_width // tile_size
    for record in image_manifest.sort_values("sample_id").itertuples(index=False):
        image, _, _ = load_grayscale_image(
            _resolve(project_root, record.source_image_path)
        )
        mask = cv2.imread(
            str(_resolve(project_root, record.source_mask_path)),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise FileNotFoundError(record.source_mask_path)
        if image.shape != mask.shape or image.shape != (expected_height, expected_width):
            raise ValueError(
                f"Native geometry mismatch for {record.sample_id}: "
                f"image={image.shape}, mask={mask.shape}"
            )
        binary = mask > 0
        foreground_counts = tile_foreground_counts(binary, tile_size)
        for tile_row in range(tile_rows):
            for tile_column in range(tile_columns):
                tile_index = tile_row * tile_columns + tile_column
                x0 = tile_column * tile_size
                y0 = tile_row * tile_size
                foreground = int(foreground_counts[tile_row, tile_column])
                pixels = tile_size * tile_size
                rows.append(
                    {
                        "sample_id": record.sample_id,
                        "file_name": record.file_name,
                        "source_regime": record.source_regime,
                        "tile_index": tile_index,
                        "tile_row": tile_row,
                        "tile_column": tile_column,
                        "x0": x0,
                        "y0": y0,
                        "x1": x0 + tile_size,
                        "y1": y0 + tile_size,
                        "tile_size": tile_size,
                        "foreground_pixels": foreground,
                        "background_pixels": pixels - foreground,
                        "foreground_fraction": foreground / pixels,
                        "source_image_path": str(record.source_image_path),
                        "source_mask_path": str(record.source_mask_path),
                    }
                )
    return pd.DataFrame(rows)


def fold_tile_records(
    tile_manifest: pd.DataFrame,
    cv_manifest: pd.DataFrame,
    outer_fold: int,
    split: str,
    tiles_per_image: int,
) -> pd.DataFrame:
    """Select tiles using only the original-image membership of one CV split."""

    selected_images = cv_manifest[
        (cv_manifest["outer_fold"] == outer_fold) & (cv_manifest["split"] == split)
    ][["sample_id", "file_name", "source_regime", "split"]].copy()
    if selected_images.empty or selected_images["sample_id"].duplicated().any():
        raise ValueError(f"Invalid image split for fold={outer_fold}, split={split}")
    result = selected_images.merge(
        tile_manifest,
        on=["sample_id", "file_name", "source_regime"],
        how="left",
        validate="one_to_many",
    )
    counts = result.groupby("sample_id").size()
    if len(counts) != len(selected_images) or not counts.eq(tiles_per_image).all():
        raise ValueError(
            f"Expected {tiles_per_image} tiles per image for fold={outer_fold}, split={split}"
        )
    result.insert(0, "outer_fold", int(outer_fold))
    return result.sort_values(["sample_id", "tile_index"]).reset_index(drop=True)


def calculate_native_normalization(
    tile_records: pd.DataFrame,
    project_root: Path,
) -> NormalizationStatistics:
    """Calculate full native-image normalization from training images only."""

    unique_images = tile_records[["sample_id", "source_image_path"]].drop_duplicates()
    pixel_count = 0
    pixel_sum = 0.0
    squared_sum = 0.0
    for record in unique_images.itertuples(index=False):
        image, _, _ = load_grayscale_image(
            _resolve(project_root, record.source_image_path)
        )
        values = image.astype(np.float64) / 255.0
        pixel_count += values.size
        pixel_sum += float(values.sum())
        squared_sum += float(np.square(values).sum())
    mean = pixel_sum / pixel_count
    variance = max(squared_sum / pixel_count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if std < 1e-8:
        raise ValueError("Training images have near-zero intensity variance")
    return NormalizationStatistics(mean=float(mean), std=std)


def calculate_tile_positive_weight(tile_records: pd.DataFrame) -> float:
    """Return background/foreground ratio across training tiles."""

    foreground = int(tile_records["foreground_pixels"].sum())
    background = int(tile_records["background_pixels"].sum())
    if foreground <= 0 or background <= 0:
        raise ValueError("Both foreground and background pixels are required")
    return background / foreground


class NativeTileDataset(Dataset):
    """Lazy, cached native-image tiles with paired geometric augmentation."""

    def __init__(
        self,
        records: pd.DataFrame,
        project_root: Path,
        normalization: NormalizationStatistics,
        augment: bool = False,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        random_seed: int = 0,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.project_root = project_root
        self.normalization = normalization
        self.augment = augment
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.vertical_flip_probability = float(vertical_flip_probability)
        self.rng = np.random.default_rng(random_seed)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_pair(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        sample_id = str(row["sample_id"])
        if sample_id not in self._cache:
            image, _, _ = load_grayscale_image(
                _resolve(self.project_root, row["source_image_path"])
            )
            mask = cv2.imread(
                str(_resolve(self.project_root, row["source_mask_path"])),
                cv2.IMREAD_GRAYSCALE,
            )
            if mask is None or image.shape != mask.shape:
                raise FileNotFoundError(f"Missing native pair for {sample_id}")
            self._cache[sample_id] = (image, mask)
        return self._cache[sample_id]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.records.iloc[index]
        image, mask = self._load_pair(row)
        x0, x1 = int(row["x0"]), int(row["x1"])
        y0, y1 = int(row["y0"]), int(row["y1"])
        image_tile = image[y0:y1, x0:x1]
        mask_tile = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
        expected = int(row["tile_size"])
        if image_tile.shape != (expected, expected) or mask_tile.shape != image_tile.shape:
            raise ValueError(f"Invalid tile geometry for {row['sample_id']}:{row['tile_index']}")

        if self.augment:
            if self.rng.random() < self.horizontal_flip_probability:
                image_tile = np.flip(image_tile, axis=1)
                mask_tile = np.flip(mask_tile, axis=1)
            if self.rng.random() < self.vertical_flip_probability:
                image_tile = np.flip(image_tile, axis=0)
                mask_tile = np.flip(mask_tile, axis=0)

        image_values = image_tile.astype(np.float32) / 255.0
        image_values = (
            image_values - self.normalization.mean
        ) / self.normalization.std
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_values[None])),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_tile[None])),
            "sample_id": str(row["sample_id"]),
            "tile_index": int(row["tile_index"]),
            "tile_row": int(row["tile_row"]),
            "tile_column": int(row["tile_column"]),
            "x0": x0,
            "y0": y0,
        }
