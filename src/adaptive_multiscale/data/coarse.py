"""Coarse-resolution data preparation and loading for M-A segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from torch.utils.data import Dataset

from adaptive_multiscale.data.ma_islands import load_grayscale_image


@dataclass(frozen=True)
class NormalizationStatistics:
    """Train-fold grayscale normalization parameters."""

    mean: float
    std: float


def _resolve(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def downsample_image(
    image: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Downsample a grayscale image with bilinear, anti-aliased resampling."""

    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, received {image.shape}")
    resized = resize(
        image,
        (target_height, target_width),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(np.rint(resized), 0, 255).astype(np.uint8)


def downsample_mask(
    mask: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Downsample a binary mask with nearest-neighbor interpolation."""

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, received {mask.shape}")
    binary = (mask > 0).astype(np.uint8)
    resized = resize(
        binary,
        (target_height, target_width),
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )
    return (resized > 0).astype(np.uint8)


def prepare_coarse_dataset(
    image_statistics: pd.DataFrame,
    project_root: Path,
    output_dir: Path,
    target_height: int,
    target_width: int,
) -> pd.DataFrame:
    """Create paired coarse images/masks and return a reproducibility manifest."""

    required = {
        "sample_id",
        "file_name",
        "source_regime",
        "image_path",
        "mask_path",
    }
    missing = required - set(image_statistics.columns)
    if missing:
        raise ValueError(f"Image manifest is missing columns: {sorted(missing)}")

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for source in image_statistics.sort_values("sample_id").itertuples(index=False):
        source_image, _, _ = load_grayscale_image(
            _resolve(project_root, source.image_path)
        )
        source_mask = cv2.imread(
            str(_resolve(project_root, source.mask_path)), cv2.IMREAD_GRAYSCALE
        )
        if source_mask is None:
            raise FileNotFoundError(source.mask_path)
        if source_image.shape != source_mask.shape:
            raise ValueError(
                f"Image/mask mismatch for {source.sample_id}: "
                f"{source_image.shape} versus {source_mask.shape}"
            )

        coarse_image = downsample_image(source_image, target_height, target_width)
        coarse_mask = downsample_mask(source_mask, target_height, target_width)
        image_path = images_dir / f"{source.sample_id}.png"
        mask_path = masks_dir / f"{source.sample_id}.png"
        if not cv2.imwrite(str(image_path), coarse_image):
            raise OSError(f"Could not write {image_path}")
        if not cv2.imwrite(str(mask_path), coarse_mask * 255):
            raise OSError(f"Could not write {mask_path}")

        reloaded_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        reloaded_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if reloaded_image is None or reloaded_image.shape != (target_height, target_width):
            raise ValueError(f"Coarse image validation failed for {source.sample_id}")
        if reloaded_mask is None or set(np.unique(reloaded_mask)) - {0, 255}:
            raise ValueError(f"Coarse mask validation failed for {source.sample_id}")

        foreground_pixels = int(coarse_mask.sum())
        total_pixels = int(coarse_mask.size)
        rows.append(
            {
                "sample_id": source.sample_id,
                "file_name": source.file_name,
                "source_regime": source.source_regime,
                "width": target_width,
                "height": target_height,
                "foreground_pixels": foreground_pixels,
                "background_pixels": total_pixels - foreground_pixels,
                "foreground_fraction": foreground_pixels / total_pixels,
                "coarse_image_path": str(image_path.relative_to(project_root)),
                "coarse_mask_path": str(mask_path.relative_to(project_root)),
                "source_image_path": str(source.image_path),
                "source_mask_path": str(source.mask_path),
                "image_resampling": "bilinear_with_anti_aliasing",
                "mask_resampling": "nearest_neighbor",
            }
        )
    return pd.DataFrame(rows)


def fold_records(
    coarse_manifest: pd.DataFrame,
    cv_manifest: pd.DataFrame,
    outer_fold: int,
    split: str,
) -> pd.DataFrame:
    """Join one fold/split to coarse paths, preserving image-level isolation."""

    selected = cv_manifest[
        (cv_manifest["outer_fold"] == outer_fold) & (cv_manifest["split"] == split)
    ].copy()
    if selected.empty:
        raise ValueError(f"No records for fold={outer_fold}, split={split}")
    if selected["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate images in fold={outer_fold}, split={split}")
    merged = selected.merge(
        coarse_manifest,
        on=["sample_id", "file_name", "source_regime"],
        how="left",
        validate="one_to_one",
    )
    if merged[["coarse_image_path", "coarse_mask_path"]].isna().any().any():
        raise ValueError(f"Missing coarse paths in fold={outer_fold}, split={split}")
    return merged.sort_values("sample_id").reset_index(drop=True)


def calculate_normalization(
    records: pd.DataFrame, project_root: Path
) -> NormalizationStatistics:
    """Compute global mean/std from training images only, in [0, 1] units."""

    pixel_count = 0
    pixel_sum = 0.0
    squared_sum = 0.0
    for path_value in records["coarse_image_path"]:
        image = cv2.imread(str(_resolve(project_root, path_value)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path_value)
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


def calculate_positive_weight(records: pd.DataFrame) -> float:
    """Return background/foreground pixel ratio for weighted BCE."""

    foreground = int(records["foreground_pixels"].sum())
    background = int(records["background_pixels"].sum())
    if foreground <= 0 or background <= 0:
        raise ValueError("Both foreground and background pixels are required")
    return background / foreground


class CoarseSegmentationDataset(Dataset):
    """Full-frame 512 x 384 coarse SEM images and binary masks."""

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
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.records.iloc[index]
        image = cv2.imread(
            str(_resolve(self.project_root, row["coarse_image_path"])),
            cv2.IMREAD_GRAYSCALE,
        )
        mask = cv2.imread(
            str(_resolve(self.project_root, row["coarse_mask_path"])),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing coarse pair for {row['sample_id']}")
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            if self.rng.random() < self.horizontal_flip_probability:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
            if self.rng.random() < self.vertical_flip_probability:
                image = np.flip(image, axis=0)
                mask = np.flip(mask, axis=0)

        image_values = image.astype(np.float32) / 255.0
        image_values = (
            image_values - self.normalization.mean
        ) / self.normalization.std
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_values[None])),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])),
            "sample_id": str(row["sample_id"]),
        }
