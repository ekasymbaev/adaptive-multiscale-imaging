#!/usr/bin/env python3
"""Train and evaluate native-tile U-Nets over five image-level folds."""

from __future__ import annotations

import argparse
import copy
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

from adaptive_multiscale.data.ma_islands import load_grayscale_image
from adaptive_multiscale.data.native_tiles import (
    NativeTileDataset,
    build_native_tile_manifest,
    calculate_native_normalization,
    calculate_tile_positive_weight,
    fold_tile_records,
)
from adaptive_multiscale.models import CompactUNet
from adaptive_multiscale.selection.tile_ranking import paired_bootstrap_interval
from adaptive_multiscale.training.losses import WeightedBCEDiceLoss
from adaptive_multiscale.training.metrics import (
    binary_segmentation_metrics,
    micro_average,
)
from adaptive_multiscale.training.reproducibility import seed_everything, select_device


METRICS = ("dice", "iou", "precision", "recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fine_model.json"))
    parser.add_argument(
        "--folds", type=int, nargs="*", help="Optional subset of configured folds."
    )
    parser.add_argument("--max-epochs", type=int, help="Optional training-epoch override.")
    parser.add_argument("--output-dir", type=Path, help="Optional output-directory override.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_model(config: dict[str, Any]) -> CompactUNet:
    return CompactUNet(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        encoder_channels=tuple(config["encoder_channels"]),
        bottleneck_dropout=float(config["bottleneck_dropout"]),
    )


def make_loader(
    records: pd.DataFrame,
    project_root: Path,
    normalization: Any,
    training: bool,
    config: dict[str, Any],
    seed: int,
) -> DataLoader:
    dataset = NativeTileDataset(
        records,
        project_root=project_root,
        normalization=normalization,
        augment=training,
        horizontal_flip_probability=float(config["horizontal_flip_probability"]),
        vertical_flip_probability=float(config["vertical_flip_probability"]),
        random_seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=training,
        num_workers=int(config["num_workers"]),
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_total = 0.0
    sample_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, masks)
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        loss_total += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
    return loss_total / sample_count


def image_metrics_from_tiles(
    tile_metrics: pd.DataFrame,
    fold: int,
    split: str,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, group in tile_metrics.groupby("sample_id", sort=True):
        metrics = micro_average(group.to_dict("records"))
        total_pixels = sum(
            int(metrics[name])
            for name in ("true_positive", "false_positive", "false_negative", "true_negative")
        )
        predicted_foreground = int(metrics["true_positive"]) + int(metrics["false_positive"])
        target_foreground = int(metrics["true_positive"]) + int(metrics["false_negative"])
        predicted_fraction = predicted_foreground / total_pixels
        target_fraction = target_foreground / total_pixels
        rows.append(
            {
                "outer_fold": fold,
                "split": split,
                "sample_id": sample_id,
                "threshold": threshold,
                "tiles": int(len(group)),
                "predicted_foreground_fraction": predicted_fraction,
                "target_foreground_fraction": target_fraction,
                "area_fraction_error_signed": predicted_fraction - target_fraction,
                "area_fraction_error_absolute": abs(predicted_fraction - target_fraction),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def evaluate_tiles(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    threshold: float,
    fold: int,
    split: str,
    full_height: int,
    full_width: int,
    tile_size: int,
    prediction_dir: Path | None = None,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Evaluate every tile and optionally reassemble complete prediction maps."""

    model.eval()
    loss_total = 0.0
    sample_count = 0
    rows: list[dict[str, object]] = []
    full_predictions: dict[str, np.ndarray] = {}
    coverage: dict[str, np.ndarray] = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = loss_function(logits, masks)
        probabilities = torch.sigmoid(logits).cpu().numpy()
        truth = masks.cpu().numpy() > 0.5
        batch_size = images.shape[0]
        loss_total += float(loss.cpu()) * batch_size
        sample_count += batch_size

        for item_index, sample_id_value in enumerate(batch["sample_id"]):
            sample_id = str(sample_id_value)
            predicted = probabilities[item_index, 0] >= threshold
            target = truth[item_index, 0]
            metrics = binary_segmentation_metrics(predicted, target)
            tile_index = int(batch["tile_index"][item_index])
            tile_row = int(batch["tile_row"][item_index])
            tile_column = int(batch["tile_column"][item_index])
            x0 = int(batch["x0"][item_index])
            y0 = int(batch["y0"][item_index])
            rows.append(
                {
                    "outer_fold": fold,
                    "split": split,
                    "sample_id": sample_id,
                    "tile_index": tile_index,
                    "tile_row": tile_row,
                    "tile_column": tile_column,
                    "x0": x0,
                    "y0": y0,
                    "threshold": threshold,
                    "foreground_probability_mean": float(probabilities[item_index, 0].mean()),
                    "predicted_foreground_fraction": float(predicted.mean()),
                    "target_foreground_fraction": float(target.mean()),
                    **metrics,
                }
            )
            if prediction_dir is not None:
                if sample_id not in full_predictions:
                    full_predictions[sample_id] = np.zeros(
                        (full_height, full_width), dtype=np.uint8
                    )
                    coverage[sample_id] = np.zeros(
                        (full_height, full_width), dtype=np.uint8
                    )
                y1, x1 = y0 + tile_size, x0 + tile_size
                full_predictions[sample_id][y0:y1, x0:x1] = predicted.astype(np.uint8)
                coverage[sample_id][y0:y1, x0:x1] += 1

    tile_frame = pd.DataFrame(rows).sort_values(["sample_id", "tile_index"])
    image_frame = image_metrics_from_tiles(tile_frame, fold, split, threshold)
    if prediction_dir is not None:
        if set(full_predictions) != set(image_frame["sample_id"]):
            raise ValueError("Reassembled prediction set does not match evaluated images")
        for sample_id, prediction in full_predictions.items():
            if not np.all(coverage[sample_id] == 1):
                raise ValueError(f"Incomplete or overlapping reconstruction for {sample_id}")
            output_path = prediction_dir / f"{sample_id}.png"
            if not cv2.imwrite(str(output_path), prediction * 255):
                raise OSError(f"Failed to save prediction {output_path}")
    return loss_total / sample_count, tile_frame, image_frame


def metric_means(metrics: pd.DataFrame) -> dict[str, float]:
    return {f"macro_{name}": float(metrics[name].mean()) for name in METRICS}


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def train_fold(
    fold: int,
    project_root: Path,
    tile_manifest: pd.DataFrame,
    cv_manifest: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    dataset_config = config["dataset"]
    training_config = config["training"]
    model_config = config["model"]
    tiles_per_image = int(dataset_config["tiles_per_image"])
    fold_seed = int(training_config["random_seed"]) + fold
    seed_everything(fold_seed)

    train_records = fold_tile_records(
        tile_manifest, cv_manifest, fold, "train", tiles_per_image
    )
    validation_records = fold_tile_records(
        tile_manifest, cv_manifest, fold, "validation", tiles_per_image
    )
    test_records = fold_tile_records(
        tile_manifest, cv_manifest, fold, "test", tiles_per_image
    )
    split_sets = [
        set(frame["sample_id"])
        for frame in (train_records, validation_records, test_records)
    ]
    if (
        split_sets[0] & split_sets[1]
        or split_sets[0] & split_sets[2]
        or split_sets[1] & split_sets[2]
    ):
        raise ValueError(f"Original-image leakage detected in fold {fold}")

    normalization = calculate_native_normalization(train_records, project_root)
    positive_weight = calculate_tile_positive_weight(train_records)
    train_loader = make_loader(
        train_records, project_root, normalization, True, training_config, fold_seed
    )
    validation_loader = make_loader(
        validation_records,
        project_root,
        normalization,
        False,
        training_config,
        fold_seed + 100,
    )
    test_loader = make_loader(
        test_records,
        project_root,
        normalization,
        False,
        training_config,
        fold_seed + 200,
    )

    model = make_model(model_config).to(device)
    loss_function = WeightedBCEDiceLoss(
        positive_weight=positive_weight,
        bce_weight=float(training_config["bce_weight"]),
        dice_weight=float(training_config["dice_weight"]),
        dice_smooth=float(training_config["soft_dice_smooth"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_config["scheduler_factor"]),
        patience=int(training_config["scheduler_patience"]),
        min_lr=float(training_config["minimum_learning_rate"]),
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"fold_{fold}_best.pt"
    history_rows: list[dict[str, object]] = []
    best_dice = -np.inf
    best_epoch = 0
    stale_epochs = 0
    fold_start = time.perf_counter()
    threshold = float(training_config["prediction_threshold"])
    full_height = int(dataset_config["native_height"])
    full_width = int(dataset_config["native_width"])
    tile_size = int(dataset_config["tile_size"])

    for epoch in range(1, int(training_config["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        validation_loss, _, validation_images = evaluate_tiles(
            model,
            validation_loader,
            loss_function,
            device,
            threshold,
            fold,
            "validation",
            full_height,
            full_width,
            tile_size,
        )
        validation_means = metric_means(validation_images)
        validation_dice = validation_means["macro_dice"]
        scheduler.step(validation_dice)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "outer_fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{
                    f"validation_{key}": value
                    for key, value in validation_means.items()
                },
                "learning_rate": learning_rate,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )

        improved = validation_dice > best_dice + float(
            training_config["early_stopping_min_delta"]
        )
        if improved:
            best_dice = validation_dice
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": 1,
                    "experiment_name": config["experiment_name"],
                    "outer_fold": fold,
                    "epoch": epoch,
                    "model_config": model_config,
                    "model_state_dict": cpu_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "normalization_mean": normalization.mean,
                    "normalization_std": normalization.std,
                    "positive_weight": positive_weight,
                    "prediction_threshold": threshold,
                    "validation_metrics": validation_means,
                    "random_seed": fold_seed,
                    "tile_size": tile_size,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        print(
            f"fold={fold} epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_image_dice={validation_dice:.4f} "
            f"lr={learning_rate:.2e}{' *' if improved else ''}",
            flush=True,
        )
        if stale_epochs >= int(training_config["early_stopping_patience"]):
            break

    last_path = checkpoint_dir / f"fold_{fold}_last.pt"
    torch.save(
        {
            "schema_version": 1,
            "experiment_name": config["experiment_name"],
            "outer_fold": fold,
            "epoch": history_rows[-1]["epoch"],
            "model_config": model_config,
            "model_state_dict": cpu_state_dict(model),
            "normalization_mean": normalization.mean,
            "normalization_std": normalization.std,
            "positive_weight": positive_weight,
            "prediction_threshold": threshold,
            "random_seed": fold_seed,
            "tile_size": tile_size,
        },
        last_path,
    )

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_validation_loss, _, best_validation_images = evaluate_tiles(
        model,
        validation_loader,
        loss_function,
        device,
        threshold,
        fold,
        "validation",
        full_height,
        full_width,
        tile_size,
    )
    test_loss, test_tiles, test_images = evaluate_tiles(
        model,
        test_loader,
        loss_function,
        device,
        threshold,
        fold,
        "test",
        full_height,
        full_width,
        tile_size,
        prediction_dir=output_dir / "predicted_masks" / f"fold_{fold}",
    )
    metadata = test_records[
        ["sample_id", "file_name", "source_regime"]
    ].drop_duplicates()
    test_images = test_images.merge(
        metadata, on="sample_id", how="left", validate="one_to_one"
    )
    test_tiles = test_tiles.merge(
        metadata, on="sample_id", how="left", validate="many_to_one"
    )
    validation_means = metric_means(best_validation_images)
    test_means = metric_means(test_images)
    test_micro = micro_average(test_images.to_dict("records"))
    fold_summary: dict[str, object] = {
        "outer_fold": fold,
        "random_seed": fold_seed,
        "train_images": len(split_sets[0]),
        "validation_images": len(split_sets[1]),
        "test_images": len(split_sets[2]),
        "train_tiles": len(train_records),
        "validation_tiles": len(validation_records),
        "test_tiles": len(test_records),
        "train_empty_tiles": int(train_records["foreground_pixels"].eq(0).sum()),
        "train_under_1pct_foreground_tiles": int(
            train_records["foreground_fraction"].lt(0.01).sum()
        ),
        "normalization_mean": normalization.mean,
        "normalization_std": normalization.std,
        "positive_weight": positive_weight,
        "best_epoch": best_epoch,
        "epochs_run": len(history_rows),
        "best_validation_loss": best_validation_loss,
        **{f"validation_{key}": value for key, value in validation_means.items()},
        "test_loss": test_loss,
        **{f"test_{key}": value for key, value in test_means.items()},
        "test_area_fraction_absolute_error_mean": float(
            test_images["area_fraction_error_absolute"].mean()
        ),
        **{
            f"test_micro_{key}": value
            for key, value in test_micro.items()
            if key in METRICS
        },
        "training_seconds": time.perf_counter() - fold_start,
        "best_checkpoint": str(best_path.relative_to(project_root))
        if best_path.is_relative_to(project_root)
        else str(best_path),
        "last_checkpoint": str(last_path.relative_to(project_root))
        if last_path.is_relative_to(project_root)
        else str(last_path),
    }
    return pd.DataFrame(history_rows), fold_summary, test_images, test_tiles


def coarse_native_comparison(
    fine_metrics: pd.DataFrame,
    image_manifest: pd.DataFrame,
    coarse_metrics: pd.DataFrame,
    project_root: Path,
    prediction_pattern: str,
    native_width: int,
    native_height: int,
) -> pd.DataFrame:
    manifest = image_manifest.set_index("sample_id")
    coarse_recorded = coarse_metrics.set_index("sample_id")
    rows: list[dict[str, object]] = []
    for fine in fine_metrics.itertuples(index=False):
        record = manifest.loc[fine.sample_id]
        target = cv2.imread(
            str(project_root / record["source_mask_path"]), cv2.IMREAD_GRAYSCALE
        )
        coarse_prediction = cv2.imread(
            str(
                project_root
                / prediction_pattern.format(
                    fold=int(fine.outer_fold), sample_id=fine.sample_id
                )
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        if target is None or coarse_prediction is None:
            raise FileNotFoundError(f"Missing comparison data for {fine.sample_id}")
        if target.shape != (native_height, native_width):
            raise ValueError(f"Native target geometry mismatch for {fine.sample_id}")
        upsampled = cv2.resize(
            coarse_prediction,
            (native_width, native_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        truth = target > 0
        coarse = binary_segmentation_metrics(upsampled, truth)
        coarse_predicted_fraction = float(upsampled.mean())
        target_fraction = float(truth.mean())
        coarse_area_signed = coarse_predicted_fraction - target_fraction
        fine_values = {name: float(getattr(fine, name)) for name in METRICS}
        row: dict[str, object] = {
            "outer_fold": int(fine.outer_fold),
            "sample_id": fine.sample_id,
            "file_name": fine.file_name,
            "source_regime": fine.source_regime,
            "target_foreground_fraction": target_fraction,
            "coarse_predicted_foreground_fraction_native": coarse_predicted_fraction,
            "fine_predicted_foreground_fraction": float(
                fine.predicted_foreground_fraction
            ),
            "coarse_area_fraction_error_signed": coarse_area_signed,
            "coarse_area_fraction_error_absolute": abs(coarse_area_signed),
            "fine_area_fraction_error_signed": float(fine.area_fraction_error_signed),
            "fine_area_fraction_error_absolute": float(
                fine.area_fraction_error_absolute
            ),
            "area_fraction_absolute_error_improvement": (
                abs(coarse_area_signed) - float(fine.area_fraction_error_absolute)
            ),
            "coarse_step2_domain_dice": float(
                coarse_recorded.loc[fine.sample_id, "dice"]
            ),
        }
        for name in METRICS:
            row[f"coarse_{name}"] = float(coarse[name])
            row[f"fine_{name}"] = fine_values[name]
            row[f"fine_minus_coarse_{name}"] = fine_values[name] - float(coarse[name])
        for name in ("true_positive", "false_positive", "false_negative", "true_negative"):
            row[f"coarse_{name}"] = int(coarse[name])
            row[f"fine_{name}"] = int(getattr(fine, name))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["outer_fold", "sample_id"])


def model_summary_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = [("all_images", comparison)] + [
        (f"regime_{regime}", comparison[comparison["source_regime"] == regime])
        for regime in ("I", "II", "III", "IV")
    ]
    for scope, frame in scopes:
        for model_name in ("coarse", "fine"):
            counts = [
                {
                    name: int(row[f"{model_name}_{name}"])
                    for name in ("true_positive", "false_positive", "false_negative", "true_negative")
                }
                for _, row in frame.iterrows()
            ]
            micro = micro_average(counts)
            prefix = f"{model_name}_"
            rows.append(
                {
                    "scope": scope,
                    "source_regime": scope.replace("regime_", "")
                    if scope.startswith("regime_")
                    else "all",
                    "model": model_name,
                    "images": len(frame),
                    **{
                        f"macro_{metric}": float(frame[f"{prefix}{metric}"].mean())
                        for metric in METRICS
                    },
                    **{
                        f"micro_{metric}": float(micro[metric])
                        for metric in METRICS
                    },
                    "area_fraction_error_absolute_mean": float(
                        frame[f"{prefix}area_fraction_error_absolute"].mean()
                    ),
                    "area_fraction_error_signed_mean": float(
                        frame[f"{prefix}area_fraction_error_signed"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_comparison_rows(
    comparison: pd.DataFrame,
    confidence_level: float,
    resamples: int,
    seed_base: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = [("all_images", comparison)] + [
        (f"regime_{regime}", comparison[comparison["source_regime"] == regime])
        for regime in ("I", "II", "III", "IV")
    ]
    definitions = [
        (metric, f"coarse_{metric}", f"fine_{metric}", "fine_minus_coarse")
        for metric in METRICS
    ] + [
        (
            "area_fraction_absolute_error",
            "coarse_area_fraction_error_absolute",
            "fine_area_fraction_error_absolute",
            "coarse_minus_fine",
        )
    ]
    comparison_index = 0
    for scope, frame in scopes:
        for metric, coarse_column, fine_column, direction in definitions:
            coarse_values = frame[coarse_column].to_numpy(dtype=np.float64)
            fine_values = frame[fine_column].to_numpy(dtype=np.float64)
            differences = (
                fine_values - coarse_values
                if direction == "fine_minus_coarse"
                else coarse_values - fine_values
            )
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
                    wilcoxon(
                        differences,
                        alternative="greater",
                        zero_method="wilcox",
                    ).pvalue
                )
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "images": len(frame),
                    "coarse_mean": float(coarse_values.mean()),
                    "fine_mean": float(fine_values.mean()),
                    "improvement_direction": direction,
                    "improvement_mean": mean,
                    "improvement_ci_low": low,
                    "improvement_ci_high": high,
                    "improvement_median": float(np.median(differences)),
                    "improved_image_fraction": float((differences > 0.0).mean()),
                    "one_sided_wilcoxon_p": p_value,
                    "confidence_level": confidence_level,
                }
            )
            comparison_index += 1
    return pd.DataFrame(rows)


def plot_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, history["outer_fold"].nunique()))
    for color, (fold, fold_history) in zip(colors, history.groupby("outer_fold")):
        label = f"Fold {fold}"
        axes[0, 0].plot(fold_history["epoch"], fold_history["train_loss"], color=color, label=label)
        axes[0, 1].plot(fold_history["epoch"], fold_history["validation_loss"], color=color, label=label)
        axes[1, 0].plot(
            fold_history["epoch"],
            fold_history["validation_macro_dice"],
            color=color,
            label=label,
        )
        axes[1, 1].plot(fold_history["epoch"], fold_history["learning_rate"], color=color, label=label)
    for axis, title, ylabel in zip(
        axes.flat,
        ("Training loss", "Validation loss", "Validation full-image Dice", "Learning rate"),
        ("Weighted BCE + soft Dice", "Weighted BCE + soft Dice", "Dice", "Rate"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    figure.suptitle("Fine U-Net training across five image-level folds", fontsize=14)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_comparison_summary(
    comparison: pd.DataFrame,
    model_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    regime_colors = comparison["source_regime"].map(
        {"I": "#3B82F6", "II": "#10B981", "III": "#F59E0B", "IV": "#8B5CF6"}
    )
    axes[0, 0].scatter(
        comparison["coarse_dice"],
        comparison["fine_dice"],
        c=regime_colors,
        edgecolor="white",
        linewidth=0.5,
        s=55,
    )
    minimum = min(comparison["coarse_dice"].min(), comparison["fine_dice"].min()) - 0.02
    axes[0, 0].plot([minimum, 1.0], [minimum, 1.0], "--", color="#64748B")
    axes[0, 0].set_xlim(minimum, 1.0)
    axes[0, 0].set_ylim(minimum, 1.0)
    axes[0, 0].set_xlabel("Upsampled coarse Dice on native mask")
    axes[0, 0].set_ylabel("Fine full-resolution Dice")
    axes[0, 0].set_title("Paired held-out image performance")

    overall = model_summary[model_summary["scope"] == "all_images"].set_index("model")
    positions = np.arange(len(METRICS))
    width = 0.36
    axes[0, 1].bar(
        positions - width / 2,
        [overall.loc["coarse", f"macro_{metric}"] for metric in METRICS],
        width,
        label="Coarse",
        color="#94A3B8",
    )
    axes[0, 1].bar(
        positions + width / 2,
        [overall.loc["fine", f"macro_{metric}"] for metric in METRICS],
        width,
        label="Fine",
        color="#2F6B8A",
    )
    axes[0, 1].set_xticks(positions, [value.title() for value in METRICS])
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Macro segmentation metrics")
    axes[0, 1].legend(frameon=False)

    regime_order = ["I", "II", "III", "IV"]
    fine_gain = comparison.groupby("source_regime")["fine_minus_coarse_dice"].mean().loc[regime_order]
    axes[1, 0].bar(
        np.arange(4),
        fine_gain,
        color=["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"],
    )
    axes[1, 0].axhline(0.0, color="#64748B", linewidth=1)
    axes[1, 0].set_xticks(np.arange(4), [f"Regime {value}" for value in regime_order])
    axes[1, 0].set_ylabel("Fine minus coarse Dice")
    axes[1, 0].set_title("Dice improvement by source regime")

    coarse_area = model_summary[
        (model_summary["scope"] != "all_images") & (model_summary["model"] == "coarse")
    ].set_index("source_regime").loc[regime_order, "area_fraction_error_absolute_mean"]
    fine_area = model_summary[
        (model_summary["scope"] != "all_images") & (model_summary["model"] == "fine")
    ].set_index("source_regime").loc[regime_order, "area_fraction_error_absolute_mean"]
    axes[1, 1].bar(positions[:4] - width / 2, coarse_area, width, label="Coarse", color="#94A3B8")
    axes[1, 1].bar(positions[:4] + width / 2, fine_area, width, label="Fine", color="#2F6B8A")
    axes[1, 1].set_xticks(positions[:4], [f"Regime {value}" for value in regime_order])
    axes[1, 1].set_ylabel("Mean absolute M-A area-fraction error")
    axes[1, 1].set_title("Area-fraction accuracy by regime")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    figure.suptitle("Full-resolution fine model versus frozen coarse model", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def select_representatives(comparison: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    largest_gain = comparison.sort_values(
        ["fine_minus_coarse_dice", "sample_id"], ascending=[False, True]
    ).iloc[0].copy()
    largest_gain["example_role"] = "largest Dice improvement"
    selected.append(largest_gain)
    largest_loss = comparison.sort_values(
        ["fine_minus_coarse_dice", "sample_id"], ascending=[True, True]
    ).iloc[0].copy()
    largest_loss["example_role"] = "smallest Dice improvement"
    selected.append(largest_loss)
    for regime in ("I", "II", "III", "IV"):
        group = comparison[comparison["source_regime"] == regime].copy()
        median = group["fine_minus_coarse_dice"].median()
        group["distance_to_median_gain"] = (
            group["fine_minus_coarse_dice"] - median
        ).abs()
        row = group.sort_values(["distance_to_median_gain", "sample_id"]).iloc[0].copy()
        row["example_role"] = f"Regime {regime} median improvement"
        selected.append(row)
    return pd.DataFrame(selected).drop_duplicates("sample_id").reset_index(drop=True)


def error_code_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    codes = np.zeros(target.shape, dtype=np.uint8)
    codes[prediction & ~target] = 1
    codes[~prediction & target] = 2
    return codes


def plot_representative_comparisons(
    representatives: pd.DataFrame,
    image_manifest: pd.DataFrame,
    project_root: Path,
    coarse_prediction_pattern: str,
    fine_prediction_root: Path,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(representatives),
        6,
        figsize=(19, 3.0 * len(representatives)),
        constrained_layout=True,
    )
    paths = image_manifest.set_index("sample_id")
    error_cmap = ListedColormap(["#111827", "#EF4444", "#22D3EE"])
    for row_index, row in representatives.iterrows():
        record = paths.loc[row["sample_id"]]
        image, _, _ = load_grayscale_image(project_root / record["source_image_path"])
        target = cv2.imread(
            str(project_root / record["source_mask_path"]), cv2.IMREAD_GRAYSCALE
        ) > 0
        coarse_small = cv2.imread(
            str(
                project_root
                / coarse_prediction_pattern.format(
                    fold=int(row["outer_fold"]), sample_id=row["sample_id"]
                )
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        fine = cv2.imread(
            str(
                fine_prediction_root
                / f"fold_{int(row['outer_fold'])}"
                / f"{row['sample_id']}.png"
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        if coarse_small is None or fine is None:
            raise FileNotFoundError(f"Missing prediction for {row['sample_id']}")
        coarse = cv2.resize(
            coarse_small,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        fine = fine > 0
        panels = (
            (image, "gray", 0, 255),
            (target, "gray", 0, 1),
            (coarse, "gray", 0, 1),
            (fine, "gray", 0, 1),
            (error_code_map(coarse, target), error_cmap, 0, 2),
            (error_code_map(fine, target), error_cmap, 0, 2),
        )
        for column, (values, cmap, minimum, maximum) in enumerate(panels):
            axes[row_index, column].imshow(values, cmap=cmap, vmin=minimum, vmax=maximum)
            axes[row_index, column].axis("off")
        axes[row_index, 0].text(
            -0.03,
            0.5,
            (
                f"{row['sample_id']} | Regime {row['source_regime']}\n"
                f"{row['example_role']}\n"
                f"ΔDice {row['fine_minus_coarse_dice']:+.3f}"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
            clip_on=False,
        )
        axes[row_index, 2].text(
            0.02,
            0.98,
            f"Dice {row['coarse_dice']:.3f}",
            transform=axes[row_index, 2].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "#111827", "alpha": 0.75, "pad": 2, "edgecolor": "none"},
        )
        axes[row_index, 3].text(
            0.02,
            0.98,
            f"Dice {row['fine_dice']:.3f}",
            transform=axes[row_index, 3].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "#111827", "alpha": 0.75, "pad": 2, "edgecolor": "none"},
        )
    for column, title in enumerate(
        (
            "Native SEM",
            "Ground truth",
            "Upsampled coarse prediction",
            "Full fine prediction",
            "Coarse error",
            "Fine error",
        )
    ):
        axes[0, column].set_title(title, fontsize=10)
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
        "Held-out native-resolution predictions and error maps",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = copy.deepcopy(load_json(config_path))
    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ValueError("--max-epochs must be positive")
        config["training"]["max_epochs"] = args.max_epochs
    output_dir = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else project_root / (args.output_dir or config["output_dir"])
    )
    dataset_config = config["dataset"]
    training_config = config["training"]
    for directory in (
        output_dir / "history",
        output_dir / "metrics",
        output_dir / "figures",
        output_dir / "predicted_masks",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if int(training_config["cpu_threads"]) > 0:
        torch.set_num_threads(int(training_config["cpu_threads"]))
    device = select_device(str(training_config["device"]))
    image_manifest = pd.read_csv(project_root / dataset_config["coarse_manifest"])
    cv_manifest = pd.read_csv(project_root / dataset_config["cv_manifest"])
    coarse_metrics = pd.read_csv(
        project_root / config["coarse_comparison"]["metrics"]
    )
    tile_manifest = build_native_tile_manifest(
        image_manifest,
        project_root,
        int(dataset_config["tile_size"]),
        int(dataset_config["native_height"]),
        int(dataset_config["native_width"]),
    )
    expected_tiles = len(image_manifest) * int(dataset_config["tiles_per_image"])
    if len(tile_manifest) != expected_tiles:
        raise ValueError(f"Expected {expected_tiles} native tiles, found {len(tile_manifest)}")
    tile_manifest.to_csv(output_dir / "metrics" / "native_tile_manifest.csv", index=False)

    folds = args.folds if args.folds is not None else training_config["folds"]
    if not folds:
        raise ValueError("At least one fold is required")
    print(
        f"device={device} torch={torch.__version__} folds={folds} "
        f"batch_size={training_config['batch_size']}",
        flush=True,
    )

    histories: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    test_images: list[pd.DataFrame] = []
    test_tiles: list[pd.DataFrame] = []
    fold_manifest_frames: list[pd.DataFrame] = []
    experiment_start = time.perf_counter()
    for fold in folds:
        fold_int = int(fold)
        for split in ("train", "validation", "test"):
            fold_manifest_frames.append(
                fold_tile_records(
                    tile_manifest,
                    cv_manifest,
                    fold_int,
                    split,
                    int(dataset_config["tiles_per_image"]),
                )
            )
        history, fold_summary, fold_test_images, fold_test_tiles = train_fold(
            fold_int,
            project_root,
            tile_manifest,
            cv_manifest,
            config,
            output_dir,
            device,
        )
        histories.append(history)
        fold_summaries.append(fold_summary)
        test_images.append(fold_test_images)
        test_tiles.append(fold_test_tiles)
        history.to_csv(output_dir / "history" / f"fold_{fold_int}_history.csv", index=False)
        if device.type == "mps":
            torch.mps.empty_cache()

    fold_tile_manifest = pd.concat(fold_manifest_frames, ignore_index=True).sort_values(
        ["outer_fold", "split", "sample_id", "tile_index"]
    )
    history_frame = pd.concat(histories, ignore_index=True)
    fold_frame = pd.DataFrame(fold_summaries).sort_values("outer_fold")
    fine_image_frame = pd.concat(test_images, ignore_index=True).sort_values(
        ["outer_fold", "sample_id"]
    )
    fine_tile_frame = pd.concat(test_tiles, ignore_index=True).sort_values(
        ["outer_fold", "sample_id", "tile_index"]
    )
    fold_tile_manifest.to_csv(
        output_dir / "metrics" / "fold_tile_manifest.csv", index=False
    )
    history_frame.to_csv(output_dir / "history" / "all_folds_history.csv", index=False)
    fold_frame.to_csv(output_dir / "metrics" / "fold_metrics.csv", index=False)
    fine_image_frame.to_csv(
        output_dir / "metrics" / "per_image_fine_metrics.csv", index=False
    )
    fine_tile_frame.to_csv(
        output_dir / "metrics" / "per_tile_test_metrics.csv", index=False
    )

    comparison = coarse_native_comparison(
        fine_image_frame,
        image_manifest,
        coarse_metrics,
        project_root,
        config["coarse_comparison"]["prediction_pattern"],
        int(dataset_config["native_width"]),
        int(dataset_config["native_height"]),
    )
    model_summary = model_summary_rows(comparison)
    statistics_config = config["statistics"]
    paired = paired_comparison_rows(
        comparison,
        float(statistics_config["confidence_level"]),
        int(statistics_config["bootstrap_resamples"]),
        int(statistics_config["bootstrap_seed"]),
    )
    comparison.to_csv(output_dir / "metrics" / "fine_vs_coarse_per_image.csv", index=False)
    model_summary.to_csv(output_dir / "metrics" / "model_summary.csv", index=False)
    paired.to_csv(output_dir / "metrics" / "paired_comparison.csv", index=False)

    representatives = select_representatives(comparison)
    representatives.to_csv(
        output_dir / "metrics" / "representative_samples.csv", index=False
    )
    plot_training_curves(history_frame, output_dir / "figures" / "training_curves.png")
    plot_comparison_summary(
        comparison,
        model_summary,
        output_dir / "figures" / "fine_vs_coarse_summary.png",
    )
    plot_representative_comparisons(
        representatives,
        image_manifest,
        project_root,
        config["coarse_comparison"]["prediction_pattern"],
        output_dir / "predicted_masks",
        output_dir / "figures" / "representative_comparisons.png",
    )

    overall_models = model_summary[model_summary["scope"] == "all_images"].set_index(
        "model"
    )
    overall_paired = paired[paired["scope"] == "all_images"]
    summary: dict[str, object] = {
        "experiment_name": config["experiment_name"],
        "completed_folds": [int(value) for value in sorted(folds)],
        "cross_validation": "five-fold image-level out-of-fold evaluation",
        "held_out_images": int(len(fine_image_frame)),
        "held_out_tiles": int(len(fine_tile_frame)),
        "tile_geometry": {
            "native_width": int(dataset_config["native_width"]),
            "native_height": int(dataset_config["native_height"]),
            "tile_size": int(dataset_config["tile_size"]),
            "tiles_per_image": int(dataset_config["tiles_per_image"]),
        },
        "patch_sampling": {
            "method": training_config["patch_sampling"],
            "reason": training_config["patch_sampling_reason"],
            "all_dataset_empty_tiles": int(tile_manifest["foreground_pixels"].eq(0).sum()),
            "all_dataset_under_1pct_tiles": int(
                tile_manifest["foreground_fraction"].lt(0.01).sum()
            ),
        },
        "model_parameters": int(
            sum(parameter.numel() for parameter in make_model(config["model"]).parameters())
        ),
        "fine_macro_mean": {
            metric: float(overall_models.loc["fine", f"macro_{metric}"])
            for metric in METRICS
        },
        "fine_pixel_micro": {
            metric: float(overall_models.loc["fine", f"micro_{metric}"])
            for metric in METRICS
        },
        "coarse_native_macro_mean": {
            metric: float(overall_models.loc["coarse", f"macro_{metric}"])
            for metric in METRICS
        },
        "coarse_native_pixel_micro": {
            metric: float(overall_models.loc["coarse", f"micro_{metric}"])
            for metric in METRICS
        },
        "fine_area_fraction_error_absolute_mean": float(
            overall_models.loc["fine", "area_fraction_error_absolute_mean"]
        ),
        "coarse_area_fraction_error_absolute_mean": float(
            overall_models.loc["coarse", "area_fraction_error_absolute_mean"]
        ),
        "paired_improvements": overall_paired.to_dict(orient="records"),
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "total_training_seconds": time.perf_counter() - experiment_start,
        "config_path": str(config_path.relative_to(project_root)),
        "config": config,
    }
    with (output_dir / "metrics" / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "run_environment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "command": " ".join(sys.argv),
                "device": str(device),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
                "config": config,
            },
            handle,
            indent=2,
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
