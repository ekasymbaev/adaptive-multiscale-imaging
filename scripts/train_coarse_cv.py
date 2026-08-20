#!/usr/bin/env python3
"""Train and evaluate the compact coarse U-Net over five image-level folds."""

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
from torch.utils.data import DataLoader

from adaptive_multiscale.data.coarse import (
    CoarseSegmentationDataset,
    NormalizationStatistics,
    calculate_normalization,
    calculate_positive_weight,
    fold_records,
)
from adaptive_multiscale.models import CompactUNet
from adaptive_multiscale.training.losses import WeightedBCEDiceLoss
from adaptive_multiscale.training.metrics import (
    binary_segmentation_metrics,
    micro_average,
)
from adaptive_multiscale.training.reproducibility import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/coarse_model.json"))
    parser.add_argument(
        "--folds", type=int, nargs="*", help="Optional subset of configured folds."
    )
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
    normalization: NormalizationStatistics,
    training: bool,
    config: dict[str, Any],
    seed: int,
) -> DataLoader:
    dataset = CoarseSegmentationDataset(
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


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    threshold: float,
    fold: int,
    split: str,
    prediction_dir: Path | None = None,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    loss_total = 0.0
    sample_count = 0
    rows: list[dict[str, object]] = []
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

        for item_index, sample_id in enumerate(batch["sample_id"]):
            predicted = probabilities[item_index, 0] >= threshold
            target = truth[item_index, 0]
            metrics = binary_segmentation_metrics(predicted, target)
            rows.append(
                {
                    "outer_fold": fold,
                    "split": split,
                    "sample_id": sample_id,
                    "threshold": threshold,
                    "foreground_probability_mean": float(
                        probabilities[item_index, 0].mean()
                    ),
                    "predicted_foreground_fraction": float(predicted.mean()),
                    "target_foreground_fraction": float(target.mean()),
                    **metrics,
                }
            )
            if prediction_dir is not None:
                output_path = prediction_dir / f"{sample_id}.png"
                if not cv2.imwrite(str(output_path), predicted.astype(np.uint8) * 255):
                    raise OSError(f"Failed to save prediction {output_path}")
    return loss_total / sample_count, pd.DataFrame(rows)


def metric_means(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        f"macro_{name}": float(metrics[name].mean())
        for name in ("dice", "iou", "precision", "recall")
    }


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def train_fold(
    fold: int,
    project_root: Path,
    coarse_manifest: pd.DataFrame,
    cv_manifest: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    training_config = config["training"]
    model_config = config["model"]
    fold_seed = int(training_config["random_seed"]) + fold
    seed_everything(fold_seed)

    train_records = fold_records(coarse_manifest, cv_manifest, fold, "train")
    validation_records = fold_records(
        coarse_manifest, cv_manifest, fold, "validation"
    )
    test_records = fold_records(coarse_manifest, cv_manifest, fold, "test")
    split_sets = [
        set(frame["sample_id"])
        for frame in (train_records, validation_records, test_records)
    ]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise ValueError(f"Image leakage detected in fold {fold}")

    normalization = calculate_normalization(train_records, project_root)
    positive_weight = calculate_positive_weight(train_records)
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

    for epoch in range(1, int(training_config["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        validation_loss, validation_metrics = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
            threshold,
            fold,
            "validation",
        )
        validation_means = metric_means(validation_metrics)
        validation_dice = validation_means["macro_dice"]
        scheduler.step(validation_dice)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "outer_fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{f"validation_{key}": value for key, value in validation_means.items()},
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
                },
                best_path,
            )
        else:
            stale_epochs += 1

        print(
            f"fold={fold} epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_dice={validation_dice:.4f} "
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
        },
        last_path,
    )

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_validation_loss, best_validation_metrics = evaluate(
        model,
        validation_loader,
        loss_function,
        device,
        threshold,
        fold,
        "validation",
    )
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        loss_function,
        device,
        threshold,
        fold,
        "test",
        prediction_dir=output_dir / "predicted_masks" / f"fold_{fold}",
    )
    test_metrics = test_metrics.merge(
        test_records[["sample_id", "file_name", "source_regime"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    validation_means = metric_means(best_validation_metrics)
    test_means = metric_means(test_metrics)
    test_micro = micro_average(test_metrics.to_dict("records"))
    fold_summary: dict[str, object] = {
        "outer_fold": fold,
        "random_seed": fold_seed,
        "train_images": len(train_records),
        "validation_images": len(validation_records),
        "test_images": len(test_records),
        "normalization_mean": normalization.mean,
        "normalization_std": normalization.std,
        "positive_weight": positive_weight,
        "best_epoch": best_epoch,
        "epochs_run": len(history_rows),
        "best_validation_loss": best_validation_loss,
        **{f"validation_{key}": value for key, value in validation_means.items()},
        "test_loss": test_loss,
        **{f"test_{key}": value for key, value in test_means.items()},
        **{
            f"test_micro_{key}": value
            for key, value in test_micro.items()
            if key in {"dice", "iou", "precision", "recall"}
        },
        "training_seconds": time.perf_counter() - fold_start,
        "best_checkpoint": str(best_path.relative_to(project_root)),
        "last_checkpoint": str(last_path.relative_to(project_root)),
    }
    return pd.DataFrame(history_rows), fold_summary, test_metrics


def plot_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, history["outer_fold"].nunique()))
    for color, (fold, fold_history) in zip(colors, history.groupby("outer_fold")):
        label = f"Fold {fold}"
        axes[0, 0].plot(
            fold_history["epoch"], fold_history["train_loss"], color=color, label=label
        )
        axes[0, 1].plot(
            fold_history["epoch"],
            fold_history["validation_loss"],
            color=color,
            label=label,
        )
        axes[1, 0].plot(
            fold_history["epoch"],
            fold_history["validation_macro_dice"],
            color=color,
            label=label,
        )
        axes[1, 1].plot(
            fold_history["epoch"],
            fold_history["learning_rate"],
            color=color,
            label=label,
        )
    titles = ["Training loss", "Validation loss", "Validation Dice", "Learning rate"]
    ylabels = ["Weighted BCE + soft Dice", "Weighted BCE + soft Dice", "Dice", "Rate"]
    for axis, title, ylabel in zip(axes.flat, titles, ylabels):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Coarse U-Net training across five image-level folds", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_example_predictions(
    per_image_metrics: pd.DataFrame,
    coarse_manifest: pd.DataFrame,
    project_root: Path,
    prediction_root: Path,
    output_path: Path,
) -> None:
    examples = []
    for _, group in per_image_metrics.groupby("outer_fold"):
        median = group["dice"].median()
        selected = group.assign(distance=(group["dice"] - median).abs()).sort_values(
            ["distance", "sample_id"]
        ).iloc[0]
        examples.append(selected)

    fig, axes = plt.subplots(len(examples), 3, figsize=(13, 15), constrained_layout=True)
    paths = coarse_manifest.set_index("sample_id")
    for row_index, row in enumerate(examples):
        path_row = paths.loc[row["sample_id"]]
        image = cv2.imread(
            str(project_root / path_row["coarse_image_path"]), cv2.IMREAD_GRAYSCALE
        )
        target = cv2.imread(
            str(project_root / path_row["coarse_mask_path"]), cv2.IMREAD_GRAYSCALE
        )
        prediction = cv2.imread(
            str(
                prediction_root
                / f"fold_{int(row['outer_fold'])}"
                / f"{row['sample_id']}.png"
            ),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None or target is None or prediction is None:
            raise FileNotFoundError(f"Missing example data for {row['sample_id']}")
        axes[row_index, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 0].set_title(
            f"Fold {int(row['outer_fold'])}: {row['sample_id']}\nCoarse SEM (512 x 384)",
            fontsize=9,
        )
        axes[row_index, 1].imshow(target > 0, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title("Ground-truth M-A mask", fontsize=9)
        axes[row_index, 2].imshow(prediction > 0, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].set_title(
            f"Predicted mask\nDice {row['dice']:.3f} | IoU {row['iou']:.3f}",
            fontsize=9,
        )
        for axis in axes[row_index]:
            axis.axis("off")
    fig.suptitle(
        "Held-out coarse U-Net predictions (median-Dice example from each fold)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def aggregate_summary(per_image: pd.DataFrame) -> dict[str, object]:
    macro = metric_means(per_image)
    micro = micro_average(per_image.to_dict("records"))
    return {
        "test_images": int(len(per_image)),
        "prediction_threshold": float(per_image["threshold"].iloc[0]),
        "macro_mean": {
            name: macro[f"macro_{name}"]
            for name in ("dice", "iou", "precision", "recall")
        },
        "macro_standard_deviation": {
            name: float(per_image[name].std(ddof=1))
            for name in ("dice", "iou", "precision", "recall")
        },
        "pixel_micro": {
            name: float(micro[name])
            for name in ("dice", "iou", "precision", "recall")
        },
        "confusion_counts": {
            name: int(micro[name])
            for name in (
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
            )
        },
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_json(config_path)
    dataset_config = config["dataset"]
    training_config = config["training"]
    output_dir = project_root / config["output_dir"]
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
    coarse_manifest_path = project_root / dataset_config["coarse_dir"] / "coarse_manifest.csv"
    if not coarse_manifest_path.is_file():
        raise FileNotFoundError(
            f"{coarse_manifest_path} does not exist; run prepare_coarse_data.py first"
        )
    coarse_manifest = pd.read_csv(coarse_manifest_path)
    cv_manifest = pd.read_csv(project_root / dataset_config["cv_manifest"])
    folds = args.folds if args.folds is not None else training_config["folds"]
    if not folds:
        raise ValueError("At least one fold is required")

    print(f"device={device} torch={torch.__version__} folds={folds}", flush=True)
    histories: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    test_metrics: list[pd.DataFrame] = []
    experiment_start = time.perf_counter()
    for fold in folds:
        history, fold_summary, fold_test = train_fold(
            int(fold),
            project_root,
            coarse_manifest,
            cv_manifest,
            config,
            output_dir,
            device,
        )
        histories.append(history)
        fold_summaries.append(fold_summary)
        test_metrics.append(fold_test)
        history.to_csv(output_dir / "history" / f"fold_{fold}_history.csv", index=False)

    history_frame = pd.concat(histories, ignore_index=True)
    fold_frame = pd.DataFrame(fold_summaries).sort_values("outer_fold")
    test_frame = pd.concat(test_metrics, ignore_index=True).sort_values(
        ["outer_fold", "sample_id"]
    )
    history_frame.to_csv(output_dir / "history" / "all_folds_history.csv", index=False)
    fold_frame.to_csv(output_dir / "metrics" / "fold_metrics.csv", index=False)
    test_frame.to_csv(output_dir / "metrics" / "per_image_test_metrics.csv", index=False)

    summary = aggregate_summary(test_frame)
    summary.update(
        {
            "experiment_name": config["experiment_name"],
            "completed_folds": [int(value) for value in sorted(folds)],
            "cross_validation": "five-fold image-level out-of-fold evaluation",
            "model_parameters": int(sum(p.numel() for p in make_model(config["model"]).parameters())),
            "coarse_dimensions": [
                int(dataset_config["target_width"]),
                int(dataset_config["target_height"]),
            ],
            "device": str(device),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "total_training_seconds": time.perf_counter() - experiment_start,
            "config_path": str(config_path.relative_to(project_root)),
        }
    )
    with (output_dir / "metrics" / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_training_curves(history_frame, output_dir / "figures" / "training_curves.png")
    plot_example_predictions(
        test_frame,
        coarse_manifest,
        project_root,
        output_dir / "predicted_masks",
        output_dir / "figures" / "example_predictions.png",
    )
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
