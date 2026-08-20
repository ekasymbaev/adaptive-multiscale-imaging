"""Binary foreground segmentation metrics with explicit confusion counts."""

from __future__ import annotations

import numpy as np


def binary_segmentation_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
    predicted = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError(f"Prediction/target shape mismatch: {predicted.shape}, {truth.shape}")
    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    true_negative = int(np.count_nonzero(~predicted & ~truth))
    return metrics_from_counts(
        true_positive, false_positive, false_negative, true_negative
    )


def metrics_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> dict[str, float | int]:
    epsilon = 1e-12
    dice = (2.0 * true_positive) / (
        2.0 * true_positive + false_positive + false_negative + epsilon
    )
    iou = true_positive / (
        true_positive + false_positive + false_negative + epsilon
    )
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    return {
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def micro_average(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("At least one metric row is required")
    return metrics_from_counts(
        sum(int(row["true_positive"]) for row in rows),
        sum(int(row["false_positive"]) for row in rows),
        sum(int(row["false_negative"]) for row in rows),
        sum(int(row["true_negative"]) for row in rows),
    )
