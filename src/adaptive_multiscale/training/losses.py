"""Segmentation losses used by the coarse-model experiment."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = torch.sum(probabilities * targets)
    denominator = torch.sum(probabilities) + torch.sum(targets)
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


class WeightedBCEDiceLoss(nn.Module):
    """Weighted binary cross-entropy plus soft Dice loss."""

    def __init__(
        self,
        positive_weight: float,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("Loss component weights must be non-negative and non-zero")
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight)))
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice_smooth = float(dice_smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.positive_weight
        )
        dice = soft_dice_loss(logits, targets, smooth=self.dice_smooth)
        return self.bce_weight * bce + self.dice_weight * dice
