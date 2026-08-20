"""Monte Carlo dropout inference with batch normalization kept frozen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MCDropoutPrediction:
    """Pixel-wise statistics from repeated stochastic forward passes."""

    mean_probability: np.ndarray
    predictive_entropy_bits: np.ndarray
    predictive_variance: np.ndarray
    dropout_modules_enabled: int


def enable_mc_dropout(model: nn.Module) -> int:
    """Freeze deterministic layers and enable only dropout modules."""

    model.eval()
    enabled = 0
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train()
            enabled += 1
    return enabled


def predictive_entropy_bits(probability: np.ndarray) -> np.ndarray:
    """Binary entropy of a foreground probability map, measured in bits."""

    values = np.asarray(probability, dtype=np.float32)
    clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
    entropy = -(
        clipped * np.log2(clipped)
        + (1.0 - clipped) * np.log2(1.0 - clipped)
    )
    return entropy.astype(np.float32, copy=False)


@torch.inference_mode()
def mc_dropout_predict(
    model: nn.Module,
    image: torch.Tensor,
    passes: int,
) -> MCDropoutPrediction:
    """Run repeated stochastic passes without changing non-dropout state."""

    if passes < 2:
        raise ValueError("MC-dropout inference requires at least two passes")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected one BCHW image, received {tuple(image.shape)}")

    dropout_modules = enable_mc_dropout(model)
    if dropout_modules == 0:
        raise ValueError("The model contains no dropout module to sample")

    samples = []
    for _ in range(passes):
        probability = torch.sigmoid(model(image))[0, 0]
        samples.append(probability.detach().cpu().numpy().astype(np.float32))
    stacked = np.stack(samples, axis=0)
    mean_probability = stacked.mean(axis=0, dtype=np.float32)
    predictive_variance = stacked.var(axis=0, dtype=np.float32)
    return MCDropoutPrediction(
        mean_probability=mean_probability,
        predictive_entropy_bits=predictive_entropy_bits(mean_probability),
        predictive_variance=predictive_variance,
        dropout_modules_enabled=dropout_modules,
    )
