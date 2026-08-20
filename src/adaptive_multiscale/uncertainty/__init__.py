"""Uncertainty estimation utilities for frozen segmentation models."""

from adaptive_multiscale.uncertainty.mc_dropout import (
    MCDropoutPrediction,
    enable_mc_dropout,
    mc_dropout_predict,
    predictive_entropy_bits,
)

__all__ = [
    "MCDropoutPrediction",
    "enable_mc_dropout",
    "mc_dropout_predict",
    "predictive_entropy_bits",
]
