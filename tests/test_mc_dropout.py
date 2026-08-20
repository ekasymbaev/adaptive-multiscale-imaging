from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from adaptive_multiscale.models import CompactUNet
from adaptive_multiscale.uncertainty.evaluation import (
    local_region_correlations,
    local_region_table,
    top_uncertainty_concentration,
    uncertainty_error_statistics,
)
from adaptive_multiscale.uncertainty.mc_dropout import (
    enable_mc_dropout,
    mc_dropout_predict,
    predictive_entropy_bits,
)


class MCDropoutTests(unittest.TestCase):
    def test_predictive_entropy_has_expected_limits(self) -> None:
        values = predictive_entropy_bits(np.array([0.0, 0.5, 1.0]))
        self.assertLess(values[0], 1e-5)
        self.assertAlmostEqual(float(values[1]), 1.0, places=6)
        self.assertLess(values[2], 1e-5)

    def test_only_dropout_is_reenabled(self) -> None:
        model = CompactUNet(bottleneck_dropout=0.5)
        enabled = enable_mc_dropout(model)
        self.assertEqual(enabled, 1)
        self.assertTrue(model.bottleneck_dropout.training)
        batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))

    def test_mc_predictions_have_nonzero_variance(self) -> None:
        torch.manual_seed(10)
        model = CompactUNet(
            encoder_channels=(4, 8, 16, 32), bottleneck_dropout=0.5
        )
        prediction = mc_dropout_predict(model, torch.randn(1, 1, 32, 32), passes=8)
        self.assertEqual(prediction.mean_probability.shape, (32, 32))
        self.assertGreater(float(prediction.predictive_variance.max()), 0.0)
        self.assertEqual(prediction.dropout_modules_enabled, 1)

    def test_uncertainty_statistics_reward_aligned_errors(self) -> None:
        uncertainty = np.array([[0.9, 0.8], [0.2, 0.1]], dtype=np.float32)
        error = np.array([[1, 1], [0, 0]], dtype=bool)
        statistics = uncertainty_error_statistics(uncertainty, error)
        concentration = top_uncertainty_concentration(uncertainty, error, [0.5])
        self.assertGreater(statistics["error_pearson"], 0.9)
        self.assertEqual(concentration["top_50pct_error_capture"], 1.0)
        self.assertEqual(concentration["top_50pct_error_enrichment"], 2.0)

    def test_local_region_grid_and_correlations(self) -> None:
        entropy = np.zeros((8, 8), dtype=np.float32)
        entropy[:4, :4] = 1.0
        variance = entropy * 0.1
        error = entropy > 0
        regions = local_region_table(
            entropy, variance, error, region_size=4, metadata={"sample_id": "test"}
        )
        self.assertEqual(len(regions), 4)
        correlations = local_region_correlations(regions)
        self.assertGreater(correlations["region_entropy_error_pearson"], 0.99)


if __name__ == "__main__":
    unittest.main()
