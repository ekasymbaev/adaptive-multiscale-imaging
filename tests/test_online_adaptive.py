from __future__ import annotations

import unittest

import numpy as np
import torch

from adaptive_multiscale.fusion import (
    fuse_probability_tiles,
    infer_selected_tile_probabilities,
    timed_call,
    validate_selected_indices,
)


class IdentityLogitModel(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class OnlineAdaptiveTests(unittest.TestCase):
    def test_online_fusion_changes_only_supplied_tiles(self) -> None:
        coarse = np.full((8, 8), 0.2, dtype=np.float32)
        fine_tiles = {1: np.full((4, 4), 0.9, dtype=np.float32)}
        fused = fuse_probability_tiles(
            coarse, fine_tiles, tile_size=4, blend_width=1, rule="hard"
        )
        np.testing.assert_allclose(fused[:4, :4], 0.2)
        np.testing.assert_allclose(fused[:4, 4:], 0.9)
        np.testing.assert_allclose(fused[4:, :], 0.2)

    def test_selected_inference_returns_no_unselected_probabilities(self) -> None:
        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        result = infer_selected_tile_probabilities(
            IdentityLogitModel(),
            image,
            selected_tile_indices=[3, 0],
            normalization_mean=0.0,
            normalization_std=1.0,
            tile_size=4,
            batch_size=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(set(result.probabilities), {0, 3})
        self.assertEqual(result.tiles_processed, 2)
        self.assertEqual(result.batches_executed, 2)
        self.assertEqual(result.probabilities[0].shape, (4, 4))

    def test_selection_validation_rejects_duplicates_and_bounds(self) -> None:
        with self.assertRaises(ValueError):
            validate_selected_indices([1, 1], tile_count=4)
        with self.assertRaises(ValueError):
            validate_selected_indices([4], tile_count=4)
        with self.assertRaises(ValueError):
            validate_selected_indices([], tile_count=4)

    def test_cpu_timing_returns_value_and_nonnegative_duration(self) -> None:
        value, seconds = timed_call(lambda: 7, torch.device("cpu"))
        self.assertEqual(value, 7)
        self.assertGreaterEqual(seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
