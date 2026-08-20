from __future__ import annotations

import unittest

import numpy as np

from adaptive_multiscale.selection.tile_ranking import (
    TileGrid,
    attach_error_evaluation,
    paired_bootstrap_interval,
    random_rank_matrix,
    selection_metrics,
    uncertainty_tile_rankings,
)


class TileRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = TileGrid(
            native_height=1536,
            native_width=2048,
            coarse_height=384,
            coarse_width=512,
            native_tile_size=256,
        )

    def test_exact_native_to_coarse_geometry(self) -> None:
        self.assertEqual(self.grid.scale_factor, 4)
        self.assertEqual(self.grid.coarse_tile_size, 64)
        self.assertEqual((self.grid.rows, self.grid.columns), (6, 8))
        self.assertEqual(self.grid.tile_count, 48)

    def test_uncertainty_ranking_does_not_require_ground_truth(self) -> None:
        entropy = np.zeros((384, 512), dtype=np.float32)
        variance = np.zeros_like(entropy)
        entropy[:64, :64] = 0.9
        variance[64:128, :64] = 0.2
        rankings = uncertainty_tile_rankings(entropy, variance, self.grid)
        self.assertEqual(len(rankings), 48)
        self.assertEqual(int(rankings.loc[rankings.tile_index == 0, "entropy_rank"].iloc[0]), 1)
        self.assertEqual(int(rankings.loc[rankings.tile_index == 8, "variance_rank"].iloc[0]), 1)
        self.assertFalse(any("error" in column for column in rankings.columns))

    def test_error_capture_uses_exact_budget(self) -> None:
        entropy = np.zeros((384, 512), dtype=np.float32)
        variance = np.zeros_like(entropy)
        entropy[:64, :64] = 0.9
        error = np.zeros_like(entropy, dtype=bool)
        error[:64, :64] = True
        rankings = uncertainty_tile_rankings(entropy, variance, self.grid)
        evaluated = attach_error_evaluation(rankings, error, self.grid)
        metrics = selection_metrics(evaluated, "entropy_rank", [2, 48])
        self.assertEqual(metrics.loc[0, "error_capture"], 1.0)
        self.assertEqual(metrics.loc[0, "coverage_fraction"], 2 / 48)
        self.assertEqual(metrics.loc[1, "error_capture"], 1.0)

    def test_random_rankings_are_reproducible_and_valid(self) -> None:
        first = random_rank_matrix(48, 20, 100, 7)
        second = random_rank_matrix(48, 20, 100, 7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (20, 48))
        self.assertTrue(all(set(row) == set(range(1, 49)) for row in first))

    def test_paired_bootstrap_interval(self) -> None:
        mean, low, high = paired_bootstrap_interval(
            np.array([0.1, 0.2, 0.3, 0.4]), 0.95, 2000, 10
        )
        self.assertAlmostEqual(mean, 0.25)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)


if __name__ == "__main__":
    unittest.main()
