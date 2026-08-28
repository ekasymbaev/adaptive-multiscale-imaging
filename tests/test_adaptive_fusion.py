import unittest

import numpy as np

from adaptive_multiscale.fusion import (
    feather_tile_weight,
    fuse_selected_tiles,
    oracle_gain_rankings,
)


class AdaptiveFusionTests(unittest.TestCase):
    def test_empty_and_hard_selection(self) -> None:
        coarse = np.zeros((8, 8), dtype=np.float32)
        fine = np.ones((8, 8), dtype=np.float32)
        unchanged = fuse_selected_tiles(coarse, fine, [], 4, 1, "hard")
        np.testing.assert_array_equal(unchanged, coarse)

        fused = fuse_selected_tiles(coarse, fine, [1], 4, 1, "hard")
        self.assertTrue(np.all(fused[:4, 4:] == 1.0))
        self.assertTrue(np.all(fused[:4, :4] == 0.0))
        self.assertTrue(np.all(fused[4:, :] == 0.0))

    def test_feathering_tapers_only_internal_edges(self) -> None:
        top_left = feather_tile_weight(8, 2, 0, 0, 2, 2)
        self.assertEqual(float(top_left[0, 0]), 1.0)
        self.assertEqual(float(top_left[-1, -1]), 0.0)
        self.assertEqual(float(top_left[2, 2]), 1.0)

        bottom_right = feather_tile_weight(8, 2, 1, 1, 2, 2)
        self.assertEqual(float(bottom_right[-1, -1]), 1.0)
        self.assertEqual(float(bottom_right[0, 0]), 0.0)

    def test_feathered_fusion_retains_coarse_at_internal_seam(self) -> None:
        coarse = np.zeros((8, 8), dtype=np.float32)
        fine = np.ones((8, 8), dtype=np.float32)
        fused = fuse_selected_tiles(coarse, fine, [0], 4, 1, "feathered")
        self.assertEqual(float(fused[1, 3]), 0.0)
        self.assertEqual(float(fused[1, 1]), 1.0)

    def test_oracle_ranks_the_only_helpful_tile_first(self) -> None:
        coarse = np.zeros((8, 8), dtype=np.float32)
        fine = np.zeros((8, 8), dtype=np.float32)
        target = np.zeros((8, 8), dtype=bool)
        fine[:4, :4] = 1.0
        target[:4, :4] = True
        rankings = oracle_gain_rankings(coarse, fine, target, 4, 1)
        best = rankings.loc[rankings["oracle_gain_rank"] == 1].iloc[0]
        self.assertEqual(int(best["tile_index"]), 0)
        self.assertGreater(int(best["oracle_gain_pixels"]), 0)

    def test_invalid_duplicate_selection_is_rejected(self) -> None:
        probability = np.zeros((8, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            fuse_selected_tiles(probability, probability, [0, 0], 4, 1)


if __name__ == "__main__":
    unittest.main()
