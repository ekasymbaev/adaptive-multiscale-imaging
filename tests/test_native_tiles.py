from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from adaptive_multiscale.data.coarse import NormalizationStatistics
from adaptive_multiscale.data.native_tiles import (
    NativeTileDataset,
    build_native_tile_manifest,
    calculate_tile_positive_weight,
    fold_tile_records,
)


class NativeTileTests(unittest.TestCase):
    def test_manifest_dataset_and_split_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.arange(512 * 512, dtype=np.uint32).reshape(512, 512) % 256
            image = image.astype(np.uint8)
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[:256, :256] = 255
            Image.fromarray(image, mode="L").save(root / "image.tif")
            self.assertTrue(cv2.imwrite(str(root / "mask.png"), mask))
            images = pd.DataFrame(
                [
                    {
                        "sample_id": "ma_test",
                        "file_name": "image.tif",
                        "source_regime": "I",
                        "source_image_path": "image.tif",
                        "source_mask_path": "mask.png",
                    }
                ]
            )
            manifest = build_native_tile_manifest(images, root, 256, 512, 512)
            self.assertEqual(len(manifest), 4)
            self.assertEqual(manifest["foreground_pixels"].tolist(), [65536, 0, 0, 0])
            cv = pd.DataFrame(
                [
                    {
                        "outer_fold": 0,
                        "sample_id": "ma_test",
                        "file_name": "image.tif",
                        "source_regime": "I",
                        "split": "train",
                    }
                ]
            )
            records = fold_tile_records(manifest, cv, 0, "train", 4)
            self.assertEqual(set(records["sample_id"]), {"ma_test"})
            self.assertAlmostEqual(calculate_tile_positive_weight(records), 3.0)
            dataset = NativeTileDataset(
                records,
                root,
                NormalizationStatistics(mean=0.5, std=0.25),
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (1, 256, 256))
            self.assertEqual(tuple(sample["mask"].shape), (1, 256, 256))
            self.assertEqual(sample["tile_index"], 0)

    def test_fold_records_rejects_incomplete_images(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "sample_id": "a",
                    "file_name": "a.tif",
                    "source_regime": "I",
                    "tile_index": 0,
                }
            ]
        )
        cv = pd.DataFrame(
            [
                {
                    "outer_fold": 0,
                    "sample_id": "a",
                    "file_name": "a.tif",
                    "source_regime": "I",
                    "split": "test",
                }
            ]
        )
        with self.assertRaises(ValueError):
            fold_tile_records(manifest, cv, 0, "test", 48)


if __name__ == "__main__":
    unittest.main()
