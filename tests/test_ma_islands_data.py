from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_multiscale.data.ma_islands import (
    MAImageRecord,
    build_cv_manifests,
    collect_native_records,
    infer_source_regime,
    rasterize_union_mask,
    tile_foreground_counts,
)


class MAIslandsDataTests(unittest.TestCase):
    def test_source_regime_inference(self) -> None:
        self.assertEqual(infer_source_regime("10_400-H-6mm-4000x06.tif"), "I")
        self.assertEqual(infer_source_regime("10_500-H-6mm-4000x06.tif"), "II")
        self.assertEqual(infer_source_regime("10-500-03-H-6mm-4000x07.tif"), "III")
        self.assertEqual(infer_source_regime("10-400-01-H-6mm-4000x06.tif"), "IV")
        with self.assertRaises(ValueError):
            infer_source_regime("unknown.tif")

    def test_rasterization_and_tile_geometry(self) -> None:
        annotation = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "iscrowd": 0,
            "segmentation": [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]],
        }
        record = MAImageRecord(
            sample_id="ma_test",
            file_name="10_400-test.tif",
            image_path=Path("unused.tif"),
            published_split="train",
            coco_image_id=1,
            width=512,
            height=256,
            source_regime="I",
            cooling_group="I+IV",
            annotations=(annotation,),
        )
        mask = rasterize_union_mask(record)
        self.assertEqual(mask.shape, (256, 512))
        self.assertEqual(set(np.unique(mask)), {0, 1})
        tile_counts = tile_foreground_counts(mask, tile_size=256)
        self.assertEqual(tile_counts.shape, (1, 2))
        self.assertGreater(tile_counts[0, 0], 0)
        self.assertEqual(tile_counts[0, 1], 0)

    def test_cv_manifests_are_image_disjoint_and_stratified(self) -> None:
        rows = []
        group_specs = [
            ("I+IV", "I", 10),
            ("II", "II", 10),
            ("III", "III", 10),
            ("I+IV", "IV", 10),
        ]
        index = 1
        for cooling_group, source_regime, count in group_specs:
            for _ in range(count):
                rows.append(
                    {
                        "sample_id": f"ma_{index:03d}",
                        "file_name": f"image_{index:03d}.tif",
                        "source_regime": source_regime,
                        "cooling_group": cooling_group,
                    }
                )
                index += 1
        image_statistics = pd.DataFrame(rows)
        assignments, expanded = build_cv_manifests(
            image_statistics,
            n_splits=5,
            random_seed=2026,
            validation_offset=1,
            stratify_by="source_regime",
        )

        self.assertEqual(len(assignments), 40)
        self.assertEqual(len(expanded), 200)
        for _, fold in expanded.groupby("outer_fold"):
            self.assertEqual(fold["sample_id"].nunique(), 40)
            self.assertEqual(
                fold["split"].value_counts().to_dict(),
                {"train": 24, "validation": 8, "test": 8},
            )
            test_counts = (
                fold[fold["split"] == "test"]["cooling_group"]
                .value_counts()
                .to_dict()
            )
            self.assertEqual(test_counts, {"I+IV": 4, "II": 2, "III": 2})
            regime_counts = (
                fold[fold["split"] == "test"]["source_regime"]
                .value_counts()
                .to_dict()
            )
            self.assertEqual(regime_counts, {"I": 2, "II": 2, "III": 2, "IV": 2})

    def test_downloaded_native_subset_has_40_pairs_when_present(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        native_root = (
            repository
            / "data/raw/ma_islands/extracted/dataset/appendix/i_ii_iii_iv_combined"
        )
        if not native_root.exists():
            self.skipTest("M-A island dataset has not been downloaded")
        records, diagnostics = collect_native_records(native_root)
        self.assertEqual(len(records), 40)
        self.assertGreater(diagnostics["annotation_count"], 0)
        self.assertEqual(
            {record.source_regime for record in records}, {"I", "II", "III", "IV"}
        )


if __name__ == "__main__":
    unittest.main()
