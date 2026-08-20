#!/usr/bin/env python3
"""Create the 512 x 384 coarse M-A island image/mask pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adaptive_multiscale.data.coarse import prepare_coarse_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/coarse_model.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    dataset = config["dataset"]
    statistics = pd.read_csv(project_root / dataset["image_statistics_manifest"])
    output_dir = project_root / dataset["coarse_dir"]
    manifest = prepare_coarse_dataset(
        statistics,
        project_root=project_root,
        output_dir=output_dir,
        target_height=int(dataset["target_height"]),
        target_width=int(dataset["target_width"]),
    )
    manifest_path = output_dir / "coarse_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(
        f"Prepared {len(manifest)} coarse image/mask pairs at "
        f"{dataset['target_width']} x {dataset['target_height']}."
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
