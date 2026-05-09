#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet import create_stage1_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one batch from the stage1 THz feature dataloader."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/ppfnet_stage1/splits/train_pairs.csv"),
        help="Manifest CSV to read.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size. Default: 4",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers. Default: 0",
    )
    parser.add_argument(
        "--normalization",
        default="none",
        choices=["none", "zscore", "minmax"],
        help="Per-sample per-channel normalization mode.",
    )
    parser.add_argument(
        "--spatial-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional fixed spatial size applied before collation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataloader = create_stage1_dataloader(
        manifest_csv=args.manifest,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        dataset_kwargs={
            "normalization": args.normalization,
            "spatial_size": tuple(args.spatial_size) if args.spatial_size else None,
        },
    )

    batch = next(iter(dataloader))

    print("manifest:", args.manifest)
    print("batch_size:", len(batch["pair_id"]))
    print("fs_features:", tuple(batch["fs_features"].shape))
    print("ts_features:", tuple(batch["ts_features"].shape))
    print("fs_valid_mask:", tuple(batch["fs_valid_mask"].shape))
    print("ts_valid_mask:", tuple(batch["ts_valid_mask"].shape))
    print("spatial_mask:", tuple(batch["spatial_mask"].shape))
    print("original_sizes:", batch["original_sizes"].tolist())
    print("first_pair_ids:", batch["pair_id"][: min(5, len(batch["pair_id"]))])
    print("fs_feature_names:", batch["fs_feature_names"])
    print("ts_feature_names:", batch["ts_feature_names"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

