#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paired RGB + FS manifest from datasets/."
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=Path("datasets/images"),
        help="Root directory of paired RGB images.",
    )
    parser.add_argument(
        "--fs-root",
        type=Path,
        default=Path("datasets/thz_seed_only/FS"),
        help="Root directory of paired FS CSV files.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/ppfnet_stage2/manifests/rgb_fs_pairs.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def build_rgb_map(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        key = "{0}/{1}".format(path.parent.name, path.stem)
        mapping[key] = path
    return mapping


def build_fs_map(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(root.rglob("*_Alignabs.csv")):
        key = "{0}/{1}".format(path.parent.name, path.stem.replace("_Alignabs", ""))
        mapping[key] = path
    return mapping


def main() -> int:
    args = parse_args()
    rgb_map = build_rgb_map(args.rgb_root)
    fs_map = build_fs_map(args.fs_root)

    shared_keys = sorted(set(rgb_map) & set(fs_map))
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    with args.out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "class_name",
                "sample_name",
                "rgb_path",
                "fs_raw_csv_path",
                "pair_key",
                "group_id",
            ],
        )
        writer.writeheader()
        for key in shared_keys:
            class_name, sample_name = key.split("/", 1)
            writer.writerow(
                {
                    "sample_id": key,
                    "class_name": class_name,
                    "sample_name": sample_name,
                    "rgb_path": str(rgb_map[key]),
                    "fs_raw_csv_path": str(fs_map[key]),
                    "pair_key": key,
                    "group_id": key,
                }
            )

    print("paired_rgb_fs:", len(shared_keys))
    print("output:", args.out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

