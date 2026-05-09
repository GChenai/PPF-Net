#!/usr/bin/env python
"""
Generate manifest CSV files from exported THz feature-map folders.

Default behavior targets stage 1:

- outputs/ppfnet_stage1/dataset_fs_features
- outputs/ppfnet_stage1/dataset_ts_features
- outputs/ppfnet_stage1/manifests

Generated files:

- fs_all.csv
- ts_all.csv
- fs_ts_pairs.csv

Optional RGB support can also generate:

- rgb_all.csv
- rgb_fs_ts_triplets.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manifest CSV files from THz feature folders."
    )
    parser.add_argument(
        "--fs-root",
        type=Path,
        default=Path("outputs/ppfnet_stage1/dataset_fs_features"),
        help="Root directory of FS feature folders.",
    )
    parser.add_argument(
        "--ts-root",
        type=Path,
        default=Path("outputs/ppfnet_stage1/dataset_ts_features"),
        help="Root directory of TS feature folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/ppfnet_stage1/manifests"),
        help="Directory where manifest CSV files will be written.",
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=None,
        help="Optional RGB root. If provided, rgb_all.csv and rgb_fs_ts_triplets.csv are generated.",
    )
    parser.add_argument(
        "--rgb-exts",
        nargs="*",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        help="Allowed RGB image extensions when --rgb-root is used.",
    )
    return parser.parse_args()


def path_to_posix(path: Path) -> str:
    return path.as_posix()


def normalized_rgb_key(class_name: str, sample_name: str) -> str:
    if sample_name.endswith("_Alignabs"):
        sample_name = sample_name[: -len("_Alignabs")]
    return "{0}/{1}".format(class_name, sample_name) if class_name else sample_name


def iter_feature_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(
        path.parent
        for path in root.rglob("metadata.json")
        if path.is_file()
    )


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def feature_record(root: Path, feature_dir: Path, modality: str) -> Dict[str, object]:
    metadata_path = feature_dir / "metadata.json"
    feature_npz_path = feature_dir / "feature_maps.npz"
    valid_mask_path = feature_dir / "valid_mask.png"

    if not metadata_path.exists():
        raise FileNotFoundError("Missing metadata.json in {0}".format(feature_dir))

    metadata = load_json(metadata_path)
    rel_dir = feature_dir.relative_to(root)

    class_name = rel_dir.parts[0] if len(rel_dir.parts) >= 2 else ""
    sample_name = rel_dir.name
    sample_id = path_to_posix(rel_dir)
    pair_key = sample_id
    rgb_key = normalized_rgb_key(class_name, sample_name)

    cube_shape = metadata.get("cube_shape", [None, None, None])
    height = cube_shape[0] if len(cube_shape) >= 1 else None
    width = cube_shape[1] if len(cube_shape) >= 2 else None
    axis_count = cube_shape[2] if len(cube_shape) >= 3 else metadata.get("axis_count")

    return {
        "modality": modality,
        "sample_id": sample_id,
        "class_name": class_name,
        "sample_name": sample_name,
        "pair_key": pair_key,
        "rgb_key": rgb_key,
        "feature_dir": path_to_posix(feature_dir),
        "feature_npz_path": path_to_posix(feature_npz_path),
        "metadata_path": path_to_posix(metadata_path),
        "valid_mask_path": path_to_posix(valid_mask_path),
        "raw_csv_path": str(metadata.get("source_file", "")),
        "signal_label": metadata.get("signal_label", ""),
        "axis_unit": metadata.get("axis_unit", ""),
        "axis_domain": metadata.get("axis_domain", ""),
        "height": height,
        "width": width,
        "axis_count": axis_count,
        "valid_pixels": metadata.get("valid_pixels"),
        "total_pixels": metadata.get("total_pixels"),
        "valid_ratio": metadata.get("valid_ratio"),
    }


def collect_feature_records(root: Path, modality: str) -> List[Dict[str, object]]:
    if not root.exists():
        raise FileNotFoundError("Feature root does not exist: {0}".format(root))

    records = [feature_record(root, feature_dir, modality) for feature_dir in iter_feature_dirs(root)]
    records.sort(key=lambda item: str(item["sample_id"]))
    return records


def image_record(root: Path, image_path: Path) -> Dict[str, object]:
    rel_path = image_path.relative_to(root)
    class_name = rel_path.parts[0] if len(rel_path.parts) >= 2 else ""
    sample_name = image_path.stem
    sample_id = path_to_posix(rel_path.with_suffix(""))
    rgb_key = "{0}/{1}".format(class_name, sample_name) if class_name else sample_name

    return {
        "rgb_id": sample_id,
        "class_name": class_name,
        "rgb_name": sample_name,
        "rgb_key": rgb_key,
        "rgb_path": path_to_posix(image_path),
    }


def collect_image_records(root: Path, extensions: Sequence[str]) -> List[Dict[str, object]]:
    if not root.exists():
        raise FileNotFoundError("RGB root does not exist: {0}".format(root))

    allowed = {ext.lower() for ext in extensions}
    records = [
        image_record(root, path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]
    records.sort(key=lambda item: str(item["rgb_id"]))
    return records


def unique_by_key(
    records: Sequence[Dict[str, object]],
    key_name: str,
    label: str,
) -> Dict[str, Dict[str, object]]:
    mapping: Dict[str, Dict[str, object]] = {}
    for record in records:
        key = str(record[key_name])
        if key in mapping:
            raise ValueError(
                "Duplicate {0} key {1!r} detected while building manifests.".format(label, key)
            )
        mapping[key] = dict(record)
    return mapping


def pair_fs_ts(
    fs_records: Sequence[Dict[str, object]],
    ts_records: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    fs_map = unique_by_key(fs_records, "pair_key", "FS")
    ts_map = unique_by_key(ts_records, "pair_key", "TS")

    pair_keys = sorted(set(fs_map) & set(ts_map))
    rows: List[Dict[str, object]] = []

    for key in pair_keys:
        fs = fs_map[key]
        ts = ts_map[key]
        rows.append(
            {
                "pair_id": key,
                "class_name": fs["class_name"],
                "sample_name": fs["sample_name"],
                "pair_key": key,
                "rgb_key": fs["rgb_key"],
                "fs_sample_id": fs["sample_id"],
                "ts_sample_id": ts["sample_id"],
                "fs_feature_dir": fs["feature_dir"],
                "ts_feature_dir": ts["feature_dir"],
                "fs_feature_npz_path": fs["feature_npz_path"],
                "ts_feature_npz_path": ts["feature_npz_path"],
                "fs_metadata_path": fs["metadata_path"],
                "ts_metadata_path": ts["metadata_path"],
                "fs_valid_mask_path": fs["valid_mask_path"],
                "ts_valid_mask_path": ts["valid_mask_path"],
                "fs_raw_csv_path": fs["raw_csv_path"],
                "ts_raw_csv_path": ts["raw_csv_path"],
                "signal_label": fs["signal_label"],
                "axis_unit": fs["axis_unit"],
                "axis_domain": fs["axis_domain"],
                "fs_height": fs["height"],
                "fs_width": fs["width"],
                "ts_height": ts["height"],
                "ts_width": ts["width"],
                "axis_count": fs["axis_count"],
                "fs_valid_pixels": fs["valid_pixels"],
                "ts_valid_pixels": ts["valid_pixels"],
                "fs_total_pixels": fs["total_pixels"],
                "ts_total_pixels": ts["total_pixels"],
                "fs_valid_ratio": fs["valid_ratio"],
                "ts_valid_ratio": ts["valid_ratio"],
            }
        )

    return rows


def build_triplets(
    pair_rows: Sequence[Dict[str, object]],
    rgb_records: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rgb_map = unique_by_key(rgb_records, "rgb_key", "RGB")
    rows: List[Dict[str, object]] = []

    for pair in pair_rows:
        rgb_key = str(pair["rgb_key"])
        if rgb_key not in rgb_map:
            continue

        rgb = rgb_map[rgb_key]
        row = dict(pair)
        row.update(
            {
                "triplet_id": pair["pair_id"],
                "rgb_id": rgb["rgb_id"],
                "rgb_name": rgb["rgb_name"],
                "rgb_path": rgb["rgb_path"],
            }
        )
        rows.append(row)

    return rows


def csv_columns(records: Sequence[Dict[str, object]], preferred: Sequence[str]) -> List[str]:
    seen = set()
    columns: List[str] = []

    for name in preferred:
        if any(name in record for record in records):
            columns.append(name)
            seen.add(name)

    for record in records:
        for name in record:
            if name not in seen:
                columns.append(name)
                seen.add(name)

    return columns


def write_csv(path: Path, rows: Sequence[Dict[str, object]], preferred_columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = csv_columns(rows, preferred_columns)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(
    fs_records: Sequence[Dict[str, object]],
    ts_records: Sequence[Dict[str, object]],
    pair_rows: Sequence[Dict[str, object]],
    rgb_records: Optional[Sequence[Dict[str, object]]] = None,
    triplet_rows: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    print("FS records:", len(fs_records))
    print("TS records:", len(ts_records))
    print("FS-TS pairs:", len(pair_rows))

    if rgb_records is not None:
        print("RGB records:", len(rgb_records))
    if triplet_rows is not None:
        print("RGB-FS-TS triplets:", len(triplet_rows))


def main() -> int:
    args = parse_args()

    fs_records = collect_feature_records(args.fs_root, modality="FS")
    ts_records = collect_feature_records(args.ts_root, modality="TS")
    pair_rows = pair_fs_ts(fs_records, ts_records)

    fs_columns = [
        "modality",
        "sample_id",
        "class_name",
        "sample_name",
        "pair_key",
        "rgb_key",
        "feature_dir",
        "feature_npz_path",
        "metadata_path",
        "valid_mask_path",
        "raw_csv_path",
        "signal_label",
        "axis_unit",
        "axis_domain",
        "height",
        "width",
        "axis_count",
        "valid_pixels",
        "total_pixels",
        "valid_ratio",
    ]
    pair_columns = [
        "pair_id",
        "class_name",
        "sample_name",
        "pair_key",
        "rgb_key",
        "fs_sample_id",
        "ts_sample_id",
        "fs_feature_dir",
        "ts_feature_dir",
        "fs_feature_npz_path",
        "ts_feature_npz_path",
        "fs_metadata_path",
        "ts_metadata_path",
        "fs_valid_mask_path",
        "ts_valid_mask_path",
        "fs_raw_csv_path",
        "ts_raw_csv_path",
        "signal_label",
        "axis_unit",
        "axis_domain",
        "fs_height",
        "fs_width",
        "ts_height",
        "ts_width",
        "axis_count",
        "fs_valid_pixels",
        "ts_valid_pixels",
        "fs_total_pixels",
        "ts_total_pixels",
        "fs_valid_ratio",
        "ts_valid_ratio",
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "fs_all.csv", fs_records, fs_columns)
    write_csv(args.out_dir / "ts_all.csv", ts_records, fs_columns)
    write_csv(args.out_dir / "fs_ts_pairs.csv", pair_rows, pair_columns)

    rgb_records: Optional[List[Dict[str, object]]] = None
    triplet_rows: Optional[List[Dict[str, object]]] = None

    if args.rgb_root is not None:
        rgb_records = collect_image_records(args.rgb_root, args.rgb_exts)
        triplet_rows = build_triplets(pair_rows, rgb_records)

        rgb_columns = [
            "rgb_id",
            "class_name",
            "rgb_name",
            "rgb_key",
            "rgb_path",
        ]
        triplet_columns = [
            "triplet_id",
            "rgb_id",
            "rgb_name",
            "rgb_path",
        ] + pair_columns

        write_csv(args.out_dir / "rgb_all.csv", rgb_records, rgb_columns)
        write_csv(args.out_dir / "rgb_fs_ts_triplets.csv", triplet_rows, triplet_columns)

    print_summary(
        fs_records=fs_records,
        ts_records=ts_records,
        pair_rows=pair_rows,
        rgb_records=rgb_records,
        triplet_rows=triplet_rows,
    )
    print("Manifest directory:", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

