#!/usr/bin/env python
"""
Generate train / val / test splits from a manifest CSV.

Default target:

- input : outputs/ppfnet_stage1/manifests/fs_ts_pairs.csv
- output: outputs/ppfnet_stage1/splits

Key design choice:

- Splits are created at the group level, not directly at the row level.
- By default, names ending with `_seedNNN` are grouped together so that
  augmented variants of the same base sample do not leak across splits.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_GROUP_STRIP_REGEX = r"_seed\d+$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train / val / test splits from a manifest CSV."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("outputs/ppfnet_stage1/manifests/fs_ts_pairs.csv"),
        help="Input manifest CSV to split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ppfnet_stage1/splits"),
        help="Directory where split files will be written.",
    )
    parser.add_argument(
        "--id-column",
        default="pair_id",
        help="Column written to train.txt / val.txt / test.txt.",
    )
    parser.add_argument(
        "--group-column",
        default="sample_name",
        help="Column used to derive the grouping key before splitting.",
    )
    parser.add_argument(
        "--class-column",
        default="class_name",
        help="Optional class column used for stratified splitting.",
    )
    parser.add_argument(
        "--group-strip-regex",
        default=DEFAULT_GROUP_STRIP_REGEX,
        help="Regex removed from the end of group-column values before grouping. "
        "Use an empty string to disable stripping.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio. Default: 0.7",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio. Default: 0.15",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio. Default: 0.15",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError("Input manifest not found: {0}".format(path))

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("Input manifest is empty: {0}".format(path))
    return rows


def validate_columns(rows: Sequence[Dict[str, str]], required: Sequence[str]) -> None:
    header = set(rows[0].keys())
    missing = [name for name in required if name not in header]
    if missing:
        raise KeyError("Manifest is missing required columns: {0}".format(", ".join(missing)))


def normalize_group_value(value: str, strip_regex: str) -> str:
    if strip_regex:
        return re.sub(strip_regex, "", value)
    return value


def compute_split_counts(total: int, ratios: Sequence[float]) -> List[int]:
    if total <= 0:
        return [0 for _ in ratios]

    raw = [ratio * total for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)

    order = sorted(
        range(len(ratios)),
        key=lambda idx: (raw[idx] - counts[idx], ratios[idx]),
        reverse=True,
    )

    for idx in order[:remainder]:
        counts[idx] += 1

    positive_targets = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(positive_targets):
        zeros = [idx for idx in positive_targets if counts[idx] == 0]
        for idx in zeros:
            donor = max(positive_targets, key=lambda j: counts[j])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[idx] += 1

    return counts


def assign_groups(
    group_to_class: Dict[str, str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    ratios = [train_ratio, val_ratio, test_ratio]
    split_names = ["train", "val", "test"]

    class_to_groups: Dict[str, List[str]] = defaultdict(list)
    for group_id, class_name in group_to_class.items():
        class_to_groups[class_name].append(group_id)

    rng = random.Random(seed)
    assignments: Dict[str, str] = {}

    for class_name, groups in sorted(class_to_groups.items()):
        groups = list(groups)
        rng.shuffle(groups)
        counts = compute_split_counts(len(groups), ratios)

        start = 0
        for split_name, count in zip(split_names, counts):
            for group_id in groups[start:start + count]:
                assignments[group_id] = split_name
            start += count

    return assignments


def write_txt(path: Path, values: Sequence[str]) -> None:
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0, got {0}".format(total_ratio))

    rows = load_rows(args.input_manifest)
    validate_columns(rows, [args.id_column, args.group_column, args.class_column])

    group_to_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    group_to_class: Dict[str, str] = {}

    for row in rows:
        group_id = normalize_group_value(row[args.group_column], args.group_strip_regex)
        class_name = row.get(args.class_column, "")
        group_to_rows[group_id].append(row)

        if group_id in group_to_class and group_to_class[group_id] != class_name:
            raise ValueError(
                "Group {0!r} appears under multiple classes: {1!r} and {2!r}".format(
                    group_id, group_to_class[group_id], class_name
                )
            )
        group_to_class[group_id] = class_name

    assignments = assign_groups(
        group_to_class=group_to_class,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    assignment_rows: List[Dict[str, str]] = []

    for row in rows:
        group_id = normalize_group_value(row[args.group_column], args.group_strip_regex)
        split_name = assignments[group_id]
        row_with_split = dict(row)
        row_with_split["group_id"] = group_id
        row_with_split["split"] = split_name
        split_rows[split_name].append(row_with_split)

    for group_id, split_name in sorted(assignments.items()):
        assignment_rows.append(
            {
                "group_id": group_id,
                "class_name": group_to_class[group_id],
                "split": split_name,
                "row_count": str(len(group_to_rows[group_id])),
            }
        )

    base_fieldnames = list(rows[0].keys()) + ["group_id", "split"]

    write_csv(output_dir / "split_assignments.csv", assignment_rows, ["group_id", "class_name", "split", "row_count"])
    write_csv(output_dir / "train_pairs.csv", split_rows["train"], base_fieldnames)
    write_csv(output_dir / "val_pairs.csv", split_rows["val"], base_fieldnames)
    write_csv(output_dir / "test_pairs.csv", split_rows["test"], base_fieldnames)

    for split_name in ("train", "val", "test"):
        ids = [row[args.id_column] for row in split_rows[split_name]]
        write_txt(output_dir / "{0}.txt".format(split_name), ids)

    train_groups = sorted(group_id for group_id, split_name in assignments.items() if split_name == "train")
    val_groups = sorted(group_id for group_id, split_name in assignments.items() if split_name == "val")
    test_groups = sorted(group_id for group_id, split_name in assignments.items() if split_name == "test")

    write_txt(output_dir / "train_groups.txt", train_groups)
    write_txt(output_dir / "val_groups.txt", val_groups)
    write_txt(output_dir / "test_groups.txt", test_groups)

    print("Rows:", len(rows))
    print("Groups:", len(assignments))
    print("Train groups:", len(train_groups), "Train rows:", len(split_rows["train"]))
    print("Val groups:", len(val_groups), "Val rows:", len(split_rows["val"]))
    print("Test groups:", len(test_groups), "Test rows:", len(split_rows["test"]))
    print("Split directory:", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

