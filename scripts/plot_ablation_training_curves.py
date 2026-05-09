#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


DEFAULT_EXPERIMENTS = [
    "stage2_rgb_fs_patch_student",
    "stage2_rgb_fs_patch_student_no_structure",
    "stage2_rgb_fs_patch_student_no_teacher_init",
    "stage2_rgb_fs_patch_student_obs30",
    "stage2_rgb_fs_patch_student_obs50",
    "stage2_rgb_fs_patch_student_obs70",
]

DEFAULT_LABELS = {
    "stage2_rgb_fs_patch_student": "Full PPF-Net",
    "stage2_rgb_fs_patch_student_no_structure": "w/o structure prior",
    "stage2_rgb_fs_patch_student_no_teacher_init": "w/o teacher init",
    "stage2_rgb_fs_patch_student_obs30": "Observed ratio 30%",
    "stage2_rgb_fs_patch_student_obs50": "Observed ratio 50%",
    "stage2_rgb_fs_patch_student_obs70": "Observed ratio 70%",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ablation results and plot training curves."
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing experiment folders.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment folder names under outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablation_summary"),
        help="Directory where summary tables and plots will be saved.",
    )
    parser.add_argument(
        "--title",
        default="PPF-Net Ablation Training Curves",
        help="Figure title prefix.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_epoch_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("tag") == "epoch"]


def detect_prefix(logs_dir: Path) -> str:
    candidates = sorted(logs_dir.glob("*_best_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_best_summary.json found in {logs_dir}")
    return candidates[0].name[: -len("_best_summary.json")]


def to_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "experiment",
        "label",
        "best_epoch",
        "val_loss",
        "val_mae",
        "val_rmse",
        "val_psnr",
        "test_loss",
        "test_mae",
        "test_rmse",
        "test_psnr",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = [
        "Experiment",
        "Best Epoch",
        "Val Loss",
        "Val MAE",
        "Val RMSE",
        "Val PSNR",
        "Test Loss",
        "Test MAE",
        "Test RMSE",
        "Test PSNR",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {0} | {1} | {2:.4f} | {3:.4f} | {4:.4f} | {5:.4f} | {6:.4f} | {7:.4f} | {8:.4f} | {9:.4f} |".format(
                row["label"],
                row["best_epoch"],
                row["val_loss"],
                row["val_mae"],
                row["val_rmse"],
                row["val_psnr"],
                row["test_loss"],
                row["test_mae"],
                row["test_rmse"],
                row["test_psnr"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(
    histories: List[Dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    specs = [
        ("loss", "Loss"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("psnr", "PSNR (dB)"),
    ]

    for ax, (metric_key, ylabel) in zip(axes.flat, specs):
        for history in histories:
            epochs = history["epochs"]
            label = history["label"]
            train_values = history["train_" + metric_key]
            val_values = history["val_" + metric_key]
            ax.plot(epochs, train_values, linewidth=1.6, alpha=0.55, label=f"{label} train")
            ax.plot(epochs, val_values, linewidth=2.0, label=f"{label} val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary_rows: List[Dict[str, object]] = []
    histories: List[Dict[str, object]] = []

    for experiment in args.experiments:
        logs_dir = args.outputs_root / experiment / "logs"
        prefix = detect_prefix(logs_dir)
        best_summary_path = logs_dir / f"{prefix}_best_summary.json"
        metrics_csv_path = logs_dir / f"{prefix}_metrics.csv"

        best = load_json(best_summary_path)
        epoch_rows = load_epoch_rows(metrics_csv_path)
        if not epoch_rows:
            raise ValueError(f"No epoch rows found in {metrics_csv_path}")

        label = DEFAULT_LABELS.get(experiment, experiment)
        summary_rows.append(
            {
                "experiment": experiment,
                "label": label,
                "best_epoch": int(best["epoch"]),
                "val_loss": float(best["val_loss"]),
                "val_mae": float(best["val_mae"]),
                "val_rmse": float(best["val_rmse"]),
                "val_psnr": float(best["val_psnr"]),
                "test_loss": float(best["test_loss"]),
                "test_mae": float(best["test_mae"]),
                "test_rmse": float(best["test_rmse"]),
                "test_psnr": float(best["test_psnr"]),
            }
        )

        histories.append(
            {
                "experiment": experiment,
                "label": label,
                "epochs": [int(row["epoch"]) for row in epoch_rows],
                "train_loss": [to_float(row, "train_loss") for row in epoch_rows],
                "val_loss": [to_float(row, "val_loss") for row in epoch_rows],
                "train_mae": [to_float(row, "train_mae") for row in epoch_rows],
                "val_mae": [to_float(row, "val_mae") for row in epoch_rows],
                "train_rmse": [to_float(row, "train_rmse") for row in epoch_rows],
                "val_rmse": [to_float(row, "val_rmse") for row in epoch_rows],
                "train_psnr": [to_float(row, "train_psnr") for row in epoch_rows],
                "val_psnr": [to_float(row, "val_psnr") for row in epoch_rows],
            }
        )

    summary_rows.sort(key=lambda item: item["test_psnr"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(args.output_dir / "ablation_summary.csv", summary_rows)
    write_summary_markdown(args.output_dir / "ablation_summary.md", summary_rows)
    plot_curves(histories, args.output_dir / "ablation_training_curves.png", args.title)

    print("saved:", args.output_dir / "ablation_summary.csv")
    print("saved:", args.output_dir / "ablation_summary.md")
    print("saved:", args.output_dir / "ablation_training_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
