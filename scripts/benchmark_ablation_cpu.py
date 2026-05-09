#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import statistics

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage2_rgb_fs_patch_model import PatchContextResidualStudent


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
        description="Benchmark PPF-Net ablation models on CPU."
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
        help="Experiment folder names to benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Synthetic batch size for CPU benchmarking.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=20,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated benchmark runs used to report mean and std.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablation_summary/cpu_benchmark"),
        help="Directory where benchmark tables will be written.",
    )
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_checkpoint(logs_dir: Path, prefix: str) -> Path:
    best_path = logs_dir.parent / "checkpoints" / f"{prefix}_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_path}")
    return best_path


def detect_prefix(logs_dir: Path) -> str:
    candidates = sorted(logs_dir.glob("*_best_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_best_summary.json found in {logs_dir}")
    return candidates[0].name[: -len("_best_summary.json")]


def spectral_length_from_checkpoint(checkpoint: Dict[str, Any]) -> int:
    data_info = checkpoint.get("data_info", {})
    if "spectral_length" in data_info:
        return int(data_info["spectral_length"])
    axis_values = data_info.get("axis_values")
    if axis_values is not None:
        return int(len(axis_values))
    return 273


def build_patch_models(
    checkpoint: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.nn.Module], callable, Dict[str, Any]]:
    args = checkpoint.get("args", {})
    data_info = checkpoint.get("data_info", {})

    student = PatchContextResidualStudent(
        rgb_in_channels=int(data_info.get("rgb_channels", 6)),
        rgb_embed_dim=int(args.get("rgb_embed_dim", 64)),
        cond_channels=int(args.get("cond_channels", 16)),
        base_channels=int(args.get("base_channels", 32)),
        dropout=float(args.get("dropout", 0.0)),
    ).to(device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()

    patch_height, patch_width = map(int, args.get("rgb_patch_size", (64, 64)))
    rgb_channels = int(data_info.get("rgb_channels", 6))
    spectral_length = 273

    masked_model_input = torch.randn(batch_size, 3, spectral_length, device=device)
    patch_mean = torch.randn(batch_size, 1, spectral_length, device=device)
    patch_std = torch.rand(batch_size, 1, spectral_length, device=device)
    patch_valid_ratio = torch.rand(batch_size, 1, device=device)
    coords_xy_norm = torch.rand(batch_size, 2, device=device)
    rgb_patch = torch.randn(batch_size, rgb_channels, patch_height, patch_width, device=device)
    baseline_reconstruction = torch.randn(batch_size, 1, spectral_length, device=device)

    def forward() -> torch.Tensor:
        with torch.no_grad():
            return student(
                masked_model_input=masked_model_input,
                patch_mean=patch_mean,
                patch_std=patch_std,
                patch_valid_ratio=patch_valid_ratio,
                coords_xy_norm=coords_xy_norm,
                rgb_patch=rgb_patch,
                baseline_reconstruction=baseline_reconstruction,
            )

    meta = {
        "spectral_length": spectral_length,
        "rgb_patch_height": patch_height,
        "rgb_patch_width": patch_width,
        "rgb_channels": rgb_channels,
        "student_params": count_parameters(student),
    }
    return [student], forward, meta


def benchmark_forward(forward_fn, warmup_iters: int, benchmark_iters: int) -> float:
    for _ in range(max(warmup_iters, 0)):
        _ = forward_fn()

    start = time.perf_counter()
    for _ in range(max(benchmark_iters, 1)):
        _ = forward_fn()
    elapsed = time.perf_counter() - start
    return elapsed / max(benchmark_iters, 1) * 1000.0


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "| Method | Params (M) | Inference Time (ms) |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {0} | {1:.3f} | {2:.3f} ± {3:.3f} |".format(
                row["label"],
                row["params_m"],
                row["avg_forward_ms_mean"],
                row["avg_forward_ms_std"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = torch.device("cpu")
    rows: List[Dict[str, object]] = []

    for experiment in args.experiments:
        logs_dir = args.outputs_root / experiment / "logs"
        prefix = detect_prefix(logs_dir)
        checkpoint_path = resolve_checkpoint(logs_dir, prefix)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        models, forward_fn, meta = build_patch_models(checkpoint, device, args.batch_size)
        measurements = [
            benchmark_forward(
                forward_fn=forward_fn,
                warmup_iters=args.warmup_iters,
                benchmark_iters=args.benchmark_iters,
            )
            for _ in range(max(args.repeats, 1))
        ]
        avg_forward_ms_mean = statistics.mean(measurements)
        avg_forward_ms_std = statistics.pstdev(measurements) if len(measurements) > 1 else 0.0

        total_params = sum(count_parameters(model) for model in models)
        row = {
            "experiment": experiment,
            "label": DEFAULT_LABELS.get(experiment, experiment),
            "device": "cpu",
            "batch_size": args.batch_size,
            "warmup_iters": args.warmup_iters,
            "benchmark_iters": args.benchmark_iters,
            "repeats": args.repeats,
            "student_params": int(meta["student_params"]),
            "total_params": int(total_params),
            "params_m": float(meta["student_params"]) / 1_000_000.0,
            "avg_forward_ms_mean": float(avg_forward_ms_mean),
            "avg_forward_ms_std": float(avg_forward_ms_std),
            "avg_forward_ms_runs": ";".join("{0:.6f}".format(value) for value in measurements),
            "spectral_length": int(meta["spectral_length"]),
            "rgb_patch_height": int(meta["rgb_patch_height"]),
            "rgb_patch_width": int(meta["rgb_patch_width"]),
            "rgb_channels": int(meta["rgb_channels"]),
        }
        rows.append(row)
        print(
            "{0}: Params(M)={1:.3f} Inference Time(ms)={2:.3f} ± {3:.3f}".format(
                row["label"],
                row["params_m"],
                row["avg_forward_ms_mean"],
                row["avg_forward_ms_std"],
            )
        )

    rows.sort(key=lambda item: item["avg_forward_ms_mean"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ablation_cpu_benchmark.csv", rows)
    write_markdown(args.output_dir / "ablation_cpu_benchmark.md", rows)
    print("saved:", args.output_dir / "ablation_cpu_benchmark.csv")
    print("saved:", args.output_dir / "ablation_cpu_benchmark.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
