#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage1_spectral_unet import SpectralUNet1D
from ppfnet.stage2_rgb_fs_patch_model import PatchContextResidualStudent
from ppfnet.stage2_spectral_baselines import build_stage2_spectral_baseline
from ppfnet.stage2_tcn_baseline import SpectralTCN1D


DEFAULT_MODELS = [
    {
        "label": "SRCNN",
        "checkpoint": "outputs/stage2_srcnn_baseline/checkpoints/stage2_srcnn_baseline_best.pt",
        "family": "srcnn",
    },
    {
        "label": "DnCNN",
        "checkpoint": "outputs/stage2_dncnn_baseline/checkpoints/stage2_dncnn_baseline_best.pt",
        "family": "dncnn",
    },
    {
        "label": "EDSR",
        "checkpoint": "outputs/stage2_edsr_baseline/checkpoints/stage2_edsr_baseline_best.pt",
        "family": "edsr",
    },
    {
        "label": "TCN",
        "checkpoint": "outputs/stage2_tcn_baseline/checkpoints/stage2_tcn_baseline_best.pt",
        "family": "tcn",
    },
    {
        "label": "Single-Modal THz Baseline",
        "checkpoint": "outputs/stage2_fs_only_baseline_random/checkpoints/stage2_fs_only_baseline_best.pt",
        "family": "fs_only",
    },
    {
        "label": "PPF-Net (Obs. 70%)",
        "checkpoint": "outputs/stage2_rgb_fs_patch_student_obs70/checkpoints/stage2_rgb_fs_patch_student_best.pt",
        "family": "ppf_patch_student_only",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated CPU benchmark for comparison models."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/comparison_benchmark_cpu_repeated.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/comparison_benchmark_cpu_repeated.md"),
    )
    return parser.parse_args()


def resolve_checkpoint_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def spectral_length_from_checkpoint(checkpoint: Dict[str, Any]) -> int:
    data_info = checkpoint.get("data_info", {})
    if "spectral_length" in data_info:
        return int(data_info["spectral_length"])
    axis_values = data_info.get("axis_values")
    if axis_values is not None:
        return int(len(axis_values))
    return 273


def benchmark_forward(forward_fn: Callable[[], torch.Tensor], warmup_iters: int, benchmark_iters: int) -> float:
    for _ in range(max(warmup_iters, 0)):
        _ = forward_fn()
    start = time.perf_counter()
    for _ in range(max(benchmark_iters, 1)):
        _ = forward_fn()
    elapsed = time.perf_counter() - start
    return elapsed / max(benchmark_iters, 1) * 1000.0


def build_standard_model(checkpoint: Dict[str, Any], family: str, device: torch.device, batch_size: int) -> tuple[nn.Module, Callable[[], torch.Tensor]]:
    args = checkpoint.get("args", {})
    spectral_length = spectral_length_from_checkpoint(checkpoint)

    if family == "tcn":
        dilations = checkpoint.get("data_info", {}).get("dilations")
        if dilations is None:
            num_blocks = int(args.get("num_blocks", 6))
            dilations = [2**idx for idx in range(max(num_blocks, 1))]
        model = SpectralTCN1D(
            in_channels=3,
            base_channels=int(args.get("base_channels", 32)),
            kernel_size=int(args.get("kernel_size", 3)),
            dilations=[int(value) for value in dilations],
            dropout=float(args.get("dropout", 0.0)),
        ).to(device)
    elif family in {"srcnn", "dncnn", "edsr"}:
        model = build_stage2_spectral_baseline(
            model_family=family,
            in_channels=3,
            base_channels=int(args.get("base_channels", 64)),
            kernel_size=int(args.get("kernel_size", 3)),
            num_blocks=int(args.get("num_blocks", 8)),
            srcnn_bottleneck_channels=int(args.get("srcnn_bottleneck_channels", 32)),
            edsr_res_scale=float(args.get("edsr_res_scale", 0.1)),
        ).to(device)
    else:
        model = SpectralUNet1D(
            in_channels=3,
            base_channels=int(args.get("base_channels", 32)),
            dropout=float(args.get("dropout", 0.0)),
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model_input = torch.randn(batch_size, 3, spectral_length, device=device)

    def forward() -> torch.Tensor:
        with torch.no_grad():
            return model(model_input)

    return model, forward


def build_patch_student_only(checkpoint: Dict[str, Any], device: torch.device, batch_size: int) -> tuple[nn.Module, Callable[[], torch.Tensor]]:
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

    return student, forward


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
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

    for spec in DEFAULT_MODELS:
        checkpoint_path = resolve_checkpoint_path(spec["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        family = str(spec["family"])

        if family == "ppf_patch_student_only":
            model, forward_fn = build_patch_student_only(checkpoint, device, args.batch_size)
        else:
            model, forward_fn = build_standard_model(checkpoint, family, device, args.batch_size)

        runs = [
            benchmark_forward(forward_fn, args.warmup_iters, args.benchmark_iters)
            for _ in range(max(args.repeats, 1))
        ]
        mean_ms = statistics.mean(runs)
        std_ms = statistics.pstdev(runs) if len(runs) > 1 else 0.0
        params = count_parameters(model)

        row = {
            "label": spec["label"],
            "family": family,
            "params": params,
            "params_m": params / 1_000_000.0,
            "avg_forward_ms_mean": mean_ms,
            "avg_forward_ms_std": std_ms,
            "runs_ms": ";".join("{0:.6f}".format(value) for value in runs),
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

    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print("saved:", args.output_csv)
    print("saved:", args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
