#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage1_spectral_unet import SpectralUNet1D
from ppfnet.stage2_spectral_baselines import build_stage2_spectral_baseline
from ppfnet.stage2_rgb_fs_model import RGBConditionedSpectralStudent, sample_local_rgb_features
from ppfnet.stage2_rgb_fs_patch_model import PatchContextResidualStudent
from ppfnet.stage2_tcn_baseline import SpectralTCN1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark parameter count and forward inference speed for PPF-Net checkpoints."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        required=True,
        help="One or more checkpoint paths to benchmark.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Benchmark device. 'auto' selects CUDA when available.",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Synthetic batch size. For patch models this is the number of pixel items; for old RGB student this is the number of image samples.",
    )
    parser.add_argument(
        "--pixels-per-sample",
        type=int,
        default=512,
        help="Synthetic per-image pixel count used for the old RGB+FS student benchmark.",
    )
    parser.add_argument("--amp", action="store_true", help="Use AMP autocast on CUDA during benchmarking.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file for benchmark results.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV file for benchmark results.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_checkpoint_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _infer_family(checkpoint: Dict[str, Any], checkpoint_path: Path) -> str:
    data_info = checkpoint.get("data_info", {})
    args = checkpoint.get("args", {})

    task = str(data_info.get("task", "")).lower()
    model_family = str(data_info.get("model_family", "")).lower()
    conditioning_type = str(data_info.get("conditioning_type", "")).lower()

    if task == "stage2_tcn_baseline" or model_family == "tcn":
        return "stage2_tcn"
    if task in {"stage2_srcnn_baseline", "stage2_dncnn_baseline", "stage2_edsr_baseline"}:
        return task.replace("_baseline", "")
    if model_family in {"srcnn", "dncnn", "edsr"}:
        return "stage2_{0}".format(model_family)
    if task == "stage2_fs_only_baseline":
        return "stage2_fs_only"
    if conditioning_type:
        return "stage2_rgb_fs_old"
    if "rgb_in_channels" in data_info or "rgb_embed_dim" in data_info:
        return "stage2_rgb_fs_old"
    if "rgb_patch_size" in args and "teacher_checkpoint" in args:
        return "stage2_rgb_fs_patch"
    return "stage1_spectral_unet"


def _spectral_length_from_checkpoint(checkpoint: Dict[str, Any]) -> int:
    data_info = checkpoint.get("data_info", {})
    if "spectral_length" in data_info:
        return int(data_info["spectral_length"])
    axis_values = data_info.get("axis_values")
    if axis_values is not None:
        return int(len(axis_values))
    return 273


class LegacySmallRGBEncoderBN(nn.Module):
    def __init__(self, in_channels: int = 6, global_embed_dim: int = 64, local_cond_channels: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )
        self.local_proj = nn.Conv2d(96, local_cond_channels, kernel_size=1)
        self.global_proj = nn.Linear(96, global_embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.stem(x)
        local_feature_map = self.local_proj(feature_map)
        pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
        global_embedding = self.global_proj(pooled)
        return local_feature_map, global_embedding


class LegacyRGBConditionedStudent(nn.Module):
    def __init__(
        self,
        rgb_in_channels: int = 6,
        rgb_embed_dim: int = 64,
        local_cond_channels: int = 16,
        global_cond_channels: int = 16,
        base_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rgb_encoder = LegacySmallRGBEncoderBN(
            in_channels=rgb_in_channels,
            global_embed_dim=rgb_embed_dim,
            local_cond_channels=local_cond_channels,
        )
        self.rgb_to_global_channels = nn.Linear(rgb_embed_dim, global_cond_channels)
        self.backbone = SpectralUNet1D(
            in_channels=3 + 2 + local_cond_channels + global_cond_channels,
            base_channels=base_channels,
            dropout=dropout,
        )

    def forward(
        self,
        masked_model_input: torch.Tensor,
        coords_xy_norm: torch.Tensor,
        rgb_images: torch.Tensor,
        pixel_to_sample_index: torch.Tensor,
    ) -> torch.Tensor:
        local_feature_map, rgb_embedding = self.rgb_encoder(rgb_images)
        spectral_length = masked_model_input.shape[-1]

        pixel_local_features = sample_local_rgb_features(
            local_feature_map,
            coords_xy_norm=coords_xy_norm,
            pixel_to_sample_index=pixel_to_sample_index,
        ).to(dtype=masked_model_input.dtype)
        pixel_global = self.rgb_to_global_channels(rgb_embedding)[pixel_to_sample_index].to(dtype=masked_model_input.dtype)

        local_channels = pixel_local_features.unsqueeze(-1).expand(-1, -1, spectral_length)
        global_channels = pixel_global.unsqueeze(-1).expand(-1, -1, spectral_length)
        coord_channels = coords_xy_norm.to(dtype=masked_model_input.dtype).unsqueeze(-1).expand(-1, -1, spectral_length)
        x = torch.cat([masked_model_input, coord_channels, local_channels, global_channels], dim=1)
        return self.backbone(x)


def build_stage1_or_fs_only_forward(
    checkpoint: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[List[torch.nn.Module], Callable[[], torch.Tensor], Dict[str, Any]]:
    args = checkpoint.get("args", {})
    model_family = str(checkpoint.get("data_info", {}).get("model_family", "unet")).lower()
    spectral_length = _spectral_length_from_checkpoint(checkpoint)

    if model_family == "tcn":
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
    elif model_family in {"srcnn", "dncnn", "edsr"}:
        model = build_stage2_spectral_baseline(
            model_family=model_family,
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
        return model(model_input)

    meta = {
        "synthetic_batch_size": batch_size,
        "total_pixel_spectra": batch_size,
        "spectral_length": spectral_length,
    }
    return [model], forward, meta


def build_old_student_forward(
    checkpoint: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    pixels_per_sample: int,
) -> tuple[List[torch.nn.Module], Callable[[], torch.Tensor], Dict[str, Any]]:
    args = checkpoint.get("args", {})
    data_info = checkpoint.get("data_info", {})
    teacher_checkpoint = torch.load(resolve_checkpoint_path(args["teacher_checkpoint"]), map_location=device)
    teacher_args = teacher_checkpoint.get("args", {})

    teacher = SpectralUNet1D(
        in_channels=3,
        base_channels=int(teacher_args.get("base_channels", 32)),
        dropout=float(teacher_args.get("dropout", 0.0)),
    ).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher.eval()

    use_legacy = "conditioning_type" not in data_info
    if use_legacy:
        student = LegacyRGBConditionedStudent(
            rgb_in_channels=int(data_info.get("rgb_in_channels", 6)),
            rgb_embed_dim=int(data_info.get("rgb_embed_dim", args.get("rgb_embed_dim", 64))),
            local_cond_channels=int(data_info.get("local_cond_channels", args.get("local_cond_channels", 16))),
            global_cond_channels=int(data_info.get("global_cond_channels", args.get("global_cond_channels", 16))),
            base_channels=int(args.get("base_channels", 32)),
            dropout=float(args.get("dropout", 0.0)),
        ).to(device)
    else:
        student = RGBConditionedSpectralStudent(
            rgb_in_channels=int(data_info.get("rgb_in_channels", 6)),
            rgb_embed_dim=int(args.get("rgb_embed_dim", 64)),
            local_cond_channels=int(args.get("local_cond_channels", 16)),
            global_cond_channels=int(args.get("global_cond_channels", 16)),
            base_channels=int(args.get("base_channels", 32)),
            dropout=float(args.get("dropout", 0.0)),
            use_local_rgb_conditioning=bool(data_info.get("use_local_rgb_conditioning", True)),
        ).to(device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()

    image_height, image_width = map(int, args.get("image_size", (224, 224)))
    rgb_channels = int(data_info.get("rgb_in_channels", 6))
    spectral_length = _spectral_length_from_checkpoint(teacher_checkpoint)
    total_pixels = batch_size * pixels_per_sample

    masked_model_input = torch.randn(total_pixels, 3, spectral_length, device=device)
    coords_xy_norm = torch.rand(total_pixels, 2, device=device)
    rgb_images = torch.randn(batch_size, rgb_channels, image_height, image_width, device=device)
    pixel_to_sample_index = torch.arange(batch_size, device=device).repeat_interleave(pixels_per_sample)

    def forward() -> torch.Tensor:
        teacher_prediction = teacher(masked_model_input)
        if use_legacy:
            return student(
                masked_model_input=masked_model_input,
                coords_xy_norm=coords_xy_norm,
                rgb_images=rgb_images,
                pixel_to_sample_index=pixel_to_sample_index,
            )
        return student(
            masked_model_input=masked_model_input,
            coords_xy_norm=coords_xy_norm,
            rgb_images=rgb_images,
            pixel_to_sample_index=pixel_to_sample_index,
            baseline_reconstruction=teacher_prediction,
        )

    meta = {
        "synthetic_batch_size": batch_size,
        "pixels_per_sample": pixels_per_sample,
        "total_pixel_spectra": total_pixels,
        "spectral_length": spectral_length,
        "rgb_height": image_height,
        "rgb_width": image_width,
        "rgb_channels": rgb_channels,
    }
    return [teacher, student], forward, meta


def build_patch_forward(
    checkpoint: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[List[torch.nn.Module], Callable[[], torch.Tensor], Dict[str, Any]]:
    args = checkpoint.get("args", {})
    data_info = checkpoint.get("data_info", {})
    teacher_checkpoint = torch.load(resolve_checkpoint_path(args["teacher_checkpoint"]), map_location=device)
    teacher_args = teacher_checkpoint.get("args", {})

    teacher = SpectralUNet1D(
        in_channels=3,
        base_channels=int(teacher_args.get("base_channels", 32)),
        dropout=float(teacher_args.get("dropout", 0.0)),
    ).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher.eval()

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
    spectral_length = _spectral_length_from_checkpoint(teacher_checkpoint)

    masked_model_input = torch.randn(batch_size, 3, spectral_length, device=device)
    patch_mean = torch.randn(batch_size, 1, spectral_length, device=device)
    patch_std = torch.rand(batch_size, 1, spectral_length, device=device)
    patch_valid_ratio = torch.rand(batch_size, 1, device=device)
    coords_xy_norm = torch.rand(batch_size, 2, device=device)
    rgb_patch = torch.randn(batch_size, rgb_channels, patch_height, patch_width, device=device)

    def forward() -> torch.Tensor:
        teacher_prediction = teacher(masked_model_input)
        return student(
            masked_model_input=masked_model_input,
            patch_mean=patch_mean,
            patch_std=patch_std,
            patch_valid_ratio=patch_valid_ratio,
            coords_xy_norm=coords_xy_norm,
            rgb_patch=rgb_patch,
            baseline_reconstruction=teacher_prediction,
        )

    meta = {
        "synthetic_batch_size": batch_size,
        "total_pixel_spectra": batch_size,
        "spectral_length": spectral_length,
        "rgb_patch_height": patch_height,
        "rgb_patch_width": patch_width,
        "rgb_channels": rgb_channels,
    }
    return [teacher, student], forward, meta


def benchmark_forward(
    forward_fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup_iters: int,
    benchmark_iters: int,
    use_amp: bool,
) -> Dict[str, float]:
    autocast_enabled = use_amp and device.type == "cuda"

    def _run_once() -> None:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                _ = forward_fn()

    for _ in range(max(warmup_iters, 0)):
        _run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(max(benchmark_iters, 1)):
        _run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / max(benchmark_iters, 1) * 1000.0
    return {
        "avg_forward_ms": avg_ms,
        "total_benchmark_sec": elapsed,
        "benchmark_iters": int(max(benchmark_iters, 1)),
    }


def write_json(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    results: List[Dict[str, Any]] = []
    for checkpoint_path in args.checkpoints:
        resolved_checkpoint = resolve_checkpoint_path(checkpoint_path)
        checkpoint = torch.load(resolved_checkpoint, map_location=device)
        family = _infer_family(checkpoint, resolved_checkpoint)

        if family in {"stage1_spectral_unet", "stage2_fs_only", "stage2_tcn", "stage2_srcnn", "stage2_dncnn", "stage2_edsr"}:
            models, forward_fn, meta = build_stage1_or_fs_only_forward(
                checkpoint=checkpoint,
                device=device,
                batch_size=args.batch_size,
            )
        elif family == "stage2_rgb_fs_old":
            models, forward_fn, meta = build_old_student_forward(
                checkpoint=checkpoint,
                device=device,
                batch_size=args.batch_size,
                pixels_per_sample=args.pixels_per_sample,
            )
        elif family == "stage2_rgb_fs_patch":
            models, forward_fn, meta = build_patch_forward(
                checkpoint=checkpoint,
                device=device,
                batch_size=args.batch_size,
            )
        else:
            raise ValueError("Unsupported checkpoint family: {0}".format(family))

        benchmark = benchmark_forward(
            forward_fn=forward_fn,
            device=device,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            use_amp=args.amp,
        )

        teacher_params = 0
        primary_params = 0
        if family in {"stage2_rgb_fs_old", "stage2_rgb_fs_patch"}:
            teacher_params = count_parameters(models[0])
            primary_params = count_parameters(models[1])
        else:
            primary_params = count_parameters(models[0])

        total_params = sum(count_parameters(model) for model in models)
        total_pixels = int(meta.get("total_pixel_spectra", args.batch_size))
        avg_ms = float(benchmark["avg_forward_ms"])

        row: Dict[str, Any] = {
            "checkpoint": str(resolved_checkpoint),
            "family": family,
            "device": str(device),
            "amp": bool(args.amp and device.type == "cuda"),
            "teacher_params": int(teacher_params),
            "primary_params": int(primary_params),
            "total_params": int(total_params),
            "avg_forward_ms": avg_ms,
            "throughput_items_per_sec": float(total_pixels / (avg_ms / 1000.0)),
            **meta,
        }
        results.append(row)
        print(
            "{0}: total_params={1} avg_forward_ms={2:.3f} throughput={3:.2f}/s".format(
                Path(resolved_checkpoint).name,
                row["total_params"],
                row["avg_forward_ms"],
                row["throughput_items_per_sec"],
            )
        )

    if args.output_json is not None:
        write_json(args.output_json, results)
    if args.output_csv is not None:
        write_csv(args.output_csv, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

