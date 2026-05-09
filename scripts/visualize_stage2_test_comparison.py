#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage1_spectral_unet import SpectralUNet1D, build_spectral_masked_inputs
from ppfnet.stage2_rgb_fs_patch_dataset import Stage2RGBFSPatchDataset
from ppfnet.stage2_rgb_fs_patch_model import PatchContextResidualStudent
from ppfnet.stage2_tcn_baseline import SpectralTCN1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize test-set spectral reconstruction comparison.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/ppfnet_stage2/splits/test_pairs.csv"),
        help="Test manifest CSV.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("outputs/ppfnet_stage2_fs_only_baseline/checkpoints/stage2_fs_only_baseline_best.pt"),
        help="FS-only baseline checkpoint.",
    )
    parser.add_argument(
        "--patch-checkpoint",
        type=Path,
        default=Path("outputs/ppfnet_stage2_rgb_fs_patch_student/checkpoints/stage2_rgb_fs_patch_student_best.pt"),
        help="RGB+FS patch checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage2_visual_compare"),
        help="Output directory for figures.",
    )
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--mask-mode", choices=["point", "band", "hybrid"], default="hybrid")
    parser.add_argument("--min-observed-ratio", type=float, default=0.25)
    parser.add_argument("--max-observed-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataset(manifest: Path, patch_checkpoint: dict) -> Stage2RGBFSPatchDataset:
    patch_args = patch_checkpoint.get("args", {})
    return Stage2RGBFSPatchDataset(
        manifest_csv=manifest,
        image_size=tuple(patch_args.get("image_size", (224, 224))),
        rgb_patch_size=tuple(patch_args.get("rgb_patch_size", (64, 64))),
        thz_patch_size=int(patch_args.get("thz_patch_size", 7)),
        normalization=patch_args.get("normalization", "none"),
        repo_root=REPO_ROOT,
        max_pixels_per_sample=patch_args.get("max_pixels_per_sample", None),
        pixel_selection_seed=int(patch_args.get("seed", 42)),
        include_structure_channels=True,
    )


def load_baseline_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    data_info = checkpoint.get("data_info", {})
    model_family = str(data_info.get("model_family", "unet")).lower()
    if model_family == "tcn":
        dilations = data_info.get("dilations")
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
    else:
        model = SpectralUNet1D(
            in_channels=3,
            base_channels=int(args.get("base_channels", 32)),
            dropout=float(args.get("dropout", 0.0)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_patch_student(checkpoint_path: Path, device: torch.device) -> tuple[PatchContextResidualStudent, SpectralUNet1D, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    data_info = checkpoint.get("data_info", {})

    teacher_checkpoint_path = Path(args["teacher_checkpoint"])
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
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
    return student, teacher, checkpoint


def plot_sample(
    output_path: Path,
    sample_id: str,
    coord: tuple[int, int],
    axis_values,
    ground_truth,
    masked_input,
    baseline_recon,
    patch_recon,
    observed_mask,
) -> None:
    gt = ground_truth.squeeze()
    masked = masked_input.squeeze()
    baseline = baseline_recon.squeeze()
    patch = patch_recon.squeeze()
    observed = observed_mask.squeeze() > 0.5
    missing = ~observed

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=180)
    ax.plot(axis_values, gt, label="ground_truth", linewidth=2.2, color="#1b4965")
    ax.plot(axis_values, baseline, label="fs_only", linewidth=1.8, color="#8d99ae")
    ax.plot(axis_values, patch, label="rgb_fs_patch", linewidth=2.0, color="#c1121f")
    ax.scatter(axis_values[observed], masked[observed], s=14, label="observed_input", color="#2a9d8f", zorder=3)
    if missing.any():
        ax.scatter(axis_values[missing], patch[missing], s=8, label="patched_points", color="#ffb703", zorder=3)

    ax.set_title("{0} @ (y={1}, x={2})".format(sample_id, coord[0], coord[1]))
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Reflectance Spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_student, teacher, patch_checkpoint = load_patch_student(args.patch_checkpoint, device)
    baseline_model = load_baseline_model(args.baseline_checkpoint, device)
    dataset = build_dataset(args.manifest, patch_checkpoint)

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(args.seed)

    summary: List[Dict[str, object]] = []

    for offset in range(args.num_samples):
        sample_index = args.start_index + offset
        if sample_index >= len(dataset):
            break

        item = dataset[sample_index]
        batch = {
            "center_spectrum": item["center_spectrum"].unsqueeze(0).to(device),
            "axis_values": item["axis_values"].unsqueeze(0).to(device),
            "patch_mean": item["patch_mean"].unsqueeze(0).to(device),
            "patch_std": item["patch_std"].unsqueeze(0).to(device),
            "patch_valid_ratio": item["patch_valid_ratio"].unsqueeze(0).to(device),
            "coord_xy_norm": item["coord_xy_norm"].unsqueeze(0).to(device),
            "rgb_patch": item["rgb_patch"].unsqueeze(0).to(device),
        }

        spectral_batch = {
            "spectrum": batch["center_spectrum"],
            "axis_values": batch["axis_values"],
        }
        masked = build_spectral_masked_inputs(
            spectral_batch,
            mask_mode=args.mask_mode,
            min_observed_ratio=args.min_observed_ratio,
            max_observed_ratio=args.max_observed_ratio,
            use_axis_channel=True,
            generator=generator,
        )

        with torch.no_grad():
            baseline_pred = baseline_model(masked["model_input"])
            baseline_recon = masked["masked_spectrum"] + baseline_pred * (1.0 - masked["observed_mask"])

            teacher_pred = teacher(masked["model_input"])
            teacher_recon = masked["masked_spectrum"] + teacher_pred * (1.0 - masked["observed_mask"])

            patch_residual = patch_student(
                masked_model_input=masked["model_input"],
                patch_mean=batch["patch_mean"],
                patch_std=batch["patch_std"],
                patch_valid_ratio=batch["patch_valid_ratio"],
                coords_xy_norm=batch["coord_xy_norm"],
                rgb_patch=batch["rgb_patch"],
                baseline_reconstruction=teacher_recon,
            )
            patch_recon = teacher_recon + patch_residual * masked["missing_mask"]
            patch_recon = masked["masked_spectrum"] + patch_recon * (1.0 - masked["observed_mask"])

        axis_values = batch["axis_values"][0].detach().cpu().numpy()
        ground_truth = batch["center_spectrum"][0].detach().cpu().numpy()
        masked_input = masked["masked_spectrum"][0].detach().cpu().numpy()
        baseline_np = baseline_recon[0].detach().cpu().numpy()
        patch_np = patch_recon[0].detach().cpu().numpy()
        observed_mask = masked["observed_mask"][0].detach().cpu().numpy()

        sample_dir = args.output_dir / "test"
        output_path = sample_dir / "{0}_y{1}_x{2}.png".format(
            str(item["sample_id"]).replace("/", "__"),
            int(item["coord_y"]),
            int(item["coord_x"]),
        )
        plot_sample(
            output_path=output_path,
            sample_id=str(item["sample_id"]),
            coord=(int(item["coord_y"]), int(item["coord_x"])),
            axis_values=axis_values,
            ground_truth=ground_truth,
            masked_input=masked_input,
            baseline_recon=baseline_np,
            patch_recon=patch_np,
            observed_mask=observed_mask,
        )

        summary.append(
            {
                "sample_index": sample_index,
                "sample_id": item["sample_id"],
                "coord_y": int(item["coord_y"]),
                "coord_x": int(item["coord_x"]),
                "output_path": str(output_path),
            }
        )

    summary_path = args.output_dir / "visualization_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved_visualizations:", len(summary))
    print("summary_path:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

