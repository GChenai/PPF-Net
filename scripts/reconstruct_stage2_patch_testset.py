#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage1_spectral_unet import SpectralUNet1D, build_spectral_masked_inputs
from ppfnet.stage2_rgb_fs_patch_dataset import Stage2RGBFSPatchDataset
from ppfnet.stage2_rgb_fs_patch_model import PatchContextResidualStudent
from ppfnet.thz_csv import assemble_cube_from_pixel_spectra, load_thz_csv, resolve_repo_relative_path, save_thz_csv
from ppfnet.thz_imaging import compute_image_map_errors, compute_thz_image_maps, save_image_maps_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct the stage2 test set with the RGB+FS patch model.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/ppfnet_stage2/splits/test_pairs.csv"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/ppfnet_stage2_rgb_fs_patch_student/checkpoints/stage2_rgb_fs_patch_student_best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppfnet_stage2_rgb_fs_patch_student/predictions/test_reconstruction"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-mode", choices=["point", "band", "hybrid"], default="hybrid")
    parser.add_argument("--min-observed-ratio", type=float, default=0.25)
    parser.add_argument("--max-observed-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest CSV is empty: {0}".format(path))
    return rows


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint["args"]
    data_info = checkpoint["data_info"]

    teacher_checkpoint_path = Path(model_args["teacher_checkpoint"])
    teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    teacher_args = teacher_checkpoint["args"]

    teacher = SpectralUNet1D(
        in_channels=3,
        base_channels=int(teacher_args.get("base_channels", 32)),
        dropout=float(teacher_args.get("dropout", 0.0)),
    ).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher.eval()

    student = PatchContextResidualStudent(
        rgb_in_channels=int(data_info.get("rgb_channels", 6)),
        rgb_embed_dim=int(model_args.get("rgb_embed_dim", 64)),
        cond_channels=int(model_args.get("cond_channels", 16)),
        base_channels=int(model_args.get("base_channels", 32)),
        dropout=float(model_args.get("dropout", 0.0)),
    ).to(device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()

    manifest_rows = load_manifest_rows(args.manifest)
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(args.seed)

    for sample_index, row in enumerate(manifest_rows):
        dataset = Stage2RGBFSPatchDataset(
            manifest_csv=args.manifest,
            image_size=tuple(model_args.get("image_size", (224, 224))),
            rgb_patch_size=tuple(model_args.get("rgb_patch_size", (64, 64))),
            thz_patch_size=int(model_args.get("thz_patch_size", 7)),
            normalization=model_args.get("normalization", "none"),
            repo_root=Path.cwd(),
            max_pixels_per_sample=None,
            pixel_selection_seed=int(model_args.get("seed", 42)),
            include_structure_channels=bool(data_info.get("include_structure_channels", True)),
        )

        # Collect only pixels from one sample.
        indices = [idx for idx, (sample_ref, _) in enumerate(dataset.index_map) if sample_ref == sample_index]
        sample_info = dataset.samples[sample_index]
        raw_csv_path = resolve_repo_relative_path(row["fs_raw_csv_path"], Path.cwd())
        cube_data = load_thz_csv(raw_csv_path)

        predicted_spectra = []
        masked_spectra = []
        observed_masks = []
        coords = []

        for start in range(0, len(indices), args.batch_size):
            chunk_indices = indices[start:start + args.batch_size]
            batch_items = [dataset[i] for i in chunk_indices]
            center_spectrum = torch.stack([item["center_spectrum"] for item in batch_items], dim=0).to(device)
            patch_mean = torch.stack([item["patch_mean"] for item in batch_items], dim=0).to(device)
            patch_std = torch.stack([item["patch_std"] for item in batch_items], dim=0).to(device)
            patch_valid_ratio = torch.stack([item["patch_valid_ratio"] for item in batch_items], dim=0).to(device)
            coord_xy_norm = torch.stack([item["coord_xy_norm"] for item in batch_items], dim=0).to(device)
            rgb_patch = torch.stack([item["rgb_patch"] for item in batch_items], dim=0).to(device)
            axis_values = torch.stack([item["axis_values"] for item in batch_items], dim=0).to(device)

            masked = build_spectral_masked_inputs(
                {"spectrum": center_spectrum, "axis_values": axis_values},
                mask_mode=args.mask_mode,
                min_observed_ratio=args.min_observed_ratio,
                max_observed_ratio=args.max_observed_ratio,
                use_axis_channel=True,
                generator=generator,
            )

            with torch.no_grad():
                teacher_prediction = teacher(masked["model_input"])
                teacher_reconstruction = masked["masked_spectrum"] + teacher_prediction * (1.0 - masked["observed_mask"])
                residual = student(
                    masked_model_input=masked["model_input"],
                    patch_mean=patch_mean,
                    patch_std=patch_std,
                    patch_valid_ratio=patch_valid_ratio,
                    coords_xy_norm=coord_xy_norm,
                    rgb_patch=rgb_patch,
                    baseline_reconstruction=teacher_reconstruction,
                )
                reconstruction = teacher_reconstruction + residual * masked["missing_mask"]
                reconstruction = masked["masked_spectrum"] + reconstruction * (1.0 - masked["observed_mask"])

            predicted_spectra.append(reconstruction.squeeze(1).detach().cpu().numpy())
            masked_spectra.append(masked["masked_spectrum"].squeeze(1).detach().cpu().numpy())
            observed_masks.append(masked["observed_mask"].squeeze(1).detach().cpu().numpy())
            coords.extend([(int(item["coord_y"]), int(item["coord_x"])) for item in batch_items])

        predicted_spectra = np.concatenate(predicted_spectra, axis=0)
        masked_spectra = np.concatenate(masked_spectra, axis=0)
        observed_masks = np.concatenate(observed_masks, axis=0)
        coords_array = np.asarray(coords, dtype=np.int32)

        target_cube = cube_data.cube.astype(np.float32)
        reconstructed_cube = assemble_cube_from_pixel_spectra(
            coords=coords_array,
            spectra=predicted_spectra,
            height=target_cube.shape[0],
            width=target_cube.shape[1],
        )
        masked_cube = assemble_cube_from_pixel_spectra(
            coords=coords_array,
            spectra=masked_spectra,
            height=target_cube.shape[0],
            width=target_cube.shape[1],
        )

        sample_id = row["sample_id"]
        sample_dir = args.output_dir / sample_id.replace("/", "__")
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            sample_dir / "reconstruction.npz",
            axis_values=cube_data.axis_values.astype(np.float32),
            valid_mask=cube_data.valid_mask.astype(np.float32),
            target_cube=target_cube,
            masked_cube=masked_cube,
            reconstructed_cube=reconstructed_cube,
            observed_masks=observed_masks.astype(np.float32),
            coords=coords_array,
        )

        header_metadata = dict(cube_data.header_metadata)
        header_metadata["鏂囦欢鍚嶇О"] = str(sample_dir / "reconstructed.imgtds")
        header_metadata["鎴愬儚妯″紡"] = "閲嶅缓鍙嶅皠鍏夎氨"

        save_thz_csv(
            sample_dir / "reconstructed.csv",
            cube=reconstructed_cube,
            axis_values=cube_data.axis_values,
            signal_label=cube_data.signal_label,
            axis_unit=cube_data.axis_unit,
            header_metadata=header_metadata,
        )
        save_thz_csv(
            sample_dir / "target.csv",
            cube=target_cube,
            axis_values=cube_data.axis_values,
            signal_label=cube_data.signal_label,
            axis_unit=cube_data.axis_unit,
            header_metadata=cube_data.header_metadata,
        )

        image_maps_target, _ = compute_thz_image_maps(
            target_cube,
            cube_data.axis_values,
            cube_data.valid_mask,
            slice_values=[1.0, 2.0, 3.0],
            band_ranges=[(0.8, 1.2), (1.8, 2.2)],
        )
        image_maps_recon, _ = compute_thz_image_maps(
            reconstructed_cube,
            cube_data.axis_values,
            cube_data.valid_mask,
            slice_values=[1.0, 2.0, 3.0],
            band_ranges=[(0.8, 1.2), (1.8, 2.2)],
        )
        image_maps_masked, _ = compute_thz_image_maps(
            masked_cube,
            cube_data.axis_values,
            cube_data.valid_mask,
            slice_values=[1.0, 2.0, 3.0],
            band_ranges=[(0.8, 1.2), (1.8, 2.2)],
        )
        image_maps_error = compute_image_map_errors(
            image_maps_target,
            image_maps_recon,
            cube_data.valid_mask,
        )

        save_image_maps_png(image_maps_target, cube_data.valid_mask, sample_dir / "images_target")
        save_image_maps_png(image_maps_masked, cube_data.valid_mask, sample_dir / "images_masked")
        save_image_maps_png(image_maps_recon, cube_data.valid_mask, sample_dir / "images_reconstructed")
        save_image_maps_png(image_maps_error, cube_data.valid_mask, sample_dir / "images_error")

        (sample_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "raw_csv_path": str(raw_csv_path),
                    "checkpoint": str(args.checkpoint),
                    "mask_mode": args.mask_mode,
                    "min_observed_ratio": args.min_observed_ratio,
                    "max_observed_ratio": args.max_observed_ratio,
                    "num_pixels": int(coords_array.shape[0]),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("saved_sample_dir:", sample_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

