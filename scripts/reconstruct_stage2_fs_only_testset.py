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
from ppfnet.stage2_spectral_baselines import build_stage2_spectral_baseline
from ppfnet.stage2_tcn_baseline import SpectralTCN1D
from ppfnet.thz_csv import (
    assemble_cube_from_pixel_spectra,
    extract_valid_pixel_spectra,
    load_thz_csv,
    resolve_repo_relative_path,
    save_thz_csv,
)
from ppfnet.thz_imaging import compute_image_map_errors, compute_thz_image_maps, save_image_maps_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct the stage2 test set with the FS-only baseline.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/ppfnet_stage2/splits/test_pairs.csv"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/ppfnet_stage2_fs_only_baseline/checkpoints/stage2_fs_only_baseline_best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ppfnet_stage2_fs_only_baseline/predictions/test_reconstruction"),
    )
    parser.add_argument("--batch-size", type=int, default=512)
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


def _infer_raw_column(rows: List[Dict[str, str]], modality: str) -> str:
    preferred = "fs_raw_csv_path" if modality == "fs" else "ts_raw_csv_path"
    if preferred in rows[0]:
        return preferred
    if "raw_csv_path" in rows[0]:
        return "raw_csv_path"
    raise KeyError("Could not infer raw CSV column for modality={0}".format(modality))


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint["args"]
    data_info = checkpoint["data_info"]
    modality = str(data_info.get("modality", "fs"))
    normalization = str(data_info.get("normalization", "none"))
    if normalization != "none":
        raise NotImplementedError(
            "CSV export currently expects normalization='none'. "
            "Run the FS-only baseline with --normalization none."
        )

    model_family = str(data_info.get("model_family", "unet")).lower()
    if model_family == "tcn":
        dilations = data_info.get("dilations")
        if dilations is None:
            num_blocks = int(model_args.get("num_blocks", 6))
            dilations = [2**idx for idx in range(max(num_blocks, 1))]
        model = SpectralTCN1D(
            in_channels=3,
            base_channels=int(model_args.get("base_channels", 32)),
            kernel_size=int(model_args.get("kernel_size", 3)),
            dilations=[int(value) for value in dilations],
            dropout=float(model_args.get("dropout", 0.0)),
        ).to(device)
    elif model_family in {"srcnn", "dncnn", "edsr"}:
        model = build_stage2_spectral_baseline(
            model_family=model_family,
            in_channels=3,
            base_channels=int(model_args.get("base_channels", data_info.get("base_channels", 64))),
            kernel_size=int(model_args.get("kernel_size", data_info.get("kernel_size", 3))),
            num_blocks=int(model_args.get("num_blocks", data_info.get("num_blocks", 8))),
            srcnn_bottleneck_channels=int(
                model_args.get("srcnn_bottleneck_channels", data_info.get("srcnn_bottleneck_channels", 32))
            ),
            edsr_res_scale=float(model_args.get("edsr_res_scale", data_info.get("edsr_res_scale", 0.1))),
        ).to(device)
    else:
        model = SpectralUNet1D(
            in_channels=3,
            base_channels=int(model_args.get("base_channels", 32)),
            dropout=float(model_args.get("dropout", 0.0)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    manifest_rows = load_manifest_rows(args.manifest)
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(args.seed)

    for row in manifest_rows:
        raw_csv_column = _infer_raw_column(manifest_rows, modality)
        raw_csv_path = resolve_repo_relative_path(row[raw_csv_column], Path.cwd())
        cube_data = load_thz_csv(raw_csv_path)
        coords, spectra = extract_valid_pixel_spectra(cube_data)

        axis_values = torch.from_numpy(cube_data.axis_values.astype(np.float32)).to(device)
        axis_values = axis_values.unsqueeze(0).repeat(args.batch_size, 1)

        predicted_batches: List[np.ndarray] = []
        masked_batches: List[np.ndarray] = []
        observed_batches: List[np.ndarray] = []

        for start in range(0, spectra.shape[0], args.batch_size):
            stop = min(start + args.batch_size, spectra.shape[0])
            spectrum_batch = torch.from_numpy(spectra[start:stop]).unsqueeze(1).to(device)
            axis_batch = axis_values[: stop - start]

            batch = {"spectrum": spectrum_batch, "axis_values": axis_batch}
            masked = build_spectral_masked_inputs(
                batch,
                mask_mode=args.mask_mode,
                min_observed_ratio=args.min_observed_ratio,
                max_observed_ratio=args.max_observed_ratio,
                use_axis_channel=True,
                generator=generator,
            )

            with torch.no_grad():
                prediction = model(masked["model_input"])
                reconstruction = masked["masked_spectrum"] + prediction * (1.0 - masked["observed_mask"])

            predicted_batches.append(reconstruction.squeeze(1).detach().cpu().numpy())
            masked_batches.append(masked["masked_spectrum"].squeeze(1).detach().cpu().numpy())
            observed_batches.append(masked["observed_mask"].squeeze(1).detach().cpu().numpy())

        predicted_spectra = np.concatenate(predicted_batches, axis=0).astype(np.float32)
        masked_spectra = np.concatenate(masked_batches, axis=0).astype(np.float32)
        observed_masks = np.concatenate(observed_batches, axis=0).astype(np.float32)

        target_cube = cube_data.cube.astype(np.float32)
        reconstructed_cube = assemble_cube_from_pixel_spectra(
            coords=coords,
            spectra=predicted_spectra,
            height=target_cube.shape[0],
            width=target_cube.shape[1],
        )
        masked_cube = assemble_cube_from_pixel_spectra(
            coords=coords,
            spectra=masked_spectra,
            height=target_cube.shape[0],
            width=target_cube.shape[1],
        )

        sample_id = row.get("sample_id", raw_csv_path.stem)
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
            coords=coords,
        )

        header_metadata = dict(cube_data.header_metadata)
        header_metadata["鏂囦欢鍚嶇О"] = str(sample_dir / "reconstructed.imgtds")
        header_metadata["鎴愬儚妯″紡"] = "FS-only鍩虹嚎閲嶅缓鍙嶅皠鍏夎氨"

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
                    "num_pixels": int(coords.shape[0]),
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

