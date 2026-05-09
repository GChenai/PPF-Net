#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage1_spectral_unet import SpectralUNet1D, build_spectral_masked_inputs
from ppfnet.thz_csv import (
    assemble_cube_from_pixel_spectra,
    extract_valid_pixel_spectra,
    load_thz_csv,
    resolve_repo_relative_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a full THz cube from pixel-wise spectral predictions.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/ppfnet_stage1_pixel_fs/checkpoints/stage1_pixel_spectral_unet_best.pt"),
        help="Checkpoint to load.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/ppfnet_stage1/splits/test_pairs.csv"),
        help="Manifest CSV from which to select a sample.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Row index inside the manifest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ppfnet_stage1_pixel_fs/predictions/cube_reconstructions"),
        help="Directory where reconstruction artifacts are written.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mask-mode", choices=["point", "band", "hybrid"], default="hybrid")
    parser.add_argument("--min-observed-ratio", type=float, default=0.25)
    parser.add_argument("--max-observed-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_manifest_rows(path: Path) -> List[Dict[str, str]]:
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


def _to_u8(array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid = valid_mask > 0.5
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    if not np.any(valid):
        gray = np.zeros_like(alpha)
        return np.stack([gray, gray, gray, alpha], axis=-1)

    values = array[valid]
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if high <= low:
        high = low + 1e-6

    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    gray = np.round(np.nan_to_num(scaled, nan=0.0) * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray, alpha], axis=-1)


def _save_preview(path: Path, array: np.ndarray, valid_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_u8(array, valid_mask), mode="RGBA").save(path)


def _compute_preview_maps(cube: np.ndarray, valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            mean_value = np.nanmean(cube, axis=-1).astype(np.float32)
            max_value = np.nanmax(cube, axis=-1).astype(np.float32)
            min_value = np.nanmin(cube, axis=-1).astype(np.float32)
    peak_to_peak = (max_value - min_value).astype(np.float32)

    mean_value[~valid_mask] = np.nan
    max_value[~valid_mask] = np.nan
    min_value[~valid_mask] = np.nan
    peak_to_peak[~valid_mask] = np.nan

    return {
        "mean_value": mean_value,
        "max_value": max_value,
        "min_value": min_value,
        "peak_to_peak": peak_to_peak,
    }


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    data_info = checkpoint["data_info"]
    modality = str(data_info["modality"])
    normalization = str(data_info.get("normalization", "none"))
    if normalization != "none":
        raise NotImplementedError(
            "Cube reconstruction export currently expects normalization='none'. "
            "Run pixel-wise training with --normalization none."
        )

    rows = _load_manifest_rows(args.manifest)
    row = rows[args.sample_index]
    raw_csv_column = _infer_raw_column(rows, modality)
    raw_csv_path = resolve_repo_relative_path(row[raw_csv_column], REPO_ROOT)

    cube_data = load_thz_csv(raw_csv_path)
    coords, spectra = extract_valid_pixel_spectra(cube_data)

    model = SpectralUNet1D(
        in_channels=3,
        base_channels=int(checkpoint_args.get("base_channels", 32)),
        dropout=float(checkpoint_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    axis_values = torch.from_numpy(cube_data.axis_values.astype(np.float32)).to(device)
    axis_values = axis_values.unsqueeze(0).repeat(args.batch_size, 1)
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(args.seed)

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

    reconstructed_spectra = np.concatenate(predicted_batches, axis=0).astype(np.float32)
    masked_spectra = np.concatenate(masked_batches, axis=0).astype(np.float32)
    observed_masks = np.concatenate(observed_batches, axis=0).astype(np.float32)

    target_cube = assemble_cube_from_pixel_spectra(
        coords=coords,
        spectra=spectra,
        height=cube_data.cube.shape[0],
        width=cube_data.cube.shape[1],
    )
    masked_cube = assemble_cube_from_pixel_spectra(
        coords=coords,
        spectra=masked_spectra,
        height=cube_data.cube.shape[0],
        width=cube_data.cube.shape[1],
    )
    reconstructed_cube = assemble_cube_from_pixel_spectra(
        coords=coords,
        spectra=reconstructed_spectra,
        height=cube_data.cube.shape[0],
        width=cube_data.cube.shape[1],
    )

    sample_id = row.get("pair_id") or row.get("sample_name") or raw_csv_path.stem
    sample_dir = args.output_dir / str(sample_id).replace("/", "__")
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sample_dir / "reconstruction.npz",
        axis_values=cube_data.axis_values.astype(np.float32),
        coords=coords.astype(np.int32),
        valid_mask=cube_data.valid_mask.astype(np.float32),
        target_cube=target_cube.astype(np.float32),
        masked_cube=masked_cube.astype(np.float32),
        reconstructed_cube=reconstructed_cube.astype(np.float32),
        observed_masks=observed_masks.astype(np.float32),
    )

    preview_maps = {
        "target": _compute_preview_maps(target_cube, cube_data.valid_mask),
        "masked": _compute_preview_maps(masked_cube, cube_data.valid_mask),
        "reconstructed": _compute_preview_maps(reconstructed_cube, cube_data.valid_mask),
    }
    for stage_name, stage_maps in preview_maps.items():
        for map_name, array in stage_maps.items():
            _save_preview(sample_dir / "{0}_{1}.png".format(stage_name, map_name), array, cube_data.valid_mask)

    metadata = {
        "sample_id": sample_id,
        "modality": modality,
        "raw_csv_path": str(raw_csv_path),
        "checkpoint": str(args.checkpoint),
        "mask_mode": args.mask_mode,
        "min_observed_ratio": args.min_observed_ratio,
        "max_observed_ratio": args.max_observed_ratio,
        "num_pixels": int(coords.shape[0]),
        "cube_shape": list(map(int, cube_data.cube.shape)),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved_sample_dir:", sample_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

