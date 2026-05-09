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

from ppfnet.stage1_spectral_unet import build_spectral_masked_inputs
from ppfnet.thz_csv import (
    assemble_cube_from_pixel_spectra,
    extract_valid_pixel_spectra,
    load_thz_csv,
    resolve_repo_relative_path,
    save_thz_csv,
)
from ppfnet.thz_imaging import compute_image_map_errors, compute_thz_image_maps, save_image_maps_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct the stage2 test set with traditional spectral interpolation baselines."
    )
    parser.add_argument("--manifest", type=Path, default=Path("outputs/ppfnet_stage2/splits/test_pairs.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ppfnet_stage2_interpolation/predictions/test_reconstruction"),
    )
    parser.add_argument(
        "--method",
        choices=["linear", "pchip", "cubic_spline"],
        default="linear",
        help="Interpolation method used to fill missing spectral positions.",
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


def _linear_interpolate(axis_values: np.ndarray, spectrum: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    observed = observed_mask > 0.5
    if observed.sum() == 0:
        return np.zeros_like(spectrum, dtype=np.float32)
    if observed.sum() == 1:
        only_value = float(spectrum[observed][0])
        return np.full_like(spectrum, only_value, dtype=np.float32)
    reconstructed = np.interp(
        axis_values,
        axis_values[observed],
        spectrum[observed],
    )
    return reconstructed.astype(np.float32, copy=False)


def _scipy_interpolate(
    method: str,
    axis_values: np.ndarray,
    spectrum: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    observed = observed_mask > 0.5
    observed_count = int(observed.sum())
    if observed_count < 2:
        return _linear_interpolate(axis_values, spectrum, observed_mask)

    x_obs = axis_values[observed]
    y_obs = spectrum[observed]

    if method == "pchip":
        from scipy.interpolate import PchipInterpolator

        interpolator = PchipInterpolator(x_obs, y_obs, extrapolate=False)
    elif method == "cubic_spline":
        from scipy.interpolate import CubicSpline

        if observed_count < 4:
            return _linear_interpolate(axis_values, spectrum, observed_mask)
        interpolator = CubicSpline(x_obs, y_obs, extrapolate=False)
    else:
        raise ValueError("Unsupported scipy interpolation method: {0}".format(method))

    reconstructed = interpolator(axis_values)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)

    left_fill = float(y_obs[0])
    right_fill = float(y_obs[-1])
    reconstructed = np.where(np.isfinite(reconstructed), reconstructed, np.nan).astype(np.float32, copy=False)

    left_invalid = ~np.isfinite(reconstructed) & (axis_values <= x_obs[0])
    right_invalid = ~np.isfinite(reconstructed) & (axis_values >= x_obs[-1])
    reconstructed[left_invalid] = left_fill
    reconstructed[right_invalid] = right_fill

    remaining_invalid = ~np.isfinite(reconstructed)
    if np.any(remaining_invalid):
        fallback = _linear_interpolate(axis_values, spectrum, observed_mask)
        reconstructed[remaining_invalid] = fallback[remaining_invalid]
    return reconstructed.astype(np.float32, copy=False)


def reconstruct_spectra(
    method: str,
    axis_values: np.ndarray,
    masked_spectra: np.ndarray,
    observed_masks: np.ndarray,
) -> np.ndarray:
    reconstructed = np.empty_like(masked_spectra, dtype=np.float32)
    for row_idx in range(masked_spectra.shape[0]):
        spectrum = masked_spectra[row_idx]
        observed_mask = observed_masks[row_idx]
        if method == "linear":
            rebuilt = _linear_interpolate(axis_values, spectrum, observed_mask)
        else:
            rebuilt = _scipy_interpolate(method, axis_values, spectrum, observed_mask)
        rebuilt[observed_mask > 0.5] = spectrum[observed_mask > 0.5]
        reconstructed[row_idx] = rebuilt.astype(np.float32, copy=False)
    return reconstructed


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_rows = load_manifest_rows(args.manifest)
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(args.seed)

    for row in manifest_rows:
        raw_csv_path = resolve_repo_relative_path(row["fs_raw_csv_path"], REPO_ROOT)
        cube_data = load_thz_csv(raw_csv_path)
        coords, spectra = extract_valid_pixel_spectra(cube_data)

        axis_values_tensor = torch.from_numpy(cube_data.axis_values.astype(np.float32)).to(device)
        axis_values_tensor = axis_values_tensor.unsqueeze(0).repeat(args.batch_size, 1)

        masked_batches: List[np.ndarray] = []
        observed_batches: List[np.ndarray] = []

        for start in range(0, spectra.shape[0], args.batch_size):
            stop = min(start + args.batch_size, spectra.shape[0])
            spectrum_batch = torch.from_numpy(spectra[start:stop]).unsqueeze(1).to(device)
            axis_batch = axis_values_tensor[: stop - start]

            masked = build_spectral_masked_inputs(
                {"spectrum": spectrum_batch, "axis_values": axis_batch},
                mask_mode=args.mask_mode,
                min_observed_ratio=args.min_observed_ratio,
                max_observed_ratio=args.max_observed_ratio,
                use_axis_channel=True,
                generator=generator,
            )

            masked_batches.append(masked["masked_spectrum"].squeeze(1).detach().cpu().numpy().astype(np.float32))
            observed_batches.append(masked["observed_mask"].squeeze(1).detach().cpu().numpy().astype(np.float32))

        masked_spectra = np.concatenate(masked_batches, axis=0).astype(np.float32)
        observed_masks = np.concatenate(observed_batches, axis=0).astype(np.float32)
        predicted_spectra = reconstruct_spectra(
            method=args.method,
            axis_values=cube_data.axis_values.astype(np.float32),
            masked_spectra=masked_spectra,
            observed_masks=observed_masks,
        )

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
        header_metadata["閺傚洣娆㈤崥宥囆?] = str(sample_dir / "reconstructed.imgtds")
        header_metadata["閹存劕鍎氬Ο鈥崇础"] = "{0} 閹绘帒鈧偐娈戦崣宥呯殸閸忓姘ㄩ柌宥呯紦".format(args.method)

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
                    "method": args.method,
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

