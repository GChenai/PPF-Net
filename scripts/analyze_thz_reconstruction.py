#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.thz_csv import load_thz_csv
from ppfnet.thz_imaging import compute_thz_image_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze reconstructed THz CSV directories with spectral and image-map metrics."
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Directory containing per-sample reconstruction folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where analysis outputs are written. Defaults to <prediction-root>/analysis.",
    )
    parser.add_argument(
        "--slice-values",
        nargs="*",
        type=float,
        default=[1.0, 2.0, 3.0],
        help="Axis values used for image-map analysis.",
    )
    parser.add_argument(
        "--band-ranges",
        nargs="*",
        default=["0.8:1.2", "1.8:2.2"],
        help="Axis ranges used for image-map and band-integral analysis.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for quick analysis/debugging.",
    )
    return parser.parse_args()


def parse_band_ranges(values: Sequence[str]) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for value in values:
        left_text, right_text = value.split(":", 1)
        left = float(left_text)
        right = float(right_text)
        if right < left:
            left, right = right, left
        ranges.append((left, right))
    return ranges


def list_sample_dirs(root: Path, max_samples: int | None) -> List[Path]:
    dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if max_samples is not None:
        dirs = dirs[:max_samples]
    return dirs


def safe_psnr(target: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> float:
    values = target[valid_mask]
    pred_values = pred[valid_mask]
    if values.size == 0:
        return float("nan")
    mse = float(np.mean((pred_values - values) ** 2))
    if mse <= 1e-12:
        return 99.0
    data_range = float(np.max(values) - np.min(values))
    if data_range <= 1e-12:
        data_range = 1.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def basic_error_metrics(target: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    values = target[valid_mask]
    pred_values = pred[valid_mask]
    if values.size == 0:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "psnr": float("nan")}
    mae = float(np.mean(np.abs(pred_values - values)))
    mse = float(np.mean((pred_values - values) ** 2))
    rmse = float(math.sqrt(mse))
    psnr = safe_psnr(target, pred, valid_mask)
    return {"mae": mae, "mse": mse, "rmse": rmse, "psnr": psnr}


def band_integral(spectra: np.ndarray, axis_values: np.ndarray, left: float, right: float) -> np.ndarray:
    selector = (axis_values >= left) & (axis_values <= right)
    if not np.any(selector):
        center = 0.5 * (left + right)
        nearest_idx = int(np.argmin(np.abs(axis_values - center)))
        selector = np.zeros_like(axis_values, dtype=bool)
        selector[nearest_idx] = True
    return np.trapz(spectra[:, selector], x=axis_values[selector], axis=1)


def analyze_spectral_metrics(
    target_cube: np.ndarray,
    pred_cube: np.ndarray,
    axis_values: np.ndarray,
    valid_mask: np.ndarray,
    band_ranges: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    target_spectra = target_cube[valid_mask]
    pred_spectra = pred_cube[valid_mask]
    finite_rows = np.isfinite(target_spectra).all(axis=1) & np.isfinite(pred_spectra).all(axis=1)
    target_spectra = target_spectra[finite_rows]
    pred_spectra = pred_spectra[finite_rows]

    if target_spectra.size == 0:
        return {}

    metrics = basic_error_metrics(target_spectra, pred_spectra, np.ones_like(target_spectra, dtype=bool))

    target_peak_idx = np.argmax(target_spectra, axis=1)
    pred_peak_idx = np.argmax(pred_spectra, axis=1)
    target_valley_idx = np.argmin(target_spectra, axis=1)
    pred_valley_idx = np.argmin(pred_spectra, axis=1)

    metrics["peak_axis_mae"] = float(np.mean(np.abs(axis_values[pred_peak_idx] - axis_values[target_peak_idx])))
    metrics["peak_value_mae"] = float(np.mean(np.abs(pred_spectra[np.arange(pred_spectra.shape[0]), pred_peak_idx] - target_spectra[np.arange(target_spectra.shape[0]), target_peak_idx])))
    metrics["valley_axis_mae"] = float(np.mean(np.abs(axis_values[pred_valley_idx] - axis_values[target_valley_idx])))
    metrics["valley_value_mae"] = float(np.mean(np.abs(pred_spectra[np.arange(pred_spectra.shape[0]), pred_valley_idx] - target_spectra[np.arange(target_spectra.shape[0]), target_valley_idx])))

    target_p2p = target_spectra.max(axis=1) - target_spectra.min(axis=1)
    pred_p2p = pred_spectra.max(axis=1) - pred_spectra.min(axis=1)
    metrics["peak_to_peak_mae"] = float(np.mean(np.abs(pred_p2p - target_p2p)))

    for left, right in band_ranges:
        target_integral = band_integral(target_spectra, axis_values, left, right)
        pred_integral = band_integral(pred_spectra, axis_values, left, right)
        key = "band_{0:.3f}_{1:.3f}_mae".format(left, right)
        metrics[key] = float(np.mean(np.abs(pred_integral - target_integral)))

    return metrics


def analyze_image_maps(
    target_cube: np.ndarray,
    pred_cube: np.ndarray,
    axis_values: np.ndarray,
    valid_mask: np.ndarray,
    slice_values: Sequence[float],
    band_ranges: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    target_maps, _ = compute_thz_image_maps(target_cube, axis_values, valid_mask, slice_values=slice_values, band_ranges=band_ranges)
    pred_maps, _ = compute_thz_image_maps(pred_cube, axis_values, valid_mask, slice_values=slice_values, band_ranges=band_ranges)
    metrics: Dict[str, float] = {}
    for name in sorted(set(target_maps) & set(pred_maps)):
        if name == "valid_mask":
            continue
        valid = np.isfinite(target_maps[name]) & np.isfinite(pred_maps[name]) & valid_mask
        map_metrics = basic_error_metrics(target_maps[name], pred_maps[name], valid)
        metrics["{0}_mae".format(name)] = map_metrics["mae"]
        metrics["{0}_rmse".format(name)] = map_metrics["rmse"]
        metrics["{0}_psnr".format(name)] = map_metrics["psnr"]
    return metrics


def analyze_sample(
    sample_dir: Path,
    slice_values: Sequence[float],
    band_ranges: Sequence[Tuple[float, float]],
) -> Dict[str, object]:
    target = load_thz_csv(sample_dir / "target.csv")
    reconstructed = load_thz_csv(sample_dir / "reconstructed.csv")

    if not np.allclose(target.axis_values, reconstructed.axis_values, atol=1e-6):
        raise ValueError("Axis values differ between target and reconstructed CSV: {0}".format(sample_dir))

    valid_mask = target.valid_mask & reconstructed.valid_mask
    spectral_metrics = analyze_spectral_metrics(
        target.cube,
        reconstructed.cube,
        target.axis_values,
        valid_mask,
        band_ranges=band_ranges,
    )
    image_metrics = analyze_image_maps(
        target.cube,
        reconstructed.cube,
        target.axis_values,
        valid_mask,
        slice_values=slice_values,
        band_ranges=band_ranges,
    )

    metrics: Dict[str, object] = {
        "sample_id": sample_dir.name,
        "num_valid_pixels": int(valid_mask.sum()),
    }
    metrics.update(spectral_metrics)
    metrics.update(image_metrics)
    return metrics


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
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


def mean_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    summary: Dict[str, object] = {"num_samples": len(rows)}
    metric_keys = [key for key in rows[0].keys() if key not in {"sample_id"}]
    for key in metric_keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not math.isnan(float(row[key]))]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def main() -> int:
    args = parse_args()
    prediction_root = args.prediction_root
    output_dir = args.output_dir or (prediction_root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    band_ranges = parse_band_ranges(args.band_ranges)
    sample_dirs = list_sample_dirs(prediction_root, args.max_samples)

    rows: List[Dict[str, object]] = []
    for sample_dir in sample_dirs:
        if not (sample_dir / "target.csv").exists() or not (sample_dir / "reconstructed.csv").exists():
            continue
        metrics = analyze_sample(
            sample_dir,
            slice_values=args.slice_values,
            band_ranges=band_ranges,
        )
        rows.append(metrics)
        print("analyzed:", sample_dir.name)

    summary = mean_summary(rows)
    write_csv(output_dir / "per_sample_metrics.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

