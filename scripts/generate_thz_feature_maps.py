#!/usr/bin/env python
"""
Generate multi-feature THz images from CSV hyperspectral cubes.

This script targets CSV files with rows shaped like:

    abs,min: 0.018...THz,max: 4.998...THz,0.018...,0.036...,...
    abs,X51,Y8,0.31,1.39,...

For each CSV, it reconstructs a cube of shape [H, W, C] and exports:

- valid_mask        -> valid seed-region mask
- mean_value        -> mean image along the spectral / temporal axis
- std_value         -> standard deviation image
- integral_value    -> integral image along the axis
- max_value         -> maximum value image
- min_value         -> minimum value image
- peak_to_peak      -> max - min image
- argmax_axis       -> axis position of the maximum value
- argmin_axis       -> axis position of the minimum value
- slice_* / band_*  -> user-requested single-axis or axis-range images

Useful THz mappings:

- peak_to_peak      -> peak-to-peak image
- max_value         -> maximum-peak image
- min_value         -> minimum-peak image
- argmax_axis       -> max flight-time image for time-domain CSVs
- argmin_axis       -> min flight-time image for time-domain CSVs
- slice_* / band_*  -> time-domain / frequency-domain / absorbance slice image
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("numpy is required to run this script.") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Pillow is required to save PNG files.") from exc


AXIS_UNIT_RE = re.compile(r"([A-Za-z]+)$")
X_RE = re.compile(r"X(\d+)$", re.IGNORECASE)
Y_RE = re.compile(r"Y(\d+)$", re.IGNORECASE)


@dataclass
class THzCube:
    source_path: Path
    cube: np.ndarray
    axis_values: np.ndarray
    valid_mask: np.ndarray
    signal_label: str
    axis_unit: str
    axis_domain: str
    header_metadata: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-feature THz images from CSV hyperspectral cubes."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV file or directory containing CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ppfnet_feature_maps"),
        help="Output directory. One subdirectory will be created per CSV file.",
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="Glob pattern used when --input is a directory. Default: *.csv",
    )
    parser.add_argument(
        "--axis-values",
        nargs="*",
        type=float,
        default=[],
        help="Axis values to export as slice images, e.g. --axis-values 1.0 2.0 3.0",
    )
    parser.add_argument(
        "--axis-ranges",
        nargs="*",
        default=[],
        help="Axis ranges to export as band-mean images, e.g. --axis-ranges 0.8:1.2 1.8:2.2",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.0,
        help="Optional averaging window for each axis value. "
        "If > 0, the slice image becomes an average over [value-window/2, value+window/2].",
    )
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Only save NPZ + metadata, do not save PNG previews.",
    )
    parser.add_argument(
        "--skip-npz",
        action="store_true",
        help="Only save PNG previews + metadata, do not save the compressed NPZ stack.",
    )
    parser.add_argument(
        "--png-low",
        type=float,
        default=1.0,
        help="Lower percentile used for PNG visualization. Default: 1",
    )
    parser.add_argument(
        "--png-high",
        type=float,
        default=99.0,
        help="Upper percentile used for PNG visualization. Default: 99",
    )
    return parser.parse_args()


def strip_trailing_empty(items: Sequence[str]) -> List[str]:
    cleaned = [item.strip() for item in items]
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return cleaned


def safe_float(text: str) -> float:
    if text == "" or text.lower() == "nan":
        return float("nan")
    return float(text)


def infer_axis_unit(*texts: str) -> str:
    for text in texts:
        match = AXIS_UNIT_RE.search(text.strip())
        if match:
            return match.group(1)
    return ""


def infer_axis_domain(axis_unit: str) -> str:
    unit = axis_unit.lower()
    if unit in {"thz", "ghz", "mhz", "khz", "hz"}:
        return "frequency"
    if unit in {"ps", "ns", "us", "ms", "s", "fs"}:
        return "time"
    return "unknown"


def parse_coord(token: str, pattern: re.Pattern[str], axis_name: str) -> int:
    match = pattern.match(token.strip())
    if not match:
        raise ValueError("Could not parse {0} coordinate from token: {1}".format(axis_name, token))
    return int(match.group(1))


def parse_thz_csv(path: Path) -> THzCube:
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            row = strip_trailing_empty(row)
            if row:
                rows.append(row)

    if len(rows) < 6:
        raise ValueError("CSV looks too short to be a THz cube: {0}".format(path))

    header_metadata: Dict[str, str] = {}
    for row in rows[:4]:
        if len(row) >= 2:
            header_metadata[row[0]] = ",".join(row[1:])

    axis_row = rows[4]
    if len(axis_row) < 4:
        raise ValueError("Could not find an axis row with at least 4 columns in: {0}".format(path))

    signal_label = axis_row[0]
    axis_values = np.asarray([safe_float(item) for item in axis_row[3:]], dtype=np.float32)
    axis_unit = infer_axis_unit(axis_row[1], axis_row[2])
    axis_domain = infer_axis_domain(axis_unit)
    axis_count = int(axis_values.shape[0])

    parsed_rows: List[Tuple[int, int, np.ndarray]] = []
    max_x = -1
    max_y = -1

    for row in rows[5:]:
        if len(row) < 3:
            continue
        try:
            x = parse_coord(row[1], X_RE, "X")
            y = parse_coord(row[2], Y_RE, "Y")
        except ValueError:
            continue

        values = [safe_float(item) for item in row[3:3 + axis_count]]
        if len(values) < axis_count:
            values.extend([float("nan")] * (axis_count - len(values)))
        spectra = np.asarray(values, dtype=np.float32)

        parsed_rows.append((x, y, spectra))
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    if not parsed_rows:
        raise ValueError("No pixel rows were parsed from: {0}".format(path))

    cube = np.full((max_y + 1, max_x + 1, axis_count), np.nan, dtype=np.float32)
    for x, y, spectra in parsed_rows:
        cube[y, x, :] = spectra

    valid_mask = np.isfinite(cube).any(axis=-1)

    return THzCube(
        source_path=path,
        cube=cube,
        axis_values=axis_values,
        valid_mask=valid_mask,
        signal_label=signal_label,
        axis_unit=axis_unit,
        axis_domain=axis_domain,
        header_metadata=header_metadata,
    )


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
    return sanitized.strip("_") or "feature"


def compute_arg_feature(
    cube: np.ndarray,
    axis_values: np.ndarray,
    valid_mask: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "max":
        filler = -np.inf
        op = np.argmax
        reducer = np.max
    elif mode == "min":
        filler = np.inf
        op = np.argmin
        reducer = np.min
    else:
        raise ValueError("Unsupported mode: {0}".format(mode))

    filled = np.where(np.isfinite(cube), cube, filler)
    indices = op(filled, axis=-1)
    values = reducer(filled, axis=-1).astype(np.float32)
    axis_positions = axis_values[indices].astype(np.float32)

    values[~valid_mask] = np.nan
    axis_positions[~valid_mask] = np.nan
    return values, axis_positions


def compute_slice_map(
    cube: np.ndarray,
    axis_values: np.ndarray,
    valid_mask: np.ndarray,
    target: float,
    window: float,
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    if axis_values.size == 0:
        raise ValueError("Axis values are empty, cannot create slice map.")

    if window > 0:
        left = target - window / 2.0
        right = target + window / 2.0
        selector = (axis_values >= left) & (axis_values <= right)
        if not np.any(selector):
            nearest_idx = int(np.argmin(np.abs(axis_values - target)))
            selector = np.zeros_like(axis_values, dtype=bool)
            selector[nearest_idx] = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = np.nanmean(cube[..., selector], axis=-1).astype(np.float32)
        data[~valid_mask] = np.nan

        name = "band_{0:.6f}_w_{1:.6f}".format(target, window)
        meta = {
            "target": float(target),
            "window": float(window),
            "selected_count": int(np.sum(selector)),
            "axis_left": float(axis_values[selector][0]),
            "axis_right": float(axis_values[selector][-1]),
        }
        return sanitize_name(name), data, meta

    nearest_idx = int(np.argmin(np.abs(axis_values - target)))
    nearest_value = float(axis_values[nearest_idx])
    data = cube[..., nearest_idx].astype(np.float32)
    data[~valid_mask] = np.nan

    name = "slice_{0:.6f}".format(nearest_value)
    meta = {
        "target": float(target),
        "window": 0.0,
        "selected_count": 1,
        "axis_left": nearest_value,
        "axis_right": nearest_value,
    }
    return sanitize_name(name), data, meta


def parse_axis_range(text: str) -> Tuple[float, float]:
    try:
        left_text, right_text = text.split(":", 1)
        left = float(left_text)
        right = float(right_text)
    except ValueError as exc:
        raise ValueError("Axis range must look like start:end, got: {0}".format(text)) from exc

    if right < left:
        left, right = right, left
    return left, right


def compute_band_map(
    cube: np.ndarray,
    axis_values: np.ndarray,
    valid_mask: np.ndarray,
    left: float,
    right: float,
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    selector = (axis_values >= left) & (axis_values <= right)
    if not np.any(selector):
        center = 0.5 * (left + right)
        nearest_idx = int(np.argmin(np.abs(axis_values - center)))
        selector = np.zeros_like(axis_values, dtype=bool)
        selector[nearest_idx] = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data = np.nanmean(cube[..., selector], axis=-1).astype(np.float32)
    data[~valid_mask] = np.nan

    name = "band_{0:.6f}_{1:.6f}".format(left, right)
    meta = {
        "target": float(0.5 * (left + right)),
        "window": float(right - left),
        "selected_count": int(np.sum(selector)),
        "axis_left": float(axis_values[selector][0]),
        "axis_right": float(axis_values[selector][-1]),
    }
    return sanitize_name(name), data, meta


def compute_feature_maps(
    cube_data: THzCube,
    axis_values: Sequence[float],
    axis_ranges: Sequence[str],
    window: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    cube = cube_data.cube
    valid_mask = cube_data.valid_mask
    axis = cube_data.axis_values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_value = np.nanmean(cube, axis=-1).astype(np.float32)
        std_value = np.nanstd(cube, axis=-1).astype(np.float32)

    if axis.size > 1 and np.all(np.isfinite(axis)):
        integral_value = np.trapz(np.nan_to_num(cube, nan=0.0), x=axis, axis=-1).astype(np.float32)
        integral_value[~valid_mask] = np.nan
    else:
        integral_value = mean_value.copy()

    max_value, argmax_axis = compute_arg_feature(cube, axis, valid_mask, mode="max")
    min_value, argmin_axis = compute_arg_feature(cube, axis, valid_mask, mode="min")

    peak_to_peak = (max_value - min_value).astype(np.float32)
    peak_to_peak[~valid_mask] = np.nan

    feature_maps: Dict[str, np.ndarray] = {
        "valid_mask": valid_mask.astype(np.float32),
        "mean_value": mean_value,
        "std_value": std_value,
        "integral_value": integral_value,
        "max_value": max_value,
        "min_value": min_value,
        "peak_to_peak": peak_to_peak,
        "argmax_axis": argmax_axis,
        "argmin_axis": argmin_axis,
    }

    derived_meta: Dict[str, Dict[str, float]] = {}

    for value in axis_values:
        name, data, meta = compute_slice_map(cube, axis, valid_mask, target=float(value), window=window)
        feature_maps[name] = data
        derived_meta[name] = meta

    for range_text in axis_ranges:
        left, right = parse_axis_range(range_text)
        name, data, meta = compute_band_map(cube, axis, valid_mask, left=left, right=right)
        feature_maps[name] = data
        derived_meta[name] = meta

    for name, data in feature_maps.items():
        if name == "valid_mask":
            continue
        data[~valid_mask] = np.nan

    return feature_maps, derived_meta


def normalize_for_png(
    array: np.ndarray,
    valid_mask: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> np.ndarray:
    finite_mask = np.isfinite(array) & valid_mask
    alpha = np.where(finite_mask, 255, 0).astype(np.uint8)

    if not np.any(finite_mask):
        return np.stack(
            [np.zeros_like(alpha), np.zeros_like(alpha), np.zeros_like(alpha), alpha],
            axis=-1,
        )

    values = array[finite_mask]
    low = float(np.percentile(values, low_pct))
    high = float(np.percentile(values, high_pct))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
        if high <= low:
            high = low + 1e-6

    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    gray = np.round(scaled * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray, alpha], axis=-1)


def save_png(
    path: Path,
    array: np.ndarray,
    valid_mask: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> None:
    rgba = normalize_for_png(array, valid_mask, low_pct=low_pct, high_pct=high_pct)
    image = Image.fromarray(rgba, mode="RGBA")
    image.save(path)


def save_mask_png(path: Path, valid_mask: np.ndarray) -> None:
    image = Image.fromarray((valid_mask.astype(np.uint8) * 255), mode="L")
    image.save(path)


def feature_summary(array: np.ndarray) -> Dict[str, Optional[float]]:
    finite = np.isfinite(array)
    if not np.any(finite):
        return {
            "finite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    values = array[finite]
    return {
        "finite_count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def build_metadata(
    cube_data: THzCube,
    feature_maps: Dict[str, np.ndarray],
    derived_meta: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    axis = cube_data.axis_values
    valid_pixels = int(np.sum(cube_data.valid_mask))
    total_pixels = int(cube_data.valid_mask.size)

    metadata: Dict[str, object] = {
        "source_file": str(cube_data.source_path),
        "signal_label": cube_data.signal_label,
        "axis_unit": cube_data.axis_unit,
        "axis_domain": cube_data.axis_domain,
        "cube_shape": list(map(int, cube_data.cube.shape)),
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
        "valid_ratio": float(valid_pixels / total_pixels) if total_pixels else 0.0,
        "axis_count": int(axis.size),
        "axis_min": float(axis[0]) if axis.size else None,
        "axis_max": float(axis[-1]) if axis.size else None,
        "axis_values": [float(value) for value in axis.tolist()],
        "header_metadata": cube_data.header_metadata,
        "feature_summaries": {
            name: feature_summary(array)
            for name, array in feature_maps.items()
        },
        "derived_feature_metadata": derived_meta,
    }
    return metadata


def collect_csv_paths(input_path: Path, pattern: str) -> Tuple[List[Path], Path]:
    if input_path.is_file():
        return [input_path], input_path.parent

    if not input_path.is_dir():
        raise FileNotFoundError("Input path does not exist: {0}".format(input_path))

    files = sorted(path for path in input_path.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError("No CSV files matched {0!r} inside {1}".format(pattern, input_path))
    return files, input_path


def output_subdir(output_root: Path, input_root: Path, csv_path: Path) -> Path:
    try:
        relative = csv_path.relative_to(input_root)
    except ValueError:
        relative = Path(csv_path.name)
    return output_root / relative.with_suffix("")


def process_file(
    csv_path: Path,
    input_root: Path,
    output_root: Path,
    axis_values: Sequence[float],
    axis_ranges: Sequence[str],
    window: float,
    save_pngs: bool,
    save_npz: bool,
    png_low: float,
    png_high: float,
) -> None:
    cube_data = parse_thz_csv(csv_path)
    feature_maps, derived_meta = compute_feature_maps(
        cube_data,
        axis_values=axis_values,
        axis_ranges=axis_ranges,
        window=window,
    )

    target_dir = output_subdir(output_root, input_root, csv_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(cube_data, feature_maps, derived_meta)

    metadata_path = target_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if save_npz:
        np.savez_compressed(target_dir / "feature_maps.npz", **feature_maps)

    if save_pngs:
        save_mask_png(target_dir / "valid_mask.png", cube_data.valid_mask)
        for name, array in feature_maps.items():
            if name == "valid_mask":
                continue
            save_png(
                target_dir / "{0}.png".format(name),
                array=array,
                valid_mask=cube_data.valid_mask,
                low_pct=png_low,
                high_pct=png_high,
            )

    print("[OK] {0} -> {1}".format(csv_path, target_dir))


def main() -> int:
    args = parse_args()

    csv_paths, input_root = collect_csv_paths(args.input, args.glob)
    args.output.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        process_file(
            csv_path=csv_path,
            input_root=input_root,
            output_root=args.output,
            axis_values=args.axis_values,
            axis_ranges=args.axis_ranges,
            window=args.window,
            save_pngs=not args.skip_png,
            save_npz=not args.skip_npz,
            png_low=args.png_low,
            png_high=args.png_high,
        )

    print("Processed {0} CSV file(s).".format(len(csv_paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

