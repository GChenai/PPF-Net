from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def compute_slice_map(
    cube: np.ndarray,
    axis_values: np.ndarray,
    valid_mask: np.ndarray,
    target: float,
    window: float = 0.0,
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    if axis_values.size == 0:
        raise ValueError("Axis values are empty.")

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
            "axis_left": float(axis_values[selector][0]),
            "axis_right": float(axis_values[selector][-1]),
            "selected_count": int(np.sum(selector)),
        }
        return name, data, meta

    nearest_idx = int(np.argmin(np.abs(axis_values - target)))
    nearest_value = float(axis_values[nearest_idx])
    data = cube[..., nearest_idx].astype(np.float32)
    data[~valid_mask] = np.nan
    name = "slice_{0:.6f}".format(nearest_value)
    meta = {
        "target": float(target),
        "window": 0.0,
        "axis_left": nearest_value,
        "axis_right": nearest_value,
        "selected_count": 1,
    }
    return name, data, meta


def compute_band_map(
    cube: np.ndarray,
    axis_values: np.ndarray,
    valid_mask: np.ndarray,
    left: float,
    right: float,
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    if right < left:
        left, right = right, left
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
        "window": float(abs(right - left)),
        "axis_left": float(axis_values[selector][0]),
        "axis_right": float(axis_values[selector][-1]),
        "selected_count": int(np.sum(selector)),
    }
    return name, data, meta


def compute_thz_image_maps(
    cube: np.ndarray,
    axis_values: np.ndarray,
    valid_mask: np.ndarray,
    slice_values: Optional[Sequence[float]] = None,
    band_ranges: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_value = np.nanmean(cube, axis=-1).astype(np.float32)
        std_value = np.nanstd(cube, axis=-1).astype(np.float32)
        max_value = np.nanmax(cube, axis=-1).astype(np.float32)
        min_value = np.nanmin(cube, axis=-1).astype(np.float32)

    peak_to_peak = (max_value - min_value).astype(np.float32)

    if axis_values.size > 1 and np.all(np.isfinite(axis_values)):
        integral_value = np.trapz(np.nan_to_num(cube, nan=0.0), x=axis_values, axis=-1).astype(np.float32)
    else:
        integral_value = mean_value.copy()

    filled_max = np.where(np.isfinite(cube), cube, -np.inf)
    filled_min = np.where(np.isfinite(cube), cube, np.inf)
    argmax_axis = axis_values[np.argmax(filled_max, axis=-1)].astype(np.float32)
    argmin_axis = axis_values[np.argmin(filled_min, axis=-1)].astype(np.float32)

    image_maps: Dict[str, np.ndarray] = {
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

    metadata: Dict[str, Dict[str, float]] = {}

    for target in slice_values or []:
        name, array, meta = compute_slice_map(cube, axis_values, valid_mask, target, window=0.0)
        image_maps[name] = array
        metadata[name] = meta

    for left, right in band_ranges or []:
        name, array, meta = compute_band_map(cube, axis_values, valid_mask, left, right)
        image_maps[name] = array
        metadata[name] = meta

    for key, array in image_maps.items():
        if key == "valid_mask":
            continue
        array[~valid_mask] = np.nan

    return image_maps, metadata


def _to_rgba(array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
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


def save_image_maps_png(
    image_maps: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    output_dir: Path | str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in image_maps.items():
        if name == "valid_mask":
            image = Image.fromarray((valid_mask.astype(np.uint8) * 255), mode="L")
            image.save(output_dir / "valid_mask.png")
            continue
        rgba = _to_rgba(array, valid_mask)
        Image.fromarray(rgba, mode="RGBA").save(output_dir / "{0}.png".format(name))


def compute_image_map_errors(
    target_maps: Dict[str, np.ndarray],
    reconstructed_maps: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    error_maps: Dict[str, np.ndarray] = {
        "valid_mask": valid_mask.astype(np.float32),
    }
    shared_keys = sorted(set(target_maps) & set(reconstructed_maps))
    for key in shared_keys:
        if key == "valid_mask":
            continue
        target = target_maps[key]
        recon = reconstructed_maps[key]
        error = np.abs(recon - target).astype(np.float32)
        error[~valid_mask] = np.nan
        error_maps[key] = error
    return error_maps
