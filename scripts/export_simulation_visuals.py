#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.thz_imaging import compute_image_map_errors, compute_thz_image_maps


MODEL_SPECS = {
    "linear_interpolation": {
        "label": "Linear Interpolation",
        "prediction_root": Path("outputs/stage2_interpolation_linear/predictions/test_reconstruction"),
        "analysis_dir": Path("outputs/stage2_interpolation_linear/predictions/test_reconstruction/analysis"),
        "kind": "model",
    },
    "tcn": {
        "label": "TCN",
        "prediction_root": Path("outputs/stage2_tcn_baseline/predictions/test_reconstruction"),
        "analysis_dir": Path("outputs/stage2_tcn_baseline/predictions/test_reconstruction/analysis"),
        "kind": "model",
    },
    "single_modal_thz_baseline": {
        "label": "Single-Modal THz Baseline",
        "prediction_root": Path("outputs/stage2_fs_only_baseline_random/predictions/test_reconstruction"),
        "analysis_dir": Path("outputs/stage2_fs_only_baseline_random/predictions/test_reconstruction/analysis"),
        "kind": "model",
    },
    "ppf_net_obs70": {
        "label": "PPF-Net (Obs. 70%)",
        "prediction_root": Path("outputs/stage2_rgb_fs_patch_student_obs70/predictions/test_reconstruction"),
        "analysis_dir": Path("outputs/stage2_rgb_fs_patch_student_obs70/predictions/test_reconstruction/analysis"),
        "kind": "model",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export simulation-test visuals for selected models."
    )
    parser.add_argument(
        "--sample-id",
        default="auto",
        help="Sample id like A__1. Default: auto (choose representative sample by median RMSE on PPF-Net Obs.70%).",
    )
    parser.add_argument(
        "--map-name",
        default="peak_to_peak",
        help="Image map used for visualization and ROI selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/simulation_visuals"),
        help="Root directory where exported visuals will be written.",
    )
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=96,
        help="Square ROI size in pixels for local zoom.",
    )
    return parser.parse_args()


def load_per_sample_metrics(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: List[Dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                "sample_id": row["sample_id"],
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "psnr": float(row["psnr"]),
            }
        )
    return parsed


def choose_representative_sample() -> str:
    rows = load_per_sample_metrics(MODEL_SPECS["ppf_net_obs70"]["analysis_dir"] / "per_sample_metrics.csv")
    ordered = sorted(rows, key=lambda item: item["rmse"])
    return str(ordered[len(ordered) // 2]["sample_id"])


def load_bundle(prediction_root: Path, sample_id: str) -> Dict[str, np.ndarray]:
    sample_dir = prediction_root / sample_id
    bundle = np.load(sample_dir / "reconstruction.npz")
    return {
        "target_cube": bundle["target_cube"].astype(np.float32),
        "masked_cube": bundle["masked_cube"].astype(np.float32),
        "reconstructed_cube": bundle["reconstructed_cube"].astype(np.float32),
        "axis_values": bundle["axis_values"].astype(np.float32),
        "valid_mask": bundle["valid_mask"].astype(np.float32) > 0.5,
    }


def metric_for_sample(analysis_dir: Path, sample_id: str) -> Dict[str, float]:
    rows = load_per_sample_metrics(analysis_dir / "per_sample_metrics.csv")
    for row in rows:
        if str(row["sample_id"]) == sample_id:
            return {"mae": float(row["mae"]), "rmse": float(row["rmse"]), "psnr": float(row["psnr"])}
    raise KeyError(f"Sample {sample_id} not found in {analysis_dir}")


def normalize_to_rgba(array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
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


def save_map_with_annotation(
    array: np.ndarray,
    valid_mask: np.ndarray,
    output_path: Path,
    title: str,
    metrics: Dict[str, float] | None = None,
) -> None:
    rgba = normalize_to_rgba(array, valid_mask)
    image = Image.fromarray(rgba, mode="RGBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def crop_zoom(array: np.ndarray, center_y: int, center_x: int, zoom_size: int) -> np.ndarray:
    half = zoom_size // 2
    y0 = max(0, center_y - half)
    x0 = max(0, center_x - half)
    y1 = min(array.shape[0], y0 + zoom_size)
    x1 = min(array.shape[1], x0 + zoom_size)
    y0 = max(0, y1 - zoom_size)
    x0 = max(0, x1 - zoom_size)
    return array[y0:y1, x0:x1]


def find_roi_center(target_map: np.ndarray, valid_mask: np.ndarray, reference_error: np.ndarray) -> Tuple[int, int]:
    error = np.where(valid_mask, reference_error, -np.inf)
    if np.isfinite(error).any():
        y, x = np.unravel_index(int(np.nanargmax(error)), error.shape)
        return int(y), int(x)
    valid_coords = np.argwhere(valid_mask)
    cy, cx = np.median(valid_coords, axis=0)
    return int(cy), int(cx)


def center_valid_coord(valid_mask: np.ndarray) -> Tuple[int, int]:
    coords = np.argwhere(valid_mask)
    center = np.array(valid_mask.shape) / 2.0
    distances = np.sum((coords - center) ** 2, axis=1)
    idx = int(np.argmin(distances))
    return int(coords[idx, 0]), int(coords[idx, 1])


def max_peak_coord(target_map: np.ndarray, valid_mask: np.ndarray) -> Tuple[int, int]:
    arr = np.where(valid_mask, target_map, -np.inf)
    y, x = np.unravel_index(int(np.nanargmax(arr)), arr.shape)
    return int(y), int(x)


def plot_spectrum_comparison(
    output_path: Path,
    sample_id: str,
    coord: Tuple[int, int],
    axis_values: np.ndarray,
    curves: Dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    palette = {
        "Ground Truth": "#1b4965",
        "Masked Input": "#2a9d8f",
        "Linear Interpolation": "#8d99ae",
        "TCN": "#6a4c93",
        "Single-Modal THz Baseline": "#f4a261",
        "PPF-Net (Obs. 70%)": "#c1121f",
    }
    for label, values in curves.items():
        linewidth = 2.4 if label in {"Ground Truth", "PPF-Net (Obs. 70%)"} else 1.8
        ax.plot(axis_values, values, label=label, linewidth=linewidth, color=palette.get(label, None))
    ax.set_title("{0} @ (y={1}, x={2})".format(sample_id, coord[0], coord[1]))
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Reflectance Spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def make_key_panel(
    output_path: Path,
    images: List[Tuple[str, Path]],
) -> None:
    opened = [(label, Image.open(path).convert("RGBA")) for label, path in images]
    panel_w = max(img.width for _, img in opened)
    panel_h = max(img.height for _, img in opened)
    cols = 3
    rows = math.ceil(len(opened) / cols)
    canvas = Image.new("RGBA", (cols * panel_w, rows * panel_h), (255, 255, 255, 255))
    for idx, (label, img) in enumerate(opened):
        row = idx // cols
        col = idx % cols
        x = col * panel_w
        y = row * panel_h
        canvas.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_sample_metrics_table(output_root: Path, sample_id: str, metrics: Dict[str, Dict[str, float]]) -> None:
    rows = [
        {
            "model": MODEL_SPECS[key]["label"],
            "mae": values["mae"],
            "rmse": values["rmse"],
            "psnr": values["psnr"],
        }
        for key, values in metrics.items()
    ]
    rows.sort(key=lambda item: item["psnr"], reverse=True)

    csv_path = output_root / "sample_metrics_summary.csv"
    md_path = output_root / "sample_metrics_summary.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "mae", "rmse", "psnr"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "mae": "{0:.4f}".format(row["mae"]),
                    "rmse": "{0:.4f}".format(row["rmse"]),
                    "psnr": "{0:.4f}".format(row["psnr"]),
                }
            )

    lines = [
        "# Sample Metrics Summary",
        "",
        "Sample ID: `{0}`".format(sample_id),
        "",
        "| Model | MAE | RMSE | PSNR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {0} | {1:.4f} | {2:.4f} | {3:.4f} |".format(
                row["model"], row["mae"], row["rmse"], row["psnr"]
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    sample_id = choose_representative_sample() if args.sample_id == "auto" else args.sample_id
    output_root = args.output_dir / sample_id
    output_root.mkdir(parents=True, exist_ok=True)

    bundles = {
        key: load_bundle(spec["prediction_root"], sample_id)
        for key, spec in MODEL_SPECS.items()
    }
    metrics = {
        key: metric_for_sample(spec["analysis_dir"], sample_id)
        for key, spec in MODEL_SPECS.items()
    }

    ref = bundles["ppf_net_obs70"]
    target_maps, _ = compute_thz_image_maps(
        ref["target_cube"], ref["axis_values"], ref["valid_mask"], slice_values=[1.0, 2.0, 3.0], band_ranges=[(0.8, 1.2), (1.8, 2.2)]
    )
    masked_maps, _ = compute_thz_image_maps(
        ref["masked_cube"], ref["axis_values"], ref["valid_mask"], slice_values=[1.0, 2.0, 3.0], band_ranges=[(0.8, 1.2), (1.8, 2.2)]
    )

    reconstructed_maps: Dict[str, Dict[str, np.ndarray]] = {}
    error_maps: Dict[str, Dict[str, np.ndarray]] = {}
    for key, bundle in bundles.items():
        recon_maps, _ = compute_thz_image_maps(
            bundle["reconstructed_cube"],
            bundle["axis_values"],
            bundle["valid_mask"],
            slice_values=[1.0, 2.0, 3.0],
            band_ranges=[(0.8, 1.2), (1.8, 2.2)],
        )
        reconstructed_maps[key] = recon_maps
        error_maps[key] = compute_image_map_errors(target_maps, recon_maps, bundle["valid_mask"])

    roi_center = find_roi_center(
        target_maps[args.map_name],
        ref["valid_mask"],
        error_maps["single_modal_thz_baseline"][args.map_name],
    )
    center_coord = center_valid_coord(ref["valid_mask"])
    peak_coord = max_peak_coord(target_maps["peak_to_peak"], ref["valid_mask"])
    spectrum_coords = [roi_center, center_coord, peak_coord]
    spectrum_labels = ["roi_center", "valid_center", "peak_response"]

    # Common folders
    gt_dir = output_root / "ground_truth"
    masked_dir = output_root / "masked_input"
    comparison_dir = output_root / "comparison"

    save_map_with_annotation(target_maps[args.map_name], ref["valid_mask"], gt_dir / f"{args.map_name}.png", "Ground Truth")
    save_map_with_annotation(masked_maps[args.map_name], ref["valid_mask"], masked_dir / f"{args.map_name}.png", "Masked Input")
    save_map_with_annotation(
        crop_zoom(target_maps[args.map_name], roi_center[0], roi_center[1], args.zoom_size),
        np.ones_like(crop_zoom(target_maps[args.map_name], roi_center[0], roi_center[1], args.zoom_size), dtype=bool),
        gt_dir / f"{args.map_name}_zoom.png",
        "Ground Truth ROI",
    )
    save_map_with_annotation(
        crop_zoom(masked_maps[args.map_name], roi_center[0], roi_center[1], args.zoom_size),
        np.ones_like(crop_zoom(masked_maps[args.map_name], roi_center[0], roi_center[1], args.zoom_size), dtype=bool),
        masked_dir / f"{args.map_name}_zoom.png",
        "Masked Input ROI",
    )

    key_panel_items: List[Tuple[str, Path]] = [
        ("Ground Truth", gt_dir / f"{args.map_name}.png"),
        ("Masked Input", masked_dir / f"{args.map_name}.png"),
    ]

    for key, spec in MODEL_SPECS.items():
        model_dir = output_root / key
        model_dir.mkdir(parents=True, exist_ok=True)

        recon_path = model_dir / f"{args.map_name}_reconstructed.png"
        error_path = model_dir / f"{args.map_name}_error.png"
        zoom_path = model_dir / f"{args.map_name}_zoom.png"

        save_map_with_annotation(
            reconstructed_maps[key][args.map_name],
            ref["valid_mask"],
            recon_path,
            spec["label"],
            metrics[key],
        )
        save_map_with_annotation(
            error_maps[key][args.map_name],
            ref["valid_mask"],
            error_path,
            spec["label"] + " Error",
            metrics[key],
        )
        zoom = crop_zoom(reconstructed_maps[key][args.map_name], roi_center[0], roi_center[1], args.zoom_size)
        save_map_with_annotation(
            zoom,
            np.ones_like(zoom, dtype=bool),
            zoom_path,
            spec["label"] + " ROI",
            metrics[key],
        )

        (model_dir / "sample_metrics.json").write_text(json.dumps(metrics[key], ensure_ascii=False, indent=2), encoding="utf-8")
        if key in {"single_modal_thz_baseline", "ppf_net_obs70"}:
            key_panel_items.append((spec["label"], recon_path))
        if key == "ppf_net_obs70":
            key_panel_items.append(("Error Map", error_path))

    for coord, label in zip(spectrum_coords, spectrum_labels):
        curves = {
            "Ground Truth": ref["target_cube"][coord[0], coord[1]],
            "Masked Input": ref["masked_cube"][coord[0], coord[1]],
            "Linear Interpolation": bundles["linear_interpolation"]["reconstructed_cube"][coord[0], coord[1]],
            "TCN": bundles["tcn"]["reconstructed_cube"][coord[0], coord[1]],
            "Single-Modal THz Baseline": bundles["single_modal_thz_baseline"]["reconstructed_cube"][coord[0], coord[1]],
            "PPF-Net (Obs. 70%)": bundles["ppf_net_obs70"]["reconstructed_cube"][coord[0], coord[1]],
        }
        plot_spectrum_comparison(
            comparison_dir / f"spectrum_curves_{label}.png",
            sample_id,
            coord,
            ref["axis_values"],
            curves,
        )

    key_panel_items.append(("Spectrum Curves", comparison_dir / "spectrum_curves_roi_center.png"))
    make_key_panel(comparison_dir / "key_visual_panel.png", key_panel_items)
    write_sample_metrics_table(output_root, sample_id, metrics)

    summary = {
        "sample_id": sample_id,
        "map_name": args.map_name,
        "roi_center_yx": [int(roi_center[0]), int(roi_center[1])],
        "spectrum_coords_yx": [[int(y), int(x)] for y, x in spectrum_coords],
        "models": {key: spec["label"] for key, spec in MODEL_SPECS.items()},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("sample_id:", sample_id)
    print("output_root:", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
