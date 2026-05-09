#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage1_spectral_unet import build_spectral_masked_inputs
from ppfnet.thz_csv import assemble_cube_from_pixel_spectra, extract_valid_pixel_spectra, load_thz_csv
from ppfnet.thz_imaging import save_image_maps_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one THz CSV as cube-style image maps with quarter-wise spectral occlusion.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("dataset/thz_seed_only/FS/E/0d-1_Alignabs_seed002.csv"),
        help="Path to the THz CSV cube.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/single_cube_occlusion/0d-1_Alignabs_seed002"),
        help="Directory for output images and metadata.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["point", "band", "hybrid"],
        default="hybrid",
        help="Same spectral masking mode used during training.",
    )
    parser.add_argument(
        "--occlusion-ratios",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Fractions hidden in quarter 2/3/4. Quarter 1 stays unmasked.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_quarter_edges(length: int, segment_count: int = 4) -> np.ndarray:
    return np.linspace(0, length, num=segment_count + 1, dtype=int)


def quarter_band_map(cube: np.ndarray, valid_mask: np.ndarray, start: int, end: int) -> np.ndarray:
    quarter = cube[..., start:end]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data = np.nanmean(quarter, axis=-1).astype(np.float32)
    data[~valid_mask] = np.nan
    return data


def build_composite_masked_cube(
    spectra: np.ndarray,
    axis_values: np.ndarray,
    coords_yx: np.ndarray,
    height: int,
    width: int,
    mask_mode: str,
    occlusion_ratios: Sequence[float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    spectral_length = int(spectra.shape[1])
    segment_count = len(occlusion_ratios) + 1
    edges = compute_quarter_edges(spectral_length, segment_count=segment_count)

    masked_spectra = spectra.copy()
    observed_masks = np.ones_like(spectra, dtype=np.float32)
    segment_labels = ["Q1: 0% occlusion"]

    device = torch.device("cpu")
    for idx, occlusion_ratio in enumerate(occlusion_ratios, start=1):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        segment_labels.append("Q{0}: {1:.0%} occlusion".format(idx + 1, float(occlusion_ratio)))
        if end <= start:
            continue

        observed_ratio = 1.0 - float(occlusion_ratio)
        if not (0.0 < observed_ratio < 1.0):
            raise ValueError("Each occlusion ratio must be between 0 and 1, exclusive: {0}".format(occlusion_ratio))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + idx)
        batch = {
            "spectrum": torch.from_numpy(spectra[:, None, start:end]).to(device=device, dtype=torch.float32),
            "axis_values": torch.from_numpy(np.repeat(axis_values[None, start:end], spectra.shape[0], axis=0)).to(
                device=device,
                dtype=torch.float32,
            ),
        }
        masked = build_spectral_masked_inputs(
            batch,
            mask_mode=mask_mode,
            min_observed_ratio=observed_ratio,
            max_observed_ratio=observed_ratio,
            use_axis_channel=True,
            generator=generator,
        )
        masked_spectra[:, start:end] = masked["masked_spectrum"].squeeze(1).detach().cpu().numpy()
        observed_masks[:, start:end] = masked["observed_mask"].squeeze(1).detach().cpu().numpy()

    masked_cube = assemble_cube_from_pixel_spectra(coords_yx, masked_spectra, height, width)
    observed_cube = assemble_cube_from_pixel_spectra(coords_yx, observed_masks, height, width, fill_value=np.nan)
    return masked_cube, observed_cube, segment_labels, edges


def build_segment_maps(
    cube: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
    segment_labels: Sequence[str],
) -> Dict[str, np.ndarray]:
    maps: Dict[str, np.ndarray] = {"valid_mask": valid_mask.astype(np.float32)}
    for idx, _ in enumerate(segment_labels):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        maps["quarter_{0}".format(idx + 1)] = quarter_band_map(cube, valid_mask, start, end)
    return maps


def build_observed_ratio_maps(
    observed_cube: np.ndarray,
    valid_mask: np.ndarray,
    edges: np.ndarray,
) -> Dict[str, np.ndarray]:
    maps: Dict[str, np.ndarray] = {"valid_mask": valid_mask.astype(np.float32)}
    for idx in range(len(edges) - 1):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ratio_map = np.nanmean(observed_cube[..., start:end], axis=-1).astype(np.float32)
        ratio_map[~valid_mask] = np.nan
        maps["quarter_{0}".format(idx + 1)] = ratio_map
    return maps


def main() -> int:
    args = parse_args()
    cube_data = load_thz_csv(args.input_csv)
    coords_yx, spectra = extract_valid_pixel_spectra(cube_data)
    if spectra.shape[0] == 0:
        raise ValueError("No valid spectra found in {0}".format(args.input_csv))

    target_cube = cube_data.cube.astype(np.float32)
    masked_cube, observed_cube, segment_labels, edges = build_composite_masked_cube(
        spectra=spectra,
        axis_values=cube_data.axis_values.astype(np.float32),
        coords_yx=coords_yx,
        height=target_cube.shape[0],
        width=target_cube.shape[1],
        mask_mode=args.mask_mode,
        occlusion_ratios=args.occlusion_ratios,
        seed=args.seed,
    )

    masked_maps = build_segment_maps(masked_cube, cube_data.valid_mask, edges, segment_labels)
    observed_ratio_maps = build_observed_ratio_maps(observed_cube, cube_data.valid_mask, edges)

    masked_dir = args.output_dir / "masked_segment_maps"
    observed_dir = args.output_dir / "observed_ratio_maps"
    save_image_maps_png(masked_maps, cube_data.valid_mask, masked_dir)
    save_image_maps_png(observed_ratio_maps, cube_data.valid_mask, observed_dir)

    metadata = {
        "input_csv": str(args.input_csv.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "mask_mode": args.mask_mode,
        "cube_shape": [int(v) for v in target_cube.shape],
        "valid_pixels": int(coords_yx.shape[0]),
        "segment_labels": segment_labels,
        "segment_edges": [int(v) for v in edges.tolist()],
        "occlusion_ratios": [0.0] + [float(v) for v in args.occlusion_ratios],
        "masked_image_paths": {
            "quarter_1": str((masked_dir / "quarter_1.png").resolve()),
            "quarter_2": str((masked_dir / "quarter_2.png").resolve()),
            "quarter_3": str((masked_dir / "quarter_3.png").resolve()),
            "quarter_4": str((masked_dir / "quarter_4.png").resolve()),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved:", args.output_dir.resolve())
    print("image_1:", (masked_dir / "quarter_1.png").resolve())
    print("image_2:", (masked_dir / "quarter_2.png").resolve())
    print("image_3:", (masked_dir / "quarter_3.png").resolve())
    print("image_4:", (masked_dir / "quarter_4.png").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

