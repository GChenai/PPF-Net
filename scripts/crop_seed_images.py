#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop RGB seed images to the seed region and preserve folder structure."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("datasets/images"),
        help="Root directory containing original RGB images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/images_cropped"),
        help="Root directory where cropped RGB images will be written.",
    )
    parser.add_argument(
        "--border-fraction",
        type=float,
        default=0.08,
        help="Fraction of the image border used to estimate the background.",
    )
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.12,
        help="Relative crop margin added around the detected seed box.",
    )
    parser.add_argument(
        "--square",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to expand the crop to a square box. Default: true",
    )
    parser.add_argument(
        "--save-mask-preview",
        action="store_true",
        help="Save binary mask previews next to cropped images.",
    )
    parser.add_argument(
        "--foreground-only",
        action="store_true",
        help="Keep only masked seed pixels inside the crop. Outside-mask pixels are replaced by background-color.",
    )
    parser.add_argument(
        "--background-color",
        choices=["white", "black"],
        default="white",
        help="Background fill color used when --foreground-only is enabled.",
    )
    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Save a tightly cropped PNG with transparent background outside the seed mask.",
    )
    return parser.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def border_mask(height: int, width: int, fraction: float) -> np.ndarray:
    border = max(4, int(round(min(height, width) * fraction)))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return mask


def compute_foreground_mask(rgb: np.ndarray, border_fraction: float) -> np.ndarray:
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)

    h, w = gray.shape
    bmask = border_mask(h, w, border_fraction)
    border_gray = gray[bmask]
    border_chroma = chroma[bmask]

    bg_median = float(np.median(border_gray))
    bg_std = float(np.std(border_gray))
    gray_threshold = bg_median - max(0.05, 1.5 * bg_std)

    chroma_threshold = max(
        float(np.quantile(border_chroma, 0.995)) + 0.015,
        float(np.mean(border_chroma) + 3.0 * np.std(border_chroma)),
    )

    mask = gray < gray_threshold
    mask |= ((gray < (bg_median - max(0.025, 0.75 * bg_std))) & (chroma > chroma_threshold))

    kernel = max(3, int(round(min(h, w) * 0.004)))
    if kernel % 2 == 0:
        kernel += 1

    close_structure = np.ones((kernel, kernel), dtype=bool)
    open_kernel = max(3, kernel // 2)
    if open_kernel % 2 == 0:
        open_kernel += 1
    open_structure = np.ones((open_kernel, open_kernel), dtype=bool)

    mask = ndimage.binary_closing(mask, structure=close_structure)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=open_structure)
    mask = ndimage.binary_fill_holes(mask)

    labels, num_labels = ndimage.label(mask)
    if num_labels <= 0:
        raise ValueError("No foreground component detected.")

    component_ids = np.arange(1, num_labels + 1)
    areas = ndimage.sum(mask, labels=labels, index=component_ids)
    largest_id = int(component_ids[int(np.argmax(areas))])
    return labels == largest_id


def expand_bbox(
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    height: int,
    width: int,
    margin_ratio: float,
    square: bool,
) -> Tuple[int, int, int, int]:
    box_h = y1 - y0 + 1
    box_w = x1 - x0 + 1
    margin_y = max(2, int(round(box_h * margin_ratio)))
    margin_x = max(2, int(round(box_w * margin_ratio)))

    y0 = max(0, y0 - margin_y)
    x0 = max(0, x0 - margin_x)
    y1 = min(height - 1, y1 + margin_y)
    x1 = min(width - 1, x1 + margin_x)

    if not square:
        return y0, x0, y1, x1

    box_h = y1 - y0 + 1
    box_w = x1 - x0 + 1
    side = max(box_h, box_w)
    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)

    y0 = int(round(cy - side / 2))
    x0 = int(round(cx - side / 2))
    y1 = y0 + side - 1
    x1 = x0 + side - 1

    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y1 >= height:
        shift = y1 - height + 1
        y0 = max(0, y0 - shift)
        y1 = height - 1
    if x1 >= width:
        shift = x1 - width + 1
        x0 = max(0, x0 - shift)
        x1 = width - 1

    return y0, x0, y1, x1


def crop_from_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    margin_ratio: float,
    square: bool,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Mask is empty.")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0, x0, y1, x1 = expand_bbox(
        int(y0), int(x0), int(y1), int(x1),
        height=rgb.shape[0],
        width=rgb.shape[1],
        margin_ratio=margin_ratio,
        square=square,
    )
    cropped = rgb[y0:y1 + 1, x0:x1 + 1]
    cropped_mask = mask[y0:y1 + 1, x0:x1 + 1]
    return cropped, cropped_mask, (y0, x0, y1, x1)


def apply_foreground_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    background_color: str,
) -> np.ndarray:
    if background_color == "white":
        background = np.ones_like(rgb, dtype=np.float32)
    elif background_color == "black":
        background = np.zeros_like(rgb, dtype=np.float32)
    else:
        raise ValueError("Unsupported background_color: {0}".format(background_color))

    mask_3d = mask[..., None].astype(np.float32)
    return rgb * mask_3d + background * (1.0 - mask_3d)


def apply_alpha_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.float32) * 255.0)[..., None]
    rgba = np.concatenate(
        [
            np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8),
            np.clip(np.round(alpha), 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    return rgba


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb.ndim != 3:
        raise ValueError("Image array must have shape [H, W, C].")
    if rgb.shape[-1] == 4:
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGBA")
    else:
        image = Image.fromarray(np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8), mode="RGB")
    image.save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image.save(path)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    image_paths = list(iter_images(args.input_root))

    for image_path in image_paths:
        rgb = load_rgb(image_path)
        mask = compute_foreground_mask(rgb, border_fraction=args.border_fraction)
        tight_mode = args.foreground_only or args.transparent_background
        margin_ratio = 0.0 if tight_mode else args.margin_ratio
        square = False if tight_mode else args.square
        cropped, cropped_mask, bbox = crop_from_mask(
            rgb,
            mask,
            margin_ratio=margin_ratio,
            square=square,
        )

        relative = image_path.relative_to(args.input_root)
        output_path = args.output_root / relative

        if args.transparent_background:
            cropped_to_save = apply_alpha_mask(cropped, cropped_mask)
            output_path = output_path.with_suffix(".png")
        elif args.foreground_only:
            cropped = apply_foreground_mask(
                cropped,
                cropped_mask,
                background_color=args.background_color,
            )
            cropped_to_save = cropped
        else:
            cropped_to_save = cropped

        save_rgb(output_path, cropped_to_save)

        if args.save_mask_preview:
            mask_path = output_path.with_name(output_path.stem + "_mask.png")
            save_mask(mask_path, mask)

        rows.append(
            {
                "sample_id": "{0}/{1}".format(image_path.parent.name, image_path.stem),
                "input_path": str(image_path),
                "output_path": str(output_path),
                "orig_height": int(rgb.shape[0]),
                "orig_width": int(rgb.shape[1]),
                "crop_y0": bbox[0],
                "crop_x0": bbox[1],
                "crop_y1": bbox[2],
                "crop_x1": bbox[3],
                "crop_height": int(cropped.shape[0]),
                "crop_width": int(cropped.shape[1]),
            }
        )

    manifest_path = args.output_root / "crop_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "input_path",
                "output_path",
                "orig_height",
                "orig_width",
                "crop_y0",
                "crop_x0",
                "crop_y1",
                "crop_x1",
                "crop_height",
                "crop_width",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("processed_images:", len(rows))
    print("output_root:", args.output_root)
    print("manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
