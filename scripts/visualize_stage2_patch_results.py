#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage1_spectral_unet import SpectralUNet1D, build_spectral_masked_inputs
from ppfnet.stage2_rgb_fs_patch_dataset import Stage2RGBFSPatchDataset
from ppfnet.stage2_rgb_fs_patch_model import PatchContextResidualStudent
from ppfnet.thz_csv import assemble_cube_from_pixel_spectra, extract_valid_pixel_spectra, load_thz_csv, resolve_repo_relative_path
from ppfnet.thz_imaging import compute_image_map_errors, compute_thz_image_maps, save_image_maps_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize stage2 patch-based RGB+FS reconstruction results.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/ppfnet_stage2_rgb_fs_patch_student/checkpoints/stage2_rgb_fs_patch_student_best.pt"),
        help="Stage2 patch checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/ppfnet_stage2/splits/test_pairs.csv"),
        help="Manifest CSV used for sample selection.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ppfnet_stage2_rgb_fs_patch_student/predictions/paper_visuals"),
        help="Directory where visualizations are written.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-mode", choices=["point", "band", "hybrid"], default="hybrid")
    parser.add_argument("--min-observed-ratio", type=float, default=0.25)
    parser.add_argument("--max-observed-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--maps",
        nargs="*",
        default=["peak_to_peak", "max_value", "band_0.800000_1.200000", "band_1.800000_2.200000"],
        help="Image map names to include in the paper figure board.",
    )
    return parser.parse_args()


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest CSV is empty: {0}".format(path))
    return rows


def make_board(
    image_dir_map: Dict[str, Path],
    map_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    font = ImageFont.load_default()
    panel_groups = ["images_target", "images_masked", "images_reconstructed", "images_error"]
    header_labels = ["Target", "Masked", "Reconstructed", "Error"]

    loaded: List[List[Image.Image]] = []
    for map_name in map_names:
        row_images = []
        for group_name in panel_groups:
            image_path = image_dir_map[group_name] / "{0}.png".format(map_name)
            row_images.append(Image.open(image_path).convert("RGBA"))
        loaded.append(row_images)

    panel_w = max(img.width for row in loaded for img in row)
    panel_h = max(img.height for row in loaded for img in row)
    row_label_w = 180
    top_h = 28
    title_h = 22
    canvas_w = row_label_w + panel_w * len(panel_groups)
    canvas_h = title_h + top_h + panel_h * len(map_names)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((6, 2), title, fill=(0, 0, 0, 255), font=font)
    for col_idx, header in enumerate(header_labels):
        x = row_label_w + col_idx * panel_w + 6
        draw.text((x, title_h + 2), header, fill=(0, 0, 0, 255), font=font)

    for row_idx, map_name in enumerate(map_names):
        y = title_h + top_h + row_idx * panel_h
        draw.text((6, y + 6), map_name, fill=(0, 0, 0, 255), font=font)
        for col_idx, img in enumerate(loaded[row_idx]):
            x = row_label_w + col_idx * panel_w
            canvas.paste(img, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def load_models(checkpoint_path: Path) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
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

    return {
        "device": device,
        "checkpoint": checkpoint,
        "model_args": model_args,
        "data_info": data_info,
        "teacher": teacher,
        "student": student,
    }


def reconstruct_patch_sample(
    model_bundle: Dict[str, object],
    manifest_rows: List[Dict[str, str]],
    manifest_path: Path,
    sample_index: int,
    batch_size: int,
    mask_mode: str,
    min_observed_ratio: float,
    max_observed_ratio: float,
    seed: int,
) -> Dict[str, object]:
    device = model_bundle["device"]
    teacher = model_bundle["teacher"]
    student = model_bundle["student"]
    model_args = model_bundle["model_args"]
    row = manifest_rows[sample_index]

    dataset = Stage2RGBFSPatchDataset(
        manifest_csv=manifest_path,
        image_size=tuple(model_args.get("image_size", (224, 224))),
        rgb_patch_size=tuple(model_args.get("rgb_patch_size", (64, 64))),
        thz_patch_size=int(model_args.get("thz_patch_size", 7)),
        normalization=model_args.get("normalization", "none"),
        repo_root=REPO_ROOT,
        max_pixels_per_sample=None,
        pixel_selection_seed=int(model_args.get("seed", 42)),
        include_structure_channels=bool(model_bundle["data_info"].get("include_structure_channels", True)),
    )

    indices = [idx for idx, (sample_ref, _) in enumerate(dataset.index_map) if sample_ref == sample_index]
    raw_csv_path = resolve_repo_relative_path(row["fs_raw_csv_path"], REPO_ROOT)
    cube_data = load_thz_csv(raw_csv_path)

    predicted_spectra = []
    masked_spectra = []
    coords = []

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)

    for start in range(0, len(indices), batch_size):
        chunk_indices = indices[start:start + batch_size]
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
            mask_mode=mask_mode,
            min_observed_ratio=min_observed_ratio,
            max_observed_ratio=max_observed_ratio,
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
        coords.extend([(int(item["coord_y"]), int(item["coord_x"])) for item in batch_items])

    coords_array = np.asarray(coords, dtype=np.int32)
    predicted_spectra = np.concatenate(predicted_spectra, axis=0).astype(np.float32)
    masked_spectra = np.concatenate(masked_spectra, axis=0).astype(np.float32)

    target_cube = cube_data.cube.astype(np.float32)
    masked_cube = assemble_cube_from_pixel_spectra(coords_array, masked_spectra, target_cube.shape[0], target_cube.shape[1])
    reconstructed_cube = assemble_cube_from_pixel_spectra(coords_array, predicted_spectra, target_cube.shape[0], target_cube.shape[1])

    return {
        "sample_id": row["sample_id"],
        "cube_data": cube_data,
        "target_cube": target_cube,
        "masked_cube": masked_cube,
        "reconstructed_cube": reconstructed_cube,
    }


def main() -> int:
    args = parse_args()
    rows = load_manifest_rows(args.manifest)
    start_index = args.start_index if args.start_index is not None else args.sample_index
    model_bundle = load_models(args.checkpoint)
    summary: List[Dict[str, object]] = []

    for offset in range(args.num_samples):
        sample_index = start_index + offset
        if sample_index >= len(rows):
            break

        result = reconstruct_patch_sample(
            model_bundle=model_bundle,
            manifest_rows=rows,
            manifest_path=args.manifest,
            sample_index=sample_index,
            batch_size=args.batch_size,
            mask_mode=args.mask_mode,
            min_observed_ratio=args.min_observed_ratio,
            max_observed_ratio=args.max_observed_ratio,
            seed=args.seed + sample_index,
        )

        cube_data = result["cube_data"]
        sample_id = str(result["sample_id"]).replace("/", "__")
        target_cube = result["target_cube"]
        masked_cube = result["masked_cube"]
        reconstructed_cube = result["reconstructed_cube"]

        target_maps, _ = compute_thz_image_maps(
            target_cube,
            cube_data.axis_values,
            cube_data.valid_mask,
            slice_values=[1.0, 2.0, 3.0],
            band_ranges=[(0.8, 1.2), (1.8, 2.2)],
        )
        masked_maps, _ = compute_thz_image_maps(
            masked_cube,
            cube_data.axis_values,
            cube_data.valid_mask,
            slice_values=[1.0, 2.0, 3.0],
            band_ranges=[(0.8, 1.2), (1.8, 2.2)],
        )
        reconstructed_maps, _ = compute_thz_image_maps(
            reconstructed_cube,
            cube_data.axis_values,
            cube_data.valid_mask,
            slice_values=[1.0, 2.0, 3.0],
            band_ranges=[(0.8, 1.2), (1.8, 2.2)],
        )
        error_maps = compute_image_map_errors(target_maps, reconstructed_maps, cube_data.valid_mask)

        sample_dir = args.output_dir / sample_id
        image_dir_map = {
            "images_target": sample_dir / "images_target",
            "images_masked": sample_dir / "images_masked",
            "images_reconstructed": sample_dir / "images_reconstructed",
            "images_error": sample_dir / "images_error",
        }
        save_image_maps_png(target_maps, cube_data.valid_mask, image_dir_map["images_target"])
        save_image_maps_png(masked_maps, cube_data.valid_mask, image_dir_map["images_masked"])
        save_image_maps_png(reconstructed_maps, cube_data.valid_mask, image_dir_map["images_reconstructed"])
        save_image_maps_png(error_maps, cube_data.valid_mask, image_dir_map["images_error"])

        board_path = sample_dir / "paper_board.png"
        make_board(
            image_dir_map=image_dir_map,
            map_names=args.maps,
            output_path=board_path,
            title="Stage2 RGB+FS Patch Reconstruction: {0}".format(sample_id),
        )

        metadata = {
            "sample_id": sample_id,
            "sample_index": sample_index,
            "maps": list(args.maps),
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "paper_board": str(board_path),
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary.append(metadata)
        print("saved_sample_dir:", sample_dir)

    summary_path = args.output_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

