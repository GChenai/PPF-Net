#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage2_rgb_fs_patch_dataset import Stage2RGBFSPatchDataset
from ppfnet.stage2_rgb_fs_patch_model import PatchContextResidualStudent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RGB-branch visualizations as separate images."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/stage2_rgb_fs_patch_student_obs70/checkpoints/stage2_rgb_fs_patch_student_best.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/stage2/splits/test_pairs.csv"),
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=["A/1", "C/64", "D/83", "E/613"],
        help="Sample ids to export. Example: C/64 D/83",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rgb_branch_visuals"),
    )
    return parser.parse_args()


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest CSV is empty: {path}")
    return rows


def find_sample_index(rows: List[Dict[str, str]], sample_id: str) -> int:
    for idx, row in enumerate(rows):
        if row.get("sample_id") == sample_id:
            return idx
    raise KeyError(f"Sample id not found in manifest: {sample_id}")


def choose_center_item(dataset: Stage2RGBFSPatchDataset, sample_index: int) -> Dict[str, object]:
    sample = dataset.samples[sample_index]
    coords = sample["coords_yx"]
    center = np.array(
        [
            float(np.mean(coords[:, 0])),
            float(np.mean(coords[:, 1])),
        ],
        dtype=np.float32,
    )
    distances = np.sum((coords.astype(np.float32) - center[None, :]) ** 2, axis=1)
    pixel_index = int(np.argmin(distances))

    for idx, (sample_ref, pix_ref) in enumerate(dataset.index_map):
        if sample_ref == sample_index and pix_ref == pixel_index:
            return dataset[idx]
    raise RuntimeError("Failed to resolve sample pixel index.")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[PatchContextResidualStudent, Dict[str, object], Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint["args"]
    data_info = checkpoint["data_info"]
    model = PatchContextResidualStudent(
        rgb_in_channels=int(data_info.get("rgb_channels", 6)),
        rgb_embed_dim=int(args.get("rgb_embed_dim", 64)),
        cond_channels=int(args.get("cond_channels", 16)),
        base_channels=int(args.get("base_channels", 32)),
        dropout=float(args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, args, data_info


def save_rgb_image(array: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=220)
    ax.imshow(np.clip(array, 0.0, 1.0))
    ax.axis("off")
    fig.tight_layout(pad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_gray_image(array: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=220)
    ax.imshow(array, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.tight_layout(pad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_heatmap_overlay(rgb: np.ndarray, heatmap: np.ndarray, path: Path) -> None:
    cmap = plt.get_cmap("inferno")
    norm = Normalize(vmin=float(np.min(heatmap)), vmax=float(np.max(heatmap)))
    colored = cmap(norm(heatmap))[..., :3]
    overlay = np.clip(0.58 * rgb + 0.42 * colored, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=220)
    ax.imshow(overlay)
    ax.axis("off")
    fig.tight_layout(pad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_args, data_info = load_model(args.checkpoint, device)
    rows = load_manifest_rows(args.manifest)

    dataset = Stage2RGBFSPatchDataset(
        manifest_csv=args.manifest,
        image_size=tuple(model_args.get("image_size", (224, 224))),
        rgb_patch_size=tuple(model_args.get("rgb_patch_size", (64, 64))),
        thz_patch_size=int(model_args.get("thz_patch_size", 7)),
        normalization=model_args.get("normalization", "none"),
        repo_root=REPO_ROOT,
        max_pixels_per_sample=None,
        pixel_selection_seed=int(model_args.get("seed", 42)),
        include_structure_channels=bool(data_info.get("include_structure_channels", True)),
    )

    summary: List[Dict[str, object]] = []
    for sample_id in args.sample_ids:
        sample_index = find_sample_index(rows, sample_id)
        item = choose_center_item(dataset, sample_index)
        rgb_patch = item["rgb_patch"].unsqueeze(0).to(device)

        with torch.no_grad():
            feature_map = model.rgb_encoder.stem(rgb_patch)
            heatmap = feature_map.abs().mean(dim=1, keepdim=False)
            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(1),
                size=rgb_patch.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze().detach().cpu().numpy().astype(np.float32)

        rgb_patch_np = item["rgb_patch"].detach().cpu().numpy().astype(np.float32)
        rgb = np.transpose(rgb_patch_np[:3], (1, 2, 0))
        seed_mask = rgb_patch_np[3] if rgb_patch_np.shape[0] > 3 else np.ones(rgb.shape[:2], dtype=np.float32)
        edge = rgb_patch_np[4] if rgb_patch_np.shape[0] > 4 else np.zeros(rgb.shape[:2], dtype=np.float32)
        distance = rgb_patch_np[5] if rgb_patch_np.shape[0] > 5 else np.zeros(rgb.shape[:2], dtype=np.float32)

        sample_dir = args.output_dir / sample_id.replace("/", "__")
        rgb_path = sample_dir / "rgb_patch.png"
        mask_path = sample_dir / "seed_mask.png"
        edge_path = sample_dir / "edge_prior.png"
        dist_path = sample_dir / "distance_prior.png"
        heat_path = sample_dir / "feature_heatmap.png"

        save_rgb_image(rgb, rgb_path)
        save_gray_image(seed_mask, mask_path)
        save_gray_image(edge, edge_path)
        save_gray_image(distance, dist_path)
        save_heatmap_overlay(rgb, heatmap, heat_path)

        summary.append(
            {
                "sample_id": sample_id,
                "coord_y": int(item["coord_y"]),
                "coord_x": int(item["coord_x"]),
                "rgb_patch": str(rgb_path),
                "seed_mask": str(mask_path),
                "edge_prior": str(edge_path),
                "distance_prior": str(dist_path),
                "feature_heatmap": str(heat_path),
            }
        )
        print("saved sample:", sample_id, "->", sample_dir)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
