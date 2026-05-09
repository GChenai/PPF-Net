#!/usr/bin/env python
from __future__ import annotations

import argparse
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

from ppfnet import Stage1FeaturePairDataset, Stage1UNet, build_masked_inputs
from ppfnet.stage1_unet import modality_uses_fs, modality_uses_ts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize stage1 U-Net reconstructions on selected samples."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/ppfnet_stage1_feature_unet/checkpoints/stage1_unet_best.pt"),
        help="Checkpoint to load.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/ppfnet_stage1/splits/val_pairs.csv"),
        help="Manifest CSV to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ppfnet_stage1_feature_unet/predictions/visualizations"),
        help="Directory where visualization PNG files are written.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting sample index inside the manifest.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Feature names to visualize. Defaults to a useful subset if omitted.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["pixel", "block", "hybrid"],
        default="hybrid",
        help="Masking mode used during visualization.",
    )
    parser.add_argument(
        "--min-observed-ratio",
        type=float,
        default=0.45,
        help="Minimum observed ratio for masking.",
    )
    parser.add_argument(
        "--max-observed-ratio",
        type=float,
        default=0.85,
        help="Maximum observed ratio for masking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to generate visualization masks.",
    )
    parser.add_argument(
        "--label-style",
        choices=["none", "short", "full"],
        default="short",
        help="How much text to draw on each visualization image.",
    )
    return parser.parse_args()


def _pick_feature_names(all_names: Sequence[str], requested: Sequence[str] | None) -> List[str]:
    if requested:
        missing = [name for name in requested if name not in all_names]
        if missing:
            raise KeyError("Requested feature(s) not found: {0}".format(", ".join(missing)))
        return list(requested)

    preferred = [
        "mean_value",
        "peak_to_peak",
        "band_0.800000_1.200000",
        "band_1.800000_2.200000",
        "slice_1.995850",
    ]
    chosen = [name for name in preferred if name in all_names]
    if not chosen:
        chosen = list(all_names[: min(4, len(all_names))])
    return chosen


def _to_u8(array: np.ndarray, valid_mask: np.ndarray, error_mode: bool = False) -> np.ndarray:
    valid = valid_mask > 0.5
    alpha = np.where(valid, 255, 0).astype(np.uint8)

    if not np.any(valid):
        gray = np.zeros_like(alpha)
        return np.stack([gray, gray, gray, alpha], axis=-1)

    values = array[valid]
    if error_mode:
        low = 0.0
        high = float(np.percentile(values, 99.0))
    else:
        low = float(np.percentile(values, 1.0))
        high = float(np.percentile(values, 99.0))

    if high <= low:
        high = low + 1e-6

    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    gray = np.round(scaled * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray, alpha], axis=-1)


def _tile_with_labels(
    panels: Sequence[np.ndarray],
    labels: Sequence[str],
    title: str,
    label_style: str,
) -> Image.Image:
    images = [Image.fromarray(panel, mode="RGBA") for panel in panels]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    font = ImageFont.load_default()
    margin = 4

    show_title = label_style == "full"
    show_labels = label_style in {"short", "full"}

    title_h = 18 if show_title else 0
    label_h = 18 if show_labels else 0
    header_h = title_h + label_h + (margin * 2 if (show_title or show_labels) else 0)
    canvas = Image.new("RGBA", (width * len(images), height + header_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if show_title:
        draw.text((margin, 2), title, fill=(0, 0, 0, 255), font=font)

    for idx, (image, label) in enumerate(zip(images, labels)):
        x = idx * width
        canvas.paste(image, (x, header_h))
        if show_labels:
            draw.text((x + margin, title_h + margin), label, fill=(0, 0, 0, 255), font=font)

    return canvas


def _save_feature_visual(
    output_path: Path,
    title: str,
    target: np.ndarray,
    masked: np.ndarray,
    recon: np.ndarray,
    valid_mask: np.ndarray,
    label_style: str,
) -> None:
    error = np.abs(recon - target)
    panels = [
        _to_u8(target, valid_mask),
        _to_u8(masked, valid_mask),
        _to_u8(recon, valid_mask),
        _to_u8(error, valid_mask, error_mode=True),
    ]
    if label_style == "full":
        labels = ["ground_truth", "masked_input", "reconstruction", "abs_error"]
    else:
        labels = ["GT", "IN", "REC", "ERR"]
    image = _tile_with_labels(panels, labels=labels, title=title, label_style=label_style)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    feature_info = checkpoint["feature_info"]
    modality_mode = checkpoint_args.get("modality_mode", feature_info.get("modality_mode", "joint"))
    mask_sharing = checkpoint_args.get("mask_sharing", feature_info.get("mask_sharing", "shared"))

    dataset = Stage1FeaturePairDataset(
        manifest_csv=args.manifest,
        fs_feature_names=feature_info["fs_feature_names"],
        ts_feature_names=feature_info["ts_feature_names"],
        normalization=checkpoint_args.get("normalization", "none"),
        spatial_size=tuple(checkpoint_args["spatial_size"]) if checkpoint_args.get("spatial_size") else None,
        include_valid_mask_channel=bool(checkpoint_args.get("include_valid_mask_channel", False)),
    )

    model = Stage1UNet(
        fs_channels=int(feature_info["fs_channels"]),
        ts_channels=int(feature_info["ts_channels"]),
        modality_mode=modality_mode,
        base_channels=int(checkpoint_args.get("base_channels", 32)),
        dropout=float(checkpoint_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fs_feature_names = dataset.fs_feature_names
    ts_feature_names = dataset.ts_feature_names
    active_feature_names = fs_feature_names if modality_mode != "ts_only" else ts_feature_names
    chosen_features = _pick_feature_names(active_feature_names, args.features)

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(args.seed)

    summary: List[Dict[str, object]] = []

    for offset in range(args.num_samples):
        sample_index = args.start_index + offset
        if sample_index >= len(dataset):
            break

        item = dataset[sample_index]
        batch = {
            "fs_features": item["fs_features"].unsqueeze(0).to(device),
            "ts_features": item["ts_features"].unsqueeze(0).to(device),
            "fs_valid_mask": item["fs_valid_mask"].unsqueeze(0).to(device),
            "ts_valid_mask": item["ts_valid_mask"].unsqueeze(0).to(device),
        }

        masked = build_masked_inputs(
            batch,
            modality_mode=modality_mode,
            mask_mode=args.mask_mode,
            mask_sharing=mask_sharing,
            min_observed_ratio=args.min_observed_ratio,
            max_observed_ratio=args.max_observed_ratio,
            generator=generator,
        )

        with torch.no_grad():
            pred_fs, pred_ts = model(
                masked_fs=masked["masked_fs"],
                masked_ts=masked["masked_ts"],
                fs_observed_mask=masked["fs_observed_mask"],
                ts_observed_mask=masked["ts_observed_mask"],
                fs_valid_mask=batch["fs_valid_mask"],
                ts_valid_mask=batch["ts_valid_mask"],
            )

        recon_fs = None
        recon_ts = None
        if pred_fs is not None and masked["masked_fs"] is not None and masked["fs_observed_mask"] is not None:
            recon_fs = masked["masked_fs"] + pred_fs * (1.0 - masked["fs_observed_mask"])
        if pred_ts is not None and masked["masked_ts"] is not None and masked["ts_observed_mask"] is not None:
            recon_ts = masked["masked_ts"] + pred_ts * (1.0 - masked["ts_observed_mask"])

        fs_np = batch["fs_features"][0].detach().cpu().numpy()
        ts_np = batch["ts_features"][0].detach().cpu().numpy()
        masked_fs_np = masked["masked_fs"][0].detach().cpu().numpy() if masked["masked_fs"] is not None else None
        masked_ts_np = masked["masked_ts"][0].detach().cpu().numpy() if masked["masked_ts"] is not None else None
        recon_fs_np = recon_fs[0].detach().cpu().numpy() if recon_fs is not None else None
        recon_ts_np = recon_ts[0].detach().cpu().numpy() if recon_ts is not None else None
        fs_valid_np = batch["fs_valid_mask"][0, 0].detach().cpu().numpy()
        ts_valid_np = batch["ts_valid_mask"][0, 0].detach().cpu().numpy()

        sample_dir = args.output_dir / item["split"] / str(item["pair_id"]).replace("/", "__")
        sample_dir.mkdir(parents=True, exist_ok=True)

        if modality_uses_fs(modality_mode):
            for feature_name in chosen_features:
                channel_idx = fs_feature_names.index(feature_name)
                _save_feature_visual(
                    sample_dir / "FS_{0}.png".format(feature_name),
                    title="FS {0} {1}".format(item["pair_id"], feature_name),
                    target=fs_np[channel_idx],
                    masked=masked_fs_np[channel_idx],
                    recon=recon_fs_np[channel_idx],
                    valid_mask=fs_valid_np,
                    label_style=args.label_style,
                )
        if modality_uses_ts(modality_mode):
            for feature_name in chosen_features:
                channel_idx = ts_feature_names.index(feature_name)
                _save_feature_visual(
                    sample_dir / "TS_{0}.png".format(feature_name),
                    title="TS {0} {1}".format(item["pair_id"], feature_name),
                    target=ts_np[channel_idx],
                    masked=masked_ts_np[channel_idx],
                    recon=recon_ts_np[channel_idx],
                    valid_mask=ts_valid_np,
                    label_style=args.label_style,
                )

        summary.append(
            {
                "sample_index": sample_index,
                "pair_id": item["pair_id"],
                "split": item["split"],
                "modality_mode": modality_mode,
                "mask_sharing": mask_sharing,
                "output_dir": str(sample_dir),
                "features": chosen_features,
            }
        )

    summary_path = args.output_dir / "visualization_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved_visualizations:", len(summary))
    print("summary_path:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

