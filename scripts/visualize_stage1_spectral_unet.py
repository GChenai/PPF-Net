#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage1_spectral_dataset import Stage1SpectralDataset
from ppfnet.stage1_spectral_unet import SpectralUNet1D, build_spectral_masked_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize spectral reconstructions.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/ppfnet_stage1_spectral_fs/checkpoints/stage1_spectral_unet_best.pt"),
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
        default=Path("outputs/ppfnet_stage1_spectral_fs/predictions/visualizations"),
        help="Directory where visualization PNG files are written.",
    )
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--mask-mode", choices=["point", "band", "hybrid"], default="hybrid")
    parser.add_argument("--min-observed-ratio", type=float, default=0.25)
    parser.add_argument("--max-observed-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _plot_sample(
    output_path: Path,
    sample_id: str,
    axis_values,
    ground_truth,
    masked_input,
    reconstruction,
    observed_mask,
) -> None:
    gt = ground_truth.squeeze()
    masked = masked_input.squeeze()
    recon = reconstruction.squeeze()
    observed = observed_mask.squeeze() > 0.5
    missing = ~observed

    fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
    ax.plot(axis_values, gt, label="ground_truth", linewidth=2.0, color="#1b4965")
    ax.plot(axis_values, recon, label="reconstruction", linewidth=2.0, color="#c1121f")
    ax.scatter(axis_values[observed], masked[observed], s=14, label="observed_input", color="#2a9d8f", zorder=3)
    if missing.any():
        ax.scatter(axis_values[missing], recon[missing], s=10, label="reconstructed_points", color="#ffb703", zorder=3)

    ax.set_title(sample_id)
    ax.set_xlabel("Axis")
    ax.set_ylabel("Spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    data_info = checkpoint["data_info"]

    dataset = Stage1SpectralDataset(
        manifest_csv=args.manifest,
        modality=data_info["modality"],
        spectrum_reduction=data_info.get("spectrum_reduction", "mean"),
        normalization=data_info.get("normalization", checkpoint_args.get("normalization", "none")),
        repo_root=REPO_ROOT,
    )

    model = SpectralUNet1D(
        in_channels=3,
        base_channels=int(checkpoint_args.get("base_channels", 32)),
        dropout=float(checkpoint_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(args.seed)

    summary: List[Dict[str, object]] = []

    for offset in range(args.num_samples):
        sample_index = args.start_index + offset
        if sample_index >= len(dataset):
            break

        item = dataset[sample_index]
        batch = {
            "spectrum": item["spectrum"].unsqueeze(0).to(device),
            "axis_values": item["axis_values"].unsqueeze(0).to(device),
        }

        masked = build_spectral_masked_inputs(
            batch,
            mask_mode=args.mask_mode,
            min_observed_ratio=args.min_observed_ratio,
            max_observed_ratio=args.max_observed_ratio,
            use_axis_channel=True,
            generator=generator,
        )

        with torch.no_grad():
            prediction = model(masked["model_input"])
            reconstruction = masked["masked_spectrum"] + prediction * (1.0 - masked["observed_mask"])

        axis_values = batch["axis_values"][0].detach().cpu().numpy()
        ground_truth = batch["spectrum"][0].detach().cpu().numpy()
        masked_input = masked["masked_spectrum"][0].detach().cpu().numpy()
        recon = reconstruction[0].detach().cpu().numpy()
        observed_mask = masked["observed_mask"][0].detach().cpu().numpy()

        sample_dir = args.output_dir / str(item["split"])
        output_path = sample_dir / "{0}.png".format(str(item["sample_id"]).replace("/", "__"))
        _plot_sample(
            output_path=output_path,
            sample_id=str(item["sample_id"]),
            axis_values=axis_values,
            ground_truth=ground_truth,
            masked_input=masked_input,
            reconstruction=recon,
            observed_mask=observed_mask,
        )

        summary.append(
            {
                "sample_index": sample_index,
                "sample_id": item["sample_id"],
                "split": item["split"],
                "output_path": str(output_path),
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

