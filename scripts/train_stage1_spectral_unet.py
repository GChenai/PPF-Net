#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppfnet.stage1_spectral_dataset import create_stage1_spectral_dataloaders
from ppfnet.stage1_spectral_unet import (
    SpectralUNet1D,
    build_spectral_masked_inputs,
    spectral_reconstruction_loss,
)
from ppfnet.stage1_spectral_utils import (
    EarlyStopping,
    accumulate_metrics,
    average_metrics,
    compute_spectral_metrics,
    plot_spectral_training_curves,
    spectral_best_summary_row,
    spectral_epoch_row,
    spectral_training_log_fieldnames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stage1 spectral reconstruction U-Net.")
    parser.add_argument("--train-manifest", type=Path, default=Path("outputs/ppfnet_stage1/splits/train_pairs.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("outputs/ppfnet_stage1/splits/val_pairs.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("outputs/ppfnet_stage1/splits/test_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppfnet_stage1_spectral_fs"))
    parser.add_argument("--modality", choices=["fs", "ts"], default="fs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mask-mode", choices=["point", "band", "hybrid"], default="hybrid")
    parser.add_argument("--min-observed-ratio", type=float, default=0.25)
    parser.add_argument("--max-observed-ratio", type=float, default=0.7)
    parser.add_argument("--l2-weight", type=float, default=0.1)
    parser.add_argument("--normalization", choices=["none", "zscore", "minmax"], default="none")
    parser.add_argument("--spectrum-reduction", choices=["mean", "median"], default="mean")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready_args(args: argparse.Namespace) -> Dict[str, object]:
    ready: Dict[str, object] = {}
    for key, value in vars(args).items():
        ready[key] = str(value) if isinstance(value, Path) else value
    return ready


def json_safe_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    safe: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and math.isnan(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def run_model_step(
    model: SpectralUNet1D,
    batch: Dict[str, object],
    args: argparse.Namespace,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    masked = build_spectral_masked_inputs(
        batch,
        mask_mode=args.mask_mode,
        min_observed_ratio=args.min_observed_ratio,
        max_observed_ratio=args.max_observed_ratio,
        use_axis_channel=True,
        generator=generator,
    )

    prediction = model(masked["model_input"])
    reconstruction = masked["masked_spectrum"] + prediction * (1.0 - masked["observed_mask"])

    loss_output = spectral_reconstruction_loss(
        prediction=reconstruction,
        target=batch["spectrum"],
        missing_mask=masked["missing_mask"],
        l2_weight=args.l2_weight,
    )
    metrics = compute_spectral_metrics(
        loss=loss_output.loss,
        reconstruction=reconstruction,
        target=batch["spectrum"],
        missing_mask=masked["missing_mask"],
    )
    return loss_output.loss, masked, metrics


def train_one_epoch(
    model: SpectralUNet1D,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    running: Dict[str, float] = {}
    step_count = 0

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            loss, _, batch_metrics = run_model_step(model, batch, args)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accumulate_metrics(running, batch_metrics)
        step_count += 1

        if step % args.log_every == 0:
            averaged = average_metrics(running, step_count)
            print(
                "epoch {0} step {1}/{2} loss={3:.4f} psnr={4:.2f}".format(
                    epoch,
                    step,
                    len(dataloader),
                    averaged["loss"],
                    averaged["psnr"],
                )
            )

    return average_metrics(running, step_count)


@torch.no_grad()
def evaluate(
    model: SpectralUNet1D,
    dataloader,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, float]:
    model.eval()
    running: Dict[str, float] = {}
    step_count = 0

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            _, _, batch_metrics = run_model_step(model, batch, args, generator=generator)
        accumulate_metrics(running, batch_metrics)
        step_count += 1

    return average_metrics(running, step_count)


def save_checkpoint(
    path: Path,
    model: SpectralUNet1D,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    data_info: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "args": json_ready_args(args),
            "data_info": data_info,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    (logs_dir / "stage1_spectral_unet_config.json").write_text(
        json.dumps(json_ready_args(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataloaders = create_stage1_spectral_dataloaders(
        train_csv=args.train_manifest,
        val_csv=args.val_manifest,
        test_csv=args.test_manifest,
        modality=args.modality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        dataset_kwargs={
            "spectrum_reduction": args.spectrum_reduction,
            "normalization": args.normalization,
            "repo_root": REPO_ROOT,
        },
    )

    train_dataset = dataloaders["train"].dataset
    data_info = {
        "modality": args.modality,
        "axis_values": train_dataset.axis_values.tolist(),
        "spectral_length": train_dataset.spectral_length,
        "spectrum_reduction": args.spectrum_reduction,
        "normalization": args.normalization,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralUNet1D(in_channels=3, base_channels=args.base_channels, dropout=args.dropout).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode="min",
    )

    metrics_path = logs_dir / "stage1_spectral_unet_metrics.csv"
    best_checkpoint_path = checkpoints_dir / "stage1_spectral_unet_best.pt"
    last_checkpoint_path = checkpoints_dir / "stage1_spectral_unet_last.pt"

    history_rows: list[Dict[str, object]] = []
    best_epoch_row: Dict[str, object] | None = None
    best_val_loss = float("inf")
    stopped_early = False

    with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spectral_training_log_fieldnames())
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = train_one_epoch(model, dataloaders["train"], optimizer, scaler, device, epoch, args)
            val_metrics = evaluate(model, dataloaders["val"], device, args, seed=args.seed + epoch)
            scheduler.step()

            lr = float(optimizer.param_groups[0]["lr"])
            elapsed = time.time() - epoch_start
            row = spectral_epoch_row(epoch, args.modality, lr, train_metrics, val_metrics, elapsed)
            history_rows.append(row)
            writer.writerow(row)
            handle.flush()

            print(
                "epoch {0}/{1} modality={2} lr={3:.6f} train_loss={4:.4f} val_loss={5:.4f} val_psnr={6:.2f} elapsed={7:.1f}s".format(
                    epoch,
                    args.epochs,
                    args.modality,
                    lr,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["psnr"],
                    elapsed,
                )
            )

            save_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                args,
                data_info,
            )

            improved, should_stop = early_stopper.step(val_metrics["loss"], epoch)
            if improved:
                best_val_loss = float(val_metrics["loss"])
                best_epoch_row = dict(row)
                save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    args,
                    data_info,
                )

            if should_stop:
                stopped_early = True
                print(
                    "early stopping triggered at epoch {0} (best epoch: {1}, best val_loss: {2:.4f})".format(
                        epoch,
                        early_stopper.best_epoch,
                        early_stopper.best_value if early_stopper.best_value is not None else float("nan"),
                    )
                )
                break

        if best_epoch_row is None:
            best_epoch_row = dict(history_rows[-1])

        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = evaluate(model, dataloaders["test"], device, args, seed=args.seed + 999)

        (logs_dir / "stage1_spectral_unet_test_metrics.json").write_text(
            json.dumps(json_safe_metrics(test_metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        best_row = spectral_best_summary_row(
            best_epoch_row=best_epoch_row,
            test_metrics=test_metrics,
            best_checkpoint_path=best_checkpoint_path,
            stopped_early=stopped_early,
            notes="best row selected by minimum val_loss",
        )
        history_rows.append(best_row)
        writer.writerow(best_row)
        handle.flush()

    (logs_dir / "stage1_spectral_unet_best_summary.json").write_text(
        json.dumps(json_safe_metrics(best_row), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_spectral_training_curves(history_rows, logs_dir, prefix="stage1_spectral_unet")

    print(
        "best_epoch={0} best_val_loss={1:.4f} test_loss={2:.4f} test_psnr={3:.2f}".format(
            best_epoch_row["epoch"],
            float(best_epoch_row["val_loss"]),
            test_metrics["loss"],
            test_metrics["psnr"],
        )
    )
    print("best checkpoint:", best_checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

