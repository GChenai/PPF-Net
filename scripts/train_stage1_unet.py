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

from ppfnet import Stage1UNet, build_masked_inputs, create_stage1_dataloaders, masked_reconstruction_loss
from ppfnet.stage1_unet import modality_uses_fs, modality_uses_ts
from ppfnet.stage1_training_utils import (
    EarlyStopping,
    accumulate_metrics,
    average_metrics,
    best_summary_row,
    compute_stage1_metrics,
    epoch_row,
    plot_training_curves,
    training_log_fieldnames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stage1 THz feature reconstruction U-Net.")
    parser.add_argument("--train-manifest", type=Path, default=Path("outputs/ppfnet_stage1/splits/train_pairs.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("outputs/ppfnet_stage1/splits/val_pairs.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("outputs/ppfnet_stage1/splits/test_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppfnet_stage1_feature_unet"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--modality-mode",
        choices=["joint", "fs_only", "ts_only"],
        default="joint",
        help="Train on reflection only, transmission only, or both jointly.",
    )
    parser.add_argument("--mask-mode", choices=["pixel", "block", "hybrid"], default="hybrid")
    parser.add_argument(
        "--mask-sharing",
        choices=["shared", "independent"],
        default="shared",
        help="When modality-mode is joint, share one mask across FS/TS or sample them independently.",
    )
    parser.add_argument("--min-observed-ratio", type=float, default=0.45)
    parser.add_argument("--max-observed-ratio", type=float, default=0.85)
    parser.add_argument("--l2-weight", type=float, default=0.1)
    parser.add_argument("--normalization", choices=["none", "zscore", "minmax"], default="none")
    parser.add_argument("--spatial-size", nargs=2, type=int, default=None, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--include-valid-mask-channel", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="Stop training if validation loss does not improve for N epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early stopping.",
    )
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
    model: Stage1UNet,
    batch: Dict[str, object],
    args: argparse.Namespace,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor | None], Dict[str, float]]:
    masked = build_masked_inputs(
        batch,
        modality_mode=args.modality_mode,
        mask_mode=args.mask_mode,
        mask_sharing=args.mask_sharing,
        min_observed_ratio=args.min_observed_ratio,
        max_observed_ratio=args.max_observed_ratio,
        generator=generator,
    )

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

    loss_output = masked_reconstruction_loss(
        pred_fs=recon_fs,
        pred_ts=recon_ts,
        target_fs=batch["fs_features"] if modality_uses_fs(args.modality_mode) else None,
        target_ts=batch["ts_features"] if modality_uses_ts(args.modality_mode) else None,
        fs_missing_mask=masked["fs_missing_mask"],
        ts_missing_mask=masked["ts_missing_mask"],
        l2_weight=args.l2_weight,
    )

    metrics = compute_stage1_metrics(
        loss=loss_output.loss,
        recon_fs=recon_fs,
        recon_ts=recon_ts,
        target_fs=batch["fs_features"] if modality_uses_fs(args.modality_mode) else None,
        target_ts=batch["ts_features"] if modality_uses_ts(args.modality_mode) else None,
        fs_missing_mask=masked["fs_missing_mask"],
        ts_missing_mask=masked["ts_missing_mask"],
        fs_valid_mask=batch["fs_valid_mask"] if modality_uses_fs(args.modality_mode) else None,
        ts_valid_mask=batch["ts_valid_mask"] if modality_uses_ts(args.modality_mode) else None,
    )
    return loss_output.loss, masked, metrics


def train_one_epoch(
    model: Stage1UNet,
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
                "epoch {0} step {1}/{2} loss={3:.4f} fs_psnr={4:.2f} ts_psnr={5:.2f}".format(
                    epoch,
                    step,
                    len(dataloader),
                    averaged["loss"],
                    averaged["fs_psnr"],
                    averaged["ts_psnr"],
                )
            )

    return average_metrics(running, step_count)


@torch.no_grad()
def evaluate(
    model: Stage1UNet,
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
    model: Stage1UNet,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    feature_info: Dict[str, object],
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
            "feature_info": feature_info,
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

    (logs_dir / "stage1_unet_config.json").write_text(
        json.dumps(json_ready_args(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataloaders = create_stage1_dataloaders(
        train_csv=args.train_manifest,
        val_csv=args.val_manifest,
        test_csv=args.test_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        dataset_kwargs={
            "normalization": args.normalization,
            "spatial_size": tuple(args.spatial_size) if args.spatial_size else None,
            "include_valid_mask_channel": args.include_valid_mask_channel,
        },
    )

    train_dataset = dataloaders["train"].dataset
    fs_channels = len(train_dataset.fs_feature_names) + (1 if args.include_valid_mask_channel else 0)
    ts_channels = len(train_dataset.ts_feature_names) + (1 if args.include_valid_mask_channel else 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage1UNet(
        fs_channels=fs_channels,
        ts_channels=ts_channels,
        modality_mode=args.modality_mode,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode="min",
    )

    feature_info = {
        "fs_feature_names": list(train_dataset.fs_feature_names),
        "ts_feature_names": list(train_dataset.ts_feature_names),
        "fs_channels": fs_channels,
        "ts_channels": ts_channels,
        "modality_mode": args.modality_mode,
        "mask_sharing": args.mask_sharing,
    }

    metrics_path = logs_dir / "stage1_unet_metrics.csv"
    best_checkpoint_path = checkpoints_dir / "stage1_unet_best.pt"
    last_checkpoint_path = checkpoints_dir / "stage1_unet_last.pt"

    history_rows: list[Dict[str, object]] = []
    best_epoch_row: Dict[str, object] | None = None
    best_val_loss = float("inf")
    stopped_early = False

    with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=training_log_fieldnames())
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_metrics = train_one_epoch(model, dataloaders["train"], optimizer, scaler, device, epoch, args)
            val_metrics = evaluate(model, dataloaders["val"], device, args, seed=args.seed + epoch)
            scheduler.step()

            lr = float(optimizer.param_groups[0]["lr"])
            elapsed = time.time() - epoch_start
            row = epoch_row(epoch, args.modality_mode, args.mask_sharing, lr, train_metrics, val_metrics, elapsed)
            history_rows.append(row)
            writer.writerow(row)
            handle.flush()

            primary_modality = "fs" if args.modality_mode != "ts_only" else "ts"
            print(
                "epoch {0}/{1} mode={2} mask={3} lr={4:.6f} train_loss={5:.4f} val_loss={6:.4f} "
                "train_{7}_psnr={8:.2f} val_{7}_psnr={9:.2f} elapsed={10:.1f}s".format(
                    epoch,
                    args.epochs,
                    args.modality_mode,
                    args.mask_sharing,
                    lr,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    primary_modality,
                    train_metrics["{0}_psnr".format(primary_modality)],
                    val_metrics["{0}_psnr".format(primary_modality)],
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
                feature_info,
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
                    feature_info,
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

        (logs_dir / "stage1_unet_test_metrics.json").write_text(
            json.dumps(json_safe_metrics(test_metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        best_row = best_summary_row(
            best_epoch_row=best_epoch_row,
            test_metrics=test_metrics,
            best_checkpoint_path=best_checkpoint_path,
            stopped_early=stopped_early,
            notes="best row selected by minimum val_loss",
        )
        history_rows.append(best_row)
        writer.writerow(best_row)
        handle.flush()

    (logs_dir / "stage1_unet_best_summary.json").write_text(
        json.dumps(json_safe_metrics(best_row), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_training_curves(history_rows, logs_dir, prefix="stage1_unet")

    final_bits = [
        "best_epoch={0}".format(best_epoch_row["epoch"]),
        "best_val_loss={0:.4f}".format(float(best_epoch_row["val_loss"])),
        "test_loss={0:.4f}".format(test_metrics["loss"]),
    ]
    if modality_uses_fs(args.modality_mode):
        final_bits.append("fs_psnr={0:.2f}".format(test_metrics["fs_psnr"]))
    if modality_uses_ts(args.modality_mode):
        final_bits.append("ts_psnr={0:.2f}".format(test_metrics["ts_psnr"]))

    print(
        " ".join(final_bits)
    )
    print("best checkpoint:", best_checkpoint_path)
    print("metrics log:", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

