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

from ppfnet.stage1_spectral_unet import SpectralUNet1D, build_spectral_masked_inputs, spectral_reconstruction_loss
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
from ppfnet.stage2_rgb_fs_dataset import create_stage2_rgb_fs_dataloader
from ppfnet.stage2_rgb_fs_model import (
    RGBConditionedSpectralStudent,
    freeze_student_backbone,
    freeze_teacher,
    initialize_student_from_teacher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RGB-conditioned FS pixel-wise spectral student model.")
    parser.add_argument("--train-manifest", type=Path, default=Path("outputs/ppfnet_stage2/splits/train_pairs.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("outputs/ppfnet_stage2/splits/val_pairs.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("outputs/ppfnet_stage2/splits/test_pairs.csv"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("outputs/ppfnet_stage1_pixel_fs/checkpoints/stage1_pixel_spectral_unet_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppfnet_stage2_rgb_fs_student"))
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="Optional backbone learning rate. Defaults to lr * 0.25 when teacher initialization is used.",
    )
    parser.add_argument(
        "--rgb-lr",
        type=float,
        default=None,
        help="Optional RGB encoder and fusion learning rate. Defaults to lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rgb-embed-dim", type=int, default=64)
    parser.add_argument("--local-cond-channels", type=int, default=16)
    parser.add_argument("--global-cond-channels", type=int, default=16)
    parser.add_argument(
        "--use-local-rgb-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to sample local RGB features by THz pixel coordinates. Disable for global-only RGB conditioning.",
    )
    parser.add_argument("--image-size", nargs=2, type=int, default=(224, 224), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--max-pixels-per-sample", type=int, default=512)
    parser.add_argument("--mask-mode", choices=["point", "band", "hybrid"], default="hybrid")
    parser.add_argument("--min-observed-ratio", type=float, default=0.25)
    parser.add_argument("--max-observed-ratio", type=float, default=0.7)
    parser.add_argument("--l2-weight", type=float, default=0.1)
    parser.add_argument("--normalization", choices=["none", "zscore", "minmax"], default="none")
    parser.add_argument(
        "--include-structure-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to append seed mask, edge map, and distance transform to RGB input.",
    )
    parser.add_argument(
        "--teacher-weight",
        type=float,
        default=0.0,
        help="Distillation weight. Default 0.0 means teacher is used only for initialization.",
    )
    parser.add_argument("--teacher-l2-weight", type=float, default=0.1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
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


def resolve_learning_rates(args: argparse.Namespace) -> tuple[float, float]:
    backbone_lr = args.backbone_lr
    if backbone_lr is None:
        backbone_lr = args.lr * 0.25 if args.teacher_checkpoint is not None else args.lr

    rgb_lr = args.rgb_lr
    if rgb_lr is None:
        rgb_lr = args.lr

    return float(backbone_lr), float(rgb_lr)


def build_optimizer(
    student: RGBConditionedSpectralStudent,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, float, float]:
    backbone_lr, rgb_lr = resolve_learning_rates(args)
    optimizer = AdamW(
        [
            {"params": student.backbone.parameters(), "lr": backbone_lr},
            {"params": student.rgb_encoder.parameters(), "lr": rgb_lr},
            {"params": student.rgb_to_global_context.parameters(), "lr": rgb_lr},
        ],
        weight_decay=args.weight_decay,
    )
    return optimizer, backbone_lr, rgb_lr


def flatten_stage2_batch(batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
    pixel_mask = batch["pixel_mask"] > 0.5
    batch_size, max_pixels = pixel_mask.shape
    if int(pixel_mask.sum().item()) == 0:
        raise ValueError("No valid pixels found in batch.")

    sample_indices = torch.arange(batch_size, device=pixel_mask.device).unsqueeze(1).expand(batch_size, max_pixels)
    pixel_to_sample_index = sample_indices[pixel_mask]

    spectra = batch["spectra"][pixel_mask]
    coords_xy_norm = batch["coords_xy_norm"][pixel_mask]
    coords_yx = batch["coords_yx"][pixel_mask]
    axis_values = batch["axis_values"][pixel_to_sample_index]

    return {
        "pixel_to_sample_index": pixel_to_sample_index,
        "spectra": spectra,
        "coords_xy_norm": coords_xy_norm,
        "coords_yx": coords_yx,
        "axis_values": axis_values,
    }


def run_model_step(
    student: RGBConditionedSpectralStudent,
    teacher: SpectralUNet1D,
    batch: Dict[str, object],
    args: argparse.Namespace,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    flat = flatten_stage2_batch(batch)
    spectral_batch = {
        "spectrum": flat["spectra"],
        "axis_values": flat["axis_values"],
    }
    masked = build_spectral_masked_inputs(
        spectral_batch,
        mask_mode=args.mask_mode,
        min_observed_ratio=args.min_observed_ratio,
        max_observed_ratio=args.max_observed_ratio,
        use_axis_channel=True,
        generator=generator,
    )

    with torch.no_grad():
        teacher_prediction = teacher(masked["model_input"])
        teacher_reconstruction = masked["masked_spectrum"] + teacher_prediction * (1.0 - masked["observed_mask"])

    student_residual = student(
        masked_model_input=masked["model_input"],
        coords_xy_norm=flat["coords_xy_norm"],
        rgb_images=batch["rgb_image"],
        pixel_to_sample_index=flat["pixel_to_sample_index"],
        baseline_reconstruction=teacher_reconstruction,
    )
    student_reconstruction = teacher_reconstruction + student_residual * masked["missing_mask"]
    student_reconstruction = masked["masked_spectrum"] + student_reconstruction * (1.0 - masked["observed_mask"])

    student_loss_output = spectral_reconstruction_loss(
        prediction=student_reconstruction,
        target=flat["spectra"],
        missing_mask=masked["missing_mask"],
        l2_weight=args.l2_weight,
    )

    residual_penalty = student_residual.abs().mul(masked["missing_mask"]).mean()
    total_loss = student_loss_output.loss + 0.01 * residual_penalty

    teacher_kd_loss = torch.tensor(0.0, device=student_reconstruction.device)
    if args.teacher_weight > 0:
        teacher_loss_output = spectral_reconstruction_loss(
            prediction=student_reconstruction,
            target=teacher_reconstruction.detach(),
            missing_mask=masked["missing_mask"],
            l2_weight=args.teacher_l2_weight,
        )
        teacher_kd_loss = teacher_loss_output.loss

    total_loss = total_loss + args.teacher_weight * teacher_kd_loss

    metrics = compute_spectral_metrics(
        loss=total_loss,
        reconstruction=student_reconstruction,
        target=flat["spectra"],
        missing_mask=masked["missing_mask"],
    )
    metrics["kd_loss"] = float(teacher_kd_loss.detach().item())
    metrics["gt_loss"] = float(student_loss_output.loss.detach().item())
    metrics["residual_penalty"] = float(residual_penalty.detach().item())

    aux = {
        "student_reconstruction": student_reconstruction,
        "teacher_reconstruction": teacher_reconstruction,
        "student_residual": student_residual,
        "missing_mask": masked["missing_mask"],
        "pixel_to_sample_index": flat["pixel_to_sample_index"],
        "coords_yx": flat["coords_yx"],
    }
    return total_loss, aux, metrics


def train_one_epoch(
    student: RGBConditionedSpectralStudent,
    teacher: SpectralUNet1D,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    student.train()
    running: Dict[str, float] = {}
    step_count = 0

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            loss, _, batch_metrics = run_model_step(student, teacher, batch, args)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accumulate_metrics(running, batch_metrics)
        step_count += 1

        if step % args.log_every == 0:
            averaged = average_metrics(running, step_count)
            print(
                "epoch {0} step {1}/{2} loss={3:.4f} gt_loss={4:.4f} kd_loss={5:.4f} psnr={6:.2f}".format(
                    epoch,
                    step,
                    len(dataloader),
                    averaged["loss"],
                    averaged["gt_loss"],
                    averaged["kd_loss"],
                    averaged["psnr"],
                )
            )

    return average_metrics(running, step_count)


@torch.no_grad()
def evaluate(
    student: RGBConditionedSpectralStudent,
    teacher: SpectralUNet1D,
    dataloader,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, float]:
    student.eval()
    running: Dict[str, float] = {}
    step_count = 0

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            _, _, batch_metrics = run_model_step(student, teacher, batch, args, generator=generator)
        accumulate_metrics(running, batch_metrics)
        step_count += 1

    return average_metrics(running, step_count)


def save_checkpoint(
    path: Path,
    student: RGBConditionedSpectralStudent,
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
            "model_state_dict": student.state_dict(),
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

    (logs_dir / "stage2_rgb_fs_student_config.json").write_text(
        json.dumps(json_ready_args(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_loader = create_stage2_rgb_fs_dataloader(
        args.train_manifest,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        dataset_kwargs={
            "repo_root": REPO_ROOT,
            "image_size": tuple(args.image_size),
            "normalization": args.normalization,
            "max_pixels_per_sample": args.max_pixels_per_sample,
            "pixel_selection_seed": args.seed,
            "include_structure_channels": args.include_structure_channels,
            "resample_pixels_each_epoch": True,
        },
    )
    val_loader = create_stage2_rgb_fs_dataloader(
        args.val_manifest,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        dataset_kwargs={
            "repo_root": REPO_ROOT,
            "image_size": tuple(args.image_size),
            "normalization": args.normalization,
            "max_pixels_per_sample": args.max_pixels_per_sample,
            "pixel_selection_seed": args.seed,
            "include_structure_channels": args.include_structure_channels,
            "resample_pixels_each_epoch": False,
        },
    )
    test_loader = create_stage2_rgb_fs_dataloader(
        args.test_manifest,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        dataset_kwargs={
            "repo_root": REPO_ROOT,
            "image_size": tuple(args.image_size),
            "normalization": args.normalization,
            "max_pixels_per_sample": args.max_pixels_per_sample,
            "pixel_selection_seed": args.seed,
            "include_structure_channels": args.include_structure_channels,
            "resample_pixels_each_epoch": False,
        },
    )

    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    teacher_args = teacher_checkpoint["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = SpectralUNet1D(
        in_channels=3,
        base_channels=int(teacher_args.get("base_channels", args.base_channels)),
        dropout=float(teacher_args.get("dropout", 0.0)),
    ).to(device)
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"])
    freeze_teacher(teacher)

    student = RGBConditionedSpectralStudent(
        rgb_in_channels=train_loader.dataset.rgb_channels,
        rgb_embed_dim=args.rgb_embed_dim,
        local_cond_channels=args.local_cond_channels,
        global_cond_channels=args.global_cond_channels,
        base_channels=args.base_channels,
        dropout=args.dropout,
        use_local_rgb_conditioning=args.use_local_rgb_conditioning,
    ).to(device)
    initialize_student_from_teacher(student, teacher.state_dict())

    optimizer, backbone_lr, rgb_lr = build_optimizer(student, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode="min",
    )

    data_info = {
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_usage": "initialization_only" if args.teacher_weight <= 0 else "initialization_plus_distillation",
        "conditioning_type": "concat_plus_film",
        "image_size": list(map(int, args.image_size)),
        "max_pixels_per_sample": args.max_pixels_per_sample,
        "normalization": args.normalization,
        "rgb_in_channels": train_loader.dataset.rgb_channels,
        "include_structure_channels": args.include_structure_channels,
        "use_local_rgb_conditioning": args.use_local_rgb_conditioning,
        "rgb_embed_dim": args.rgb_embed_dim,
        "local_cond_channels": args.local_cond_channels,
        "global_cond_channels": args.global_cond_channels,
        "backbone_lr": backbone_lr,
        "rgb_lr": rgb_lr,
        "train_resample_pixels_each_epoch": True,
    }

    metrics_path = logs_dir / "stage2_rgb_fs_student_metrics.csv"
    best_checkpoint_path = checkpoints_dir / "stage2_rgb_fs_student_best.pt"
    last_checkpoint_path = checkpoints_dir / "stage2_rgb_fs_student_last.pt"

    history_rows: list[Dict[str, object]] = []
    best_epoch_row: Dict[str, object] | None = None
    best_val_loss = float("inf")
    stopped_early = False

    with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=spectral_training_log_fieldnames() + [
                "train_gt_loss",
                "train_kd_loss",
                "train_residual_penalty",
                "val_gt_loss",
                "val_kd_loss",
                "val_residual_penalty",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loader.dataset.set_epoch(epoch)
            freeze_student_backbone(student, freeze=epoch <= args.freeze_backbone_epochs)
            epoch_start = time.time()
            train_metrics = train_one_epoch(student, teacher, train_loader, optimizer, scaler, device, epoch, args)
            val_metrics = evaluate(student, teacher, val_loader, device, args, seed=args.seed + epoch)
            scheduler.step()

            backbone_group_lr = float(optimizer.param_groups[0]["lr"])
            rgb_group_lr = float(optimizer.param_groups[1]["lr"])
            lr = float(max(backbone_group_lr, rgb_group_lr))
            elapsed = time.time() - epoch_start
            row = spectral_epoch_row(epoch, "rgb_fs_student", lr, train_metrics, val_metrics, elapsed)
            row["train_gt_loss"] = train_metrics["gt_loss"]
            row["train_kd_loss"] = train_metrics["kd_loss"]
            row["train_residual_penalty"] = train_metrics["residual_penalty"]
            row["val_gt_loss"] = val_metrics["gt_loss"]
            row["val_kd_loss"] = val_metrics["kd_loss"]
            row["val_residual_penalty"] = val_metrics["residual_penalty"]
            history_rows.append(row)
            writer.writerow(row)
            handle.flush()

            print(
                "epoch {0}/{1} lr_backbone={2:.6f} lr_rgb={3:.6f} train_loss={4:.4f} val_loss={5:.4f} val_psnr={6:.2f} kd={7:.4f} res={8:.4f} elapsed={9:.1f}s".format(
                    epoch,
                    args.epochs,
                    backbone_group_lr,
                    rgb_group_lr,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["psnr"],
                    val_metrics["kd_loss"],
                    val_metrics["residual_penalty"],
                    elapsed,
                )
            )

            save_checkpoint(
                last_checkpoint_path,
                student,
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
                    student,
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
        student.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = evaluate(student, teacher, test_loader, device, args, seed=args.seed + 999)

        (logs_dir / "stage2_rgb_fs_student_test_metrics.json").write_text(
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
        best_row["train_gt_loss"] = best_epoch_row.get("train_gt_loss", "")
        best_row["train_kd_loss"] = best_epoch_row.get("train_kd_loss", "")
        best_row["train_residual_penalty"] = best_epoch_row.get("train_residual_penalty", "")
        best_row["val_gt_loss"] = best_epoch_row.get("val_gt_loss", "")
        best_row["val_kd_loss"] = best_epoch_row.get("val_kd_loss", "")
        best_row["val_residual_penalty"] = best_epoch_row.get("val_residual_penalty", "")
        history_rows.append(best_row)
        writer.writerow(best_row)
        handle.flush()

    (logs_dir / "stage2_rgb_fs_student_best_summary.json").write_text(
        json.dumps(json_safe_metrics(best_row), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_spectral_training_curves(history_rows, logs_dir, prefix="stage2_rgb_fs_student")

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

