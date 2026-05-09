from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import torch

from .stage1_training_utils import EarlyStopping, accumulate_metrics, average_metrics


def compute_spectral_metrics(
    loss: torch.Tensor,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
) -> Dict[str, float]:
    denom = missing_mask.sum().clamp_min(1.0)
    abs_error = (reconstruction - target).abs() * missing_mask
    sq_error = (reconstruction - target).pow(2) * missing_mask

    mae = abs_error.sum() / denom
    mse = sq_error.sum() / denom
    rmse = torch.sqrt(mse.clamp_min(1e-12))

    psnr_values: List[torch.Tensor] = []
    batch_size = reconstruction.shape[0]
    for batch_idx in range(batch_size):
        mask = missing_mask[batch_idx, 0] > 0.5
        if int(mask.sum().item()) == 0:
            continue
        pred = reconstruction[batch_idx, 0]
        gt = target[batch_idx, 0]
        mse_sample = (pred[mask] - gt[mask]).pow(2).mean()
        data_range = (gt.max() - gt.min()).clamp_min(1e-6)
        if float(mse_sample.item()) <= 1e-12:
            psnr_sample = torch.tensor(99.0, device=reconstruction.device)
        else:
            psnr_sample = 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse_sample)
        psnr_values.append(psnr_sample)

    psnr = torch.stack(psnr_values).mean() if psnr_values else torch.tensor(99.0, device=reconstruction.device)

    return {
        "loss": float(loss.detach().item()),
        "mae": float(mae.detach().item()),
        "mse": float(mse.detach().item()),
        "rmse": float(rmse.detach().item()),
        "psnr": float(psnr.detach().item()),
    }


def spectral_training_log_fieldnames() -> List[str]:
    return [
        "epoch",
        "tag",
        "modality",
        "lr",
        "train_loss",
        "train_mae",
        "train_mse",
        "train_rmse",
        "train_psnr",
        "val_loss",
        "val_mae",
        "val_mse",
        "val_rmse",
        "val_psnr",
        "test_loss",
        "test_mae",
        "test_mse",
        "test_rmse",
        "test_psnr",
        "elapsed_sec",
        "best_checkpoint_path",
        "stopped_early",
        "notes",
    ]


def spectral_epoch_row(
    epoch: int,
    modality: str,
    lr: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    elapsed_sec: float,
) -> Dict[str, object]:
    return {
        "epoch": epoch,
        "tag": "epoch",
        "modality": modality,
        "lr": lr,
        "train_loss": train_metrics["loss"],
        "train_mae": train_metrics["mae"],
        "train_mse": train_metrics["mse"],
        "train_rmse": train_metrics["rmse"],
        "train_psnr": train_metrics["psnr"],
        "val_loss": val_metrics["loss"],
        "val_mae": val_metrics["mae"],
        "val_mse": val_metrics["mse"],
        "val_rmse": val_metrics["rmse"],
        "val_psnr": val_metrics["psnr"],
        "elapsed_sec": elapsed_sec,
        "best_checkpoint_path": "",
        "stopped_early": "",
        "notes": "",
    }


def spectral_best_summary_row(
    best_epoch_row: Dict[str, object],
    test_metrics: Dict[str, float],
    best_checkpoint_path: Path,
    stopped_early: bool,
    notes: str,
) -> Dict[str, object]:
    row = {key: "" for key in spectral_training_log_fieldnames()}
    row.update(best_epoch_row)
    row["tag"] = "best_summary"
    row["test_loss"] = test_metrics["loss"]
    row["test_mae"] = test_metrics["mae"]
    row["test_mse"] = test_metrics["mse"]
    row["test_rmse"] = test_metrics["rmse"]
    row["test_psnr"] = test_metrics["psnr"]
    row["best_checkpoint_path"] = str(best_checkpoint_path)
    row["stopped_early"] = str(bool(stopped_early))
    row["notes"] = notes
    return row


def _plot_lines(
    output_path: Path,
    epochs: Sequence[int],
    title: str,
    ylabel: str,
    series: Dict[str, Sequence[float]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    plotted = 0
    for label, values in series.items():
        finite_values = [value for value in values if value is not None and math.isfinite(float(value))]
        if not finite_values:
            continue
        ax.plot(epochs, values, label=label, linewidth=2)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_spectral_training_curves(
    history_rows: Sequence[Dict[str, object]],
    output_dir: Path,
    prefix: str = "stage1_spectral_unet",
) -> None:
    epoch_rows = [row for row in history_rows if row.get("tag") == "epoch"]
    if not epoch_rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in epoch_rows]
    _plot_lines(
        output_dir / "{0}_loss_curve.png".format(prefix),
        epochs,
        title="Loss",
        ylabel="Loss",
        series={
            "train_loss": [float(row["train_loss"]) for row in epoch_rows],
            "val_loss": [float(row["val_loss"]) for row in epoch_rows],
        },
    )
    _plot_lines(
        output_dir / "{0}_mae_curve.png".format(prefix),
        epochs,
        title="MAE",
        ylabel="MAE",
        series={
            "train_mae": [float(row["train_mae"]) for row in epoch_rows],
            "val_mae": [float(row["val_mae"]) for row in epoch_rows],
        },
    )
    _plot_lines(
        output_dir / "{0}_rmse_curve.png".format(prefix),
        epochs,
        title="RMSE",
        ylabel="RMSE",
        series={
            "train_rmse": [float(row["train_rmse"]) for row in epoch_rows],
            "val_rmse": [float(row["val_rmse"]) for row in epoch_rows],
        },
    )
    _plot_lines(
        output_dir / "{0}_psnr_curve.png".format(prefix),
        epochs,
        title="PSNR",
        ylabel="PSNR (dB)",
        series={
            "train_psnr": [float(row["train_psnr"]) for row in epoch_rows],
            "val_psnr": [float(row["val_psnr"]) for row in epoch_rows],
        },
    )

