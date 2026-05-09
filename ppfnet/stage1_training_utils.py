from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import torch


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    mode: str = "min"
    best_value: float | None = None
    best_epoch: int = 0
    num_bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        return value > (self.best_value + self.min_delta)

    def step(self, value: float, epoch: int) -> tuple[bool, bool]:
        improved = self._is_improvement(value)
        if improved:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        should_stop = self.enabled and self.num_bad_epochs >= self.patience
        return improved, should_stop


def accumulate_metrics(running: Dict[str, float], batch_metrics: Dict[str, float]) -> None:
    for key, value in batch_metrics.items():
        running[key] = running.get(key, 0.0) + float(value)


def average_metrics(running: Dict[str, float], steps: int) -> Dict[str, float]:
    divisor = max(steps, 1)
    return {key: value / divisor for key, value in running.items()}


def _masked_scalar_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    expanded_missing = missing_mask.expand_as(prediction)
    denom = expanded_missing.sum().clamp_min(1.0)

    abs_error = (prediction - target).abs() * expanded_missing
    sq_error = (prediction - target).pow(2) * expanded_missing

    mae = abs_error.sum() / denom
    mse = sq_error.sum() / denom
    rmse = torch.sqrt(mse.clamp_min(1e-12))

    psnr_values: List[torch.Tensor] = []
    batch_size, channels = prediction.shape[:2]
    for batch_idx in range(batch_size):
        spatial_missing = missing_mask[batch_idx, 0] > 0.5
        spatial_valid = valid_mask[batch_idx, 0] > 0.5
        if int(spatial_missing.sum().item()) == 0:
            continue

        for channel_idx in range(channels):
            pred_ch = prediction[batch_idx, channel_idx]
            target_ch = target[batch_idx, channel_idx]
            mse_ch = (pred_ch[spatial_missing] - target_ch[spatial_missing]).pow(2).mean()

            range_source = target_ch[spatial_valid] if int(spatial_valid.sum().item()) > 0 else target_ch[spatial_missing]
            if range_source.numel() == 0:
                continue

            data_range = (range_source.max() - range_source.min()).clamp_min(1e-6)
            if float(mse_ch.item()) <= 1e-12:
                psnr_ch = torch.tensor(99.0, device=prediction.device)
            else:
                psnr_ch = 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse_ch)
            psnr_values.append(psnr_ch)

    if psnr_values:
        psnr = torch.stack(psnr_values).mean()
    else:
        psnr = torch.tensor(99.0, device=prediction.device)

    return {
        "{0}_mae".format(prefix): float(mae.detach().item()),
        "{0}_mse".format(prefix): float(mse.detach().item()),
        "{0}_rmse".format(prefix): float(rmse.detach().item()),
        "{0}_psnr".format(prefix): float(psnr.detach().item()),
    }


def compute_stage1_metrics(
    loss: torch.Tensor,
    recon_fs: torch.Tensor | None,
    recon_ts: torch.Tensor | None,
    target_fs: torch.Tensor | None,
    target_ts: torch.Tensor | None,
    fs_missing_mask: torch.Tensor | None,
    ts_missing_mask: torch.Tensor | None,
    fs_valid_mask: torch.Tensor | None,
    ts_valid_mask: torch.Tensor | None,
) -> Dict[str, float]:
    def empty_metric_block(prefix: str) -> Dict[str, float]:
        return {
            "{0}_mae".format(prefix): float("nan"),
            "{0}_mse".format(prefix): float("nan"),
            "{0}_rmse".format(prefix): float("nan"),
            "{0}_psnr".format(prefix): float("nan"),
        }

    metrics = {"loss": float(loss.detach().item())}

    if recon_fs is not None and target_fs is not None and fs_missing_mask is not None and fs_valid_mask is not None:
        metrics.update(
            _masked_scalar_metrics(
                prediction=recon_fs,
                target=target_fs,
                missing_mask=fs_missing_mask,
                valid_mask=fs_valid_mask,
                prefix="fs",
            )
        )
    else:
        metrics.update(empty_metric_block("fs"))

    if recon_ts is not None and target_ts is not None and ts_missing_mask is not None and ts_valid_mask is not None:
        metrics.update(
            _masked_scalar_metrics(
                prediction=recon_ts,
                target=target_ts,
                missing_mask=ts_missing_mask,
                valid_mask=ts_valid_mask,
                prefix="ts",
            )
        )
    else:
        metrics.update(empty_metric_block("ts"))

    return metrics


def training_log_fieldnames() -> List[str]:
    return [
        "epoch",
        "tag",
        "modality_mode",
        "mask_sharing",
        "lr",
        "train_loss",
        "train_fs_mae",
        "train_ts_mae",
        "train_fs_mse",
        "train_ts_mse",
        "train_fs_rmse",
        "train_ts_rmse",
        "train_fs_psnr",
        "train_ts_psnr",
        "val_loss",
        "val_fs_mae",
        "val_ts_mae",
        "val_fs_mse",
        "val_ts_mse",
        "val_fs_rmse",
        "val_ts_rmse",
        "val_fs_psnr",
        "val_ts_psnr",
        "test_loss",
        "test_fs_mae",
        "test_ts_mae",
        "test_fs_mse",
        "test_ts_mse",
        "test_fs_rmse",
        "test_ts_rmse",
        "test_fs_psnr",
        "test_ts_psnr",
        "elapsed_sec",
        "best_checkpoint_path",
        "stopped_early",
        "notes",
    ]


def epoch_row(
    epoch: int,
    modality_mode: str,
    mask_sharing: str,
    lr: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    elapsed_sec: float,
) -> Dict[str, object]:
    return {
        "epoch": epoch,
        "tag": "epoch",
        "modality_mode": modality_mode,
        "mask_sharing": mask_sharing,
        "lr": lr,
        "train_loss": train_metrics["loss"],
        "train_fs_mae": train_metrics["fs_mae"],
        "train_ts_mae": train_metrics["ts_mae"],
        "train_fs_mse": train_metrics["fs_mse"],
        "train_ts_mse": train_metrics["ts_mse"],
        "train_fs_rmse": train_metrics["fs_rmse"],
        "train_ts_rmse": train_metrics["ts_rmse"],
        "train_fs_psnr": train_metrics["fs_psnr"],
        "train_ts_psnr": train_metrics["ts_psnr"],
        "val_loss": val_metrics["loss"],
        "val_fs_mae": val_metrics["fs_mae"],
        "val_ts_mae": val_metrics["ts_mae"],
        "val_fs_mse": val_metrics["fs_mse"],
        "val_ts_mse": val_metrics["ts_mse"],
        "val_fs_rmse": val_metrics["fs_rmse"],
        "val_ts_rmse": val_metrics["ts_rmse"],
        "val_fs_psnr": val_metrics["fs_psnr"],
        "val_ts_psnr": val_metrics["ts_psnr"],
        "elapsed_sec": elapsed_sec,
        "best_checkpoint_path": "",
        "stopped_early": "",
        "notes": "",
    }


def best_summary_row(
    best_epoch_row: Dict[str, object],
    test_metrics: Dict[str, float],
    best_checkpoint_path: Path,
    stopped_early: bool,
    notes: str,
) -> Dict[str, object]:
    row = {key: "" for key in training_log_fieldnames()}
    row.update(best_epoch_row)
    row["epoch"] = best_epoch_row["epoch"]
    row["tag"] = "best_summary"
    row["test_loss"] = test_metrics["loss"]
    row["test_fs_mae"] = test_metrics["fs_mae"]
    row["test_ts_mae"] = test_metrics["ts_mae"]
    row["test_fs_mse"] = test_metrics["fs_mse"]
    row["test_ts_mse"] = test_metrics["ts_mse"]
    row["test_fs_rmse"] = test_metrics["fs_rmse"]
    row["test_ts_rmse"] = test_metrics["ts_rmse"]
    row["test_fs_psnr"] = test_metrics["fs_psnr"]
    row["test_ts_psnr"] = test_metrics["ts_psnr"]
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


def plot_training_curves(history_rows: Sequence[Dict[str, object]], output_dir: Path, prefix: str = "stage1_unet") -> None:
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
            "train_fs_mae": [float(row["train_fs_mae"]) for row in epoch_rows],
            "train_ts_mae": [float(row["train_ts_mae"]) for row in epoch_rows],
            "val_fs_mae": [float(row["val_fs_mae"]) for row in epoch_rows],
            "val_ts_mae": [float(row["val_ts_mae"]) for row in epoch_rows],
        },
    )
    _plot_lines(
        output_dir / "{0}_rmse_curve.png".format(prefix),
        epochs,
        title="RMSE",
        ylabel="RMSE",
        series={
            "train_fs_rmse": [float(row["train_fs_rmse"]) for row in epoch_rows],
            "train_ts_rmse": [float(row["train_ts_rmse"]) for row in epoch_rows],
            "val_fs_rmse": [float(row["val_fs_rmse"]) for row in epoch_rows],
            "val_ts_rmse": [float(row["val_ts_rmse"]) for row in epoch_rows],
        },
    )
    _plot_lines(
        output_dir / "{0}_psnr_curve.png".format(prefix),
        epochs,
        title="PSNR",
        ylabel="PSNR (dB)",
        series={
            "train_fs_psnr": [float(row["train_fs_psnr"]) for row in epoch_rows],
            "train_ts_psnr": [float(row["train_ts_psnr"]) for row in epoch_rows],
            "val_fs_psnr": [float(row["val_fs_psnr"]) for row in epoch_rows],
            "val_ts_psnr": [float(row["val_ts_psnr"]) for row in epoch_rows],
        },
    )
