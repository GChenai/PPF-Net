#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


HIGHLIGHT_LABEL = "PPF-Net(Random Observed)"
BEST_TRADEOFF_LABEL = "Observed ratio 70%"
ACCENT = "#2a9d8f"
SECONDARY = "#4c78a8"
BEST_OUTLINE = "#e76f51"
NEUTRAL = "#a7b1c2"
GRID = "#d9dee7"
BG = "#f6f8fb"
COMIC_FONT_PATH = Path(r"C:\Windows\Fonts\comic.ttf")
TIMES_FONT_PATH = Path(r"C:\Windows\Fonts\times.ttf")


def load_font_properties(path: Path, fallback_family: str):
    if path.exists():
        return font_manager.FontProperties(fname=str(path))
    return font_manager.FontProperties(family=fallback_family)


COMIC_FONT = load_font_properties(COMIC_FONT_PATH, "Comic Sans MS")
TIMES_FONT = load_font_properties(TIMES_FONT_PATH, "Times New Roman")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-style ablation figures from ablation_summary.csv."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/ablation_summary/ablation_summary.csv"),
        help="Input ablation summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablation_summary/publication_figures"),
        help="Directory where figures will be saved.",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=Path("outputs/ablation_summary/cpu_benchmark/ablation_cpu_benchmark.csv"),
        help="Optional CPU benchmark CSV used to build the final trade-off figure.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: List[Dict[str, object]] = []
    for row in raw_rows:
        rows.append(
            {
                "experiment": row["experiment"],
                "label": row["label"],
                "best_epoch": int(row["best_epoch"]),
                "val_loss": float(row["val_loss"]),
                "val_mae": float(row["val_mae"]),
                "val_rmse": float(row["val_rmse"]),
                "val_psnr": float(row["val_psnr"]),
                "test_loss": float(row["test_loss"]),
                "test_mae": float(row["test_mae"]),
                "test_rmse": float(row["test_rmse"]),
                "test_psnr": float(row["test_psnr"]),
            }
        )
    return rows


def load_benchmark_rows(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: Dict[str, Dict[str, object]] = {}
    for row in raw_rows:
        rows[row["experiment"]] = {
            "params_m": float(row["params_m"]),
            "avg_forward_ms_mean": float(row["avg_forward_ms_mean"]),
            "avg_forward_ms_std": float(row.get("avg_forward_ms_std", 0.0) or 0.0),
        }
    return rows


def label_color(label: str) -> str:
    if label == BEST_TRADEOFF_LABEL:
        return ACCENT
    return SECONDARY


def setup_axis(ax) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    ax.grid(False, axis="y")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_psnr_ranking(rows: List[Dict[str, object]], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda item: item["test_psnr"])
    labels = [row["label"] for row in ordered]
    values = [row["test_psnr"] for row in ordered]
    colors = [label_color(label) for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=220)
    setup_axis(ax)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="none", height=0.62)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Test PSNR (dB)")
    ax.set_title("Ablation Ranking by Test PSNR")

    x_min = min(values) - 0.08
    x_max = max(values) + 0.12
    ax.set_xlim(x_min, x_max)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#1f2937",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_val_test_dumbbell(rows: List[Dict[str, object]], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda item: item["test_psnr"])
    labels = [row["label"] for row in ordered]
    val_psnr = [row["val_psnr"] for row in ordered]
    test_psnr = [row["test_psnr"] for row in ordered]

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=220)
    setup_axis(ax)
    y = np.arange(len(labels))

    for idx, row in enumerate(ordered):
        ax.plot(
            [row["val_psnr"], row["test_psnr"]],
            [idx, idx],
            color=NEUTRAL,
            linewidth=2.4,
            solid_capstyle="round",
            zorder=1,
        )
        c = label_color(row["label"])
        ax.scatter(row["val_psnr"], idx, s=70, color="#ffffff", edgecolor=c, linewidth=1.8, zorder=3)
        ax.scatter(row["test_psnr"], idx, s=70, color=c, edgecolor="#ffffff", linewidth=0.8, zorder=4)

    ax.set_yticks(y, labels)
    ax.set_xlabel("PSNR (dB)")
    ax.set_title("Validation vs Test PSNR")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=SECONDARY, markersize=8, label="Validation"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=SECONDARY, markeredgecolor="white", markersize=8, label="Test"),
        ],
        frameon=False,
        loc="lower right",
    )

    x_values = val_psnr + test_psnr
    ax.set_xlim(min(x_values) - 0.08, max(x_values) + 0.08)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_scatter(rows: List[Dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=220)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in rows:
        color = label_color(row["label"])
        ax.scatter(
            row["test_loss"],
            row["test_psnr"],
            s=130,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            row["test_loss"] + 0.0008,
            row["test_psnr"] + 0.01,
            row["label"],
            fontsize=8.5,
            color="#1f2937",
        )

    ax.set_xlabel("Test Loss")
    ax.set_ylabel("Test PSNR (dB)")
    ax.set_title("Loss-PSNR Trade-off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_final_tradeoff(rows: List[Dict[str, object]], benchmark_map: Dict[str, Dict[str, object]], output_path: Path) -> None:
    merged = []
    for row in rows:
        bench = benchmark_map.get(str(row["experiment"]))
        if bench is None:
            continue
        merged.append(
            {
                **row,
                **bench,
            }
        )

    if not merged:
        return

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8.8, 7.0),
        dpi=240,
        sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.0},
    )
    for ax in (ax_top, ax_bottom):
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, linewidth=0.9, alpha=0.9)
        for spine in ax.spines.values():
            spine.set_visible(False)

    x = np.array([float(row["avg_forward_ms_mean"]) for row in merged], dtype=float)
    y = np.array([float(row["test_psnr"]) for row in merged], dtype=float)
    c = np.array([float(row["test_rmse"]) for row in merged], dtype=float)
    s = np.array([float(row.get("params_m", 1.0)) for row in merged], dtype=float)

    size = 700 * (s / max(float(s.max()), 1e-6))
    scatter = None
    for ax in (ax_top, ax_bottom):
        scatter = ax.scatter(
            x,
            y,
            c=c,
            s=size,
            cmap="viridis_r",
            edgecolor="white",
            linewidth=1.4,
            alpha=0.96,
            zorder=3,
        )

    highlight = next((row for row in merged if row["label"] == HIGHLIGHT_LABEL), None)
    best_tradeoff = next((row for row in merged if row["label"] == BEST_TRADEOFF_LABEL), None)
    if highlight is not None:
        for ax in (ax_top, ax_bottom):
            ax.scatter(
                [highlight["avg_forward_ms_mean"]],
                [highlight["test_psnr"]],
                s=[920],
                facecolor="none",
                edgecolor="#1d3557",
                linewidth=2.2,
                zorder=4,
            )
    if best_tradeoff is not None:
        for ax in (ax_top, ax_bottom):
            ax.scatter(
                [best_tradeoff["avg_forward_ms_mean"]],
                [best_tradeoff["test_psnr"]],
                s=[980],
                facecolor="none",
                edgecolor=BEST_OUTLINE,
                linewidth=2.4,
                linestyle="--",
                zorder=5,
            )

    for row in merged:
        label = str(row["label"])
        label_text = label.replace("Observed ratio ", "Obs. ")
        dx = 0.45
        dy = 0.01
        if label == HIGHLIGHT_LABEL:
            dx = 0.55
            dy = 0.015
        if label == BEST_TRADEOFF_LABEL:
            label_text = f"{label_text} (recommended)"
            dx = 0.55
            dy = 0.03
        if "teacher" in label.lower():
            dx = -3.2
            dy = 0.02
        y_text = float(row["test_psnr"])
        if label == "Single-Modal THz Baseline":
            dx = 0.55
            dy = 0.03
        target_ax = ax_bottom if y_text < 23.0 else ax_top
        target_ax.text(
            float(row["avg_forward_ms_mean"]) + dx,
            y_text + dy,
            label_text,
            fontsize=8.8,
            color="#1f2937",
            fontproperties=COMIC_FONT,
        )

    ax_bottom.set_xlabel("Inference Time on CPU (ms)", fontproperties=TIMES_FONT, fontsize=12)
    ax_top.set_ylabel("Test PSNR (dB)", fontproperties=TIMES_FONT, fontsize=12)
    ax_top.set_title("Ablation Trade-off Between Accuracy and Efficiency", fontproperties=TIMES_FONT, fontsize=14)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_pad = max((x_max - x_min) * 0.15, 1.0)
    ax_top.set_xlim(x_min - x_pad, x_max + x_pad)

    split_threshold = 23.0
    top_values = y[y >= split_threshold]
    low_values = y[y < split_threshold]
    top_min = np.floor((float(top_values.min()) - 0.05) * 100.0) / 100.0
    top_max = np.ceil((float(top_values.max()) + 0.05) * 100.0) / 100.0
    bottom_min = np.floor((float(low_values.min()) - 0.10) * 100.0) / 100.0
    bottom_max = np.ceil((float(low_values.max()) + 0.12) * 100.0) / 100.0

    ax_top.set_ylim(top_min, top_max)
    ax_bottom.set_ylim(bottom_min, bottom_max)
    ax_top.set_yticks(np.arange(top_min, top_max + 1e-6, 0.1))
    ax_bottom.set_yticks(np.arange(bottom_min, bottom_max + 1e-6, 0.2))
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bottom.tick_params(top=False)

    # Only show the axis break marks, without leaving a visible white gap in the plot area.
    d = 0.008
    kwargs_top = dict(transform=ax_top.transAxes, color="#374151", clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kwargs_top)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_top)
    kwargs_bottom = dict(transform=ax_bottom.transAxes, color="#374151", clip_on=False, linewidth=1.2)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs_bottom)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_bottom)

    ax_top.annotate(
        "Better",
        xy=(x_min, y_max),
        xytext=(x_min + 2.0, top_max - 0.05),
        arrowprops=dict(arrowstyle="->", lw=1.3, color="#374151"),
        fontsize=9,
        color="#1f2937",
        fontproperties=COMIC_FONT,
    )
    ax_top.text(
        x_min - x_pad * 0.55,
        top_max - (top_max - top_min) * 0.10,
        "faster",
        rotation=0,
        fontsize=8,
        color="#1f2937",
        fontproperties=COMIC_FONT,
    )
    ax_top.text(
        x_min + x_pad * 0.10,
        top_max - (top_max - top_min) * 0.06,
        "higher PSNR",
        fontsize=8,
        color="#1f2937",
        fontproperties=COMIC_FONT,
    )

    cbar = fig.colorbar(scatter, ax=[ax_top, ax_bottom], pad=0.08, fraction=0.045)
    cbar.set_label("Test RMSE", rotation=90, fontproperties=TIMES_FONT, fontsize=11)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(TIMES_FONT)

    note = "Color: lower RMSE is better"
    ax_bottom.text(
        0.98,
        0.03,
        note,
        transform=ax_bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#1f2937",
        fontproperties=COMIC_FONT,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SECONDARY,
            markeredgecolor="white",
            markersize=8,
            label="Ablation setting",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#1d3557",
            markeredgewidth=2.0,
            markersize=10,
            label="PPF-Net(Random Observed)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=BEST_OUTLINE,
            markeredgewidth=2.0,
            markersize=10,
            label="Recommended trade-off",
        ),
    ]
    legend = ax_top.legend(handles=legend_handles, frameon=False, loc="lower left", fontsize=8)
    for text in legend.get_texts():
        text.set_fontproperties(TIMES_FONT)

    for tick in ax_bottom.get_xticklabels():
        tick.set_fontproperties(TIMES_FONT)
    for tick in ax_top.get_yticklabels():
        tick.set_fontproperties(TIMES_FONT)
    for tick in ax_bottom.get_yticklabels():
        tick.set_fontproperties(TIMES_FONT)

    fig.tight_layout(rect=(0.02, 0.02, 0.90, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    benchmark_map = load_benchmark_rows(args.benchmark_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_psnr_ranking(rows, args.output_dir / "ablation_psnr_ranking.png")
    plot_val_test_dumbbell(rows, args.output_dir / "ablation_val_test_dumbbell.png")
    plot_tradeoff_scatter(rows, args.output_dir / "ablation_loss_psnr_tradeoff.png")
    plot_final_tradeoff(rows, benchmark_map, args.output_dir / "ablation_efficiency_psnr_rmse.png")

    print("saved:", args.output_dir / "ablation_psnr_ranking.png")
    print("saved:", args.output_dir / "ablation_val_test_dumbbell.png")
    print("saved:", args.output_dir / "ablation_loss_psnr_tradeoff.png")
    print("saved:", args.output_dir / "ablation_efficiency_psnr_rmse.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
