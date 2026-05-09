#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


COMIC_FONT_PATH = Path(r"C:\Windows\Fonts\comic.ttf")
TIMES_FONT_PATH = Path(r"C:\Windows\Fonts\times.ttf")
BG = "#f6f8fb"
GRID = "#d9dee7"
TEXT = "#1f2937"
PPF_OUTLINE = "#1d3557"
PPF_FILL = "#c1121f"


def load_font_properties(path: Path, fallback_family: str):
    if path.exists():
        return font_manager.FontProperties(fname=str(path))
    return font_manager.FontProperties(family=fallback_family)


COMIC_FONT = load_font_properties(COMIC_FONT_PATH, "Comic Sans MS")
TIMES_FONT = load_font_properties(TIMES_FONT_PATH, "Times New Roman")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-style comparison figure."
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=Path("outputs/comparison_benchmark_cpu_repeated.csv"),
        help="CPU repeated benchmark CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison_summary/publication_figures"),
        help="Directory where the figure will be saved.",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: Dict[str, Dict[str, float]] = {}
    for row in rows:
        result[row["label"]] = {
            "params_m": float(row["params_m"]),
            "time_mean": float(row["avg_forward_ms_mean"]),
            "time_std": float(row["avg_forward_ms_std"]),
        }
    return result


def comparison_rows(benchmark_map: Dict[str, Dict[str, float]]) -> List[Dict[str, object]]:
    rows = [
        {"label": "SRCNN", "rmse": 1.0322, "psnr": 15.1871},
        {"label": "DnCNN", "rmse": 0.9893, "psnr": 15.5602},
        {"label": "EDSR", "rmse": 0.5533, "psnr": 19.9324},
        {"label": "TCN", "rmse": 0.3735, "psnr": 19.0498},
        {"label": "Single-Modal THz Baseline", "rmse": 0.3291, "psnr": 20.1946},
        {"label": "PPF-Net (Obs. 70%)", "rmse": 0.2977, "psnr": 22.1069},
    ]
    for row in rows:
        row.update(benchmark_map[row["label"]])
    return rows


def plot_comparison_tradeoff(rows: List[Dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.6), dpi=240)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.9, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    x = np.array([float(row["time_mean"]) for row in rows], dtype=float)
    y = np.array([float(row["psnr"]) for row in rows], dtype=float)
    c = np.array([float(row["rmse"]) for row in rows], dtype=float)
    s = np.array([float(row["params_m"]) for row in rows], dtype=float)
    size = 850 * (s / max(float(s.max()), 1e-6)) + 90

    scatter = ax.scatter(
        x,
        y,
        c=c,
        s=size,
        cmap="viridis_r",
        edgecolor="white",
        linewidth=1.4,
        alpha=0.95,
        zorder=3,
    )

    ppf_row = next((row for row in rows if row["label"] == "PPF-Net (Obs. 70%)"), None)
    if ppf_row is not None:
        ax.scatter(
            [ppf_row["time_mean"]],
            [ppf_row["psnr"]],
            s=[1020],
            facecolor="none",
            edgecolor=PPF_OUTLINE,
            linewidth=2.6,
            zorder=5,
        )
        ax.scatter(
            [ppf_row["time_mean"]],
            [ppf_row["psnr"]],
            s=[240],
            color=PPF_FILL,
            edgecolor="white",
            linewidth=0.9,
            zorder=6,
        )

    for row in rows:
        label = str(row["label"])
        label_text = label
        dx = 0.45
        dy = 0.12
        if label == "Single-Modal THz Baseline":
            dx = 0.55
            dy = 0.22
            label_text = "U-Net"
        elif label == "PPF-Net (Obs. 70%)":
            dx = 0.55
            dy = 0.25
            label_text = "PPF-Net"
        elif label == "TCN":
            dx = 0.45
            dy = -0.42
        elif label == "EDSR":
            dx = 0.40
            dy = 0.12
        elif label == "DnCNN":
            dx = 0.30
            dy = 0.15
        elif label == "SRCNN":
            dx = 0.30
            dy = -0.45

        ax.text(
            float(row["time_mean"]) + dx,
            float(row["psnr"]) + dy,
            label_text,
            fontsize=9.0,
            color=TEXT,
            fontproperties=COMIC_FONT,
        )

    ax.set_xlabel("Inference Time on CPU (ms)", fontproperties=TIMES_FONT, fontsize=12)
    ax.set_ylabel("Test PSNR (dB)", fontproperties=TIMES_FONT, fontsize=12)
    ax.set_title("Comparison of Accuracy-Efficiency Trade-off", fontproperties=TIMES_FONT, fontsize=14)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_pad = max((x_max - x_min) * 0.22, 1.0)
    y_pad = max((y_max - y_min) * 0.18, 0.2)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.annotate(
        "Better",
        xy=(x_min, y_max),
        xytext=(x_min + 2.0, y_max - 1.0),
        arrowprops=dict(arrowstyle="->", lw=1.3, color=TEXT),
        fontsize=9,
        color=TEXT,
        fontproperties=COMIC_FONT,
    )
    ax.text(
        x_min - x_pad * 0.40,
        y_max - y_pad * 0.15,
        "faster",
        fontsize=8,
        color=TEXT,
        fontproperties=COMIC_FONT,
    )
    ax.text(
        x_min + x_pad * 0.12,
        y_max + y_pad * 0.22,
        "higher PSNR",
        fontsize=8,
        color=TEXT,
        fontproperties=COMIC_FONT,
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.04)
    cbar.set_label("Test RMSE", rotation=90, fontproperties=TIMES_FONT, fontsize=11)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(TIMES_FONT)

    note = "Color: lower RMSE is better"
    ax.text(
        0.98,
        0.03,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=TEXT,
        fontproperties=COMIC_FONT,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d1495b",
            markeredgecolor="white",
            markersize=8,
            label="Compared method",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=PPF_OUTLINE,
            markeredgewidth=2.0,
            markersize=10,
            label="PPF-Net",
        ),
    ]
    legend = ax.legend(handles=legend_handles, frameon=False, loc="lower left", fontsize=8)
    for text in legend.get_texts():
        text.set_fontproperties(TIMES_FONT)

    for tick in ax.get_xticklabels():
        tick.set_fontproperties(TIMES_FONT)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(TIMES_FONT)

    fig.tight_layout(rect=(0.02, 0.02, 0.94, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    benchmark_map = load_benchmark(args.benchmark_csv)
    rows = comparison_rows(benchmark_map)
    output_path = args.output_dir / "comparison_efficiency_psnr_rmse.png"
    plot_comparison_tradeoff(rows, output_path)
    print("saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
