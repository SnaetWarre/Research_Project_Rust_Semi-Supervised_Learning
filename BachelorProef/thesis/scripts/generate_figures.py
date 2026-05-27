#!/usr/bin/env python3
"""Generate thesis figures from the local figure specification data."""

from __future__ import annotations

import shutil
from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[3]
THESIS_DIR = REPO_ROOT / "BachelorProef" / "thesis"
FIG_DIR = THESIS_DIR / "figures"

BLUE = "#004C97"
GRAY = "#888888"
LIGHT_GRAY = "#B0B0B0"
GRID = "#E5E5E5"
TEXT = "#333333"
RED = "#C0392B"
GREEN = "#2E8B57"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "svg.fonttype": "none",
    }
)


def mm_to_inches(width: float, height: float) -> tuple[float, float]:
    return width / 25.4, height / 25.4


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "plantvillage_ssl/output/experiments/label_efficiency").mkdir(
        parents=True, exist_ok=True
    )
    (REPO_ROOT / "plantvillage_ssl/output/experiments/class_scaling").mkdir(
        parents=True, exist_ok=True
    )
    (REPO_ROOT / "plantvillage_ssl/output/experiments/new_class_position").mkdir(
        parents=True, exist_ok=True
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def copy_to_thesis_figures(path: Path) -> None:
    thesis_path = FIG_DIR / path.name
    thesis_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, thesis_path)
    print(f"wrote {thesis_path.relative_to(REPO_ROOT)}")


def save_experiment_figure(fig: plt.Figure, path: Path) -> None:
    save(fig, path)
    copy_to_thesis_figures(path)


def strip_axes(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str = "white",
    edge: str = TEXT,
    color: str = TEXT,
    lw: float = 1.2,
    size: float = 8,
    weight: str = "normal",
    wrap: int | None = None,
) -> Rectangle:
    rect = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=lw)
    ax.add_patch(rect)
    label = fill(text, wrap) if wrap else text
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        weight=weight,
    )
    return rect


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = TEXT,
    dashed: bool = False,
    lw: float = 1.3,
    mutation_scale: float = 10,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            linestyle="--" if dashed else "-",
            shrinkA=2,
            shrinkB=2,
        )
    )


def style_data_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)
    ax.tick_params(length=3, width=0.8)
    ax.set_axisbelow(True)


def figure_deployment_comparison() -> None:
    fig, ax = plt.subplots(figsize=mm_to_inches(160, 100))
    strip_axes(ax, (0, 160), (0, 100))

    ax.text(42, 93, "Python / PyTorch stack", ha="center", weight="bold", fontsize=11)
    ax.text(116, 93, "Rust / Burn binary", ha="center", weight="bold", fontsize=11)

    x, y, w = 14, 14, 56
    blocks = [
        ("Python interpreter", 18),
        ("PyTorch + CUDA", 30),
        ("TorchVision, NumPy,\nPillow", 11),
        ("Model weights\n(~5 MB)", 8),
        ("Application code", 8),
    ]
    current = y
    fills = ["#F0F0F0", "#D8D8D8", "#E8E8E8", "#F5F5F5", "#FAFAFA"]
    for (label, h), face in zip(blocks, fills):
        draw_box(ax, x, current, w, h, label, face=face, edge="#666666", size=6.8)
        current += h
    ax.text(x + w + 6, y + 37.5, "~2-5 GB\ntotal", ha="left", va="center", fontsize=9)

    bx, by, bw, bh = 97, 37, 43, 39
    draw_box(
        ax,
        bx,
        by,
        bw,
        bh,
        "",
        face=BLUE,
        edge=BLUE,
        lw=1.4,
    )
    inner = [
        ("Burn runtime", 13),
        ("Model weights\n(~5 MB)", 12),
        ("Application code", 14),
    ]
    cy = by
    for label, h in inner:
        draw_box(
            ax,
            bx + 3,
            cy + 2.2,
            bw - 6,
            h - 3.2,
            label,
            face=BLUE,
            edge="white",
            color="white",
            size=7.0,
            lw=0.8,
        )
        cy += h
    ax.text(bx + bw / 2, by + bh + 6, "Single binary: 26 MB", ha="center", fontsize=10, weight="bold", color=BLUE)

    draw_box(
        ax,
        8,
        1.5,
        70,
        9,
        "Requires: Wi-Fi install / Docker /\npre-installed environment",
        face="white",
        edge=LIGHT_GRAY,
        size=7.2,
    )
    draw_box(
        ax,
        84,
        1.5,
        70,
        9,
        "Fits on: USB stick / Bluetooth /\nbrief mobile data",
        face="white",
        edge=BLUE,
        color=BLUE,
        size=7.2,
    )

    save(fig, FIG_DIR / "deployment_comparison.svg")


def figure_pipeline_flowchart() -> None:
    fig, ax = plt.subplots(figsize=mm_to_inches(160, 115))
    strip_axes(ax, (0, 160), (0, 115))

    def node(x: float, y: float, w: float, h: float, text: str, face: str = "white") -> None:
        draw_box(ax, x, y, w, h, text, face=face, edge=BLUE if face != "white" else LIGHT_GRAY, color="white" if face != "white" else TEXT, size=8, wrap=18)

    def diamond(cx: float, cy: float, w: float, h: float, text: str) -> None:
        pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
        ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=BLUE, linewidth=1.3))
        ax.text(cx, cy, fill(text, 14), ha="center", va="center", fontsize=8)

    center_x = 80
    node(56, 103, 48, 10, "Start: labeled data (20%)", BLUE)
    node(56, 88, 48, 10, "Train initial CNN")
    node(52, 73, 56, 10, "Inference on unlabeled stream")
    diamond(center_x, 58, 50, 18, "Confidence >= 0.9?")
    node(56, 38, 48, 10, "Accept as pseudo-label")
    node(118, 53, 34, 10, "Reject / discard")
    diamond(center_x, 25, 46, 16, "Buffer >= 200?")
    node(6, 20, 46, 11, "Retrain CNN on labeled + pseudo")
    diamond(38, 8, 42, 14, "Validation plateau?")
    node(108, 3, 36, 10, "Final model", BLUE)

    arrow(ax, (80, 103), (80, 98))
    arrow(ax, (80, 88), (80, 83))
    arrow(ax, (80, 73), (80, 67))
    arrow(ax, (80, 49), (80, 48))
    ax.text(89, 50, "Yes", fontsize=7, color=BLUE)
    arrow(ax, (105, 58), (118, 58))
    ax.text(109, 61, "No", fontsize=7, color=GRAY)
    arrow(ax, (80, 38), (80, 33))
    arrow(ax, (57, 25), (52, 25))
    ax.text(54, 29, "Yes", fontsize=7, color=BLUE)
    arrow(ax, (29, 20), (33, 15))
    arrow(ax, (59, 8), (108, 8), color=BLUE)
    ax.text(72, 10, "Yes", fontsize=7, color=BLUE)
    ax.plot([17, 3, 3], [8, 8, 78], color=GRAY, linewidth=1.3, linestyle="--")
    arrow(ax, (3, 78), (52, 78), color=GRAY, dashed=True)
    ax.text(9, 80, "No", fontsize=7, color=GRAY)
    arrow(ax, (103, 25), (121, 71), color=GRAY, dashed=True)
    arrow(ax, (121, 71), (108, 78), color=GRAY, dashed=True)
    ax.text(105, 28, "No", fontsize=7, color=GRAY)

    save(fig, FIG_DIR / "pipeline_flowchart.svg")


def figure_framework_deployment() -> None:
    fig, ax = plt.subplots(figsize=mm_to_inches(160, 110))
    strip_axes(ax, (0, 160), (0, 110))

    rows = [88, 55, 22]
    labels = ["Burn", "Candle", "tch-rs"]
    row_colors = [BLUE, GRAY, "#666666"]
    for y, label, color in zip(rows, labels, row_colors):
        ax.text(8, y + 6, label, ha="left", va="center", weight="bold", fontsize=11, color=color)
        draw_box(ax, 28, y, 33, 12, "Rust source\ncode + model", face="white", edge=color, size=7.5)

    arrow(ax, (62, 94), (91, 94), color=BLUE)
    ax.text(76.5, 99, "cargo build --release", ha="center", fontsize=7, color=BLUE)
    draw_box(ax, 93, 88, 48, 12, "Static binary\n(CPU/CUDA/WGPU)", face=BLUE, edge=BLUE, color="white", size=7.5)
    ax.text(117, 83, "~26 MB, no runtime deps", ha="center", fontsize=7.5, color=BLUE)

    arrow(ax, (62, 61), (85, 61), color=GRAY)
    ax.text(73.5, 66, "cargo build", ha="center", fontsize=7, color=GRAY)
    draw_box(ax, 87, 57, 28, 10, "Static\nbinary", face="white", edge=GRAY, size=7.5)
    draw_box(ax, 121, 57, 28, 10, "WASM\nmodule", face="white", edge=GRAY, size=7.5)
    ax.text(118, 51, "Lightweight, inference-focused", ha="center", fontsize=7.5, color=GRAY)

    arrow(ax, (62, 28), (78, 28), color="#666666")
    ax.text(70, 33, "cargo build", ha="center", fontsize=7, color="#666666")
    draw_box(ax, 80, 23, 26, 10, "Rust\nbinary", face="white", edge="#666666", size=7.5)
    arrow(ax, (107, 28), (122, 28), color="#666666", dashed=True)
    ax.text(114, 34, "requires\nat runtime", ha="center", fontsize=7, color="#666666")
    draw_box(
        ax,
        124,
        21,
        31,
        14,
        "LibTorch shared\nlibrary (~1.5 GB)",
        face="#F0F0F0",
        edge="#666666",
        size=7.2,
    )
    ax.text(119, 15, "Full PyTorch compatibility, large runtime dependency", ha="center", fontsize=7.5, color="#666666")

    save(fig, FIG_DIR / "framework_deployment.svg")


def figure_forgetting_strategies() -> None:
    fig, ax = plt.subplots(figsize=mm_to_inches(160, 90))
    strip_axes(ax, (0, 160), (0, 90))

    col_x = [6, 58, 110]
    titles = ["Regularization", "Rehearsal", "Architecture"]
    for x, title in zip(col_x, titles):
        ax.text(x + 22, 81, title, ha="center", weight="bold", fontsize=11, color=BLUE if title == "Rehearsal" else TEXT)
        draw_box(ax, x, 18, 44, 56, "", face="white", edge=GRID, lw=1.0)

    # Regularization: weight matrix with locked cells.
    x0, y0 = col_x[0] + 12, 56
    for i in range(4):
        for j in range(4):
            face = "#F7F7F7"
            edge = LIGHT_GRAY
            if (i, j) in {(1, 1), (2, 0), (2, 3)}:
                face = "#FCE8E6"
                edge = RED
            ax.add_patch(Rectangle((x0 + j * 5, y0 - i * 5), 4, 4, facecolor=face, edgecolor=edge, linewidth=0.8))
    ax.text(col_x[0] + 22, 36, "Protect important weights", ha="center", va="center", fontsize=6.2, color=RED)
    ax.text(col_x[0] + 22, 27, "EWC: protect weights\nLwF: distil old outputs", ha="center", va="center", fontsize=6.8)

    # Rehearsal: buffer feeding training loop.
    bx, by = col_x[1] + 8, 52
    for offset in [0, 3, 6]:
        draw_box(ax, bx + offset, by - offset, 14, 10, "", face="#F7F7F7", edge=LIGHT_GRAY)
    draw_box(ax, col_x[1] + 24, 49, 14, 10, "Train", face=BLUE, edge=BLUE, color="white", size=7)
    arrow(ax, (col_x[1] + 24, 54), (col_x[1] + 22, 54), color=BLUE)
    arrow(ax, (col_x[1] + 31, 49), (col_x[1] + 15, 45), color=BLUE, dashed=True)
    arrow(ax, (col_x[1] + 15, 45), (col_x[1] + 13, 52), color=BLUE, dashed=True)
    ax.text(col_x[1] + 22, 38, "Memory buffer\nreplay", ha="center", va="center", fontsize=6.8, color=BLUE)
    ax.text(col_x[1] + 22, 28, "Replay old examples\nGEM: constrain updates", ha="center", va="center", fontsize=6.8)

    # Architecture: add columns and lateral links.
    xbase = col_x[2] + 9
    for c, color in enumerate([LIGHT_GRAY, BLUE, GREEN]):
        xs = xbase + c * 9
        for yy in [63, 55, 47]:
            ax.add_patch(plt.Circle((xs, yy), 2.2, facecolor="white", edgecolor=color, linewidth=1.1))
        if c > 0:
            for yy in [63, 55, 47]:
                ax.plot([xs - 9 + 2.2, xs - 2.2], [yy, yy], color=color, linewidth=0.8)
    ax.text(col_x[2] + 22, 37, "Add/freeze capacity", ha="center", va="center", fontsize=6.2, color=GREEN)
    ax.text(col_x[2] + 22, 27, "Prog. Nets: add columns\nPackNet: prune/freeze", ha="center", va="center", fontsize=6.5)

    ax.text(
        80,
        8,
        "This project uses plain fine-tuning to measure forgetting directly.",
        ha="center",
        va="center",
        fontsize=8,
        color=TEXT,
    )

    save(fig, FIG_DIR / "forgetting_strategies.svg")


def figure_label_efficiency_curve() -> None:
    x = np.array([5, 10, 25, 50, 100, 200, 500])
    y = np.array([34.21, 36.84, 57.89, 72.37, 85.53, 88.75, 94.47])

    fig, ax = plt.subplots(figsize=mm_to_inches(120, 80))
    style_data_ax(ax)
    ax.plot(x, y, color=BLUE, marker="o", markersize=5.5, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlim(4, 600)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("Labeled images per class")
    ax.set_ylabel("Accuracy (%)")
    ax.axvline(100, color=GRAY, linestyle="--", linewidth=1)
    ax.annotate(
        "min viable: 100",
        xy=(100, 85.53),
        xytext=(132, 76),
        arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 1},
        fontsize=8,
        color=BLUE,
    )
    save_experiment_figure(fig, REPO_ROOT / "plantvillage_ssl/output/experiments/label_efficiency/label_efficiency_curve.svg")


def figure_label_efficiency_bars() -> None:
    labels = ["5", "10", "25", "50", "100", "200", "500"]
    y = np.array([34.21, 36.84, 57.89, 72.37, 85.53, 88.75, 94.47])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=mm_to_inches(120, 80))
    style_data_ax(ax)
    bars = ax.bar(x, y, width=0.6, color=BLUE)
    bars[4].set_edgecolor(TEXT)
    bars[4].set_linewidth(1.2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Labeled images per class")
    ax.set_ylabel("Accuracy (%)")
    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=45,
        )
    save_experiment_figure(fig, REPO_ROOT / "plantvillage_ssl/output/experiments/label_efficiency/label_efficiency_bars.svg")


def figure_class_scaling() -> None:
    metrics = ["Base acc.\n(before)", "Base acc.\n(after)", "New class\nacc.", "Overall\nacc."]
    small_acc = np.array([99.83, 99.62, 100.00, 99.68])
    large_acc = np.array([98.76, 97.50, 96.98, 97.49])
    small_forget = 0.21
    large_forget = 1.26

    fig = plt.figure(figsize=mm_to_inches(140, 90))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.4], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in (ax1, ax2):
        style_data_ax(ax)

    x = np.arange(len(metrics))
    width = 0.35
    b1 = ax1.bar(x - width / 2, small_acc, width, color=BLUE, label="5->6 classes")
    b2 = ax1.bar(x + width / 2, large_acc, width, color=LIGHT_GRAY, label="30->31 classes")
    ax1.set_ylim(95, 101.0)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x, metrics)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=8, frameon=False, ncol=2)
    for bars in (b1, b2):
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=7)

    fx = np.arange(1)
    f1 = ax2.bar(fx - width / 2, [small_forget], width, color=BLUE)
    f2 = ax2.bar(fx + width / 2, [large_forget], width, color=LIGHT_GRAY)
    ax2.set_ylim(0, 1.55)
    ax2.set_ylabel("Forgetting (pp)")
    ax2.set_xticks(fx, ["Forgetting"])
    for bars in (f1, f2):
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{bar.get_height():.2f} pp", ha="center", va="bottom", fontsize=7)
    ax2.annotate(
        "6x more forgetting",
        xy=(width / 2, large_forget),
        xytext=(-0.06, 1.35),
        ha="center",
        fontsize=8,
        color=TEXT,
        arrowprops={"arrowstyle": "->", "linewidth": 1, "color": TEXT},
    )
    fig.text(0.5, 0.015, "Training time: 1,573 s (small) vs 8,359 s (large)", ha="center", fontsize=8)
    save_experiment_figure(fig, REPO_ROOT / "plantvillage_ssl/output/experiments/class_scaling/class_scaling_comparison.svg")


def figure_new_class_accuracy() -> None:
    x = np.array([5, 10, 25, 50, 100])
    small = np.array([3.62, 5.11, 60.03, 84.27, 95.16])
    large = np.array([0.00, 0.17, 19.66, 25.62, 55.10])

    fig, ax = plt.subplots(figsize=mm_to_inches(120, 80))
    style_data_ax(ax)
    ax.plot(x, small, color=BLUE, marker="o", linewidth=2, markersize=5.5, label="6th class (small base)")
    ax.plot(x, large, color=GRAY, marker="s", linestyle="--", linewidth=2, markersize=5, label="31st class (large base)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xlabel("Labeled samples for new class")
    ax.set_ylabel("New class accuracy (%)")
    ax.axhline(70, color=TEXT, linestyle="--", linewidth=1)
    ax.text(3, 72, "70% threshold", fontsize=8, color=TEXT)
    ax.annotate("crosses at 50 samples", xy=(50, 84.27), xytext=(54, 74), fontsize=8, color=BLUE, arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 1})
    ax.annotate("does not reach 70%", xy=(100, 55.10), xytext=(60, 47), fontsize=8, color=GRAY, arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 1})
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save_experiment_figure(fig, REPO_ROOT / "plantvillage_ssl/output/experiments/new_class_position/new_class_accuracy_curve.svg")


def figure_position_comparison() -> None:
    metrics = ["New class\naccuracy", "Forgetting", "Overall\naccuracy"]
    small = np.array([84.27, -2.84, 92.52])
    large = np.array([25.62, 0.62, 88.29])
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=mm_to_inches(120, 80))
    style_data_ax(ax)
    b1 = ax.bar(x - width / 2, small, width, color=BLUE, label="5->6 classes")
    b2 = ax.bar(x + width / 2, large, width, color=LIGHT_GRAY, label="30->31 classes")
    ax.axhline(0, color=TEXT, linewidth=1)
    ax.set_ylim(-5, 95)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Percentage / percentage points")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=8, frameon=False, ncol=2)
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            offset = 1.2 if value >= 0 else -1.2
            va = "bottom" if value >= 0 else "top"
            suffix = " pp" if abs(bar.get_x() + bar.get_width() / 2 - 1) < 0.3 else "%"
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.2f}{suffix}", ha="center", va=va, fontsize=7)
    ax.annotate("", xy=(0.42, 84.27), xytext=(0.42, 25.62), arrowprops={"arrowstyle": "<->", "color": TEXT, "linewidth": 1})
    ax.text(0.52, 58, "-58.7 pp gap", ha="left", fontsize=8, color=TEXT)
    save_experiment_figure(fig, REPO_ROOT / "plantvillage_ssl/output/experiments/new_class_position/position_comparison_50.svg")


def figure_forgetting_curve() -> None:
    x = np.array([5, 10, 25, 50, 100])
    small = np.array([0.42, 1.42, -0.25, -2.84, -2.50])
    large = np.array([-0.70, 0.37, 0.15, 0.62, 0.55])

    fig, ax = plt.subplots(figsize=mm_to_inches(120, 80))
    style_data_ax(ax)
    ax.plot(x, small, color=BLUE, marker="o", linewidth=2, markersize=5.5, label="5->6 classes")
    ax.plot(x, large, color=GRAY, marker="s", linestyle="--", linewidth=2, markersize=5, label="30->31 classes")
    ax.axhline(0, color=TEXT, linewidth=1.1)
    ax.text(36, -0.28, "no forgetting", fontsize=8, color=TEXT)
    ax.set_xlim(0, 105)
    ax.set_ylim(-3.5, 2.0)
    ax.set_xticks(x)
    ax.set_xlabel("Labeled samples for new class")
    ax.set_ylabel("Forgetting (percentage points)")
    ax.annotate(
        "implicit regularisation",
        xy=(50, -2.84),
        xytext=(38, -3.28),
        fontsize=8,
        color=BLUE,
        arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 1},
    )
    ax.annotate(
        "measurable forgetting",
        xy=(100, 0.55),
        xytext=(62, 1.12),
        fontsize=8,
        color=GRAY,
        arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 1},
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8, frameon=False)
    save_experiment_figure(fig, REPO_ROOT / "plantvillage_ssl/output/experiments/new_class_position/forgetting_curve.svg")


def main() -> None:
    ensure_dirs()
    figure_deployment_comparison()
    figure_pipeline_flowchart()
    figure_framework_deployment()
    figure_forgetting_strategies()
    figure_label_efficiency_curve()
    figure_label_efficiency_bars()
    figure_class_scaling()
    figure_new_class_accuracy()
    figure_position_comparison()
    figure_forgetting_curve()


if __name__ == "__main__":
    main()
