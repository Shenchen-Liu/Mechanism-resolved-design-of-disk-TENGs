#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = SCRIPT_DIR / ".mplconfig_fig1_icons"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


CANVAS_SIZE = (9.0, 6.0)
PNG_DPI = 100
SAFE_PAD = 0.05

ICON_MECHANISM = "fig1_icon_mechanism_landscape"
ICON_WINDOW = "fig1_icon_design_window"
ICON_DECISION = "fig1_icon_decision_map"
ICON_SAFE = "fig1_icon_safe_region"
ALL_ICONS = (ICON_MECHANISM, ICON_WINDOW, ICON_DECISION, ICON_SAFE)

CAP_BLUE = "#2C7FB8"
MIX_GREY = "#B3B3B3"
CHARGE_RED = "#D7301F"
SAFE_GREEN = "#BFE2D8"
SAFE_GREEN_EDGE = "#5A7C72"
ASHBY_GREY = "#DCE3E8"
CRITERION_GREY = "#8E98A2"
MECHANISM_CMAP = LinearSegmentedColormap.from_list(
    "fig1_mechanism",
    [(0.0, CAP_BLUE), (0.48, MIX_GREY), (1.0, CHARGE_RED)],
)
FIG3_VIRIDIS = LinearSegmentedColormap.from_list(
    "fig1_fig3_viridis",
    ["#440154", "#31688E", "#35B779", "#FDE725"],
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": PNG_DPI,
            "savefig.dpi": PNG_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.85,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate abstract mini-icons for the Fig.1 right-side output cards."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory for SVG/PNG exports. Defaults to the current Fig.1 support directory.",
    )
    parser.add_argument(
        "--icons",
        nargs="+",
        choices=ALL_ICONS,
        default=list(ALL_ICONS),
        help="Subset of icons to export.",
    )
    return parser.parse_args()


def make_icon_canvas():
    fig = plt.figure(figsize=CANVAS_SIZE, dpi=PNG_DPI)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([SAFE_PAD, SAFE_PAD, 1.0 - 2.0 * SAFE_PAD, 1.0 - 2.0 * SAFE_PAD])
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    return fig, ax


def style_no_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_icon(fig, stem: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{stem}.svg"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(svg_path, transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(png_path, transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    return svg_path, png_path


def export_mechanism_landscape(out_dir: Path) -> tuple[Path, Path]:
    fig, ax = make_icon_canvas()

    x = np.linspace(0.0, 1.0, 420)
    y = np.linspace(0.0, 1.0, 240)
    xx, yy = np.meshgrid(x, y)
    diag = 0.60 * xx + 0.50 * yy
    centerline = 0.76 - 0.48 * xx + 0.06 * np.sin(1.05 * np.pi * (xx - 0.06))
    signed_dist = yy - centerline
    band = np.exp(-((signed_dist / 0.17) ** 2))

    field = diag
    field += (0.50 - field) * (0.54 * band)
    field += 0.040 * np.exp(-(((xx - 0.54) / 0.15) ** 2 + ((signed_dist) / 0.08) ** 2))
    field -= 0.015 * np.exp(-(((xx - 0.26) / 0.16) ** 2 + ((signed_dist) / 0.10) ** 2))
    field += 0.012 * np.sin(1.55 * np.pi * yy) * np.exp(-((xx - 0.50) / 0.34) ** 2)
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)
    field = np.clip(0.5 + 1.22 * (field - 0.5), 0.0, 1.0)

    ax.imshow(
        field,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap=MECHANISM_CMAP,
        interpolation="bicubic",
        aspect="auto",
        zorder=1,
    )
    ax.contourf(
        xx,
        yy,
        band,
        levels=[0.38, 1.01],
        colors=[MIX_GREY],
        alpha=0.14,
        zorder=2,
    )
    curve_x = np.linspace(0.18, 0.84, 160)
    center_curve = 0.76 - 0.48 * curve_x + 0.06 * np.sin(1.05 * np.pi * (curve_x - 0.06))
    upper_curve = center_curve + 0.10
    lower_curve = center_curve - 0.10
    ax.plot(
        curve_x,
        upper_curve,
        color="#F5F5F5",
        linewidth=1.0,
        alpha=0.34,
        zorder=3,
        solid_capstyle="round",
    )
    ax.plot(
        curve_x,
        lower_curve,
        color="#F5F5F5",
        linewidth=1.0,
        alpha=0.34,
        zorder=3,
        solid_capstyle="round",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    style_no_axes(ax)
    return save_icon(fig, ICON_MECHANISM, out_dir)


def export_design_window(out_dir: Path) -> tuple[Path, Path]:
    x = np.linspace(0.0, 1.0, 180)
    y = np.linspace(0.0, 1.0, 140)
    xx, yy = np.meshgrid(x, y)

    xf = 1.0 - xx
    yf = 1.0 - yy

    field = 0.34 * xf + 0.27 * yf + 0.23 * (xf * yf)
    field += 0.86 * np.exp(-(((xf - 0.75) / 0.18) ** 2 + ((yf - 0.72) / 0.10) ** 2))
    field += 0.28 * np.exp(-(((xf - 0.67) / 0.16) ** 2 + ((yf - 0.67) / 0.12) ** 2))
    field -= 0.14 * np.exp(-(((xf - 0.83) / 0.09) ** 2 + ((yf - 0.78) / 0.08) ** 2))
    field -= 0.10 * np.exp(-(((xf - 0.70) / 0.07) ** 2 + ((yf - 0.60) / 0.06) ** 2))
    field -= 0.10 * np.exp(-(((xf - 0.12) / 0.30) ** 2 + ((yf - 0.12) / 0.28) ** 2))
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)

    fig, ax = make_icon_canvas()
    ax.imshow(
        field,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap=FIG3_VIRIDIS,
        interpolation="bicubic",
        aspect="auto",
        zorder=1,
    )
    ax.contour(
        xx,
        yy,
        field,
        levels=[0.77],
        colors=["#FFFFFF"],
        linewidths=1.5,
        linestyles=[(0, (3.0, 2.2))],
        alpha=0.95,
        zorder=3,
    )

    max_idx = np.unravel_index(np.argmax(field), field.shape)
    peak_x = x[max_idx[1]]
    peak_y = y[max_idx[0]]
    ax.scatter(
        peak_x,
        peak_y,
        s=58,
        marker="o",
        facecolor="none",
        edgecolor="#FFFFFF",
        linewidth=1.35,
        alpha=0.92,
        zorder=4,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    style_no_axes(ax)
    return save_icon(fig, ICON_WINDOW, out_dir)


def export_decision_map(out_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(19)

    x_all = rng.uniform(0.10, 0.92, size=15)
    y_all = rng.uniform(0.12, 0.88, size=15)
    x_safe = np.array([0.18, 0.22, 0.25, 0.28, 0.24], dtype=float)
    y_safe = np.array([0.18, 0.25, 0.21, 0.15, 0.12], dtype=float)

    criterion_x = 0.32
    criterion_y = 0.34

    fig, ax = make_icon_canvas()
    ax.scatter(
        x_all,
        y_all,
        s=40,
        color="#D1D5DB",
        alpha=0.72,
        linewidths=0,
        zorder=1,
    )
    ax.axvline(
        criterion_x,
        color="#6B7280",
        linewidth=1.4,
        linestyle=(0, (6.0, 4.0)),
        alpha=0.9,
        zorder=2,
    )
    ax.axhline(
        criterion_y,
        color="#6B7280",
        linewidth=1.4,
        linestyle=(0, (6.0, 4.0)),
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        x_safe,
        y_safe,
        s=58,
        facecolor="#2E7D52",
        edgecolor="#2E7D52",
        linewidth=0.6,
        alpha=0.98,
        zorder=3,
    )

    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.98, 0.06)
    style_no_axes(ax)
    return save_icon(fig, ICON_DECISION, out_dir)


def export_safe_region(out_dir: Path) -> tuple[Path, Path]:
    x = np.linspace(0.0, 1.0, 240)
    y = np.linspace(0.0, 1.0, 180)
    xx, yy = np.meshgrid(x, y)

    bg = 0.82 * xx + 0.72 * yy
    bg += 0.09 * np.exp(-(((xx - 0.28) / 0.26) ** 2 + ((yy - 0.30) / 0.24) ** 2))
    bg -= 0.08 * np.exp(-(((xx - 0.78) / 0.28) ** 2 + ((yy - 0.78) / 0.24) ** 2))
    bg += 0.025 * np.sin(0.95 * np.pi * xx) * np.cos(0.78 * np.pi * yy)
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-12)
    bg = np.clip(0.5 + 1.22 * (bg - 0.5), 0.0, 1.0)

    region = 1.08 * np.exp(-(((xx - 0.42) / 0.29) ** 2 + ((yy - 0.43) / 0.15) ** 2))
    region += 0.28 * np.exp(-(((xx - 0.29) / 0.09) ** 2 + ((yy - 0.47) / 0.10) ** 2))
    region += 0.42 * np.exp(-(((xx - 0.57) / 0.14) ** 2 + ((yy - 0.42) / 0.13) ** 2))
    region -= 0.46 * np.exp(-(((xx - 0.43) / 0.10) ** 2 + ((yy - 0.34) / 0.08) ** 2))
    region -= 0.16 * np.exp(-(((xx - 0.26) / 0.07) ** 2 + ((yy - 0.40) / 0.07) ** 2))
    region -= 0.14 * np.exp(-(((xx - 0.63) / 0.08) ** 2 + ((yy - 0.52) / 0.08) ** 2))

    fig, ax = make_icon_canvas()
    ax.imshow(
        bg,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap=LinearSegmentedColormap.from_list(
            "safe_region_bg",
            ["#D3F2E4", "#CBE6E0", "#BCD5D9", "#9DBAC4"],
        ),
        interpolation="bicubic",
        aspect="auto",
        alpha=0.82,
        zorder=0,
    )

    for xpos in np.linspace(0.10, 0.90, 5):
        ax.plot([xpos, xpos], [0.06, 0.94], color="#D8E6E6", lw=0.8, alpha=0.11, zorder=1)
    for ypos in np.linspace(0.12, 0.88, 4):
        ax.plot([0.06, 0.94], [ypos, ypos], color="#D8E6E6", lw=0.8, alpha=0.11, zorder=1)

    ax.contourf(
        xx,
        yy,
        region,
        levels=[0.60, region.max() + 1e-6],
        colors=["#9AB3B7"],
        alpha=0.38,
        zorder=2,
    )
    ax.contour(
        xx,
        yy,
        region,
        levels=[0.60],
        colors=["#59686A"],
        linewidths=2.0,
        alpha=0.98,
        zorder=3,
    )
    curve_x = np.linspace(0.36, 0.50, 64)
    curve_y = 0.48 + 0.024 * np.sin(np.linspace(0.20, np.pi - 0.30, 64))
    ax.plot(
        curve_x,
        curve_y,
        color="#738A8D",
        lw=1.30,
        alpha=0.80,
        zorder=4,
        solid_capstyle="round",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    style_no_axes(ax)
    return save_icon(fig, ICON_SAFE, out_dir)


def main() -> None:
    args = parse_args()
    configure_style()

    exporters = {
        ICON_MECHANISM: export_mechanism_landscape,
        ICON_WINDOW: export_design_window,
        ICON_DECISION: export_decision_map,
        ICON_SAFE: export_safe_region,
    }

    for icon in args.icons:
        svg_path, png_path = exporters[icon](args.out_dir)
        print(f"[ok] {svg_path}")
        print(f"[ok] {png_path}")


if __name__ == "__main__":
    main()
