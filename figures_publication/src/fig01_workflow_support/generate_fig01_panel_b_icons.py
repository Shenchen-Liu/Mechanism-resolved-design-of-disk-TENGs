#!/usr/bin/env python3
from __future__ import annotations

import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "panel_b_icons"
MPLCONFIGDIR = SCRIPT_DIR / ".mplconfig_panel_b_icons"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import griddata
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree


OUT_SIZE_PT = (96.0, 72.0)
OUT_SIZE_IN = (OUT_SIZE_PT[0] / 72.0, OUT_SIZE_PT[1] / 72.0)
DPI = 144

MECHANISM_TABLE = ROOT / "outputs_mechanism_multitask" / "mechanism_analysis_table.csv"
ASHBY_CSV = ROOT / "figures_publication" / "src" / "fig04_t11_ashby_data.csv"
VALIDATION_CSV = ROOT / "figures_publication" / "src" / "fig05_validation_points.csv"
MANUAL_OPEN_TOOL_SCREENSHOT = SCRIPT_DIR / "open_design_tool_manual.png"

OKABE_ITO = {
    "V1": "#E69F00",
    "V2": "#56B4E9",
    "V3": "#009E73",
}
FIG2C_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fig2c_blue_gray_red",
    ["#2C7FB8", "#BDBDBD", "#D7301F"],
)
FIG4_CV_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fig4_cv_muted",
    ["#2E7F9E", "#4DB29F", "#69B86A", "#B8D65B"],
)

ICON_BUILDERS = (
    ("panel_b_icon_comsol_dataset", "fig01_panel_b_icon_01_comsol_dataset.svg"),
    ("panel_b_icon_channel_decomposition", "fig01_panel_b_icon_02_channel_decomposition.svg"),
    ("panel_b_icon_landscape_construction", "fig01_panel_b_icon_03_landscape_construction.svg"),
    ("panel_b_icon_robust_screening", "fig01_panel_b_icon_04_robust_screening.svg"),
    ("panel_b_icon_unseen_validation", "fig01_panel_b_icon_05_unseen_validation.svg"),
    ("panel_b_icon_open_design_tool", "fig01_panel_b_icon_06_open_design_tool.png"),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def make_figure():
    fig = plt.figure(figsize=OUT_SIZE_IN, dpi=DPI)
    ax = fig.add_axes([0.12, 0.14, 0.80, 0.74])
    return fig, ax


def style_axes(
    ax,
    *,
    show_labels: bool = False,
    full_frame_light: bool = True,
    tick_size: float = 6.0,
    axis_color: str = "#64748B",
    frame_color: str = "#D1D5DB",
    axis_width: float = 0.8,
    frame_width: float = 0.7,
) -> None:
    ax.spines["top"].set_visible(full_frame_light)
    ax.spines["right"].set_visible(full_frame_light)
    ax.spines["bottom"].set_color(axis_color)
    ax.spines["left"].set_color(axis_color)
    ax.spines["bottom"].set_linewidth(axis_width)
    ax.spines["left"].set_linewidth(axis_width)
    ax.spines["top"].set_color(frame_color)
    ax.spines["right"].set_color(frame_color)
    ax.spines["top"].set_linewidth(frame_width)
    ax.spines["right"].set_linewidth(frame_width)
    ax.tick_params(axis="both", which="major", labelsize=tick_size, colors=axis_color, length=2.1, pad=1.0)
    if not show_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def save_svg(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", transparent=True)
    plt.close(fig)


def sample_evenly(df: pd.DataFrame, n: int, sort_col: str) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    ordered = df.sort_values(sort_col).reset_index(drop=True)
    idx = np.linspace(0, len(ordered) - 1, n, dtype=int)
    return ordered.iloc[idx].copy()


def select_spread_points(df: pd.DataFrame, n: int = 18) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    work = df[["logQ", "logInvC", "logFOMS"]].copy()
    for col in work.columns:
        span = work[col].max() - work[col].min()
        if span > 0:
            work[col] = (work[col] - work[col].min()) / span
    points = work.to_numpy()
    score = 0.55 * points[:, 0] + 0.30 * (1.0 - points[:, 1]) + 0.15 * points[:, 2]
    first = int(np.argmax(score))
    chosen = [first]
    remaining = set(range(len(df))) - {first}
    while len(chosen) < n and remaining:
        chosen_pts = points[chosen]
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            dist = np.sqrt(((chosen_pts - points[idx]) ** 2).sum(axis=1)).min()
            weighted = dist + 0.12 * score[idx]
            if weighted > best_score:
                best_score = weighted
                best_idx = idx
        chosen.append(best_idx)
        remaining.remove(best_idx)
    return df.iloc[sorted(chosen)].copy()


def compute_local_f_charge(df: pd.DataFrame, k: int = 50) -> np.ndarray:
    coords = np.column_stack([df["logQ"].to_numpy(), df["logInvC"].to_numpy()])
    tree = BallTree(coords)
    k = min(k, len(df))
    out = np.full(len(df), np.nan)
    for i, point in enumerate(coords):
        _, idx = tree.query(point.reshape(1, -1), k=k)
        local_q = coords[idx[0], 0]
        local_c = coords[idx[0], 1]
        sq = np.std(local_q)
        sc = np.std(local_c)
        denom = sq + sc
        out[i] = sq / denom if denom > 1e-12 else 0.5
    return out


def build_comsol_dataset(path: Path) -> None:
    df = pd.read_csv(MECHANISM_TABLE)
    grid = (
        df.groupby(["hh", "dd"])["logFOMS"]
        .mean()
        .unstack("dd")
        .sort_index()
    )

    fig, ax = make_figure()
    ax.set_position([0.12, 0.20, 0.72, 0.66])
    ny, nx = grid.shape
    x_edges = np.arange(nx + 1)
    y_edges = np.arange(ny + 1)
    ax.pcolormesh(
        x_edges,
        y_edges,
        grid.to_numpy(),
        cmap="viridis",
        shading="flat",
        edgecolors=(1.0, 1.0, 1.0, 0.20),
        linewidth=0.35,
    )
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xticks([1.8, 4.8])
    ax.set_yticks([2.2, 6.6])
    style_axes(ax, show_labels=False)
    fig.text(
        0.955,
        0.028,
        "N=1,944",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#6B7280",
    )
    save_svg(fig, path)


def build_channel_decomposition(path: Path) -> None:
    df = pd.read_csv(MECHANISM_TABLE)
    sample = select_spread_points(df, 18)
    x = sample["logQ"].to_numpy()
    y = sample["logInvC"].to_numpy()
    c = sample["logFOMS"].to_numpy()

    fig, ax = make_figure()
    ax.scatter(x, y, c=c, cmap="viridis", s=13, linewidths=0.0, alpha=0.72, zorder=2)
    x_pad = 0.06 * (x.max() - x.min())
    y_pad = 0.08 * (y.max() - y.min())
    x_lo, x_hi = x.min() - x_pad, x.max() + x_pad
    y_lo, y_hi = y.min() - y_pad, y.max() + y_pad
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    x_line = np.linspace(x_lo, x_hi, 100)
    y_line = np.linspace(y_hi - 0.10 * (y_hi - y_lo), y_lo + 0.10 * (y_hi - y_lo), 100)
    ax.plot(x_line, y_line, color="#94A3B8", lw=0.8, alpha=0.58, ls=(0, (2.5, 2.0)), zorder=1)
    ax.set_xticks(np.quantile(x, [0.28, 0.78]))
    ax.set_yticks(np.quantile(y, [0.30, 0.78]))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    style_axes(ax, show_labels=False, tick_size=6.0)
    save_svg(fig, path)


def build_landscape_construction(path: Path) -> None:
    df = pd.read_csv(MECHANISM_TABLE)
    df = df.copy()
    df["f_charge_local"] = compute_local_f_charge(df)
    roi = df[
        (df["logQ"] >= -20.00)
        & (df["logQ"] <= -19.13)
        & (df["logInvC"] >= 10.94)
        & (df["logInvC"] <= 11.24)
    ].copy()
    roi["f_charge_display"] = np.clip(
        0.5 + 3.85 * (roi["f_charge_local"] - 0.5),
        0.06,
        0.94,
    )
    gx = np.linspace(roi["logQ"].min(), roi["logQ"].max(), 60)
    gy = np.linspace(roi["logInvC"].min(), roi["logInvC"].max(), 40)
    xx, yy = np.meshgrid(gx, gy)
    zz = griddata(
        roi[["logQ", "logInvC"]].to_numpy(),
        roi["f_charge_display"].to_numpy(),
        (xx, yy),
        method="linear",
    )
    if np.isnan(zz).any():
        zz_near = griddata(
            roi[["logQ", "logInvC"]].to_numpy(),
            roi["f_charge_display"].to_numpy(),
            (xx, yy),
            method="nearest",
        )
        zz = np.where(np.isnan(zz), zz_near, zz)

    fig, ax = make_figure()
    ax.imshow(
        zz,
        origin="lower",
        extent=(roi["logQ"].min(), roi["logQ"].max(), roi["logInvC"].min(), roi["logInvC"].max()),
        cmap=plt.cm.coolwarm,
        interpolation="lanczos",
        aspect="auto",
        norm=mcolors.TwoSlopeNorm(vmin=0.06, vcenter=0.5, vmax=0.94),
    )
    ax.set_xlim(roi["logQ"].min(), roi["logQ"].max())
    ax.set_ylim(roi["logInvC"].min(), roi["logInvC"].max())
    ax.set_xticks(np.quantile(roi["logQ"], [0.15, 0.85]))
    ax.set_yticks(np.quantile(roi["logInvC"], [0.15, 0.85]))
    style_axes(ax, show_labels=False, axis_color="#94A3B8", frame_color="#D6DBE1", axis_width=0.7, frame_width=0.65)
    save_svg(fig, path)


def build_robust_screening(path: Path) -> None:
    df = pd.read_csv(ASHBY_CSV)
    sub = df[df["scenario"] == "low_E_mid_d"].copy()
    n_values = np.sort(sub["n"].unique())
    hh_values = np.sort(sub["hh"].unique())
    cv_map = (
        sub.pivot(index="hh", columns="n", values="cv_pct")
        .reindex(index=hh_values, columns=n_values)
        .to_numpy()
    )
    safe_map = (
        sub.pivot(index="hh", columns="n", values="safe_zone")
        .reindex(index=hh_values, columns=n_values)
        .fillna(False)
        .to_numpy(dtype=float)
    )
    safe_idx = np.argwhere(safe_map > 0.5)
    y0, x0 = safe_idx.min(axis=0)
    y1, x1 = safe_idx.max(axis=0)
    y0 = max(0, y0 - 2)
    y1 = min(safe_map.shape[0] - 1, y1 + 2)
    x0 = max(0, x0 - 1)
    x1 = min(safe_map.shape[1] - 1, x1 + 1)
    cv_crop = cv_map[y0 : y1 + 1, x0 : x1 + 1]
    safe_crop = safe_map[y0 : y1 + 1, x0 : x1 + 1]
    x = np.arange(cv_crop.shape[1], dtype=float)
    y = np.arange(cv_crop.shape[0], dtype=float)
    xx, yy = np.meshgrid(x, y)
    gx = np.linspace(x.min(), x.max(), 90)
    gy = np.linspace(y.min(), y.max(), 120)
    gxx, gyy = np.meshgrid(gx, gy)
    cv_fine = griddata((xx.ravel(), yy.ravel()), cv_crop.ravel(), (gxx, gyy), method="cubic")
    if np.isnan(cv_fine).any():
        cv_near = griddata((xx.ravel(), yy.ravel()), cv_crop.ravel(), (gxx, gyy), method="nearest")
        cv_fine = np.where(np.isnan(cv_fine), cv_near, cv_fine)
    safe_fine = griddata((xx.ravel(), yy.ravel()), safe_crop.ravel(), (gxx, gyy), method="nearest")

    fig, ax = make_figure()
    ax.imshow(
        cv_fine,
        origin="lower",
        extent=(x.min(), x.max(), y.min(), y.max()),
        cmap=FIG4_CV_CMAP,
        interpolation="bicubic",
        aspect="auto",
    )
    cv_levels = [np.nanquantile(cv_fine, 0.50)]
    ax.contour(
        gxx,
        gyy,
        cv_fine,
        levels=cv_levels,
        colors=["#90A89E"],
        linewidths=0.55,
        alpha=0.75,
        zorder=1.8,
    )
    ax.contourf(
        gxx,
        gyy,
        safe_fine,
        levels=[0.5, 1.1],
        colors=["#D8F0E4"],
        alpha=0.72,
        zorder=2,
    )
    ax.contour(
        gxx,
        gyy,
        safe_fine,
        levels=[0.5],
        colors=["#556B65"],
        linewidths=0.95,
        zorder=3,
    )
    ax.set_xticks(np.linspace(x.min(), x.max(), 2))
    ax.set_yticks(np.linspace(y.min(), y.max(), 2))
    style_axes(ax, show_labels=False)
    save_svg(fig, path)


def build_unseen_validation(path: Path) -> None:
    df = pd.read_csv(VALIDATION_CSV).copy()
    df["true_log"] = np.log10(df["FOMS_direct_true"])
    df["pred_log"] = np.log10(df["FOMS_direct_pred"])
    label_map = {"validate1": "V1", "validate2": "V2", "validate3": "V3"}

    fig, ax = make_figure()
    lo = min(df["true_log"].min(), df["pred_log"].min())
    hi = max(df["true_log"].max(), df["pred_log"].max())
    ax.plot([lo, hi], [lo, hi], color="#94A3B8", lw=0.9, zorder=1)

    x_plot = df["true_log"].to_numpy().copy()
    y_plot = df["pred_log"].to_numpy().copy()
    dense_mask = (x_plot >= np.quantile(x_plot, 0.72)) & (y_plot >= np.quantile(y_plot, 0.72))
    dense_idx = np.where(dense_mask)[0]
    if len(dense_idx) > 0:
        span = hi - lo
        offsets = np.array(
            [
                [-0.015, 0.008],
                [0.014, -0.010],
                [-0.010, -0.012],
                [0.011, 0.010],
                [-0.006, 0.015],
                [0.016, 0.004],
                [-0.014, -0.004],
                [0.005, -0.016],
            ]
        ) * span
        for j, idx in enumerate(dense_idx):
            dx, dy = offsets[j % len(offsets)]
            x_plot[idx] += dx
            y_plot[idx] += dy

    for dataset, label in label_map.items():
        sub = df[df["dataset"] == dataset].copy()
        idx = sub.index.to_numpy()
        ax.scatter(
            x_plot[idx],
            y_plot[idx],
            s=8,
            color=OKABE_ITO[label],
            linewidths=0.0,
            alpha=0.90,
            zorder=2,
        )

    r2 = r2_score(df["true_log"], df["pred_log"])
    ax.text(
        0.03,
        0.96,
        f"OOD, R²log={r2:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        fontweight="bold",
        color="#111827",
    )
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(np.linspace(lo, hi, 2))
    ax.set_yticks(np.linspace(lo, hi, 2))
    style_axes(ax, show_labels=False)
    ax.tick_params(labelbottom=False, labelleft=False)
    save_svg(fig, path)


def build_open_design_tool(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if MANUAL_OPEN_TOOL_SCREENSHOT.exists():
        with Image.open(MANUAL_OPEN_TOOL_SCREENSHOT) as img:
            img = img.convert("RGBA")
            w, h = img.size
            crop_w = int(w * 0.52)
            crop_h = int(h * 0.40)
            left = int(w * 0.20)
            top = int(h * 0.18)
            cropped = img.crop((left, top, left + crop_w, top + crop_h))
            out = cropped.resize((int(OUT_SIZE_PT[0] * 2), int(OUT_SIZE_PT[1] * 2)), Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", out.size, (255, 255, 255, 0))
            canvas.paste(out, (0, 0))
            border = Image.new("RGBA", out.size, (0, 0, 0, 0))
            canvas.save(path)
    else:
        fig, ax = make_figure()
        ax.set_axis_off()
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, edgecolor="#CBD5E1", linewidth=1.0, transform=ax.transAxes))
        ax.text(0.5, 0.5, "manual\nscreenshot", ha="center", va="center", fontsize=7, color="#94A3B8", transform=ax.transAxes)
        fig.savefig(path, format="png", transparent=True)
        plt.close(fig)
        return

    # Draw border on the saved PNG.
    with Image.open(path) as img:
        img = img.convert("RGBA")
        arr = np.array(img)
        arr[0:2, :, :3] = [203, 213, 225]
        arr[-2:, :, :3] = [203, 213, 225]
        arr[:, 0:2, :3] = [203, 213, 225]
        arr[:, -2:, :3] = [203, 213, 225]
        Image.fromarray(arr).save(path)


BUILD_FN = {
    "panel_b_icon_comsol_dataset": build_comsol_dataset,
    "panel_b_icon_channel_decomposition": build_channel_decomposition,
    "panel_b_icon_landscape_construction": build_landscape_construction,
    "panel_b_icon_robust_screening": build_robust_screening,
    "panel_b_icon_unseen_validation": build_unseen_validation,
    "panel_b_icon_open_design_tool": build_open_design_tool,
}


def generate_panel_b_icons(out_dir: Path | None = None) -> dict[str, Path]:
    configure_style()
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for key, filename in ICON_BUILDERS:
        path = out_dir / filename
        BUILD_FN[key](path)
        outputs[key] = path
    return outputs


def main() -> None:
    outputs = generate_panel_b_icons()
    for key, path in outputs.items():
        print(f"[ok] {key}: {path}")


if __name__ == "__main__":
    main()
