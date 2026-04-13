#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the main-text publication figures and source tables.

The script reads the released model, processed datasets, and exported analysis
artifacts, then writes figure files into `figures_publication/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig_publication"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter, zoom as nd_zoom
from sklearn.neighbors import BallTree

from utils_mechanism_multitask import (
    REGIME_CAPACITANCE,
    REGIME_CHARGE,
    REGIME_MIXED,
    compute_design_regime_grid,
    compute_robustness_grid,
)

DATA_DIR = ROOT / "data"
MECHANISM_TABLE = ROOT / "outputs_mechanism_multitask" / "mechanism_analysis_table.csv"
FIG01_FINAL_STEM = "fig01_method_workflow"
TEST_METRICS = ROOT / "outputs_multitask_physics" / "test_metrics.json"
CV_RESULTS = ROOT / "outputs" / "cross_validation" / "cv_results.json"
DESIGN_RULES_RAW = ROOT / "outputs" / "comparison_tables" / "table_design_rules.csv"

OUT_ROOT = ROOT / "figures_publication"
MAIN_DIR = OUT_ROOT / "main"
PREVIEW_DIR = OUT_ROOT / "preview"
SRC_DIR = OUT_ROOT / "src"

FORMATS = ("png", "pdf", "svg")
ZOOM_LOGQ_MIN = -18.5
N_VALUES = np.array([2, 4, 8, 16, 32, 64], dtype=float)
HH_VALUES = np.geomspace(0.00390625, 1.0, 18)
DESIGN_DELTA_LOG = 0.02
DESIGN_DOMINANCE_THRESHOLD = 0.62
PERTURB_FRAC = 0.10
REGIME_K = 50
FIG3_BALANCE_ALPHA = 1.00
FIG4_CV_LAMBDA = 0.45
FIG4_RETENTION_GAMMA = 0.20
SAFE_SMOOTH_SIGMA = 0.9

SCENARIOS = [
    {
        "E": 1.0,
        "dd": 0.125,
        "label": "low_E_mid_d",
        "title": r"low $\varepsilon$, mid $d/R$",
    },
    {
        "E": 3.0,
        "dd": 0.125,
        "label": "mid_E_mid_d",
        "title": r"mid $\varepsilon$, mid $d/R$",
    },
    {
        "E": 10.0,
        "dd": 0.125,
        "label": "high_E_mid_d",
        "title": r"high $\varepsilon$, mid $d/R$",
    },
    {
        "E": 3.0,
        "dd": 0.5,
        "label": "mid_E_large_d",
        "title": r"mid $\varepsilon$, large $d/R$",
    },
]

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
BRIDGE = {
    "cap": "#2C7FB8",
    "mixed": "#B9BFC7",
    "charge": "#E64B35",
    "text": "#1F2937",
    "muted": "#6B7280",
    "bg": "#F8FAFC",
}

FIG3_SCENARIOS = SCENARIOS[:3]
FIG4_HEATMAP_SCENARIOS = SCENARIOS[:2]
ASHBY_SCENARIOS = SCENARIOS[:3]
SAFE_CV_THRESHOLD = 0.05
SAFE_WORST_RATIO_THRESHOLD = 0.90
REGIME_COLORS = {
    REGIME_CAPACITANCE: "#2C7FB8",
    REGIME_MIXED: "#BDBDBD",
    REGIME_CHARGE: "#D7301F",
}
DOMINANCE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dominance_blue_gray_red",
    [
        (0.0, REGIME_COLORS[REGIME_CAPACITANCE]),
        (0.5, REGIME_COLORS[REGIME_MIXED]),
        (1.0, REGIME_COLORS[REGIME_CHARGE]),
    ],
)
FIG4_CV_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "fig4_cv_muted",
    ["#2E7F9E", "#4DB29F", "#69B86A", "#B8D65B"],
)
SAFE_GREEN = "#BFE2D8"
SAFE_GREEN_SOFT = "#A7D6C9"
SAFE_CANDIDATE_FACE = SAFE_GREEN_SOFT
SAFE_CANDIDATE_ALPHA = 0.96
SAFE_CANDIDATE_EDGE = "#5A7C72"
REPRESENTATIVE_FACE = "white"
REPRESENTATIVE_EDGE = "#25313A"
REPRESENTATIVE_SIZE = 62
REPRESENTATIVE_MARKERSIZE = 6.1
REPRESENTATIVE_EDGEWIDTH = 1.1
ASHBY_ALL_FACE = "#DCE3E8"
ASHBY_ALL_ALPHA = 0.52
LINE_COLORS = {
    "q2": "#CC6B00",
    "invc": "#1D4E89",
    "foms": "#1B7F3A",
}
# fig3g 顶部三个区域说明，坐标使用数据坐标，便于手动微调。
FIG3G_REGION_LABELS = (
    {"x": 2.0, "y": 1.065, "text": "cap.-dom.", "color": "#1D4E89", "ha": "left"},
    {"x": 12.0, "y": 1.065, "text": "mixed", "color": "#6B7280", "ha": "center"},
    {
        "x": 75.0,
        "y": 1.065,
        "text": "charge-dom.",
        "color": "#8A3B12",
        "ha": "right",
    },
)
FIG_TITLE_SIZE = 8.2
FIG_AXIS_LABEL_SIZE = 8.6
FIG_TICK_LABEL_SIZE = 7.0
FIG_CBAR_TICK_SIZE = 7.0
FIG_PANEL_LABEL_SIZE = 10.4
FIG_CALLOUT_SIZE = 7.0
FIG_CALLOUT_SMALL_SIZE = 6.8
SHORT_DASH = (0, (2.2, 1.8))
PERF_REFERENCE_DASH = (0, (1.5, 2.5))
PERF_REFERENCE_COLOR = "#D8DDE2"
PERF_REFERENCE_ALPHA = 0.72
PERF_REFERENCE_WIDTH = 0.42
SAFE_ZONE_ALPHA = 0.50
SAFE_ZONE_EDGE_ALPHA = 0.98
SAFE_ZONE_EDGE = "#526D66"
CRITERION_LINE_COLOR = "#8E98A2"
CRITERION_LINE_ALPHA = 0.78
CRITERION_LINE_WIDTH = 0.72
FIG_TITLE_PAD = 6.0
ALL_FIG_KEYS = ("fig01", "fig02", "fig03", "fig04", "fig05")


def rel_path(path: Path) -> str:
    """Return a repository-relative POSIX path for public manifests."""
    return path.relative_to(ROOT).as_posix()


def ensure_dirs():
    for path in [OUT_ROOT, MAIN_DIR, PREVIEW_DIR, SRC_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_style():
    sns.set_theme(style="white", context="paper", font="Arial")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def hide_extra_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label, x=-0.14, y=1.08):
    label = str(label).lower()
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=FIG_PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
    )


def style_colorbar(
    cbar, label, *, fontsize=FIG_AXIS_LABEL_SIZE, labelpad=2.0, n_ticks=5
):
    axis = cbar.ax.yaxis
    if getattr(cbar, "orientation", "vertical") == "horizontal":
        axis = cbar.ax.xaxis
    axis.set_major_locator(MaxNLocator(nbins=n_ticks))
    cbar.set_label(label, fontsize=fontsize, labelpad=labelpad)
    cbar.ax.tick_params(labelsize=FIG_CBAR_TICK_SIZE)
    cbar.update_ticks()


def style_mechanism_axes(ax):
    ax.grid(alpha=0.14, linewidth=0.4)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=4.0,
        width=0.9,
        color="#344054",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=2.5,
        width=0.6,
        color="#667085",
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    hide_extra_spines(ax)


def save_figure(fig, stem):
    main_files = []
    for ext in FORMATS:
        path = MAIN_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
        main_files.append(rel_path(path))
    preview_path = PREVIEW_DIR / f"{stem}.png"
    fig.savefig(preview_path, bbox_inches="tight", pad_inches=0.06, dpi=300)
    return main_files, rel_path(preview_path)


def install_fig1_assets():
    """Copy the finalized Fig.1 assets into the publication tree."""
    src_files = {}
    for ext in FORMATS:
        src_candidate = SRC_DIR / f"{FIG01_FINAL_STEM}.{ext}"
        main_candidate = MAIN_DIR / f"{FIG01_FINAL_STEM}.{ext}"
        if src_candidate.exists():
            src_files[ext] = src_candidate
        elif main_candidate.exists():
            src_files[ext] = main_candidate
        else:
            src_files[ext] = src_candidate
    missing = [str(path) for path in src_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing finalized Fig.1 assets: " + ", ".join(missing)
        )

    main_files: list[str] = []
    for ext in FORMATS:
        src_path = src_files[ext]
        dst_path = MAIN_DIR / src_path.name
        if src_path.resolve() != dst_path.resolve():
            shutil.copy2(src_path, dst_path)
        main_files.append(rel_path(dst_path))

    preview_src = src_files["png"]
    preview_path = PREVIEW_DIR / preview_src.name
    if preview_src.resolve() != preview_path.resolve():
        shutil.copy2(preview_src, preview_path)

    editable_source = SRC_DIR / f"{FIG01_FINAL_STEM}.drawio"
    editable_xml = SRC_DIR / f"{FIG01_FINAL_STEM}.xml"
    payload = {
        "main_files": main_files,
        "preview_file": rel_path(preview_path),
        "placeholder": False,
        "replacement_needed": False,
        "source": rel_path(editable_source if editable_source.exists() else preview_src),
    }
    if editable_xml.exists():
        payload["editable_xml"] = rel_path(editable_xml)
    return payload


def make_predict_fn(model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device):
    from predict_multitask_physics import predict_batch

    def predict_fn(n_arr, E_arr, dd_arr, hh_arr):
        df = pd.DataFrame(
            {
                "n": n_arr,
                "E": E_arr,
                "dd": dd_arr,
                "hh": hh_arr,
            }
        )
        result = predict_batch(
            df,
            model=model,
            scaler_X=scaler_X,
            scaler_qsc=scaler_qsc,
            scaler_invc=scaler_invc,
            scaler_foms=scaler_foms,
            device=device,
        )
        return {
            "Qsc_MACRS": result["Qsc_MACRS_pred"].to_numpy(),
            "invC_sum": result["invC_sum_pred"].to_numpy(),
            "FOMS_direct": result["FOMS_direct_pred"].to_numpy(),
            "FOMS_phys": result["FOMS_phys_pred"].to_numpy(),
        }

    return predict_fn


def compute_local_f_charge(df):
    logq = df["logQ"].to_numpy()
    loginvc = df["logInvC"].to_numpy()
    coords = np.column_stack([logq, loginvc])
    tree = BallTree(coords)
    k = min(REGIME_K, len(df))
    out = np.full(len(df), np.nan)
    for idx, point in enumerate(coords):
        _, nn_idx = tree.query(point.reshape(1, -1), k=k)
        local_q = logq[nn_idx[0]]
        local_c = loginvc[nn_idx[0]]
        sq = np.std(local_q)
        sc = np.std(local_c)
        denom = sq + sc
        out[idx] = sq / denom if denom > 1e-12 else 0.5
    return out


def log2_mesh():
    return np.meshgrid(np.log2(N_VALUES), np.log2(HH_VALUES))


def configure_design_axes(ax, show_xlabel, show_ylabel):
    hh_ticks = np.array([0.004, 0.01, 0.03, 0.1, 0.3, 1.0], dtype=float)
    hh_ticks = hh_ticks[(hh_ticks >= HH_VALUES.min()) & (hh_ticks <= HH_VALUES.max())]
    ax.set_xticks(np.log2(N_VALUES))
    ax.set_xticklabels([str(int(v)) for v in N_VALUES], fontsize=FIG_TICK_LABEL_SIZE)
    ax.set_yticks(np.log2(hh_ticks))
    ax.set_yticklabels([f"{v:g}" for v in hh_ticks], fontsize=FIG_TICK_LABEL_SIZE)
    ax.set_xlabel(
        r"$n$ (log scale)" if show_xlabel else "",
        fontsize=FIG_AXIS_LABEL_SIZE,
        labelpad=1.5,
    )
    ax.set_ylabel(
        r"$h/R$ (log scale)" if show_ylabel else "",
        fontsize=FIG_AXIS_LABEL_SIZE,
        labelpad=1.5,
    )
    ax.tick_params(length=2.5, pad=2, labelsize=FIG_TICK_LABEL_SIZE)
    hide_extra_spines(ax)


def format_scenario_title(scenario):
    return scenario["title"]


def annotation_offset(row, col, n_rows, n_cols):
    dx = 6
    dy = 5
    ha = "left"
    va = "bottom"
    if col >= n_cols - 2:
        dx = -6
        ha = "right"
    if row >= n_rows - 3:
        dy = -6
        va = "top"
    elif row <= 2:
        dy = 6
        va = "bottom"
    return dx, dy, ha, va


def pick_best_index(values, mask, prefer_max=True):
    valid_idx = np.argwhere(mask)
    if len(valid_idx) == 0:
        return None
    subset = values[mask]
    order = np.nanargmax(subset) if prefer_max else np.nanargmin(subset)
    return tuple(valid_idx[order])


def normalize_map(values, mask):
    out = np.zeros_like(values, dtype=float)
    subset = values[mask]
    if subset.size == 0:
        return out
    vmin = np.nanmin(subset)
    vmax = np.nanmax(subset)
    span = vmax - vmin
    if not np.isfinite(span) or span < 1e-12:
        out[mask] = 1.0
        return out
    out[mask] = (values[mask] - vmin) / span
    return out


def select_design_point(result):
    foms = result["foms"]
    f_charge = result["f_charge"]
    top30 = foms >= np.nanpercentile(foms, 70)
    norm_fom = normalize_map(foms, top30)
    balance_penalty = np.abs(f_charge - 0.5) / 0.5
    score = np.full_like(foms, -np.inf, dtype=float)
    score[top30] = norm_fom[top30] - FIG3_BALANCE_ALPHA * balance_penalty[top30]

    idx = pick_best_index(score, top30, prefer_max=True)
    label = "recommend"
    if idx is None:
        idx = np.unravel_index(np.nanargmax(foms), foms.shape)
        label = "high-performance"

    row, col = idx
    return {
        "row": int(row),
        "col": int(col),
        "label": label,
        "n": float(N_VALUES[col]),
        "hh": float(HH_VALUES[row]),
        "foms": float(foms[row, col]),
        "f_charge": float(f_charge[row, col]),
        "regime": float(result["regime"][row, col]),
        "score": float(score[row, col]) if np.isfinite(score[row, col]) else np.nan,
        "balance_penalty": float(balance_penalty[row, col]),
    }


def select_robust_points(foms_map, cv_map, worst_ratio_map):
    top30 = foms_map >= np.nanpercentile(foms_map, 70)
    safe = top30 & (cv_map <= 0.05)
    norm_fom = normalize_map(foms_map, top30)
    norm_cv = normalize_map(cv_map, top30)
    norm_retention = normalize_map(worst_ratio_map, top30)
    score = np.full_like(foms_map, -np.inf, dtype=float)
    score[top30] = (
        norm_fom[top30]
        - FIG4_CV_LAMBDA * norm_cv[top30]
        + FIG4_RETENTION_GAMMA * norm_retention[top30]
    )

    label = "recommend"
    rep_idx = pick_best_index(score, top30, prefer_max=True)
    if rep_idx is None:
        rep_idx = np.unravel_index(np.nanargmin(cv_map), cv_map.shape)
        label = "low-CV point"

    hp_idx = np.unravel_index(np.nanargmax(foms_map), foms_map.shape)
    return {
        "safe_mask": safe,
        "safe_overlay": gaussian_filter(safe.astype(float), sigma=SAFE_SMOOTH_SIGMA),
        "recommended": {
            "row": int(rep_idx[0]),
            "col": int(rep_idx[1]),
            "label": label,
            "n": float(N_VALUES[rep_idx[1]]),
            "hh": float(HH_VALUES[rep_idx[0]]),
            "foms": float(foms_map[rep_idx]),
            "cv": float(cv_map[rep_idx]),
            "worst_ratio": float(worst_ratio_map[rep_idx]),
            "score": float(score[rep_idx]) if np.isfinite(score[rep_idx]) else np.nan,
        },
        "high_perf": {
            "row": int(hp_idx[0]),
            "col": int(hp_idx[1]),
            "n": float(N_VALUES[hp_idx[1]]),
            "hh": float(HH_VALUES[hp_idx[0]]),
            "foms": float(foms_map[hp_idx]),
            "cv": float(cv_map[hp_idx]),
            "worst_ratio": float(worst_ratio_map[hp_idx]),
        },
    }


def add_bridge_card(ax, x, y, w, h, accent, title, lines, *, highlight_last=False):
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="#D0D5DD",
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.add_patch(
        Rectangle(
            (x, y), 0.018, h, transform=ax.transAxes, facecolor=accent, edgecolor="none"
        )
    )
    ax.text(
        x + 0.032,
        y + h - 0.04,
        title,
        transform=ax.transAxes,
        fontsize=FIG_CALLOUT_SIZE,
        fontweight="bold",
        color=BRIDGE["text"],
        va="top",
        clip_on=True,
        clip_path=card,
    )
    line_gap = 0.036 if len(lines) >= 4 else 0.041
    line_fontsize = 6.0 if len(lines) >= 4 else 6.3
    start_y = y + h - 0.088
    for i, line in enumerate(lines):
        is_last = highlight_last and i == len(lines) - 1
        ax.text(
            x + 0.032,
            start_y - i * line_gap,
            line,
            transform=ax.transAxes,
            fontsize=line_fontsize,
            fontweight="semibold" if is_last else "medium",
            fontstyle="italic" if is_last else "normal",
            color=accent if is_last else "#475467",
            va="top",
            clip_on=True,
            clip_path=card,
        )


def build_fig2():
    print("\n[Fig.2] Mechanism landscape ...")
    df = pd.read_csv(MECHANISM_TABLE)
    df["f_charge_local"] = compute_local_f_charge(df)
    df["logFOMS_norm_n"] = np.log10(np.maximum(df["FOMS_norm_n"].to_numpy(), 1e-30))
    zoom_df = df[df["logQ"] >= ZOOM_LOGQ_MIN].copy()

    perf_vmin_full = np.nanpercentile(df["logFOMS_norm_n"], 3)
    perf_vmax_full = np.nanpercentile(df["logFOMS_norm_n"], 98)
    # Unified colorbar range for A and B
    perf_vmin_shared = perf_vmin_full
    perf_vmax_shared = perf_vmax_full
    fc_norm = mcolors.TwoSlopeNorm(vmin=0.35, vcenter=0.5, vmax=0.65)

    # --- Wider figure, more generous spacing ---
    fig = plt.figure(figsize=(7.5, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # --- Panel A: full overview ---
    hb_a = ax_a.hexbin(
        df["logQ"],
        df["logInvC"],
        C=df["logFOMS_norm_n"],
        reduce_C_function=np.mean,
        gridsize=28,
        cmap="viridis",
        mincnt=1,
        linewidths=0.30,
        edgecolors="white",
        vmin=perf_vmin_shared,
        vmax=perf_vmax_shared,
    )
    cb_a = fig.colorbar(hb_a, ax=ax_a, pad=0.02, fraction=0.046)
    style_colorbar(cb_a, r"Mean $\log_{10}(FOM_S)$")
    zoom_ymin = zoom_df["logInvC"].min() - 0.04
    zoom_ymax = zoom_df["logInvC"].max() + 0.04
    ax_a.add_patch(
        Rectangle(
            (ZOOM_LOGQ_MIN, zoom_ymin),
            df["logQ"].max() - ZOOM_LOGQ_MIN + 0.03,
            zoom_ymax - zoom_ymin,
            fill=False,
            linewidth=0.82,
            edgecolor="#667085",
        )
    )
    ax_a.text(
        ZOOM_LOGQ_MIN + 2,
        zoom_ymax + 0.045,
        "zoom band",
        fontsize=FIG_CALLOUT_SMALL_SIZE,
        color="#667085",
        ha="right",
        va="bottom",
    )
    ax_a.set_title(
        "Mechanism-space overview",
        loc="left",
        fontweight="bold",
        fontsize=8,
        pad=FIG_TITLE_PAD,
    )
    ax_a.set_xlabel(r"$\log_{10}(Q_{sc,MACRS}^{2})$", fontsize=8)
    ax_a.set_ylabel(r"$\log_{10}(1/C_{start}+1/C_{end})$", fontsize=8)
    style_mechanism_axes(ax_a)
    add_panel_label(ax_a, "A", x=-0.13, y=1.04)

    # --- Panel B: zoom performance (same colorbar as A) ---
    hb_b = ax_b.hexbin(
        zoom_df["logQ"],
        zoom_df["logInvC"],
        C=zoom_df["logFOMS_norm_n"],
        reduce_C_function=np.mean,
        gridsize=24,
        cmap="viridis",
        mincnt=1,
        linewidths=0.35,
        edgecolors="white",
        vmin=perf_vmin_shared,
        vmax=perf_vmax_shared,
    )
    cb_b = fig.colorbar(hb_b, ax=ax_b, pad=0.02, fraction=0.046)
    style_colorbar(cb_b, r"Mean $\log_{10}(FOM_S)$")
    ax_b.set_title(
        "Performance landscape (zoom)",
        loc="left",
        fontweight="bold",
        fontsize=8,
        pad=FIG_TITLE_PAD,
    )
    ax_b.set_xlabel(r"$\log_{10}(Q_{sc,MACRS}^{2})$", fontsize=8)
    ax_b.set_ylabel(r"$\log_{10}(1/C_{start}+1/C_{end})$", fontsize=8, labelpad=2)
    style_mechanism_axes(ax_b)
    add_panel_label(ax_b, "B", x=-0.13, y=1.04)

    # --- Panel C: f_charge continuous ---
    hb_c = ax_c.hexbin(
        zoom_df["logQ"],
        zoom_df["logInvC"],
        C=zoom_df["f_charge_local"],
        reduce_C_function=np.mean,
        gridsize=24,
        cmap=DOMINANCE_CMAP,
        norm=fc_norm,
        mincnt=1,
        linewidths=0.35,
        edgecolors="white",
    )
    cb_c = fig.colorbar(hb_c, ax=ax_c, pad=0.02, fraction=0.046)
    style_colorbar(cb_c, r"Mean local $f_{\mathrm{charge}}$")
    ax_c.set_title(
        "Mechanism-dominance map",
        loc="left",
        fontweight="bold",
        fontsize=8,
        pad=FIG_TITLE_PAD,
    )
    ax_c.set_xlabel(r"$\log_{10}(Q_{sc,MACRS}^{2})$", fontsize=8)
    ax_c.set_ylabel(r"$\log_{10}(1/C_{start}+1/C_{end})$", fontsize=8)
    # Place labels outside data clusters with semi-transparent backgrounds
    ax_c.text(
        0.03,
        0.97,
        "charge-\nleaning",
        transform=ax_c.transAxes,
        fontsize=FIG_CALLOUT_SMALL_SIZE,
        color=REGIME_COLORS[REGIME_CHARGE],
        fontstyle="italic",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75
        ),
    )
    ax_c.text(
        0.97,
        0.03,
        "capacitance-\nleaning",
        transform=ax_c.transAxes,
        fontsize=FIG_CALLOUT_SMALL_SIZE,
        color=REGIME_COLORS[REGIME_CAPACITANCE],
        fontstyle="italic",
        va="bottom",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75
        ),
    )
    style_mechanism_axes(ax_c)
    add_panel_label(ax_c, "C", x=-0.13, y=1.04)

    ax_d.set_facecolor(BRIDGE["bg"])
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")
    ax_d.set_title(
        "Design-space translation",
        loc="left",
        fontweight="bold",
        fontsize=8,
        pad=FIG_TITLE_PAD,
    )
    add_panel_label(ax_d, "D", x=-0.13, y=1.04)
    ax_d.text(
        0.00,
        0.94,
        "schematic regions (left) → parameter trends (right)",
        transform=ax_d.transAxes,
        fontsize=FIG_CALLOUT_SMALL_SIZE,
        color=BRIDGE["muted"],
        va="top",
    )

    sketch = FancyBboxPatch(
        (0.02, 0.13),
        0.47,
        0.72,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=0.9,
        edgecolor="#CBD5E1",
        facecolor="white",
        transform=ax_d.transAxes,
    )
    ax_d.add_patch(sketch)
    ax_d.text(
        0.05,
        0.82,
        "Mechanism-space sketch",
        transform=ax_d.transAxes,
        fontsize=FIG_CALLOUT_SIZE,
        fontweight="bold",
        color=BRIDGE["text"],
    )
    ax_d.add_patch(
        FancyBboxPatch(
            (0.06, 0.18),
            0.35,
            0.56,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            linewidth=0.0,
            facecolor="#EEF2F7",
            transform=ax_d.transAxes,
        )
    )
    ax_d.add_patch(
        FancyBboxPatch(
            (0.355, 0.22),
            0.065,
            0.46,
            boxstyle="round,pad=0.002,rounding_size=0.01",
            linewidth=0.0,
            facecolor="#C9E3F5",
            transform=ax_d.transAxes,
        )
    )
    ax_d.add_patch(
        FancyBboxPatch(
            (0.08, 0.61),
            0.23,
            0.08,
            boxstyle="round,pad=0.002,rounding_size=0.01",
            linewidth=0.0,
            facecolor="#F7C9C1",
            transform=ax_d.transAxes,
        )
    )
    ax_d.add_patch(
        FancyBboxPatch(
            (0.08, 0.22),
            0.16,
            0.07,
            boxstyle="round,pad=0.002,rounding_size=0.01",
            linewidth=0.0,
            facecolor="#F7C9C1",
            transform=ax_d.transAxes,
        )
    )
    ax_d.text(
        0.405,
        0.695,
        "blue edge",
        transform=ax_d.transAxes,
        fontsize=6.2,
        color=BRIDGE["cap"],
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.75
        ),
    )
    ax_d.text(
        0.215,
        0.47,
        "gray core",
        transform=ax_d.transAxes,
        fontsize=6.5,
        color=BRIDGE["muted"],
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.72
        ),
    )
    ax_d.text(
        0.155,
        0.695,
        "red zone",
        transform=ax_d.transAxes,
        fontsize=6.2,
        color=BRIDGE["charge"],
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.75
        ),
    )
    ax_d.text(
        0.135,
        0.305,
        "red zone",
        transform=ax_d.transAxes,
        fontsize=6.5,
        color=BRIDGE["charge"],
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.75
        ),
    )
    ax_d.text(
        0.175,
        0.65,
        "A",
        transform=ax_d.transAxes,
        fontsize=6.5,
        fontweight="bold",
        color=BRIDGE["charge"],
        ha="center",
        va="center",
    )
    ax_d.text(
        0.135,
        0.255,
        "B",
        transform=ax_d.transAxes,
        fontsize=6.5,
        fontweight="bold",
        color=BRIDGE["charge"],
        ha="center",
        va="center",
    )

    add_bridge_card(
        ax_d,
        0.54,
        0.635,
        0.44,
        0.225,
        BRIDGE["cap"],
        "Blue edge",
        [r"low $d/R$", r"low $h/R$", r"low $n$", "capacitance-led"],
        highlight_last=True,
    )
    add_bridge_card(
        ax_d,
        0.54,
        0.385,
        0.44,
        0.225,
        BRIDGE["mixed"],
        "Gray core",
        [r"moderate $n$, $d/R$", r"low–mid $h/R$", "co-varying channels"],
    )
    add_bridge_card(
        ax_d,
        0.54,
        0.135,
        0.44,
        0.225,
        BRIDGE["charge"],
        "Red zones",
        [
            r"A: low $n$, high $h/R$",
            r"B: high $n$, high $d/R$, low $h/R$",
            "charge-led",
        ],
        highlight_last=True,
    )

    arrows = [
        ((0.42, 0.52), (0.54, 0.76), BRIDGE["cap"], "arc3,rad=-0.08"),
        ((0.31, 0.47), (0.54, 0.52), "#7C8797", "arc3,rad=0.0"),
        ((0.24, 0.63), (0.54, 0.33), BRIDGE["charge"], "arc3,rad=-0.10"),
        ((0.21, 0.255), (0.54, 0.25), BRIDGE["charge"], "arc3,rad=0.10"),
    ]
    for start, end, color, cs in arrows:
        ax_d.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax_d.transAxes,
                arrowstyle="->",
                mutation_scale=8,
                linewidth=1.0,
                connectionstyle=cs,
                color=color,
            )
        )

    ax_d.text(
        0.50,
        0.09,
        "schematic only; explicit windows in Fig. 3",
        transform=ax_d.transAxes,
        fontsize=FIG_CALLOUT_SMALL_SIZE,
        color=BRIDGE["muted"],
        va="center",
    )

    fig.subplots_adjust(left=0.085, right=0.972, top=0.93, bottom=0.085)
    main_files, preview_file = save_figure(fig, "fig02_mechanism_landscape")
    plt.close(fig)
    return {"main_files": main_files, "preview_file": preview_file}


def normalize_log_series(values):
    log_values = np.log10(np.maximum(values, 1e-30))
    vmin = np.nanmin(log_values)
    vmax = np.nanmax(log_values)
    span = vmax - vmin
    if span < 1e-12:
        return np.full_like(log_values, 0.5)
    return (log_values - vmin) / span


def select_phase_point(result):
    foms = result["foms"]
    f_charge = result["f_charge"]
    top30 = foms >= np.nanpercentile(foms, 70)
    norm_fom = normalize_map(foms, top30)
    balance_penalty = np.abs(f_charge - 0.5) / 0.5
    score = np.full_like(foms, -np.inf, dtype=float)
    score[top30] = norm_fom[top30] - balance_penalty[top30]
    idx = pick_best_index(score, top30, prefer_max=True)
    if idx is None:
        idx = np.unravel_index(np.nanargmax(foms), foms.shape)
    row, col = idx
    return {
        "row": int(row),
        "col": int(col),
        "n": float(N_VALUES[col]),
        "hh": float(HH_VALUES[row]),
        "foms": float(foms[row, col]),
        "f_charge": float(f_charge[row, col]),
        "regime": float(result["regime"][row, col]),
    }


def select_transition_row(design):
    f_map = design["f_charge"]
    log_foms = np.log10(np.maximum(design["foms"], 1e-30))
    global_min = float(np.nanmin(log_foms))
    global_max = float(np.nanmax(log_foms))
    best_row = None
    best_score = -np.inf

    for row in range(f_map.shape[0]):
        f_row = np.asarray(f_map[row, :], dtype=float)
        f_min = float(np.nanmin(f_row))
        f_max = float(np.nanmax(f_row))
        if not (f_min < 0.5 < f_max):
            continue

        span = f_max - f_min
        diff = np.diff(f_row)
        valid_diff = diff[np.abs(diff) > 1e-6]
        if valid_diff.size >= 2:
            sign_changes = int(
                np.sum(np.sign(valid_diff[1:]) * np.sign(valid_diff[:-1]) < 0)
            )
        else:
            sign_changes = 0

        cross_idx = None
        for idx in range(len(f_row) - 1):
            left = f_row[idx] - 0.5
            right = f_row[idx + 1] - 0.5
            if left == 0.0 or left * right < 0 or right == 0.0:
                cross_idx = idx
                break
        if cross_idx is None:
            continue

        center = 0.5 * (len(f_row) - 2)
        edge_penalty = abs(cross_idx - center) / max(center, 1.0)
        row_perf = float(np.nanmedian(log_foms[row, :]))
        perf_norm = (row_perf - global_min) / (global_max - global_min + 1e-12)
        score = (
            1.00 * span + 0.25 * perf_norm - 0.25 * sign_changes - 0.20 * edge_penalty
        )
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is not None:
        return int(best_row)

    fallback_row = None
    fallback_score = -np.inf
    for row in range(f_map.shape[0]):
        f_row = np.asarray(f_map[row, :], dtype=float)
        span = float(np.nanmax(f_row) - np.nanmin(f_row))
        dist = float(np.nanmin(np.abs(f_row - 0.5)))
        row_perf = float(np.nanmedian(log_foms[row, :]))
        perf_norm = (row_perf - global_min) / (global_max - global_min + 1e-12)
        score = span - dist + 0.15 * perf_norm
        if score > fallback_score:
            fallback_score = score
            fallback_row = row
    return int(
        fallback_row
        if fallback_row is not None
        else np.nanargmax(np.nanstd(f_map, axis=1))
    )


def select_safe_mask(foms_map, cv_map, worst_ratio_map):
    return (
        (foms_map >= np.nanpercentile(foms_map, 70))
        & (cv_map <= SAFE_CV_THRESHOLD)
        & (worst_ratio_map >= SAFE_WORST_RATIO_THRESHOLD)
    )


def build_refined_safe_window(mask, *, sigma=0.75, refine=8):
    smooth_mask = gaussian_filter(mask.astype(float), sigma=sigma)
    refined_mask = nd_zoom(smooth_mask, zoom=refine, order=3)
    x_fine = np.linspace(
        np.log2(N_VALUES[0]), np.log2(N_VALUES[-1]), refined_mask.shape[1]
    )
    y_fine = np.linspace(
        np.log2(HH_VALUES[0]), np.log2(HH_VALUES[-1]), refined_mask.shape[0]
    )
    return x_fine, y_fine, refined_mask


def _iter_true_runs(mask_1d):
    start = None
    for idx, flag in enumerate(mask_1d):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, len(mask_1d)


def draw_safe_window_boundary_closure(
    ax, x_vals, y_vals, refined_mask, *, level, color, linewidth, alpha
):
    edge_specs = [
        ("left", refined_mask[:, 0] >= level),
        ("right", refined_mask[:, -1] >= level),
        ("bottom", refined_mask[0, :] >= level),
        ("top", refined_mask[-1, :] >= level),
    ]
    for edge_name, edge_mask in edge_specs:
        for start, end in _iter_true_runs(edge_mask):
            if edge_name == "left":
                ax.plot(
                    np.full(end - start, x_vals[0]),
                    y_vals[start:end],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=6.2,
                    solid_capstyle="round",
                )
            elif edge_name == "right":
                ax.plot(
                    np.full(end - start, x_vals[-1]),
                    y_vals[start:end],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=6.2,
                    solid_capstyle="round",
                )
            elif edge_name == "bottom":
                ax.plot(
                    x_vals[start:end],
                    np.full(end - start, y_vals[0]),
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=6.2,
                    solid_capstyle="round",
                )
            else:
                ax.plot(
                    x_vals[start:end],
                    np.full(end - start, y_vals[-1]),
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=6.2,
                    solid_capstyle="round",
                )


def select_robust_point(foms_map, cv_map, worst_ratio_map):
    top30 = foms_map >= np.nanpercentile(foms_map, 70)
    norm_fom = normalize_map(np.log10(np.maximum(foms_map, 1e-30)), top30)
    norm_cv = normalize_map(cv_map, top30)
    norm_ret = normalize_map(worst_ratio_map, top30)
    score = np.full_like(foms_map, -np.inf, dtype=float)
    score[top30] = norm_fom[top30] - 0.45 * norm_cv[top30] + 0.20 * norm_ret[top30]
    idx = pick_best_index(score, top30, prefer_max=True)
    if idx is None:
        idx = np.unravel_index(np.nanargmin(cv_map), cv_map.shape)
    row, col = idx
    return {
        "row": int(row),
        "col": int(col),
        "n": float(N_VALUES[col]),
        "hh": float(HH_VALUES[row]),
        "foms": float(foms_map[row, col]),
        "cv": float(cv_map[row, col]),
        "worst_ratio": float(worst_ratio_map[row, col]),
        "score": float(score[row, col]) if np.isfinite(score[row, col]) else np.nan,
    }


def segment_bounds(values):
    ratios = values[1:] / values[:-1]
    left = values[0] / np.sqrt(ratios[0])
    mids = np.sqrt(values[:-1] * values[1:])
    right = values[-1] * np.sqrt(ratios[-1])
    return np.concatenate([[left], mids, [right]])


def draw_fcharge_background(ax, f_charge_row):
    bounds = segment_bounds(N_VALUES)
    for i, f_value in enumerate(np.asarray(f_charge_row, dtype=float)):
        if f_value < 0.4:
            color = REGIME_COLORS[REGIME_CAPACITANCE]
        elif f_value > 0.6:
            color = REGIME_COLORS[REGIME_CHARGE]
        else:
            color = REGIME_COLORS[REGIME_MIXED]
        ax.axvspan(bounds[i], bounds[i + 1], color=color, alpha=0.12, lw=0, zorder=0)


def add_curve_label(ax, x, y, text, color, *, fontsize=FIG_CALLOUT_SMALL_SIZE):
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=fontsize,
        ha="center",
        va="center",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9)],
        zorder=6,
    )


def build_design_rows(predict_fn, scenarios):
    rows = []
    for scenario in scenarios:
        design = compute_design_regime_grid(
            predict_fn,
            E_fixed=scenario["E"],
            dd_fixed=scenario["dd"],
            n_values=N_VALUES,
            hh_values=HH_VALUES,
            delta_log=DESIGN_DELTA_LOG,
            dominance_threshold=DESIGN_DOMINANCE_THRESHOLD,
        )
        rows.append(
            {
                "scenario": scenario,
                "design": design,
                "phase_point": select_phase_point(design),
            }
        )
    return rows


def build_ashby_frame(rows):
    ashby_rows = []
    for row in rows:
        scenario = row["scenario"]
        design = row["design"]
        robust = row["robust"]
        safe_mask = row["safe_mask"]
        for i_h, hh in enumerate(HH_VALUES):
            for i_n, n in enumerate(N_VALUES):
                ashby_rows.append(
                    {
                        "scenario": scenario["label"],
                        "scenario_title": scenario["title"],
                        "n": float(n),
                        "hh": float(hh),
                        "foms": float(design["foms"][i_h, i_n]),
                        "log10_foms": float(
                            np.log10(max(design["foms"][i_h, i_n], 1e-30))
                        ),
                        "cv_pct": float(robust["cv_map"][i_h, i_n] * 100.0),
                        "worst_ratio_pct": float(
                            robust["worst_ratio_map"][i_h, i_n] * 100.0
                        ),
                        "regime": int(design["regime"][i_h, i_n]),
                        "safe_zone": bool(safe_mask[i_h, i_n]),
                    }
                )
    return pd.DataFrame(ashby_rows)


def select_recommended_point(df):
    candidates = df.copy()
    perf_norm = (candidates["log10_foms"] - candidates["log10_foms"].min()) / (
        candidates["log10_foms"].max() - candidates["log10_foms"].min() + 1e-12
    )
    cv_norm = (candidates["cv_pct"] - candidates["cv_pct"].min()) / (
        candidates["cv_pct"].max() - candidates["cv_pct"].min() + 1e-12
    )
    candidates["value_score"] = perf_norm - 0.55 * cv_norm
    return candidates.sort_values("value_score", ascending=False).iloc[0]


def _draw_ashby_panel(ax, ashby_df, safe_df, core_recommend, fom_threshold):
    ax.axvline(
        SAFE_CV_THRESHOLD * 100.0,
        color=CRITERION_LINE_COLOR,
        lw=CRITERION_LINE_WIDTH,
        ls=SHORT_DASH,
        alpha=CRITERION_LINE_ALPHA,
        zorder=2,
    )
    ax.axhline(
        fom_threshold,
        color=CRITERION_LINE_COLOR,
        lw=CRITERION_LINE_WIDTH,
        ls=SHORT_DASH,
        alpha=CRITERION_LINE_ALPHA,
        zorder=2,
    )
    ax.text(
        14.75,
        fom_threshold + 0.018,
        r"top-30% global ($FOM_S$) criterion",
        fontsize=FIG_CALLOUT_SMALL_SIZE,
        color="#5F6B76",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.72),
        zorder=5,
    )
    ax.text(
        SAFE_CV_THRESHOLD * 100.0 + 0.14,
        -2.38,
        "CV = 5% criterion",
        fontsize=FIG_CALLOUT_SMALL_SIZE,
        color="#5F6B76",
        ha="left",
        va="bottom",
        rotation=90,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.72),
        zorder=5,
    )
    ax.scatter(
        ashby_df["cv_pct"],
        ashby_df["log10_foms"],
        s=12,
        color=ASHBY_ALL_FACE,
        alpha=ASHBY_ALL_ALPHA,
        linewidths=0,
        zorder=1,
    )
    if not safe_df.empty:
        ax.scatter(
            safe_df["cv_pct"],
            safe_df["log10_foms"],
            s=30,
            marker="o",
            facecolors=SAFE_CANDIDATE_FACE,
            edgecolors=SAFE_CANDIDATE_EDGE,
            linewidths=0.35,
            alpha=SAFE_CANDIDATE_ALPHA,
            zorder=4,
        )
    if core_recommend is not None:
        ax.scatter(
            core_recommend["cv_pct"],
            core_recommend["log10_foms"],
            s=REPRESENTATIVE_SIZE,
            marker="o",
            facecolor=REPRESENTATIVE_FACE,
            edgecolor=REPRESENTATIVE_EDGE,
            linewidth=REPRESENTATIVE_EDGEWIDTH,
            zorder=9,
        )

    hide_extra_spines(ax)
    ax.tick_params(axis="y", pad=1.8)
    ax.set_xlabel("CV (%)")
    ax.set_ylabel(r"$\log_{10}(FOM_S)$")
    ax.yaxis.set_label_coords(-0.085, 0.5)
    ax.set_title(
        "Global decision map",
        loc="left",
        fontweight="bold",
        pad=5,
        fontsize=8,
    )
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(-2.5, -1.15)


def _build_ashby_legend():
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ASHBY_ALL_FACE,
            markeredgecolor="none",
            markersize=4.6,
            alpha=ASHBY_ALL_ALPHA,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SAFE_CANDIDATE_FACE,
            markeredgecolor=SAFE_CANDIDATE_EDGE,
            markeredgewidth=0.35,
            markersize=5.0,
            alpha=SAFE_CANDIDATE_ALPHA,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=REPRESENTATIVE_FACE,
            markeredgecolor=REPRESENTATIVE_EDGE,
            markeredgewidth=REPRESENTATIVE_EDGEWIDTH,
            markersize=REPRESENTATIVE_MARKERSIZE,
        ),
    ]
    labels = [
        "all pooled designs",
        "safe candidates",
        "representative candidate",
    ]
    return handles, labels


def build_fig3(predict_fn):
    rows = build_design_rows(predict_fn, FIG3_SCENARIOS)
    n_grid, hh_grid = log2_mesh()
    all_logfoms = np.concatenate(
        [np.log10(np.maximum(row["design"]["foms"], 1e-30)).ravel() for row in rows]
    )
    fmin = np.nanpercentile(all_logfoms, 2)
    fmax = np.nanpercentile(all_logfoms, 98)

    fig = plt.figure(figsize=(11.55, 6.0))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 0.95, 1.06],
        hspace=0.28,
        wspace=0.42,
        left=0.072,
        right=0.972,
        bottom=0.12,
        top=0.84,
    )

    label_ord = ord("a")
    top_axes = []
    bottom_axes = []
    foms_im = None
    fcharge_im = None
    export_rows = []
    mid_row = next(row for row in rows if row["scenario"]["label"] == "mid_E_mid_d")
    mid_slice_row = select_transition_row(mid_row["design"])
    mid_slice_hh = float(HH_VALUES[mid_slice_row])

    for col_idx, row in enumerate(rows):
        scenario = row["scenario"]
        design = row["design"]
        rep = row["phase_point"]

        ax_hm = fig.add_subplot(gs[0, col_idx])
        ax_fc = fig.add_subplot(gs[1, col_idx], sharex=ax_hm, sharey=ax_hm)
        top_axes.append(ax_hm)
        bottom_axes.append(ax_fc)

        log_foms = np.log10(np.maximum(design["foms"], 1e-30))
        foms_im = ax_hm.pcolormesh(
            n_grid,
            hh_grid,
            log_foms,
            cmap="viridis",
            shading="nearest",
            vmin=fmin,
            vmax=fmax,
        )
        if scenario["label"] == "mid_E_mid_d":
            ax_hm.axhline(
                np.log2(mid_slice_hh),
                color="white",
                lw=1.1,
                ls=SHORT_DASH,
                zorder=7,
            )
            ax_hm.scatter(
                np.log2(rep["n"]),
                np.log2(rep["hh"]),
                s=38,
                facecolor="white",
                edgecolor="#111827",
                linewidth=0.9,
                zorder=8,
            )
        ax_hm.set_title(
            scenario["title"],
            loc="left",
            fontweight="bold",
            pad=FIG_TITLE_PAD,
            fontsize=FIG_TITLE_SIZE,
        )
        configure_design_axes(ax_hm, show_xlabel=False, show_ylabel=(col_idx == 0))
        add_panel_label(ax_hm, chr(label_ord), x=-0.15, y=1.05)
        label_ord += 1

        fcharge_im = ax_fc.pcolormesh(
            n_grid,
            hh_grid,
            design["f_charge"],
            cmap=DOMINANCE_CMAP,
            shading="nearest",
            norm=mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
        )
        ax_fc.contour(
            n_grid,
            hh_grid,
            design["f_charge"],
            levels=[0.5],
            colors=["#FFFFFF"],
            linewidths=1.05,
            linestyles=[SHORT_DASH],
            alpha=0.98,
        )
        if scenario["label"] == "mid_E_mid_d":
            ax_fc.axhline(
                np.log2(mid_slice_hh),
                color="#111827",
                lw=1.0,
                ls=SHORT_DASH,
                zorder=7,
            )
        configure_design_axes(ax_fc, show_xlabel=True, show_ylabel=(col_idx == 0))
        add_panel_label(ax_fc, chr(label_ord), x=-0.15, y=1.05)
        label_ord += 1

        export_rows.append(
            {
                "scenario": scenario["label"],
                "phase_n": rep["n"],
                "phase_hh": rep["hh"],
                "phase_foms": rep["foms"],
                "phase_f_charge": (
                    rep["f_charge"] if scenario["label"] == "mid_E_mid_d" else np.nan
                ),
                "mid_slice_hh": mid_slice_hh,
            }
        )

    ax_slice = fig.add_subplot(gs[:, 3])
    mid_design = mid_row["design"]
    mid_slice_regime = mid_design["regime"][mid_slice_row, :]
    q2_norm = normalize_log_series(mid_design["qsc"][mid_slice_row, :] ** 2)
    invc_norm = normalize_log_series(mid_design["invc"][mid_slice_row, :])
    foms_norm = normalize_log_series(mid_design["foms"][mid_slice_row, :])
    draw_fcharge_background(ax_slice, mid_design["f_charge"][mid_slice_row, :])
    ax_slice.plot(
        N_VALUES,
        q2_norm,
        color=LINE_COLORS["q2"],
        lw=1.8,
        ls="-",
        marker="o",
        ms=4,
    )
    ax_slice.plot(
        N_VALUES,
        invc_norm,
        color=LINE_COLORS["invc"],
        lw=1.8,
        ls="--",
        marker="s",
        ms=3.8,
    )
    ax_slice.plot(
        N_VALUES,
        foms_norm,
        color=LINE_COLORS["foms"],
        lw=2.0,
        ls="-.",
        marker="D",
        ms=3.8,
    )
    ax_slice.set_xscale("log", base=2)
    ax_slice.set_xticks(N_VALUES)
    ax_slice.set_xticklabels([str(int(v)) for v in N_VALUES])
    ax_slice.set_xlim(N_VALUES[0] / 1.3, N_VALUES[-1] * 1.3)
    ax_slice.set_ylim(-0.02, 1.14)
    ax_slice.set_xlabel(r"$n$ (log scale)", labelpad=1.5)
    ax_slice.set_ylabel("normalized value", labelpad=0.6)
    ax_slice.yaxis.set_label_coords(-0.14, 0.5)
    ax_slice.tick_params(length=2.5, pad=2, labelsize=FIG_TICK_LABEL_SIZE)
    hide_extra_spines(ax_slice)
    ax_slice.text(
        0.03,
        0.04,
        rf"slice at $h/R={mid_slice_hh:.3g}$",
        transform=ax_slice.transAxes,
        fontsize=FIG_CALLOUT_SIZE,
        color="#475467",
    )
    ax_slice.set_title(
        "mechanism transition along n",
        loc="left",
        fontweight="bold",
        pad=FIG_TITLE_PAD,
        fontsize=FIG_TITLE_SIZE,
    )
    for region_label in FIG3G_REGION_LABELS:
        ax_slice.text(
            region_label["x"],
            region_label["y"],
            region_label["text"],
            transform=ax_slice.transData,
            fontsize=FIG_CALLOUT_SMALL_SIZE,
            color=region_label["color"],
            ha=region_label["ha"],
            va="bottom",
        )
    add_curve_label(ax_slice, 14.5, 0.75, r"$Q_{sc}^{2}$", LINE_COLORS["q2"])
    add_curve_label(ax_slice, 6.3, 0.43, r"$1/C_{\mathrm{sum}}$", LINE_COLORS["invc"])
    add_curve_label(ax_slice, 24, 0.92, r"$FOM_S$", LINE_COLORS["foms"])
    add_panel_label(ax_slice, chr(label_ord), x=-0.11, y=1.05)

    for n_value, q2_v, invc_v, foms_v, regime_v in zip(
        N_VALUES, q2_norm, invc_norm, foms_norm, mid_slice_regime
    ):
        export_rows.append(
            {
                "scenario": "mid_E_mid_d",
                "slice_n": float(n_value),
                "slice_hh": mid_slice_hh,
                "q2_norm": float(q2_v),
                "invc_norm": float(invc_v),
                "foms_norm": float(foms_v),
                "slice_regime": int(regime_v),
            }
        )

    cbar1 = fig.colorbar(foms_im, ax=top_axes, pad=0.012, fraction=0.021, aspect=26)
    style_colorbar(cbar1, r"$\log_{10}(FOM_S)$", labelpad=1)
    cbar2 = fig.colorbar(
        fcharge_im, ax=bottom_axes, pad=0.012, fraction=0.021, aspect=26
    )
    style_colorbar(cbar2, r"$f_{\mathrm{charge}}$", labelpad=1)

    fig.suptitle(
        "Design landscapes and mechanism transition across ε",
        y=0.965,
        fontsize=10,
        fontweight="bold",
    )

    csv_path = SRC_DIR / "fig03_t11_three_scene_slice_data.csv"
    pd.DataFrame(export_rows).to_csv(csv_path, index=False)
    main_files, preview_file = save_figure(
        fig, "fig03_design_landscapes_mechanism_transition"
    )
    plt.close(fig)
    return {
        "main_files": main_files,
        "preview_file": preview_file,
        "csv": rel_path(csv_path),
    }


def build_fig4(predict_fn):
    ashby_rows_source = build_design_rows(predict_fn, ASHBY_SCENARIOS)
    for row in ashby_rows_source:
        foms_map, cv_map, worst_ratio_map = compute_robustness_grid(
            predict_fn,
            E_fixed=row["scenario"]["E"],
            dd_fixed=row["scenario"]["dd"],
            n_values=N_VALUES,
            hh_values=HH_VALUES,
            perturb_frac=PERTURB_FRAC,
        )
        row["robust"] = {
            "foms_map": foms_map,
            "cv_map": cv_map,
            "worst_ratio_map": worst_ratio_map,
            "recommended": select_robust_point(foms_map, cv_map, worst_ratio_map),
        }
        row["safe_mask"] = select_safe_mask(foms_map, cv_map, worst_ratio_map)

    rows = [
        row
        for row in ashby_rows_source
        if row["scenario"]["label"] in {s["label"] for s in FIG4_HEATMAP_SCENARIOS}
    ]

    n_grid, hh_grid = log2_mesh()
    fig = plt.figure(figsize=(11.4, 4.95))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.38],
        wspace=0.34,
        left=0.068,
        right=0.972,
        bottom=0.32,
        top=0.80,
    )

    ashby_df = build_ashby_frame(ashby_rows_source)
    fom_threshold = float(np.nanpercentile(ashby_df["log10_foms"], 70))
    safe_df = ashby_df[
        (ashby_df["cv_pct"] <= SAFE_CV_THRESHOLD * 100.0)
        & (ashby_df["worst_ratio_pct"] >= SAFE_WORST_RATIO_THRESHOLD * 100.0)
        & (ashby_df["log10_foms"] >= fom_threshold)
    ].copy()
    candidate_df = safe_df if not safe_df.empty else ashby_df
    core_recommend = select_recommended_point(candidate_df)

    cv_all = np.concatenate([row["robust"]["cv_map"].ravel() * 100.0 for row in rows])
    cv_vmax = np.nanpercentile(cv_all, 95)

    label_ord = ord("a")
    cv_im = None
    heatmap_axes = []
    for idx, row in enumerate(rows):
        scenario = row["scenario"]
        design = row["design"]
        robust = row["robust"]
        ax = fig.add_subplot(gs[0, idx])
        heatmap_axes.append(ax)
        cv_im = ax.pcolormesh(
            n_grid,
            hh_grid,
            robust["cv_map"] * 100.0,
            cmap=FIG4_CV_CMAP,
            shading="nearest",
            vmin=0.0,
            vmax=cv_vmax,
        )
        if np.any(row["safe_mask"]):
            x_safe, y_safe, smooth_mask = build_refined_safe_window(
                row["safe_mask"], sigma=0.75, refine=8
            )
            ax.contourf(
                x_safe,
                y_safe,
                smooth_mask,
                levels=[0.5, 1.1],
                colors=[SAFE_GREEN],
                alpha=SAFE_ZONE_ALPHA,
                zorder=5,
            )
            ax.contour(
                x_safe,
                y_safe,
                smooth_mask,
                levels=[0.5],
                colors=[SAFE_ZONE_EDGE],
                linewidths=1.0,
                linestyles=["-"],
                alpha=SAFE_ZONE_EDGE_ALPHA,
                zorder=6,
            )
            draw_safe_window_boundary_closure(
                ax,
                x_safe,
                y_safe,
                smooth_mask,
                level=0.5,
                color=SAFE_ZONE_EDGE,
                linewidth=1.0,
                alpha=SAFE_ZONE_EDGE_ALPHA,
            )
        ax.set_title(
            scenario["title"],
            loc="left",
            fontweight="bold",
            pad=FIG_TITLE_PAD,
            fontsize=8,
        )
        configure_design_axes(ax, show_xlabel=False, show_ylabel=(idx == 0))
        add_panel_label(ax, chr(label_ord), x=-0.08, y=1.05)
        label_ord += 1

    ax_ashby = fig.add_subplot(gs[0, 2])
    _draw_ashby_panel(ax_ashby, ashby_df, safe_df, core_recommend, fom_threshold)
    add_panel_label(ax_ashby, chr(label_ord), x=-0.10, y=1.04)

    legend_handles, legend_labels = _build_ashby_legend()
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.055),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=FIG_CALLOUT_SMALL_SIZE,
        columnspacing=1.0,
        handlelength=1.1,
    )

    heatmap_left = heatmap_axes[0].get_position().x0
    heatmap_right = heatmap_axes[-1].get_position().x1
    heatmap_bottom = min(ax.get_position().y0 for ax in heatmap_axes)
    cax = fig.add_axes(
        [heatmap_left, heatmap_bottom - 0.075, heatmap_right - heatmap_left, 0.022]
    )
    cbar = fig.colorbar(cv_im, cax=cax, orientation="horizontal")
    style_colorbar(cbar, "CV (%)", fontsize=7.3, labelpad=1.5)
    cbar.ax.tick_params(pad=1.5)

    fig.suptitle(
        "Robust manufacturable windows and performance–variability trade-off",
        y=0.94,
        fontsize=10,
        fontweight="bold",
    )

    csv_path = SRC_DIR / "fig04_t11_ashby_data.csv"
    ashby_df.to_csv(csv_path, index=False)
    main_files, preview_file = save_figure(fig, "fig04_robust_design_map")
    plt.close(fig)
    return {
        "main_files": main_files,
        "preview_file": preview_file,
        "csv": rel_path(csv_path),
        "recommend": core_recommend.to_dict(),
    }


def build_fig5(model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device):
    from predict_multitask_physics import compute_combined_ood_metrics, validate_ood

    validate_inputs = [
        ("validate1", DATA_DIR / "validate_foms_macrs.csv"),
        ("validate2", DATA_DIR / "validate2_foms_macrs.csv"),
        ("validate3", DATA_DIR / "validate3_foms_macrs.csv"),
    ]

    results = {}
    all_rows = []
    for name, csv_path in validate_inputs:
        result_df, _ = validate_ood(
            str(csv_path),
            model=model,
            scaler_X=scaler_X,
            scaler_qsc=scaler_qsc,
            scaler_invc=scaler_invc,
            scaler_foms=scaler_foms,
            device=device,
        )
        result_df = result_df.copy()
        result_df["dataset"] = name
        results[name] = result_df
        all_rows.append(result_df)

    combined_df = pd.concat(all_rows, ignore_index=True)
    combined_df.to_csv(SRC_DIR / "fig05_validation_points.csv", index=False)
    metrics = compute_combined_ood_metrics(
        [results["validate1"], results["validate2"], results["validate3"]]
    )
    (SRC_DIR / "fig05_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.18, wspace=0.32)
    targets = [
        ("$Q_{sc}$", "Qsc_true", "Qsc_pred", "qsc"),
        ("$1/C_{sum}$", "invC_true", "invC_pred", "invc"),
        ("$FOM_S$", "FOMS_direct_true", "FOMS_direct_pred", "foms_direct"),
    ]
    dataset_labels = {"validate1": "V1", "validate2": "V2", "validate3": "V3"}

    handles = []
    for ax, (title, true_col, pred_col, metric_key), label_char in zip(
        axes, targets, ["a", "b", "c"]
    ):
        for idx, (dataset, df) in enumerate(results.items()):
            h = ax.scatter(
                df[true_col],
                df[pred_col],
                s=24,
                marker=["o", "s", "D"][idx],
                color=OKABE_ITO[idx],
                edgecolors="#111827",
                linewidths=0.35,
                alpha=0.85,
                label=dataset_labels[dataset],
            )
            if len(handles) < 3:
                handles.append(h)

        all_true = combined_df[true_col].to_numpy()
        all_pred = combined_df[pred_col].to_numpy()
        pos = (all_true > 0) & (all_pred > 0)
        lo = min(all_true[pos].min(), all_pred[pos].min()) * 0.80
        hi = max(all_true[pos].max(), all_pred[pos].max()) * 1.20
        ax.plot(
            [lo, hi], [lo, hi], linestyle=SHORT_DASH, color="#667085", linewidth=0.9
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"True {title}", fontsize=FIG_AXIS_LABEL_SIZE)
        ax.set_ylabel(f"Predicted {title}", fontsize=FIG_AXIS_LABEL_SIZE)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        m = metrics[metric_key]
        ax.text(
            0.05,
            0.95,
            f"$R^2_{{\\log10}}$ = {m['r2_log10']:.3f}\nMAPE = {m['mape']:.1f}%\nN = {metrics['n_points']}",
            transform=ax.transAxes,
            fontsize=FIG_CALLOUT_SMALL_SIZE,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="#D0D5DD",
                alpha=0.96,
            ),
        )
        hide_extra_spines(ax)
        add_panel_label(ax, label_char, x=-0.16, y=1.10)

    fig.legend(
        handles=handles,
        labels=["V1", "V2", "V3"],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, 0.97),
        fontsize=7.5,
        columnspacing=2.0,
    )
    main_files, preview_file = save_figure(
        fig, "fig05_unseen_structural_dielectric_validation"
    )
    plt.close(fig)
    return {"main_files": main_files, "preview_file": preview_file}


def write_table_csv(df: pd.DataFrame, filename: str):
    path = SRC_DIR / filename
    df.to_csv(path, index=False)
    return rel_path(path)


def export_main_table_sources():
    """Export main-text table source data for LaTeX rendering."""
    test_metrics = json.loads(TEST_METRICS.read_text(encoding="utf-8"))
    cv_results = json.loads(CV_RESULTS.read_text(encoding="utf-8"))
    fig05_metrics_path = SRC_DIR / "fig05_validation_metrics.json"
    if not fig05_metrics_path.exists():
        raise FileNotFoundError(
            "Missing fig05_validation_metrics.json. Build Fig.5 before exporting main tables."
        )
    fig05_metrics = json.loads(fig05_metrics_path.read_text(encoding="utf-8"))

    table1_rows = [
        {
            "validation_dimension": "Test",
            "qsc_r2_log10": test_metrics["qsc"]["r2_log10"],
            "qsc_r2_log10_std": "",
            "invc_r2_log10": test_metrics["invc"]["r2_log10"],
            "invc_r2_log10_std": "",
            "foms_r2_log10": test_metrics["foms_direct"]["r2_log10"],
            "foms_r2_log10_std": "",
            "consistency_pearson_r": test_metrics["consistency"]["pearson_r"],
            "consistency_pearson_r_std": "",
            "sample_size": test_metrics["_metadata"].get("n_test", ""),
            "role": "In-distribution fitting capability",
        },
        {
            "validation_dimension": "5-fold CV",
            "qsc_r2_log10": cv_results["val_aggregated"]["qsc"]["r2_log10"]["mean"],
            "qsc_r2_log10_std": cv_results["val_aggregated"]["qsc"]["r2_log10"]["std"],
            "invc_r2_log10": cv_results["val_aggregated"]["invc"]["r2_log10"]["mean"],
            "invc_r2_log10_std": cv_results["val_aggregated"]["invc"]["r2_log10"]["std"],
            "foms_r2_log10": cv_results["val_aggregated"]["foms_direct"]["r2_log10"]["mean"],
            "foms_r2_log10_std": cv_results["val_aggregated"]["foms_direct"]["r2_log10"]["std"],
            "consistency_pearson_r": cv_results["val_aggregated"]["consistency"]["pearson_r"]["mean"],
            "consistency_pearson_r_std": cv_results["val_aggregated"]["consistency"]["pearson_r"]["std"],
            "sample_size": "5 folds",
            "role": "Interpolation stability",
        },
        {
            "validation_dimension": "Unseen structural-dielectric combinations",
            "qsc_r2_log10": fig05_metrics["qsc"]["r2_log10"],
            "qsc_r2_log10_std": "",
            "invc_r2_log10": fig05_metrics["invc"]["r2_log10"],
            "invc_r2_log10_std": "",
            "foms_r2_log10": fig05_metrics["foms_direct"]["r2_log10"],
            "foms_r2_log10_std": "",
            "consistency_pearson_r": fig05_metrics["consistency"]["pearson_r"],
            "consistency_pearson_r_std": "",
            "sample_size": fig05_metrics["n_points"],
            "role": "Design extrapolation usability",
        },
    ]
    table1_path = write_table_csv(
        pd.DataFrame(table1_rows), "table01_model_capability_summary.csv"
    )

    design_rules = pd.read_csv(DESIGN_RULES_RAW)
    table3_df = pd.DataFrame(
        {
            "rule_id": design_rules["#"],
            "design_guideline": design_rules["Design Rule"],
            "applicable_condition": design_rules["Applicable Condition"],
            "physical_basis": design_rules["Physical Mechanism"],
            "quantitative_evidence": design_rules["Quantitative Evidence"],
            "figure_anchor": design_rules["Corresponding Figure"],
        }
    )
    table2_path = write_table_csv(table3_df, "table02_design_guidelines.csv")

    return {
        "table01": table1_path,
        "table02": table2_path,
    }


def write_manifest(outputs, tables):
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": rel_path(OUT_ROOT),
        "figures": outputs,
        "tables": tables,
    }
    (SRC_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate publication figures and main-table source files."
    )
    parser.add_argument(
        "--figs",
        nargs="+",
        choices=ALL_FIG_KEYS,
        default=list(ALL_FIG_KEYS),
        help="Subset of figures to build, e.g. --figs fig02 fig03",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    targets = list(dict.fromkeys(args.figs))

    print("=" * 72)
    print("Publication figure generation")
    print("=" * 72)
    print(f"[目标] {', '.join(targets)}")

    ensure_dirs()
    configure_style()

    outputs = {}
    if "fig01" in targets:
        outputs["fig01"] = install_fig1_assets()
    if "fig02" in targets:
        outputs["fig02"] = build_fig2()

    need_prediction_stack = any(key in targets for key in ("fig03", "fig04", "fig05"))
    if need_prediction_stack:
        from predict_multitask_physics import load_model_and_scalers
        from utils_multitask_physics import get_device

        device = get_device()
        print(f"[设备] {device}")

        model, scaler_X, scaler_qsc, scaler_invc, scaler_foms = load_model_and_scalers(
            device=device,
            model_path=str(
                ROOT / "checkpoints_multitask_physics" / "physics_multitask_best.pth"
            ),
            artifact_dir=str(ROOT / "artifacts_multitask_physics"),
        )
        predict_fn = make_predict_fn(
            model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device
        )

        if "fig03" in targets:
            outputs["fig03"] = build_fig3(predict_fn)
        if "fig04" in targets:
            outputs["fig04"] = build_fig4(predict_fn)
        if "fig05" in targets:
            outputs["fig05"] = build_fig5(
                model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device
            )

    tables = export_main_table_sources()
    write_manifest(outputs, tables)

    print(f"\n[完成] figures_publication 已生成: {OUT_ROOT}")
    for key, value in outputs.items():
        print(f"  - {key}: {value['preview_file']}")
    for key, value in tables.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
