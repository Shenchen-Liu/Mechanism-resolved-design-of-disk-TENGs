#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the supporting-information figures and source tables.

The script reads released JSON and CSV artifacts and writes the final SI assets
into `figures_publication/si` and `figures_publication/src/si`.
"""

from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures_publication" / "si"
SOURCE_DIR = ROOT / "figures_publication" / "src" / "si"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig_si"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────

TEST_METRICS = ROOT / "outputs_multitask_physics" / "test_metrics.json"
OOD_METRICS = ROOT / "outputs_multitask_physics" / "ood_metrics.json"
BASELINE_RES = ROOT / "outputs" / "baselines" / "baseline_results.json"
BASELINE_OOD = ROOT / "outputs" / "baselines" / "baseline_ood_results.json"
CV_RESULTS = ROOT / "outputs" / "cross_validation" / "cv_results.json"
THRESH_SENS = ROOT / "outputs" / "sensitivity" / "threshold_sensitivity.csv"
GRID_SENS = ROOT / "outputs" / "sensitivity" / "grid_resolution_sensitivity.csv"
GRID_MAP_COMPARISON = ROOT / "outputs" / "sensitivity" / "fig_s8_grid_resolution.png"
GRID_SPATIAL_METRICS = SOURCE_DIR / "grid_resolution_spatial_metrics.json"

VAL1_CSV = ROOT / "data" / "multitask_validate1_results.csv"
VAL2_CSV = ROOT / "data" / "multitask_validate2_results.csv"
VAL3_CSV = ROOT / "data" / "multitask_validate3_results.csv"

# ── Publication style ────────────────────────────────────────────────────
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
REGIME_COLORS = {"charge": "#D7301F", "cap": "#2C7FB8", "mixed": "#BDBDBD"}
SAFE_GREEN = "#2B8A3E"
TASK_COLORS = {
    "qsc": "#0072B2",
    "invc": "#009E73",
    "foms_direct": "#D55E00",
    "foms_phys": "#CC79A7",
}
DATASET_COLORS = {
    "V1": "#0072B2",
    "V2": "#D55E00",
    "V3": "#009E73",
}
NEUTRAL_COLORS = {
    "light": "#B8C2CF",
    "dark": "#6B7280",
}
REF_LINE_COLOR = "#A8B0B8"
AGREEMENT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "agreement_green", ["#F8FAFC", "#CFE9D6", SAFE_GREEN]
)
CHARGE_RATIO_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "charge_ratio", ["#F8F6F2", "#E5DDD3", "#CBBCAA", "#A88B74"]
)
REGIME_CMAP = ListedColormap(
    [REGIME_COLORS["cap"], REGIME_COLORS["mixed"], REGIME_COLORS["charge"]]
)
REGIME_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], REGIME_CMAP.N)
FIG_TITLE_SIZE = 7.8
FIG_AXIS_LABEL_SIZE = 8.6
FIG_TICK_LABEL_SIZE = 7.0
FIG_PANEL_LABEL_SIZE = 10.4
FIG_NOTE_SIZE = 6.8
SHORT_DASH = (0, (2.2, 1.8))
SUPTITLE_SIZE = 9
PANEL_LABEL_X = -0.18
PANEL_LABEL_Y = 1.07
PANEL_LABEL_FONT = 9.2
PANEL_TITLE_X = 0.02
PANEL_TITLE_Y = 1.035
LEGEND_FONTSIZE = 6.2
ERRBAR_LW = 0.75
ABLATION_TEST_ROWS = [
    [r"$Q_{sc}$ " + r"$R^2_{\log_{10}}$", "0.9898", "0.9835±0.0014", "-0.0063"],
    [r"$C^{-1}_{sum}$ " + r"$R^2_{\log_{10}}$", "0.9966", "0.9942±0.0012", "-0.0025"],
    [r"FOMS$_{direct}$ " + r"$R^2_{\log_{10}}$", "0.9817", "0.9805±0.0040", "-0.0011"],
    ["Consistency Pearson", "0.9684", "0.9396±0.0247", "-0.0288"],
]
ABLATION_OOD_ROWS = [
    ["V1", r"FOMS $R^2_{\log_{10}}$", "0.8720", "0.9564", "+0.0844"],
    ["V1", "Consistency Pearson", "0.9770", "0.9700", "-0.0070"],
    ["V2", r"FOMS $R^2_{\log_{10}}$", "0.9538", "0.9489", "-0.0049"],
    ["V2", "Consistency Pearson", "0.9665", "0.9890", "+0.0225"],
    ["V3", r"FOMS $R^2_{\log_{10}}$", "0.9653", "0.9586", "-0.0067"],
    ["V3", "Consistency Pearson", "0.9886", "0.9900", "+0.0014"],
]


def rel_path(path: Path) -> str:
    """Return a repository-relative POSIX path for public manifests."""
    return path.relative_to(ROOT).as_posix()


def configure_style():
    """Match main-text publication style."""
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


def add_panel_label(ax, label, x=PANEL_LABEL_X, y=PANEL_LABEL_Y, fontsize=None):
    label = str(label).lower()
    if fontsize is None:
        fontsize = PANEL_LABEL_FONT
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
    )


def add_panel_header(
    ax,
    label,
    title,
    *,
    x=PANEL_LABEL_X,
    title_x=PANEL_TITLE_X,
    y=PANEL_LABEL_Y,
    title_y=PANEL_TITLE_Y,
    label_fontsize=None,
    title_fontsize=FIG_TITLE_SIZE,
):
    add_panel_label(ax, label, x=x, y=y, fontsize=label_fontsize)
    ax.text(
        title_x,
        title_y,
        title,
        transform=ax.transAxes,
        fontsize=title_fontsize,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_fig(fig, name):
    """Save figure in PNG + PDF to the final SI figure directory."""
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")


def write_table_csv(df: pd.DataFrame, filename: str):
    """Persist table source data for downstream LaTeX rendering."""
    path = SOURCE_DIR / filename
    df.to_csv(path, index=False)
    print(f"  Wrote table data: {path.name}")


def write_si_manifest():
    """Write a compact manifest for final SI assets."""
    figures = {}
    for stem in [
        "figS01_surrogate_architecture",
        "figS02_heldout_logscale_consistency",
        "figS03_model_selection_id_vs_ood",
        "figS04_cross_validation_stability",
        "figS05_ood_error_decomposition",
        "figS06_regime_parameter_sensitivity",
        "figS07_grid_refinement_stability",
    ]:
        files = [rel_path(p) for p in sorted(OUT_DIR.glob(f"{stem}.*"))]
        if files:
            figures[stem] = files

    tables = {}
    for path in sorted(SOURCE_DIR.glob("tableS*.csv")):
        tables[path.stem] = rel_path(path)

    manifest = {
        "output_root": rel_path(ROOT / "figures_publication"),
        "figure_dir": rel_path(OUT_DIR),
        "source_dir": rel_path(SOURCE_DIR),
        "figures": figures,
        "tables": tables,
        "grid_spatial_metrics": rel_path(GRID_SPATIAL_METRICS),
    }
    manifest_path = SOURCE_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Wrote manifest: {manifest_path.name}")


def remove_stale_outputs():
    """Delete deprecated SI artifacts from older layouts if they exist."""
    for name in [
        "fig_s1_architecture",
        "fig_s2_baseline_comparison",
        "fig_s3_parity_supplement",
        "fig_s3b_foms_magnitude_bands",
    ]:
        for ext in ("png", "pdf"):
            path = OUT_DIR / f"{name}.{ext}"
            if path.exists():
                path.unlink()
                print(f"  Removed stale artifact: {path.name}")


def draw_image_panel(
    ax,
    path: Path,
    title: str,
    label: str,
    *,
    x=PANEL_LABEL_X,
    title_x=PANEL_TITLE_X,
    y=PANEL_LABEL_Y,
    title_y=PANEL_TITLE_Y,
    note: str | None = None,
):
    """Render an existing PNG as a panel in a composite figure."""
    if path.exists():
        img = plt.imread(str(path))
        ax.imshow(img)
        ax.axis("off")
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"Missing file:\n{path.name}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )
    add_panel_header(ax, label, title, x=x, title_x=title_x, y=y, title_y=title_y)
    if note:
        ax.text(
            0.02,
            -0.08,
            note,
            transform=ax.transAxes,
            fontsize=FIG_NOTE_SIZE,
            color="#555",
            va="top",
        )


def load_grid_spatial_metrics():
    """Load precomputed S7 spatial-agreement metrics if available."""
    if GRID_SPATIAL_METRICS.exists():
        with open(GRID_SPATIAL_METRICS) as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════════
# Fig.S2 — Held-out log-scale parity + consistency
# ══════════════════════════════════════════════════════════════════════════
def fig_s2_heldout_prediction_structure():
    """
    Four-panel figure:
    (a) direct-vs-physical consistency
    (b) FOMS_direct parity
    (c) FOMS_phys parity
    (d) FOMS magnitude bands
    """
    with open(TEST_METRICS) as f:
        tm = json.load(f)

    fig = plt.figure(figsize=(17.5 / 2.54, 12.2 / 2.54))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.12, 1.12, 0.92], height_ratios=[1.22, 1.06]
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[:, 2])

    draw_image_panel(
        ax_a,
        ROOT / "outputs_multitask_physics" / "direct_vs_phys_consistency_log.png",
        "Direct–physical consistency",
        "a",
    )
    draw_image_panel(
        ax_b,
        ROOT / "outputs_multitask_physics" / "parity_foms_direct_log.png",
        "Direct FOMS parity",
        "b",
    )
    draw_image_panel(
        ax_c,
        ROOT / "outputs_multitask_physics" / "parity_foms_phys_log.png",
        "Reconstructed FOMS parity",
        "c",
    )

    ax = ax_d
    bands = tm.get("foms_by_magnitude_band")
    if bands:
        band_names = [
            "Low\n(<10" + r"$^{-8}$" + ")",
            "Mid\n(10" + r"$^{-8}$" + "~10" + r"$^{-5}$" + ")",
            "High\n(" + r"$\geq$" + "10" + r"$^{-5}$" + ")",
        ]
        band_keys = ["foms_band_low", "foms_band_mid", "foms_band_high"]
        r2_vals = [bands[k]["r2_log10"] for k in band_keys]
        n_samples = [bands[k]["n_samples"] for k in band_keys]
        colors_band = [TASK_COLORS["foms_direct"], REGIME_COLORS["mixed"], SAFE_GREEN]

        ax.bar(range(3), r2_vals, color=colors_band, edgecolor="white", lw=0.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(band_names, fontsize=6.0)
        ax.set_ylabel(r"$R^2_{\log_{10}}$", fontsize=FIG_AXIS_LABEL_SIZE)
        ax.axhline(0.95, color=REF_LINE_COLOR, ls=SHORT_DASH, lw=0.7, alpha=0.75)
        for i, (v, n) in enumerate(zip(r2_vals, n_samples)):
            y_text = max(v, 0) + 0.03
            va = "bottom"
            x_text = i
            ha = "center"
            if v < 0.1:
                y_text = 0.02
                va = "bottom"
                x_text = i + 0.03
                ha = "center"
            ax.text(
                x_text,
                y_text,
                "N=" + f"{n}\n" + r"$R^2_{\log_{10}}$" + f"={v:.3f}",
                ha=ha,
                va=va,
                fontsize=5.5,
            )
        ax.set_ylim(-0.3, 1.18)
        hide_extra_spines(ax)
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Missing magnitude-band summary",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )
    add_panel_header(ax, "d", "Magnitude-stratified FOMS fidelity")

    fig.suptitle(
        "Held-out predictions remain accurate and self-consistent on the log scale",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95], w_pad=1.4, h_pad=1.2)
    save_fig(fig, "figS02_heldout_logscale_consistency")


# ══════════════════════════════════════════════════════════════════════════
# Fig.S3 — Baseline capability comparison
# ══════════════════════════════════════════════════════════════════════════
def fig_s3_baseline_comparison():
    """Two-panel: (a) test-set R²_log10, (b) OOD FOMS R²_log10."""
    with open(BASELINE_RES) as f:
        bl = json.load(f)
    with open(BASELINE_OOD) as f:
        bl_ood = json.load(f)
    with open(OOD_METRICS) as f:
        ood = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 7.0 / 2.54))

    # ── (a) Test-set R²_log10 ──
    ax = axes[0]
    models = ["XGB (independent ×3)", "MLP (no consistency)", "MLP (+ consistency)"]
    main_names = ["XGBoost", "Multi-MLP", "Multi-MLP", "Ours"]
    sub_names = ["(indep. ×3)", "(no cons.)", "(+ cons.)", "(Transformer)"]
    targets = ["qsc", "invc", "foms_direct"]
    target_labels = [r"$Q_{sc}$", r"$C^{-1}_{sum}$", r"FOMS$_{direct}$"]
    colors = [TASK_COLORS["qsc"], TASK_COLORS["invc"], TASK_COLORS["foms_direct"]]

    x = np.arange(len(main_names))
    width = 0.20

    for j, (tgt, tgt_label, c) in enumerate(zip(targets, target_labels, colors)):
        vals, errs = [], []
        for m in models:
            vals.append(bl["baselines"][m][tgt]["r2_log10"]["mean"])
            errs.append(bl["baselines"][m][tgt]["r2_log10"]["std"])
        vals.append(bl["our_model"][tgt]["r2_log10"])
        errs.append(0)

        offset = (j - 1) * width
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            label=tgt_label,
            color=c,
            edgecolor="white",
            linewidth=0.45,
            capsize=2,
            error_kw={"lw": ERRBAR_LW},
        )

    ax.set_ylabel(r"$R^2_{\log_{10}}$ (test set)")
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(main_names))
    ax.tick_params(axis="x", length=0)
    for xpos, main, sub in zip(x, main_names, sub_names):
        ax.text(
            xpos,
            -0.085,
            main,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.2,
            color="#111827",
        )
        ax.text(
            xpos,
            -0.165,
            sub,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.2,
            color="#4B5563",
        )
    ax.set_ylim(0.88, 1.0)
    ax.legend(
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.25),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.5,
        columnspacing=1.2,
    )
    ax.axhline(0.95, color=REF_LINE_COLOR, ls=SHORT_DASH, lw=0.7, alpha=0.75)
    ax.text(3.58, 0.9513, "0.95 threshold", fontsize=4.8, color="gray", alpha=0.75)
    hide_extra_spines(ax)
    add_panel_header(ax, "a", "Held-out accuracy is competitive", y=1.07)

    # ── (b) OOD FOMS R²_log10 ──
    ax = axes[1]
    val_sets = ["validate1", "validate2", "validate3"]
    val_labels = ["V1 (n OOD)", "V2 (n+E OOD)", "V3 (multi-scene)"]
    val_colors = [DATASET_COLORS["V1"], DATASET_COLORS["V2"], DATASET_COLORS["V3"]]

    x = np.arange(len(main_names))
    width = 0.20

    for j, (vs, vl, vc) in enumerate(zip(val_sets, val_labels, val_colors)):
        vals = []
        for m in models:
            vals.append(bl_ood[vs][m]["foms_direct"]["r2_log10"])
        vals.append(ood[vs]["foms_direct"]["r2_log10"])

        offset = (j - 1) * width
        edge = "#98A2B3" if vs == "validate2" else "white"
        lw = 0.8 if vs == "validate2" else 0.45
        ax.bar(
            x + offset, vals, width, label=vl, color=vc, edgecolor=edge, linewidth=lw
        )

    ax.set_ylabel(r"FOMS $R^2_{\log_{10}}$ (OOD)")
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(main_names))
    ax.tick_params(axis="x", length=0)
    for xpos, main, sub in zip(x, main_names, sub_names):
        ax.text(
            xpos,
            -0.085,
            main,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.2,
            color="#111827",
        )
        ax.text(
            xpos,
            -0.165,
            sub,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.2,
            color="#4B5563",
        )
    ax.set_ylim(-0.8, 1.0)
    ax.axhline(0, color="gray", ls="-", lw=0.5, alpha=0.4)
    ax.axhline(1.0, color=REF_LINE_COLOR, ls=SHORT_DASH, lw=0.7, alpha=0.75)
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.25), frameon=False, fontsize=LEGEND_FONTSIZE)
    hide_extra_spines(ax)
    add_panel_header(ax, "b", "OOD ranking changes sharply", y=1.07)

    fig.suptitle(
        "In-distribution competitiveness does not ensure OOD reliability",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.94], w_pad=2.5)
    save_fig(fig, "figS03_model_selection_id_vs_ood")


# ══════════════════════════════════════════════════════════════════════════
# Fig.S4 — 5-fold CV distribution
# ══════════════════════════════════════════════════════════════════════════
def fig_s4_cross_validation():
    """Three panels: (a) R²_log10 by fold, (b) train-val gap, (c) consistency."""
    with open(CV_RESULTS) as f:
        cv = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(19.2 / 2.54, 7.2 / 2.54))

    # ── (a) R²_log10 by target across folds ──
    ax = axes[0]
    targets = ["qsc", "invc", "foms_direct", "foms_phys"]
    target_labels = [r"$Q_{sc}$", r"$C^{-1}_{sum}$", r"FOMS$_{dir}$", r"FOMS$_{phys}$"]
    colors_cv = [
        TASK_COLORS["qsc"],
        TASK_COLORS["invc"],
        TASK_COLORS["foms_direct"],
        TASK_COLORS["foms_phys"],
    ]

    fold_data = cv["per_fold_val_r2_log10"]
    positions = np.arange(len(targets))

    for i, (tgt, label, c) in enumerate(zip(targets, target_labels, colors_cv)):
        vals = fold_data[tgt]
        # Jittered strip plot
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            positions[i] + jitter,
            vals,
            color=c,
            s=25,
            alpha=0.62,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        # Mean ± std
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        ax.errorbar(
            positions[i] + 0.25,
            mean_v,
            yerr=std_v,
            fmt="D",
            color=c,
            markersize=5,
            capsize=3,
            lw=ERRBAR_LW,
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(target_labels, fontsize=7)
    ax.set_ylabel(r"Val $R^2_{\log_{10}}$")
    ax.set_ylim(0.952, 0.999)
    ax.axhline(0.95, color=REF_LINE_COLOR, ls=SHORT_DASH, lw=0.7, alpha=0.75)
    hide_extra_spines(ax)
    add_panel_header(ax, "a", r"Validation R$^2_{\log_{10}}$ remains high", x=-0.16)

    # ── (b) Train-val gap ──
    ax = axes[1]
    gaps = []
    gap_labels = []
    gap_colors_list = []
    for tgt, label, c in zip(targets, target_labels, colors_cv):
        train_vals = [fd["train_metrics"][tgt]["r2_log10"] for fd in cv["fold_details"]]
        val_vals = fold_data[tgt]
        gap = [t - v for t, v in zip(train_vals, val_vals)]
        gaps.append(gap)
        gap_labels.append(label)
        gap_colors_list.append(c)

    flierprops = dict(
        marker="o",
        markersize=4,
        markerfacecolor="white",
        markeredgecolor="#111827",
        markeredgewidth=0.8,
        alpha=0.9,
    )
    bp = ax.boxplot(
        gaps,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "#333", "lw": 1.0},
        boxprops={"linewidth": 0.9, "edgecolor": "#374151"},
        whiskerprops={"linewidth": 0.9, "color": "#374151"},
        capprops={"linewidth": 0.9, "color": "#374151"},
        flierprops=flierprops,
    )
    for patch, c in zip(bp["boxes"], gap_colors_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(gap_labels, fontsize=7)
    ax.set_ylabel(r"Train - Val $R^2_{\log_{10}}$")
    ax.axhline(0, color="gray", ls="-", lw=0.5, alpha=0.4)
    hide_extra_spines(ax)
    add_panel_header(ax, "b", "Train–validation gap remains small", x=-0.16)

    # ── (c) Consistency across folds ──
    ax = axes[2]
    pearson_vals = [
        fd["val_metrics"]["consistency"]["pearson_r"] for fd in cv["fold_details"]
    ]
    spearman_vals = [
        fd["val_metrics"]["consistency"]["spearman_r"] for fd in cv["fold_details"]
    ]
    folds = range(1, 6)

    pearson_color = TASK_COLORS["qsc"]
    spearman_color = TASK_COLORS["foms_direct"]
    ax.plot(
        folds,
        pearson_vals,
        "o-",
        color=pearson_color,
        label="Pearson r",
        markersize=5,
        lw=1.2,
    )
    ax.plot(
        folds,
        spearman_vals,
        "s--",
        color=spearman_color,
        label=r"Spearman $\rho$",
        markersize=5,
        lw=1.2,
    )

    mean_p = np.mean(pearson_vals)
    ax.axhline(mean_p, color=REF_LINE_COLOR, ls=SHORT_DASH, lw=0.7, alpha=0.75)
    ax.text(
        5.20,
        mean_p,
        f"{mean_p:.3f}",
        fontsize=6,
        color=pearson_color,
        va="center",
        ha="right",
    )

    ax.set_xlabel("Fold", fontsize=FIG_AXIS_LABEL_SIZE)
    ax.set_ylabel("Correlation", fontsize=FIG_AXIS_LABEL_SIZE)
    ax.set_ylim(0.92, 1.005)
    ax.tick_params(labelsize=FIG_TICK_LABEL_SIZE)
    ax.legend(
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        loc="center left",
        bbox_to_anchor=(1.02, 0.22),
    )
    hide_extra_spines(ax)
    add_panel_header(ax, "c", "Consistency remains stable across folds", x=-0.20)

    fig.suptitle(
        "Cross-validation confirms stable interpolation and limited overfitting",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.16, wspace=0.58)
    save_fig(fig, "figS04_cross_validation_stability")


# ══════════════════════════════════════════════════════════════════════════
# Fig.S5 — OOD error decomposition
# ══════════════════════════════════════════════════════════════════════════
def fig_s5_ood_decomposition():
    """
    Three-panel figure:
    (a) Per-n FOMS relative error across V1/V2/V3
    (b) V3 ID-n vs OOD-n error comparison
    (c) Per-scenario FOMS MAE_log10
    """
    with open(OOD_METRICS) as f:
        ood = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(17.5 / 2.54, 7.0 / 2.54))

    # ── (a) Per-n FOMS error across validation sets ──
    ax = axes[0]
    val_colors = {
        "validate1": DATASET_COLORS["V1"],
        "validate2": DATASET_COLORS["V2"],
        "validate3": DATASET_COLORS["V3"],
    }
    val_labels = {"validate1": "V1", "validate2": "V2", "validate3": "V3"}

    for vset, color in val_colors.items():
        if vset in ood.get("per_n_breakdown", {}):
            breakdown = ood["per_n_breakdown"][vset]
            n_vals = sorted([int(k) for k in breakdown.keys()])
            errors = [breakdown[str(n)]["foms_direct_rel_error_%"] for n in n_vals]
            ax.plot(
                n_vals,
                errors,
                "o-",
                color=color,
                label=val_labels[vset],
                markersize=4,
                lw=1.1,
                alpha=0.85,
            )

    ax.set_xlabel("n (number of segments)")
    ax.set_ylabel("FOMS relative error (%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xticklabels(
        [r"$2^2$", r"$2^3$", r"$2^4$", r"$2^5$"], fontsize=FIG_TICK_LABEL_SIZE
    )
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper left")
    ax.axhline(20, color=REF_LINE_COLOR, ls=SHORT_DASH, lw=0.7, alpha=0.75)
    ax.text(52, 20.6, "20%", fontsize=4.8, color="gray", ha="left")
    hide_extra_spines(ax)
    add_panel_header(ax, "a", "Scenario shifts dominate OOD error", x=-0.16)

    # ── (b) V3 ID-n vs OOD-n ──
    ax = axes[1]
    if "validate3_error_decomposition" in ood:
        decomp = ood["validate3_error_decomposition"]
        groups = ["id_n", "ood_n"]
        group_labels = ["ID-n\n(n=4,16; N=6)", "OOD-n\n(n=3,7,24; N=9)"]
        bar_colors = [NEUTRAL_COLORS["light"], NEUTRAL_COLORS["dark"]]

        # Data is nested: decomp[group]["foms_direct"]["r2_log10"]
        r2_vals = []
        mae_vals = []
        for g in groups:
            gd = decomp.get(g, {})
            fd = gd.get("foms_direct", {})
            r2_vals.append(fd.get("r2_log10", 0))
            mae_vals.append(fd.get("mae_log10", 0))

        x_pos = np.arange(2)
        bars = ax.bar(
            x_pos, r2_vals, color=bar_colors, edgecolor="#475467", lw=0.7, width=0.5
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, fontsize=7)
        ax.set_ylabel(r"FOMS $R^2_{\log_{10}}$")
        ax.set_ylim(0.9, 1.0)

        for i, v in enumerate(r2_vals):
            ax.text(
                i, v + 0.003, f"{v:.3f}", ha="center", fontsize=6.5, fontweight="bold"
            )

        ax.axhline(0.95, color=REF_LINE_COLOR, ls=SHORT_DASH, lw=0.7, alpha=0.75)
    else:
        ax.text(
            0.5,
            0.5,
            "No decomposition data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )

    hide_extra_spines(ax)
    add_panel_header(ax, "b", "Unseen n is not the main failure mode", x=-0.16)

    # ── (c) Per-scenario MAE_log10 ──
    ax = axes[2]
    if "validate3_scenarios" in ood:
        scenarios = ood["validate3_scenarios"]
        sc_names = sorted(scenarios.keys())
        sc_labels_map = {
            "A": r"A ($\varepsilon$=1.5)",
            "B": r"B ($\varepsilon$=6)",
            "C": r"C ($\varepsilon$=8)",
        }
        sc_colors = [DATASET_COLORS["V1"], DATASET_COLORS["V2"], DATASET_COLORS["V3"]]

        mae_foms = []
        pearson_cons = []
        for sc in sc_names:
            sd = scenarios[sc]
            # Nested structure: scenarios[sc]["foms_direct"]["mae_log10"]
            fd = sd.get("foms_direct", {})
            mae_foms.append(fd.get("mae_log10", 0))
            cd = sd.get("consistency", {})
            pearson_cons.append(cd.get("pearson_r", 0))

        x_pos = np.arange(len(sc_names))
        bars = ax.bar(
            x_pos,
            mae_foms,
            color=sc_colors[: len(sc_names)],
            edgecolor="white",
            lw=0.5,
            width=0.5,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([sc_labels_map.get(s, s) for s in sc_names], fontsize=6.5)
        ax.set_ylabel(r"FOMS MAE$_{\log_{10}}$")

        y_offset = max(mae_foms) * 0.035
        for i, v in enumerate(mae_foms):
            ax.text(i, v + y_offset, f"{v:.3f}", ha="center", fontsize=6.5)

    else:
        ax.text(
            0.5,
            0.5,
            "No scenario data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )

    hide_extra_spines(ax)
    add_panel_header(ax, "c", "Scenario C is easiest under V3", x=-0.16)

    fig.suptitle(
        "OOD degradation is driven more by scenario shift than by unseen n alone",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.13, wspace=0.72)
    save_fig(fig, "figS05_ood_error_decomposition")


# ══════════════════════════════════════════════════════════════════════════
# Fig.S6 — Regime boundary sensitivity
# ══════════════════════════════════════════════════════════════════════════
def fig_s6_regime_sensitivity():
    """
    Two-panel: (a) charge/cap ratio heatmap (k × threshold),
    (b) charge-dominant % line plot by threshold for different k.
    """
    df = pd.read_csv(THRESH_SENS)

    fig, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 7.0 / 2.54))

    # ── (a) Charge/cap ratio heatmap ──
    ax = axes[0]
    pivot = df.pivot(
        index="k_neighbors", columns="dominance_threshold", values="charge_cap_ratio"
    )
    hm = sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap=CHARGE_RATIO_CMAP,
        ax=ax,
        cbar_kws={"label": "Charge/cap", "shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 6.5, "ha": "center", "va": "center"},
    )
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.4, length=1.8)
    cbar.set_label("Charge/cap", fontsize=5.6, labelpad=2.5)
    ax.set_xlabel("Dominance threshold")
    ax.set_ylabel("k (neighbors)")
    ax.set_xticklabels(
        [f"{float(t.get_text()):.2f}" for t in ax.get_xticklabels()],
        fontsize=FIG_TICK_LABEL_SIZE,
    )
    add_panel_header(ax, "a", "Charge dominance exceeds capacitance")

    # ── (b) Charge-dominant % by threshold ──
    ax = axes[1]
    k_values = sorted(df["k_neighbors"].unique())
    k_colors = [
        TASK_COLORS["qsc"],
        TASK_COLORS["foms_direct"],
        TASK_COLORS["invc"],
        TASK_COLORS["foms_phys"],
        "#E69F00",
    ][: len(k_values)]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    markers = ["o", "s", "^", "D", "v"]

    for idx, (k, c) in enumerate(zip(k_values, k_colors)):
        sub = df[df["k_neighbors"] == k].sort_values("dominance_threshold")
        ax.plot(
            sub["dominance_threshold"],
            sub["charge_pct"],
            color=c,
            label=f"k={k}",
            markersize=4,
            lw=1.15,
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],
        )

    # Reference line at k=50, thr=0.62
    ref_row = df[(df["k_neighbors"] == 50) & (df["dominance_threshold"] == 0.62)]
    if len(ref_row) > 0:
        ax.axvline(0.62, color="#C6CDD4", ls=":", lw=0.55, alpha=0.34)
        ax.plot(
            0.62,
            ref_row["charge_pct"].values[0],
            "*",
            color="red",
            markersize=10,
            zorder=5,
            label="Reference",
        )

    ax.set_xlabel("Dominance threshold")
    ax.set_ylabel("Charge-dominant (%)")
    ax.set_xticks([0.55, 0.58, 0.62, 0.65, 0.70])
    ax.set_xticklabels(
        [f"{v:.2f}" for v in [0.55, 0.58, 0.62, 0.65, 0.70]],
        fontsize=FIG_TICK_LABEL_SIZE,
    )
    ax.legend(
        frameon=False,
        fontsize=LEGEND_FONTSIZE * 0.9,
        ncol=2,
        loc="upper right",
        markerscale=0.9,
        handlelength=1.35,
        handletextpad=0.28,
        columnspacing=0.72,
        borderaxespad=0.18,
        labelspacing=0.3,
    )
    hide_extra_spines(ax)
    add_panel_header(ax, "b", "Reference setting lies in a stable zone")

    fig.suptitle(
        "Regime asymmetry remains robust across threshold and neighborhood choices",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.16, wspace=0.72)
    save_fig(fig, "figS06_regime_parameter_sensitivity")


# ══════════════════════════════════════════════════════════════════════════
# Fig.S7 — Grid resolution sensitivity
# ══════════════════════════════════════════════════════════════════════════
def fig_s7_grid_resolution():
    """Composite figure: fraction stability + spatial agreement + map comparison."""
    df = pd.read_csv(GRID_SENS)
    metrics = load_grid_spatial_metrics()

    fig = plt.figure(figsize=(25.0 / 2.54, 7.8 / 2.54))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.05, 0.34, 1.0, 0.16, 1.85])
    ax_frac = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_agreement = fig.add_subplot(gs[0, 2])
    ax_map = fig.add_subplot(gs[0, 4])
    ax_legend.axis("off")

    # ── (a) Fraction stability ──
    ax = ax_frac
    x = np.arange(len(df))
    width = 0.25
    colors_regime = [
        REGIME_COLORS["charge"],
        REGIME_COLORS["cap"],
        REGIME_COLORS["mixed"],
    ]

    ax.bar(
        x - width,
        df["charge_pct"],
        width,
        color=colors_regime[0],
        label="Charge-dominant",
        edgecolor="white",
        lw=0.5,
    )
    ax.bar(
        x,
        df["cap_pct"],
        width,
        color=colors_regime[1],
        label="Capacitance-dominant",
        edgecolor="white",
        lw=0.5,
    )
    ax.bar(
        x + width,
        df["mixed_pct"],
        width,
        color=colors_regime[2],
        label="Mixed",
        edgecolor="white",
        lw=0.5,
    )

    res_labels = []
    for _, row in df.iterrows():
        hh_n = int(row["hh_n_points"])
        n_grid = int(row["n_grid_points"])
        label_str = f"hh={hh_n}\n({n_grid} pts)"
        if hh_n == 9:
            label_str += "\n(train)"
        elif hh_n == 18:
            label_str += "\n(2" + r"$\times$" + ")"
        elif hh_n == 36:
            label_str += "\n(4" + r"$\times$" + ")"
        res_labels.append(label_str)

    ax.set_xticks(x)
    ax.set_xticklabels(res_labels, fontsize=6.5)
    ax.set_ylabel("Percentage (%)")
    handles, _ = ax.get_legend_handles_labels()
    ax_legend.legend(
        handles,
        ["Charge-dom.", "Cap-dom.", "Mixed"],
        frameon=False,
        fontsize=5.9,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(-0.18, 0.98),
        borderaxespad=0.0,
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=0.38,
    )
    hide_extra_spines(ax)
    ax.set_ylim(0, 75)
    add_panel_header(ax, "a", "Class fractions change minimally")

    # ── (b) Pairwise spatial agreement ──
    ax = ax_agreement
    if metrics:
        labels = [9, 18, 36]
        agreement = np.eye(3)
        mean_iou = np.eye(3)
        index = {v: i for i, v in enumerate(labels)}
        for item in metrics["pairwise_metrics"]:
            i, j = index[item["pair"][0]], index[item["pair"][1]]
            agreement[i, j] = agreement[j, i] = item["agreement"]
            mean_iou[i, j] = mean_iou[j, i] = item["mean_iou"]

        hm = sns.heatmap(
            agreement,
            ax=ax,
            cmap=AGREEMENT_CMAP,
            vmin=0.90,
            vmax=1.0,
            cbar_kws={
                "label": "Pixel-wise agreement",
                "shrink": 0.82,
                "fraction": 0.065,
                "pad": 0.035,
            },
            linewidths=0.5,
            linecolor="white",
            annot=False,
        )
        ax.set_xticklabels([f"hh={v}" for v in labels], rotation=0, fontsize=6.5)
        ax.set_yticklabels([f"hh={v}" for v in labels], rotation=0, fontsize=6.5)
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6.0, length=2.0)
        cbar.set_label("Pixel-wise agreement", fontsize=6.3, labelpad=3)

        for i in range(3):
            for j in range(3):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{agreement[i, j]*100:.1f}%\nIoU {mean_iou[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="#173b1a" if agreement[i, j] > 0.9 else "#333",
                )

    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Missing spatial-agreement metrics.\nRun code/compute_si_grid_spatial_metrics.py",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )
    add_panel_header(ax, "b", "Spatial agreement stays high")

    # ── (c) Map comparison ──
    ax = ax_map
    ax.axis("off")
    if metrics and metrics.get("maps"):
        map_records = sorted(metrics["maps"], key=lambda item: item["hh_n_points"])
        n_values = metrics.get("n_values", [2, 4, 8, 16, 32, 64])
        lefts = [0.02, 0.35, 0.68]
        width = 0.28
        for idx, rec in enumerate(map_records[:3]):
            inset = ax.inset_axes([lefts[idx], 0.16, width, 0.66])
            regime = np.asarray(rec["regime"], dtype=float)
            hh_values = np.asarray(rec["hh_values"], dtype=float)

            inset.imshow(
                regime,
                cmap=REGIME_CMAP,
                norm=REGIME_NORM,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
            )
            inset.set_title(
                f"hh = {rec['hh_n_points']}", fontsize=6.8, fontweight="bold", pad=2
            )
            inset.set_xticks(np.arange(len(n_values)))
            inset.set_xticklabels([str(v) for v in n_values], fontsize=5.2)

            y_idx = [0, max(0, len(hh_values) // 2), len(hh_values) - 1]
            y_idx = sorted(set(y_idx))
            inset.set_yticks(y_idx)
            inset.set_yticklabels([f"{hh_values[i]:.1g}" for i in y_idx], fontsize=5.0)

            if idx == 0:
                inset.set_ylabel("h/R", fontsize=6.0)
            else:
                inset.set_yticklabels([])
                inset.set_yticks([])
            inset.set_xlabel("n", fontsize=6.0, labelpad=1)
            inset.tick_params(length=1.8, pad=1)
            hide_extra_spines(inset)

            frac_row = df[df["hh_n_points"] == rec["hh_n_points"]].iloc[0]
            inset.text(
                0.98,
                0.98,
                f"C {frac_row['charge_pct']:.0f}%\nM {frac_row['mixed_pct']:.0f}%\nP {frac_row['cap_pct']:.0f}%",
                transform=inset.transAxes,
                fontsize=4.8,
                color="#4B5563",
                ha="right",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    alpha=0.78,
                    edgecolor="none",
                ),
            )
    else:
        ax.text(
            0.5,
            0.5,
            "Missing regime-map matrices in grid_resolution_spatial_metrics.json",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
            color="gray",
        )
    add_panel_header(ax, "c", "Boundaries remain visually stable", x=-0.08)

    fig.suptitle(
        "Grid refinement preserves regime structure and boundary agreement",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
        y=1.02,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.79, bottom=0.17, wspace=0.18)
    save_fig(fig, "figS07_grid_refinement_stability")


# ══════════════════════════════════════════════════════════════════════════
# Table S1-S4 — source data export for LaTeX
# ══════════════════════════════════════════════════════════════════════════
def export_table_s1_head_metrics():
    """Export Table S1 source data as CSV for LaTeX rendering."""
    with open(TEST_METRICS) as f:
        tm = json.load(f)
    with open(CV_RESULTS) as f:
        cv = json.load(f)
    with open(OOD_METRICS) as f:
        ood = json.load(f)

    rows = []
    for key, label in [
        ("qsc", "Qsc"),
        ("invc", "invC"),
        ("foms_direct", "FOMS_direct"),
        ("foms_phys", "FOMS_phys"),
    ]:
        rows.append(
            {
                "head": label,
                "test_r2_log10": tm[key]["r2_log10"],
                "cv_r2_log10_mean": cv["val_aggregated"][key]["r2_log10"]["mean"],
                "cv_r2_log10_std": cv["val_aggregated"][key]["r2_log10"]["std"],
                "v1_r2_log10": ood["validate1"][key]["r2_log10"],
                "v2_r2_log10": ood["validate2"][key]["r2_log10"],
                "v3_r2_log10": ood["validate3"][key]["r2_log10"],
                "test_mae_log10": tm[key]["mae_log10"],
            }
        )

    write_table_csv(pd.DataFrame(rows), "tableS1_head_metrics.csv")


def export_table_s2_cv_fold_details():
    """Export Table S2 source data as CSV for LaTeX rendering."""
    with open(CV_RESULTS) as f:
        cv = json.load(f)

    rows = []
    for fd in cv["fold_details"]:
        vm = fd["val_metrics"]
        rows.append(
            {
                "fold": fd["fold"],
                "n_val": fd["n_val"],
                "qsc_r2_log10": vm["qsc"]["r2_log10"],
                "invc_r2_log10": vm["invc"]["r2_log10"],
                "foms_direct_r2_log10": vm["foms_direct"]["r2_log10"],
                "foms_phys_r2_log10": vm["foms_phys"]["r2_log10"],
                "consistency_pearson": vm["consistency"]["pearson_r"],
                "consistency_spearman": vm["consistency"]["spearman_r"],
            }
        )

    write_table_csv(pd.DataFrame(rows), "tableS2_cv_fold_details.csv")


def export_table_s3_hyperparameters():
    """Export Table S3 source data as CSV for LaTeX rendering."""
    rows = [
        {"hyperparameter": "embed_dim", "value": "256"},
        {"hyperparameter": "nhead", "value": "4"},
        {"hyperparameter": "num_layers", "value": "2"},
        {"hyperparameter": "dropout", "value": "0.02"},
        {"hyperparameter": "lambda_consistency", "value": "0.3"},
        {"hyperparameter": "lambda_qsc", "value": "1.0"},
        {"hyperparameter": "lambda_invc", "value": "1.0"},
        {"hyperparameter": "lambda_foms", "value": "1.0"},
        {"hyperparameter": "learning_rate", "value": "5e-4"},
        {"hyperparameter": "batch_size", "value": "32"},
        {"hyperparameter": "max_epochs", "value": "300"},
        {"hyperparameter": "early_stopping_patience", "value": "50"},
        {"hyperparameter": "lr_scheduler", "value": "CosineAnnealing"},
        {"hyperparameter": "T_0", "value": "50"},
        {"hyperparameter": "T_mult", "value": "2"},
        {"hyperparameter": "seed", "value": "42"},
        {"hyperparameter": "train_val_test_split", "value": "80/10/10"},
        {"hyperparameter": "n_total_samples", "value": "1944"},
    ]
    write_table_csv(pd.DataFrame(rows), "tableS3_hyperparameters.csv")


def export_table_s4_consistency_ablation():
    """Export Table S4 source data as CSV for LaTeX rendering."""
    test_df = pd.DataFrame(
        [
            {
                "metric": row[0],
                "t1_with_consistency": row[1],
                "t5_no_consistency": row[2],
                "delta": row[3],
            }
            for row in ABLATION_TEST_ROWS
        ]
    )
    ood_df = pd.DataFrame(
        [
            {
                "validation_set": row[0],
                "metric": row[1],
                "t1_with_consistency": row[2],
                "t5_no_consistency": row[3],
                "delta": row[4],
            }
            for row in ABLATION_OOD_ROWS
        ]
    )
    write_table_csv(test_df, "tableS4_consistency_ablation_test.csv")
    write_table_csv(ood_df, "tableS4_consistency_ablation_ood.csv")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    configure_style()
    print("=" * 60)
    print("Generating final SI assets...")
    print(f"Figure directory: {OUT_DIR}")
    print(f"Table/source directory: {SOURCE_DIR}")
    print("=" * 60)
    remove_stale_outputs()

    print("\n[1/6] Fig.S2 — Held-out parity + consistency + magnitude bands")
    fig_s2_heldout_prediction_structure()

    print("\n[2/6] Fig.S3 — Baseline capability comparison")
    fig_s3_baseline_comparison()

    print("\n[3/6] Fig.S4 — 5-fold CV distribution")
    fig_s4_cross_validation()

    print("\n[4/6] Fig.S5 — OOD error decomposition")
    fig_s5_ood_decomposition()

    print("\n[5/6] Fig.S6 — Regime boundary sensitivity")
    fig_s6_regime_sensitivity()

    print("\n[6/6] Fig.S7 — Grid resolution sensitivity")
    fig_s7_grid_resolution()

    print("\n[Table S1] Export head metrics CSV")
    export_table_s1_head_metrics()

    print("\n[Table S2] Export fold-wise CV CSV")
    export_table_s2_cv_fold_details()

    print("\n[Table S3] Export hyperparameter CSV")
    export_table_s3_hyperparameters()

    print("\n[Table S4] Export ablation CSVs")
    export_table_s4_consistency_ablation()

    print("\n[Manifest] Write SI asset manifest")
    write_si_manifest()

    # Summary
    print("\n" + "=" * 60)
    outputs = sorted(OUT_DIR.glob("*"))
    print(f"Generated {len(outputs)} figure assets:")
    for p in outputs:
        print(f"  {p.name}")
    sources = sorted(SOURCE_DIR.glob("*"))
    print(f"Generated {len(sources)} source assets:")
    for p in sources:
        print(f"  {p.name}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
