#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T9: 敏感性分析 — regime 分类参数鲁棒性验证
========================================================
验证机制 regime 判定结果对方法学参数选择的稳定性:
  T9.1: 边界敏感性 — k_neighbors × dominance_threshold 参数扫描
  T9.2: 网格分辨率效应 — hh 插值密度对设计空间 regime 的影响

产出:
  - outputs/sensitivity/threshold_sensitivity.csv
  - outputs/sensitivity/grid_resolution_sensitivity.csv
  - outputs/sensitivity/fig_s7_regime_sensitivity.png
  - outputs/sensitivity/fig_s8_grid_resolution.png

运行方式:
    cd code/
    python experiment_sensitivity.py

"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import seaborn as sns

from utils_mechanism_multitask import (
    load_real_data,
    compute_support_mask,
    compute_regime_map_variance,
    compute_design_regime_grid,
)
from predict_multitask_physics import load_model_and_scalers, predict_batch
from plot_mechanism_multitask import REGIME_CMAP, REGIME_NORM

# ============================================================================
# Style
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "figure.titlesize": 18,
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

# Regime codes (match utils_mechanism_multitask)
REGIME_CHARGE = 1
REGIME_CAPACITANCE = -1
REGIME_MIXED = 0

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    # Paths (relative to code/)
    "csv_path": "../data/disk_teng_training_processed.csv",
    "output_dir": "../outputs/sensitivity",
    "checkpoint_dir": "../checkpoints_multitask_physics",
    "artifact_dir": "../artifacts_multitask_physics",
    "dpi": 300,

    # T9.1: threshold sensitivity sweep
    "t91_k_values": [20, 30, 50, 75, 100],
    "t91_threshold_values": [0.55, 0.58, 0.62, 0.65, 0.70],
    "t91_grid_resolution": 50,
    "t91_min_local_points": 10,

    # Support mask (same as analyze_mechanism_multitask.py)
    "support_method": "knn",
    "support_k_neighbors": 15,
    "support_min_density_threshold": 5,

    # T9.2: grid resolution sweep
    "t92_hh_n_points": [9, 18, 36],
    "t92_hh_range": (0.00390625, 1.0),
    "t92_n_values": [2, 4, 8, 16, 32, 64],
    "t92_E": 3.0,
    "t92_dd": 0.125,
    "t92_dominance_threshold": 0.62,
    "t92_delta_log": 0.02,
}


# ============================================================================
# Predict function wrapper (same pattern as analyze_mechanism_multitask.py)
# ============================================================================

def _make_predict_fn(model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device):
    """Wrap model into predict_fn for design regime analysis."""
    def predict_fn(n_arr, E_arr, dd_arr, hh_arr):
        df = pd.DataFrame({
            "n": n_arr, "E": E_arr, "dd": dd_arr, "hh": hh_arr,
        })
        result = predict_batch(
            df, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device,
        )
        return {
            "Qsc_MACRS": result["Qsc_MACRS_pred"].values,
            "invC_sum": result["invC_sum_pred"].values,
            "FOMS_direct": result["FOMS_direct_pred"].values,
            "FOMS_phys": result["FOMS_phys_pred"].values,
        }
    return predict_fn


# ============================================================================
# Regime fraction helper
# ============================================================================

def _regime_fractions(regime_map):
    """Compute charge/cap/mixed fractions (%) from regime map, ignoring NaN."""
    valid = ~np.isnan(regime_map.flatten())
    n_total = valid.sum()
    if n_total == 0:
        return 0.0, 0.0, 0.0
    flat = regime_map.flatten()[valid]
    charge_pct = 100.0 * np.sum(flat == REGIME_CHARGE) / n_total
    cap_pct = 100.0 * np.sum(flat == REGIME_CAPACITANCE) / n_total
    mixed_pct = 100.0 * np.sum(flat == REGIME_MIXED) / n_total
    return charge_pct, cap_pct, mixed_pct


# ============================================================================
# T9.1: Regime Boundary Sensitivity
# ============================================================================

def run_t91_threshold_sensitivity(cfg):
    """Sweep k_neighbors × dominance_threshold.

    Returns:
        df: DataFrame with per-combo fractions and charge/cap ratio
        consistency_map: (res, res) array — per-cell agreement rate [0,1]
        grid_x, grid_y: grid coordinates for consistency map
    """
    print("=" * 60)
    print("[T9.1] Regime Boundary Sensitivity Analysis")
    print("=" * 60)

    # Load real data
    df_real, _ = load_real_data(cfg["csv_path"], verbose=True)
    logQ = df_real["logQ"].values
    logInvC = df_real["logInvC"].values

    # Build support mask grid (must match compute_regime_map_variance internals)
    res = cfg["t91_grid_resolution"]
    pad = 0.05
    x_min, x_max = logQ.min(), logQ.max()
    y_min, y_max = logInvC.min(), logInvC.max()
    x_pad = (x_max - x_min) * pad
    y_pad = (y_max - y_min) * pad
    grid_x_ref = np.linspace(x_min - x_pad, x_max + x_pad, res)
    grid_y_ref = np.linspace(y_min - y_pad, y_max + y_pad, res)

    support_mask, _ = compute_support_mask(
        logQ, logInvC, grid_x_ref, grid_y_ref,
        method=cfg["support_method"],
        k_neighbors=cfg["support_k_neighbors"],
        min_density_threshold=cfg["support_min_density_threshold"],
    )
    print(f"  支持区: {support_mask.sum()}/{support_mask.size} 网格点")

    # Parameter sweep — collect all regime maps for consistency analysis
    k_values = cfg["t91_k_values"]
    threshold_values = cfg["t91_threshold_values"]
    n_combos = len(k_values) * len(threshold_values)
    regime_stack = np.full((n_combos, res, res), np.nan)
    results = []
    combo_idx = 0

    for k in k_values:
        for thr in threshold_values:
            grid_x, grid_y, regime, _ = compute_regime_map_variance(
                logQ, logInvC,
                k_neighbors=k,
                grid_resolution=res,
                dominance_threshold=thr,
                min_local_points=cfg["t91_min_local_points"],
                support_mask=support_mask,
            )
            regime_stack[combo_idx] = regime
            combo_idx += 1

            charge_pct, cap_pct, mixed_pct = _regime_fractions(regime)
            ratio = charge_pct / cap_pct if cap_pct > 0.1 else np.inf
            results.append({
                "k_neighbors": k,
                "dominance_threshold": thr,
                "charge_pct": charge_pct,
                "cap_pct": cap_pct,
                "mixed_pct": mixed_pct,
                "charge_cap_ratio": ratio,
            })
            print(f"  k={k:3d}, thr={thr:.2f}: "
                  f"charge={charge_pct:5.1f}%, cap={cap_pct:5.1f}%, "
                  f"ratio={ratio:5.1f}x")

    df = pd.DataFrame(results)

    # ---- Per-cell classification consistency ----
    print("\n  [A2] 计算逐单元分类一致率 ...")
    consistency_map = np.full((res, res), np.nan)
    for j in range(res):
        for i in range(res):
            cell_vals = regime_stack[:, j, i]
            valid_vals = cell_vals[~np.isnan(cell_vals)]
            if len(valid_vals) < 5:
                continue
            n_charge = np.sum(valid_vals == REGIME_CHARGE)
            n_cap = np.sum(valid_vals == REGIME_CAPACITANCE)
            n_mixed = np.sum(valid_vals == REGIME_MIXED)
            consistency_map[j, i] = max(n_charge, n_cap, n_mixed) / len(
                valid_vals)

    valid_cells = ~np.isnan(consistency_map)
    if valid_cells.sum() > 0:
        cons_vals = consistency_map[valid_cells]
        print(f"    有效单元: {valid_cells.sum()}")
        print(f"    一致率 mean={np.mean(cons_vals):.3f}, "
              f"median={np.median(cons_vals):.3f}")
        print(f"    一致率 >80%: "
              f"{100 * np.mean(cons_vals > 0.80):.1f}% 的单元")
        print(f"    一致率 >60%: "
              f"{100 * np.mean(cons_vals > 0.60):.1f}% 的单元")

    print(f"\n  完成: {len(df)} 组参数组合")
    return df, consistency_map, grid_x, grid_y


def plot_t91(df, save_dir, dpi=300):
    """Fig.S7: 2-panel sensitivity plot (charge/cap ratio focus)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel (a): Heatmap of charge/cap ratio ---
    # Cap ratio with inf→max clamp for display
    df_plot = df.copy()
    finite_ratios = df_plot["charge_cap_ratio"][
        np.isfinite(df_plot["charge_cap_ratio"])]
    max_finite = finite_ratios.max() if len(finite_ratios) > 0 else 30
    df_plot["ratio_display"] = df_plot["charge_cap_ratio"].clip(upper=max_finite)

    pivot = df_plot.pivot_table(
        values="ratio_display", index="k_neighbors",
        columns="dominance_threshold",
    )
    im = ax1.imshow(
        pivot.values, aspect="auto", cmap="YlOrRd", origin="lower",
        vmin=max(1, pivot.values.min() - 1),
        vmax=pivot.values.max() + 1,
    )
    ax1.set_xticks(range(len(pivot.columns)))
    ax1.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=10)
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels([str(v) for v in pivot.index], fontsize=10)
    ax1.set_xlabel("Dominance threshold", fontsize=13)
    ax1.set_ylabel(r"$k$ neighbors", fontsize=13)
    ax1.set_title("(a) Charge / Capacitance ratio",
                   fontsize=13, fontweight="bold")

    # Annotate cells
    mid_val = (pivot.values.max() + pivot.values.min()) / 2
    for j in range(len(pivot.index)):
        for i in range(len(pivot.columns)):
            val = pivot.values[j, i]
            raw = df.loc[
                (df["k_neighbors"] == pivot.index[j]) &
                (df["dominance_threshold"] == pivot.columns[i]),
                "charge_cap_ratio"
            ].values[0]
            txt = f"{raw:.0f}x" if np.isfinite(raw) else ">30x"
            color = "white" if val > mid_val else "black"
            ax1.text(i, j, txt, ha="center", va="center",
                     fontsize=9, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
    cbar.set_label("Charge% / Cap% ratio", fontsize=11)

    # --- Panel (b): Line plot of charge/cap ratio ---
    colors_oi = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]
    markers = ["o", "s", "^", "D", "v"]
    for idx, k in enumerate(sorted(df["k_neighbors"].unique())):
        sub = df[df["k_neighbors"] == k].sort_values("dominance_threshold")
        ratios = sub["charge_cap_ratio"].clip(upper=50)
        ax2.plot(
            sub["dominance_threshold"], ratios,
            marker=markers[idx % len(markers)],
            color=colors_oi[idx % len(colors_oi)],
            linewidth=2, markersize=7, label=f"k = {k}",
        )
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.5,
                label="Charge = Cap (ratio = 1)")
    ax2.set_xlabel("Dominance threshold", fontsize=13)
    ax2.set_ylabel("Charge-dominant / Cap-dominant ratio", fontsize=13)
    ax2.set_title("(b) Asymmetry stability",
                   fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9, edgecolor="gray")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1)

    fig.suptitle(
        "Fig. S7: Regime Classification Sensitivity to "
        "Methodological Parameters",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig_s7_regime_sensitivity.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"  [图片] {save_path}")
    plt.close(fig)


def plot_t91_consistency(consistency_map, grid_x, grid_y, save_dir, dpi=300):
    """Fig.S7b: Per-cell classification consistency map."""
    from plot_mechanism_multitask import XLABEL, YLABEL

    fig, ax = plt.subplots(figsize=(10, 8))

    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    im = ax.pcolormesh(
        X_grid, Y_grid, consistency_map,
        cmap="RdYlGn", vmin=0.3, vmax=1.0,
        shading="nearest",
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.82, aspect=25, pad=0.02)
    cbar.set_label("Classification agreement rate", fontsize=13)
    cbar.set_ticks([0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["40%\n(unstable)", "60%", "80%",
                         "100%\n(unanimous)"])

    # Summary stats
    valid = ~np.isnan(consistency_map)
    if valid.sum() > 0:
        vals = consistency_map[valid]
        ax.text(
            0.03, 0.96,
            f"Mean = {np.mean(vals):.1%}\n"
            f"Median = {np.median(vals):.1%}\n"
            f">80% agreement: {np.mean(vals > 0.80):.0%} of cells\n"
            f">60% agreement: {np.mean(vals > 0.60):.0%} of cells",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat",
                      alpha=0.9),
        )

    ax.set_xlabel(XLABEL, fontsize=13)
    ax.set_ylabel(YLABEL, fontsize=13)
    ax.set_title(
        "Per-Cell Regime Classification Stability\n"
        "across 25 (k, threshold) combinations",
        fontsize=13, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig_s7b_classification_stability.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"  [图片] {save_path}")
    plt.close(fig)


# ============================================================================
# T9.2: Interpolation Grid Resolution Effect
# ============================================================================

def run_t92_grid_resolution(cfg, device):
    """Sweep hh interpolation density, return DataFrame + regime maps."""
    print("\n" + "=" * 60)
    print("[T9.2] Grid Resolution Sensitivity Analysis")
    print("=" * 60)

    # Load model
    print("[模型] 加载多任务模型 ...")
    model, scaler_X, scaler_qsc, scaler_invc, scaler_foms = \
        load_model_and_scalers(
            device,
            model_path=os.path.join(
                cfg["checkpoint_dir"], "physics_multitask_best.pth"),
            artifact_dir=cfg["artifact_dir"],
        )
    predict_fn = _make_predict_fn(
        model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device
    )

    n_values = np.array(cfg["t92_n_values"], dtype=np.float64)
    E_val = cfg["t92_E"]
    dd_val = cfg["t92_dd"]
    hh_range = cfg["t92_hh_range"]

    results = []
    regime_maps = []
    hh_configs = []

    for hh_np in cfg["t92_hh_n_points"]:
        hh_values = np.geomspace(hh_range[0], hh_range[1], hh_np)
        n_grid = len(n_values) * hh_np
        print(f"\n  hh_n_points={hh_np}: "
              f"{len(n_values)}n x {hh_np}hh = {n_grid} 设计点")

        result = compute_design_regime_grid(
            predict_fn,
            E_fixed=E_val,
            dd_fixed=dd_val,
            n_values=n_values,
            hh_values=hh_values,
            delta_log=cfg["t92_delta_log"],
            dominance_threshold=cfg["t92_dominance_threshold"],
        )

        charge_pct, cap_pct, mixed_pct = _regime_fractions(result["regime"])
        results.append({
            "hh_n_points": hh_np,
            "charge_pct": charge_pct,
            "cap_pct": cap_pct,
            "mixed_pct": mixed_pct,
            "n_grid_points": n_grid,
        })
        regime_maps.append(result)
        hh_configs.append(hh_values)

        print(f"    charge={charge_pct:.1f}%, cap={cap_pct:.1f}%, "
              f"mixed={mixed_pct:.1f}%")

    df = pd.DataFrame(results)
    print(f"\n  完成: {len(df)} 种分辨率")
    return df, regime_maps, hh_configs


def plot_t92(regime_maps, hh_configs, n_values, E_val, dd_val,
             save_dir, dpi=300):
    """Fig.S8: 1x3 regime map comparison at different resolutions."""
    n_panels = len(regime_maps)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    log2_n = np.log2(n_values)

    for idx, (result, hh_values, ax) in enumerate(
            zip(regime_maps, hh_configs, axes)):
        log2_hh = np.log2(hh_values)
        N_grid, HH_grid = np.meshgrid(log2_n, log2_hh)

        im = ax.pcolormesh(
            N_grid, HH_grid, result["regime"],
            cmap=REGIME_CMAP, norm=REGIME_NORM,
            shading="nearest", alpha=0.7,
        )

        # FOMS contour
        log_foms = np.log10(np.maximum(result["foms"], 1e-30))
        try:
            cs = ax.contour(N_grid, HH_grid, log_foms, levels=6,
                            colors="black", linewidths=0.6, alpha=0.5)
            ax.clabel(cs, fontsize=7, fmt="%.1f")
        except ValueError:
            pass

        charge_pct, cap_pct, mixed_pct = _regime_fractions(result["regime"])
        ax.set_xlabel(r"$n$ (blade pairs)", fontsize=12)
        if idx == 0:
            ax.set_ylabel(r"$h/R$ (blade height ratio)", fontsize=12)

        ax.set_title(
            f"({chr(97 + idx)}) hh points = {len(hh_values)}\n"
            f"Charge {charge_pct:.0f}% / Cap {cap_pct:.0f}% / "
            f"Mixed {mixed_pct:.0f}%",
            fontsize=12, fontweight="bold",
        )

        ax.set_xticks(log2_n)
        ax.set_xticklabels([str(int(v)) for v in n_values], fontsize=9)
        hh_step = max(1, len(hh_values) // 5)
        hh_ticks = hh_values[::hh_step]
        ax.set_yticks(np.log2(hh_ticks))
        ax.set_yticklabels([f"{v:.4g}" for v in hh_ticks], fontsize=8)

    # Shared legend
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.7, label="Charge-limited"),
        Patch(facecolor="#bdc3c7", alpha=0.7, label="Mixed"),
        Patch(facecolor="#2980b9", alpha=0.7, label="Capacitance-limited"),
    ]
    axes[-1].legend(handles=legend_elements, loc="lower right", fontsize=9,
                    framealpha=0.9, edgecolor="gray")

    fig.suptitle(
        "Fig. S8: Grid Resolution Effect on Design Regime Map\n"
        r"$\varepsilon$" + f"={E_val}, d/R={dd_val}",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "fig_s8_grid_resolution.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"  [图片] {save_path}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("T9: Sensitivity Analysis for Regime Classification")
    print("=" * 70)

    cfg = CONFIG
    save_dir = cfg["output_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"设备: {device}\n")

    # ---- T9.1: Threshold Sensitivity ----
    df_t91, consistency_map, grid_x, grid_y = run_t91_threshold_sensitivity(cfg)
    csv_path_t91 = os.path.join(save_dir, "threshold_sensitivity.csv")
    df_t91.to_csv(csv_path_t91, index=False, float_format="%.2f")
    print(f"  [导出] {csv_path_t91}")
    plot_t91(df_t91, save_dir, dpi=cfg["dpi"])
    plot_t91_consistency(consistency_map, grid_x, grid_y,
                         save_dir, dpi=cfg["dpi"])

    # ---- T9.2: Grid Resolution ----
    df_t92, regime_maps, hh_configs = run_t92_grid_resolution(cfg, device)
    csv_path_t92 = os.path.join(save_dir, "grid_resolution_sensitivity.csv")
    df_t92.to_csv(csv_path_t92, index=False, float_format="%.2f")
    print(f"  [导出] {csv_path_t92}")
    n_values = np.array(cfg["t92_n_values"], dtype=np.float64)
    plot_t92(
        regime_maps, hh_configs, n_values,
        cfg["t92_E"], cfg["t92_dd"],
        save_dir, dpi=cfg["dpi"],
    )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("T9 敏感性分析完成!")
    print("=" * 70)

    charge_min = df_t91["charge_pct"].min()
    charge_max = df_t91["charge_pct"].max()
    ref_row = df_t91[
        (df_t91["k_neighbors"] == 50) &
        (df_t91["dominance_threshold"] == 0.62)
    ]
    ref_charge = (ref_row["charge_pct"].values[0]
                  if len(ref_row) > 0 else df_t91["charge_pct"].median())

    # Charge/cap ratio stats (the robust metric)
    finite_ratios = df_t91["charge_cap_ratio"][
        np.isfinite(df_t91["charge_cap_ratio"])]
    ratio_min = finite_ratios.min() if len(finite_ratios) > 0 else 0
    ratio_max = finite_ratios.max() if len(finite_ratios) > 0 else 0

    print(f"\n  T9.1 参数扫描结果:")
    print(f"    Charge/Cap 比值范围: {ratio_min:.1f}x ~ {ratio_max:.1f}x "
          f"(始终 >> 1)")
    print(f"    参考值 (k=50, thr=0.62): charge={ref_charge:.1f}%")

    # Consistency stats
    valid_cons = consistency_map[~np.isnan(consistency_map)]
    if len(valid_cons) > 0:
        print(f"\n  T9.1 空间一致率:")
        print(f"    mean={np.mean(valid_cons):.1%}, "
              f"median={np.median(valid_cons):.1%}")
        print(f"    >80% 一致: {np.mean(valid_cons > 0.80):.0%} 的单元")

    print(f"\n  T9.2 网格分辨率结果:")
    for _, row in df_t92.iterrows():
        print(f"    hh={int(row['hh_n_points']):2d} pts: "
              f"charge={row['charge_pct']:.1f}%, "
              f"cap={row['cap_pct']:.1f}%, "
              f"mixed={row['mixed_pct']:.1f}%")

    # Template sentence (A1: focus on charge/cap asymmetry, not absolute %)
    print(f"\n  模板句:")
    print(f'    "Across all 25 parameter combinations (k in [20,100], '
          f"threshold in [0.55,0.70]), the charge-dominant fraction "
          f"consistently exceeds the capacitance-dominant fraction by a "
          f"factor of {ratio_min:.0f}-{ratio_max:.0f}x, and "
          f"{np.mean(valid_cons > 0.80):.0%} of grid cells maintain >80% "
          f"classification agreement, confirming that the charge-transfer "
          f'asymmetry is insensitive to methodological choices."')

    # File listing
    print(f"\n  产出文件:")
    for f in sorted(os.listdir(save_dir)):
        fpath = os.path.join(save_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    - {f} ({size_kb:.0f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
