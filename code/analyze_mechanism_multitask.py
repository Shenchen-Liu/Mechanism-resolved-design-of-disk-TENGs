#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main mechanism-analysis entry point.

This script combines the trained multitask surrogate with the processed disk
TENG dataset to build mechanism maps, design-regime maps, and robustness maps.
"""

import os
import numpy as np
import pandas as pd
import torch

# 本模块
from utils_mechanism_multitask import (
    safe_log10,
    load_real_data,
    compute_support_mask,
    compute_regime_map_variance,
    compute_design_regime_grid,
    compute_robustness_grid,
    print_regime_statistics,
    print_global_correlations,
    export_analysis_table,
)
from plot_mechanism_multitask import (
    plot_real_landscape,
    plot_regime_map,
    plot_overlay,
    plot_model_landscape,
    plot_consistency,
    plot_design_regime,
    plot_robustness,
    CBAR_FOMS,
    CBAR_FOMS_NORM_N,
)

# 多任务模型推理
from predict_multitask_physics import load_model_and_scalers, predict_batch


# ============================================================================
# 配置参数 — 集中管理
# ============================================================================

CONFIG = {
    # ---- 数据路径 (相对于 code/) ----
    "csv_path": "../data/disk_teng_training_processed.csv",
    "output_dir": "../outputs_mechanism_multitask",

    # ---- 模型路径 ----
    "checkpoint_dir": "../checkpoints_multitask_physics",
    "artifact_dir": "../artifacts_multitask_physics",

    # ---- 图片设置 ----
    "dpi": 300,
    "fig_size_landscape": (11, 8.5),
    "fig_size_regime": (10, 8),
    "fig_size_consistency": (8, 8),
    "fig_size_design": (10, 8),

    # ---- zoom 设置 ----
    "zoom_logQ_min": -18.5,

    # ---- 机制空间 regime 分析参数 (方差占比法) ----
    "regime_k_neighbors": 50,
    "regime_grid_resolution": 50,
    "regime_dominance_threshold": 0.62,  # >0.62 = charge, <0.38 = cap
    "regime_min_local_points": 10,
g ----
    "support_method": "knn",
    "support_k_neighbors": 15,
    "support_min_density_threshold": 5,

    # ---- 模型增强网格 ----
    "model_n_values": [2, 4, 8, 16, 32, 64],
    "model_E_values": [1, 2, 3, 5, 7, 10],
    "model_dd_geomspace": (0.03125, 1.0, 12),
    "model_hh_geomspace": (0.00390625, 1.0, 18),

    # ---- 设计空间 regime 参数 ----
    "design_n_values": [2, 4, 8, 16, 32, 64],
    "design_hh_geomspace": (0.00390625, 1.0, 18),
    # 展示多组 (E, dd) 条件
    "design_scenarios": [
        {"E": 1.0, "dd": 0.125, "label": "low_E_small_d"},
        {"E": 3.0, "dd": 0.125, "label": "mid_E_small_d"},
        {"E": 10.0, "dd": 0.125, "label": "high_E_small_d"},
        {"E": 3.0, "dd": 0.5, "label": "mid_E_large_d"},
    ],
    "design_delta_log": 0.02,
    "design_dominance_threshold": 0.62,

    # ---- 鲁棒性分析参数 ----
    "robustness_perturb_frac": 0.10,  # ±10%
}


# ============================================================================
# 模型预测函数封装
# ============================================================================

def _make_predict_fn(model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device):
    """
    将模型封装为简洁的预测函数, 供 design regime 和 robustness 分析使用。

    Returns:
        predict_fn: callable(n_arr, E_arr, dd_arr, hh_arr) -> dict
    """
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
# 模型增强数据生成
# ============================================================================

def generate_multitask_predictions(cfg, device):
    """
    用多任务模型在稠密参数网格上生成预测。

    输出所有量 (Qsc, invC, FOMS_direct, FOMS_phys) 均来自同一模型，
    不混合插值。
    """
    print("[模型] 加载多任务模型 ...")
    model, scaler_X, scaler_qsc, scaler_invc, scaler_foms = load_model_and_scalers(
        device,
        model_path=os.path.join(cfg["checkpoint_dir"], "physics_multitask_best.pth"),
        artifact_dir=cfg["artifact_dir"],
    )

    n_values = np.array(cfg["model_n_values"])
    E_values = np.array(cfg["model_E_values"])
    dd_values = np.geomspace(*cfg["model_dd_geomspace"])
    hh_values = np.geomspace(*cfg["model_hh_geomspace"])

    grid_list = []
    for n in n_values:
        for E in E_values:
            for dd in dd_values:
                for hh in hh_values:
                    grid_list.append([n, E, dd, hh])

    df_grid = pd.DataFrame(grid_list, columns=["n", "E", "dd", "hh"])
    print(f"  稠密网格: {len(df_grid)} 点")

    df_pred = predict_batch(
        df_grid, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device,
    )

    valid = (
        (df_pred["Qsc_MACRS_pred"] > 0) &
        (df_pred["invC_sum_pred"] > 0) &
        (df_pred["FOMS_direct_pred"] > 0)
    )
    df_pred = df_pred[valid].copy().reset_index(drop=True)
    print(f"  有效预测点: {len(df_pred)}")

    df_pred["logQ_pred"] = safe_log10(df_pred["Qsc_MACRS_pred"].values ** 2)
    df_pred["logInvC_pred"] = safe_log10(df_pred["invC_sum_pred"].values)

    return df_pred, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Mechanism analysis")
    print("  (landscape, regime map, design map, robustness)")
    print("=" * 70)

    cfg = CONFIG
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"设备: {device}\n")
    print("[1/9] Load processed dataset ...")
    df_real, epsilon_col = load_real_data(cfg["csv_path"], verbose=True)

    logQ = df_real["logQ"].values
    logInvC = df_real["logInvC"].values
    foms = df_real["FOMS"].values

    zoom_xlim = (cfg["zoom_logQ_min"], logQ.max() + 0.2)

    print("\n[2/9] Export ground-truth mechanism landscape ...")

    plot_real_landscape(
        logQ, logInvC, foms,
        save_path=os.path.join(output_dir, "real_mechanism_landscape_full.png"),
        color_label=CBAR_FOMS,
        title="Mechanism Landscape from Ground-Truth TENG Data (Full)",
        zoom_xlim=None, dpi=cfg["dpi"], figsize=cfg["fig_size_landscape"],
    )
    plot_real_landscape(
        logQ, logInvC, foms,
        save_path=os.path.join(output_dir, "real_mechanism_landscape_zoom.png"),
        color_label=CBAR_FOMS,
        title="Mechanism Landscape — High-Information Region (Zoom)",
        zoom_xlim=zoom_xlim, dpi=cfg["dpi"], figsize=cfg["fig_size_landscape"],
    )

    print("\n[3/9] Build support mask ...")

    pad = 0.05
    x_min, x_max = logQ.min(), logQ.max()
    y_min, y_max = logInvC.min(), logInvC.max()
    x_pad = (x_max - x_min) * pad
    y_pad = (y_max - y_min) * pad
    res = cfg["regime_grid_resolution"]

    grid_x_ref = np.linspace(x_min - x_pad, x_max + x_pad, res)
    grid_y_ref = np.linspace(y_min - y_pad, y_max + y_pad, res)

    support_mask, _density = compute_support_mask(
        logQ, logInvC,
        grid_x_ref, grid_y_ref,
        method=cfg["support_method"],
        k_neighbors=cfg["support_k_neighbors"],
        min_density_threshold=cfg["support_min_density_threshold"],
    )
    n_supported = support_mask.sum()
    n_total_cells = support_mask.size
    print(f"  支持区: {n_supported}/{n_total_cells} 网格点 "
          f"({100 * n_supported / n_total_cells:.1f}%)")

    print("\n[4/9] Compute mechanism-space regime map ...")
    print("  Using variance-fraction classification on log(FOMS/n).")

    grid_x, grid_y, regime, f_charge_map = compute_regime_map_variance(
        logQ, logInvC,
        k_neighbors=cfg["regime_k_neighbors"],
        grid_resolution=cfg["regime_grid_resolution"],
        dominance_threshold=cfg["regime_dominance_threshold"],
        min_local_points=cfg["regime_min_local_points"],
        support_mask=support_mask,
    )

    regime_stats = print_regime_statistics(regime, label="FOMS/n")

    foms_plot = df_real["FOMS_norm_n"].values

    plot_regime_map(
        grid_x, grid_y, regime,
        save_path=os.path.join(output_dir, "regime_map_norm_n_full.png"),
        title="Variance-Fraction Regime — FOMS/n (Full)",
        zoom_xlim=None, dpi=cfg["dpi"], figsize=cfg["fig_size_regime"],
    )
    plot_regime_map(
        grid_x, grid_y, regime,
        save_path=os.path.join(output_dir, "regime_map_norm_n_zoom.png"),
        title="Variance-Fraction Regime — FOMS/n (Zoom)",
        zoom_xlim=zoom_xlim, dpi=cfg["dpi"], figsize=cfg["fig_size_regime"],
    )

    plot_overlay(
        logQ, logInvC, foms_plot,
        grid_x, grid_y, regime,
        save_path=os.path.join(output_dir, "real_overlay_norm_n_full.png"),
        color_label=CBAR_FOMS_NORM_N,
        title="Mechanism Landscape + Regime — FOMS/n (Full)",
        zoom_xlim=None, dpi=cfg["dpi"], figsize=cfg["fig_size_landscape"],
    )
    plot_overlay(
        logQ, logInvC, foms_plot,
        grid_x, grid_y, regime,
        save_path=os.path.join(output_dir, "real_overlay_norm_n_zoom.png"),
        color_label=CBAR_FOMS_NORM_N,
        title="Mechanism Landscape + Regime — FOMS/n (Zoom)",
        zoom_xlim=zoom_xlim, dpi=cfg["dpi"], figsize=cfg["fig_size_landscape"],
    )

    print("\n[5/9] Export model-based landscape maps ...")

    model = scaler_X = scaler_qsc = scaler_invc = scaler_foms = None
    predict_fn = None

    try:
        (df_pred, model, scaler_X, scaler_qsc,
         scaler_invc, scaler_foms) = generate_multitask_predictions(cfg, device)

        predict_fn = _make_predict_fn(
            model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device
        )

        logQ_model = df_pred["logQ_pred"].values
        logInvC_model = df_pred["logInvC_pred"].values
        foms_direct_model = df_pred["FOMS_direct_pred"].values
        foms_phys_model = df_pred["FOMS_phys_pred"].values

        for suffix, xlim in [("full", None), ("zoom", zoom_xlim)]:
            plot_model_landscape(
                logQ_model, logInvC_model, foms_direct_model,
                logQ, logInvC, foms,
                save_path=os.path.join(
                    output_dir, f"multitask_landscape_foms_direct_{suffix}.png"),
                color_label=CBAR_FOMS,
                title=f"Multi-Task Model: FOMS_direct ({suffix.title()})",
                zoom_xlim=xlim, dpi=cfg["dpi"], figsize=cfg["fig_size_landscape"],
            )

        for suffix, xlim in [("full", None), ("zoom", zoom_xlim)]:
            plot_model_landscape(
                logQ_model, logInvC_model, foms_phys_model,
                logQ, logInvC, foms,
                save_path=os.path.join(
                    output_dir, f"multitask_landscape_foms_phys_{suffix}.png"),
                color_label=CBAR_FOMS,
                title=f"Physics-Reconstructed Landscape ({suffix.title()})",
                zoom_xlim=xlim, dpi=cfg["dpi"], figsize=cfg["fig_size_landscape"],
            )

        print("\n[6/9] Export consistency plot ...")
        plot_consistency(
            foms_direct_model, foms_phys_model,
            save_path=os.path.join(
                output_dir, "multitask_direct_vs_phys_consistency.png"),
            dpi=cfg["dpi"], figsize=cfg["fig_size_consistency"],
        )

    except Exception as e:
        print(f"\n  [WARNING] model-based landscape export failed: {e}")
        import traceback
        traceback.print_exc()
        print("  Ground-truth analysis outputs are still valid.")

    print("\n[7/9] Export summary table and global trends ...")

    export_analysis_table(
        df_real, epsilon_col,
        output_path=os.path.join(output_dir, "mechanism_analysis_table.csv"),
    )
    print_global_correlations(df_real)

    print("\n" + "=" * 60)
    print("Mechanism-space regime summary")
    print("=" * 60)
    if regime_stats is not None:
        print(f"  Charge-dominant : {regime_stats['pct_charge']:.1f}%")
        print(f"  Cap-dominant    : {regime_stats['pct_capacitance']:.1f}%")
        print(f"  Mixed           : {regime_stats['pct_mixed']:.1f}%")
        balance = abs(regime_stats["pct_charge"] - regime_stats["pct_capacitance"])
        print(f"  |Charge% - Cap%| = {balance:.1f}pp")

    print("\n[8/9] Export design-space regime maps ...")

    if predict_fn is not None:
        n_design = np.array(cfg["design_n_values"], dtype=np.float64)
        hh_design = np.geomspace(*cfg["design_hh_geomspace"])

        for scenario in cfg["design_scenarios"]:
            E_val = scenario["E"]
            dd_val = scenario["dd"]
            label = scenario["label"]

            print(f"\n  --- E={E_val}, dd={dd_val} ({label}) ---")

            result = compute_design_regime_grid(
                predict_fn,
                E_fixed=E_val,
                dd_fixed=dd_val,
                n_values=n_design,
                hh_values=hh_design,
                delta_log=cfg["design_delta_log"],
                dominance_threshold=cfg["design_dominance_threshold"],
            )

            regime_flat = result["regime"].flatten()
            valid_regime = regime_flat[~np.isnan(regime_flat)]
            n_total = len(valid_regime)
            if n_total > 0:
                n_charge = np.sum(valid_regime == 1)
                n_cap = np.sum(valid_regime == -1)
                n_mixed = np.sum(valid_regime == 0)
                print(f"    Charge-limited:      {n_charge:4d} ({100*n_charge/n_total:.1f}%)")
                print(f"    Capacitance-limited: {n_cap:4d} ({100*n_cap/n_total:.1f}%)")
                print(f"    Mixed:               {n_mixed:4d} ({100*n_mixed/n_total:.1f}%)")

            plot_design_regime(
                n_design, hh_design,
                result["foms"], result["regime"], result["f_charge"],
                E_val, dd_val,
                save_path=os.path.join(
                    output_dir, f"design_regime_{label}.png"),
                dpi=cfg["dpi"], figsize=cfg["fig_size_design"],
            )
    else:
        print("  [SKIP] model not available")

    print("\n[9/9] Export robustness maps ...")

    if predict_fn is not None:
        perturb_frac = cfg["robustness_perturb_frac"]
        perturb_pct = int(perturb_frac * 100)

        for scenario in cfg["design_scenarios"]:
            E_val = scenario["E"]
            dd_val = scenario["dd"]
            label = scenario["label"]

            print(f"\n  --- E={E_val}, dd={dd_val} ({label}) ---")

            foms_map, cv_map, worst_ratio_map = compute_robustness_grid(
                predict_fn,
                E_fixed=E_val,
                dd_fixed=dd_val,
                n_values=n_design,
                hh_values=hh_design,
                perturb_frac=perturb_frac,
            )

            cv_pct = cv_map * 100
            print(f"    CV 统计: median={np.median(cv_pct):.2f}%, "
                  f"max={np.max(cv_pct):.2f}%, "
                  f"<5%区域占比={100*np.mean(cv_pct < 5):.1f}%")
            print(f"    Worst-case retention: "
                  f"median={np.median(worst_ratio_map)*100:.1f}%, "
                  f"min={np.min(worst_ratio_map)*100:.1f}%")

            plot_robustness(
                n_design, hh_design,
                cv_map, foms_map, worst_ratio_map,
                E_val, dd_val, perturb_pct,
                save_path=os.path.join(
                    output_dir, f"robustness_{label}.png"),
                dpi=cfg["dpi"], figsize=cfg["fig_size_design"],
            )
    else:
        print("  [SKIP] model not available")

    print("\n" + "=" * 70)
    print("Analysis finished")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  - {f} ({size_kb:.0f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
