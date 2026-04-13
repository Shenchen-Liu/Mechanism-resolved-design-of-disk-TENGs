#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务物理一致模型 - 推理脚本
========================================================
加载训练好的 PhysicsMultiTaskTransformer 模型，
输入结构参数 (n, E, dd, hh)，输出:
  1. Qsc_MACRS
  2. invC_sum
  3. FOMS_direct
  4. FOMS_phys

与原版 main.py 的推理流程完全独立。
为后续 Web 系统预留函数化接口。

使用方法:
    python predict_multitask_physics.py --n 4 --E 3.4 --dd 0.25 --hh 0.125
    python predict_multitask_physics.py --batch_file input.csv

"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils_multitask_physics import (
    load_scalers_multitask,
    compute_foms_phys,
    get_device,
    compute_metrics,
    compute_consistency_metrics,
)
from model_multitask_physics import PhysicsMultiTaskTransformer


# ============================================================================
# 路径配置（相对于 code/ 目录）
# ============================================================================

CHECKPOINT_DIR = "../checkpoints_multitask_physics"
ARTIFACT_DIR = "../artifacts_multitask_physics"
DEFAULT_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "physics_multitask_best.pth")


# ============================================================================
# 核心推理函数（Web API 友好接口）
# ============================================================================

def predict_single(
    n, E, dd, hh,
    model=None,
    scaler_X=None,
    scaler_qsc=None,
    scaler_invc=None,
    scaler_foms=None,
    device=None,
):
    """
    单样本推理：输入结构参数，输出多任务预测结果。

    此函数为 Web API 预留的标准接口，可直接被 Flask/FastAPI 调用。

    Args:
        n:    叶片对数 (无量纲整数)
        E:    介电常数 (无量纲)
        dd:   间隙比例 (无量纲)
        hh:   高度比例 (无量纲)
        model:      已加载的模型实例（若为 None 则自动加载）
        scaler_X:   特征归一化器
        scaler_qsc: Qsc 目标归一化器
        scaler_invc: invC 目标归一化器
        scaler_foms: FOMS 目标归一化器
        device:     计算设备

    Returns:
        dict: {
            'Qsc_MACRS':    float,  # 短路电荷差 (C)
            'invC_sum':     float,  # 倒电容之和 (1/F)
            'FOMS_direct':  float,  # 模型直接预测的 FOMS
            'FOMS_phys':    float,  # 物理公式重构的 FOMS
        }
    """
    if device is None:
        device = get_device()

    # 自动加载模型和 scaler（懒加载）
    if model is None or scaler_X is None:
        model, scaler_X, scaler_qsc, scaler_invc, scaler_foms = load_model_and_scalers(device)

    model.eval()

    # 准备输入
    X_raw = np.array([[n, E, dd, hh]], dtype=np.float64)
    X_scaled = scaler_X.transform(X_raw)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(X_tensor)

    # 逆归一化到 log10 值，再 10**x 得到原始物理量级
    pred_qsc = 10.0 ** scaler_qsc.inverse_transform(
        outputs["pred_qsc"].cpu().numpy()
    )[0, 0]
    pred_invc = 10.0 ** scaler_invc.inverse_transform(
        outputs["pred_invc_sum"].cpu().numpy()
    )[0, 0]
    pred_foms_direct = 10.0 ** scaler_foms.inverse_transform(
        outputs["pred_foms_direct"].cpu().numpy()
    )[0, 0]

    # 计算 FOMS_phys（参考 calculate_foms.py 的物理组合公式）
    pred_foms_phys = compute_foms_phys(pred_qsc, pred_invc, n)

    return {
        "Qsc_MACRS": float(pred_qsc),
        "invC_sum": float(pred_invc),
        "FOMS_direct": float(pred_foms_direct),
        "FOMS_phys": float(pred_foms_phys),
    }


def predict_batch(
    df_input,
    model=None,
    scaler_X=None,
    scaler_qsc=None,
    scaler_invc=None,
    scaler_foms=None,
    device=None,
):
    """
    批量推理：输入 DataFrame (含 n, E, dd, hh 列)，返回预测结果。

    Args:
        df_input: pd.DataFrame, 必须包含列 ['n', 'E', 'dd', 'hh']
        model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device: 同上

    Returns:
        pd.DataFrame: 输入 + 预测结果
    """
    if device is None:
        device = get_device()

    if model is None or scaler_X is None:
        model, scaler_X, scaler_qsc, scaler_invc, scaler_foms = load_model_and_scalers(device)

    model.eval()

    feature_cols = ["n", "E", "dd", "hh"]
    X_raw = df_input[feature_cols].values
    X_scaled = scaler_X.transform(X_raw)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)

    # 逆归一化到 log10 值，再 10**x 得到原始物理量级
    pred_qsc = 10.0 ** scaler_qsc.inverse_transform(outputs["pred_qsc"].cpu().numpy())
    pred_invc = 10.0 ** scaler_invc.inverse_transform(outputs["pred_invc_sum"].cpu().numpy())
    pred_foms_direct = 10.0 ** scaler_foms.inverse_transform(outputs["pred_foms_direct"].cpu().numpy())

    # 计算 FOMS_phys（参考 calculate_foms.py 的物理组合公式）
    n_values = df_input["n"].values
    pred_foms_phys = compute_foms_phys(
        pred_qsc.flatten(), pred_invc.flatten(), n_values
    )

    result = df_input.copy()
    result["Qsc_MACRS_pred"] = pred_qsc.flatten()
    result["invC_sum_pred"] = pred_invc.flatten()
    result["FOMS_direct_pred"] = pred_foms_direct.flatten()
    result["FOMS_phys_pred"] = pred_foms_phys.flatten()

    return result


def validate_ood(
    csv_path,
    model,
    scaler_X,
    scaler_qsc,
    scaler_invc,
    scaler_foms,
    device,
):
    """
    OOD 验证：加载验证集 CSV，计算预测误差，返回结果 DataFrame 和指标

    Args:
        csv_path: 验证集 CSV 路径（含 n, E, dd, hh, Qsc_MACRS, inv_C_start, inv_C_end, FOMS）
        model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device: 模型和归一化器

    Returns:
        result_df: 包含真实值、预测值、误差的 DataFrame
        metrics: 指标字典
    """
    # 加载验证集
    df = pd.read_csv(csv_path)
    print(f"\n[OOD 验证] 加载: {csv_path}, 样本数: {len(df)}")

    # 计算 invC_sum
    df["invC_sum"] = df["inv_C_start"] + df["inv_C_end"]

    # 准备输入
    df_input = df[["n", "E", "dd", "hh"]].copy()

    # 批量预测
    result = predict_batch(df_input, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device)

    # 合并真实值（适配 FOMS 或 FOMS_direct 列名）
    result["Qsc_true"] = df["Qsc_MACRS"].values
    result["invC_true"] = df["invC_sum"].values
    foms_col = "FOMS_direct" if "FOMS_direct" in df.columns else "FOMS"
    result["FOMS_direct_true"] = df[foms_col].values
    result["FOMS_phys_true"] = compute_foms_phys(df["Qsc_MACRS"].values, df["invC_sum"].values, df["n"].values)

    # 重命名预测列以统一命名
    result.rename(columns={
        "Qsc_MACRS_pred": "Qsc_pred",
        "invC_sum_pred": "invC_pred",
    }, inplace=True)

    # 计算误差
    result["Qsc_abs_error"] = np.abs(result["Qsc_pred"] - result["Qsc_true"])
    result["Qsc_rel_error_%"] = (result["Qsc_abs_error"] / np.abs(result["Qsc_true"])) * 100.0

    result["invC_abs_error"] = np.abs(result["invC_pred"] - result["invC_true"])
    result["invC_rel_error_%"] = (result["invC_abs_error"] / np.abs(result["invC_true"])) * 100.0

    result["FOMS_direct_abs_error"] = np.abs(result["FOMS_direct_pred"] - result["FOMS_direct_true"])
    result["FOMS_direct_rel_error_%"] = (result["FOMS_direct_abs_error"] / np.abs(result["FOMS_direct_true"])) * 100.0

    result["FOMS_phys_abs_error"] = np.abs(result["FOMS_phys_pred"] - result["FOMS_phys_true"])
    result["FOMS_phys_rel_error_%"] = (result["FOMS_phys_abs_error"] / np.abs(result["FOMS_phys_true"])) * 100.0

    # 计算聚合指标
    metrics = {}
    metrics["qsc"] = compute_metrics(result["Qsc_true"].values, result["Qsc_pred"].values, name="Qsc")
    metrics["invc"] = compute_metrics(result["invC_true"].values, result["invC_pred"].values, name="invC")
    metrics["foms_direct"] = compute_metrics(result["FOMS_direct_true"].values, result["FOMS_direct_pred"].values, name="FOMS_direct")
    metrics["foms_phys"] = compute_metrics(result["FOMS_phys_true"].values, result["FOMS_phys_pred"].values, name="FOMS_phys")
    metrics["consistency"] = compute_consistency_metrics(
        result["FOMS_direct_pred"].values,
        result["FOMS_phys_pred"].values,
        name="Consistency"
    )

    # 按 n 分组统计
    print("\n  === 按 n 分组误差 ===")
    for n_val in sorted(result["n"].unique()):
        subset = result[result["n"] == n_val]
        qsc_mae = subset["Qsc_abs_error"].mean()
        invc_mae = subset["invC_abs_error"].mean()
        foms_mae = subset["FOMS_direct_abs_error"].mean()
        print(f"  n={n_val:.0f}: Qsc MAE={qsc_mae:.3e}, invC MAE={invc_mae:.3e}, FOMS MAE={foms_mae:.3e}")

    # 找出最大误差样本
    worst_idx = result["FOMS_direct_rel_error_%"].idxmax()
    worst_row = result.loc[worst_idx]
    print(f"\n  [最大误差] n={worst_row['n']:.0f}, E={worst_row['E']:.2f}, "
          f"FOMS_direct 相对误差={worst_row['FOMS_direct_rel_error_%']:.2f}%")

    return result, metrics


def plot_ood_parity(results_dict, output_dir, metrics_dict=None):
    """
    绘制 OOD 验证的 parity plots with R²/MAE_log10 annotations + consistency plot

    Args:
        results_dict: {"validate1": df1, "validate2": df2}
        output_dir: 输出目录
        metrics_dict: {"validate1": metrics1, "validate2": metrics2}（可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    })

    # Okabe-Ito colorblind-safe colors (supports up to 5 datasets)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
    markers = ["o", "s", "D", "^", "v"]

    targets = [
        ("Qsc", "Qsc_true", "Qsc_pred", "Qsc_MACRS (C)", "qsc"),
        ("invC", "invC_true", "invC_pred", "invC_sum (1/F)", "invc"),
        ("FOMS_direct", "FOMS_direct_true", "FOMS_direct_pred", "FOMS_direct", "foms_direct"),
        ("FOMS_phys", "FOMS_phys_true", "FOMS_phys_pred", "FOMS_phys", "foms_phys"),
    ]

    for name, true_col, pred_col, label, metric_key in targets:
        fig, ax = plt.subplots(1, 1, figsize=(3.35, 3.35))

        annotation_lines = []
        for i, (key, df) in enumerate(results_dict.items()):
            y_true = df[true_col].values
            y_pred = df[pred_col].values
            ax.scatter(y_true, y_pred, alpha=0.6, s=40, marker=markers[i],
                       color=colors[i], label=key, edgecolors="k", linewidths=0.5)

            # Per-dataset metrics annotation
            if metrics_dict and key in metrics_dict and metric_key in metrics_dict[key]:
                m = metrics_dict[key][metric_key]
                r2 = m.get("r2", None)
                mae_l = m.get("mae_log10", None)
                parts = []
                if r2 is not None:
                    parts.append(f"R²={r2:.3f}")
                if mae_l is not None:
                    parts.append(f"MAE(log10)={mae_l:.3f}")
                if parts:
                    annotation_lines.append(f"{key}: {', '.join(parts)}")

        # Diagonal
        all_true = np.concatenate([df[true_col].values for df in results_dict.values()])
        all_pred = np.concatenate([df[pred_col].values for df in results_dict.values()])
        lo = min(all_true.min(), all_pred.min())
        hi = max(all_true.max(), all_pred.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "--", color="gray", lw=1, alpha=0.7, label="y = x")

        ax.set_xlabel(f"True {label}", fontsize=9)
        ax.set_ylabel(f"Predicted {label}", fontsize=9)
        ax.set_title(f"OOD: {name}", fontsize=10)
        ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="gray")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

        if annotation_lines:
            ax.text(0.05, 0.95, "\n".join(annotation_lines), transform=ax.transAxes,
                    fontsize=6.5, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        save_path = os.path.join(output_dir, f"ood_parity_{name.lower()}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [图] 已保存: {save_path}")

    # ---- Log-scale OOD parity plots for multi-order quantities ----
    log_targets = [
        ("Qsc", "Qsc_true", "Qsc_pred", "Qsc_MACRS (C)", "qsc"),
        ("FOMS_direct", "FOMS_direct_true", "FOMS_direct_pred", "FOMS_direct", "foms_direct"),
        ("FOMS_phys", "FOMS_phys_true", "FOMS_phys_pred", "FOMS_phys", "foms_phys"),
    ]

    for name, true_col, pred_col, label, metric_key in log_targets:
        fig, ax = plt.subplots(1, 1, figsize=(3.35, 3.35))

        annotation_lines = []
        for i, (key, df) in enumerate(results_dict.items()):
            y_true = df[true_col].values
            y_pred = df[pred_col].values
            pos = (y_true > 0) & (y_pred > 0)
            ax.scatter(y_true[pos], y_pred[pos], alpha=0.6, s=40, marker=markers[i],
                       color=colors[i], label=key, edgecolors="k", linewidths=0.5)

            if metrics_dict and key in metrics_dict and metric_key in metrics_dict[key]:
                m = metrics_dict[key][metric_key]
                r2_l = m.get("r2_log10", m.get("r2", None))
                mae_l = m.get("mae_log10", None)
                parts = []
                if r2_l is not None:
                    parts.append(f"R²(log)={r2_l:.3f}")
                if mae_l is not None:
                    parts.append(f"MAE(log10)={mae_l:.3f}")
                if parts:
                    annotation_lines.append(f"{key}: {', '.join(parts)}")

        ax.set_xscale("log")
        ax.set_yscale("log")

        all_true = np.concatenate([df[true_col].values for df in results_dict.values()])
        all_pred = np.concatenate([df[pred_col].values for df in results_dict.values()])
        pos_all = (all_true > 0) & (all_pred > 0)
        lo = min(all_true[pos_all].min(), all_pred[pos_all].min()) * 0.5
        hi = max(all_true[pos_all].max(), all_pred[pos_all].max()) * 2.0
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, alpha=0.7, label="y = x")

        ax.set_xlabel(f"True {label}", fontsize=9)
        ax.set_ylabel(f"Predicted {label}", fontsize=9)
        ax.set_title(f"OOD: {name} (log)", fontsize=10)
        ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="gray")
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(labelsize=7)

        if annotation_lines:
            ax.text(0.05, 0.95, "\n".join(annotation_lines), transform=ax.transAxes,
                    fontsize=6.5, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        save_path = os.path.join(output_dir, f"ood_parity_{name.lower()}_log.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [图] 已保存: {save_path}")

    # ---- Consistency parity plot: FOMS_direct_pred vs FOMS_phys_pred ----
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 3.35))
    annotation_lines = []
    for i, (key, df) in enumerate(results_dict.items()):
        foms_d = df["FOMS_direct_pred"].values
        foms_p = df["FOMS_phys_pred"].values
        ax.scatter(foms_d, foms_p, alpha=0.6, s=40, marker=markers[i],
                   color=colors[i], label=key, edgecolors="k", linewidths=0.5)

        if metrics_dict and key in metrics_dict and "consistency" in metrics_dict[key]:
            m = metrics_dict[key]["consistency"]
            pearson = m.get("pearson_r", None)
            if pearson is not None:
                annotation_lines.append(f"{key}: Pearson r={pearson:.3f}")

    all_d = np.concatenate([df["FOMS_direct_pred"].values for df in results_dict.values()])
    all_p = np.concatenate([df["FOMS_phys_pred"].values for df in results_dict.values()])
    lo = min(all_d.min(), all_p.min())
    hi = max(all_d.max(), all_p.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "--", color="gray", lw=1, alpha=0.7, label="y = x")

    ax.set_xlabel("FOMS_direct (predicted)", fontsize=9)
    ax.set_ylabel("FOMS_phys (predicted)", fontsize=9)
    ax.set_title("OOD: Consistency", fontsize=10)
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="gray")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)

    if annotation_lines:
        ax.text(0.05, 0.95, "\n".join(annotation_lines), transform=ax.transAxes,
                fontsize=6.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    save_path = os.path.join(output_dir, "ood_parity_consistency.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 已保存: {save_path}")

    # ---- Log-scale consistency parity plot ----
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 3.35))
    annotation_lines = []
    for i, (key, df) in enumerate(results_dict.items()):
        foms_d = df["FOMS_direct_pred"].values
        foms_p = df["FOMS_phys_pred"].values
        pos = (foms_d > 0) & (foms_p > 0)
        ax.scatter(foms_d[pos], foms_p[pos], alpha=0.6, s=40, marker=markers[i],
                   color=colors[i], label=key, edgecolors="k", linewidths=0.5)

        if metrics_dict and key in metrics_dict and "consistency" in metrics_dict[key]:
            m = metrics_dict[key]["consistency"]
            pearson = m.get("pearson_r", None)
            if pearson is not None:
                annotation_lines.append(f"{key}: Pearson r={pearson:.3f}")

    all_d = np.concatenate([df["FOMS_direct_pred"].values for df in results_dict.values()])
    all_p = np.concatenate([df["FOMS_phys_pred"].values for df in results_dict.values()])
    pos_all = (all_d > 0) & (all_p > 0)
    lo = min(all_d[pos_all].min(), all_p[pos_all].min()) * 0.5
    hi = max(all_d[pos_all].max(), all_p[pos_all].max()) * 2.0

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, alpha=0.7, label="y = x")

    ax.set_xlabel("FOMS_direct (predicted)", fontsize=9)
    ax.set_ylabel("FOMS_phys (predicted)", fontsize=9)
    ax.set_title("OOD: Consistency (log)", fontsize=10)
    ax.legend(fontsize=7, frameon=True, fancybox=False, edgecolor="gray")
    ax.grid(True, alpha=0.2, which="both")
    ax.tick_params(labelsize=7)

    if annotation_lines:
        ax.text(0.05, 0.95, "\n".join(annotation_lines), transform=ax.transAxes,
                fontsize=6.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    save_path = os.path.join(output_dir, "ood_parity_consistency_log.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 已保存: {save_path}")


# ============================================================================
# V3 分析辅助函数
# ============================================================================

def _to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types recursively."""
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_v3_decomposition(result_df):
    """
    V3 误差分解: ID-n (n 在训练网格中) vs OOD-n (n 不在训练网格中)

    训练网格 n 值: {2, 4, 8, 16, 32, 64}
    V3 n 值: {3, 4, 7, 16, 24}
    ID-n: n=4, n=16 (6 points, pure parameter-space extrapolation)
    OOD-n: n=3, n=7, n=24 (9 points, compound extrapolation)

    同时计算每个场景 (A/B/C) 的独立指标.
    """
    TRAIN_N = {2, 4, 8, 16, 32, 64}

    SCENARIOS = {
        "A": {"E": 1.5, "dd": 0.075, "hh": 0.02},
        "B": {"E": 6.0, "dd": 0.05, "hh": 0.01},
        "C": {"E": 8.0, "dd": 0.15, "hh": 0.04},
    }

    decomposition = {}

    # ID-n vs OOD-n
    id_mask = result_df["n"].isin(TRAIN_N)
    ood_mask = ~id_mask

    for label, mask in [("id_n", id_mask), ("ood_n", ood_mask)]:
        subset = result_df[mask]
        if len(subset) == 0:
            continue
        sub_metrics = {}
        sub_metrics["n_points"] = int(len(subset))
        sub_metrics["n_values"] = sorted([int(x) for x in subset["n"].unique()])
        sub_metrics["qsc"] = compute_metrics(
            subset["Qsc_true"].values, subset["Qsc_pred"].values, name=f"Qsc_{label}")
        sub_metrics["invc"] = compute_metrics(
            subset["invC_true"].values, subset["invC_pred"].values, name=f"invC_{label}")
        sub_metrics["foms_direct"] = compute_metrics(
            subset["FOMS_direct_true"].values, subset["FOMS_direct_pred"].values, name=f"FOMS_direct_{label}")
        sub_metrics["foms_phys"] = compute_metrics(
            subset["FOMS_phys_true"].values, subset["FOMS_phys_pred"].values, name=f"FOMS_phys_{label}")
        sub_metrics["consistency"] = compute_consistency_metrics(
            subset["FOMS_direct_pred"].values, subset["FOMS_phys_pred"].values, name=f"Consistency_{label}")
        decomposition[label] = sub_metrics

    # Per-scenario metrics
    scenarios = {}
    for scenario_name, params in SCENARIOS.items():
        mask = (
            (np.isclose(result_df["E"], params["E"])) &
            (np.isclose(result_df["dd"], params["dd"])) &
            (np.isclose(result_df["hh"], params["hh"]))
        )
        subset = result_df[mask]
        if len(subset) == 0:
            continue
        sc_metrics = {}
        sc_metrics["n_points"] = int(len(subset))
        sc_metrics["params"] = params
        sc_metrics["qsc"] = compute_metrics(
            subset["Qsc_true"].values, subset["Qsc_pred"].values, name=f"Qsc_V3{scenario_name}")
        sc_metrics["invc"] = compute_metrics(
            subset["invC_true"].values, subset["invC_pred"].values, name=f"invC_V3{scenario_name}")
        sc_metrics["foms_direct"] = compute_metrics(
            subset["FOMS_direct_true"].values, subset["FOMS_direct_pred"].values, name=f"FOMS_direct_V3{scenario_name}")
        sc_metrics["foms_phys"] = compute_metrics(
            subset["FOMS_phys_true"].values, subset["FOMS_phys_pred"].values, name=f"FOMS_phys_V3{scenario_name}")
        sc_metrics["consistency"] = compute_consistency_metrics(
            subset["FOMS_direct_pred"].values, subset["FOMS_phys_pred"].values, name=f"Consistency_V3{scenario_name}")
        scenarios[scenario_name] = sc_metrics

    return decomposition, scenarios


def compute_combined_ood_metrics(results_list):
    """Compute aggregate metrics across all OOD validation sets."""
    combined = pd.concat(results_list, ignore_index=True)
    metrics = {}
    metrics["n_points"] = int(len(combined))
    metrics["qsc"] = compute_metrics(
        combined["Qsc_true"].values, combined["Qsc_pred"].values, name="Qsc_combined")
    metrics["invc"] = compute_metrics(
        combined["invC_true"].values, combined["invC_pred"].values, name="invC_combined")
    metrics["foms_direct"] = compute_metrics(
        combined["FOMS_direct_true"].values, combined["FOMS_direct_pred"].values, name="FOMS_direct_combined")
    metrics["foms_phys"] = compute_metrics(
        combined["FOMS_phys_true"].values, combined["FOMS_phys_pred"].values, name="FOMS_phys_combined")
    metrics["consistency"] = compute_consistency_metrics(
        combined["FOMS_direct_pred"].values, combined["FOMS_phys_pred"].values, name="Consistency_combined")
    return metrics


# ============================================================================
# 模型加载
# ============================================================================

def load_model_and_scalers(device, model_path=None, artifact_dir=None):
    """
    加载多任务模型和所有 scaler

    Args:
        device:       计算设备
        model_path:   模型权重路径（默认使用 best checkpoint）
        artifact_dir: scaler 文件目录

    Returns:
        model, scaler_X, scaler_qsc, scaler_invc, scaler_foms
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if artifact_dir is None:
        artifact_dir = ARTIFACT_DIR

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n"
            f"请先运行训练: python train_multitask_physics.py"
        )

    # 加载 scaler
    scaler_X, scaler_qsc, scaler_invc, scaler_foms = load_scalers_multitask(artifact_dir)

    # 加载 checkpoint（支持新旧两种格式）
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_config" in ckpt:
        # 新格式: 包含 model_config 元数据
        config = ckpt["model_config"]
        state_dict = ckpt["state_dict"]
        print(f"[加载] checkpoint 含 model_config: {config}")
    else:
        # 旧格式: 仅 state_dict，推断 embed_dim，其余使用训练默认值
        state_dict = ckpt
        embed_dim = state_dict["feature_embedding.weight"].shape[0]
        config = {
            "input_dim": 4,
            "embed_dim": embed_dim,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.05,
        }
        print(f"[加载] 旧格式 checkpoint, 推断配置: {config}")

    # 创建并加载模型
    model = PhysicsMultiTaskTransformer(
        input_dim=config["input_dim"],
        embed_dim=config["embed_dim"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[加载] 模型: {model_path} (embed_dim={config['embed_dim']})")
    print(f"[加载] Scaler: {artifact_dir}/")

    return model, scaler_X, scaler_qsc, scaler_invc, scaler_foms


# ============================================================================
# 命令行入口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多任务物理一致模型推理 (PhysicsMultiTaskTransformer)"
    )
    parser.add_argument("--n", type=float, default=None, help="叶片对数")
    parser.add_argument("--E", type=float, default=None, help="介电常数")
    parser.add_argument("--dd", type=float, default=None, help="间隙比例 (无量纲)")
    parser.add_argument("--hh", type=float, default=None, help="高度比例 (无量纲)")
    parser.add_argument("--batch_file", type=str, default=None, help="批量输入 CSV 文件")
    parser.add_argument("--output_file", type=str, default=None, help="批量输出 CSV 文件")
    parser.add_argument("--model_path", type=str, default=None, help="模型权重路径")
    parser.add_argument("--artifact_dir", type=str, default=None, help="Scaler 文件目录")
    parser.add_argument("--validate", action="store_true", help="OOD 验证模式（validate1 + validate2）")
    parser.add_argument("--validate_dir", type=str, default="../data", help="验证集目录")
    parser.add_argument("--output_dir", type=str, default="../outputs_multitask_physics", help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print("\n" + "=" * 60)
    print("  多任务物理一致模型推理")
    print("=" * 60)

    # 加载模型
    model, scaler_X, scaler_qsc, scaler_invc, scaler_foms = load_model_and_scalers(
        device, args.model_path, args.artifact_dir
    )

    if args.validate:
        # OOD 验证模式
        print("\n[模式] OOD 验证 (--validate)")
        print("  评估 validate1, validate2 数据集 (+ validate3 if available)\n")

        validate1_path = os.path.join(args.validate_dir, "validate_foms_macrs.csv")
        validate2_path = os.path.join(args.validate_dir, "validate2_foms_macrs.csv")
        validate3_path = os.path.join(args.validate_dir, "validate3_foms_macrs.csv")

        if not os.path.exists(validate1_path) or not os.path.exists(validate2_path):
            print(f"[错误] 验证集文件不存在:")
            print(f"  {validate1_path}")
            print(f"  {validate2_path}")
            return

        # 验证 validate1
        result1, metrics1 = validate_ood(
            validate1_path, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device
        )
        output1_path = os.path.join(args.validate_dir, "multitask_validate1_results.csv")
        result1.to_csv(output1_path, index=False, float_format="%.10e")
        print(f"\n[保存] validate1 结果: {output1_path}")

        # 验证 validate2
        result2, metrics2 = validate_ood(
            validate2_path, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device
        )
        output2_path = os.path.join(args.validate_dir, "multitask_validate2_results.csv")
        result2.to_csv(output2_path, index=False, float_format="%.10e")
        print(f"\n[保存] validate2 结果: {output2_path}")

        # 验证 validate3 (自动检测)
        result3 = None
        metrics3 = None
        if os.path.exists(validate3_path):
            result3, metrics3 = validate_ood(
                validate3_path, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device
            )
            output3_path = os.path.join(args.validate_dir, "multitask_validate3_results.csv")
            result3.to_csv(output3_path, index=False, float_format="%.10e")
            print(f"\n[保存] validate3 结果: {output3_path}")

        # 构建结果和指标字典
        results_dict = {"validate1": result1, "validate2": result2}
        ood_metrics = {
            "validate1": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics1.items()},
            "validate2": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics2.items()},
        }
        if result3 is not None:
            results_dict["validate3"] = result3
            ood_metrics["validate3"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics3.items()}

        # 绘制 parity plots（传递指标用于标注）
        print("\n[可视化] 生成 OOD parity plots...")
        plot_ood_parity(results_dict, args.output_dir, metrics_dict=ood_metrics)

        # 计算按 n 分组误差并保存
        per_n_breakdown = {}
        validate_items = [("validate1", result1), ("validate2", result2)]
        if result3 is not None:
            validate_items.append(("validate3", result3))
        for key, df in validate_items:
            per_n = {}
            for n_val in sorted(df["n"].unique()):
                subset = df[df["n"] == n_val]
                per_n[str(int(n_val))] = {
                    "qsc_rel_error_%": float(subset["Qsc_rel_error_%"].mean()),
                    "invc_rel_error_%": float(subset["invC_rel_error_%"].mean()),
                    "foms_direct_rel_error_%": float(subset["FOMS_direct_rel_error_%"].mean()),
                    "foms_phys_rel_error_%": float(subset["FOMS_phys_rel_error_%"].mean()),
                }
            per_n_breakdown[key] = per_n
        ood_metrics["per_n_breakdown"] = per_n_breakdown

        # V3 误差分解与场景分析
        if result3 is not None:
            print("\n[分析] V3 误差分解 (ID-n vs OOD-n) + 场景分析...")
            decomposition, scenarios = compute_v3_decomposition(result3)
            ood_metrics["validate3_error_decomposition"] = _to_json_serializable(decomposition)
            ood_metrics["validate3_scenarios"] = _to_json_serializable(scenarios)

            # 合并全部验证集的聚合指标
            combined = compute_combined_ood_metrics([result1, result2, result3])
            ood_metrics["combined"] = _to_json_serializable(combined)

            # 打印 V3 误差分解摘要
            print("\n  === V3 误差分解 ===")
            for label in ["id_n", "ood_n"]:
                if label in decomposition:
                    d = decomposition[label]
                    print(f"  {label} (n={d['n_values']}, {d['n_points']} points):")
                    print(f"    FOMS_direct R²_log10={d['foms_direct']['r2_log10']:.3f}, "
                          f"MAE_log10={d['foms_direct']['mae_log10']:.3f}")
                    print(f"    Consistency Pearson={d['consistency']['pearson_r']:.3f}")

            print("\n  === V3 场景分析 ===")
            for sc_name, sc_data in scenarios.items():
                p = sc_data["params"]
                print(f"  场景 {sc_name} (E={p['E']}, dd={p['dd']}, hh={p['hh']}, {sc_data['n_points']} points):")
                print(f"    FOMS_direct R²_log10={sc_data['foms_direct']['r2_log10']:.3f}, "
                      f"MAE_log10={sc_data['foms_direct']['mae_log10']:.3f}")

            print(f"\n  === 合并指标 ({combined['n_points']} points) ===")
            print(f"    Qsc R²_log10={combined['qsc']['r2_log10']:.3f}")
            print(f"    invC R²_log10={combined['invc']['r2_log10']:.3f}")
            print(f"    FOMS_direct R²_log10={combined['foms_direct']['r2_log10']:.3f}")
            print(f"    Consistency Pearson={combined['consistency']['pearson_r']:.3f}")

        # 保存 OOD 指标汇总
        ood_json_path = os.path.join(args.output_dir, "ood_metrics.json")
        with open(ood_json_path, "w", encoding="utf-8") as f:
            json.dump(ood_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n[保存] OOD 指标汇总: {ood_json_path}")

        print("\n" + "=" * 60)
        print("  OOD 验证完成！")
        print("=" * 60)
        n_sets = 2 + (1 if result3 is not None else 0)
        n_total = len(result1) + len(result2) + (len(result3) if result3 is not None else 0)
        print(f"  验证集数量: {n_sets} (共 {n_total} 个 OOD 点)")
        print(f"  结果 CSV: {args.validate_dir}/")
        print(f"  图表:     {args.output_dir}/")
        print(f"  指标:     {ood_json_path}")
        print("=" * 60 + "\n")
        return

    if args.batch_file:
        # 批量推理
        df_input = pd.read_csv(args.batch_file)
        print(f"\n[批量推理] 输入文件: {args.batch_file}, 样本数: {len(df_input)}")

        result = predict_batch(
            df_input, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device
        )

        output_file = args.output_file or args.batch_file.replace(".csv", "_multitask_pred.csv")
        result.to_csv(output_file, index=False, float_format="%.10e")
        print(f"[输出] 结果已保存: {output_file}")
        print(result.head())

    elif args.n is not None and args.E is not None and args.dd is not None and args.hh is not None:
        # 单样本推理
        print(f"\n[单样本推理] n={args.n}, E={args.E}, dd={args.dd}, hh={args.hh}")

        result = predict_single(
            args.n, args.E, args.dd, args.hh,
            model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device,
        )

        print("\n  === 预测结果 ===")
        print(f"  Qsc_MACRS    = {result['Qsc_MACRS']:.6e}  (C)")
        print(f"  invC_sum     = {result['invC_sum']:.6e}  (1/F)")
        print(f"  FOMS_direct  = {result['FOMS_direct']:.6e}")
        print(f"  FOMS_phys    = {result['FOMS_phys']:.6e}")
        print(f"  (FOMS_phys 由 Qsc_MACRS 和 invC_sum 通过物理公式重构)")

    else:
        print("\n  用法:")
        print("  1) 单样本:  python predict_multitask_physics.py --n 4 --E 3.4 --dd 0.25 --hh 0.125")
        print("  2) 批量:    python predict_multitask_physics.py --batch_file input.csv")
        print("  3) 帮助:    python predict_multitask_physics.py --help")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
