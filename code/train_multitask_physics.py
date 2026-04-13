#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务物理一致模型 - 训练脚本
========================================================
训练 PhysicsMultiTaskTransformer 模型，同时预测:
  1. Qsc_MACRS
  2. invC_sum
  3. FOMS_direct

并使用 FOMS_phys (物理公式重构) 做一致性约束。

与原版 main.py 完全独立，不修改任何原有文件。

关键设计:
  - 目标变量先取 log10 再 MinMaxScaler（压缩 11 个数量级的动态范围）
  - 一致性损失在 log10 空间直接计算（数值稳定）
  - 评估指标在原始物理量级计算（10**x 逆变换）

使用方法:
    python train_multitask_physics.py
    python train_multitask_physics.py --epochs 300 --lr 0.0005
    python train_multitask_physics.py --debug

"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 本地模块
from utils_multitask_physics import (
    load_multitask_data,
    save_scalers,
    compute_metrics,
    compute_consistency_metrics,
    compute_foms_phys,
    compute_log10_foms_phys_torch,
    setup_seed,
    get_device,
    EPSILON_0,
    SIGMA,
    PI,
    R,
)
from model_multitask_physics import PhysicsMultiTaskTransformer


# ============================================================================
# 1. 配置参数（集中管理）
# ============================================================================

# --- 多任务损失权重 ---
LAMBDA_QSC = 1.0  # Qsc_MACRS 回归损失权重
LAMBDA_INVC = 1.0  # invC_sum 回归损失权重
LAMBDA_FOMS = 1.0  # FOMS_direct 回归损失权重
LAMBDA_CONSISTENCY = 0.5  # FOMS_direct 与 FOMS_phys 一致性损失权重

# --- 路径配置（相对于 code/ 目录）---
DATA_PATH = "../data/disk_teng_training_processed.csv"
CHECKPOINT_DIR = "../checkpoints_multitask_physics"
ARTIFACT_DIR = "../artifacts_multitask_physics"
OUTPUT_DIR = "../outputs_multitask_physics"


# ============================================================================
# 2. 命令行参数
# ============================================================================


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多任务物理一致模型训练 (PhysicsMultiTaskTransformer)"
    )
    parser.add_argument("--debug", action="store_true", help="调试模式（仅 1 epoch）")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="仅评估模式（加载最佳模型，评估测试集，保存指标）",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.0005, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="数据文件路径")
    parser.add_argument(
        "--experiment_mode",
        type=str,
        default="full",
        choices=["reproduction", "full"],
        help="实验模式: reproduction (dd<=0.5) 或 full (全量数据)",
    )
    # 损失权重
    parser.add_argument("--lambda_qsc", type=float, default=LAMBDA_QSC)
    parser.add_argument("--lambda_invc", type=float, default=LAMBDA_INVC)
    parser.add_argument("--lambda_foms", type=float, default=LAMBDA_FOMS)
    parser.add_argument("--lambda_consistency", type=float, default=LAMBDA_CONSISTENCY)
    # 模型结构
    parser.add_argument("--embed_dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--nhead", type=int, default=4, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer 层数")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout 比例")

    return parser.parse_args()


# ============================================================================
# 3. 训练一个 epoch
# ============================================================================


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler_qsc,
    scaler_invc,
    scaler_foms,
    lambda_qsc,
    lambda_invc,
    lambda_foms,
    lambda_consistency,
    debug=False,
):
    """
    训练一个 epoch

    一致性损失在 log10 空间计算:
    scaler 作用于 log10(y)，所以 inverse_transform 得到的是 log10 值。
    用 compute_log10_foms_phys_torch 直接在 log10 空间算 FOMS_phys，
    再归一化到同一 scaler 空间后与 FOMS_direct 比较。

    Returns:
        dict: 各项损失的平均值
    """
    model.train()
    criterion = nn.MSELoss()

    total_loss_sum = 0.0
    loss_qsc_sum = 0.0
    loss_invc_sum = 0.0
    loss_foms_sum = 0.0
    loss_cons_sum = 0.0
    n_batches = 0

    # 预计算 scaler 参数（log10 空间的 min 和 range）
    qsc_min_log = torch.tensor(scaler_qsc.data_min_[0], dtype=torch.float32)
    qsc_range_log = torch.tensor(scaler_qsc.data_range_[0], dtype=torch.float32)
    invc_min_log = torch.tensor(scaler_invc.data_min_[0], dtype=torch.float32)
    invc_range_log = torch.tensor(scaler_invc.data_range_[0], dtype=torch.float32)
    foms_min_log = torch.tensor(scaler_foms.data_min_[0], dtype=torch.float32)
    foms_range_log = torch.tensor(scaler_foms.data_range_[0], dtype=torch.float32)

    pbar = tqdm(loader, desc="Training", leave=False, disable=debug)

    for batch_idx, (features, t_qsc, t_invc, t_foms, raw_n) in enumerate(pbar):
        features = features.to(device)
        t_qsc = t_qsc.to(device)
        t_invc = t_invc.to(device)
        t_foms = t_foms.to(device)
        raw_n = raw_n.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        # --- 三个回归损失（在归一化的 log10 空间中计算）---
        L_q = criterion(outputs["pred_qsc"], t_qsc)
        L_c = criterion(outputs["pred_invc_sum"], t_invc)
        L_f = criterion(outputs["pred_foms_direct"], t_foms)

        # --- 一致性损失：在 log10 空间直接计算 ---
        # 逆 MinMaxScaler → 得到 log10 值
        pred_log10_qsc = outputs["pred_qsc"] * qsc_range_log.to(
            device
        ) + qsc_min_log.to(device)
        pred_log10_invc = outputs["pred_invc_sum"] * invc_range_log.to(
            device
        ) + invc_min_log.to(device)

        # 在 log10 空间用物理公式计算 log10(FOMS_phys)
        # 参考 calculate_foms.py 的物理组合公式
        log10_foms_phys = compute_log10_foms_phys_torch(
            pred_log10_qsc, pred_log10_invc, raw_n.unsqueeze(-1)
        )

        # 将 log10(FOMS_phys) 归一化到与 FOMS_direct 相同的 scaler 空间
        foms_phys_normalized = (
            log10_foms_phys - foms_min_log.to(device)
        ) / foms_range_log.to(device)

        # 一致性损失
        L_consistency = criterion(
            outputs["pred_foms_direct"], foms_phys_normalized.detach()
        )

        # --- 总损失 ---
        L_total = (
            lambda_qsc * L_q
            + lambda_invc * L_c
            + lambda_foms * L_f
            + lambda_consistency * L_consistency
        )

        # NaN 检查
        if torch.isnan(L_total) or torch.isinf(L_total):
            print(f"\n  Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
            continue

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += L_total.item()
        loss_qsc_sum += L_q.item()
        loss_invc_sum += L_c.item()
        loss_foms_sum += L_f.item()
        loss_cons_sum += L_consistency.item()
        n_batches += 1

        pbar.set_postfix({"loss": L_total.item()})

        if debug and batch_idx == 0:
            print(f"  DEBUG: batch 0 loss={L_total.item():.6f}")
            break

    if n_batches == 0:
        return {"total": float("nan"), "qsc": 0, "invc": 0, "foms": 0, "consistency": 0}

    return {
        "total": total_loss_sum / n_batches,
        "qsc": loss_qsc_sum / n_batches,
        "invc": loss_invc_sum / n_batches,
        "foms": loss_foms_sum / n_batches,
        "consistency": loss_cons_sum / n_batches,
    }


# ============================================================================
# 4. 验证
# ============================================================================


def validate(
    model,
    loader,
    device,
    scaler_qsc,
    scaler_invc,
    scaler_foms,
    lambda_qsc,
    lambda_invc,
    lambda_foms,
    lambda_consistency,
):
    """
    在验证集上评估模型

    Returns:
        dict: 各项损失的平均值
    """
    model.eval()
    criterion = nn.MSELoss()

    total_loss_sum = 0.0
    loss_qsc_sum = 0.0
    loss_invc_sum = 0.0
    loss_foms_sum = 0.0
    loss_cons_sum = 0.0
    n_batches = 0

    qsc_min_log = scaler_qsc.data_min_[0]
    qsc_range_log = scaler_qsc.data_range_[0]
    invc_min_log = scaler_invc.data_min_[0]
    invc_range_log = scaler_invc.data_range_[0]
    foms_min_log = scaler_foms.data_min_[0]
    foms_range_log = scaler_foms.data_range_[0]

    with torch.no_grad():
        for features, t_qsc, t_invc, t_foms, raw_n in loader:
            features = features.to(device)
            t_qsc = t_qsc.to(device)
            t_invc = t_invc.to(device)
            t_foms = t_foms.to(device)
            raw_n = raw_n.to(device)

            outputs = model(features)

            L_q = criterion(outputs["pred_qsc"], t_qsc)
            L_c = criterion(outputs["pred_invc_sum"], t_invc)
            L_f = criterion(outputs["pred_foms_direct"], t_foms)

            # 一致性损失（log10 空间）
            pred_log10_qsc = outputs["pred_qsc"] * qsc_range_log + qsc_min_log
            pred_log10_invc = outputs["pred_invc_sum"] * invc_range_log + invc_min_log
            log10_foms_phys = compute_log10_foms_phys_torch(
                pred_log10_qsc, pred_log10_invc, raw_n.unsqueeze(-1)
            )
            foms_phys_norm = (log10_foms_phys - foms_min_log) / foms_range_log
            L_consistency = criterion(outputs["pred_foms_direct"], foms_phys_norm)

            L_total = (
                lambda_qsc * L_q
                + lambda_invc * L_c
                + lambda_foms * L_f
                + lambda_consistency * L_consistency
            )

            total_loss_sum += L_total.item()
            loss_qsc_sum += L_q.item()
            loss_invc_sum += L_c.item()
            loss_foms_sum += L_f.item()
            loss_cons_sum += L_consistency.item()
            n_batches += 1

    if n_batches == 0:
        return {"total": float("nan"), "qsc": 0, "invc": 0, "foms": 0, "consistency": 0}

    return {
        "total": total_loss_sum / n_batches,
        "qsc": loss_qsc_sum / n_batches,
        "invc": loss_invc_sum / n_batches,
        "foms": loss_foms_sum / n_batches,
        "consistency": loss_cons_sum / n_batches,
    }


# ============================================================================
# 5. 全面评估（测试集）
# ============================================================================


def full_evaluation(model, loader, device, scaler_qsc, scaler_invc, scaler_foms):
    """
    在 val/test 上进行全面评估，返回原始物理量级指标。

    注意: scaler.inverse_transform 返回 log10 值，需要 10**x 得到原始物理量。

    Returns:
        results: dict, 各变量的 {mae, rmse, r2} 以及一致性指标
        arrays:  dict, 真实值和预测值数组（用于绘图）
    """
    model.eval()

    all_pred_qsc, all_true_qsc = [], []
    all_pred_invc, all_true_invc = [], []
    all_pred_foms, all_true_foms = [], []
    all_raw_n = []

    with torch.no_grad():
        for features, t_qsc, t_invc, t_foms, raw_n in loader:
            features = features.to(device)
            outputs = model(features)

            all_pred_qsc.append(outputs["pred_qsc"].cpu().numpy())
            all_pred_invc.append(outputs["pred_invc_sum"].cpu().numpy())
            all_pred_foms.append(outputs["pred_foms_direct"].cpu().numpy())
            all_true_qsc.append(t_qsc.numpy())
            all_true_invc.append(t_invc.numpy())
            all_true_foms.append(t_foms.numpy())
            all_raw_n.append(raw_n.numpy())

    # 合并
    pred_qsc_norm = np.vstack(all_pred_qsc)
    pred_invc_norm = np.vstack(all_pred_invc)
    pred_foms_norm = np.vstack(all_pred_foms)
    true_qsc_norm = np.vstack(all_true_qsc)
    true_invc_norm = np.vstack(all_true_invc)
    true_foms_norm = np.vstack(all_true_foms)
    raw_n = np.concatenate(all_raw_n)

    # 逆 MinMaxScaler → log10 值
    pred_qsc_log = scaler_qsc.inverse_transform(pred_qsc_norm)
    true_qsc_log = scaler_qsc.inverse_transform(true_qsc_norm)
    pred_invc_log = scaler_invc.inverse_transform(pred_invc_norm)
    true_invc_log = scaler_invc.inverse_transform(true_invc_norm)
    pred_foms_log = scaler_foms.inverse_transform(pred_foms_norm)
    true_foms_log = scaler_foms.inverse_transform(true_foms_norm)

    # 10**x → 原始物理量级
    pred_qsc = 10.0**pred_qsc_log
    true_qsc = 10.0**true_qsc_log
    pred_invc = 10.0**pred_invc_log
    true_invc = 10.0**true_invc_log
    pred_foms = 10.0**pred_foms_log
    true_foms = 10.0**true_foms_log

    # 计算 FOMS_phys（参考 calculate_foms.py 的物理组合公式）
    pred_foms_phys = compute_foms_phys(
        pred_qsc.flatten(), pred_invc.flatten(), raw_n
    ).reshape(-1, 1)

    # 评估各项指标
    print("\n  === 各目标评估指标 (原始物理量级) ===")
    m_qsc = compute_metrics(true_qsc, pred_qsc, "Qsc_MACRS")
    m_invc = compute_metrics(true_invc, pred_invc, "invC_sum")
    m_foms_direct = compute_metrics(true_foms, pred_foms, "FOMS_direct")
    m_foms_phys = compute_metrics(true_foms, pred_foms_phys, "FOMS_phys")

    print("\n  === FOMS_direct vs FOMS_phys 一致性 ===")
    m_consistency = compute_consistency_metrics(
        pred_foms.flatten(), pred_foms_phys.flatten()
    )

    results = {
        "qsc": m_qsc,
        "invc": m_invc,
        "foms_direct": m_foms_direct,
        "foms_phys": m_foms_phys,
        "consistency": m_consistency,
    }

    arrays = {
        "true_qsc": true_qsc,
        "pred_qsc": pred_qsc,
        "true_invc": true_invc,
        "pred_invc": pred_invc,
        "true_foms": true_foms,
        "pred_foms_direct": pred_foms,
        "pred_foms_phys": pred_foms_phys,
    }

    return results, arrays


def save_test_metrics(
    test_results, output_dir, model_config=None, n_test=None, test_arrays=None
):
    """
    保存测试集指标到 JSON 文件

    Args:
        test_results: full_evaluation() 返回的 results 字典
        output_dir: 输出目录路径
        model_config: 模型超参数字典（可选）
        n_test: 测试集样本数（可选）
        test_arrays: full_evaluation() 返回的 arrays 字典（用于分区域分析）
    """
    from datetime import datetime

    # 转换 numpy 类型为 Python 原生类型（JSON 可序列化）
    metrics = {}
    for key, value_dict in test_results.items():
        metrics[key] = {k: float(v) for k, v in value_dict.items()}

    # 添加元数据
    metadata = {"timestamp": datetime.now().isoformat()}
    if model_config:
        metadata["model_config"] = model_config
    if n_test is not None:
        metadata["n_test"] = int(n_test)
    metrics["_metadata"] = metadata

    # 按 FOMS 量级分区域分析（A2: 解决 R² 被高值区域主导的问题）
    if test_arrays is not None and "true_foms" in test_arrays:
        true_foms = np.asarray(test_arrays["true_foms"]).flatten()
        pred_foms = np.asarray(test_arrays["pred_foms_direct"]).flatten()
        bands = [
            ("foms_band_low", true_foms < 1e-8),
            ("foms_band_mid", (true_foms >= 1e-8) & (true_foms < 1e-5)),
            ("foms_band_high", true_foms >= 1e-5),
        ]
        foms_by_band = {}
        for band_name, mask in bands:
            n_pts = int(mask.sum())
            if n_pts > 0:
                t, p = true_foms[mask], pred_foms[mask]
                pos = (t > 0) & (p > 0)
                if np.any(pos):
                    log_t = np.log10(t[pos])
                    log_p = np.log10(p[pos])
                    mae_l = float(np.mean(np.abs(log_t - log_p)))
                    ss_r = np.sum((log_t - log_p) ** 2)
                    ss_t = np.sum((log_t - np.mean(log_t)) ** 2)
                    r2_l = float(1.0 - ss_r / ss_t) if ss_t > 0 else 0.0
                else:
                    mae_l, r2_l = float("inf"), 0.0
                foms_by_band[band_name] = {
                    "n_samples": n_pts,
                    "mae_log10": mae_l,
                    "r2_log10": r2_l,
                }
                print(
                    f"  FOMS {band_name}: n={n_pts}, R²_log10={r2_l:.4f}, MAE_log10={mae_l:.4f}"
                )
        metrics["foms_by_magnitude_band"] = foms_by_band

    output_path = os.path.join(output_dir, "test_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[保存] 测试集指标已保存: {output_path}")


# ============================================================================
# 6. 可视化
# ============================================================================


def compute_shared_log_limits(*arrays):
    """Compute one shared positive log-range across multiple arrays."""
    positives = []
    for arr in arrays:
        arr = np.asarray(arr).flatten()
        pos = arr[arr > 0]
        if len(pos) > 0:
            positives.append(pos)
    if not positives:
        return (1e-12, 1.0)
    merged = np.concatenate(positives)
    return merged.min() * 0.5, merged.max() * 2.0


def plot_parity(
    y_true,
    y_pred,
    title,
    save_path,
    metrics=None,
    log_scale=False,
    axis_limits=None,
    annotation_loc=(0.05, 0.95),
    annotation_keys=None,
):
    """绘制 parity plot with optional R²/MAPE/MAE_log10 annotations"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    ax.scatter(y_true, y_pred, alpha=0.5, s=16, edgecolors="none", c="#0072B2")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        if axis_limits is not None:
            lo, hi = axis_limits
        else:
            pos = (y_true > 0) & (y_pred > 0)
            lo = min(y_true[pos].min(), y_pred[pos].min()) * 0.5
            hi = max(y_true[pos].max(), y_pred[pos].max()) * 2.0
    else:
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = (hi - lo) * 0.05
        lo, hi = lo - margin, hi + margin

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle=(0, (2.2, 1.8)),
        color="#A8B0B8",
        lw=1.0,
        alpha=0.95,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True", fontsize=10)
    ax.set_ylabel("Predicted", fontsize=10)
    if title:
        ax.set_title(title, fontsize=10.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2, which="both" if log_scale else "major")
    ax.tick_params(labelsize=7.6)

    # Metrics annotation textbox
    if metrics:
        if annotation_keys is None:
            annotation_keys = ["r2_log10", "mae_log10"] if log_scale else ["r2", "mae"]
        lines = []
        for key in annotation_keys:
            if key == "r2_log10" and "r2_log10" in metrics:
                lines.append(r"$R^2_{\log_{10}}$" + f" = {metrics['r2_log10']:.4f}")
            elif key == "mae_log10" and "mae_log10" in metrics:
                lines.append(r"MAE$_{\log_{10}}$" + f" = {metrics['mae_log10']:.3f}")
            elif key == "pearson_r" and "pearson_r" in metrics:
                lines.append(f"Pearson = {metrics['pearson_r']:.4f}")
            elif key == "r2" and "r2" in metrics:
                lines.append(f"R² = {metrics['r2']:.4f}")
            elif key == "mae" and "mae" in metrics:
                lines.append(f"MAE = {metrics['mae']:.3e}")
        if lines:
            ax.text(
                annotation_loc[0],
                annotation_loc[1],
                "\n".join(lines),
                transform=ax.transAxes,
                fontsize=7.6,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="#F8E6BF",
                    edgecolor="#4B5563",
                    alpha=0.95,
                ),
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 已保存: {save_path}")


def plot_loss_curves(history, save_path):
    """绘制训练/验证损失曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    keys = ["total", "qsc", "invc", "foms", "consistency"]
    titles = ["Total Loss", "Qsc Loss", "InvC Loss", "FOMS Loss", "Consistency Loss"]

    for i, (key, title) in enumerate(zip(keys, titles)):
        ax = axes[i // 3][i % 3]
        ax.plot(history["train_" + key], label="Train", alpha=0.8)
        ax.plot(history["val_" + key], label="Val", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 隐藏第 6 个子图
    axes[1][2].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [图] 已保存: {save_path}")


# ============================================================================
# 7. 主训练流程
# ============================================================================


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = get_device()

    # 创建输出目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("  多任务物理一致模型训练 (PhysicsMultiTaskTransformer)")
    print("=" * 65)
    print(f"  设备: {device}")
    print(f"  数据: {args.data_path}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(
        f"  损失权重: λ_qsc={args.lambda_qsc}, λ_invc={args.lambda_invc}, "
        f"λ_foms={args.lambda_foms}, λ_cons={args.lambda_consistency}"
    )
    print(
        f"  模型: embed_dim={args.embed_dim}, nhead={args.nhead}, "
        f"layers={args.num_layers}, dropout={args.dropout}"
    )
    print("=" * 65 + "\n")

    # ====== 加载数据 ======
    (
        train_loader,
        val_loader,
        test_loader,
        scaler_X,
        scaler_qsc,
        scaler_invc,
        scaler_foms,
        raw_n_train,
        raw_n_val,
        raw_n_test,
    ) = load_multitask_data(
        data_path=args.data_path,
        experiment_mode=args.experiment_mode,
        debug=args.debug,
        batch_size=args.batch_size,
        random_state=args.seed,
    )

    # ====== 保存 scaler ======
    if not args.debug:
        save_scalers(scaler_X, scaler_qsc, scaler_invc, scaler_foms, ARTIFACT_DIR)

    # ====== 仅评估模式 ======
    if args.eval_only:
        print("\n[模式] 仅评估模式 (--eval_only)")
        print("  加载最佳模型，评估验证集和测试集，保存指标到 JSON\n")

        # 创建模型（需要配置参数）
        model = PhysicsMultiTaskTransformer(
            input_dim=4,
            embed_dim=args.embed_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        # 加载最佳模型
        best_path = os.path.join(CHECKPOINT_DIR, "physics_multitask_best.pth")
        if not os.path.exists(best_path):
            print(f"[错误] 未找到最佳模型: {best_path}")
            return

        ckpt = torch.load(best_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"[加载] 最佳模型: {best_path}")

        # 验证集评估
        print("\n" + "=" * 50)
        print("  验证集评估 (Validation)")
        print("=" * 50)
        val_results, val_arrays = full_evaluation(
            model, val_loader, device, scaler_qsc, scaler_invc, scaler_foms
        )

        # 测试集评估
        print("\n" + "=" * 50)
        print("  测试集评估 (Test)")
        print("=" * 50)
        test_results, test_arrays = full_evaluation(
            model, test_loader, device, scaler_qsc, scaler_invc, scaler_foms
        )

        # 保存测试集指标（含元数据）
        model_config = {
            "embed_dim": args.embed_dim,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lambda_consistency": args.lambda_consistency,
        }
        save_test_metrics(
            test_results,
            OUTPUT_DIR,
            model_config=model_config,
            n_test=len(test_loader.dataset),
            test_arrays=test_arrays,
        )

        # 生成 parity plots（带 R²/MAPE 标注）
        print("\n[可视化] 生成评估图表...")
        plot_parity(
            test_arrays["true_qsc"],
            test_arrays["pred_qsc"],
            "Qsc_MACRS: True vs Predicted",
            os.path.join(OUTPUT_DIR, "parity_qsc.png"),
            metrics=test_results["qsc"],
        )
        plot_parity(
            test_arrays["true_invc"],
            test_arrays["pred_invc"],
            "invC_sum: True vs Predicted",
            os.path.join(OUTPUT_DIR, "parity_invc_sum.png"),
            metrics=test_results["invc"],
        )
        plot_parity(
            test_arrays["true_foms"],
            test_arrays["pred_foms_direct"],
            "FOMS_direct: True vs Predicted",
            os.path.join(OUTPUT_DIR, "parity_foms_direct.png"),
            metrics=test_results["foms_direct"],
        )
        plot_parity(
            test_arrays["true_foms"],
            test_arrays["pred_foms_phys"],
            "FOMS_phys: True vs Predicted",
            os.path.join(OUTPUT_DIR, "parity_foms_phys.png"),
            metrics=test_results["foms_phys"],
        )
        plot_parity(
            test_arrays["pred_foms_direct"],
            test_arrays["pred_foms_phys"],
            "FOMS_direct vs FOMS_phys (Consistency)",
            os.path.join(OUTPUT_DIR, "direct_vs_phys_consistency.png"),
        )

        # Log-scale parity plots for multi-order quantities
        foms_log_limits = compute_shared_log_limits(
            test_arrays["true_foms"],
            test_arrays["pred_foms_direct"],
            test_arrays["pred_foms_phys"],
        )
        for name, true_key, pred_key, res_key in [
            ("Qsc_MACRS", "true_qsc", "pred_qsc", "qsc"),
            ("FOMS_direct", "true_foms", "pred_foms_direct", "foms_direct"),
            ("FOMS_phys", "true_foms", "pred_foms_phys", "foms_phys"),
        ]:
            title = f"{name}: True vs Predicted (log)"
            axis_limits = None
            annotation_loc = (0.05, 0.95)
            metrics_payload = test_results[res_key]
            if name == "FOMS_direct":
                title = ""
                axis_limits = foms_log_limits
                metrics_payload = None
            elif name == "FOMS_phys":
                title = ""
                axis_limits = foms_log_limits
                metrics_payload = None
            plot_parity(
                test_arrays[true_key],
                test_arrays[pred_key],
                title,
                os.path.join(OUTPUT_DIR, f"parity_{res_key}_log.png"),
                metrics=metrics_payload,
                log_scale=True,
                axis_limits=axis_limits,
                annotation_loc=annotation_loc,
                annotation_keys=["r2_log10", "mae_log10"],
            )
        plot_parity(
            test_arrays["pred_foms_direct"],
            test_arrays["pred_foms_phys"],
            "",
            os.path.join(OUTPUT_DIR, "direct_vs_phys_consistency_log.png"),
            metrics=None,
            log_scale=True,
            axis_limits=foms_log_limits,
            annotation_loc=(0.05, 0.95),
            annotation_keys=["r2_log10", "mae_log10"],
        )

        print("\n" + "=" * 65)
        print("  评估完成！")
        print("=" * 65)
        print(f"  指标:   {OUTPUT_DIR}/test_metrics.json")
        print(f"  图表:   {OUTPUT_DIR}/")
        print("=" * 65 + "\n")
        return

    # ====== 创建模型 ======
    model = PhysicsMultiTaskTransformer(
        input_dim=4,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    epochs = 1 if args.debug else args.epochs

    # CosineAnnealingWarmRestarts: 周期性重启，避免陷入局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=1e-6,
    )

    # ====== 训练循环 ======
    best_val_loss = float("inf")
    patience = 50  # 配合 cosine 周期，给足探索空间
    patience_counter = 0

    history = {
        "train_total": [],
        "train_qsc": [],
        "train_invc": [],
        "train_foms": [],
        "train_consistency": [],
        "val_total": [],
        "val_qsc": [],
        "val_invc": [],
        "val_foms": [],
        "val_consistency": [],
    }

    for epoch in range(epochs):
        # 训练
        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler_qsc,
            scaler_invc,
            scaler_foms,
            args.lambda_qsc,
            args.lambda_invc,
            args.lambda_foms,
            args.lambda_consistency,
            debug=args.debug,
        )

        if np.isnan(train_losses["total"]):
            print("Training stopped: NaN loss detected")
            break

        # 验证
        val_losses = validate(
            model,
            val_loader,
            device,
            scaler_qsc,
            scaler_invc,
            scaler_foms,
            args.lambda_qsc,
            args.lambda_invc,
            args.lambda_foms,
            args.lambda_consistency,
        )

        scheduler.step()

        # 记录历史
        for key in ["total", "qsc", "invc", "foms", "consistency"]:
            history["train_" + key].append(train_losses[key])
            history["val_" + key].append(val_losses[key])

        # 打印日志
        if not args.debug:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train: total={train_losses['total']:.6f} "
                f"(q={train_losses['qsc']:.4f} c={train_losses['invc']:.4f} "
                f"f={train_losses['foms']:.4f} cons={train_losses['consistency']:.4f}) | "
                f"Val: {val_losses['total']:.6f} | LR: {lr:.6f}"
            )

        # 保存 best checkpoint
        if val_losses["total"] < best_val_loss and not args.debug:
            best_val_loss = val_losses["total"]
            best_path = os.path.join(CHECKPOINT_DIR, "physics_multitask_best.pth")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": {
                        "input_dim": 4,
                        "embed_dim": args.embed_dim,
                        "nhead": args.nhead,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                    },
                },
                best_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if not args.debug and patience_counter >= patience:
                print(f"Early Stopping at epoch {epoch+1}")
                break

    # 保存 last checkpoint
    if not args.debug:
        last_path = os.path.join(CHECKPOINT_DIR, "physics_multitask_last.pth")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": {
                    "input_dim": 4,
                    "embed_dim": args.embed_dim,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
            },
            last_path,
        )
        print(f"\n[保存] Last checkpoint: {last_path}")
        print(
            f"[保存] Best checkpoint: {os.path.join(CHECKPOINT_DIR, 'physics_multitask_best.pth')}"
        )

    # ====== 加载 best 模型用于评估 ======
    if not args.debug:
        best_path = os.path.join(CHECKPOINT_DIR, "physics_multitask_best.pth")
        ckpt = torch.load(best_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

    # ====== 验证集评估 ======
    print("\n" + "=" * 50)
    print("  验证集评估 (Validation)")
    print("=" * 50)
    val_results, val_arrays = full_evaluation(
        model, val_loader, device, scaler_qsc, scaler_invc, scaler_foms
    )

    # ====== 测试集评估 ======
    print("\n" + "=" * 50)
    print("  测试集评估 (Test)")
    print("=" * 50)
    test_results, test_arrays = full_evaluation(
        model, test_loader, device, scaler_qsc, scaler_invc, scaler_foms
    )

    # ====== 保存测试集指标 ======
    if not args.debug:
        model_config = {
            "embed_dim": args.embed_dim,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lambda_consistency": args.lambda_consistency,
        }
        save_test_metrics(
            test_results,
            OUTPUT_DIR,
            model_config=model_config,
            n_test=len(test_loader.dataset),
            test_arrays=test_arrays,
        )

    # ====== 绘图 ======
    if not args.debug:
        print("\n[可视化] 生成评估图表...")

        plot_parity(
            test_arrays["true_qsc"],
            test_arrays["pred_qsc"],
            "Qsc_MACRS: True vs Predicted",
            os.path.join(OUTPUT_DIR, "parity_qsc.png"),
            metrics=test_results["qsc"],
        )
        plot_parity(
            test_arrays["true_invc"],
            test_arrays["pred_invc"],
            "invC_sum: True vs Predicted",
            os.path.join(OUTPUT_DIR, "parity_invc_sum.png"),
            metrics=test_results["invc"],
        )
        plot_parity(
            test_arrays["true_foms"],
            test_arrays["pred_foms_direct"],
            "FOMS_direct: True vs Predicted",
            os.path.join(OUTPUT_DIR, "parity_foms_direct.png"),
            metrics=test_results["foms_direct"],
        )
        plot_parity(
            test_arrays["true_foms"],
            test_arrays["pred_foms_phys"],
            "FOMS_phys: True vs Predicted",
            os.path.join(OUTPUT_DIR, "parity_foms_phys.png"),
            metrics=test_results["foms_phys"],
        )
        plot_parity(
            test_arrays["pred_foms_direct"],
            test_arrays["pred_foms_phys"],
            "FOMS_direct vs FOMS_phys (Consistency)",
            os.path.join(OUTPUT_DIR, "direct_vs_phys_consistency.png"),
        )

        # Log-scale parity plots for multi-order quantities
        foms_log_limits = compute_shared_log_limits(
            test_arrays["true_foms"],
            test_arrays["pred_foms_direct"],
            test_arrays["pred_foms_phys"],
        )
        for name, true_key, pred_key, res_key in [
            ("Qsc_MACRS", "true_qsc", "pred_qsc", "qsc"),
            ("FOMS_direct", "true_foms", "pred_foms_direct", "foms_direct"),
            ("FOMS_phys", "true_foms", "pred_foms_phys", "foms_phys"),
        ]:
            title = f"{name}: True vs Predicted (log)"
            axis_limits = None
            annotation_loc = (0.05, 0.95)
            metrics_payload = test_results[res_key]
            if name == "FOMS_direct":
                title = ""
                axis_limits = foms_log_limits
                metrics_payload = None
            elif name == "FOMS_phys":
                title = ""
                axis_limits = foms_log_limits
                metrics_payload = None
            plot_parity(
                test_arrays[true_key],
                test_arrays[pred_key],
                title,
                os.path.join(OUTPUT_DIR, f"parity_{res_key}_log.png"),
                metrics=metrics_payload,
                log_scale=True,
                axis_limits=axis_limits,
                annotation_loc=annotation_loc,
                annotation_keys=["r2_log10", "mae_log10"],
            )
        plot_parity(
            test_arrays["pred_foms_direct"],
            test_arrays["pred_foms_phys"],
            "",
            os.path.join(OUTPUT_DIR, "direct_vs_phys_consistency_log.png"),
            metrics=None,
            log_scale=True,
            axis_limits=foms_log_limits,
            annotation_loc=(0.05, 0.95),
            annotation_keys=["r2_log10", "mae_log10"],
        )

        plot_loss_curves(
            history,
            os.path.join(OUTPUT_DIR, "loss_curves_multitask.png"),
        )

    print("\n" + "=" * 65)
    print("  训练完成！")
    print("=" * 65)
    print(f"  权重:   {CHECKPOINT_DIR}/")
    print(f"  Scaler: {ARTIFACT_DIR}/")
    print(f"  输出:   {OUTPUT_DIR}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
