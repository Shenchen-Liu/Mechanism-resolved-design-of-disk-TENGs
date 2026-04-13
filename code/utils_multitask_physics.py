#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务物理一致模型 - 工具函数模块
========================================================
包含数据加载、FOMS_phys 物理公式计算、评估指标等独立工具函数。
与原版 main.py / utils.py 完全独立，不复用原有函数。

"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# ============================================================================
# 1. 物理常数（与 calculate_foms.py 保持一致）
# ============================================================================

EPSILON_0 = 8.854e-12    # 真空介电常数 (F/m)
SIGMA = 1e-5             # 表面电荷密度 (C/m^2)
R = 0.015                # 圆盘半径 (m)
PI = np.pi               # 圆周率
EFFECTIVE_AREA = 0.5 * PI * R**2  # 有效摩擦面积（圆盘面积的一半）
Q_TOTAL = SIGMA * EFFECTIVE_AREA  # 总摩擦电荷量


# ============================================================================
# 2. FOMS_phys 物理公式计算
# ============================================================================

def compute_foms_phys(Qsc_MACRS, invC_sum, n, E=None):
    """
    根据 calculate_foms.py 的物理组合公式，由中间物理量重构 FOMS。

    参考 calculate_foms.py 第 160-165 行:
        term_const = (n * EPSILON_0) / (SIGMA**2 * PI**2 * R**3)
        term_Q = Qsc_MACRS**2
        term_C = invC_start + invC_end   (即 invC_sum)
        FOMS = 2 * term_const * term_Q * term_C

    其中乘以 2 是对有效面积（圆盘面积的一半）的补偿系数。

    注意: E (介电常数) 在 calculate_foms.py 的 FOMS 公式中并未直接出现，
    它通过影响电容间接体现在 invC_start/invC_end 中。因此这里只需要 n。

    Args:
        Qsc_MACRS: 短路电荷差 (C), 标量或数组
        invC_sum:  倒电容之和 = invC_start + invC_end (1/F), 标量或数组
        n:         叶片对数 (无量纲整数), 标量或数组
        E:         介电常数 (未在公式中直接使用，保留接口兼容性)

    Returns:
        FOMS_phys: 与数据集中 FOMS 同量纲同尺度的物理重构值
    """
    # 参考 calculate_foms.py 的物理组合公式
    term_const = (n * EPSILON_0) / (SIGMA**2 * PI**2 * R**3)
    term_Q = Qsc_MACRS ** 2
    term_C = invC_sum
    FOMS_phys = 2.0 * term_const * term_Q * term_C  # 乘以2补偿有效面积
    return FOMS_phys


def compute_foms_phys_torch(Qsc_MACRS, invC_sum, n):
    """
    FOMS_phys 的 PyTorch 张量版本，用于训练时的一致性损失计算。

    参考 calculate_foms.py 的物理组合公式 (第 160-165 行):
        FOMS = 2 * (n * EPSILON_0) / (SIGMA^2 * PI^2 * R^3) * Qsc_MACRS^2 * invC_sum

    Args:
        Qsc_MACRS: torch.Tensor, 短路电荷差 (原始物理量级)
        invC_sum:  torch.Tensor, 倒电容之和 (原始物理量级)
        n:         torch.Tensor, 叶片对数 (原始值)

    Returns:
        FOMS_phys: torch.Tensor, 物理重构的 FOMS
    """
    term_const = (n * EPSILON_0) / (SIGMA**2 * PI**2 * R**3)
    term_Q = Qsc_MACRS ** 2
    term_C = invC_sum
    FOMS_phys = 2.0 * term_const * term_Q * term_C
    return FOMS_phys


def compute_log10_foms_phys_torch(log10_qsc, log10_invc, n):
    """
    在 log10 空间下直接计算 log10(FOMS_phys)，避免大数值精度问题。

    由物理公式:
        FOMS = 2 * (n * ε₀) / (σ² * π² * R³) * Qsc² * invC_sum

    取 log10:
        log10(FOMS) = log10(2 * ε₀ / (σ² * π² * R³))
                    + log10(n)
                    + 2 * log10(Qsc)
                    + log10(invC_sum)

    Args:
        log10_qsc:  torch.Tensor, log10(Qsc_MACRS)
        log10_invc: torch.Tensor, log10(invC_sum)
        n:          torch.Tensor, 叶片对数 (原始值，非 log)

    Returns:
        log10_foms_phys: torch.Tensor
    """
    # 预计算常数部分的 log10
    # 2 * ε₀ / (σ² * π² * R³)
    const = 2.0 * EPSILON_0 / (SIGMA**2 * PI**2 * R**3)
    log10_const = np.log10(const)  # 标量常数

    log10_foms = (
        log10_const
        + torch.log10(n)
        + 2.0 * log10_qsc
        + log10_invc
    )
    return log10_foms


# ============================================================================
# 3. 数据集类
# ============================================================================

class MultiTaskTENGDataset(Dataset):
    """
    多任务 TENG 数据集类

    存储输入特征和三个目标变量 (Qsc_MACRS, invC_sum, FOMS)。
    同时保留原始 n 值用于 FOMS_phys 的计算。
    """

    def __init__(self, features, targets_qsc, targets_invc, targets_foms, raw_n):
        """
        Args:
            features:     numpy array, shape (N, 4), 归一化后的输入特征
            targets_qsc:  numpy array, shape (N, 1), 归一化后的 Qsc_MACRS
            targets_invc: numpy array, shape (N, 1), 归一化后的 invC_sum
            targets_foms: numpy array, shape (N, 1), 归一化后的 FOMS
            raw_n:        numpy array, shape (N,), 原始 n 值 (用于 FOMS_phys 计算)
        """
        self.features = torch.FloatTensor(features)
        self.targets_qsc = torch.FloatTensor(targets_qsc)
        self.targets_invc = torch.FloatTensor(targets_invc)
        self.targets_foms = torch.FloatTensor(targets_foms)
        self.raw_n = torch.FloatTensor(raw_n)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.targets_qsc[idx],
            self.targets_invc[idx],
            self.targets_foms[idx],
            self.raw_n[idx],
        )


# ============================================================================
# 4. 数据加载与预处理
# ============================================================================

def load_multitask_data(
    data_path,
    experiment_mode="full",
    debug=False,
    test_size=0.2,
    val_ratio=0.5,
    batch_size=32,
    random_state=42,
):
    """
    加载并预处理多任务数据。

    与原版 main.py 的关键区别:
    1. 目标变量 = 3 个 (Qsc_MACRS, invC_sum, FOMS)
    2. 先划分 train/val/test，再仅在 train 上 fit scaler（避免数据泄漏）
    3. 每个目标变量单独归一化
    4. 使用原始特征值（raw 模式，不做 log 变换）

    输入特征: n, E, dd, hh (均为无量纲设计参数)
    - n:  叶片对数
    - E:  介电常数 (epsilon)
    - dd: 间隙比例 (无量纲)
    - hh: 高度比例 (无量纲)

    Args:
        data_path:        CSV 文件路径
        experiment_mode:  'full' (全量) 或 'reproduction' (dd <= 0.5)
        debug:            调试模式
        test_size:        train 之外的比例 (默认 0.2)
        val_ratio:        从 test_size 中划分 val 的比例 (默认 0.5, 即 val=10%, test=10%)
        batch_size:       批大小
        random_state:     随机种子

    Returns:
        train_loader, val_loader, test_loader,
        scaler_X, scaler_qsc, scaler_invc, scaler_foms,
        raw_n_train, raw_n_val, raw_n_test
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[数据] 原始样本数: {len(df)}")

    # 检查必要列
    feature_cols = ["n", "E", "dd", "hh"]
    target_cols = ["Qsc_MACRS", "inv_C_start", "inv_C_end", "FOMS"]
    required_cols = feature_cols + target_cols
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV 缺少必需列: {missing_cols}. 现有列: {df.columns.tolist()}")

    # 构造 invC_sum
    df["invC_sum"] = df["inv_C_start"] + df["inv_C_end"]

    # 实验模式过滤
    if experiment_mode == "reproduction":
        df = df[df["dd"] <= 0.5].reset_index(drop=True)
        print(f"[数据] reproduction 模式过滤后: {len(df)} 样本")

    # 过滤无效值
    valid_mask = (
        (df["Qsc_MACRS"] > 0) &
        (df["inv_C_start"] > 0) &
        (df["inv_C_end"] > 0) &
        (df["invC_sum"] > 0) &
        (df["FOMS"] >= 0)
    )
    df = df[valid_mask].reset_index(drop=True)
    print(f"[数据] 过滤无效值后有效样本数: {len(df)}")

    if debug:
        df = df.head(50)
        print(f"[数据] DEBUG 模式: 仅使用前 50 条")

    # 提取特征和目标
    X = df[feature_cols].values
    # 对目标变量取 log10 以压缩动态范围（Qsc 跨 ~7 个数量级, FOMS 跨 ~11 个数量级）
    # MinMaxScaler 直接作用于原始值会把 99% 的样本压到接近 0，模型无法学习
    # 过滤掉 FOMS==0 的极端情况（log10 不可用）
    df = df[df["FOMS"] > 0].reset_index(drop=True)
    X = df[feature_cols].values

    y_qsc = np.log10(df["Qsc_MACRS"].values).reshape(-1, 1)
    y_invc = np.log10(df["invC_sum"].values).reshape(-1, 1)
    y_foms = np.log10(df["FOMS"].values).reshape(-1, 1)
    raw_n = df["n"].values  # 保留原始 n 值用于 FOMS_phys 计算

    print(f"[数据] 目标已做 log10 变换: "
          f"Qsc=[{y_qsc.min():.2f},{y_qsc.max():.2f}], "
          f"invC=[{y_invc.min():.2f},{y_invc.max():.2f}], "
          f"FOMS=[{y_foms.min():.2f},{y_foms.max():.2f}]")

    # ====== 先划分 train/val/test，再 fit scaler ======
    indices = np.arange(len(X))
    idx_train, idx_temp = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=val_ratio, random_state=random_state
    )

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_qsc_train, y_qsc_val, y_qsc_test = y_qsc[idx_train], y_qsc[idx_val], y_qsc[idx_test]
    y_invc_train, y_invc_val, y_invc_test = y_invc[idx_train], y_invc[idx_val], y_invc[idx_test]
    y_foms_train, y_foms_val, y_foms_test = y_foms[idx_train], y_foms[idx_val], y_foms[idx_test]
    raw_n_train, raw_n_val, raw_n_test = raw_n[idx_train], raw_n[idx_val], raw_n[idx_test]

    print(f"[数据] 划分: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    # ====== 仅在 train 上 fit scaler ======
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    scaler_qsc = MinMaxScaler(feature_range=(0, 1))
    scaler_qsc.fit(y_qsc_train)
    y_qsc_train = scaler_qsc.transform(y_qsc_train)
    y_qsc_val = scaler_qsc.transform(y_qsc_val)
    y_qsc_test = scaler_qsc.transform(y_qsc_test)

    scaler_invc = MinMaxScaler(feature_range=(0, 1))
    scaler_invc.fit(y_invc_train)
    y_invc_train = scaler_invc.transform(y_invc_train)
    y_invc_val = scaler_invc.transform(y_invc_val)
    y_invc_test = scaler_invc.transform(y_invc_test)

    scaler_foms = MinMaxScaler(feature_range=(0, 1))
    scaler_foms.fit(y_foms_train)
    y_foms_train = scaler_foms.transform(y_foms_train)
    y_foms_val = scaler_foms.transform(y_foms_val)
    y_foms_test = scaler_foms.transform(y_foms_test)

    # 创建 DataLoader
    if debug:
        batch_size = 16

    train_ds = MultiTaskTENGDataset(X_train, y_qsc_train, y_invc_train, y_foms_train, raw_n_train)
    val_ds = MultiTaskTENGDataset(X_val, y_qsc_val, y_invc_val, y_foms_val, raw_n_val)
    test_ds = MultiTaskTENGDataset(X_test, y_qsc_test, y_invc_test, y_foms_test, raw_n_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return (
        train_loader, val_loader, test_loader,
        scaler_X, scaler_qsc, scaler_invc, scaler_foms,
        raw_n_train, raw_n_val, raw_n_test,
    )


# ============================================================================
# 5. Scaler 保存与加载
# ============================================================================

def save_scalers(scaler_X, scaler_qsc, scaler_invc, scaler_foms, save_dir):
    """保存所有 scaler 到独立文件"""
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X_multitask.pkl"))
    joblib.dump(scaler_qsc, os.path.join(save_dir, "scaler_y_qsc.pkl"))
    joblib.dump(scaler_invc, os.path.join(save_dir, "scaler_y_invc.pkl"))
    joblib.dump(scaler_foms, os.path.join(save_dir, "scaler_y_foms.pkl"))
    print(f"[保存] Scaler 已保存至 {save_dir}")


def load_scalers_multitask(save_dir):
    """加载多任务模型的所有 scaler"""
    scaler_X = joblib.load(os.path.join(save_dir, "scaler_X_multitask.pkl"))
    scaler_qsc = joblib.load(os.path.join(save_dir, "scaler_y_qsc.pkl"))
    scaler_invc = joblib.load(os.path.join(save_dir, "scaler_y_invc.pkl"))
    scaler_foms = joblib.load(os.path.join(save_dir, "scaler_y_foms.pkl"))
    return scaler_X, scaler_qsc, scaler_invc, scaler_foms


# ============================================================================
# 6. 评估指标
# ============================================================================

def compute_metrics(y_true, y_pred, name=""):
    """
    计算回归评估指标: MAE, RMSE, R², MAPE, MAE_log10

    Args:
        y_true: numpy array, 真实值
        y_pred: numpy array, 预测值
        name:   变量名称（用于打印）

    Returns:
        dict: {'mae': ..., 'rmse': ..., 'r2': ..., 'mape': ..., 'mae_log10': ...}
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE (%) with guard against zero-valued true values
    nonzero_mask = np.abs(y_true) > 1e-30
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0
    else:
        mape = float("inf")

    # Log10-space metrics — for multi-order-of-magnitude quantities
    positive_mask = (y_true > 0) & (y_pred > 0)
    if np.any(positive_mask):
        log_true = np.log10(y_true[positive_mask])
        log_pred = np.log10(y_pred[positive_mask])
        mae_log10 = float(np.mean(np.abs(log_true - log_pred)))
        # R² in log10 space — scale-invariant goodness-of-fit
        ss_res_log = np.sum((log_true - log_pred) ** 2)
        ss_tot_log = np.sum((log_true - np.mean(log_true)) ** 2)
        r2_log10 = float(1.0 - (ss_res_log / ss_tot_log)) if ss_tot_log > 0 else 0.0
    else:
        mae_log10 = float("inf")
        r2_log10 = 0.0

    if name:
        print(f"  {name}: R²={r2:.4f}, R²_log10={r2_log10:.4f}, "
              f"MAE_log10={mae_log10:.4f}, MAPE={mape:.2f}%")

    return {"mae": mae, "rmse": rmse, "r2": r2, "r2_log10": r2_log10,
            "mape": mape, "mae_log10": mae_log10}


def compute_consistency_metrics(foms_direct, foms_phys, name="FOMS consistency"):
    """
    计算 FOMS_direct 与 FOMS_phys 之间的一致性指标

    Args:
        foms_direct: numpy array, 模型直接预测的 FOMS
        foms_phys:   numpy array, 物理公式重构的 FOMS

    Returns:
        dict: {'mae': ..., 'pearson_r': ..., 'spearman_r': ...}
    """
    foms_direct = np.asarray(foms_direct).flatten()
    foms_phys = np.asarray(foms_phys).flatten()

    mae = np.mean(np.abs(foms_direct - foms_phys))

    # Pearson 相关系数
    if len(foms_direct) > 2:
        pearson_r, pearson_p = stats.pearsonr(foms_direct, foms_phys)
        spearman_r, spearman_p = stats.spearmanr(foms_direct, foms_phys)
    else:
        pearson_r = spearman_r = 0.0

    print(f"  {name}: MAE={mae:.6e}, Pearson={pearson_r:.4f}, Spearman={spearman_r:.4f}")

    return {
        "mae": mae,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
    }


# ============================================================================
# 7. 设备与随机种子
# ============================================================================

def setup_seed(seed=42):
    """设置随机种子以保证复现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """自动检测设备 (CPU/GPU/MPS)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
