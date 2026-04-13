#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务机制景观分析 - 工具函数模块
========================================================
提供机制空间分析所需的全部工具函数，包括:
  1. 安全 log 计算
  2. 数据加载与有效性过滤
  3. 归一化指标构造 (FOMS/n, FOMS/(n*epsilon))
  4. 高密度支持区 masking (基于真实数据)
  5. 局部回归 regime 判定 (含局部标准化)
  6. 参数窗口图辅助工具 (预埋)

与旧版 experiment_4_phase_diagram.py 完全独立。
所有核心物理假设在 docstring 中明确说明。

"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from scipy.stats import pearsonr, spearmanr


# ============================================================================
# 1. 安全 log 计算 — 统一封装，避免 log(0)
# ============================================================================

def safe_log10(x, eps=1e-30):
    """
    计算 log10，对零和负值做安全下限保护。

    物理目的: 将原始物理量 (Qsc^2, invC_sum, FOMS 等) 转换到对数空间，
    用于机制景观分析。eps 远小于任何真实物理量，不会影响实际计算。

    Args:
        x: 数值或数组
        eps: 下限保护值 (默认 1e-30)

    Returns:
        log10(max(x, eps))
    """
    return np.log10(np.maximum(np.asarray(x, dtype=np.float64), eps))


# ============================================================================
# 2. 数据加载与有效性过滤
# ============================================================================

# epsilon 列名候选列表 (按优先级排序)
_EPSILON_COL_CANDIDATES = ["E", "epsilon", "eps_r", "er", "dielectric"]


def _find_epsilon_column(df):
    """
    自动识别 epsilon (介电常数) 列名。

    按候选列表顺序匹配。若找不到任何匹配列，raise KeyError。

    Args:
        df: pd.DataFrame

    Returns:
        str: 匹配到的列名

    Raises:
        KeyError: 未找到任何 epsilon 列
    """
    for col in _EPSILON_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"未找到 epsilon 列! 候选列名: {_EPSILON_COL_CANDIDATES}\n"
        f"实际列名: {list(df.columns)}\n"
        f"请在数据中提供介电常数列，或在候选列表中添加对应列名。"
    )


def load_real_data(csv_path, verbose=True):
    """
    加载真实数据 (ground-truth)，进行有效性过滤并构造衍生列。

    必须包含的列: n, inv_C_start, inv_C_end, Qsc_MACRS, FOMS
    必须包含 epsilon 列 (自动识别: E, epsilon, eps_r 等)

    衍生列:
        - invC_sum = inv_C_start + inv_C_end
        - logQ     = log10(Qsc_MACRS^2)
        - logInvC  = log10(invC_sum)
        - logFOMS  = log10(FOMS)
        - FOMS_norm_n   = FOMS / n
        - FOMS_norm_ne  = FOMS / (n * epsilon)
        - logFOMS_norm_n  = log10(FOMS_norm_n)
        - logFOMS_norm_ne = log10(FOMS_norm_ne)

    有效性过滤条件:
        - Qsc_MACRS > 0
        - inv_C_start > 0
        - inv_C_end > 0
        - invC_sum > 0
        - FOMS > 0 (排除恰好为零的极端情况，log10 不可用)

    Args:
        csv_path: 数据文件路径
        verbose: 是否打印加载信息

    Returns:
        df: pd.DataFrame, 过滤后的数据 (含所有衍生列)
        epsilon_col: str, 识别到的 epsilon 列名
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    n_raw = len(df)

    # 检查必须列
    required_base = ["n", "inv_C_start", "inv_C_end", "Qsc_MACRS", "FOMS"]
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必须列: {missing}. 实际列: {list(df.columns)}")

    # 识别 epsilon 列
    epsilon_col = _find_epsilon_column(df)

    # 构造 invC_sum
    df["invC_sum"] = df["inv_C_start"] + df["inv_C_end"]

    # 有效性过滤
    mask = (
        (df["Qsc_MACRS"] > 0) &
        (df["inv_C_start"] > 0) &
        (df["inv_C_end"] > 0) &
        (df["invC_sum"] > 0) &
        (df["FOMS"] > 0) &
        (df[epsilon_col] > 0) &
        (df["n"] > 0)
    )
    df = df[mask].copy().reset_index(drop=True)
    n_valid = len(df)

    # 机制空间坐标
    df["logQ"] = safe_log10(df["Qsc_MACRS"].values ** 2)
    df["logInvC"] = safe_log10(df["invC_sum"].values)
    df["logFOMS"] = safe_log10(df["FOMS"].values)

    # 归一化指标
    # 物理目的:
    #   SI 公式中 FOMS 正比于 n (通过 term_const)。
    #   为了比较 Qsc^2 与 invC_sum 两个"机制通道"的局部相对主导性，
    #   先去掉 SI 公式中的显式前因子 n (方案A) 或 n*epsilon (方案B)。
    #   这不是说 n 或 epsilon 不重要，而是为机制比较做归一化。
    df["FOMS_norm_n"] = df["FOMS"] / df["n"]
    df["FOMS_norm_ne"] = df["FOMS"] / (df["n"] * df[epsilon_col])
    df["logFOMS_norm_n"] = safe_log10(df["FOMS_norm_n"].values)
    df["logFOMS_norm_ne"] = safe_log10(df["FOMS_norm_ne"].values)

    if verbose:
        print(f"[数据加载] 文件: {csv_path}")
        print(f"  原始样本数: {n_raw}")
        print(f"  有效样本数: {n_valid} (过滤 {n_raw - n_valid} 条)")
        print(f"  epsilon 列名: '{epsilon_col}'")
        print(f"  FOMS 范围: [{df['FOMS'].min():.6e}, {df['FOMS'].max():.6e}]")
        print(f"  Qsc_MACRS 范围: [{df['Qsc_MACRS'].min():.6e}, {df['Qsc_MACRS'].max():.6e}]")
        print(f"  invC_sum 范围: [{df['invC_sum'].min():.6e}, {df['invC_sum'].max():.6e}]")
        print(f"  logQ 范围: [{df['logQ'].min():.2f}, {df['logQ'].max():.2f}]")
        print(f"  logInvC 范围: [{df['logInvC'].min():.2f}, {df['logInvC'].max():.2f}]")
        print(f"  n 值分布: {sorted(df['n'].unique())}")
        print(f"  {epsilon_col} 值分布: {sorted(df[epsilon_col].unique())}")

    return df, epsilon_col


# ============================================================================
# 3. 高密度支持区 masking — 基于真实数据
# ============================================================================

def compute_support_mask(
    logQ_data, logInvC_data,
    grid_x, grid_y,
    method="knn",
    k_neighbors=15,
    min_density_threshold=5,
    radius=None,
):
    """
    在 (logQ, logInvC) 网格上构造真实数据高密度支持区 mask。

    只有真实数据支持的区域才允许做 regime 判定。
    低支持区留白 (insufficient support)。

    支持区只由真实数据决定，不允许使用模型预测或增强数据。

    Args:
        logQ_data:    真实数据的 logQ 数组
        logInvC_data: 真实数据的 logInvC 数组
        grid_x:       网格横轴坐标
        grid_y:       网格纵轴坐标
        method:       支持区识别方法 ("knn" 或 "radius")
        k_neighbors:  kNN 方法的邻居数
        min_density_threshold: 最低支持点数阈值
        radius:       固定半径方法的搜索半径 (若 method="radius")

    Returns:
        support_mask: 2D bool array (len(grid_y), len(grid_x))
                      True = 高支持区, False = 低支持区
        density_map:  2D float array, 各网格点的局部数据密度
    """
    coords = np.column_stack([logQ_data, logInvC_data])
    tree = BallTree(coords)

    ny, nx = len(grid_y), len(grid_x)
    density_map = np.zeros((ny, nx))

    for j, gy in enumerate(grid_y):
        for i, gx in enumerate(grid_x):
            query = np.array([[gx, gy]])

            if method == "knn":
                # 查询 k_neighbors 个最近邻的最远距离
                # 密度 = k / 最远邻距离的面积
                dist, _ = tree.query(query, k=min(k_neighbors, len(coords)))
                # 用 k 邻域内的实际点数和平均距离作为密度代理
                # 这里直接用 k 个邻居中在合理距离内的数量
                max_dist = dist[0, -1]
                # 用一个参考距离来判断支持是否充足
                # 参考距离 = 数据范围的一个合理比例
                count_in_range = len(dist[0])
                density_map[j, i] = count_in_range / (max_dist + 1e-10)

            elif method == "radius":
                if radius is None:
                    # 自动设定半径为数据范围的 5%
                    range_x = logQ_data.max() - logQ_data.min()
                    range_y = logInvC_data.max() - logInvC_data.min()
                    radius = 0.05 * max(range_x, range_y)
                count = tree.query_radius(query, r=radius, count_only=True)[0]
                density_map[j, i] = count

    # 支持区判定
    if method == "knn":
        # 使用密度百分位阈值: 密度低于中位数的 30% 则认为不足
        threshold = np.percentile(density_map[density_map > 0], 20)
        support_mask = density_map >= threshold
    else:
        support_mask = density_map >= min_density_threshold

    return support_mask, density_map


# ============================================================================
# 4. 方差占比法 regime 判定
# ============================================================================

# Regime 编码
REGIME_CHARGE = 1        # charge-dominant
REGIME_CAPACITANCE = -1  # capacitance-dominant
REGIME_MIXED = 0         # mixed / transitional
REGIME_UNKNOWN = np.nan  # insufficient support


def compute_regime_map_variance(
    logQ, logInvC,
    k_neighbors=50,
    grid_resolution=50,
    dominance_threshold=0.62,
    min_local_points=10,
    support_mask=None,
):
    """
    基于局部方差占比的 regime 判定 (替代局部回归法)。

    物理依据:
        FOMS/n ∝ Qsc² × invC_sum
        在 log 空间: log(FOMS/n) = log(Qsc²) + log(invC_sum) + const
        即 logQ 和 logInvC 的系数**完全相等** (均为 1)。

    因此, "哪个通道主导 FOMS 变异" 完全取决于该通道在局部邻域中的**变异幅度**:
        f_charge = std(logQ) / (std(logQ) + std(logInvC))

    这避免了旧版局部回归法因动态范围差异 (logQ 跨 13 个数量级 vs logInvC 跨 2.5 个)
    导致回归系数系统性偏向 charge-dominant 的问题。

    Args:
        logQ:                log10(Qsc_MACRS^2)
        logInvC:             log10(invC_sum)
        k_neighbors:         局部邻域的近邻数
        grid_resolution:     网格分辨率 (每轴)
        dominance_threshold: 主导性阈值 (>threshold 为 charge, <1-threshold 为 cap)
        min_local_points:    最少邻域点数
        support_mask:        2D bool array, 高支持区 mask (可选)

    Returns:
        grid_x, grid_y: 1D 网格坐标
        regime_map:      2D regime 编码 (1/-1/0/NaN)
        f_charge_map:    2D charge 变异占比
    """
    coords = np.column_stack([logQ, logInvC])
    tree = BallTree(coords)

    pad = 0.05
    x_min, x_max = logQ.min(), logQ.max()
    y_min, y_max = logInvC.min(), logInvC.max()
    x_pad = (x_max - x_min) * pad
    y_pad = (y_max - y_min) * pad

    grid_x = np.linspace(x_min - x_pad, x_max + x_pad, grid_resolution)
    grid_y = np.linspace(y_min - y_pad, y_max + y_pad, grid_resolution)

    regime = np.full((grid_resolution, grid_resolution), np.nan)
    f_charge_map = np.full((grid_resolution, grid_resolution), np.nan)

    k = min(k_neighbors, len(logQ))

    for j, gy in enumerate(grid_y):
        for i, gx in enumerate(grid_x):
            if support_mask is not None and not support_mask[j, i]:
                continue

            query = np.array([[gx, gy]])
            dist, idx = tree.query(query, k=k)
            idx = idx[0]

            if len(idx) < min_local_points:
                continue

            local_logQ = logQ[idx]
            local_logInvC = logInvC[idx]

            std_q = np.std(local_logQ)
            std_c = np.std(local_logInvC)

            denom = std_q + std_c
            if denom < 1e-12:
                regime[j, i] = REGIME_MIXED
                f_charge_map[j, i] = 0.5
                continue

            f_charge = std_q / denom
            f_charge_map[j, i] = f_charge

            if f_charge > dominance_threshold:
                regime[j, i] = REGIME_CHARGE
            elif f_charge < 1.0 - dominance_threshold:
                regime[j, i] = REGIME_CAPACITANCE
            else:
                regime[j, i] = REGIME_MIXED

    return grid_x, grid_y, regime, f_charge_map


# ============================================================================
# 4b. 设计空间 regime 分析 — 基于模型梯度分解
# ============================================================================

def compute_design_regime_grid(
    predict_fn,
    E_fixed,
    dd_fixed,
    n_values,
    hh_values,
    delta_log=0.02,
    dominance_threshold=0.62,
):
    """
    在 (n, hh) 设计参数空间中计算 regime 分布。

    物理思路:
        FOMS ∝ n × Qsc² × invC_sum
        对设计者而言, 关键问题是: 调节 n 或 hh 时, FOMS 的改善
        主要通过 Qsc 通道还是 invC 通道实现?

    方法:
        在每个 (n, hh) 点, 用有限差分计算:
        - ∂log(Qsc)/∂log(n), ∂log(Qsc)/∂log(hh)
        - ∂log(invC)/∂log(n), ∂log(invC)/∂log(hh)

        通过 FOMS 公式链式法则, charge 通道的灵敏度梯度范数为:
            S_charge = sqrt( (2·∂logQsc/∂logn)² + (2·∂logQsc/∂loghh)² )
            (因子 2 来自 FOMS ∝ Qsc²)

        capacitance 通道的灵敏度梯度范数为:
            S_cap = sqrt( (∂logInvC/∂logn)² + (∂logInvC/∂loghh)² )

        f_charge = S_charge / (S_charge + S_cap)

    Args:
        predict_fn:  callable(n_arr, E_arr, dd_arr, hh_arr) -> dict
                     返回 {'Qsc_MACRS': ..., 'invC_sum': ..., 'FOMS_direct': ...}
        E_fixed:     固定介电常数
        dd_fixed:    固定间隙参数
        n_values:    1D array, n 采样值 (如 [2, 4, 8, 16, 32, 64])
        hh_values:   1D array, hh 采样值 (geomspace)
        delta_log:   有限差分步长 (log 空间)
        dominance_threshold: 主导性阈值

    Returns:
        dict: {
            'foms':         2D (nh, nn),  FOMS 预测值
            'qsc':          2D, Qsc 预测值
            'invc':         2D, invC 预测值
            'regime':       2D, regime 编码
            'f_charge':     2D, charge 灵敏度占比
            'charge_sens':  2D, charge 通道灵敏度范数
            'cap_sens':     2D, cap 通道灵敏度范数
        }
    """
    nn_len = len(n_values)
    nh_len = len(hh_values)

    N_grid, HH_grid = np.meshgrid(n_values, hh_values)  # (nh, nn)
    n_flat = N_grid.flatten().astype(np.float64)
    hh_flat = HH_grid.flatten().astype(np.float64)
    E_flat = np.full_like(n_flat, E_fixed)
    dd_flat = np.full_like(n_flat, dd_fixed)

    # 基准预测
    base = predict_fn(n_flat, E_flat, dd_flat, hh_flat)
    qsc_base = np.maximum(base["Qsc_MACRS"], 1e-30)
    invc_base = np.maximum(base["invC_sum"], 1e-30)
    foms_base = np.maximum(base["FOMS_direct"], 1e-30)

    # 扰动 n (log 空间)
    factor = 10 ** delta_log
    pred_n_plus = predict_fn(n_flat * factor, E_flat, dd_flat, hh_flat)
    pred_n_minus = predict_fn(n_flat / factor, E_flat, dd_flat, hh_flat)

    dlogQsc_dlogn = (
        np.log10(np.maximum(pred_n_plus["Qsc_MACRS"], 1e-30))
        - np.log10(np.maximum(pred_n_minus["Qsc_MACRS"], 1e-30))
    ) / (2 * delta_log)
    dlogInvC_dlogn = (
        np.log10(np.maximum(pred_n_plus["invC_sum"], 1e-30))
        - np.log10(np.maximum(pred_n_minus["invC_sum"], 1e-30))
    ) / (2 * delta_log)

    # 扰动 hh (log 空间)
    pred_hh_plus = predict_fn(n_flat, E_flat, dd_flat, hh_flat * factor)
    pred_hh_minus = predict_fn(n_flat, E_flat, dd_flat, hh_flat / factor)

    dlogQsc_dloghh = (
        np.log10(np.maximum(pred_hh_plus["Qsc_MACRS"], 1e-30))
        - np.log10(np.maximum(pred_hh_minus["Qsc_MACRS"], 1e-30))
    ) / (2 * delta_log)
    dlogInvC_dloghh = (
        np.log10(np.maximum(pred_hh_plus["invC_sum"], 1e-30))
        - np.log10(np.maximum(pred_hh_minus["invC_sum"], 1e-30))
    ) / (2 * delta_log)

    # 灵敏度范数 (FOMS ∝ Qsc², 所以 charge 通道乘 2)
    charge_sens = np.sqrt(
        (2 * dlogQsc_dlogn) ** 2 + (2 * dlogQsc_dloghh) ** 2
    ).reshape(nh_len, nn_len)
    cap_sens = np.sqrt(
        dlogInvC_dlogn ** 2 + dlogInvC_dloghh ** 2
    ).reshape(nh_len, nn_len)

    total_sens = charge_sens + cap_sens + 1e-30
    f_charge_map = charge_sens / total_sens

    regime_map = np.full((nh_len, nn_len), REGIME_MIXED, dtype=float)
    regime_map[f_charge_map > dominance_threshold] = REGIME_CHARGE
    regime_map[f_charge_map < 1.0 - dominance_threshold] = REGIME_CAPACITANCE

    return {
        "foms": foms_base.reshape(nh_len, nn_len),
        "qsc": qsc_base.reshape(nh_len, nn_len),
        "invc": invc_base.reshape(nh_len, nn_len),
        "regime": regime_map,
        "f_charge": f_charge_map,
        "charge_sens": charge_sens,
        "cap_sens": cap_sens,
    }


# ============================================================================
# 4c. 鲁棒性分析 — 参数扰动下的性能稳定性
# ============================================================================

def compute_robustness_grid(
    predict_fn,
    E_fixed,
    dd_fixed,
    n_values,
    hh_values,
    perturb_frac=0.1,
):
    """
    在 (n, hh) 设计空间中计算 FOMS 对参数扰动的鲁棒性。

    对每个设计点, 施加 dd 和 hh 的 ±perturb_frac 扰动 (4 个角 + 基准 = 5 个样本),
    计算 FOMS 的变异系数 (CV = std/mean) 作为鲁棒性度量。

    CV 低的区域意味着：即使加工精度有 ±10% 偏差, FOMS 依然稳定。
    这是材料学论文中审稿人重视的"可制造性"指标。

    Args:
        predict_fn:   callable, 预测函数
        E_fixed:      固定介电常数
        dd_fixed:     固定间隙参数 (作为基准, 也被扰动)
        n_values:     1D array, n 采样值
        hh_values:    1D array, hh 采样值
        perturb_frac: 扰动幅度 (如 0.1 = ±10%)

    Returns:
        foms_map: 2D (nh, nn), 基准 FOMS
        cv_map:   2D (nh, nn), 变异系数
        worst_ratio_map: 2D, 最差情况下 FOMS / 基准 FOMS
    """
    nn_len = len(n_values)
    nh_len = len(hh_values)

    N_grid, HH_grid = np.meshgrid(n_values, hh_values)
    n_flat = N_grid.flatten().astype(np.float64)
    hh_flat = HH_grid.flatten().astype(np.float64)
    E_flat = np.full_like(n_flat, E_fixed)
    dd_flat = np.full_like(n_flat, dd_fixed)

    # 基准
    base = predict_fn(n_flat, E_flat, dd_flat, hh_flat)
    foms_base = base["FOMS_direct"]

    # 收集所有扰动样本
    foms_samples = [foms_base]
    for sign_h in [-1, 1]:
        for sign_d in [-1, 1]:
            hh_p = hh_flat * (1 + sign_h * perturb_frac)
            dd_p = dd_flat * (1 + sign_d * perturb_frac)
            pred = predict_fn(n_flat, E_flat, dd_p, hh_p)
            foms_samples.append(pred["FOMS_direct"])

    foms_stack = np.stack(foms_samples, axis=0)  # (5, N)
    mean_foms = np.mean(foms_stack, axis=0)
    std_foms = np.std(foms_stack, axis=0)
    cv = std_foms / (mean_foms + 1e-30)
    worst = np.min(foms_stack, axis=0)
    worst_ratio = worst / (foms_base + 1e-30)

    return (
        foms_base.reshape(nh_len, nn_len),
        cv.reshape(nh_len, nn_len),
        worst_ratio.reshape(nh_len, nn_len),
    )


def print_regime_statistics(regime_map, label=""):
    """
    打印 regime 分析统计。

    Args:
        regime_map: 2D array, regime 编码
        label: 标签 (如 "FOMS/n" 或 "FOMS/(n*epsilon)")
    """
    valid = ~np.isnan(regime_map)
    n_total = valid.sum()
    if n_total == 0:
        print(f"  [{label}] 无有效网格点")
        return

    n_charge = np.sum(regime_map[valid] == REGIME_CHARGE)
    n_cap = np.sum(regime_map[valid] == REGIME_CAPACITANCE)
    n_mixed = np.sum(regime_map[valid] == REGIME_MIXED)
    n_insuff = np.sum(np.isnan(regime_map))

    print(f"  [{label}] Regime 统计 ({n_total} 有效网格点):")
    print(f"    Charge-dominant:     {n_charge:4d} ({100 * n_charge / n_total:5.1f}%)")
    print(f"    Capacitance-dominant:{n_cap:4d} ({100 * n_cap / n_total:5.1f}%)")
    print(f"    Mixed/transitional:  {n_mixed:4d} ({100 * n_mixed / n_total:5.1f}%)")
    print(f"    Insufficient support:{n_insuff:4d}")

    return {
        "label": label,
        "n_total": int(n_total),
        "n_charge": int(n_charge),
        "n_capacitance": int(n_cap),
        "n_mixed": int(n_mixed),
        "n_insufficient": int(n_insuff),
        "pct_charge": float(100 * n_charge / n_total),
        "pct_capacitance": float(100 * n_cap / n_total),
        "pct_mixed": float(100 * n_mixed / n_total),
    }


# ============================================================================
# 5. 全局趋势描述 (仅用于日志输出)
# ============================================================================

def print_global_correlations(df, targets=None):
    """
    打印 logQ, logInvC 与各目标之间的全局相关性 (Pearson, Spearman)。

    仅作描述性参考，不应用于局部机制判定。

    Args:
        df: 包含 logQ, logInvC 及目标列的 DataFrame
        targets: 目标列名列表 (默认: logFOMS, logFOMS_norm_n, logFOMS_norm_ne)
    """
    if targets is None:
        targets = ["logFOMS", "logFOMS_norm_n", "logFOMS_norm_ne"]

    print("\n" + "=" * 60)
    print("全局趋势描述 (仅供参考，不替代局部 regime 判定)")
    print("=" * 60)

    for tgt in targets:
        if tgt not in df.columns:
            continue
        r_q, p_q = pearsonr(df["logQ"], df[tgt])
        r_c, p_c = pearsonr(df["logInvC"], df[tgt])
        rho_q, _ = spearmanr(df["logQ"], df[tgt])
        rho_c, _ = spearmanr(df["logInvC"], df[tgt])

        print(f"\n  {tgt}:")
        print(f"    vs logQ:    Pearson r={r_q:+.4f} (p={p_q:.2e}), Spearman rho={rho_q:+.4f}")
        print(f"    vs logInvC: Pearson r={r_c:+.4f} (p={p_c:.2e}), Spearman rho={rho_c:+.4f}")

    print("-" * 60)


# ============================================================================
# 6. 参数窗口图辅助工具 (预埋)
# ============================================================================

def get_discrete_param_levels(df, param_col, n_levels=None):
    """
    获取参数的离散采样位置。

    n, dd, hh 等参数在 disk TENG 设计中本身是离散的，
    不建议机械线性拉伸，而应按实际离散值展示。

    Args:
        df:        DataFrame
        param_col: 参数列名 (如 "n", "dd", "hh")
        n_levels:  若指定，从实际值中均匀采样 n_levels 个

    Returns:
        levels: sorted array of unique parameter values
    """
    levels = np.sort(df[param_col].unique())
    if n_levels is not None and len(levels) > n_levels:
        # 均匀采样索引
        idx = np.linspace(0, len(levels) - 1, n_levels, dtype=int)
        levels = levels[idx]
    return levels


def make_log_ticks(values):
    """
    为对数采样的参数值生成 tick 位置和标签。

    适用于 n, dd, hh 等跨多个数量级的参数。
    tick 位置使用 log2 或 log10，标签显示真实值。

    Args:
        values: 真实参数值数组

    Returns:
        tick_positions: 对数空间中的位置
        tick_labels:    对应的真实值字符串
    """
    values = np.sort(np.unique(values))
    # 使用 log2 (对 n, dd, hh 更自然)
    tick_positions = np.log2(values)
    tick_labels = []
    for v in values:
        if v >= 1:
            tick_labels.append(f"{v:g}")
        else:
            tick_labels.append(f"{v:.4g}")
    return tick_positions, tick_labels


def filter_by_params(df, fixed_params):
    """
    按固定参数条件过滤数据。

    用于后续参数窗口图: 固定某些参数，观察剩余参数空间中的机制结构。

    Args:
        df:           DataFrame
        fixed_params: dict, 如 {"E": 3, "dd": 0.25}
                      值可以是单值或列表 (允许多个)

    Returns:
        df_filtered: 过滤后的 DataFrame

    Example:
        df_sub = filter_by_params(df, {"E": 3, "hh": [0.0625, 0.125]})
    """
    mask = pd.Series(True, index=df.index)
    for col, val in fixed_params.items():
        if col not in df.columns:
            raise KeyError(f"过滤列 '{col}' 不存在于 DataFrame 中")
        if isinstance(val, (list, tuple, np.ndarray)):
            mask &= df[col].isin(val)
        else:
            mask &= (df[col] == val)
    return df[mask].copy()


# ============================================================================
# 7. 分析表导出
# ============================================================================

def export_analysis_table(df, epsilon_col, output_path):
    """
    导出机制分析汇总表。

    Args:
        df:          经过 load_real_data 处理后的 DataFrame
        epsilon_col: epsilon 列名
        output_path: 输出 CSV 路径
    """
    export_cols = [
        "n", epsilon_col, "dd", "hh",
        "Qsc_MACRS", "inv_C_start", "inv_C_end", "invC_sum",
        "FOMS", "FOMS_norm_n", "FOMS_norm_ne",
        "logQ", "logInvC", "logFOMS", "logFOMS_norm_n", "logFOMS_norm_ne",
    ]
    # 只导出存在的列
    cols = [c for c in export_cols if c in df.columns]
    df_out = df[cols].copy()

    # 重命名 epsilon 列为统一名称
    if epsilon_col != "epsilon":
        df_out = df_out.rename(columns={epsilon_col: "epsilon"})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False, float_format="%.10e")
    print(f"[导出] 分析表已保存: {output_path} ({len(df_out)} 行)")
