#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务机制景观分析 - 绘图模块
========================================================
生成机制景观图、regime map、overlay 图、模型增强图等。
所有图片同时输出 full (全域) 和 zoom (高信息区) 两个版本。

图片类型:
  A. 真实数据机制景观图 (primary figure)
  B. Regime map (norm_n / norm_ne)
  C. Overlay (real data + regime background)
  D. 模型增强机制景观图 (supplementary)
  E. 一致性比较图 (optional)

与旧版 experiment_4_phase_diagram.py 完全独立。

"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import seaborn as sns

# ============================================================================
# PLOT STYLE
# ============================================================================
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "figure.titlesize": 18,
    "figure.dpi": 150,
})

# ============================================================================
# AXIS LABELS
# ============================================================================
XLABEL = r"$\log_{10}(Q_{sc,MACRS}^{\,2})$ — Charge-transfer term"
YLABEL = r"$\log_{10}(1/C_{start}+1/C_{end})$ — Capacitance term"
CBAR_FOMS = r"$FOM_S$"
CBAR_FOMS_NORM_N = r"$FOM_S / n$"
CBAR_FOMS_NORM_NE = r"$FOM_S / (n \cdot \varepsilon)$"

# Regime 颜色方案
REGIME_CMAP = ListedColormap(["#2980b9", "#bdc3c7", "#e74c3c"])
REGIME_BOUNDS = [-1.5, -0.5, 0.5, 1.5]
REGIME_NORM = BoundaryNorm(REGIME_BOUNDS, REGIME_CMAP.N)

# ============================================================================
# 通用绘图辅助
# ============================================================================

def _apply_zoom(ax, zoom_xlim):
    """对坐标轴应用 zoom 范围 (仅限 x 轴)。"""
    if zoom_xlim is not None:
        ax.set_xlim(zoom_xlim)


def _save_fig(fig, save_path, dpi=300):
    """统一保存图片。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"  [图片] {save_path}")
    plt.close(fig)


# ============================================================================
# A. 真实数据机制景观图
# ============================================================================

def plot_real_landscape(
    logQ, logInvC, foms,
    save_path,
    color_label=CBAR_FOMS,
    title="Mechanism Landscape from Ground-Truth TENG Data",
    zoom_xlim=None,
    dpi=300,
    figsize=(11, 8.5),
):
    """
    真实数据机制景观图: 散点图，颜色映射 FOMS (或归一化指标)。

    这是论文主图候选。只包含 ground-truth 数据，
    不含模型插值或平滑，读者可直接观察数据分布和 FOMS 梯度。

    Args:
        logQ:        log10(Qsc_MACRS^2)
        logInvC:     log10(invC_sum)
        foms:        颜色值 (FOMS 或归一化指标)
        save_path:   输出路径
        color_label: colorbar 标签
        title:       图标题
        zoom_xlim:   zoom 区间 (如 (-18.5, None))
        dpi:         图片分辨率
        figsize:     图片尺寸
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    vmax = np.percentile(foms, 99)
    sc = ax.scatter(
        logQ, logInvC,
        c=foms, cmap="viridis",
        s=45, edgecolors="black", linewidths=0.6,
        zorder=5, vmin=0, vmax=vmax, alpha=0.9,
        label="Ground Truth (COMSOL)",
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.015, shrink=0.88, aspect=30)
    cbar.set_label(color_label, fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel(XLABEL, fontsize=14)
    ax.set_ylabel(YLABEL, fontsize=14)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9,
              edgecolor="gray", fancybox=True)
    ax.tick_params(labelsize=11)

    _apply_zoom(ax, zoom_xlim)
    plt.tight_layout()
    _save_fig(fig, save_path, dpi)


# ============================================================================
# B. Regime Map
# ============================================================================

def plot_regime_map(
    grid_x, grid_y, regime_map,
    save_path,
    title="Charge- vs Capacitance-Dominant Regime Map",
    zoom_xlim=None,
    dpi=300,
    figsize=(10, 8),
):
    """
    Regime 判定图: 背景色表示 charge/capacitance/mixed 主导区。

    低支持区 (NaN) 显示为白色，图例中标注 "Insufficient support"。

    Args:
        grid_x:     网格 x 坐标
        grid_y:     网格 y 坐标
        regime_map: 2D regime 编码 (1/-1/0/NaN)
        save_path:  输出路径
        title:      图标题
        zoom_xlim:  zoom 区间
        dpi:        分辨率
        figsize:    图片尺寸
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    im = ax.pcolormesh(
        X_grid, Y_grid, regime_map,
        cmap=REGIME_CMAP, norm=REGIME_NORM,
        shading="nearest", alpha=0.85,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, aspect=25, pad=0.02)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["Capacitance\ndominant", "Mixed /\nTransitional", "Charge\ndominant"])
    cbar.ax.tick_params(labelsize=11)

    # 图例: 低支持区
    legend_elements = [
        Patch(facecolor="white", edgecolor="gray", label="Insufficient support"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10,
              framealpha=0.9, edgecolor="gray", fancybox=True)

    ax.set_xlabel(XLABEL, fontsize=14)
    ax.set_ylabel(YLABEL, fontsize=14)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.tick_params(labelsize=11)

    _apply_zoom(ax, zoom_xlim)
    plt.tight_layout()
    _save_fig(fig, save_path, dpi)


# ============================================================================
# C. Overlay: 真实数据 + regime 背景
# ============================================================================

def plot_overlay(
    logQ, logInvC, foms,
    grid_x, grid_y, regime_map,
    save_path,
    color_label=CBAR_FOMS,
    title="Mechanism Landscape with Regime Overlay",
    zoom_xlim=None,
    dpi=300,
    figsize=(11, 8.5),
):
    """
    真实数据散点 + regime 背景叠加。

    散点颜色映射 FOMS，背景半透明映射 regime 区。
    低支持区为白色。

    Args:
        logQ, logInvC, foms: 真实数据坐标与颜色
        grid_x, grid_y, regime_map: regime 网格
        save_path, color_label, title, zoom_xlim, dpi, figsize: 同上
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 背景: regime
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    ax.pcolormesh(
        X_grid, Y_grid, regime_map,
        cmap=REGIME_CMAP, norm=REGIME_NORM,
        shading="nearest", alpha=0.25, zorder=1,
    )

    # 前景: 真实数据
    vmax = np.percentile(foms, 99)
    sc = ax.scatter(
        logQ, logInvC,
        c=foms, cmap="viridis",
        s=45, edgecolors="black", linewidths=0.6,
        zorder=5, vmin=0, vmax=vmax, alpha=0.9,
        label="Ground Truth (COMSOL)",
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.015, shrink=0.88, aspect=30)
    cbar.set_label(color_label, fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    # 图例
    legend_elements = [
        sc.legend_elements()[0][0],
        Patch(facecolor="#e74c3c", alpha=0.4, label="Charge-dominant"),
        Patch(facecolor="#bdc3c7", alpha=0.4, label="Mixed"),
        Patch(facecolor="#2980b9", alpha=0.4, label="Capacitance-dominant"),
        Patch(facecolor="white", edgecolor="gray", alpha=0.4, label="Insufficient support"),
    ]
    ax.legend(
        handles=legend_elements,
        labels=["Ground Truth", "Charge-dominant", "Mixed",
                "Capacitance-dominant", "Insufficient support"],
        loc="upper right", fontsize=9, framealpha=0.9,
        edgecolor="gray", fancybox=True,
    )

    ax.set_xlabel(XLABEL, fontsize=14)
    ax.set_ylabel(YLABEL, fontsize=14)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(labelsize=11)

    _apply_zoom(ax, zoom_xlim)
    plt.tight_layout()
    _save_fig(fig, save_path, dpi)


# ============================================================================
# D. 模型增强机制景观图
# ============================================================================

def plot_model_landscape(
    logQ_model, logInvC_model, foms_model,
    logQ_real, logInvC_real, foms_real,
    save_path,
    color_label=CBAR_FOMS,
    title="Mechanism-Informed Performance Landscape from Multi-Task Model",
    zoom_xlim=None,
    dpi=300,
    figsize=(11, 8.5),
):
    """
    模型增强机制景观图: 模型预测点为半透明背景，真实数据叠加。

    NOTE: 这是"连续化辅助可视化"，不是严格 phase diagram。
    标题和图例中明确标注数据来源。

    Args:
        logQ_model, logInvC_model, foms_model: 模型预测坐标与 FOMS
        logQ_real, logInvC_real, foms_real: 真实数据坐标与 FOMS
        save_path, color_label, title, zoom_xlim, dpi, figsize: 同上
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    vmax = np.percentile(foms_real, 99)

    # 背景: 模型预测
    ax.scatter(
        logQ_model, logInvC_model,
        c=foms_model, cmap="viridis",
        s=8, alpha=0.3, vmin=0, vmax=vmax,
        zorder=2, label="Multi-task model predictions",
    )

    # 前景: 真实数据
    sc = ax.scatter(
        logQ_real, logInvC_real,
        c=foms_real, cmap="viridis",
        s=45, edgecolors="black", linewidths=0.6,
        zorder=5, vmin=0, vmax=vmax, alpha=0.9,
        label="Ground Truth (COMSOL)",
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.015, shrink=0.88, aspect=30)
    cbar.set_label(color_label, fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel(XLABEL, fontsize=14)
    ax.set_ylabel(YLABEL, fontsize=14)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
              edgecolor="gray", fancybox=True)
    ax.tick_params(labelsize=11)

    _apply_zoom(ax, zoom_xlim)
    plt.tight_layout()
    _save_fig(fig, save_path, dpi)


# ============================================================================
# E. FOMS_direct vs FOMS_phys 一致性图
# ============================================================================

def plot_consistency(
    foms_direct, foms_phys,
    save_path,
    title="Multi-Task Model: FOMS_direct vs FOMS_phys Consistency",
    dpi=300,
    figsize=(8, 8),
):
    """
    模型输出一致性图: FOMS_direct vs FOMS_phys 的散点对比。

    物理意义: FOMS_direct 是模型直接预测，FOMS_phys 是由模型预测的
    Qsc_MACRS 和 invC_sum 通过 SI 公式重构。两者一致说明模型内部
    对物理约束的遵守程度高。

    Args:
        foms_direct: 模型直接预测的 FOMS
        foms_phys:   物理重构的 FOMS
        save_path:   输出路径
        title, dpi, figsize: 同上
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 过滤有效值
    valid = (foms_direct > 0) & (foms_phys > 0)
    fd = foms_direct[valid]
    fp = foms_phys[valid]

    log_fd = np.log10(fd)
    log_fp = np.log10(fp)

    sc = ax.scatter(
        log_fd, log_fp,
        s=15, alpha=0.5, c="steelblue", edgecolors="none",
    )

    # 对角线
    lims = [
        min(log_fd.min(), log_fp.min()) - 0.5,
        max(log_fd.max(), log_fp.max()) + 0.5,
    ]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.7, label="y = x")

    # 相关性
    from scipy.stats import pearsonr, spearmanr
    r_p, _ = pearsonr(log_fd, log_fp)
    r_s, _ = spearmanr(log_fd, log_fp)
    ax.text(
        0.05, 0.92,
        f"Pearson r = {r_p:.4f}\nSpearman rho = {r_s:.4f}\nN = {len(fd)}",
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel(r"$\log_{10}(FOM_{S,direct})$", fontsize=14)
    ax.set_ylabel(r"$\log_{10}(FOM_{S,phys})$", fontsize=14)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    _save_fig(fig, save_path, dpi)


# Unicode plus-minus symbol
PLUS_MINUS = "\u00B1"


# ============================================================================
# F. 设计空间 Regime Map (n-h 平面)
# ============================================================================

def plot_design_regime(
    n_values, hh_values, foms_map, regime_map, f_charge_map,
    E_val, dd_val,
    save_path,
    dpi=300,
    figsize=(10, 8),
):
    """
    设计参数空间 regime map: (n, hh) 平面, 背景色 = regime, 等值线 = FOMS。

    这是论文"设计窗口"图的核心可视化:
    - 告诉设计者: 对于给定材料(E)和间隙(dd), 在 (n, h) 空间中
      哪里是 charge-limited, 哪里是 capacitance-limited
    - FOMS 等值线帮助定位最优设计区域
    """
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.8, figsize[1]),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.30})

    log2_n = np.log2(n_values)
    log2_hh = np.log2(hh_values)
    N_grid, HH_grid = np.meshgrid(log2_n, log2_hh)

    # --- 左图: Regime map + FOMS contour ---
    ax = axes[0]
    im = ax.pcolormesh(
        N_grid, HH_grid, regime_map,
        cmap=REGIME_CMAP, norm=REGIME_NORM,
        shading="nearest", alpha=0.6,
    )

    log_foms = np.log10(np.maximum(foms_map, 1e-30))
    cs = ax.contour(N_grid, HH_grid, log_foms, levels=8,
                    colors="black", linewidths=0.8, alpha=0.7)
    ax.clabel(cs, fontsize=8, fmt="%.1f")

    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.6, label="Charge-limited"),
        Patch(facecolor="#bdc3c7", alpha=0.6, label="Mixed"),
        Patch(facecolor="#2980b9", alpha=0.6, label="Capacitance-limited"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              framealpha=0.9, edgecolor="gray")

    ax.set_xlabel(r"$n$ (blade pairs)", fontsize=13)
    ax.set_ylabel(r"$h/R$ (blade height ratio)", fontsize=13)
    ax.set_title(
        f"Design Regime Map\n"
        r"$\varepsilon$" + f"={E_val}, d/R={dd_val}",
        fontsize=13, fontweight="bold",
    )

    ax.set_xticks(log2_n)
    ax.set_xticklabels([str(int(v)) for v in n_values], fontsize=10)
    hh_step = max(1, len(hh_values) // 6)
    hh_ticks = hh_values[::hh_step]
    ax.set_yticks(np.log2(hh_ticks))
    ax.set_yticklabels([f"{v:.4g}" for v in hh_ticks], fontsize=9)

    # --- 右图: f_charge 连续热力图 ---
    ax2 = axes[1]
    im2 = ax2.pcolormesh(
        N_grid, HH_grid, f_charge_map,
        cmap="RdBu_r", vmin=0.0, vmax=1.0,
        shading="nearest",
    )
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=25, pad=0.02)
    cbar2.set_label("Charge sensitivity fraction", fontsize=12)
    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar2.set_ticklabels(["0\n(Cap.)", "0.25", "0.5\n(Balanced)", "0.75", "1.0\n(Charge)"])

    cs2 = ax2.contour(N_grid, HH_grid, log_foms, levels=8,
                      colors="black", linewidths=0.6, alpha=0.5)
    ax2.clabel(cs2, fontsize=7, fmt="%.1f")

    ax2.set_xlabel(r"$n$ (blade pairs)", fontsize=13)
    ax2.set_ylabel(r"$h/R$ (blade height ratio)", fontsize=13)
    ax2.set_title(
        f"Charge-Sensitivity Fraction\n"
        r"$\varepsilon$" + f"={E_val}, d/R={dd_val}",
        fontsize=13, fontweight="bold",
    )

    ax2.set_xticks(log2_n)
    ax2.set_xticklabels([str(int(v)) for v in n_values], fontsize=10)
    ax2.set_yticks(np.log2(hh_ticks))
    ax2.set_yticklabels([f"{v:.4g}" for v in hh_ticks], fontsize=9)

    fig.suptitle(
        "Mechanism-Resolved Design Window for Disk TENG",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"  [图片] {save_path}")
    plt.close(fig)


# ============================================================================
# G. 鲁棒性 Map (n-h 平面)
# ============================================================================

def plot_robustness(
    n_values, hh_values, cv_map, foms_map, worst_ratio_map,
    E_val, dd_val, perturb_pct,
    save_path,
    dpi=300,
    figsize=(10, 8),
):
    """
    鲁棒性分析图: (n, hh) 平面, 颜色 = CV, 等值线 = FOMS。

    CV (变异系数) 低的区域: 加工误差 ±perturb_pct% 下 FOMS 仍稳定。
    """
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.8, figsize[1]),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.30})

    log2_n = np.log2(n_values)
    log2_hh = np.log2(hh_values)
    N_grid, HH_grid = np.meshgrid(log2_n, log2_hh)
    log_foms = np.log10(np.maximum(foms_map, 1e-30))

    # --- 左图: CV map ---
    ax = axes[0]
    cv_pct = cv_map * 100
    im = ax.pcolormesh(
        N_grid, HH_grid, cv_pct,
        cmap="YlOrRd", vmin=0, vmax=np.percentile(cv_pct, 95),
        shading="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25, pad=0.02)
    cbar.set_label("CV of FOMS (%)", fontsize=12)

    cs = ax.contour(N_grid, HH_grid, log_foms, levels=8,
                    colors="black", linewidths=0.6, alpha=0.5)
    ax.clabel(cs, fontsize=7, fmt="%.1f")

    ax.set_xlabel(r"$n$ (blade pairs)", fontsize=13)
    ax.set_ylabel(r"$h/R$ (blade height ratio)", fontsize=13)
    ax.set_title(
        f"Robustness: CV under {PLUS_MINUS}{perturb_pct}% perturbation\n"
        r"$\varepsilon$" + f"={E_val}, d/R={dd_val}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(log2_n)
    ax.set_xticklabels([str(int(v)) for v in n_values], fontsize=10)
    hh_step = max(1, len(hh_values) // 6)
    hh_ticks = hh_values[::hh_step]
    ax.set_yticks(np.log2(hh_ticks))
    ax.set_yticklabels([f"{v:.4g}" for v in hh_ticks], fontsize=9)

    # --- 右图: worst-case ratio ---
    ax2 = axes[1]
    im2 = ax2.pcolormesh(
        N_grid, HH_grid, worst_ratio_map * 100,
        cmap="RdYlGn", vmin=50, vmax=100,
        shading="nearest",
    )
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=25, pad=0.02)
    cbar2.set_label("Worst-case FOMS / Baseline (%)", fontsize=12)

    cs2 = ax2.contour(N_grid, HH_grid, log_foms, levels=8,
                      colors="black", linewidths=0.6, alpha=0.5)
    ax2.clabel(cs2, fontsize=7, fmt="%.1f")

    ax2.set_xlabel(r"$n$ (blade pairs)", fontsize=13)
    ax2.set_ylabel(r"$h/R$ (blade height ratio)", fontsize=13)
    ax2.set_title(
        f"Worst-Case Retention under {PLUS_MINUS}{perturb_pct}%\n"
        r"$\varepsilon$" + f"={E_val}, d/R={dd_val}",
        fontsize=13, fontweight="bold",
    )
    ax2.set_xticks(log2_n)
    ax2.set_xticklabels([str(int(v)) for v in n_values], fontsize=10)
    ax2.set_yticks(np.log2(hh_ticks))
    ax2.set_yticklabels([f"{v:.4g}" for v in hh_ticks], fontsize=9)

    fig.suptitle(
        "Robustness Window for Disk TENG Design",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"  [图片] {save_path}")
    plt.close(fig)


# ============================================================================
# H. Regime 对比图: 原始数据 vs 模型预测 (物理流形重构)
# ============================================================================

def plot_regime_comparison(
    real_data,
    model,
    scaler,
    save_path=None,
    grid_res_raw=80,
    grid_res_model=200,
    k_neighbors=30,
    sigma_smooth=1.5,
    n_param_dd=35,
    n_param_hh=35,
    dpi=300,
    figsize=(20, 8.5),
):
    """
    并排 Regime Map 对比图 — 证明模型对物理流形的重构能力。

    生成两张子图:
      (a) 原始实验数据: 逐点计算 variance ratio (k-NN),
          使用 scipy.interpolate.griddata 线性插值填满网格。
          → 典型呈现锯齿状边界, 反映原始采样率的局限。
      (b) 模型预测: 在 (n, E, dd, hh) 4D 参数空间密集采样,
          投影到 (logQ², logInvC) 机制空间后, 基于 k-NN variance ratio
          在 200×200 精细网格上计算, 再经 Gaussian 平滑。
          → 平滑连续的相位边界, 展示模型对物理流形的拟合质量。

    背景: variance ratio 连续双色渐变云图
          (RdBu_r: 蓝=电容主导, 红=电荷主导)
    叠加: log10(FOMS/n) 等值线 (黑色), 展示机制与性能的关联

    物理依据:
        FOMS/n ∝ Qsc² × invC_sum
        log(FOMS/n) = logQ² + logInvC + const

        ratio = Var(log Q²) / (Var(log Q²) + Var(log 1/C))

        不做离散判定 (0 或 1), 保留连续 ratio 值,
        直接在双色渐变云图上展示机制的连续过渡。

    Args:
        real_data:       DataFrame, 须含 logQ, logInvC, FOMS, n, E, dd, hh
        model:           PhysicsMultiTaskTransformer (已加载权重)
        scaler:          dict {'scaler_X', 'scaler_qsc', 'scaler_invc', 'scaler_foms'}
                         或 tuple (scaler_X, scaler_qsc, scaler_invc, scaler_foms)
        save_path:       输出路径 (None 则不保存)
        grid_res_raw:    版本A网格分辨率 (默认 80)
        grid_res_model:  版本B网格分辨率 (默认 200)
        k_neighbors:     k-NN 近邻数 (方差计算)
        sigma_smooth:    版本B Gaussian 平滑 sigma
        n_param_dd:      模型预测 dd 参数采样密度
        n_param_hh:      模型预测 hh 参数采样密度
        dpi:             图片分辨率
        figsize:         图片尺寸

    Returns:
        fig: matplotlib Figure 对象
    """
    import torch
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from sklearn.neighbors import BallTree

    # ---- 解析 scaler ----
    if isinstance(scaler, dict):
        scaler_X = scaler["scaler_X"]
        scaler_qsc = scaler["scaler_qsc"]
        scaler_invc = scaler["scaler_invc"]
        scaler_foms_sc = scaler["scaler_foms"]
    else:
        scaler_X, scaler_qsc, scaler_invc, scaler_foms_sc = scaler

    # ---- 1. 提取真实数据 ----
    logQ = real_data["logQ"].values
    logInvC = real_data["logInvC"].values
    foms = real_data["FOMS"].values

    # 使用归一化 FOMS/n (等值线用途), 与 regime_map_norm_n 逻辑一致
    if "FOMS_norm_n" in real_data.columns:
        foms_for_contour = real_data["FOMS_norm_n"].values
        contour_label = r"$\log_{10}(FOM_S / n)$"
        print("  [Info] 使用归一化 FOMS/n 绘制等值线")
    else:
        foms_for_contour = foms
        contour_label = r"$\log_{10}(FOM_S)$"
        print("  [Warning] 未找到 FOMS_norm_n 列, 使用原始 FOMS")
    log_foms = np.log10(np.maximum(foms_for_contour, 1e-30))

    # ---- 2. 共享坐标范围 ----
    pad_frac = 0.05
    x_min, x_max = logQ.min(), logQ.max()
    y_min, y_max = logInvC.min(), logInvC.max()
    x_pad = (x_max - x_min) * pad_frac
    y_pad = (y_max - y_min) * pad_frac
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    # 共享 FOMS 等值线 levels (从真实数据范围计算, 保证两图一致)
    levels_foms = np.linspace(
        np.percentile(log_foms, 5),
        np.percentile(log_foms, 95),
        10,
    )

    # ================================================================
    # 版本 A: 原始数据 — 逐点 variance ratio + griddata 线性插值
    # ================================================================
    print("  [版本A] 计算逐点 variance ratio (k-NN) ...")
    coords_real = np.column_stack([logQ, logInvC])
    tree_real = BallTree(coords_real)
    k_real = min(k_neighbors, len(logQ))

    # 向量化: 批量查询所有真实点的 k-NN
    _, idx_all = tree_real.query(coords_real, k=k_real)
    logQ_nbrs = logQ[idx_all]          # (N, k)
    logInvC_nbrs = logInvC[idx_all]    # (N, k)
    var_q = np.var(logQ_nbrs, axis=1)
    var_c = np.var(logInvC_nbrs, axis=1)
    denom_real = var_q + var_c
    ratio_pts = np.where(denom_real > 1e-24, var_q / denom_real, 0.5)

    # griddata 线性插值到网格
    gx_a = np.linspace(*xlim, grid_res_raw)
    gy_a = np.linspace(*ylim, grid_res_raw)
    GX_a, GY_a = np.meshgrid(gx_a, gy_a)

    ratio_grid_a = griddata(coords_real, ratio_pts, (GX_a, GY_a), method="linear")
    lfoms_grid_a = griddata(coords_real, log_foms, (GX_a, GY_a), method="linear")

    print(f"    采样点: {len(logQ)},  网格: {grid_res_raw}x{grid_res_raw}")
    print(f"    variance ratio 范围: [{np.nanmin(ratio_grid_a):.3f}, {np.nanmax(ratio_grid_a):.3f}]")

    # ================================================================
    # 版本 B: 模型密集预测 → 投影到机制空间 → 精细 regime 计算
    # ================================================================
    print("  [版本B] 生成模型密集预测 ...")
    device = next(model.parameters()).device

    n_vals = np.array([2, 4, 8, 16, 32, 64], dtype=np.float64)
    E_vals = np.array([1, 2, 3, 5, 7, 10], dtype=np.float64)
    dd_vals = np.geomspace(0.03125, 1.0, n_param_dd)
    hh_vals = np.geomspace(0.00390625, 1.0, n_param_hh)

    # 4D 参数网格
    params = np.array(
        [[n, E, dd, hh]
         for n in n_vals for E in E_vals
         for dd in dd_vals for hh in hh_vals],
        dtype=np.float64,
    )
    print(f"    参数网格: {len(params)} 点 "
          f"({len(n_vals)}n x {len(E_vals)}E x {len(dd_vals)}dd x {len(hh_vals)}hh)")

    X_scaled = scaler_X.transform(params)
    model.eval()

    # 分批推理 (避免内存溢出)
    log_qsc_list, log_invc_list, log_foms_list = [], [], []
    batch_size = 4096

    for start in range(0, len(X_scaled), batch_size):
        end = min(start + batch_size, len(X_scaled))
        X_batch = torch.FloatTensor(X_scaled[start:end]).to(device)
        with torch.no_grad():
            out = model(X_batch)
        log_qsc_list.append(
            scaler_qsc.inverse_transform(out["pred_qsc"].cpu().numpy()))
        log_invc_list.append(
            scaler_invc.inverse_transform(out["pred_invc_sum"].cpu().numpy()))
        log_foms_list.append(
            scaler_foms_sc.inverse_transform(
                out["pred_foms_direct"].cpu().numpy()))

    # 拼接并转换到原始物理量级 (scaler 反变换给出 log10 值)
    log_qsc_pred = np.concatenate(log_qsc_list, axis=0).flatten()
    log_invc_pred = np.concatenate(log_invc_list, axis=0).flatten()
    log_foms_pred = np.concatenate(log_foms_list, axis=0).flatten()

    qsc_pred = 10.0 ** log_qsc_pred
    invc_pred = 10.0 ** log_invc_pred
    foms_pred = 10.0 ** log_foms_pred

    # 归一化 FOMS/n (与版本A保持一致)
    n_pred = params[:, 0]  # n 是参数网格第一列
    foms_norm_pred = foms_pred / np.maximum(n_pred, 1e-30)

    # 过滤有效值
    valid = (qsc_pred > 0) & (invc_pred > 0) & (foms_pred > 0)
    logQ_m = np.log10(np.maximum(qsc_pred[valid] ** 2, 1e-30))
    logInvC_m = np.log10(np.maximum(invc_pred[valid], 1e-30))
    log_foms_m = np.log10(np.maximum(foms_norm_pred[valid], 1e-30))

    print(f"    有效预测点: {valid.sum()}")
    print(f"    logQ_model 范围: [{logQ_m.min():.2f}, {logQ_m.max():.2f}]")
    print(f"    logInvC_model 范围: [{logInvC_m.min():.2f}, {logInvC_m.max():.2f}]")

    # 向量化 k-NN variance ratio 计算
    print(f"  [版本B] 精细网格 variance ratio 计算 "
          f"({grid_res_model}x{grid_res_model}) ...")
    coords_model = np.column_stack([logQ_m, logInvC_m])
    tree_model = BallTree(coords_model)

    gx_b = np.linspace(*xlim, grid_res_model)
    gy_b = np.linspace(*ylim, grid_res_model)
    GX_b, GY_b = np.meshgrid(gx_b, gy_b)

    grid_pts = np.column_stack([GX_b.ravel(), GY_b.ravel()])
    k_model = min(k_neighbors, len(logQ_m))
    _, idx_model = tree_model.query(grid_pts, k=k_model)

    logQ_m_nbrs = logQ_m[idx_model]         # (M, k)
    logInvC_m_nbrs = logInvC_m[idx_model]   # (M, k)
    var_q_m = np.var(logQ_m_nbrs, axis=1)
    var_c_m = np.var(logInvC_m_nbrs, axis=1)
    denom_m = var_q_m + var_c_m
    ratio_flat = np.where(denom_m > 1e-24, var_q_m / denom_m, 0.5)
    ratio_grid_b = ratio_flat.reshape(grid_res_model, grid_res_model)

    # Gaussian 平滑 — 消除离散采样伪影, 获得出版级平滑边界
    ratio_grid_b = gaussian_filter(ratio_grid_b, sigma=sigma_smooth)

    # FOMS 等值线 (从模型预测点 griddata 插值)
    lfoms_grid_b = griddata(
        coords_model, log_foms_m, (GX_b, GY_b), method="linear",
    )

    print(f"    variance ratio 范围 (平滑后): "
          f"[{ratio_grid_b.min():.3f}, {ratio_grid_b.max():.3f}]")

    # ================================================================
    # 绘图: 并排对比
    # ================================================================
    print("  [绘图] 生成对比图 ...")
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08},
    )

    cmap = plt.cm.RdBu_r  # 蓝 (Capacitance) → 白 (Mixed) → 红 (Charge)

    # ---- Panel A: 原始数据插值 ----
    im_a = ax_a.pcolormesh(
        GX_a, GY_a, ratio_grid_a,
        cmap=cmap, vmin=0, vmax=1,
        shading="auto", alpha=0.85,
    )

    # FOMS contours on A
    mask_a_valid = np.isfinite(lfoms_grid_a)
    if mask_a_valid.sum() > 50:
        try:
            cs_a = ax_a.contour(
                GX_a, GY_a, lfoms_grid_a,
                levels=levels_foms, colors="k",
                linewidths=0.6, alpha=0.5,
            )
            ax_a.clabel(cs_a, fontsize=7, fmt="%.1f")
        except ValueError:
            pass

    # 真实采样点散点
    ax_a.scatter(
        logQ, logInvC, c="dimgray", s=6, alpha=0.35, zorder=3,
        label=f"Experimental samples (N={len(logQ)})",
    )

    ax_a.set_xlabel(XLABEL, fontsize=13)
    ax_a.set_ylabel(YLABEL, fontsize=13)
    ax_a.set_title(
        "(a) Raw Experimental Data — Variance Ratio\n"
        f"(griddata linear interpolation, {grid_res_raw}x{grid_res_raw})",
        fontsize=13, fontweight="bold",
    )
    ax_a.set_xlim(xlim)
    ax_a.set_ylim(ylim)
    ax_a.legend(loc="upper left", fontsize=9, framealpha=0.9, edgecolor="gray")
    ax_a.tick_params(labelsize=11)

    # ---- Panel B: 模型预测 ----
    im_b = ax_b.pcolormesh(
        GX_b, GY_b, ratio_grid_b,
        cmap=cmap, vmin=0, vmax=1,
        shading="auto", alpha=0.85,
    )

    # FOMS contours on B
    mask_b_valid = np.isfinite(lfoms_grid_b)
    if mask_b_valid.sum() > 50:
        try:
            cs_b = ax_b.contour(
                GX_b, GY_b, lfoms_grid_b,
                levels=levels_foms, colors="k",
                linewidths=0.6, alpha=0.5,
            )
            ax_b.clabel(cs_b, fontsize=7, fmt="%.1f")
        except ValueError:
            pass

    ax_b.set_xlabel(XLABEL, fontsize=13)
    ax_b.set_ylabel("")
    ax_b.set_title(
        "(b) Multi-Task Model Prediction — Variance Ratio\n"
        f"({grid_res_model}x{grid_res_model} grid + "
        f"Gaussian $\\sigma$={sigma_smooth})",
        fontsize=13, fontweight="bold",
    )
    ax_b.set_xlim(xlim)
    ax_b.set_ylim(ylim)
    ax_b.tick_params(labelsize=11, labelleft=False)

    # ---- 共享 Colorbar ----
    cbar = fig.colorbar(
        im_b, ax=[ax_a, ax_b],
        shrink=0.82, aspect=30, pad=0.03,
    )
    cbar.set_label(
        r"$\mathrm{Var}(\log Q^2)\,/\,"
        r"[\mathrm{Var}(\log Q^2)+\mathrm{Var}(\log 1/C)]$",
        fontsize=13,
    )
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([
        "0.0\n(Cap.\ndominant)",
        "0.25",
        "0.5\n(Balanced)",
        "0.75",
        "1.0\n(Charge\ndominant)",
    ])
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        "Variance-Ratio Regime: Raw Data vs Model Prediction",
        fontsize=15, fontweight="bold", y=1.02,
    )

    fig.subplots_adjust(left=0.06, right=0.86, top=0.88, bottom=0.10,
                        wspace=0.08)

    if save_path:
        _save_fig(fig, save_path, dpi)

    return fig


# ============================================================================
# Standalone runner
# ============================================================================

if __name__ == "__main__":
    import sys
    import torch as _torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils_mechanism_multitask import load_real_data
    from predict_multitask_physics import load_model_and_scalers

    print("=" * 60)
    print("  Regime Comparison Plot — Standalone Runner")
    print("=" * 60)

    # 路径配置 (相对于 code/)
    csv_path = "../data/disk_teng_training_processed.csv"
    checkpoint_dir = "../checkpoints_multitask_physics"
    artifact_dir = "../artifacts_multitask_physics"
    output_dir = "../outputs_mechanism_multitask"
    os.makedirs(output_dir, exist_ok=True)

    # 设备
    if _torch.cuda.is_available():
        _device = _torch.device("cuda")
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        _device = _torch.device("mps")
    else:
        _device = _torch.device("cpu")
    print(f"设备: {_device}")

    # 加载数据
    df_real, _ = load_real_data(csv_path, verbose=True)

    # 加载模型和 scaler
    _model, _sX, _sq, _si, _sf = load_model_and_scalers(
        _device,
        model_path=os.path.join(checkpoint_dir, "physics_multitask_best.pth"),
        artifact_dir=artifact_dir,
    )

    _scaler = {
        "scaler_X": _sX,
        "scaler_qsc": _sq,
        "scaler_invc": _si,
        "scaler_foms": _sf,
    }

    # 生成对比图
    plot_regime_comparison(
        real_data=df_real,
        model=_model,
        scaler=_scaler,
        save_path=os.path.join(output_dir, "regime_comparison.png"),
        dpi=300,
    )

    print("\n" + "=" * 60)
    print("  完成!")
    print("=" * 60)
