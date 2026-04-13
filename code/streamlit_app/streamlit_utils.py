#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit 工具函数 — Disk TENG Design Tool 共享组件
===================================================
提供缓存模型加载、场景预设、regime / safe-region 选择逻辑、
单点 regime 计算、OOD 检测，以及 Plotly 叠加辅助函数。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# 路径配置 — 优先使用 streamlit_app 自带 core/，必要时回退到父项目
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent                 # streamlit_app/
APP_ROOT = _THIS_DIR
CORE_DIR = APP_ROOT / "core"
_LEGACY_CODE_DIR = _THIS_DIR.parent                         # code/
_LEGACY_PROJECT_DIR = _LEGACY_CODE_DIR.parent               # project root

for path in [CORE_DIR, APP_ROOT, _LEGACY_CODE_DIR]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _resolve_repo_root():
    """Resolve the asset root for standalone or in-project execution."""
    candidates = [APP_ROOT, _LEGACY_PROJECT_DIR]
    required_dirs = [
        "checkpoints_multitask_physics",
        "artifacts_multitask_physics",
        "figures_publication",
    ]
    for root in candidates:
        if all((root / name).exists() for name in required_dirs):
            return root
    return APP_ROOT


REPO_ROOT = _resolve_repo_root()

# 模型 / scaler 绝对路径 (不依赖 CWD)
_CHECKPOINT_DIR = REPO_ROOT / "checkpoints_multitask_physics"
_ARTIFACT_DIR = REPO_ROOT / "artifacts_multitask_physics"
_MODEL_PATH = _CHECKPOINT_DIR / "physics_multitask_best.pth"

# ---------------------------------------------------------------------------
# 参数常量 (来自训练数据 / 最新论文图设定)
# ---------------------------------------------------------------------------
N_VALUES = [2, 4, 8, 16, 32, 64]
E_VALUES = [1, 2, 3, 5, 7, 10]
DD_TRAIN = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
HH_TRAIN = [0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

# 稠密网格 (用于设计推荐的暴力搜索)
DD_GRID = np.geomspace(0.03125, 1.0, 12)
HH_GRID = np.geomspace(0.00390625, 1.0, 18)
HH_PUBLICATION = HH_GRID.copy()

# 最新 Fig.3 / Fig.4 语义常量
DESIGN_DELTA_LOG = 0.02
DESIGN_DOMINANCE_THRESHOLD = 0.62
PERTURB_FRAC_DEFAULT = 0.10
SAFE_CV_THRESHOLD = 0.05                # 分数形式, 即 5%
SAFE_WORST_RATIO_THRESHOLD = 0.90       # 分数形式, 即 90%
FIG3_BALANCE_ALPHA = 1.00
FIG4_CV_LAMBDA = 0.45
FIG4_RETENTION_GAMMA = 0.20
SAFE_SMOOTH_SIGMA = 0.9
DEFAULT_SCENARIO_LABEL = "mid_E_mid_d"

SCENARIO_PRESETS = [
    {
        "label": "low_E_mid_d",
        "title": "low ε, mid d/R",
        "E": 1.0,
        "dd": 0.125,
    },
    {
        "label": "mid_E_mid_d",
        "title": "mid ε, mid d/R",
        "E": 3.0,
        "dd": 0.125,
    },
    {
        "label": "high_E_mid_d",
        "title": "high ε, mid d/R",
        "E": 10.0,
        "dd": 0.125,
    },
    {
        "label": "mid_E_large_d",
        "title": "mid ε, large d/R",
        "E": 3.0,
        "dd": 0.5,
    },
]

# regime 配色 (与论文图语义保持一致)
REGIME_COLORS = {
    "Charge-dominant": "#E64B35",
    "Mixed": "#B9BFC7",
    "Capacitance-dominant": "#2C7FB8",
}
SAFE_REGION_COLOR = "#2F9E44"
RECOMMEND_EDGE_COLOR = "#111827"
HIGH_PERF_COLOR = "#D7301F"
REGIME_CODE_TO_LABEL = {
    -1: "Capacitance-dominant",
    0: "Mixed",
    1: "Charge-dominant",
}


# ---------------------------------------------------------------------------
# 缓存模型加载
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model_and_scalers():
    """Load model + 4 scalers once per server process."""
    from predict_multitask_physics import load_model_and_scalers
    from utils_multitask_physics import get_device

    device = get_device()
    model, scaler_X, scaler_qsc, scaler_invc, scaler_foms = load_model_and_scalers(
        device,
        model_path=str(_MODEL_PATH),
        artifact_dir=str(_ARTIFACT_DIR),
    )
    return model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device


# ---------------------------------------------------------------------------
# predict_fn 封装 (供 compute_design_regime_grid / compute_robustness_grid 使用)
# ---------------------------------------------------------------------------
def make_predict_fn(model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device):
    """Wrap model into batch predict_fn matching utils_mechanism_multitask API."""
    from predict_multitask_physics import predict_batch

    def predict_fn(n_arr, E_arr, dd_arr, hh_arr):
        df = pd.DataFrame({"n": n_arr, "E": E_arr, "dd": dd_arr, "hh": hh_arr})
        result = predict_batch(df, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device)
        return {
            "Qsc_MACRS": result["Qsc_MACRS_pred"].values,
            "invC_sum": result["invC_sum_pred"].values,
            "FOMS_direct": result["FOMS_direct_pred"].values,
            "FOMS_phys": result["FOMS_phys_pred"].values,
        }

    return predict_fn


# ---------------------------------------------------------------------------
# 场景 / 文案辅助
# ---------------------------------------------------------------------------
def get_scenario_preset(label):
    """按 label 获取场景预设。"""
    for scenario in SCENARIO_PRESETS:
        if scenario["label"] == label:
            return scenario
    return next(item for item in SCENARIO_PRESETS if item["label"] == DEFAULT_SCENARIO_LABEL)


# ---------------------------------------------------------------------------
# Regime 分类
# ---------------------------------------------------------------------------
def classify_regime(f_charge, threshold=DESIGN_DOMINANCE_THRESHOLD):
    """Return (label, color) based on charge-sensitivity fraction."""
    if f_charge > threshold:
        return "Charge-dominant", REGIME_COLORS["Charge-dominant"]
    if f_charge < 1.0 - threshold:
        return "Capacitance-dominant", REGIME_COLORS["Capacitance-dominant"]
    return "Mixed", REGIME_COLORS["Mixed"]


def classify_regime_code(regime_code):
    """Map regime code (-1/0/1) to (label, color)."""
    label = REGIME_CODE_TO_LABEL.get(int(regime_code), "Mixed")
    return label, REGIME_COLORS[label]


# ---------------------------------------------------------------------------
# 单点 regime 计算 (简化版 compute_design_regime_grid)
# ---------------------------------------------------------------------------
def compute_single_point_regime(predict_fn, n, E, dd, hh, delta_log=DESIGN_DELTA_LOG):
    """
    Compute f_charge for a single design point via gradient sensitivity.

    Uses the same finite-difference approach as compute_design_regime_grid.
    """
    factor = 10 ** delta_log
    _arr = lambda v: np.array([v], dtype=np.float64)

    # 扰动 n
    pred_n_plus = predict_fn(_arr(n * factor), _arr(E), _arr(dd), _arr(hh))
    pred_n_minus = predict_fn(_arr(n / factor), _arr(E), _arr(dd), _arr(hh))

    dlogQsc_dlogn = (
        np.log10(max(pred_n_plus["Qsc_MACRS"][0], 1e-30))
        - np.log10(max(pred_n_minus["Qsc_MACRS"][0], 1e-30))
    ) / (2 * delta_log)
    dlogInvC_dlogn = (
        np.log10(max(pred_n_plus["invC_sum"][0], 1e-30))
        - np.log10(max(pred_n_minus["invC_sum"][0], 1e-30))
    ) / (2 * delta_log)

    # 扰动 hh
    pred_hh_plus = predict_fn(_arr(n), _arr(E), _arr(dd), _arr(hh * factor))
    pred_hh_minus = predict_fn(_arr(n), _arr(E), _arr(dd), _arr(hh / factor))

    dlogQsc_dloghh = (
        np.log10(max(pred_hh_plus["Qsc_MACRS"][0], 1e-30))
        - np.log10(max(pred_hh_minus["Qsc_MACRS"][0], 1e-30))
    ) / (2 * delta_log)
    dlogInvC_dloghh = (
        np.log10(max(pred_hh_plus["invC_sum"][0], 1e-30))
        - np.log10(max(pred_hh_minus["invC_sum"][0], 1e-30))
    ) / (2 * delta_log)

    # FOMS ∝ Qsc², charge 通道乘 2
    charge_sens = np.sqrt((2 * dlogQsc_dlogn) ** 2 + (2 * dlogQsc_dloghh) ** 2)
    cap_sens = np.sqrt(dlogInvC_dlogn ** 2 + dlogInvC_dloghh ** 2)

    f_charge = charge_sens / (charge_sens + cap_sens + 1e-30)
    return float(f_charge)


# ---------------------------------------------------------------------------
# 网格选择与推荐逻辑
# ---------------------------------------------------------------------------
def pick_best_index(values, mask, prefer_max=True):
    """在 mask 指定的区域中选取最优下标。"""
    valid_idx = np.argwhere(mask)
    if len(valid_idx) == 0:
        return None
    subset = values[mask]
    order = np.nanargmax(subset) if prefer_max else np.nanargmin(subset)
    return tuple(valid_idx[order])


def normalize_map(values, mask):
    """对 mask 内的数据做 0-1 归一化。"""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
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


def build_safe_mask(
    foms_map,
    cv_map,
    worst_ratio_map,
    cv_threshold=SAFE_CV_THRESHOLD,
    retention_threshold=SAFE_WORST_RATIO_THRESHOLD,
    top_percentile=70,
):
    """构建 safe region 掩码。"""
    top_mask = foms_map >= np.nanpercentile(foms_map, top_percentile)
    return (
        top_mask
        & (cv_map <= cv_threshold)
        & (worst_ratio_map >= retention_threshold)
    )


def select_design_point(design_result, n_values=None, hh_values=None):
    """按 Fig.3 语义选择平衡型高性能设计点。"""
    n_values = np.asarray(n_values if n_values is not None else N_VALUES, dtype=float)
    hh_values = np.asarray(hh_values if hh_values is not None else HH_TRAIN, dtype=float)

    foms = np.asarray(design_result["foms"], dtype=float)
    f_charge = np.asarray(design_result["f_charge"], dtype=float)
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
        "n": float(n_values[col]),
        "hh": float(hh_values[row]),
        "foms": float(foms[row, col]),
        "f_charge": float(f_charge[row, col]),
        "regime": float(design_result["regime"][row, col]),
        "score": float(score[row, col]) if np.isfinite(score[row, col]) else np.nan,
    }


def select_robust_points(foms_map, cv_map, worst_ratio_map, n_values=None, hh_values=None):
    """按 Fig.4 语义返回 safe region、推荐点与 peak-performance 点。"""
    n_values = np.asarray(n_values if n_values is not None else N_VALUES, dtype=float)
    hh_values = np.asarray(hh_values if hh_values is not None else HH_TRAIN, dtype=float)

    top30 = foms_map >= np.nanpercentile(foms_map, 70)
    safe_mask = build_safe_mask(foms_map, cv_map, worst_ratio_map)
    norm_fom = normalize_map(foms_map, top30)
    norm_cv = normalize_map(cv_map, top30)
    norm_retention = normalize_map(worst_ratio_map, top30)

    score = np.full_like(foms_map, -np.inf, dtype=float)
    score[top30] = (
        norm_fom[top30]
        - FIG4_CV_LAMBDA * norm_cv[top30]
        + FIG4_RETENTION_GAMMA * norm_retention[top30]
    )

    rep_idx = pick_best_index(score, top30, prefer_max=True)
    label = "recommend"
    if rep_idx is None:
        rep_idx = np.unravel_index(np.nanargmin(cv_map), cv_map.shape)
        label = "low-CV point"

    hp_idx = np.unravel_index(np.nanargmax(foms_map), foms_map.shape)
    safe_overlay = gaussian_filter(safe_mask.astype(float), sigma=SAFE_SMOOTH_SIGMA)

    return {
        "safe_mask": safe_mask,
        "safe_overlay": safe_overlay,
        "recommended": {
            "row": int(rep_idx[0]),
            "col": int(rep_idx[1]),
            "label": label,
            "n": float(n_values[rep_idx[1]]),
            "hh": float(hh_values[rep_idx[0]]),
            "foms": float(foms_map[rep_idx]),
            "cv": float(cv_map[rep_idx]),
            "worst_ratio": float(worst_ratio_map[rep_idx]),
            "score": float(score[rep_idx]) if np.isfinite(score[rep_idx]) else np.nan,
        },
        "high_perf": {
            "row": int(hp_idx[0]),
            "col": int(hp_idx[1]),
            "n": float(n_values[hp_idx[1]]),
            "hh": float(hh_values[hp_idx[0]]),
            "foms": float(foms_map[hp_idx]),
            "cv": float(cv_map[hp_idx]),
            "worst_ratio": float(worst_ratio_map[hp_idx]),
        },
    }


def normalize_log_series(values):
    """Normalize log10-transformed positive values to [0, 1]."""
    log_values = np.log10(np.maximum(np.asarray(values, dtype=float), 1e-30))
    vmin = float(np.nanmin(log_values))
    vmax = float(np.nanmax(log_values))
    span = vmax - vmin
    if not np.isfinite(span) or span < 1e-12:
        return np.full_like(log_values, 0.5)
    return (log_values - vmin) / span


def select_transition_row(design_result):
    """Choose the Fig.3-style row that best shows a mechanism transition across n."""
    f_map = np.asarray(design_result["f_charge"], dtype=float)
    log_foms = np.log10(np.maximum(np.asarray(design_result["foms"], dtype=float), 1e-30))
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
            if left == 0.0 or right == 0.0 or left * right < 0:
                cross_idx = idx
                break
        if cross_idx is None:
            continue

        center = 0.5 * (len(f_row) - 2)
        edge_penalty = abs(cross_idx - center) / max(center, 1.0)
        row_perf = float(np.nanmedian(log_foms[row, :]))
        perf_norm = (row_perf - global_min) / (global_max - global_min + 1e-12)
        score = 1.00 * span + 0.25 * perf_norm - 0.25 * sign_changes - 0.20 * edge_penalty
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

    if fallback_row is not None:
        return int(fallback_row)
    return int(np.nanargmax(np.nanstd(f_map, axis=1)))


def segment_bounds(values):
    """Convert discrete logarithmic samples into segment boundaries."""
    values = np.asarray(values, dtype=float)
    ratios = values[1:] / values[:-1]
    left = values[0] / np.sqrt(ratios[0])
    mids = np.sqrt(values[:-1] * values[1:])
    right = values[-1] * np.sqrt(ratios[-1])
    return np.concatenate([[left], mids, [right]])


# ---------------------------------------------------------------------------
# Plotly 样式辅助
# ---------------------------------------------------------------------------
def get_regime_colorscale():
    """离散 regime 颜色条。"""
    return [
        [0.00, REGIME_COLORS["Capacitance-dominant"]],
        [0.3333, REGIME_COLORS["Capacitance-dominant"]],
        [0.3334, REGIME_COLORS["Mixed"]],
        [0.6666, REGIME_COLORS["Mixed"]],
        [0.6667, REGIME_COLORS["Charge-dominant"]],
        [1.00, REGIME_COLORS["Charge-dominant"]],
    ]


def add_foms_contours(fig, x_values, y_values, foms_map, line_color="rgba(255,255,255,0.85)"):
    """向热图叠加 FOMS 等值线。"""
    log_foms = np.log10(np.maximum(np.asarray(foms_map, dtype=float), 1e-30))
    vmin = float(np.nanmin(log_foms))
    vmax = float(np.nanmax(log_foms))
    span = vmax - vmin
    if not np.isfinite(span) or span < 1e-12:
        return fig

    size = max(span / 5.0, 0.15)
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=log_foms,
            contours=dict(
                coloring="none",
                showlabels=False,
                start=vmin,
                end=vmax,
                size=size,
            ),
            line=dict(color=line_color, width=1.3),
            showscale=False,
            hoverinfo="skip",
        )
    )
    return fig


def add_safe_region_overlay(fig, x_values, y_values, safe_overlay, safe_mask=None):
    """向热图叠加 safe region 阴影与边界。"""
    if safe_mask is not None and not np.any(safe_mask):
        return fig

    fig.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=safe_overlay,
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "rgba(255,255,255,0.00)"],
                [0.45, "rgba(255,255,255,0.00)"],
                [1.0, "rgba(47,158,68,0.22)"],
            ],
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=safe_overlay,
            contours=dict(coloring="none", start=0.5, end=0.5, size=1.0),
            line=dict(color=SAFE_REGION_COLOR, width=2.0),
            showscale=False,
            hoverinfo="skip",
        )
    )
    return fig


def add_point_marker(
    fig,
    point,
    name,
    symbol="circle",
    fill_color="white",
    line_color=RECOMMEND_EDGE_COLOR,
    size=12,
):
    """向图中添加关键设计点标记。"""
    hover_parts = [
        f"n={point['n']:.0f}",
        f"hh={point['hh']:.4f}",
    ]
    if "foms" in point:
        hover_parts.append(f"FOMS={point['foms']:.3e}")
    if "cv" in point:
        hover_parts.append(f"CV={point['cv'] * 100:.2f}%")
    if "worst_ratio" in point:
        hover_parts.append(f"Retention={point['worst_ratio'] * 100:.1f}%")

    fig.add_trace(
        go.Scatter(
            x=[point["n"]],
            y=[point["hh"]],
            mode="markers",
            name=name,
            marker=dict(
                symbol=symbol,
                size=size,
                color=fill_color,
                line=dict(color=line_color, width=2.0),
            ),
            hovertemplate="<br>".join(hover_parts) + "<extra>" + name + "</extra>",
        )
    )
    return fig


# ---------------------------------------------------------------------------
# OOD 检测
# ---------------------------------------------------------------------------
def is_ood(n, E, dd, hh):
    """Check if parameters are outside training grid. Return list of warnings."""
    warnings = []
    if n not in N_VALUES:
        warnings.append(f"n = {n} is outside training grid {N_VALUES}")
    if E not in E_VALUES:
        warnings.append(f"E = {E} is outside training grid {E_VALUES}")
    if dd < 0.03125 or dd > 1.0:
        warnings.append(f"dd = {dd:.4f} is outside training range [0.03125, 1.0]")
    if hh < 0.00390625 or hh > 1.0:
        warnings.append(f"hh = {hh:.6f} is outside training range [0.00390625, 1.0]")
    return warnings
