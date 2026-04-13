#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page 1: Performance Predictor
=============================
输入 4 个设计参数 → 预测 Qsc, invC, FOMS + mechanism-dominance 分类 + 一致性校验。
"""

import streamlit as st

from streamlit_utils import (
    DESIGN_DOMINANCE_THRESHOLD,
    E_VALUES,
    N_VALUES,
    classify_regime,
    compute_single_point_regime,
    get_model_and_scalers,
    is_ood,
    make_predict_fn,
)

st.set_page_config(page_title="Performance Predictor", layout="wide")
st.title("Performance Predictor")
st.markdown(
    "Input a structural–dielectric combination to predict TENG metrics and "
    "interpret its mechanism-dominance behavior."
)

# 加载模型
model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device = get_model_and_scalers()

# ---------------------------------------------------------------------------
# Sidebar 输入
# ---------------------------------------------------------------------------
st.sidebar.header("Design Parameters")

use_custom_n = st.sidebar.checkbox("Custom n (OOD)", value=False)
if use_custom_n:
    n = st.sidebar.number_input("n (sector pairs)", min_value=1, max_value=128, value=4, step=1)
else:
    n = st.sidebar.select_slider("n (sector pairs)", options=N_VALUES, value=4)

use_custom_E = st.sidebar.checkbox("Custom E (OOD)", value=False)
if use_custom_E:
    E = st.sidebar.number_input("E (dielectric constant)", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
else:
    E = st.sidebar.select_slider("E (dielectric constant)", options=E_VALUES, value=3)

dd = st.sidebar.slider(
    "dd (gap ratio d/R)",
    min_value=0.03125,
    max_value=1.0,
    value=0.125,
    step=0.001,
    format="%.4f",
)

hh = st.sidebar.slider(
    "hh (thickness ratio h/R)",
    min_value=0.00390625,
    max_value=1.0,
    value=0.0625,
    step=0.001,
    format="%.6f",
)

# ---------------------------------------------------------------------------
# OOD 检测
# ---------------------------------------------------------------------------
ood_warnings = is_ood(n, E, dd, hh)
if ood_warnings:
    st.warning(
        "**Out-of-Distribution Warning** — The following parameters are outside "
        "the training grid. Predictions for this structural–dielectric combination may be less reliable.\n\n"
        + "\n".join(f"- {w}" for w in ood_warnings)
    )

# ---------------------------------------------------------------------------
# 预测
# ---------------------------------------------------------------------------
from predict_multitask_physics import predict_single

result = predict_single(n, E, dd, hh, model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device)

# ---------------------------------------------------------------------------
# 预测结果展示
# ---------------------------------------------------------------------------
st.subheader("Prediction Results")

col1, col2, col3 = st.columns(3)
col1.metric("Q_sc (C)", f"{result['Qsc_MACRS']:.4e}")
col2.metric("invC_sum (1/F)", f"{result['invC_sum']:.4e}")
col3.metric("FOMS (direct)", f"{result['FOMS_direct']:.4e}")

# ---------------------------------------------------------------------------
# Mechanism-dominance 分类
# ---------------------------------------------------------------------------
st.subheader("Mechanism-dominance")

predict_fn = make_predict_fn(model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device)
f_charge = compute_single_point_regime(predict_fn, n, E, dd, hh)
regime_label, regime_color = classify_regime(f_charge)

rcol1, rcol2 = st.columns([1, 2])
rcol1.markdown(
    f'<div style="background-color:{regime_color}; color:white; '
    f'padding:12px 20px; border-radius:8px; font-size:18px; '
    f'font-weight:bold; text-align:center;">{regime_label}</div>',
    unsafe_allow_html=True,
)
rcol2.markdown(
    f"**f_charge** = {f_charge:.3f}\n\n"
    f"- f_charge > {DESIGN_DOMINANCE_THRESHOLD:.2f} → Charge-dominant\n"
    f"- f_charge < {1.0 - DESIGN_DOMINANCE_THRESHOLD:.2f} → Capacitance-dominant\n"
    f"- otherwise → Mixed\n\n"
    f"This is the same mechanism-dominance interpretation used in the design-space explorer."
)

# ---------------------------------------------------------------------------
# 物理一致性校验
# ---------------------------------------------------------------------------
st.subheader("Physics Consistency Check")

foms_d = result["FOMS_direct"]
foms_p = result["FOMS_phys"]
rel_dev = abs(foms_d - foms_p) / (abs(foms_d) + 1e-30) * 100

ccol1, ccol2, ccol3 = st.columns(3)
ccol1.metric("FOMS_direct", f"{foms_d:.4e}")
ccol2.metric("FOMS_phys", f"{foms_p:.4e}")
ccol3.metric("Relative Deviation", f"{rel_dev:.2f}%")

if rel_dev < 10:
    st.success(f"Consistency check passed (deviation {rel_dev:.2f}% < 10%).")
elif rel_dev < 30:
    st.warning(f"Moderate deviation ({rel_dev:.2f}%). Predictions may need caution.")
else:
    st.error(f"Large deviation ({rel_dev:.2f}%). This point may be unreliable.")

st.caption(
    "The multitask surrogate predicts in scaled log10 space, then restores physical values via "
    "scaler.inverse_transform → 10**x."
)

# ---------------------------------------------------------------------------
# 参数回显
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(f"Input: n={n}, E={E}, dd={dd:.4f}, hh={hh:.6f} | Device: {device}")
