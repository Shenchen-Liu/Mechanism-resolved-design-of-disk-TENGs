#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page 2: Design Space Explorer
==============================
Interactive publication-aligned scene explorer for Fig.3/Fig.4 logic.
"""

import streamlit as st

from publication_runtime import (
    DEFAULT_MAIN_SCENARIO_LABEL,
    build_cv_figure,
    build_mechanism_figure,
    build_performance_figure,
    build_retention_figure,
    build_transition_slice_figure,
    compute_publication_scene,
    get_main_scenario_preset,
    get_main_scenario_presets,
    register_predict_fn,
)
from streamlit_utils import (
    DD_TRAIN,
    E_VALUES,
    classify_regime,
    get_model_and_scalers,
    make_predict_fn,
)


st.set_page_config(page_title="Design Space Explorer", layout="wide")
st.title("Design Space Explorer")
st.markdown(
    "Draw a publication-aligned design scene using the current Fig.3/Fig.4 logic. "
    "The heatmaps, transition slice, safe-zone overlay, and representative point all "
    "reuse the same constants and selection rules as the publication scripts."
)

model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device = get_model_and_scalers()
predict_fn = make_predict_fn(
    model,
    scaler_X,
    scaler_qsc,
    scaler_invc,
    scaler_foms,
    device,
)
predict_fn_key = register_predict_fn(predict_fn)

presets = get_main_scenario_presets()
preset_labels = [item.label for item in presets]

st.sidebar.header("Scene Controls")
parameter_mode = st.sidebar.radio(
    "Parameter mode",
    ["Preset scene", "Custom scene"],
    index=0,
)

if parameter_mode == "Preset scene":
    scenario_label = st.sidebar.selectbox(
        "Scene preset",
        options=preset_labels,
        index=preset_labels.index(DEFAULT_MAIN_SCENARIO_LABEL),
        format_func=lambda value: f"{get_main_scenario_preset(value).title} ({value})",
    )
    preset = get_main_scenario_preset(scenario_label)
    scene_title = preset.title
    E_fixed = preset.E
    dd_fixed = preset.dd
else:
    scenario_label = "custom"
    scene_title = st.sidebar.text_input(
        "Custom title",
        value="custom structural–dielectric scene",
    )
    E_fixed = st.sidebar.slider(
        "E (dielectric constant)",
        min_value=0.1,
        max_value=20.0,
        value=3.0,
        step=0.1,
        help="Custom mode accepts off-grid values as well.",
    )
    dd_fixed = st.sidebar.slider(
        "d/R",
        min_value=0.001,
        max_value=1.0,
        value=0.1250,
        step=0.001,
        format="%.4f",
        help="The publication presets use d/R = 0.125 or 0.5.",
    )

with st.sidebar.expander("Advanced controls", expanded=False):
    dominance_threshold = st.slider(
        "Dominance threshold",
        min_value=0.50,
        max_value=0.75,
        value=0.62,
        step=0.01,
    )
    perturb_frac = st.slider(
        "Perturbation fraction",
        min_value=0.05,
        max_value=0.20,
        value=0.10,
        step=0.01,
    )
    if st.button("Clear cached scenes"):
        st.cache_data.clear()

scene = compute_publication_scene(
    predict_fn_key,
    float(E_fixed),
    float(dd_fixed),
    float(dominance_threshold),
    float(perturb_frac),
)
regime_label, regime_color = classify_regime(
    scene["phase_point"]["f_charge"],
    threshold=dominance_threshold,
)
recommended_safe = bool(
    scene["safe_mask"][scene["robust_point"]["row"], scene["robust_point"]["col"]]
)

if parameter_mode == "Custom scene":
    if float(E_fixed) not in {float(value) for value in E_VALUES}:
        st.warning(
            f"`E={E_fixed:.3f}` is outside the publication preset grid {list(E_VALUES)}. "
            "The surrogate can still interpolate/extrapolate, but this view is now in custom mode."
        )
    if float(dd_fixed) not in {float(value) for value in DD_TRAIN}:
        st.info(
            f"`d/R={dd_fixed:.4f}` is not one of the publication preset slices {list(DD_TRAIN)}."
        )

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
metric_col1.metric("Scene", scenario_label)
metric_col2.metric("E / d/R", f"{E_fixed:.2f} / {dd_fixed:.4f}")
metric_col3.metric("Phase point", f"n={scene['phase_point']['n']:.0f}")
metric_col4.metric("Representative", f"h/R={scene['robust_point']['hh']:.4f}")
metric_col5.metric("Safe-zone", "Yes" if recommended_safe else "No")

st.caption(
    f"Current scene title: `{scene_title}` | dominance threshold = `{dominance_threshold:.2f}` | "
    f"perturbation = `±{perturb_frac:.0%}`"
)

top_left, top_right = st.columns(2)
with top_left:
    st.plotly_chart(
        build_performance_figure(scene, title=f"{scene_title} · performance"),
        use_container_width=True,
    )
with top_right:
    st.plotly_chart(
        build_mechanism_figure(scene, title=f"{scene_title} · mechanism"),
        use_container_width=True,
    )

mid_left, mid_right = st.columns(2)
with mid_left:
    st.plotly_chart(
        build_cv_figure(
            scene,
            title=f"{scene_title} · robust CV map",
        ),
        use_container_width=True,
    )
with mid_right:
    st.plotly_chart(
        build_retention_figure(
            scene,
            title=f"{scene_title} · worst-case retention",
        ),
        use_container_width=True,
    )

st.plotly_chart(build_transition_slice_figure(scene), use_container_width=True)

st.markdown(
    f"**Mechanism-dominance**: <span style='color:{regime_color}; font-weight:700'>{regime_label}</span>  \n"
    f"**Phase-point f_charge** = {scene['phase_point']['f_charge']:.3f}  \n"
    f"**Representative CV / retention** = {scene['robust_point']['cv'] * 100:.2f}% / "
    f"{scene['robust_point']['worst_ratio'] * 100:.1f}%  \n"
    f"**Transition slice** = h/R {scene['transition_hh']:.4f}",
    unsafe_allow_html=True,
)
