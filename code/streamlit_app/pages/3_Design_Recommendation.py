#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page 3: Design Recommendation
==============================
Shared publication-aligned pooled design recommendation explorer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from publication_runtime import (
    build_decision_map_figure,
    compute_recommendation_search,
    get_recommendation_search_preset,
    get_recommendation_search_presets,
    load_main_publication_manifest,
    register_predict_fn,
)
from streamlit_utils import DD_GRID, E_VALUES, HH_GRID, N_VALUES, get_model_and_scalers, make_predict_fn


def _parse_numeric_list(raw: str, *, cast) -> list[float]:
    values = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(cast(text))
    return sorted(set(values))


def _sort_results(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    if sort_by == "Recommended score":
        return df.sort_values(["recommend_score", "FOMS"], ascending=[False, False])
    if sort_by == "FOMS (highest)":
        return df.sort_values("FOMS", ascending=False)
    if sort_by == "CV (lowest)":
        return df.sort_values("CV_pct", ascending=True)
    return df.sort_values("Retention_pct", ascending=False)


st.set_page_config(page_title="Design Recommendation", layout="wide")
st.title("Design Recommendation")
st.markdown(
    "Search a pooled design space with the same Fig.4-style safe-zone and recommendation "
    "semantics used by the publication workflow. Use a preset search pool for publication "
    "alignment or switch to custom mode for user-defined screening."
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

main_manifest = load_main_publication_manifest()
fig04_reference = None
if main_manifest is not None:
    fig04_reference = main_manifest.get("figures", {}).get("fig04", {}).get("recommend")

if fig04_reference is not None:
    ref_col1, ref_col2, ref_col3, ref_col4 = st.columns(4)
    ref_col1.metric("Fig.4 ref n", f"{fig04_reference['n']:.0f}")
    ref_col2.metric("Fig.4 ref h/R", f"{fig04_reference['hh']:.4f}")
    ref_col3.metric("Fig.4 ref CV", f"{fig04_reference['cv_pct']:.2f}%")
    ref_col4.metric("Fig.4 ref log10(FOMS)", f"{fig04_reference['log10_foms']:.2f}")
    st.caption(
        f"Reference scenario: `{fig04_reference['scenario']}` | safe-zone = `{fig04_reference['safe_zone']}` | "
        f"value score = `{fig04_reference['value_score']:.3f}`"
    )

st.sidebar.header("Search Controls")
parameter_mode = st.sidebar.radio(
    "Parameter mode",
    ["Preset search space", "Custom search space"],
    index=0,
)

presets = get_recommendation_search_presets()
preset_labels = [item.label for item in presets]

if parameter_mode == "Preset search space":
    preset_label = st.sidebar.selectbox(
        "Search preset",
        options=preset_labels,
        index=0,
        format_func=lambda value: get_recommendation_search_preset(value).title,
    )
    preset = get_recommendation_search_preset(preset_label)
    selected_n = list(preset.n_values)
    selected_E = list(preset.E_values)
    selected_dd = list(preset.dd_values)
    selected_hh = list(preset.hh_values)
    st.sidebar.caption(preset.description)
else:
    preset_label = "custom"
    try:
        selected_n = _parse_numeric_list(
            st.sidebar.text_input(
                "n values (comma-separated)",
                value=", ".join(str(int(value)) for value in N_VALUES),
            ),
            cast=int,
        )
    except ValueError:
        st.sidebar.error("Invalid n values. Use comma-separated numbers.")
        selected_n = []

    try:
        selected_E = _parse_numeric_list(
            st.sidebar.text_input(
                "E values (comma-separated)",
                value=", ".join(str(value) for value in E_VALUES),
            ),
            cast=float,
        )
    except ValueError:
        st.sidebar.error("Invalid E values. Use comma-separated numbers.")
        selected_E = []

    dd_range = st.sidebar.slider(
        "d/R range",
        min_value=float(DD_GRID[0]),
        max_value=float(DD_GRID[-1]),
        value=(float(DD_GRID[0]), float(DD_GRID[-1])),
        format="%.4f",
    )
    hh_range = st.sidebar.slider(
        "h/R range",
        min_value=float(HH_GRID[0]),
        max_value=float(HH_GRID[-1]),
        value=(float(HH_GRID[0]), float(HH_GRID[-1])),
        format="%.6f",
    )
    selected_dd = [
        float(value) for value in DD_GRID if dd_range[0] <= float(value) <= dd_range[1]
    ]
    selected_hh = [
        float(value) for value in HH_GRID if hh_range[0] <= float(value) <= hh_range[1]
    ]

with st.sidebar.expander("Constraints", expanded=True):
    min_foms = st.number_input(
        "Min FOMS",
        min_value=0.0,
        value=0.0,
        format="%.2e",
        help="Set 0 to disable this constraint.",
    )
    max_cv = st.number_input(
        "Max CV (%)",
        min_value=0.0,
        max_value=100.0,
        value=100.0,
        step=1.0,
        help="Set 100 to disable this constraint.",
    )
    min_retention = st.number_input(
        "Min retention (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        help="Set 0 to disable this constraint.",
    )

with st.sidebar.expander("Advanced controls", expanded=False):
    perturb_frac = st.slider(
        "Perturbation fraction",
        min_value=0.05,
        max_value=0.20,
        value=0.10,
        step=0.01,
    )
    top_n = st.slider("Top-N results", min_value=5, max_value=50, value=10)
    sort_by = st.selectbox(
        "Sort by",
        ["Recommended score", "FOMS (highest)", "CV (lowest)", "Retention (highest)"],
    )

search_btn = st.sidebar.button("Search", type="primary")

if search_btn:
    st.session_state["design_recommendation_params"] = {
        "preset_label": preset_label,
        "selected_n": tuple(float(value) for value in selected_n),
        "selected_E": tuple(float(value) for value in selected_E),
        "selected_dd": tuple(float(value) for value in selected_dd),
        "selected_hh": tuple(float(value) for value in selected_hh),
        "perturb_frac": float(perturb_frac),
        "min_foms": float(min_foms),
        "max_cv": float(max_cv),
        "min_retention": float(min_retention),
        "top_n": int(top_n),
        "sort_by": sort_by,
    }

params = st.session_state.get("design_recommendation_params")
if params is None:
    st.info("Configure the search pool in the sidebar and click **Search**.")
    st.caption(
        "Default safe-zone semantics follow the publication settings: top 30% FOMS, CV ≤ 5%, "
        "and worst-case retention ≥ 90%."
    )
else:
    if not (
        params["selected_n"]
        and params["selected_E"]
        and params["selected_dd"]
        and params["selected_hh"]
    ):
        st.error("The current search-space definition is empty. Adjust the sidebar inputs.")
    else:
        ood_n = [value for value in params["selected_n"] if value not in {float(v) for v in N_VALUES}]
        ood_E = [value for value in params["selected_E"] if value not in {float(v) for v in E_VALUES}]
        if ood_n or ood_E:
            notice_parts = []
            if ood_n:
                notice_parts.append(f"n = {ood_n}")
            if ood_E:
                notice_parts.append(f"E = {ood_E}")
            st.warning(
                "**OOD notice** — The current search pool includes values outside the training grid: "
                + ", ".join(notice_parts)
                + ". These recommendations may be less reliable."
            )

        search_result = compute_recommendation_search(
            predict_fn_key,
            params["selected_n"],
            params["selected_E"],
            params["selected_dd"],
            params["selected_hh"],
            params["perturb_frac"],
            params["min_foms"],
            params["max_cv"],
            params["min_retention"],
        )
        pool_title = (
            get_recommendation_search_preset(params["preset_label"]).title
            if params["preset_label"] != "custom"
            else "Custom search space"
        )

        df_filtered = _sort_results(search_result["filtered_results"], params["sort_by"])
        recommended_row = search_result["recommended_row"]
        peak_row = search_result["peak_row"]
        frontier_df = search_result["frontier_df"]
        if len(df_filtered) > 0 and recommended_row is not None:
            recommended_row = df_filtered[df_filtered.index == recommended_row.name].iloc[0]
        if len(df_filtered) > 0 and peak_row is not None:
            peak_row = df_filtered[df_filtered.index == peak_row.name].iloc[0]

        st.success(
            f"Evaluated **{search_result['n_designs']:,}** designs, "
            f"**{search_result['n_feasible']:,}** satisfy the constraints, and "
            f"**{search_result['safe_count']:,}** remain in the safe region."
        )

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Search pool", pool_title)
        summary_col2.metric("n / E", f"{len(params['selected_n'])} / {len(params['selected_E'])}")
        summary_col3.metric("d/R / h/R", f"{len(params['selected_dd'])} / {len(params['selected_hh'])}")
        summary_col4.metric("Perturbation", f"±{params['perturb_frac']:.0%}")

        st.caption(
            f"Current pool size = `{len(params['selected_n']) * len(params['selected_E']) * len(params['selected_dd']) * len(params['selected_hh']):,}` "
            f"| sort = `{params['sort_by']}` | constraints = "
            f"`FOMS ≥ {params['min_foms']:.2e}`, `CV ≤ {params['max_cv']:.1f}%`, "
            f"`retention ≥ {params['min_retention']:.1f}%`"
        )

        st.subheader("Recommendation Summary")
        if recommended_row is None:
            st.info("No feasible designs remain after the current constraints.")
        else:
            safe_text = "Yes" if bool(recommended_row["safe_zone"]) else "No"
            st.markdown(
                f"**Recommended design ({search_result['recommended_label']})**: "
                f"n={recommended_row['n']:.0f}, E={recommended_row['E']:.2f}, "
                f"d/R={recommended_row['dd']:.4f}, h/R={recommended_row['hh']:.6f}, "
                f"FOMS={recommended_row['FOMS']:.3e}, CV={recommended_row['CV_pct']:.2f}%, "
                f"Retention={recommended_row['Retention_pct']:.1f}%, safe-zone={safe_text}."
            )
            if peak_row is not None:
                st.caption(
                    f"Peak-performance design: n={peak_row['n']:.0f}, E={peak_row['E']:.2f}, "
                    f"d/R={peak_row['dd']:.4f}, h/R={peak_row['hh']:.6f}, "
                    f"FOMS={peak_row['FOMS']:.3e}, CV={peak_row['CV_pct']:.2f}%, "
                    f"Retention={peak_row['Retention_pct']:.1f}%."
                )

        if len(df_filtered) > 0:
            st.subheader("Design Decision Map")
            decision_fig = build_decision_map_figure(
                df_filtered,
                recommended_row=recommended_row,
                foms_threshold=search_result["foms_threshold"],
            )
            st.plotly_chart(decision_fig, use_container_width=True)
            st.caption(
                "This Plotly panel keeps the released Fig.4 styling as closely as possible while "
                "adding hover details for every candidate."
            )

        st.subheader(f"Top-{min(params['top_n'], len(df_filtered))} Designs")
        df_display = df_filtered.head(params["top_n"]).reset_index(drop=True)
        if len(df_display) == 0:
            st.info("No rows to display for the current constraints.")
        else:
            df_display.index = df_display.index + 1
            df_display.index.name = "Rank"
            st.dataframe(
                df_display.style.format(
                    {
                        "n": "{:.0f}",
                        "E": "{:.2f}",
                        "dd": "{:.4f}",
                        "hh": "{:.6f}",
                        "FOMS": "{:.4e}",
                        "log10_FOMS": "{:.2f}",
                        "Qsc": "{:.4e}",
                        "invC_sum": "{:.4e}",
                        "CV_pct": "{:.2f}",
                        "Retention_pct": "{:.1f}",
                        "recommend_score": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )
            st.download_button(
                "Download filtered results (CSV)",
                df_filtered.to_csv(index=False),
                file_name="design_recommendation.csv",
                mime="text/csv",
            )
