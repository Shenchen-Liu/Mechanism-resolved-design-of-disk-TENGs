#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page 4: Publication Figures
===========================
Main-text + supporting-information asset browser and aligned interactive explorer.
"""

from pathlib import Path

import streamlit as st

from publication_runtime import (
    DEFAULT_MAIN_SCENARIO_LABEL,
    DEFAULT_SI_REFERENCE_K,
    DEFAULT_SI_REFERENCE_THRESHOLD,
    build_cv_figure,
    build_mechanism_figure,
    build_performance_figure,
    build_retention_figure,
    build_si_s6_heatmap,
    build_si_s6_line_plot,
    build_si_s7_agreement_plot,
    build_si_s7_fraction_plot,
    build_transition_slice_figure,
    compute_publication_scene,
    get_main_scenario_preset,
    get_main_scenario_presets,
    load_grid_sensitivity_frame,
    load_grid_spatial_metrics,
    load_main_publication_manifest,
    load_publication_table,
    load_si_publication_manifest,
    load_threshold_sensitivity_frame,
    pick_preview_asset,
    register_predict_fn,
    resolve_publication_path,
    summarise_main_manifest,
    summarise_si_manifest,
)
from streamlit_utils import classify_regime, get_model_and_scalers, make_predict_fn


st.set_page_config(page_title="Publication Figures", layout="wide")
st.title("Publication Asset Explorer")
st.markdown(
    "Browse the released publication assets and inspect Streamlit views that stay aligned "
    "with `generate_publication_figures.py` and `generate_si_assets.py`. Main-text and SI "
    "interactions both support preset and custom parameter modes."
)


MAIN_FIGURE_TITLES = {
    "fig01": "Workflow Overview",
    "fig02": "Mechanism Landscape",
    "fig03": "Design Landscapes And Transition",
    "fig04": "Robust Design Map",
    "fig05": "Unseen Structural–Dielectric Validation",
}
SI_FIGURE_TITLES = {
    "figS01_surrogate_architecture": "Surrogate Architecture",
    "figS02_heldout_logscale_consistency": "Held-out Log-scale Consistency",
    "figS03_model_selection_id_vs_ood": "Model Selection ID vs OOD",
    "figS04_cross_validation_stability": "Cross-validation Stability",
    "figS05_ood_error_decomposition": "OOD Error Decomposition",
    "figS06_regime_parameter_sensitivity": "Regime Parameter Sensitivity",
    "figS07_grid_refinement_stability": "Grid Refinement Stability",
}


def _render_asset_downloads(files: list[Path], *, key_prefix: str) -> None:
    for idx, file_path in enumerate(files):
        if not file_path.exists():
            continue
        st.download_button(
            f"Download {file_path.name}",
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime="application/octet-stream",
            key=f"{key_prefix}_{idx}",
        )


def _render_csv_preview(path: str | None, *, key_prefix: str) -> None:
    table = load_publication_table(path)
    if table.empty:
        return
    resolved_path = resolve_publication_path(path)
    if resolved_path is None:
        return
    st.caption(f"Previewing `{resolved_path.name}` ({len(table):,} rows)")
    st.dataframe(table.head(12), use_container_width=True)
    st.download_button(
        f"Download {resolved_path.name}",
        data=resolved_path.read_bytes(),
        file_name=resolved_path.name,
        mime="text/csv",
        key=f"{key_prefix}_csv",
    )


def _render_main_assets(manifest: dict | None) -> None:
    if manifest is None:
        st.warning(
            "Main-text publication manifest not found. Run `python code/generate_publication_figures.py` first."
        )
        return

    for fig_key, fig_info in manifest.get("figures", {}).items():
        title = MAIN_FIGURE_TITLES.get(fig_key, fig_key)
        with st.expander(f"{fig_key.upper()} · {title}", expanded=(fig_key in {"fig03", "fig04"})):
            preview = fig_info.get("preview_file")
            preview_path = (
                resolve_publication_path(preview)
                if preview
                else pick_preview_asset(fig_info.get("main_files", []))
            )
            files = [
                resolved
                for path in fig_info.get("main_files", [])
                if (resolved := resolve_publication_path(path)) is not None
            ]

            preview_col, asset_col = st.columns([1.5, 1.0])
            with preview_col:
                if preview_path and preview_path.exists() and preview_path.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".svg",
                }:
                    st.image(str(preview_path), use_container_width=True)
                elif preview_path is not None:
                    st.info(f"Preview asset available: `{preview_path.name}`")
                else:
                    st.info("No preview asset recorded for this figure.")

            with asset_col:
                st.markdown("**Assets**")
                _render_asset_downloads(files, key_prefix=f"{fig_key}_asset")
                if fig_info.get("recommend"):
                    recommend = fig_info["recommend"]
                    st.markdown("**Representative candidate**")
                    st.markdown(
                        f"`n={recommend['n']:.0f}`, `h/R={recommend['hh']:.4f}`, "
                        f"`CV={recommend['cv_pct']:.2f}%`, `log10(FOMS)={recommend['log10_foms']:.2f}`"
                    )

            _render_csv_preview(fig_info.get("csv"), key_prefix=f"{fig_key}_table")

    st.markdown("### Main-text source tables")
    for table_key, table_path in manifest.get("tables", {}).items():
        resolved_path = resolve_publication_path(table_path)
        label = resolved_path.name if resolved_path is not None else table_key
        with st.expander(f"{table_key} · {label}", expanded=False):
            _render_csv_preview(table_path, key_prefix=table_key)


def _render_si_assets(manifest: dict | None) -> None:
    if manifest is None:
        st.warning(
            "SI asset manifest not found. Run `python code/generate_si_assets.py` first."
        )
        return

    for fig_key, files_raw in manifest.get("figures", {}).items():
        title = SI_FIGURE_TITLES.get(fig_key, fig_key)
        with st.expander(f"{fig_key} · {title}", expanded=(fig_key in {"figS06_regime_parameter_sensitivity", "figS07_grid_refinement_stability"})):
            preview = pick_preview_asset(files_raw)
            files = [
                resolved
                for path in files_raw
                if (resolved := resolve_publication_path(path)) is not None
            ]

            preview_col, asset_col = st.columns([1.5, 1.0])
            with preview_col:
                if preview and preview.exists() and preview.suffix.lower() in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".svg",
                }:
                    st.image(str(preview), use_container_width=True)
                elif preview is not None:
                    st.info(f"Preview asset available: `{preview.name}`")
                else:
                    st.info("No preview asset recorded for this figure.")

            with asset_col:
                st.markdown("**Assets**")
                _render_asset_downloads(files, key_prefix=f"{fig_key}_asset")

    st.markdown("### SI source tables")
    for table_key, table_path in manifest.get("tables", {}).items():
        resolved_path = resolve_publication_path(table_path)
        label = resolved_path.name if resolved_path is not None else table_key
        with st.expander(f"{table_key} · {label}", expanded=False):
            _render_csv_preview(table_path, key_prefix=table_key)

    metrics_path = manifest.get("grid_spatial_metrics")
    if metrics_path:
        with st.expander("grid_spatial_metrics.json", expanded=False):
            metrics = load_grid_spatial_metrics()
            if metrics is not None:
                resolved_path = resolve_publication_path(metrics_path)
                st.json(metrics)
                if resolved_path is not None:
                    st.download_button(
                        "Download grid_spatial_metrics.json",
                        data=resolved_path.read_bytes(),
                        file_name=resolved_path.name,
                        mime="application/json",
                        key="grid_spatial_metrics_download",
                    )


main_manifest = load_main_publication_manifest()
si_manifest = load_si_publication_manifest()
main_summary = summarise_main_manifest(main_manifest)
si_summary = summarise_si_manifest(si_manifest)

sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
sum_col1.metric("Main figures", f"{main_summary['figure_count']}")
sum_col2.metric("Main tables", f"{main_summary['table_count']}")
sum_col3.metric("SI figures", f"{si_summary['figure_count']}")
sum_col4.metric("SI tables", f"{si_summary['table_count']}")
if main_manifest is not None:
    st.caption(f"Main-text manifest timestamp: `{main_summary.get('generated_at', 'unknown')}`")

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

assets_tab, main_tab, si_tab = st.tabs(
    ["Asset Browser", "Main Text Explorer", "SI Explorer"]
)

with assets_tab:
    main_asset_tab, si_asset_tab = st.tabs(["Main Text Assets", "SI Assets"])
    with main_asset_tab:
        _render_main_assets(main_manifest)
    with si_asset_tab:
        _render_si_assets(si_manifest)

with main_tab:
    presets = get_main_scenario_presets()
    preset_labels = [item.label for item in presets]

    control_col, detail_col = st.columns([0.95, 2.05])
    with control_col:
        st.subheader("Scene Controls")
        parameter_mode = st.radio(
            "Parameter mode",
            ["Preset scene", "Custom scene"],
            index=0,
            key="publication_main_parameter_mode",
        )
        if parameter_mode == "Preset scene":
            scenario_label = st.selectbox(
                "Scene preset",
                options=preset_labels,
                index=preset_labels.index(DEFAULT_MAIN_SCENARIO_LABEL),
                format_func=lambda value: f"{get_main_scenario_preset(value).title} ({value})",
                key="publication_main_scene_preset",
            )
            preset = get_main_scenario_preset(scenario_label)
            scene_title = preset.title
            E_fixed = preset.E
            dd_fixed = preset.dd
        else:
            scenario_label = "custom"
            scene_title = st.text_input(
                "Custom title",
                value="custom publication scene",
                key="publication_main_custom_title",
            )
            E_fixed = st.number_input(
                "E (dielectric constant)",
                min_value=0.1,
                value=3.0,
                step=0.1,
                key="publication_main_custom_E",
            )
            dd_fixed = st.number_input(
                "d/R",
                min_value=0.001,
                value=0.1250,
                step=0.01,
                format="%.4f",
                key="publication_main_custom_dd",
            )

        with st.expander("Advanced controls", expanded=False):
            dominance_threshold = st.slider(
                "Dominance threshold",
                min_value=0.50,
                max_value=0.75,
                value=0.62,
                step=0.01,
                key="publication_main_dom_threshold",
            )
            perturb_frac = st.slider(
                "Perturbation fraction",
                min_value=0.05,
                max_value=0.20,
                value=0.10,
                step=0.01,
                key="publication_main_perturb_frac",
            )
    with detail_col:
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
        safe_flag = bool(
            scene["safe_mask"][scene["robust_point"]["row"], scene["robust_point"]["col"]]
        )

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Scene", scenario_label)
        metric_col2.metric("E / d/R", f"{E_fixed:.2f} / {dd_fixed:.4f}")
        metric_col3.metric("Phase point", f"n={scene['phase_point']['n']:.0f}, h/R={scene['phase_point']['hh']:.4f}")
        metric_col4.metric("Representative safe-zone", "Yes" if safe_flag else "No")

        st.caption(
            f"Title: `{scene_title}` | phase-point mechanism = `{regime_label}` | "
            f"threshold = `{dominance_threshold:.2f}` | perturbation = `±{perturb_frac:.0%}`"
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

        bottom_left, bottom_right = st.columns(2)
        with bottom_left:
            st.plotly_chart(
                build_cv_figure(
                    scene,
                    title=f"{scene_title} · robust CV map",
                ),
                use_container_width=True,
            )
        with bottom_right:
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
            f"**Representative CV / retention** = {scene['robust_point']['cv'] * 100:.2f}% / "
            f"{scene['robust_point']['worst_ratio'] * 100:.1f}%  \n"
            f"**Transition slice** = h/R {scene['transition_hh']:.4f}",
            unsafe_allow_html=True,
        )

with si_tab:
    threshold_df = load_threshold_sensitivity_frame()
    grid_df = load_grid_sensitivity_frame()
    grid_metrics = load_grid_spatial_metrics()

    if threshold_df.empty or grid_df.empty:
        st.warning(
            "SI explorer source tables are incomplete. Run `python code/generate_si_assets.py` "
            "to regenerate the current SI CSV/JSON sources."
        )
    else:
        control_col, detail_col = st.columns([0.95, 2.05])
        with control_col:
            st.subheader("SI Controls")
            si_family = st.selectbox(
                "Figure family",
                ["Fig.S6 regime sensitivity", "Fig.S7 grid refinement"],
                key="publication_si_family",
            )
            si_mode = st.radio(
                "Parameter mode",
                ["Preset settings", "Custom settings"],
                index=0,
                key="publication_si_mode",
            )

            if si_family == "Fig.S6 regime sensitivity":
                all_k_values = sorted(int(value) for value in threshold_df["k_neighbors"].unique())
                all_thresholds = sorted(
                    float(value) for value in threshold_df["dominance_threshold"].unique()
                )
                if si_mode == "Preset settings":
                    selected_k = all_k_values
                    threshold_range = (all_thresholds[0], all_thresholds[-1])
                    reference_k = DEFAULT_SI_REFERENCE_K
                    reference_threshold = DEFAULT_SI_REFERENCE_THRESHOLD
                else:
                    selected_k = st.multiselect(
                        "k values",
                        options=all_k_values,
                        default=all_k_values,
                        key="publication_si_s6_k_values",
                    )
                    threshold_range = st.slider(
                        "Threshold range",
                        min_value=float(all_thresholds[0]),
                        max_value=float(all_thresholds[-1]),
                        value=(float(all_thresholds[0]), float(all_thresholds[-1])),
                        step=0.01,
                        key="publication_si_s6_threshold_range",
                    )
                    reference_k = st.selectbox(
                        "Reference k",
                        options=selected_k if selected_k else all_k_values,
                        index=0,
                        key="publication_si_s6_reference_k",
                    )
                    reference_threshold = st.slider(
                        "Reference threshold",
                        min_value=float(threshold_range[0]),
                        max_value=float(threshold_range[1]),
                        value=float(DEFAULT_SI_REFERENCE_THRESHOLD),
                        step=0.01,
                        key="publication_si_s6_reference_threshold",
                    )
            else:
                all_resolutions = sorted(int(value) for value in grid_df["hh_n_points"].unique())
                if si_mode == "Preset settings":
                    selected_resolutions = all_resolutions
                else:
                    selected_resolutions = st.multiselect(
                        "Grid resolutions",
                        options=all_resolutions,
                        default=all_resolutions,
                        key="publication_si_s7_resolutions",
                    )

        with detail_col:
            if si_family == "Fig.S6 regime sensitivity":
                if not selected_k:
                    st.warning("Select at least one `k` value.")
                else:
                    st.caption(
                        "The plots below are data-driven counterparts to `fig_s6_regime_sensitivity()` "
                        "with the same source CSV and reference settings."
                    )
                    left, right = st.columns(2)
                    with left:
                        st.plotly_chart(
                            build_si_s6_heatmap(
                                threshold_df,
                                k_values=selected_k,
                                threshold_range=threshold_range,
                            ),
                            use_container_width=True,
                        )
                    with right:
                        st.plotly_chart(
                            build_si_s6_line_plot(
                                threshold_df,
                                k_values=selected_k,
                                threshold_range=threshold_range,
                                reference_k=int(reference_k),
                                reference_threshold=float(reference_threshold),
                            ),
                            use_container_width=True,
                        )
            else:
                if not selected_resolutions:
                    st.warning("Select at least one grid resolution.")
                else:
                    st.caption(
                        "The plots below are data-driven counterparts to `fig_s7_grid_resolution()` "
                        "with the same CSV/JSON sources used by the final SI figure."
                    )
                    left, right = st.columns(2)
                    with left:
                        st.plotly_chart(
                            build_si_s7_fraction_plot(
                                grid_df,
                                resolutions=selected_resolutions,
                            ),
                            use_container_width=True,
                        )
                    with right:
                        agreement_fig = build_si_s7_agreement_plot(
                            grid_metrics,
                            resolutions=selected_resolutions,
                        )
                        if agreement_fig is None:
                            st.info(
                                "Pairwise spatial-agreement JSON is missing. "
                                "Run `python code/compute_si_grid_spatial_metrics.py` if needed."
                            )
                        else:
                            st.plotly_chart(agreement_fig, use_container_width=True)
