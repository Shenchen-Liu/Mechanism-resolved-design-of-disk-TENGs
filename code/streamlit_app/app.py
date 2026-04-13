#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Disk TENG Design Tool.

Launch with: `streamlit run code/streamlit_app/app.py`
"""

import streamlit as st

st.set_page_config(
    page_title="Disk TENG Design Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 预热模型缓存
from publication_runtime import load_main_publication_manifest, load_si_publication_manifest
from streamlit_utils import get_model_and_scalers

with st.spinner("Loading model and scalers..."):
    model, scaler_X, scaler_qsc, scaler_invc, scaler_foms, device = (
        get_model_and_scalers()
    )
publication_manifest = load_main_publication_manifest()
si_manifest = load_si_publication_manifest()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.success("Model loaded successfully.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Physical constants**\n"
    "- EPSILON_0 = 8.854 × 10⁻¹² F/m\n"
    "- SIGMA = 1 × 10⁻⁵ C/m²\n"
    "- R = 0.015 m"
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("Disk TENG Design Tool")
st.markdown("**Mechanism-Resolved Performance Landscape Explorer**")

st.markdown(
    """
This interactive tool accompanies the paper on disk-type triboelectric
nanogenerator (TENG) design optimization. A physics-constrained multi-task
Transformer surrogate model simultaneously predicts short-circuit charge
(`Q_sc`), inverse capacitance (`invC_sum`), and structural figure of merit
(`FOMS`) from four dimensionless design parameters.

The current Streamlit workflow mirrors the released figure logic for:
- Fig.3-style **performance landscape** and **mechanism-dominance map**
- Fig.4-style **safe region** and **design decision** analysis
- SI-style **regime sensitivity** and **grid-refinement stability** diagnostics
- interactive recommendation of robust structural–dielectric combinations

Use the sidebar pages to explore:

| Page | Description |
|------|-------------|
| **Performance Predictor** | Input one structural–dielectric combination and inspect predicted metrics, mechanism-dominance, and physics consistency |
| **Design Space Explorer** | Explore a publication-aligned Fig.3/Fig.4 scene with an 18-point `hh` grid, mechanism-transition slice, CV, retention, and recommended design points |
| **Design Recommendation** | Search the full design space with constraints and view a design decision map with safe-zone and robust recommendation semantics |
| **Publication Figures** | Browse main-text and SI assets, then inspect interactive counterparts of the released figure scripts |
"""
)

st.markdown("---")

st.subheader("Workflow Alignment")
st.markdown(
    """
The current app is synchronized with the released figure workflow:
- structural inputs: `n`, `E`, `d/R`, `h/R`
- surrogate outputs: `Q_sc`, `invC_sum`, `FOMS`
- analysis outputs: `mechanism landscape`, `design windows`, `safe regions`
- final deliverables: `design rules` and `screening tool`
"""
)

if publication_manifest is not None:
    figure_count = len(publication_manifest.get("figures", {}))
    si_figure_count = len(si_manifest.get("figures", {})) if si_manifest is not None else 0
    st.info(
        f"Publication assets detected: main text `{figure_count}` figures, SI `{si_figure_count}` figures. "
        f"Main-text manifest timestamp: `{publication_manifest.get('generated_at', 'unknown time')}`."
    )
else:
    st.warning(
        "Publication manifest not found. Run `python code/generate_publication_figures.py` "
        "and `python code/generate_si_assets.py` to populate the Publication Figures page."
    )

st.markdown("---")

st.subheader("Recommended workflow")
workflow_col1, workflow_col2, workflow_col3 = st.columns(3)
workflow_col1.markdown(
    "**1. Predict a point**  \n"
    "Use *Performance Predictor* to inspect `Q_sc`, `invC_sum`, `FOMS`, and the local mechanism-dominance label."
)
workflow_col2.markdown(
    "**2. Explore a scene**  \n"
    "Use *Design Space Explorer* to compare performance, mechanism-dominance, and robustness windows for a fixed structural–dielectric combination."
)
workflow_col3.markdown(
    "**3. Make a decision**  \n"
    "Use *Design Recommendation* to screen the dense grid, identify safe regions, and compare peak-performance versus robust recommendations."
)

st.caption(
    "For paper-ready review, open the `Publication Figures` page to inspect both exported assets and the aligned interactive explorers."
)

st.markdown("---")

# Model summary
st.subheader("Model Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Architecture", "Transformer")
col2.metric("Embed Dim / Heads", "256 / 4")
col3.metric("Encoder Layers", "2")
col4.metric("Dropout", "0.02")

st.markdown("")

st.markdown("**Test-set performance** (R²_log10 — primary metric):")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Q_sc", "0.990")
mcol2.metric("invC_sum", "0.997")
mcol3.metric("FOMS_direct", "0.982")
mcol4.metric("FOMS_phys", "0.984")

st.markdown("---")

st.subheader("Input Parameters")
st.markdown(
    """
| Symbol | Name | Description | Training range |
|--------|------|-------------|----------------|
| **n** | Sector pairs | Number of sector pairs on the disk | {2, 4, 8, 16, 32, 64} |
| **E** | Dielectric constant | Relative permittivity of dielectric layer | {1, 2, 3, 5, 7, 10} |
| **dd** | Gap ratio (d/R) | Air gap to disk radius ratio | [0.031, 1.0] |
| **hh** | Thickness ratio (h/R) | Dielectric thickness to disk radius ratio | [0.004, 1.0] |
"""
)

st.markdown("---")
st.caption("Disk TENG Design Tool — open-source companion to the released study.")
