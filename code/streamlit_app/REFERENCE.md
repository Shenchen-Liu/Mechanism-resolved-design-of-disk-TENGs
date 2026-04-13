# Streamlit App Reference

This file maps the app pages to the released paper assets.

## Page Mapping

| App page | Main role | Related assets |
|---|---|---|
| `1_Performance_Predictor.py` | predict a single design point | model inference |
| `2_Design_Space_Explorer.py` | inspect figure-aligned design scenes | Fig.3, Fig.4 |
| `3_Design_Recommendation.py` | pooled screening and recommendation | Fig.4 decision logic |
| `4_Publication_Figures.py` | browse exported figures and tables | Fig.2-Fig.5, Fig.S1-Fig.S7 |

## Shared Runtime Files

| File | Role |
|---|---|
| `publication_runtime.py` | shared figure and table helpers |
| `streamlit_utils.py` | model loading and plotting utilities |
| `../generate_publication_figures.py` | main-text figure logic |
| `../generate_si_assets.py` | SI figure logic |
| `../predict_multitask_physics.py` | inference helpers |
| `../utils_mechanism_multitask.py` | regime and robustness helpers |

## Recommended Repository Description

An open-source Streamlit-based design tool that exposes the physics-consistent
multitask surrogate, mechanism-aware design maps, robustness screening, and the
released publication assets for disk-type TENG design.
