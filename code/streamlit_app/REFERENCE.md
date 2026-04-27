# Streamlit App Reference

This file maps the app pages to the released AFM/Wiley manuscript assets and clarifies how the open design interface supports the reported workflow.

## Page Mapping

| App page | Main role | Related assets |
|---|---|---|
| `1_Performance_Predictor.py` | predict a single design point | channel-resolved surrogate inference |
| `2_Design_Space_Explorer.py` | inspect figure-aligned design scenes | Fig.3, Fig.4 |
| `3_Design_Recommendation.py` | screen tolerance-aware candidate windows | Fig.4 decision logic |
| `4_Publication_Figures.py` | browse exported figures and dynamic SI checks | Fig.2-Fig.5, Fig.S1-Fig.S8 |

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

Code, data, released surrogate checkpoint, publication assets, and Streamlit interface for charge-capacitance channel decomposition and fabrication-tolerant disk-TENG design.

## Scope Note

The Streamlit interface reuses the locked multitask surrogate, mechanism metrics, and robustness-screening logic used for the main-text figures. It should be used within the validated structural-dielectric domain rather than as an independent extrapolation engine.
