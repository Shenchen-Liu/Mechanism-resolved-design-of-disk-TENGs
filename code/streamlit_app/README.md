# Disk TENG Streamlit App

Interactive design tool for the released disk TENG surrogate.

## Main Features

- single-point prediction for `Q_sc`, `invC_sum`, `FOMS_direct`, and `FOMS_phys`
- figure-aligned design-space exploration
- pooled recommendation with robustness screening
- publication asset browsing for main-text and SI figures

## Runtime Layout

The app lives in `code/streamlit_app/` and reads shared Python modules from the
parent `code/` directory. The following directories are expected at the
repository root:

- `checkpoints_multitask_physics/`
- `artifacts_multitask_physics/`
- `figures_publication/`
- `outputs/`
- `outputs_multitask_physics/`
- `outputs_mechanism_multitask/`

## Installation

```bash
pip install -r code/streamlit_app/requirements.txt
```

## Run

```bash
streamlit run code/streamlit_app/app.py
```

## Notes

- Publication manifests use repository-relative paths.
- The app can be launched from the repository root or from `code/streamlit_app/`.
- Page-to-figure mapping is summarized in [REFERENCE.md](./REFERENCE.md).
