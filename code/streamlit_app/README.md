# Disk TENG Open Design Interface

Interactive delivery layer for the released disk-TENG workflow described in the manuscript **"From Charge-Transfer to Capacitance Limits: Mechanism-Resolved Design of Disk Triboelectric Nanogenerators with Fabrication Tolerance"**.

The app reuses the released physics-consistent multi-output surrogate, mechanism metrics, and robustness-screening logic used for the manuscript figures. It is intended for bounded exploration within the validated structural-dielectric domain.

## Main Features

- Single-point prediction for `Q_sc,MACRS`, `C^{-1}_sum`, `FOM_S,direct`, and physically reconstructed `FOM_S,phys`.
- Figure-aligned design-space exploration across electrode-pair number, dielectric constant, `d/R`, and `h/R`.
- Tolerance-aware candidate-window screening under geometric perturbations.
- Publication asset browsing for the released main-text and supporting-information figures.

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
- The interface documents the same numerical core used by the manuscript; it is not an independent source of new scientific results.
