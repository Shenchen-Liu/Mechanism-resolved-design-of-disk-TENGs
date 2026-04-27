# Charge–Capacitance Channel Decomposition Reveals Fabrication-Tolerant Design Windows for Disk Triboelectric Nanogenerators

This repository supports the AFM/Wiley manuscript **"Charge–Capacitance Channel Decomposition Reveals Fabrication-Tolerant Design Windows for Disk Triboelectric Nanogenerators"**. It provides the final processed datasets, released physics-consistent multi-output surrogate checkpoint, analysis scripts, publication assets, and the Streamlit-based open design interface used for prediction, design-space exploration, and fabrication-tolerance-aware recommendation.

The workflow decomposes the structural figure of merit (`FOM_S`) into a charge-transfer channel (`Q_sc,MACRS`) and a capacitance channel (`C^{-1}_sum`). A shared-backbone multitask surrogate predicts `Q_sc,MACRS`, `C^{-1}_sum`, and `FOM_S` from disk-TENG design variables: electrode-pair number, dielectric constant, dielectric-thickness-to-radius ratio (`h/R`), and air-gap-to-radius ratio (`d/R`).

## AFM Manuscript Abstract

Disk triboelectric nanogenerator (TENG) design pursues high structural figure of merit (`FOM_S`), yet fabricated air gaps and dielectric thicknesses fluctuate at the ±5–10% level, and that tolerance band collides directly with the steep gradient around the nominal optimum. Under ±10% symmetric geometric perturbation, worst-case `FOM_S` retention near the peak frontier falls to 2.7%, so designs that win on paper can lose most of their predicted output once geometric variation is folded back in. We decompose `FOM_S` into a charge-transfer channel (`Q_sc,MACRS`) and a capacitance channel (`C^{-1}_sum`), and train a physics-consistent multi-output surrogate on 1,944 COMSOL-derived designs to jointly predict `Q_sc,MACRS`, `C^{-1}_sum`, and `FOM_S` across electrode-pair number, dielectric-thickness-to-radius ratio (`h/R`), air-gap-to-radius ratio (`d/R`), and dielectric constant. Evaluating 7,776 design points reveals that 58.6% of the explored space is charge-dominant, 36.1% is mixed-regime, and 5.3% is capacitance-dominant. Raising dielectric constant shifts the dominant mechanism toward capacitance-limited behavior; a larger air gap reinforces charge-limited behavior. Mixed-regime windows tolerate the same ±10% perturbation far better than peak-`FOM_S` candidates and supply an explicit fabrication-tolerant target. The surrogate reaches pooled out-of-distribution `FOM_S` `R^2_log10 = 0.914` on 43 unseen structural–dielectric combinations. The combined channel decomposition, mechanism mapping, and tolerance screening let designers identify the limiting mechanism in each region and select fabrication-tolerant candidates inside the validated domain before any device is built.

## AFM Manuscript Summary

- Training data: 1,944 final processed COMSOL-derived disk-TENG designs.
- Dense design-space evaluation: 7,776 supported design points.
- Mechanism distribution under the reference setting: 58.6% charge-dominant, 36.1% mixed-regime, and 5.3% capacitance-dominant.
- External validation: 43 unseen structural-dielectric combinations across three released validation sets.
- Pooled out-of-distribution performance: `FOM_S` `R^2_log10 = 0.914`.
- Robustness finding: mixed-regime windows tolerate ±10% geometric perturbations better than designs near the nominal peak-`FOM_S` frontier.

The AFM/Wiley manuscript uses these repository assets to support a mechanism-resolved design workflow: decompose `FOM_S` into charge-transfer and capacitance channels, map which channel limits each region of the disk-TENG design space, and screen designs for fabrication-tolerant performance rather than nominal peak performance alone.

## GitHub About Metadata

Recommended GitHub repository description:

> Code, data, released surrogate checkpoint, publication assets, and Streamlit interface for charge-capacitance channel decomposition and fabrication-tolerant disk-TENG design.

Recommended topics:

`triboelectric-nanogenerator`, `teng`, `surrogate-modeling`, `physics-informed-ml`, `transformer`, `materials-informatics`, `design-optimization`, `robust-design`, `streamlit`

## Repository Contents

- Final processed training dataset and three final processed external validation datasets.
- Released prediction result tables for the three external validation sets.
- Released multitask surrogate checkpoint and scaler files.
- Scripts for training, inference, cross-validation, mechanism analysis, robustness analysis, and figure generation.
- Exported main-text figure assets and supporting-information assets, including the open design interface figure (`Fig. S8`).
- Streamlit app for single-point prediction, mechanism-aware design-space exploration, and tolerance-aware candidate-window screening.

## Data Files

Final processed datasets in `data/`:

- `disk_teng_training_processed.csv`
- `disk_teng_validation_v1_processed.csv`
- `disk_teng_validation_v2_processed.csv`
- `disk_teng_validation_v3_processed.csv`

Prediction result tables in `data/`:

- `disk_teng_validation_v1_predictions.csv`
- `disk_teng_validation_v2_predictions.csv`
- `disk_teng_validation_v3_predictions.csv`

## Repository Layout

```text
mechanism-resolved-performance-landscape-and-tolerance-aware-design-windows-for-disk-triboelectric-nanogenerators/
├── code/
│   ├── train_multitask_physics.py
│   ├── predict_multitask_physics.py
│   ├── analyze_mechanism_multitask.py
│   ├── generate_publication_figures.py
│   ├── generate_si_assets.py
│   └── streamlit_app/
├── data/
├── artifacts_multitask_physics/
├── checkpoints_multitask_physics/
├── figures_publication/
├── outputs/
├── outputs_multitask_physics/
└── outputs_mechanism_multitask/
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a single prediction:

```bash
python code/predict_multitask_physics.py --n 4 --E 3 --dd 0.125 --hh 0.0625
```

Regenerate the publication assets:

```bash
python code/generate_publication_figures.py
python code/generate_si_assets.py
```

Launch the Streamlit open design interface:

```bash
streamlit run code/streamlit_app/app.py
```

## Scope Notes

- Paths in manifests are stored relative to the repository root.
- Only final processed dataset files are included in the public package.
- Raw and intermediate dataset-building files are not included.
- Existing exported figures are included so the repository can be inspected without rerunning the full workflow.
- The Streamlit interface is a delivery layer for the reported workflow and reuses the released surrogate, mechanism metrics, and robustness-screening logic.
- Use predictions within the validated structural-dielectric domain described in the manuscript.
