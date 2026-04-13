# Mechanism-Resolved Performance Landscape for Disk TENG

Code and data package for the disk-type triboelectric nanogenerator (TENG) study.

This repository contains:

- the processed training dataset and three external validation datasets
- the released multitask surrogate checkpoint and scaler files
- scripts for training, validation, mechanism analysis, robustness screening, and figure generation
- exported main-text and supporting-information figure assets
- a Streamlit app for prediction, design-space exploration, and recommendation

## Repository Layout

```text
disk-teng-code-data-availability/
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
├── outputs/
├── outputs_multitask_physics/
├── outputs_mechanism_multitask/
└── figures_publication/
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

Launch the Streamlit tool:

```bash
streamlit run code/streamlit_app/app.py
```

## Notes

- Paths in manifests are stored relative to the repository root.
- The public package excludes manuscript drafting files, personal directories, and local cache files.
- Existing exported figures are included so the repository can be used directly without rerunning the full workflow.
- Intermediate hyperparameter-sweep outputs and auxiliary baseline weight files are intentionally omitted from the public package.
- Intermediate Fig.1 assembly assets are omitted; the final editable `drawio` source and released figure files are retained.
