<div align="center">

<img src="docs/assets/article_banner.png" alt="Article banner: Charge–Capacitance Channel Decomposition Reveals Fabrication-Tolerant Design Windows for Disk Triboelectric Nanogenerators" width="100%">

# Mechanism-resolved Design of Disk TENGs

**Open-source data, surrogate model, and design interface for mechanism-aware, fabrication-tolerant disk triboelectric nanogenerator design.**

<p>
  <a href="https://doi.org/10.3390/ma19122607">
    <img src="https://img.shields.io/badge/DOI-10.3390%2Fma19122607-blue" alt="DOI">
  </a>
  <a href="https://www.mdpi.com/1996-1944/19/12/2607">
    <img src="https://img.shields.io/badge/Published%20in-Materials%202026-gold" alt="Published in Materials">
  </a>
  <a href="https://www.mdpi.com/1996-1944/19/12/2607">
    <img src="https://img.shields.io/badge/Article-Open%20Access-success" alt="Open Access Article">
  </a>
  <img src="https://img.shields.io/badge/Model-Physics--consistent%20multitask%20surrogate-purple" alt="Physics-consistent surrogate model">
  <img src="https://img.shields.io/badge/App-Streamlit%20design%20interface-FF4B4B" alt="Streamlit interface">
</p>

</div>

---

## Publication

This repository accompanies the following open-access article:

> **Shenchen Liu, Yangshi Shao, Xuhong Feng, Zehui Lin, Xiaoming Jing, and Everett X. Wang**
>
> **Charge–Capacitance Channel Decomposition Reveals Fabrication-Tolerant Design Windows for Disk Triboelectric Nanogenerators**
>
> *Materials* **2026**, *19*(12), 2607.
>
> https://doi.org/10.3390/ma19122607

The study introduces a charge–capacitance channel decomposition strategy for disk triboelectric nanogenerators. Instead of optimizing the structural figure of merit as a single scalar target, the workflow separates the charge-transfer and capacitance-related pathways, enabling mechanism-resolved prediction, design-space mapping, and fabrication-tolerance screening.

This repository provides the open implementation used in the paper, including processed datasets, external validation sets, trained model artifacts, prediction tables, figure-generation scripts, and a Streamlit-based design interface.

## How to Cite

If you use the data, model, scripts, or design interface in this repository, please cite:

```bibtex
@article{liu2026charge,
  title   = {Charge--Capacitance Channel Decomposition Reveals Fabrication-Tolerant
             Design Windows for Disk Triboelectric Nanogenerators},
  author  = {Liu, Shenchen and Shao, Yangshi and Feng, Xuhong and
             Lin, Zehui and Jing, Xiaoming and Wang, Everett X.},
  journal = {Materials},
  year    = {2026},
  volume  = {19},
  number  = {12},
  pages   = {2607},
  doi     = {10.3390/ma19122607},
  url     = {https://www.mdpi.com/1996-1944/19/12/2607}
}
```

> GitHub also provides a **"Cite this repository"** button in the sidebar, powered by the [`CITATION.cff`](CITATION.cff) file.

---

## Repository Contents

- **Final processed training dataset** and three external validation datasets.
- **Released prediction result tables** for the three external validation sets.
- **Released multitask surrogate checkpoint** and scaler files.
- **Scripts** for training, inference, cross-validation, mechanism analysis, robustness analysis, and figure generation.
- **Exported main-text and SI figure assets**, including the open design interface figure (Fig. S8).
- **Streamlit app** for single-point prediction, mechanism-aware design-space exploration, and tolerance-aware candidate-window screening.

## Data Files

Final processed datasets in `data/`:

| File | Description |
|------|-------------|
| `disk_teng_training_processed.csv` | Training data (1,944 COMSOL-derived designs) |
| `disk_teng_validation_v1_processed.csv` | External validation set 1 |
| `disk_teng_validation_v2_processed.csv` | External validation set 2 |
| `disk_teng_validation_v3_processed.csv` | External validation set 3 |
| `disk_teng_validation_v1_predictions.csv` | Prediction results for validation set 1 |
| `disk_teng_validation_v2_predictions.csv` | Prediction results for validation set 2 |
| `disk_teng_validation_v3_predictions.csv` | Prediction results for validation set 3 |

## Repository Layout

```text
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
├── outputs_mechanism_multitask/
└── docs/
    └── assets/                  # Article banner image
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

Run the Random Forest baseline:

```bash
python code/rf_baseline_experiment.py --n_jobs 1
```

Regenerate publication assets:

```bash
python code/generate_publication_figures.py
python code/generate_si_assets.py
```

Launch the Streamlit design interface:

```bash
streamlit run code/streamlit_app/app.py
```

## Article and Supporting Files

| Resource | Link |
|----------|------|
| Paper | https://www.mdpi.com/1996-1944/19/12/2607 |
| DOI | https://doi.org/10.3390/ma19122607 |

## Scope Notes

- Paths in manifests are stored relative to the repository root.
- Only final processed dataset files are included in the public package.
- Raw and intermediate dataset-building files are not included.
- Existing exported figures are included so the repository can be inspected without rerunning the full workflow.
- The Streamlit interface is a delivery layer for the reported workflow and reuses the released surrogate, mechanism metrics, and robustness-screening logic.
- Use predictions within the validated structural-dielectric domain described in the manuscript.

## License

This project is released under the [MIT License](LICENSE).
