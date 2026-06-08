# Random Forest Baseline Summary

## Protocol

- Data split: same fixed 80/10/10 partition used by T4 (`random_state=42`).
- Split sizes: train=1555, val=194, test=195.
- Inputs: raw `n`, `E`, `dd`, `hh` features; tree models do not require scaling.
- Targets: separate Random Forest regressors trained on log10 Qsc, log10 invC, and log10 FOMS.
- Evaluation: same log-space metrics and direct-vs-physical FOMS consistency functions as T4.

## Selected Hyperparameters

| Target | Hyperparameters |
|---|---|
| qsc | `{"max_depth": 16, "max_features": 1.0, "min_samples_leaf": 1, "n_estimators": 600}` |
| invc | `{"max_depth": null, "max_features": 1.0, "min_samples_leaf": 1, "n_estimators": 600}` |
| foms | `{"max_depth": 16, "max_features": 1.0, "min_samples_leaf": 1, "n_estimators": 600}` |

## In-Distribution Test Summary

| Model | Qsc R2_log10 | invC R2_log10 | FOMS R2_log10 | Consistency Pearson |
|---|---:|---:|---:|---:|
| Random Forest (independent x3) | 0.9953±0.0001 | 0.9970±0.0001 | 0.9903±0.0001 | 0.9828±0.0003 |

## OOD Summary

| Model | V1 FOMS | V1 Cons. | V2 FOMS | V2 Cons. | V3 FOMS | V3 Cons. | Avg FOMS | Avg Cons. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest (x3) | 0.6402 | 0.8942 | 0.5133 | 0.8258 | 0.8923 | 0.9026 | 0.6819 | 0.8742 |

## Manuscript Use

The RF baseline should be described as an added tree-based control evaluated under the same split and OOD protocol. If its random-split score is strong but OOD behavior is weaker than the Transformer, the safe claim is that model selection is driven by OOD channel coherence and mechanism/tolerance outputs rather than by ID accuracy alone.
