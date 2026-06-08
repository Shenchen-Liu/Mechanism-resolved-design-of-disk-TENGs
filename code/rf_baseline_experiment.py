#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1/M3: Random Forest baseline for MDPI revision.

This script mirrors the T4 baseline structure:
  - same fixed 80/10/10 split from code/experiment_baselines.py
  - three independent single-output models for Qsc, invC, and FOMS
  - log10-space targets and the same metric functions
  - V1/V2/V3 OOD evaluation with the seed=42 models

Outputs are written to `outputs/baselines/` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "code" / "experiment_baselines.py").exists() and (path / "data").exists():
            return path
    raise FileNotFoundError("Could not find repository root from script path.")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SCRIPT_DIR)

TRAIN_SEEDS = [42, 123, 456]

EPSILON_0 = 8.854e-12
SIGMA = 1e-5
R = 0.015
PI = np.pi

DATA_PATH = REPO_ROOT / "data" / "disk_teng_training_processed.csv"
OOD_FILES = [
    ("validate1", REPO_ROOT / "data" / "disk_teng_validation_v1_processed.csv"),
    ("validate2", REPO_ROOT / "data" / "disk_teng_validation_v2_processed.csv"),
    ("validate3", REPO_ROOT / "data" / "disk_teng_validation_v3_processed.csv"),
]
EXISTING_ID_TABLE = REPO_ROOT / "outputs" / "baselines" / "baseline_comparison.csv"
EXISTING_OOD_TABLE = REPO_ROOT / "outputs" / "comparison_tables" / "table2_ood_generalization.csv"


def load_data_for_baselines(data_path: str | Path, random_state: int = 42) -> dict[str, np.ndarray]:
    """Load public processed data and reproduce the fixed 80/10/10 split."""
    df = pd.read_csv(data_path)
    print(f"[data] raw samples: {len(df)}")

    feature_cols = ["n", "E", "dd", "hh"]
    df["invC_sum"] = df["inv_C_start"] + df["inv_C_end"]

    valid_mask = (
        (df["Qsc_MACRS"] > 0)
        & (df["inv_C_start"] > 0)
        & (df["inv_C_end"] > 0)
        & (df["invC_sum"] > 0)
        & (df["FOMS"] > 0)
    )
    df = df[valid_mask].reset_index(drop=True)
    print(f"[data] valid samples: {len(df)}")

    X = df[feature_cols].values
    y_qsc = np.log10(df["Qsc_MACRS"].values).reshape(-1, 1)
    y_invc = np.log10(df["invC_sum"].values).reshape(-1, 1)
    y_foms = np.log10(df["FOMS"].values).reshape(-1, 1)
    raw_n = df["n"].values

    indices = np.arange(len(X))
    idx_train, idx_temp = train_test_split(indices, test_size=0.2, random_state=random_state)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=random_state)
    print(f"[data] split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    return {
        "X_train": X[idx_train],
        "X_val": X[idx_val],
        "X_test": X[idx_test],
        "y_qsc_train": y_qsc[idx_train],
        "y_qsc_val": y_qsc[idx_val],
        "y_qsc_test": y_qsc[idx_test],
        "y_invc_train": y_invc[idx_train],
        "y_invc_val": y_invc[idx_val],
        "y_invc_test": y_invc[idx_test],
        "y_foms_train": y_foms[idx_train],
        "y_foms_val": y_foms[idx_val],
        "y_foms_test": y_foms[idx_test],
        "raw_n_train": raw_n[idx_train],
        "raw_n_val": raw_n[idx_val],
        "raw_n_test": raw_n[idx_test],
    }


def load_ood_data(ood_path: str | Path):
    """Load one public OOD validation CSV in the same log-target format."""
    df = pd.read_csv(ood_path)
    feature_cols = ["n", "E", "dd", "hh"]
    df["invC_sum"] = df["inv_C_start"] + df["inv_C_end"]

    X = df[feature_cols].values
    y_qsc = np.log10(df["Qsc_MACRS"].values).reshape(-1, 1)
    y_invc = np.log10(df["invC_sum"].values).reshape(-1, 1)
    foms_col = "FOMS_direct" if "FOMS_direct" in df.columns else "FOMS"
    y_foms = np.log10(df[foms_col].values).reshape(-1, 1)
    raw_n = df["n"].values
    return X, y_qsc, y_invc, y_foms, raw_n


def compute_foms_phys(Qsc_MACRS, invC_sum, n):
    term_const = (n * EPSILON_0) / (SIGMA**2 * PI**2 * R**3)
    return 2.0 * term_const * (Qsc_MACRS**2) * invC_sum


def compute_metrics(y_true, y_pred, name=""):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    nonzero_mask = np.abs(y_true) > 1e-30
    mape = (
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0
        if np.any(nonzero_mask)
        else float("inf")
    )

    positive_mask = (y_true > 0) & (y_pred > 0)
    if np.any(positive_mask):
        log_true = np.log10(y_true[positive_mask])
        log_pred = np.log10(y_pred[positive_mask])
        mae_log10 = float(np.mean(np.abs(log_true - log_pred)))
        ss_res_log = np.sum((log_true - log_pred) ** 2)
        ss_tot_log = np.sum((log_true - np.mean(log_true)) ** 2)
        r2_log10 = float(1.0 - (ss_res_log / ss_tot_log)) if ss_tot_log > 0 else 0.0
    else:
        mae_log10 = float("inf")
        r2_log10 = 0.0

    if name:
        print(
            f"  {name}: R2={r2:.4f}, R2_log10={r2_log10:.4f}, "
            f"MAE_log10={mae_log10:.4f}, MAPE={mape:.2f}%"
        )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "r2_log10": r2_log10,
        "mape": mape,
        "mae_log10": mae_log10,
    }


def compute_consistency_metrics(foms_direct, foms_phys, name="FOMS consistency"):
    foms_direct = np.asarray(foms_direct).flatten()
    foms_phys = np.asarray(foms_phys).flatten()
    mae = np.mean(np.abs(foms_direct - foms_phys))

    if len(foms_direct) > 2:
        pearson_r, _ = stats.pearsonr(foms_direct, foms_phys)
        spearman_r, _ = stats.spearmanr(foms_direct, foms_phys)
    else:
        pearson_r = spearman_r = 0.0

    print(f"  {name}: MAE={mae:.6e}, Pearson={pearson_r:.4f}, Spearman={spearman_r:.4f}")
    return {"mae": mae, "pearson_r": pearson_r, "spearman_r": spearman_r}


def aggregate_seed_results(all_seed_results):
    agg = {}
    for target in all_seed_results[0].keys():
        agg[target] = {}
        for metric in all_seed_results[0][target].keys():
            vals = [r[target][metric] for r in all_seed_results]
            agg[target][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def format_mean_std(mean, std, fmt=".4f"):
    return f"{mean:{fmt}}±{std:{fmt}}"


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def tune_rf(X_train: np.ndarray, y_train: np.ndarray, target_name: str, n_jobs: int, quick: bool) -> dict[str, Any]:
    if quick:
        param_grid = {
            "n_estimators": [200],
            "max_depth": [None],
            "min_samples_leaf": [1],
            "max_features": [1.0],
        }
    else:
        param_grid = {
            "n_estimators": [300, 600],
            "max_depth": [None, 16, 24],
            "min_samples_leaf": [1, 2],
            "max_features": [1.0, "sqrt"],
        }

    model = RandomForestRegressor(
        random_state=42,
        bootstrap=True,
        n_jobs=n_jobs,
    )
    search = GridSearchCV(
        model,
        param_grid,
        scoring="r2",
        cv=3,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train.ravel())
    print(f"  [RF-{target_name}] best_params={search.best_params_}, CV R2={search.best_score_:.4f}")
    return dict(search.best_params_)


def train_rf_single(X_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any], seed: int, n_jobs: int):
    model = RandomForestRegressor(
        **params,
        random_state=seed,
        bootstrap=True,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train.ravel())
    return model


def evaluate_rf_ensemble(models: dict[str, Any], data: dict[str, np.ndarray], raw_n: np.ndarray, dataset_name: str):
    X = data["X"]
    y_qsc_log10 = data["y_qsc"]
    y_invc_log10 = data["y_invc"]
    y_foms_log10 = data["y_foms"]

    pred_qsc_log10 = models["qsc"].predict(X).reshape(-1, 1)
    pred_invc_log10 = models["invc"].predict(X).reshape(-1, 1)
    pred_foms_log10 = models["foms"].predict(X).reshape(-1, 1)

    true_qsc = 10.0 ** y_qsc_log10
    pred_qsc = 10.0 ** pred_qsc_log10
    true_invc = 10.0 ** y_invc_log10
    pred_invc = 10.0 ** pred_invc_log10
    true_foms = 10.0 ** y_foms_log10
    pred_foms = 10.0 ** pred_foms_log10

    pred_foms_phys = compute_foms_phys(
        pred_qsc.flatten(), pred_invc.flatten(), raw_n
    ).reshape(-1, 1)

    tag = f"RF {dataset_name}"
    m_qsc = compute_metrics(true_qsc, pred_qsc, f"  {tag} Qsc")
    m_invc = compute_metrics(true_invc, pred_invc, f"  {tag} invC")
    m_foms = compute_metrics(true_foms, pred_foms, f"  {tag} FOMS")
    m_foms_phys = compute_metrics(true_foms, pred_foms_phys, f"  {tag} FOMS_phys")
    m_cons = compute_consistency_metrics(
        pred_foms.flatten(), pred_foms_phys.flatten(), f"  {tag} consistency"
    )

    return {
        "qsc": m_qsc,
        "invc": m_invc,
        "foms_direct": m_foms,
        "foms_phys": m_foms_phys,
        "consistency": m_cons,
    }


def format_rf_id_row(aggregated: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {"Model": "Random Forest (independent x3)"}
    for target in ["qsc", "invc", "foms_direct"]:
        row[f"{target}_R2_log10"] = format_mean_std(
            aggregated[target]["r2_log10"]["mean"],
            aggregated[target]["r2_log10"]["std"],
        )
        row[f"{target}_MAE_log10"] = format_mean_std(
            aggregated[target]["mae_log10"]["mean"],
            aggregated[target]["mae_log10"]["std"],
        )
    row["consistency_pearson"] = format_mean_std(
        aggregated["consistency"]["pearson_r"]["mean"],
        aggregated["consistency"]["pearson_r"]["std"],
    )
    row["consistency_spearman"] = format_mean_std(
        aggregated["consistency"]["spearman_r"]["mean"],
        aggregated["consistency"]["spearman_r"]["std"],
    )
    return row


def write_outputs(
    output_dir: Path,
    best_params: dict[str, Any],
    seed_results: dict[int, Any],
    aggregated: dict[str, Any],
    ood_results: dict[str, Any],
    data: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "protocol": {
            "data_path": str(DATA_PATH.relative_to(REPO_ROOT)),
            "split": "80/10/10 with random_state=42 via code/experiment_baselines.py",
            "targets": ["log10(Qsc_MACRS)", "log10(inv_C_start + inv_C_end)", "log10(FOMS)"],
            "features": ["n", "E", "dd", "hh"],
            "train_size": int(len(data["X_train"])),
            "val_size": int(len(data["X_val"])),
            "test_size": int(len(data["X_test"])),
            "train_seeds": TRAIN_SEEDS,
            "ood_sets": [name for name, _ in OOD_FILES],
        },
        "best_params": best_params,
        "aggregated_test": aggregated,
        "per_seed_test": seed_results,
    }
    (output_dir / "rf_baseline_results.json").write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    ood_json = {
        "protocol": "Seed=42 RF models evaluated on V1/V2/V3 with the same metric functions as T4.",
        "results": ood_results,
    }
    (output_dir / "rf_baseline_ood_results.json").write_text(
        json.dumps(json_safe(ood_json), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rf_row = format_rf_id_row(aggregated)
    pd.DataFrame([rf_row]).to_csv(output_dir / "rf_baseline_comparison.csv", index=False)

    if EXISTING_ID_TABLE.exists():
        existing = pd.read_csv(EXISTING_ID_TABLE)
        combined = pd.concat([existing, pd.DataFrame([rf_row])], ignore_index=True)
        combined.to_csv(output_dir / "p1_baseline_comparison_with_rf.csv", index=False)

    ood_row: dict[str, Any] | None = None
    if ood_results:
        ood_row = {"Model": "Random Forest (x3)"}
        foms_vals = []
        cons_vals = []
        for ood_name in ["validate1", "validate2", "validate3"]:
            result = ood_results[ood_name]
            foms = result["foms_direct"]["r2_log10"]
            cons = result["consistency"]["pearson_r"]
            ood_row[f"{ood_name} FOMS"] = f"{foms:.4f}"
            ood_row[f"{ood_name} Cons."] = f"{cons:.4f}"
            foms_vals.append(foms)
            cons_vals.append(cons)
        ood_row["Avg FOMS"] = f"{float(np.mean(foms_vals)):.4f}"
        ood_row["Avg Cons."] = f"{float(np.mean(cons_vals)):.4f}"
        pd.DataFrame([ood_row]).to_csv(output_dir / "rf_baseline_ood_summary.csv", index=False)

        if EXISTING_OOD_TABLE.exists():
            existing_ood = pd.read_csv(EXISTING_OOD_TABLE)
            combined_ood = pd.concat([existing_ood, pd.DataFrame([ood_row])], ignore_index=True)
            combined_ood.to_csv(output_dir / "p1_ood_comparison_with_rf.csv", index=False)

    write_summary_markdown(output_dir, rf_row, ood_row, best_params, data)


def write_summary_markdown(
    output_dir: Path,
    rf_row: dict[str, str],
    ood_row: dict[str, Any] | None,
    best_params: dict[str, Any],
    data: dict[str, Any],
) -> None:
    lines = [
        "# Random Forest Baseline Summary",
        "",
        "## Protocol",
        "",
        "- Data split: same fixed 80/10/10 partition used by T4 (`random_state=42`).",
        f"- Split sizes: train={len(data['X_train'])}, val={len(data['X_val'])}, test={len(data['X_test'])}.",
        "- Inputs: raw `n`, `E`, `dd`, `hh` features; tree models do not require scaling.",
        "- Targets: separate Random Forest regressors trained on log10 Qsc, log10 invC, and log10 FOMS.",
        "- Evaluation: same log-space metrics and direct-vs-physical FOMS consistency functions as T4.",
        "",
        "## Selected Hyperparameters",
        "",
        "| Target | Hyperparameters |",
        "|---|---|",
    ]
    for target, params in best_params.items():
        lines.append(f"| {target} | `{json.dumps(params, sort_keys=True)}` |")

    lines.extend(
        [
            "",
            "## In-Distribution Test Summary",
            "",
            "| Model | Qsc R2_log10 | invC R2_log10 | FOMS R2_log10 | Consistency Pearson |",
            "|---|---:|---:|---:|---:|",
            (
                f"| Random Forest (independent x3) | {rf_row['qsc_R2_log10']} | "
                f"{rf_row['invc_R2_log10']} | {rf_row['foms_direct_R2_log10']} | "
                f"{rf_row['consistency_pearson']} |"
            ),
        ]
    )
    if ood_row is not None:
        lines.extend(
            [
                "",
                "## OOD Summary",
                "",
                "| Model | V1 FOMS | V1 Cons. | V2 FOMS | V2 Cons. | V3 FOMS | V3 Cons. | Avg FOMS | Avg Cons. |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| Random Forest (x3) | {ood_row['validate1 FOMS']} | {ood_row['validate1 Cons.']} | "
                    f"{ood_row['validate2 FOMS']} | {ood_row['validate2 Cons.']} | "
                    f"{ood_row['validate3 FOMS']} | {ood_row['validate3 Cons.']} | "
                    f"{ood_row['Avg FOMS']} | {ood_row['Avg Cons.']} |"
                ),
            ]
        )
    else:
        lines.extend(["", "## OOD Summary", "", "OOD evaluation was skipped for this run."])

    lines.extend(
        [
            "",
            "## Manuscript Use",
            "",
            "The RF baseline should be described as an added tree-based control evaluated under the same split and OOD protocol. If its random-split score is strong but OOD behavior is weaker than the Transformer, the safe claim is that model selection is driven by OOD channel coherence and mechanism/tolerance outputs rather than by ID accuracy alone.",
        ]
    )
    (output_dir / "rf_baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 Random Forest baseline experiment")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "baselines")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Use a one-point grid for fast smoke testing.")
    parser.add_argument("--skip_ood", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("P1/M3 Random Forest baseline")
    print("=" * 72)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Output dir: {args.output_dir}")

    data = load_data_for_baselines(str(DATA_PATH))
    print(f"Split sizes: train={len(data['X_train'])}, val={len(data['X_val'])}, test={len(data['X_test'])}")

    targets = {
        "qsc": data["y_qsc_train"],
        "invc": data["y_invc_train"],
        "foms": data["y_foms_train"],
    }
    best_params = {}
    print("\n[Step 1] RF compact GridSearchCV on the training split")
    for target_name, y_train in targets.items():
        best_params[target_name] = tune_rf(
            data["X_train"], y_train, target_name, args.n_jobs, args.quick
        )

    print("\n[Step 2] Train RF models across seeds and evaluate fixed test split")
    seed_results = {}
    seed42_models = None
    test_data = {
        "X": data["X_test"],
        "y_qsc": data["y_qsc_test"],
        "y_invc": data["y_invc_test"],
        "y_foms": data["y_foms_test"],
    }
    for seed in TRAIN_SEEDS:
        print(f"\n  Seed={seed}")
        models = {}
        for target_name in ["qsc", "invc", "foms"]:
            models[target_name] = train_rf_single(
                data["X_train"],
                data[f"y_{target_name}_train"],
                best_params[target_name],
                seed,
                args.n_jobs,
            )
        seed_results[seed] = evaluate_rf_ensemble(models, test_data, data["raw_n_test"], f"test seed={seed}")
        if seed == 42:
            seed42_models = models

    aggregated = aggregate_seed_results(list(seed_results.values()))

    ood_results = {}
    if not args.skip_ood:
        print("\n[Step 3] OOD evaluation with seed=42 models")
        assert seed42_models is not None
        for ood_name, ood_path in OOD_FILES:
            if not ood_path.exists():
                print(f"  [skip] Missing OOD file: {ood_path}")
                continue
            print(f"\n  {ood_name}")
            X_ood, y_qsc_ood, y_invc_ood, y_foms_ood, raw_n_ood = load_ood_data(str(ood_path))
            ood_data = {
                "X": X_ood,
                "y_qsc": y_qsc_ood,
                "y_invc": y_invc_ood,
                "y_foms": y_foms_ood,
            }
            ood_results[ood_name] = evaluate_rf_ensemble(seed42_models, ood_data, raw_n_ood, ood_name)

    missing_ood = {name for name, _ in OOD_FILES} - set(ood_results)
    if missing_ood and not args.skip_ood:
        raise RuntimeError(f"Missing OOD results for: {sorted(missing_ood)}")

    print("\n[Step 4] Write outputs")
    write_outputs(args.output_dir, best_params, seed_results, aggregated, ood_results, data)
    print(f"Done. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
