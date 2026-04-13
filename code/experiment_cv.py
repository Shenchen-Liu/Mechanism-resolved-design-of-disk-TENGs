#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T6: 5-Fold Cross-Validation
========================================================
Stratified 5-fold CV (stratified by n values) for PhysicsMultiTaskTransformer.
Same hyperparameters as T1 (Experiment E).

Reports mean ± std for R², R²_log10, MAE_log10, MAPE for each output head,
plus consistency Pearson between FOMS_direct and FOMS_phys.

Usage:
    cd code/
    python experiment_cv.py
    python experiment_cv.py --debug         # quick test (1 epoch per fold)
    python experiment_cv.py --n_folds 10    # 10-fold CV

"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

from utils_multitask_physics import (
    MultiTaskTENGDataset,
    compute_metrics,
    compute_consistency_metrics,
    compute_foms_phys,
    compute_log10_foms_phys_torch,
    setup_seed,
    get_device,
    EPSILON_0, SIGMA, PI, R,
)
from model_multitask_physics import PhysicsMultiTaskTransformer
from torch.utils.data import DataLoader

# ============================================================================
# 1. Configuration — Must match T1 Experiment E exactly
# ============================================================================

T1_CONFIG = {
    "embed_dim": 256,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.02,
    "lr": 0.0005,
    "batch_size": 32,
    "epochs": 300,
    "patience": 50,
    "lambda_qsc": 1.0,
    "lambda_invc": 1.0,
    "lambda_foms": 1.0,
    "lambda_consistency": 0.3,
}

DATA_PATH = "../data/disk_teng_training_processed.csv"
OUTPUT_DIR = "../outputs/cross_validation"


# ============================================================================
# 2. Data loading (manual CV split)
# ============================================================================

def load_cv_data(data_path):
    """
    Load full dataset for CV splitting.

    Returns:
        X: raw features (N, 4)
        y_qsc: log10(Qsc), shape (N, 1)
        y_invc: log10(invC), shape (N, 1)
        y_foms: log10(FOMS), shape (N, 1)
        raw_n: original n values (N,)
        n_labels: n values for stratification (N,)
    """
    df = pd.read_csv(data_path)
    feature_cols = ["n", "E", "dd", "hh"]
    df["invC_sum"] = df["inv_C_start"] + df["inv_C_end"]

    # Filter (same logic as utils_multitask_physics.py)
    valid_mask = (
        (df["Qsc_MACRS"] > 0) &
        (df["inv_C_start"] > 0) &
        (df["inv_C_end"] > 0) &
        (df["invC_sum"] > 0) &
        (df["FOMS"] >= 0)
    )
    df = df[valid_mask].reset_index(drop=True)
    df = df[df["FOMS"] > 0].reset_index(drop=True)

    X = df[feature_cols].values
    y_qsc = np.log10(df["Qsc_MACRS"].values).reshape(-1, 1)
    y_invc = np.log10(df["invC_sum"].values).reshape(-1, 1)
    y_foms = np.log10(df["FOMS"].values).reshape(-1, 1)
    raw_n = df["n"].values
    n_labels = df["n"].values  # for stratification

    print(f"[数据] 总样本数: {len(df)}")
    print(f"[数据] n 值分布: {dict(pd.Series(n_labels).value_counts().sort_index())}")

    return X, y_qsc, y_invc, y_foms, raw_n, n_labels


# ============================================================================
# 3. Training one fold
# ============================================================================

def train_one_fold(X_train, y_qsc_train, y_invc_train, y_foms_train, raw_n_train,
                   X_val, y_qsc_val, y_invc_val, y_foms_val, raw_n_val,
                   config, seed, device, debug=False):
    """
    Train and evaluate one CV fold.

    Returns:
        val_results: dict with metrics per head + consistency
        train_results: dict with metrics per head + consistency
    """
    setup_seed(seed)

    # Fit scalers on train fold only
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(X_train)
    X_tr_scaled = scaler_X.transform(X_train)
    X_va_scaled = scaler_X.transform(X_val)

    scaler_qsc = MinMaxScaler(feature_range=(0, 1))
    scaler_qsc.fit(y_qsc_train)
    y_qsc_tr = scaler_qsc.transform(y_qsc_train)
    y_qsc_va = scaler_qsc.transform(y_qsc_val)

    scaler_invc = MinMaxScaler(feature_range=(0, 1))
    scaler_invc.fit(y_invc_train)
    y_invc_tr = scaler_invc.transform(y_invc_train)
    y_invc_va = scaler_invc.transform(y_invc_val)

    scaler_foms = MinMaxScaler(feature_range=(0, 1))
    scaler_foms.fit(y_foms_train)
    y_foms_tr = scaler_foms.transform(y_foms_train)
    y_foms_va = scaler_foms.transform(y_foms_val)

    # DataLoaders
    train_ds = MultiTaskTENGDataset(X_tr_scaled, y_qsc_tr, y_invc_tr, y_foms_tr, raw_n_train)
    val_ds = MultiTaskTENGDataset(X_va_scaled, y_qsc_va, y_invc_va, y_foms_va, raw_n_val)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    # Model
    model = PhysicsMultiTaskTransformer(
        input_dim=4,
        embed_dim=config["embed_dim"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config["lr"],
                            weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )
    criterion = nn.MSELoss()

    # Scaler params for consistency loss
    qsc_min_log = torch.tensor(scaler_qsc.data_min_[0], dtype=torch.float32, device=device)
    qsc_range_log = torch.tensor(scaler_qsc.data_range_[0], dtype=torch.float32, device=device)
    invc_min_log = torch.tensor(scaler_invc.data_min_[0], dtype=torch.float32, device=device)
    invc_range_log = torch.tensor(scaler_invc.data_range_[0], dtype=torch.float32, device=device)
    foms_min_log = torch.tensor(scaler_foms.data_min_[0], dtype=torch.float32, device=device)
    foms_range_log = torch.tensor(scaler_foms.data_range_[0], dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    epochs = 1 if debug else config["epochs"]

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        for features, t_qsc, t_invc, t_foms, raw_n in train_loader:
            features = features.to(device)
            t_qsc = t_qsc.to(device)
            t_invc = t_invc.to(device)
            t_foms = t_foms.to(device)
            raw_n_batch = raw_n.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            L_q = criterion(outputs["pred_qsc"], t_qsc)
            L_c = criterion(outputs["pred_invc_sum"], t_invc)
            L_f = criterion(outputs["pred_foms_direct"], t_foms)

            # Consistency loss
            pred_log10_qsc = outputs["pred_qsc"] * qsc_range_log + qsc_min_log
            pred_log10_invc = outputs["pred_invc_sum"] * invc_range_log + invc_min_log
            log10_foms_phys = compute_log10_foms_phys_torch(
                pred_log10_qsc, pred_log10_invc, raw_n_batch.unsqueeze(-1)
            )
            foms_phys_norm = (log10_foms_phys - foms_min_log) / foms_range_log
            L_cons = criterion(outputs["pred_foms_direct"], foms_phys_norm.detach())

            L_total = (
                config["lambda_qsc"] * L_q +
                config["lambda_invc"] * L_c +
                config["lambda_foms"] * L_f +
                config["lambda_consistency"] * L_cons
            )

            if torch.isnan(L_total) or torch.isinf(L_total):
                continue

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if debug:
                break

        scheduler.step()

        # --- Validation loss ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for features, t_qsc, t_invc, t_foms, raw_n in val_loader:
                features = features.to(device)
                t_qsc = t_qsc.to(device)
                t_invc = t_invc.to(device)
                t_foms = t_foms.to(device)
                raw_n_batch = raw_n.to(device)

                outputs = model(features)
                L_q = criterion(outputs["pred_qsc"], t_qsc)
                L_c = criterion(outputs["pred_invc_sum"], t_invc)
                L_f = criterion(outputs["pred_foms_direct"], t_foms)

                pred_log10_qsc = outputs["pred_qsc"] * qsc_range_log + qsc_min_log
                pred_log10_invc = outputs["pred_invc_sum"] * invc_range_log + invc_min_log
                log10_foms_phys = compute_log10_foms_phys_torch(
                    pred_log10_qsc, pred_log10_invc, raw_n_batch.unsqueeze(-1)
                )
                foms_phys_norm = (log10_foms_phys - foms_min_log) / foms_range_log
                L_cons = criterion(outputs["pred_foms_direct"], foms_phys_norm)

                L_total = (
                    config["lambda_qsc"] * L_q +
                    config["lambda_invc"] * L_c +
                    config["lambda_foms"] * L_f +
                    config["lambda_consistency"] * L_cons
                )
                val_loss_sum += L_total.item()
                val_batches += 1

        avg_val = val_loss_sum / max(val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"] and not debug:
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # --- Evaluate on val fold ---
    def evaluate_fold(loader, dataset_name):
        all_pred_qsc, all_true_qsc = [], []
        all_pred_invc, all_true_invc = [], []
        all_pred_foms, all_true_foms = [], []
        all_raw_n = []

        with torch.no_grad():
            for features, t_qsc, t_invc, t_foms, raw_n in loader:
                features = features.to(device)
                outputs = model(features)

                all_pred_qsc.append(outputs["pred_qsc"].cpu().numpy())
                all_pred_invc.append(outputs["pred_invc_sum"].cpu().numpy())
                all_pred_foms.append(outputs["pred_foms_direct"].cpu().numpy())
                all_true_qsc.append(t_qsc.numpy())
                all_true_invc.append(t_invc.numpy())
                all_true_foms.append(t_foms.numpy())
                all_raw_n.append(raw_n.numpy())

        pred_qsc_norm = np.vstack(all_pred_qsc)
        true_qsc_norm = np.vstack(all_true_qsc)
        pred_invc_norm = np.vstack(all_pred_invc)
        true_invc_norm = np.vstack(all_true_invc)
        pred_foms_norm = np.vstack(all_pred_foms)
        true_foms_norm = np.vstack(all_true_foms)
        raw_n_arr = np.concatenate(all_raw_n)

        # Inverse transform
        pred_qsc_phys = 10.0 ** scaler_qsc.inverse_transform(pred_qsc_norm)
        true_qsc_phys = 10.0 ** scaler_qsc.inverse_transform(true_qsc_norm)
        pred_invc_phys = 10.0 ** scaler_invc.inverse_transform(pred_invc_norm)
        true_invc_phys = 10.0 ** scaler_invc.inverse_transform(true_invc_norm)
        pred_foms_phys = 10.0 ** scaler_foms.inverse_transform(pred_foms_norm)
        true_foms_phys = 10.0 ** scaler_foms.inverse_transform(true_foms_norm)

        # FOMS_phys
        pred_foms_phys_recon = compute_foms_phys(
            pred_qsc_phys.flatten(), pred_invc_phys.flatten(), raw_n_arr
        ).reshape(-1, 1)

        m_qsc = compute_metrics(true_qsc_phys, pred_qsc_phys, f"  {dataset_name} Qsc")
        m_invc = compute_metrics(true_invc_phys, pred_invc_phys, f"  {dataset_name} invC")
        m_foms = compute_metrics(true_foms_phys, pred_foms_phys, f"  {dataset_name} FOMS_direct")
        m_foms_p = compute_metrics(true_foms_phys, pred_foms_phys_recon, f"  {dataset_name} FOMS_phys")
        m_cons = compute_consistency_metrics(
            pred_foms_phys.flatten(), pred_foms_phys_recon.flatten(),
            f"  {dataset_name} consistency"
        )

        return {
            "qsc": m_qsc, "invc": m_invc,
            "foms_direct": m_foms, "foms_phys": m_foms_p,
            "consistency": m_cons,
        }

    val_results = evaluate_fold(val_loader, "Val")
    train_results = evaluate_fold(train_loader, "Train")

    return train_results, val_results


# ============================================================================
# 4. Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="T6: 5-Fold Cross-Validation")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--n_folds", type=int, default=5, help="CV 折数 (默认 5)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  T6: {args.n_folds}-Fold Cross-Validation (Stratified by n)")
    print("=" * 70)
    print(f"  设备: {device}")
    print(f"  配置: 与 T1 (Experiment E) 完全一致")
    print(f"  种子: {args.seed}")
    print("=" * 70)

    # --- Load data ---
    print("\n[Step 1] 加载数据")
    X, y_qsc, y_invc, y_foms, raw_n, n_labels = load_cv_data(DATA_PATH)

    # --- Stratified K-Fold ---
    print(f"\n[Step 2] {args.n_folds}-Fold 分层交叉验证")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    all_train_results = []
    all_val_results = []
    fold_details = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, n_labels)):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx+1}/{args.n_folds}")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        print(f"  Val n 分布: {dict(pd.Series(n_labels[val_idx]).value_counts().sort_index())}")
        print(f"{'='*60}")

        train_results, val_results = train_one_fold(
            X[train_idx], y_qsc[train_idx], y_invc[train_idx], y_foms[train_idx], raw_n[train_idx],
            X[val_idx], y_qsc[val_idx], y_invc[val_idx], y_foms[val_idx], raw_n[val_idx],
            T1_CONFIG, args.seed + fold_idx, device, debug=args.debug,
        )

        all_train_results.append(train_results)
        all_val_results.append(val_results)

        fold_details.append({
            "fold": fold_idx + 1,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_n_distribution": dict(pd.Series(n_labels[val_idx]).value_counts().sort_index().items()),
            "train_metrics": {
                target: {k: float(v) for k, v in metrics.items()}
                for target, metrics in train_results.items()
            },
            "val_metrics": {
                target: {k: float(v) for k, v in metrics.items()}
                for target, metrics in val_results.items()
            },
        })

    # --- Aggregate results ---
    print(f"\n{'='*70}")
    print(f"  {args.n_folds}-Fold CV 汇总")
    print(f"{'='*70}")

    def aggregate_folds(all_results):
        agg = {}
        targets = list(all_results[0].keys())
        for target in targets:
            agg[target] = {}
            metrics = all_results[0][target].keys()
            for metric in metrics:
                vals = [r[target][metric] for r in all_results]
                agg[target][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": [float(v) for v in vals],
                }
        return agg

    val_agg = aggregate_folds(all_val_results)
    train_agg = aggregate_folds(all_train_results)

    # Print summary table
    print(f"\n  {'输出头':<18} {'Train R²_log10':>18} {'Val R²_log10':>18} {'Val MAE_log10':>15} {'Val MAPE(%)':>12}")
    print("  " + "-" * 85)

    summary_rows = []
    for target, display in [("qsc", "Qsc_MACRS"), ("invc", "invC_sum"),
                            ("foms_direct", "FOMS_direct"), ("foms_phys", "FOMS_phys")]:
        tr_r2 = train_agg[target]["r2_log10"]
        va_r2 = val_agg[target]["r2_log10"]
        va_mae = val_agg[target]["mae_log10"]
        va_mape = val_agg[target]["mape"]

        tr_str = f"{tr_r2['mean']:.4f}±{tr_r2['std']:.4f}"
        va_r2_str = f"{va_r2['mean']:.4f}±{va_r2['std']:.4f}"
        va_mae_str = f"{va_mae['mean']:.3f}±{va_mae['std']:.3f}"
        va_mape_str = f"{va_mape['mean']:.1f}±{va_mape['std']:.1f}"

        print(f"  {display:<18} {tr_str:>18} {va_r2_str:>18} {va_mae_str:>15} {va_mape_str:>12}")

        summary_rows.append({
            "Output Head": display,
            "Train R²_log10": tr_str,
            "Val R²_log10": va_r2_str,
            "Val R²": f"{val_agg[target]['r2']['mean']:.4f}±{val_agg[target]['r2']['std']:.4f}",
            "Val MAE_log10": va_mae_str,
            "Val MAPE(%)": va_mape_str,
        })

    # Consistency
    cons = val_agg["consistency"]
    cons_pearson = f"{cons['pearson_r']['mean']:.4f}±{cons['pearson_r']['std']:.4f}"
    cons_spearman = f"{cons['spearman_r']['mean']:.4f}±{cons['spearman_r']['std']:.4f}"
    print(f"\n  一致性: Pearson={cons_pearson}, Spearman={cons_spearman}")

    # Check for overfitting
    print(f"\n  过拟合检查:")
    for target, display in [("qsc", "Qsc"), ("invc", "invC"), ("foms_direct", "FOMS")]:
        tr_mean = train_agg[target]["r2_log10"]["mean"]
        va_mean = val_agg[target]["r2_log10"]["mean"]
        gap = tr_mean - va_mean
        status = "OK" if gap < 0.02 else "注意"
        print(f"    {display}: Train-Val gap = {gap:.4f} ({status})")

    # Check std
    foms_std = val_agg["foms_direct"]["r2_log10"]["std"]
    print(f"\n  FOMS R²_log10 std = {foms_std:.4f} {'(< 0.02 ✓)' if foms_std < 0.02 else '(≥ 0.02, 需关注)'}")

    # --- Save results ---
    print(f"\n[Step 3] 保存结果")

    # JSON
    json_data = {
        "_metadata": {
            "task": "T6_cross_validation",
            "timestamp": datetime.now().isoformat(),
            "config": T1_CONFIG,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "n_total_samples": len(X),
        },
        "val_aggregated": {
            target: {
                metric: {"mean": vals["mean"], "std": vals["std"]}
                for metric, vals in metrics.items()
            }
            for target, metrics in val_agg.items()
        },
        "train_aggregated": {
            target: {
                metric: {"mean": vals["mean"], "std": vals["std"]}
                for metric, vals in metrics.items()
            }
            for target, metrics in train_agg.items()
        },
        "fold_details": fold_details,
        "per_fold_val_r2_log10": {
            target: val_agg[target]["r2_log10"]["values"]
            for target in ["qsc", "invc", "foms_direct", "foms_phys"]
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "cv_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  [保存] {json_path}")

    # CSV summary
    csv_path = os.path.join(OUTPUT_DIR, "cv_results.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"  [保存] {csv_path}")

    # Per-fold detail CSV
    detail_rows = []
    for fold_info in fold_details:
        fold = fold_info["fold"]
        for target in ["qsc", "invc", "foms_direct", "foms_phys"]:
            row = {
                "Fold": fold,
                "Target": target,
                "n_val": fold_info["n_val"],
            }
            for metric in ["r2", "r2_log10", "mae_log10", "mape"]:
                row[metric] = fold_info["val_metrics"][target][metric]
            if target != "consistency":
                detail_rows.append(row)
        # Consistency row
        detail_rows.append({
            "Fold": fold,
            "Target": "consistency",
            "n_val": fold_info["n_val"],
            "r2": None,
            "r2_log10": None,
            "mae_log10": fold_info["val_metrics"]["consistency"]["mae"],
            "mape": None,
            "pearson_r": fold_info["val_metrics"]["consistency"]["pearson_r"],
            "spearman_r": fold_info["val_metrics"]["consistency"]["spearman_r"],
        })

    detail_csv_path = os.path.join(OUTPUT_DIR, "cv_fold_details.csv")
    pd.DataFrame(detail_rows).to_csv(detail_csv_path, index=False)
    print(f"  [保存] {detail_csv_path}")

    print(f"\n{'='*70}")
    print(f"  T6 完成！")
    print(f"{'='*70}")
    print(f"  输出目录: {OUTPUT_DIR}/")
    print(f"  - cv_results.json       (完整结构化数据)")
    print(f"  - cv_results.csv        (汇总表, SI Table S2 来源)")
    print(f"  - cv_fold_details.csv   (逐折详细指标)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
