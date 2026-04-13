#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4: Baseline Training — XGBoost×3 + Multi-MLP (±consistency)
========================================================
比较 PhysicsMultiTaskTransformer 与基线模型，用于论文 Table 1。

基线:
  1. XGB-Qsc:  XGBRegressor, tuned, MSE on log10(Qsc)
  2. XGB-invC: XGBRegressor, tuned, MSE on log10(invC)
  3. XGB-FOMS: XGBRegressor, tuned, MSE on log10(FOMS)
  4. Multi-MLP (no consistency): 3-layer MLP, 3 heads, MSE only
  5. Multi-MLP (+ consistency):  Same MLP, 3 heads, MSE + λ_cons=0.3

论证: 物理一致性约束提供独立模型无法提供的结构保证。

Usage:
    cd code/
    python experiment_baselines.py
    python experiment_baselines.py --skip_tuning   # 跳过 XGBoost 调参，使用默认参数
    python experiment_baselines.py --debug          # 快速测试

"""

import os
import sys
import json
import argparse
import numpy as np

# Fix OpenMP crash on macOS with XGBoost
os.environ.setdefault('OMP_NUM_THREADS', '1')

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb

# 本地模块
from utils_multitask_physics import (
    compute_metrics,
    compute_consistency_metrics,
    compute_foms_phys,
    compute_log10_foms_phys_torch,
    setup_seed,
    get_device,
    EPSILON_0, SIGMA, PI, R,
)


# ============================================================================
# 1. 路径配置
# ============================================================================

DATA_PATH = "../data/disk_teng_training_processed.csv"
OOD_FILES = [
    ("validate1", "../data/disk_teng_validation_v1_processed.csv"),
    ("validate2", "../data/disk_teng_validation_v2_processed.csv"),
    ("validate3", "../data/disk_teng_validation_v3_processed.csv"),
]
OUTPUT_DIR = "../outputs/baselines"
OUR_MODEL_METRICS = "../outputs_multitask_physics/test_metrics.json"

# 训练种子 (数据划分固定为 42, 这 3 个仅影响模型训练随机性)
TRAIN_SEEDS = [42, 123, 456]


# ============================================================================
# 2. 数据加载 — 复用 T1 的精确划分逻辑
# ============================================================================

def load_data_for_baselines(data_path, random_state=42):
    """
    复用 load_multitask_data() 的精确数据划分逻辑。

    对 XGBoost: 返回原始特征 + log10 目标 (无 MinMaxScaler)
    对 MLP:     返回 MinMaxScaler 归一化后的数据

    Returns:
        dict with keys:
            'X_train', 'X_val', 'X_test': raw features (N, 4)
            'y_qsc_train/val/test':  log10(Qsc), shape (N, 1)
            'y_invc_train/val/test': log10(invC), shape (N, 1)
            'y_foms_train/val/test': log10(FOMS), shape (N, 1)
            'raw_n_train/val/test':  original n values
            'X_train_scaled', 'X_val_scaled', 'X_test_scaled': MinMaxScaler transformed
            'y_qsc_train_scaled', etc.: MinMaxScaler transformed targets
            'scaler_X', 'scaler_qsc', 'scaler_invc', 'scaler_foms': fitted scalers
    """
    df = pd.read_csv(data_path)
    print(f"[数据] 原始样本数: {len(df)}")

    feature_cols = ["n", "E", "dd", "hh"]
    df["invC_sum"] = df["inv_C_start"] + df["inv_C_end"]

    # 过滤无效值 (与 utils_multitask_physics.py 完全一致)
    valid_mask = (
        (df["Qsc_MACRS"] > 0) &
        (df["inv_C_start"] > 0) &
        (df["inv_C_end"] > 0) &
        (df["invC_sum"] > 0) &
        (df["FOMS"] >= 0)
    )
    df = df[valid_mask].reset_index(drop=True)
    df = df[df["FOMS"] > 0].reset_index(drop=True)
    print(f"[数据] 过滤后有效样本数: {len(df)}")

    X = df[feature_cols].values
    y_qsc = np.log10(df["Qsc_MACRS"].values).reshape(-1, 1)
    y_invc = np.log10(df["invC_sum"].values).reshape(-1, 1)
    y_foms = np.log10(df["FOMS"].values).reshape(-1, 1)
    raw_n = df["n"].values

    # 精确复制 T1 的划分逻辑
    indices = np.arange(len(X))
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.2, random_state=random_state
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=random_state
    )

    print(f"[数据] 划分: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    data = {}

    # Raw (for XGBoost)
    data['X_train'] = X[idx_train]
    data['X_val'] = X[idx_val]
    data['X_test'] = X[idx_test]
    data['y_qsc_train'] = y_qsc[idx_train]
    data['y_qsc_val'] = y_qsc[idx_val]
    data['y_qsc_test'] = y_qsc[idx_test]
    data['y_invc_train'] = y_invc[idx_train]
    data['y_invc_val'] = y_invc[idx_val]
    data['y_invc_test'] = y_invc[idx_test]
    data['y_foms_train'] = y_foms[idx_train]
    data['y_foms_val'] = y_foms[idx_val]
    data['y_foms_test'] = y_foms[idx_test]
    data['raw_n_train'] = raw_n[idx_train]
    data['raw_n_val'] = raw_n[idx_val]
    data['raw_n_test'] = raw_n[idx_test]

    # MinMaxScaler (for MLP, fit on train only)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_X.fit(data['X_train'])
    data['X_train_scaled'] = scaler_X.transform(data['X_train'])
    data['X_val_scaled'] = scaler_X.transform(data['X_val'])
    data['X_test_scaled'] = scaler_X.transform(data['X_test'])

    scaler_qsc = MinMaxScaler(feature_range=(0, 1))
    scaler_qsc.fit(data['y_qsc_train'])
    data['y_qsc_train_scaled'] = scaler_qsc.transform(data['y_qsc_train'])
    data['y_qsc_val_scaled'] = scaler_qsc.transform(data['y_qsc_val'])
    data['y_qsc_test_scaled'] = scaler_qsc.transform(data['y_qsc_test'])

    scaler_invc = MinMaxScaler(feature_range=(0, 1))
    scaler_invc.fit(data['y_invc_train'])
    data['y_invc_train_scaled'] = scaler_invc.transform(data['y_invc_train'])
    data['y_invc_val_scaled'] = scaler_invc.transform(data['y_invc_val'])
    data['y_invc_test_scaled'] = scaler_invc.transform(data['y_invc_test'])

    scaler_foms = MinMaxScaler(feature_range=(0, 1))
    scaler_foms.fit(data['y_foms_train'])
    data['y_foms_train_scaled'] = scaler_foms.transform(data['y_foms_train'])
    data['y_foms_val_scaled'] = scaler_foms.transform(data['y_foms_val'])
    data['y_foms_test_scaled'] = scaler_foms.transform(data['y_foms_test'])

    data['scaler_X'] = scaler_X
    data['scaler_qsc'] = scaler_qsc
    data['scaler_invc'] = scaler_invc
    data['scaler_foms'] = scaler_foms

    return data


def load_ood_data(ood_path):
    """
    加载 OOD 验证集，返回与训练集相同格式的数据。

    Returns:
        X: raw features (N, 4)
        y_qsc, y_invc, y_foms: log10 targets (N, 1)
        raw_n: original n values (N,)
    """
    df = pd.read_csv(ood_path)
    feature_cols = ["n", "E", "dd", "hh"]
    df["invC_sum"] = df["inv_C_start"] + df["inv_C_end"]

    X = df[feature_cols].values
    y_qsc = np.log10(df["Qsc_MACRS"].values).reshape(-1, 1)
    y_invc = np.log10(df["invC_sum"].values).reshape(-1, 1)
    # 列名可能是 FOMS 或 FOMS_direct
    foms_col = "FOMS_direct" if "FOMS_direct" in df.columns else "FOMS"
    y_foms = np.log10(df[foms_col].values).reshape(-1, 1)
    raw_n = df["n"].values

    return X, y_qsc, y_invc, y_foms, raw_n


# ============================================================================
# 3. XGBoost 基线
# ============================================================================

def tune_xgboost(X_train, y_train, X_val, y_val, target_name, skip_tuning=False):
    """
    GridSearchCV 调参 + 返回最佳 XGBRegressor。

    Args:
        X_train, y_train: 训练集 (raw features, log10 targets)
        X_val, y_val: 验证集
        target_name: 目标名称 (用于打印)
        skip_tuning: 跳过调参，使用默认参数

    Returns:
        best_params: dict
    """
    if skip_tuning:
        best_params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
        }
        print(f"  [XGB-{target_name}] 跳过调参, 使用默认: {best_params}")
        return best_params

    param_grid = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500, 1000],
    }
    fixed_params = {
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'objective': 'reg:squarederror',
        'verbosity': 0,
        'random_state': 42,
        'nthread': 1,
    }

    model = xgb.XGBRegressor(**fixed_params)
    gs = GridSearchCV(
        model, param_grid,
        scoring='r2',
        cv=3,
        n_jobs=1,
        verbose=0,
    )
    gs.fit(X_train, y_train.ravel())

    best_params = gs.best_params_
    best_score = gs.best_score_

    print(f"  [XGB-{target_name}] 最佳参数: {best_params}, CV R²={best_score:.4f}")
    return best_params


def train_xgboost_single(X_train, y_train, params, seed):
    """训练单个 XGBRegressor。"""
    fixed_params = {
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'objective': 'reg:squarederror',
        'verbosity': 0,
        'nthread': 1,
    }
    model = xgb.XGBRegressor(
        **params,
        **fixed_params,
        random_state=seed,
    )
    model.fit(X_train, y_train.ravel())
    return model


def evaluate_xgboost_ensemble(models_dict, data, raw_n, dataset_name="test"):
    """
    评估 3 个独立 XGBoost 模型 (Qsc/invC/FOMS)。

    Args:
        models_dict: {'qsc': model, 'invc': model, 'foms': model}
        data: {'X': features, 'y_qsc': log10, 'y_invc': log10, 'y_foms': log10}
        raw_n: original n values
        dataset_name: for printing

    Returns:
        results: dict with metrics for each head + consistency
    """
    X = data['X']
    y_qsc_log10 = data['y_qsc']
    y_invc_log10 = data['y_invc']
    y_foms_log10 = data['y_foms']

    # Predict in log10 space
    pred_qsc_log10 = models_dict['qsc'].predict(X).reshape(-1, 1)
    pred_invc_log10 = models_dict['invc'].predict(X).reshape(-1, 1)
    pred_foms_log10 = models_dict['foms'].predict(X).reshape(-1, 1)

    # Convert to physical space
    true_qsc = 10.0 ** y_qsc_log10
    pred_qsc = 10.0 ** pred_qsc_log10
    true_invc = 10.0 ** y_invc_log10
    pred_invc = 10.0 ** pred_invc_log10
    true_foms = 10.0 ** y_foms_log10
    pred_foms = 10.0 ** pred_foms_log10

    # FOMS_phys from predicted Qsc and invC
    pred_foms_phys = compute_foms_phys(
        pred_qsc.flatten(), pred_invc.flatten(), raw_n
    ).reshape(-1, 1)

    # Compute metrics
    m_qsc = compute_metrics(true_qsc, pred_qsc, f"  XGB Qsc ({dataset_name})")
    m_invc = compute_metrics(true_invc, pred_invc, f"  XGB invC ({dataset_name})")
    m_foms = compute_metrics(true_foms, pred_foms, f"  XGB FOMS ({dataset_name})")
    m_foms_phys = compute_metrics(true_foms, pred_foms_phys, f"  XGB FOMS_phys ({dataset_name})")
    m_cons = compute_consistency_metrics(
        pred_foms.flatten(), pred_foms_phys.flatten(),
        f"  XGB consistency ({dataset_name})"
    )

    return {
        'qsc': m_qsc, 'invc': m_invc,
        'foms_direct': m_foms, 'foms_phys': m_foms_phys,
        'consistency': m_cons,
    }


# ============================================================================
# 4. Multi-MLP 模型定义
# ============================================================================

class MultiTaskMLP(nn.Module):
    """
    3-layer MLP with 3 independent heads.
    Architecture: 4 → 256 → ReLU → 128 → ReLU → 64 → ReLU → 3 heads (→1 each)
    """

    def __init__(self, input_dim=4, hidden_dims=(256, 128, 64)):
        super(MultiTaskMLP, self).__init__()

        # Shared backbone
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # 3 independent heads
        self.head_qsc = nn.Linear(hidden_dims[-1], 1)
        self.head_invc = nn.Linear(hidden_dims[-1], 1)
        self.head_foms = nn.Linear(hidden_dims[-1], 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        shared = self.backbone(x)
        return {
            'pred_qsc': self.head_qsc(shared),
            'pred_invc_sum': self.head_invc(shared),
            'pred_foms_direct': self.head_foms(shared),
        }


# ============================================================================
# 5. MLP 训练与评估
# ============================================================================

def train_mlp(data, seed, use_consistency=False, lambda_consistency=0.3,
              epochs=300, lr=5e-4, patience=50, debug=False):
    """
    训练 Multi-MLP 模型。

    Args:
        data: 数据字典 (from load_data_for_baselines)
        seed: 训练种子
        use_consistency: 是否使用一致性损失
        lambda_consistency: 一致性损失权重
        epochs: 训练轮数
        lr: 学习率
        patience: 早停耐心
        debug: 调试模式

    Returns:
        model: 训练好的模型
    """
    setup_seed(seed)
    device = get_device()

    model = MultiTaskMLP(input_dim=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )
    criterion = nn.MSELoss()

    # Prepare data tensors
    X_train = torch.FloatTensor(data['X_train_scaled']).to(device)
    y_qsc_train = torch.FloatTensor(data['y_qsc_train_scaled']).to(device)
    y_invc_train = torch.FloatTensor(data['y_invc_train_scaled']).to(device)
    y_foms_train = torch.FloatTensor(data['y_foms_train_scaled']).to(device)
    raw_n_train = torch.FloatTensor(data['raw_n_train']).to(device)

    X_val = torch.FloatTensor(data['X_val_scaled']).to(device)
    y_qsc_val = torch.FloatTensor(data['y_qsc_val_scaled']).to(device)
    y_invc_val = torch.FloatTensor(data['y_invc_val_scaled']).to(device)
    y_foms_val = torch.FloatTensor(data['y_foms_val_scaled']).to(device)
    raw_n_val = torch.FloatTensor(data['raw_n_val']).to(device)

    # Precompute scaler params for consistency loss
    scaler_qsc = data['scaler_qsc']
    scaler_invc = data['scaler_invc']
    scaler_foms = data['scaler_foms']

    qsc_min_log = torch.tensor(scaler_qsc.data_min_[0], dtype=torch.float32, device=device)
    qsc_range_log = torch.tensor(scaler_qsc.data_range_[0], dtype=torch.float32, device=device)
    invc_min_log = torch.tensor(scaler_invc.data_min_[0], dtype=torch.float32, device=device)
    invc_range_log = torch.tensor(scaler_invc.data_range_[0], dtype=torch.float32, device=device)
    foms_min_log = torch.tensor(scaler_foms.data_min_[0], dtype=torch.float32, device=device)
    foms_range_log = torch.tensor(scaler_foms.data_range_[0], dtype=torch.float32, device=device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    if debug:
        epochs = 5

    cons_tag = "+cons" if use_consistency else "no_cons"
    pbar = tqdm(range(epochs), desc=f"MLP({cons_tag}) seed={seed}", leave=False)

    for epoch in pbar:
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)

        L_q = criterion(outputs['pred_qsc'], y_qsc_train)
        L_c = criterion(outputs['pred_invc_sum'], y_invc_train)
        L_f = criterion(outputs['pred_foms_direct'], y_foms_train)
        L_total = L_q + L_c + L_f

        if use_consistency:
            pred_log10_qsc = outputs['pred_qsc'] * qsc_range_log + qsc_min_log
            pred_log10_invc = outputs['pred_invc_sum'] * invc_range_log + invc_min_log
            log10_foms_phys = compute_log10_foms_phys_torch(
                pred_log10_qsc, pred_log10_invc, raw_n_train.unsqueeze(-1)
            )
            foms_phys_norm = (log10_foms_phys - foms_min_log) / foms_range_log
            L_consistency = criterion(outputs['pred_foms_direct'], foms_phys_norm.detach())
            L_total = L_total + lambda_consistency * L_consistency

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_L_q = criterion(val_out['pred_qsc'], y_qsc_val)
            val_L_c = criterion(val_out['pred_invc_sum'], y_invc_val)
            val_L_f = criterion(val_out['pred_foms_direct'], y_foms_val)
            val_loss = val_L_q + val_L_c + val_L_f

            if use_consistency:
                v_log10_qsc = val_out['pred_qsc'] * qsc_range_log + qsc_min_log
                v_log10_invc = val_out['pred_invc_sum'] * invc_range_log + invc_min_log
                v_log10_foms_phys = compute_log10_foms_phys_torch(
                    v_log10_qsc, v_log10_invc, raw_n_val.unsqueeze(-1)
                )
                v_foms_phys_norm = (v_log10_foms_phys - foms_min_log) / foms_range_log
                v_L_cons = criterion(val_out['pred_foms_direct'], v_foms_phys_norm)
                val_loss = val_loss + lambda_consistency * v_L_cons

        pbar.set_postfix({'val_loss': val_loss.item()})

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model


def evaluate_mlp(model, data, raw_n, scaler_qsc, scaler_invc, scaler_foms,
                 scaler_X, dataset_name="test", X_raw=None):
    """
    评估 MLP 模型。

    Args:
        model: trained MultiTaskMLP
        data: scaled input features (N, 4)
        raw_n: original n values
        scaler_qsc/invc/foms: fitted MinMaxScaler for inverse transform
        scaler_X: fitted MinMaxScaler for features (used when X_raw is provided)
        dataset_name: for printing
        X_raw: raw features to scale (for OOD), if None uses data directly

    Returns:
        results dict
    """
    device = next(model.parameters()).device

    if X_raw is not None:
        X_scaled = scaler_X.transform(X_raw)
    else:
        X_scaled = data

    X_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        pred_qsc_norm = outputs['pred_qsc'].cpu().numpy()
        pred_invc_norm = outputs['pred_invc_sum'].cpu().numpy()
        pred_foms_norm = outputs['pred_foms_direct'].cpu().numpy()

    # Inverse transform: scaler → log10 → 10^x
    pred_qsc_log = scaler_qsc.inverse_transform(pred_qsc_norm)
    pred_invc_log = scaler_invc.inverse_transform(pred_invc_norm)
    pred_foms_log = scaler_foms.inverse_transform(pred_foms_norm)

    pred_qsc = 10.0 ** pred_qsc_log
    pred_invc = 10.0 ** pred_invc_log
    pred_foms = 10.0 ** pred_foms_log

    return pred_qsc, pred_invc, pred_foms


def evaluate_mlp_full(model, data, split, scaler_qsc, scaler_invc, scaler_foms,
                      scaler_X, dataset_name="test"):
    """
    Full evaluation of MLP on a dataset split.

    Args:
        model: trained MultiTaskMLP
        data: full data dict
        split: 'test', 'val', or 'ood'
        scaler_*: fitted scalers
        dataset_name: for printing

    Returns:
        results dict with metrics
    """
    if split == 'ood':
        # OOD data passed directly
        X_raw = data['X']
        y_qsc_log10 = data['y_qsc']
        y_invc_log10 = data['y_invc']
        y_foms_log10 = data['y_foms']
        raw_n = data['raw_n']
        pred_qsc, pred_invc, pred_foms = evaluate_mlp(
            model, None, raw_n, scaler_qsc, scaler_invc, scaler_foms,
            scaler_X, dataset_name, X_raw=X_raw,
        )
    else:
        X_scaled = data[f'X_{split}_scaled']
        y_qsc_log10 = data[f'y_qsc_{split}']
        y_invc_log10 = data[f'y_invc_{split}']
        y_foms_log10 = data[f'y_foms_{split}']
        raw_n = data[f'raw_n_{split}']
        pred_qsc, pred_invc, pred_foms = evaluate_mlp(
            model, X_scaled, raw_n, scaler_qsc, scaler_invc, scaler_foms,
            scaler_X, dataset_name,
        )

    true_qsc = 10.0 ** y_qsc_log10
    true_invc = 10.0 ** y_invc_log10
    true_foms = 10.0 ** y_foms_log10

    # FOMS_phys
    pred_foms_phys = compute_foms_phys(
        pred_qsc.flatten(), pred_invc.flatten(), raw_n
    ).reshape(-1, 1)

    tag = f"MLP {dataset_name}"
    m_qsc = compute_metrics(true_qsc, pred_qsc, f"  {tag} Qsc")
    m_invc = compute_metrics(true_invc, pred_invc, f"  {tag} invC")
    m_foms = compute_metrics(true_foms, pred_foms, f"  {tag} FOMS")
    m_foms_phys = compute_metrics(true_foms, pred_foms_phys, f"  {tag} FOMS_phys")
    m_cons = compute_consistency_metrics(
        pred_foms.flatten(), pred_foms_phys.flatten(),
        f"  {tag} consistency"
    )

    return {
        'qsc': m_qsc, 'invc': m_invc,
        'foms_direct': m_foms, 'foms_phys': m_foms_phys,
        'consistency': m_cons,
    }


# ============================================================================
# 6. 聚合与报告
# ============================================================================

def aggregate_seed_results(all_seed_results):
    """
    Aggregate results across seeds → mean ± std.

    Args:
        all_seed_results: list of dicts (one per seed), each with
            {target: {metric_name: value}}

    Returns:
        dict: {target: {metric_name: {'mean': ..., 'std': ...}}}
    """
    agg = {}
    targets = all_seed_results[0].keys()
    for target in targets:
        agg[target] = {}
        metric_names = all_seed_results[0][target].keys()
        for metric in metric_names:
            vals = [r[target][metric] for r in all_seed_results]
            agg[target][metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            }
    return agg


def format_mean_std(mean, std, fmt=".4f"):
    """Format mean±std string."""
    return f"{mean:{fmt}}±{std:{fmt}}"


def print_comparison_table(all_results, our_model_metrics):
    """
    打印完整对比表 (中文)。

    Args:
        all_results: dict of {model_name: aggregated_results}
        our_model_metrics: dict from test_metrics.json
    """
    print("\n" + "=" * 100)
    print("  T4 基线对比总结 (测试集)")
    print("=" * 100)

    header = f"{'模型':<30} {'Qsc R²_log10':>14} {'invC R²_log10':>14} {'FOMS R²_log10':>14} {'一致性 Pearson':>16}"
    print(header)
    print("-" * 100)

    for model_name, results in all_results.items():
        qsc_r2 = format_mean_std(results['qsc']['r2_log10']['mean'],
                                 results['qsc']['r2_log10']['std'])
        invc_r2 = format_mean_std(results['invc']['r2_log10']['mean'],
                                  results['invc']['r2_log10']['std'])
        foms_r2 = format_mean_std(results['foms_direct']['r2_log10']['mean'],
                                  results['foms_direct']['r2_log10']['std'])
        cons_r = format_mean_std(results['consistency']['pearson_r']['mean'],
                                 results['consistency']['pearson_r']['std'])
        print(f"  {model_name:<28} {qsc_r2:>14} {invc_r2:>14} {foms_r2:>14} {cons_r:>16}")

    # Our model
    if our_model_metrics:
        qsc_r2 = f"{our_model_metrics['qsc']['r2_log10']:.4f}"
        invc_r2 = f"{our_model_metrics['invc']['r2_log10']:.4f}"
        foms_r2 = f"{our_model_metrics['foms_direct']['r2_log10']:.4f}"
        cons_r = f"{our_model_metrics['consistency']['pearson_r']:.4f}"
        print("-" * 100)
        print(f"  {'Ours (Transformer+cons.)':<28} {qsc_r2:>14} {invc_r2:>14} {foms_r2:>14} {cons_r:>16}")

    print("=" * 100 + "\n")


def save_results(all_results, all_raw_results, output_dir, our_model_metrics):
    """
    保存结果到 JSON 和 CSV。

    Args:
        all_results: dict of {model_name: aggregated_results}
        all_raw_results: dict of {model_name: list of per-seed results}
        output_dir: output directory
        our_model_metrics: dict from test_metrics.json
    """
    os.makedirs(output_dir, exist_ok=True)

    # JSON: structured metrics
    json_data = {
        'baselines': {},
        'our_model': {},
    }

    for model_name, results in all_results.items():
        json_data['baselines'][model_name] = results

    if our_model_metrics:
        json_data['our_model'] = {
            k: {mk: float(mv) for mk, mv in v.items()}
            for k, v in our_model_metrics.items()
            if not k.startswith('_') and k != 'foms_by_magnitude_band'
        }

    # Also save per-seed raw results
    json_data['per_seed'] = {}
    for model_name, seed_results in all_raw_results.items():
        json_data['per_seed'][model_name] = {}
        for seed, result in seed_results.items():
            json_data['per_seed'][model_name][str(seed)] = {
                target: {k: float(v) for k, v in metrics.items()}
                for target, metrics in result.items()
            }

    json_path = os.path.join(output_dir, "baseline_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"[保存] {json_path}")

    # CSV: flat table for paper Table 1
    rows = []
    for model_name, results in all_results.items():
        row = {'Model': model_name}
        for target in ['qsc', 'invc', 'foms_direct']:
            r2_mean = results[target]['r2_log10']['mean']
            r2_std = results[target]['r2_log10']['std']
            mae_mean = results[target]['mae_log10']['mean']
            mae_std = results[target]['mae_log10']['std']
            row[f'{target}_R2_log10'] = format_mean_std(r2_mean, r2_std)
            row[f'{target}_MAE_log10'] = format_mean_std(mae_mean, mae_std)

        cons_mean = results['consistency']['pearson_r']['mean']
        cons_std = results['consistency']['pearson_r']['std']
        row['consistency_pearson'] = format_mean_std(cons_mean, cons_std)
        spear_mean = results['consistency']['spearman_r']['mean']
        spear_std = results['consistency']['spearman_r']['std']
        row['consistency_spearman'] = format_mean_std(spear_mean, spear_std)

        rows.append(row)

    # Add our model
    if our_model_metrics:
        row = {'Model': 'Ours (Transformer+cons.)'}
        for target in ['qsc', 'invc', 'foms_direct']:
            row[f'{target}_R2_log10'] = f"{our_model_metrics[target]['r2_log10']:.4f}"
            row[f'{target}_MAE_log10'] = f"{our_model_metrics[target]['mae_log10']:.4f}"
        row['consistency_pearson'] = f"{our_model_metrics['consistency']['pearson_r']:.4f}"
        row['consistency_spearman'] = f"{our_model_metrics['consistency']['spearman_r']:.4f}"
        rows.append(row)

    csv_path = os.path.join(output_dir, "baseline_comparison.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[保存] {csv_path}")


# ============================================================================
# 7. Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="T4: Baseline Training")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--skip_tuning", action="store_true",
                        help="跳过 XGBoost GridSearchCV，使用默认参数")
    parser.add_argument("--skip_ood", action="store_true",
                        help="跳过 OOD 评估")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  T4: Baseline Training — XGBoost×3 + Multi-MLP (±consistency)")
    print("=" * 70)

    # --- Load data ---
    print("\n[Step 1] 加载数据 (复用 T1 精确划分)")
    data = load_data_for_baselines(DATA_PATH)
    n_test = len(data['X_test'])
    print(f"  测试集大小: {n_test}")

    # Load our model metrics for comparison
    our_model_metrics = None
    if os.path.exists(OUR_MODEL_METRICS):
        with open(OUR_MODEL_METRICS, 'r') as f:
            our_model_metrics = json.load(f)
        print(f"  已加载我方模型指标: {OUR_MODEL_METRICS}")

    # Storage for all results
    all_aggregated = {}   # model_name -> aggregated metrics
    all_raw = {}          # model_name -> {seed: metrics}

    # =================================================================
    # Step 2: XGBoost ×3 with tuning
    # =================================================================
    print("\n" + "=" * 70)
    print("  [Step 2] XGBoost ×3 独立模型")
    print("=" * 70)

    # Tune on train → validate using fixed seed=42 split
    print("\n  --- XGBoost 调参 (GridSearchCV on train set) ---")
    targets_xgb = {
        'qsc': (data['X_train'], data['y_qsc_train'], data['X_val'], data['y_qsc_val']),
        'invc': (data['X_train'], data['y_invc_train'], data['X_val'], data['y_invc_val']),
        'foms': (data['X_train'], data['y_foms_train'], data['X_val'], data['y_foms_val']),
    }
    best_params_xgb = {}
    for target_name, (X_tr, y_tr, X_va, y_va) in targets_xgb.items():
        best_params_xgb[target_name] = tune_xgboost(
            X_tr, y_tr, X_va, y_va, target_name,
            skip_tuning=args.skip_tuning or args.debug,
        )

    # Train with 3 seeds
    print("\n  --- XGBoost 多种子训练 ---")
    xgb_seed_results_test = {}
    xgb_seed_results_ood = {}
    xgb_models_best = {}  # store best models for saving

    for seed in TRAIN_SEEDS:
        print(f"\n  Seed={seed}:")
        models = {}
        for target_name in ['qsc', 'invc', 'foms']:
            X_tr = data['X_train']
            y_tr = data[f'y_{target_name}_train']
            models[target_name] = train_xgboost_single(
                X_tr, y_tr, best_params_xgb[target_name], seed
            )

        # Evaluate on test
        test_data = {
            'X': data['X_test'],
            'y_qsc': data['y_qsc_test'],
            'y_invc': data['y_invc_test'],
            'y_foms': data['y_foms_test'],
        }
        result_test = evaluate_xgboost_ensemble(
            models, test_data, data['raw_n_test'], f"test seed={seed}"
        )
        xgb_seed_results_test[seed] = result_test

        if seed == 42:
            xgb_models_best = models

    # Aggregate XGBoost test results
    all_aggregated['XGB (independent ×3)'] = aggregate_seed_results(
        list(xgb_seed_results_test.values())
    )
    all_raw['XGB (independent ×3)'] = xgb_seed_results_test

    # Save best XGBoost models
    for target_name, model in xgb_models_best.items():
        model_path = os.path.join(OUTPUT_DIR, f"xgb_{target_name}_best.json")
        model.save_model(model_path)
        print(f"  [保存] {model_path}")

    # =================================================================
    # Step 3: Multi-MLP (no consistency)
    # =================================================================
    print("\n" + "=" * 70)
    print("  [Step 3] Multi-MLP (no consistency)")
    print("=" * 70)

    mlp_no_cons_seed_results = {}
    mlp_no_cons_best_model = None

    for seed in TRAIN_SEEDS:
        print(f"\n  Seed={seed}:")
        model = train_mlp(data, seed, use_consistency=False, debug=args.debug)
        result = evaluate_mlp_full(
            model, data, 'test',
            data['scaler_qsc'], data['scaler_invc'], data['scaler_foms'],
            data['scaler_X'], f"test seed={seed}"
        )
        mlp_no_cons_seed_results[seed] = result
        if seed == 42:
            mlp_no_cons_best_model = model

    all_aggregated['MLP (no consistency)'] = aggregate_seed_results(
        list(mlp_no_cons_seed_results.values())
    )
    all_raw['MLP (no consistency)'] = mlp_no_cons_seed_results

    # Save best MLP model
    mlp_path = os.path.join(OUTPUT_DIR, "mlp_no_consistency.pth")
    torch.save(mlp_no_cons_best_model.state_dict(), mlp_path)
    print(f"  [保存] {mlp_path}")

    # =================================================================
    # Step 4: Multi-MLP (+ consistency)
    # =================================================================
    print("\n" + "=" * 70)
    print("  [Step 4] Multi-MLP (+ consistency, λ=0.3)")
    print("=" * 70)

    mlp_cons_seed_results = {}
    mlp_cons_best_model = None

    for seed in TRAIN_SEEDS:
        print(f"\n  Seed={seed}:")
        model = train_mlp(
            data, seed, use_consistency=True, lambda_consistency=0.3,
            debug=args.debug,
        )
        result = evaluate_mlp_full(
            model, data, 'test',
            data['scaler_qsc'], data['scaler_invc'], data['scaler_foms'],
            data['scaler_X'], f"test seed={seed}"
        )
        mlp_cons_seed_results[seed] = result
        if seed == 42:
            mlp_cons_best_model = model

    all_aggregated['MLP (+ consistency)'] = aggregate_seed_results(
        list(mlp_cons_seed_results.values())
    )
    all_raw['MLP (+ consistency)'] = mlp_cons_seed_results

    mlp_path = os.path.join(OUTPUT_DIR, "mlp_with_consistency.pth")
    torch.save(mlp_cons_best_model.state_dict(), mlp_path)
    print(f"  [保存] {mlp_path}")

    # =================================================================
    # Step 5: OOD Evaluation (if not skipped)
    # =================================================================
    if not args.skip_ood:
        print("\n" + "=" * 70)
        print("  [Step 5] OOD 验证 (validate1 + validate2 + validate3)")
        print("=" * 70)

        ood_results = {}

        for ood_name, ood_path in OOD_FILES:
            if not os.path.exists(ood_path):
                print(f"  [跳过] OOD 文件不存在: {ood_path}")
                continue

            print(f"\n  --- {ood_name} ---")
            X_ood, y_qsc_ood, y_invc_ood, y_foms_ood, raw_n_ood = load_ood_data(ood_path)
            ood_data = {
                'X': X_ood,
                'y_qsc': y_qsc_ood,
                'y_invc': y_invc_ood,
                'y_foms': y_foms_ood,
                'raw_n': raw_n_ood,
            }

            ood_model_results = {}

            # XGBoost (use seed=42 best models)
            print(f"\n  XGBoost on {ood_name}:")
            xgb_ood = evaluate_xgboost_ensemble(
                xgb_models_best, ood_data, raw_n_ood, ood_name
            )
            ood_model_results['XGB (independent ×3)'] = xgb_ood

            # MLP no consistency (seed=42)
            print(f"\n  MLP (no cons.) on {ood_name}:")
            mlp_no_cons_ood = evaluate_mlp_full(
                mlp_no_cons_best_model, ood_data, 'ood',
                data['scaler_qsc'], data['scaler_invc'], data['scaler_foms'],
                data['scaler_X'], ood_name,
            )
            ood_model_results['MLP (no consistency)'] = mlp_no_cons_ood

            # MLP + consistency (seed=42)
            print(f"\n  MLP (+cons.) on {ood_name}:")
            mlp_cons_ood = evaluate_mlp_full(
                mlp_cons_best_model, ood_data, 'ood',
                data['scaler_qsc'], data['scaler_invc'], data['scaler_foms'],
                data['scaler_X'], ood_name,
            )
            ood_model_results['MLP (+ consistency)'] = mlp_cons_ood

            ood_results[ood_name] = ood_model_results

        # Save OOD results to the main JSON
        ood_json = {}
        for ood_name, models_results in ood_results.items():
            ood_json[ood_name] = {}
            for model_name, metrics in models_results.items():
                ood_json[ood_name][model_name] = {
                    target: {k: float(v) for k, v in m.items()}
                    for target, m in metrics.items()
                }

        ood_path = os.path.join(OUTPUT_DIR, "baseline_ood_results.json")
        with open(ood_path, 'w', encoding='utf-8') as f:
            json.dump(ood_json, f, indent=2, ensure_ascii=False)
        print(f"\n  [保存] {ood_path}")

    # =================================================================
    # Step 6: Results aggregation & comparison
    # =================================================================
    print("\n" + "=" * 70)
    print("  [Step 6] 结果汇总")
    print("=" * 70)

    # Print comparison table
    print_comparison_table(all_aggregated, our_model_metrics)

    # Save all results
    save_results(all_aggregated, all_raw, OUTPUT_DIR, our_model_metrics)

    print("\n" + "=" * 70)
    print("  T4 完成！")
    print("=" * 70)
    print(f"  输出目录: {OUTPUT_DIR}/")
    print(f"  - baseline_results.json     (结构化指标)")
    print(f"  - baseline_comparison.csv   (论文 Table 1)")
    print(f"  - xgb_*_best.json           (XGBoost 模型)")
    print(f"  - mlp_*.pth                 (MLP 模型)")
    if not args.skip_ood:
        print(f"  - baseline_ood_results.json (OOD 指标)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
