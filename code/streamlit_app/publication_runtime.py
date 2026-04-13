#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared publication/SI runtime helpers for Streamlit pages.

This module treats the current publication-generation scripts as the single
source of truth. Streamlit pages use these helpers to keep interactive plots
aligned with the final main-text and supporting-information asset builders.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
from streamlit_utils import (
    APP_ROOT,
    CORE_DIR,
    DD_GRID,
    E_VALUES,
    HH_GRID,
    N_VALUES,
    add_foms_contours,
    add_point_marker,
    normalize_log_series,
)

for path in [CORE_DIR, APP_ROOT]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_publication_figures as pub_main
import generate_si_assets as pub_si


REPO_ROOT = pub_main.ROOT
MAIN_MANIFEST_PATH = pub_main.SRC_DIR / "manifest.json"
SI_MANIFEST_PATH = pub_si.SOURCE_DIR / "manifest.json"

DOMINANCE_COLORSCALE = [
    [0.0, pub_main.REGIME_COLORS[pub_main.REGIME_CAPACITANCE]],
    [0.5, pub_main.REGIME_COLORS[pub_main.REGIME_MIXED]],
    [1.0, pub_main.REGIME_COLORS[pub_main.REGIME_CHARGE]],
]
FIG4_CV_COLORSCALE = [
    [0.0, pub_main.mcolors.to_hex(pub_main.FIG4_CV_CMAP(0.0))],
    [0.33, pub_main.mcolors.to_hex(pub_main.FIG4_CV_CMAP(0.33))],
    [0.66, pub_main.mcolors.to_hex(pub_main.FIG4_CV_CMAP(0.66))],
    [1.0, pub_main.mcolors.to_hex(pub_main.FIG4_CV_CMAP(1.0))],
]
SI_CHARGE_RATIO_COLORSCALE = [
    [0.0, pub_main.mcolors.to_hex(pub_si.CHARGE_RATIO_CMAP(0.0))],
    [0.33, pub_main.mcolors.to_hex(pub_si.CHARGE_RATIO_CMAP(0.33))],
    [0.66, pub_main.mcolors.to_hex(pub_si.CHARGE_RATIO_CMAP(0.66))],
    [1.0, pub_main.mcolors.to_hex(pub_si.CHARGE_RATIO_CMAP(1.0))],
]
SI_AGREEMENT_COLORSCALE = [
    [0.0, pub_main.mcolors.to_hex(pub_si.AGREEMENT_CMAP(0.0))],
    [0.5, pub_main.mcolors.to_hex(pub_si.AGREEMENT_CMAP(0.5))],
    [1.0, pub_main.mcolors.to_hex(pub_si.AGREEMENT_CMAP(1.0))],
]

DEFAULT_MAIN_SCENARIO_LABEL = "mid_E_mid_d"
DEFAULT_SI_REFERENCE_K = 50
DEFAULT_SI_REFERENCE_THRESHOLD = 0.62

_PREDICT_FN_REGISTRY: dict[int, Any] = {}


def resolve_publication_path(path_value: str | Path | None) -> Path | None:
    """Resolve absolute or repository-relative manifest paths."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


@dataclass(frozen=True)
class ScenarioPreset:
    """Reusable preset for main-text design scenes."""

    label: str
    title: str
    E: float
    dd: float


@dataclass(frozen=True)
class RecommendationSearchPreset:
    """Reusable preset for design-recommendation search pools."""

    label: str
    title: str
    description: str
    n_values: tuple[float, ...]
    E_values: tuple[float, ...]
    dd_values: tuple[float, ...]
    hh_values: tuple[float, ...]


MAIN_SCENARIO_PRESETS = tuple(
    ScenarioPreset(
        label=item["label"],
        title=item["title"],
        E=float(item["E"]),
        dd=float(item["dd"]),
    )
    for item in pub_main.SCENARIOS
)
RECOMMENDATION_SEARCH_PRESETS = (
    RecommendationSearchPreset(
        label="fig04_pooled_publication",
        title="Fig.4 pooled publication pool",
        description=(
            "Matches the pooled Fig.4 design-candidate space: publication n grid, "
            "E in {1, 3, 10}, d/R fixed at 0.125, and the 18-point h/R grid."
        ),
        n_values=tuple(float(value) for value in pub_main.N_VALUES),
        E_values=tuple(float(item["E"]) for item in pub_main.ASHBY_SCENARIOS),
        dd_values=tuple(sorted({float(item["dd"]) for item in pub_main.ASHBY_SCENARIOS})),
        hh_values=tuple(float(value) for value in pub_main.HH_VALUES),
    ),
    RecommendationSearchPreset(
        label="full_dense_training_pool",
        title="Full dense training pool",
        description=(
            "Covers the full training-aligned search pool used by the current "
            "Streamlit recommender: training n/E sets with dense d/R and h/R grids."
        ),
        n_values=tuple(float(value) for value in N_VALUES),
        E_values=tuple(float(value) for value in E_VALUES),
        dd_values=tuple(float(value) for value in DD_GRID),
        hh_values=tuple(float(value) for value in HH_GRID),
    ),
)


def get_main_scenario_presets() -> tuple[ScenarioPreset, ...]:
    """Return publication-aligned scene presets."""
    return MAIN_SCENARIO_PRESETS


def get_main_scenario_preset(label: str) -> ScenarioPreset:
    """Resolve a scenario preset by label."""
    for preset in MAIN_SCENARIO_PRESETS:
        if preset.label == label:
            return preset
    for preset in MAIN_SCENARIO_PRESETS:
        if preset.label == DEFAULT_MAIN_SCENARIO_LABEL:
            return preset
    return MAIN_SCENARIO_PRESETS[0]


def get_recommendation_search_presets() -> tuple[RecommendationSearchPreset, ...]:
    """Return design-recommendation search-space presets."""
    return RECOMMENDATION_SEARCH_PRESETS


def get_recommendation_search_preset(label: str) -> RecommendationSearchPreset:
    """Resolve a recommendation preset by label."""
    for preset in RECOMMENDATION_SEARCH_PRESETS:
        if preset.label == label:
            return preset
    return RECOMMENDATION_SEARCH_PRESETS[0]


def register_predict_fn(predict_fn: Any) -> int:
    """Register a predict function for cached scene computation."""
    key = id(predict_fn)
    _PREDICT_FN_REGISTRY[key] = predict_fn
    return key


def _require_predict_fn(predict_fn_key: int) -> Any:
    predict_fn = _PREDICT_FN_REGISTRY.get(predict_fn_key)
    if predict_fn is None:
        raise RuntimeError(
            "Predict function is not registered. Recreate the page session and retry."
        )
    return predict_fn


@st.cache_data
def _read_json(path_str: str) -> dict[str, Any] | None:
    path = resolve_publication_path(path_str)
    if path is None:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def _read_csv(path_str: str) -> pd.DataFrame:
    path = resolve_publication_path(path_str)
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_main_publication_manifest() -> dict[str, Any] | None:
    """Load the main-text publication manifest."""
    return _read_json(str(MAIN_MANIFEST_PATH))


def load_si_publication_manifest() -> dict[str, Any] | None:
    """Load the SI publication manifest."""
    return _read_json(str(SI_MANIFEST_PATH))


def load_publication_table(path: str | Path | None) -> pd.DataFrame:
    """Read a CSV table if it exists, otherwise return an empty frame."""
    if path is None:
        return pd.DataFrame()
    return _read_csv(str(path))


def load_grid_spatial_metrics() -> dict[str, Any] | None:
    """Load the precomputed SI grid-comparison metrics JSON."""
    return _read_json(str(pub_si.GRID_SPATIAL_METRICS))


def load_threshold_sensitivity_frame() -> pd.DataFrame:
    """Load Fig.S6 source data."""
    return _read_csv(str(pub_si.THRESH_SENS))


def load_grid_sensitivity_frame() -> pd.DataFrame:
    """Load Fig.S7 source data."""
    return _read_csv(str(pub_si.GRID_SENS))


def pick_preview_asset(files: list[str] | tuple[str, ...]) -> Path | None:
    """Choose the most display-friendly asset from a file list."""
    candidates = [
        resolved for item in files if (resolved := resolve_publication_path(item)) is not None
    ]
    if not candidates:
        return None

    for suffix in (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".drawio"):
        for path in candidates:
            if path.suffix.lower() == suffix and path.exists():
                return path

    for path in candidates:
        if path.exists():
            return path
    return None


def summarise_main_manifest(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Return lightweight summary metrics for the main-text manifest."""
    if manifest is None:
        return {"figure_count": 0, "table_count": 0}
    return {
        "figure_count": len(manifest.get("figures", {})),
        "table_count": len(manifest.get("tables", {})),
        "generated_at": manifest.get("generated_at", "unknown"),
    }


def summarise_si_manifest(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Return lightweight summary metrics for the SI manifest."""
    if manifest is None:
        return {"figure_count": 0, "table_count": 0}
    return {
        "figure_count": len(manifest.get("figures", {})),
        "table_count": len(manifest.get("tables", {})),
    }


@st.cache_data(show_spinner="Computing publication-aligned scene...")
def compute_publication_scene(
    predict_fn_key: int,
    E_fixed: float,
    dd_fixed: float,
    dominance_threshold: float,
    perturb_frac: float,
) -> dict[str, Any]:
    """
    Build a Fig.3/Fig.4 scene using the released helper logic.

    The main-text generation script remains the source of truth for constants
    and selection rules. Streamlit only wraps the outputs for interactivity.
    """
    from utils_mechanism_multitask import (
        compute_design_regime_grid,
        compute_robustness_grid,
    )

    predict_fn = _require_predict_fn(predict_fn_key)
    design = compute_design_regime_grid(
        predict_fn,
        E_fixed=E_fixed,
        dd_fixed=dd_fixed,
        n_values=pub_main.N_VALUES,
        hh_values=pub_main.HH_VALUES,
        delta_log=pub_main.DESIGN_DELTA_LOG,
        dominance_threshold=dominance_threshold,
    )
    foms_map, cv_map, worst_ratio_map = compute_robustness_grid(
        predict_fn,
        E_fixed=E_fixed,
        dd_fixed=dd_fixed,
        n_values=pub_main.N_VALUES,
        hh_values=pub_main.HH_VALUES,
        perturb_frac=perturb_frac,
    )
    phase_point = pub_main.select_phase_point(design)
    robust_point = pub_main.select_robust_point(foms_map, cv_map, worst_ratio_map)
    safe_mask = pub_main.select_safe_mask(foms_map, cv_map, worst_ratio_map)
    transition_row = pub_main.select_transition_row(design)

    return {
        "E_fixed": float(E_fixed),
        "dd_fixed": float(dd_fixed),
        "design": design,
        "foms_map": foms_map,
        "cv_map": cv_map,
        "worst_ratio_map": worst_ratio_map,
        "phase_point": phase_point,
        "robust_point": robust_point,
        "safe_mask": safe_mask,
        "safe_overlay": gaussian_filter(
            safe_mask.astype(float), sigma=pub_main.SAFE_SMOOTH_SIGMA
        ),
        "transition_row": int(transition_row),
        "transition_hh": float(pub_main.HH_VALUES[transition_row]),
        "perf_vmin": float(
            np.nanpercentile(np.log10(np.maximum(design["foms"], 1e-30)), 2)
        ),
        "perf_vmax": float(
            np.nanpercentile(np.log10(np.maximum(design["foms"], 1e-30)), 98)
        ),
        "cv_vmax": float(np.nanpercentile(cv_map * 100.0, 95)),
    }


def get_hh_tick_values() -> np.ndarray:
    """Return the publication hh tick positions used in Streamlit plots."""
    ticks = np.array([0.004, 0.01, 0.03, 0.1, 0.3, 1.0], dtype=float)
    mask = (ticks >= np.min(pub_main.HH_VALUES)) & (ticks <= np.max(pub_main.HH_VALUES))
    return ticks[mask]


def _normalize_subset(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1] within the masked subset."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(values, dtype=float)
    subset = values[mask]
    if subset.size == 0:
        return out
    vmin = float(np.nanmin(subset))
    vmax = float(np.nanmax(subset))
    span = vmax - vmin
    if not np.isfinite(span) or span < 1e-12:
        out[mask] = 1.0
        return out
    out[mask] = (values[mask] - vmin) / span
    return out


def add_decision_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Attach pooled Fig.4 decision semantics to a search result frame."""
    out = df.copy()
    if len(out) == 0:
        out["foms_top30"] = []
        out["safe_zone"] = []
        out["recommend_score"] = []
        return out, np.nan

    foms_values = out["FOMS"].to_numpy(dtype=float)
    cv_values = out["CV_pct"].to_numpy(dtype=float)
    retention_values = out["Retention_pct"].to_numpy(dtype=float)

    foms_threshold = float(np.nanpercentile(foms_values, 70))
    top30 = foms_values >= foms_threshold
    norm_fom = _normalize_subset(foms_values, top30)
    norm_cv = _normalize_subset(cv_values, top30)
    norm_retention = _normalize_subset(retention_values, top30)

    score = np.full(len(out), -np.inf, dtype=float)
    score[top30] = (
        norm_fom[top30]
        - pub_main.FIG4_CV_LAMBDA * norm_cv[top30]
        + pub_main.FIG4_RETENTION_GAMMA * norm_retention[top30]
    )
    safe_zone = (
        top30
        & (cv_values <= pub_main.SAFE_CV_THRESHOLD * 100.0)
        & (retention_values >= pub_main.SAFE_WORST_RATIO_THRESHOLD * 100.0)
    )

    out["foms_top30"] = top30
    out["safe_zone"] = safe_zone
    out["recommend_score"] = score
    return out, foms_threshold


def select_recommended_row(df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Pick the pooled design recommendation from a filtered decision frame."""
    if len(df) == 0:
        return None, "none"

    candidates = df[np.isfinite(df["recommend_score"])].copy()
    if len(candidates) > 0:
        return (
            candidates.sort_values("recommend_score", ascending=False).iloc[0],
            "recommend",
        )
    return df.sort_values("CV_pct", ascending=True).iloc[0], "low-CV point"


def compute_pareto_frontier(
    df: pd.DataFrame,
    *,
    x_col: str = "CV_pct",
    y_col: str = "log10_FOMS",
) -> pd.DataFrame:
    """Extract the non-dominated frontier in the CV-FOMS plane."""
    if len(df) == 0:
        return df.copy()

    ordered = df.sort_values([x_col, y_col], ascending=[True, False]).copy()
    keep = []
    best_y = -np.inf
    for idx, row in ordered.iterrows():
        if row[y_col] > best_y + 1e-12:
            keep.append(idx)
            best_y = row[y_col]
    return ordered.loc[keep].sort_values(x_col)


@st.cache_data(show_spinner="Searching publication-aligned design pool...")
def compute_recommendation_search(
    predict_fn_key: int,
    n_values: tuple[float, ...],
    E_values: tuple[float, ...],
    dd_values: tuple[float, ...],
    hh_values: tuple[float, ...],
    perturb_frac: float,
    min_foms: float,
    max_cv: float,
    min_retention: float,
) -> dict[str, Any]:
    """Evaluate a pooled design search space using the shared publication semantics."""
    predict_fn = _require_predict_fn(predict_fn_key)
    n_values = tuple(float(value) for value in n_values)
    E_values = tuple(float(value) for value in E_values)
    dd_values = tuple(float(value) for value in dd_values)
    hh_values = tuple(float(value) for value in hh_values)

    if not n_values or not E_values or not dd_values or not hh_values:
        raise ValueError("Search-space definitions must not be empty.")

    grid = list(itertools.product(n_values, E_values, dd_values, hh_values))
    df_grid = pd.DataFrame(grid, columns=["n", "E", "dd", "hh"])
    base_result = predict_fn(
        df_grid["n"].to_numpy(dtype=float),
        df_grid["E"].to_numpy(dtype=float),
        df_grid["dd"].to_numpy(dtype=float),
        df_grid["hh"].to_numpy(dtype=float),
    )
    foms_base = np.asarray(base_result["FOMS_direct"], dtype=float)

    foms_samples = [foms_base]
    for sign_h in (-1.0, 1.0):
        for sign_d in (-1.0, 1.0):
            dd_pert = df_grid["dd"].to_numpy(dtype=float) * (1.0 + sign_d * perturb_frac)
            hh_pert = df_grid["hh"].to_numpy(dtype=float) * (1.0 + sign_h * perturb_frac)
            pert_result = predict_fn(
                df_grid["n"].to_numpy(dtype=float),
                df_grid["E"].to_numpy(dtype=float),
                dd_pert,
                hh_pert,
            )
            foms_samples.append(np.asarray(pert_result["FOMS_direct"], dtype=float))

    foms_stack = np.stack(foms_samples, axis=0)
    mean_foms = np.mean(foms_stack, axis=0)
    std_foms = np.std(foms_stack, axis=0)
    cv_pct = std_foms / (mean_foms + 1e-30) * 100.0
    retention_pct = np.min(foms_stack, axis=0) / (foms_base + 1e-30) * 100.0

    df_result = pd.DataFrame(
        {
            "n": df_grid["n"].to_numpy(dtype=float),
            "E": df_grid["E"].to_numpy(dtype=float),
            "dd": df_grid["dd"].to_numpy(dtype=float),
            "hh": df_grid["hh"].to_numpy(dtype=float),
            "FOMS": foms_base,
            "log10_FOMS": np.log10(np.maximum(foms_base, 1e-30)),
            "Qsc": np.asarray(base_result["Qsc_MACRS"], dtype=float),
            "invC_sum": np.asarray(base_result["invC_sum"], dtype=float),
            "CV_pct": cv_pct,
            "Retention_pct": retention_pct,
        }
    )
    df_result, foms_threshold = add_decision_columns(df_result)

    mask = np.ones(len(df_result), dtype=bool)
    if min_foms > 0:
        mask &= df_result["FOMS"] >= min_foms
    if max_cv < 100:
        mask &= df_result["CV_pct"] <= max_cv
    if min_retention > 0:
        mask &= df_result["Retention_pct"] >= min_retention

    df_filtered = df_result[mask].copy()
    recommended_row, recommended_label = select_recommended_row(df_filtered)
    peak_row = (
        None
        if len(df_filtered) == 0
        else df_filtered.sort_values("FOMS", ascending=False).iloc[0]
    )
    frontier_df = compute_pareto_frontier(df_filtered)

    return {
        "all_results": df_result,
        "filtered_results": df_filtered,
        "n_designs": len(df_result),
        "n_feasible": int(mask.sum()),
        "foms_threshold": foms_threshold,
        "safe_count": int(df_filtered["safe_zone"].sum()) if len(df_filtered) > 0 else 0,
        "recommended_row": recommended_row,
        "recommended_label": recommended_label,
        "peak_row": peak_row,
        "frontier_df": frontier_df,
    }


def build_decision_map_figure(
    df_filtered: pd.DataFrame,
    *,
    recommended_row: pd.Series | None,
    foms_threshold: float,
) -> go.Figure:
    """Build a Plotly decision map styled to closely match the released Fig.4 panel."""
    fig = go.Figure()

    if len(df_filtered) > 0:
        feasible_only = df_filtered[~df_filtered["safe_zone"]].copy()
        safe_df = df_filtered[df_filtered["safe_zone"]].copy()

        if len(feasible_only) > 0:
            feasible_custom = np.empty((len(feasible_only), 9), dtype=object)
            feasible_custom[:, 0] = feasible_only["n"].to_numpy()
            feasible_custom[:, 1] = feasible_only["E"].to_numpy()
            feasible_custom[:, 2] = feasible_only["dd"].to_numpy()
            feasible_custom[:, 3] = feasible_only["hh"].to_numpy()
            feasible_custom[:, 4] = feasible_only["FOMS"].to_numpy()
            feasible_custom[:, 5] = feasible_only["CV_pct"].to_numpy()
            feasible_custom[:, 6] = feasible_only["Retention_pct"].to_numpy()
            feasible_custom[:, 7] = feasible_only["recommend_score"].to_numpy()
            feasible_custom[:, 8] = np.where(
                feasible_only["safe_zone"].to_numpy(dtype=bool), "Yes", "No"
            )
            fig.add_trace(
                go.Scatter(
                    x=feasible_only["CV_pct"],
                    y=feasible_only["log10_FOMS"],
                    mode="markers",
                    name="all pooled designs",
                    marker=dict(
                        size=5.0,
                        color=pub_main.ASHBY_ALL_FACE,
                        opacity=pub_main.ASHBY_ALL_ALPHA,
                        line=dict(width=0.0, color="rgba(0,0,0,0)"),
                    ),
                    customdata=feasible_custom,
                    hovertemplate=(
                        "n=%{customdata[0]:.0f}<br>E=%{customdata[1]:.2f}<br>d/R=%{customdata[2]:.4f}"
                        "<br>h/R=%{customdata[3]:.6f}<br>FOMS=%{customdata[4]:.3e}"
                        "<br>CV=%{customdata[5]:.2f}%<br>Retention=%{customdata[6]:.1f}%"
                        "<br>Score=%{customdata[7]:.3f}<br>safe-zone=%{customdata[8]}"
                        "<extra>all pooled designs</extra>"
                    ),
                )
            )

        if len(safe_df) > 0:
            safe_custom = np.empty((len(safe_df), 9), dtype=object)
            safe_custom[:, 0] = safe_df["n"].to_numpy()
            safe_custom[:, 1] = safe_df["E"].to_numpy()
            safe_custom[:, 2] = safe_df["dd"].to_numpy()
            safe_custom[:, 3] = safe_df["hh"].to_numpy()
            safe_custom[:, 4] = safe_df["FOMS"].to_numpy()
            safe_custom[:, 5] = safe_df["CV_pct"].to_numpy()
            safe_custom[:, 6] = safe_df["Retention_pct"].to_numpy()
            safe_custom[:, 7] = safe_df["recommend_score"].to_numpy()
            safe_custom[:, 8] = np.where(
                safe_df["safe_zone"].to_numpy(dtype=bool), "Yes", "No"
            )
            fig.add_trace(
                go.Scatter(
                    x=safe_df["CV_pct"],
                    y=safe_df["log10_FOMS"],
                    mode="markers",
                    name="safe candidates",
                    marker=dict(
                        size=8.0,
                        color=pub_main.SAFE_CANDIDATE_FACE,
                        opacity=pub_main.SAFE_CANDIDATE_ALPHA,
                        line=dict(
                            width=0.8,
                            color=pub_main.SAFE_CANDIDATE_EDGE,
                        ),
                    ),
                    customdata=safe_custom,
                    hovertemplate=(
                        "n=%{customdata[0]:.0f}<br>E=%{customdata[1]:.2f}<br>d/R=%{customdata[2]:.4f}"
                        "<br>h/R=%{customdata[3]:.6f}<br>FOMS=%{customdata[4]:.3e}"
                        "<br>CV=%{customdata[5]:.2f}%<br>Retention=%{customdata[6]:.1f}%"
                        "<br>Score=%{customdata[7]:.3f}<br>safe-zone=%{customdata[8]}"
                        "<extra>safe candidates</extra>"
                    ),
                )
            )

    fig.add_vline(
        x=pub_main.SAFE_CV_THRESHOLD * 100.0,
        line_width=pub_main.CRITERION_LINE_WIDTH,
        line_dash="dash",
        line_color=pub_main.CRITERION_LINE_COLOR,
        opacity=pub_main.CRITERION_LINE_ALPHA,
    )
    if np.isfinite(foms_threshold):
        fig.add_hline(
            y=np.log10(max(foms_threshold, 1e-30)),
            line_width=pub_main.CRITERION_LINE_WIDTH,
            line_dash="dash",
            line_color=pub_main.CRITERION_LINE_COLOR,
            opacity=pub_main.CRITERION_LINE_ALPHA,
        )

    if recommended_row is not None:
        fig.add_trace(
            go.Scatter(
                x=[recommended_row["CV_pct"]],
                y=[recommended_row["log10_FOMS"]],
                mode="markers",
                name="representative candidate",
                marker=dict(
                    symbol="circle",
                    size=10.0,
                    color=pub_main.REPRESENTATIVE_FACE,
                    line=dict(
                        width=pub_main.REPRESENTATIVE_EDGEWIDTH * 1.4,
                        color=pub_main.REPRESENTATIVE_EDGE,
                    ),
                ),
                hovertemplate=(
                    f"n={recommended_row['n']:.0f}<br>E={recommended_row['E']:.2f}<br>d/R={recommended_row['dd']:.4f}"
                    f"<br>h/R={recommended_row['hh']:.6f}<br>FOMS={recommended_row['FOMS']:.3e}"
                    f"<br>CV={recommended_row['CV_pct']:.2f}%<br>Retention={recommended_row['Retention_pct']:.1f}%"
                    f"<br>Score={recommended_row['recommend_score']:.3f}"
                    "<br>safe-zone=True<extra>representative candidate</extra>"
                ),
            )
        )

    fig.update_layout(
        height=430,
        xaxis_title="CV (%)",
        yaxis_title="log10(FOMS)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.26,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=30, t=55, b=90),
        title=dict(
            text="Global decision map",
            x=0.0,
            xanchor="left",
            font=dict(size=14),
        ),
        plot_bgcolor="white",
    )
    fig.update_xaxes(range=[0.0, 15.0], showline=True, linecolor="black", ticks="outside")
    fig.update_yaxes(range=[-2.5, -1.15], showline=True, linecolor="black", ticks="outside")
    if np.isfinite(foms_threshold):
        fig.add_annotation(
            x=14.75,
            y=np.log10(max(foms_threshold, 1e-30)) + 0.018,
            text="top-30% global (FOM_S) criterion",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=11, color="#5F6B76"),
            bgcolor="rgba(255,255,255,0.72)",
            bordercolor="rgba(0,0,0,0)",
        )
    fig.add_annotation(
        x=pub_main.SAFE_CV_THRESHOLD * 100.0 + 0.14,
        y=-2.38,
        text="CV = 5% criterion",
        textangle=90,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=11, color="#5F6B76"),
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor="rgba(0,0,0,0)",
    )
    return fig


def apply_design_layout(
    fig: go.Figure,
    title: str,
    colorbar_title: str,
    *,
    height: int = 330,
) -> go.Figure:
    """Apply a publication-style layout to 2D design maps."""
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="n (sector pairs)",
            type="log",
            tickmode="array",
            tickvals=pub_main.N_VALUES,
            ticktext=[str(int(value)) for value in pub_main.N_VALUES],
        ),
        yaxis=dict(
            title="h/R",
            type="log",
            tickmode="array",
            tickvals=get_hh_tick_values(),
            ticktext=[f"{value:g}" for value in get_hh_tick_values()],
        ),
        height=height,
        margin=dict(l=60, r=30, t=55, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    if fig.data and hasattr(fig.data[0], "colorbar"):
        fig.data[0].colorbar.title = colorbar_title
    return fig


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a hex color into an rgba() string for Plotly overlays."""
    color = color.lstrip("#")
    if len(color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _add_refined_safe_region_overlay(fig: go.Figure, safe_mask: np.ndarray) -> go.Figure:
    """Add the refined Fig.4 safe-window fill and boundary in Plotly."""
    if not np.any(safe_mask):
        return fig

    x_safe, y_safe, smooth_mask = pub_main.build_refined_safe_window(
        np.asarray(safe_mask, dtype=bool),
        sigma=0.75,
        refine=8,
    )
    x_actual = 2 ** np.asarray(x_safe, dtype=float)
    y_actual = 2 ** np.asarray(y_safe, dtype=float)

    fig.add_trace(
        go.Heatmap(
            x=x_actual,
            y=y_actual,
            z=smooth_mask,
            zmin=0.0,
            zmax=1.0,
            colorscale=[
                [0.0, "rgba(255,255,255,0.00)"],
                [0.499, "rgba(255,255,255,0.00)"],
                [0.5, _hex_to_rgba(pub_main.SAFE_GREEN, pub_main.SAFE_ZONE_ALPHA)],
                [1.0, _hex_to_rgba(pub_main.SAFE_GREEN, pub_main.SAFE_ZONE_ALPHA)],
            ],
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Contour(
            x=x_actual,
            y=y_actual,
            z=smooth_mask,
            contours=dict(coloring="none", start=0.5, end=0.5, size=1.0),
            line=dict(color=pub_main.SAFE_ZONE_EDGE, width=1.6),
            opacity=pub_main.SAFE_ZONE_EDGE_ALPHA,
            showscale=False,
            hoverinfo="skip",
        )
    )

    edge_specs = [
        ("left", smooth_mask[:, 0] >= 0.5),
        ("right", smooth_mask[:, -1] >= 0.5),
        ("bottom", smooth_mask[0, :] >= 0.5),
        ("top", smooth_mask[-1, :] >= 0.5),
    ]
    for edge_name, edge_mask in edge_specs:
        for start, end in pub_main._iter_true_runs(edge_mask):
            if edge_name == "left":
                x_vals = [x_actual[0]] * (end - start)
                y_vals = y_actual[start:end]
            elif edge_name == "right":
                x_vals = [x_actual[-1]] * (end - start)
                y_vals = y_actual[start:end]
            elif edge_name == "bottom":
                x_vals = x_actual[start:end]
                y_vals = [y_actual[0]] * (end - start)
            else:
                x_vals = x_actual[start:end]
                y_vals = [y_actual[-1]] * (end - start)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    line=dict(color=pub_main.SAFE_ZONE_EDGE, width=1.6),
                    opacity=pub_main.SAFE_ZONE_EDGE_ALPHA,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    return fig


def _apply_fig4_heatmap_layout(
    fig: go.Figure,
    *,
    title: str,
    colorbar_title: str,
) -> go.Figure:
    """Apply a Plotly layout that visually tracks the released Fig.4 heatmap panel."""
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=14)),
        xaxis=dict(
            title="",
            type="log",
            tickmode="array",
            tickvals=pub_main.N_VALUES,
            ticktext=[str(int(value)) for value in pub_main.N_VALUES],
            showline=True,
            linecolor="black",
            ticks="outside",
            ticklen=4,
        ),
        yaxis=dict(
            title="h/R (log scale)",
            type="log",
            tickmode="array",
            tickvals=get_hh_tick_values(),
            ticktext=[f"{value:g}" for value in get_hh_tick_values()],
            showline=True,
            linecolor="black",
            ticks="outside",
            ticklen=4,
        ),
        height=380,
        margin=dict(l=72, r=55, t=58, b=46),
        plot_bgcolor="white",
    )
    if fig.data and hasattr(fig.data[0], "colorbar"):
        fig.data[0].colorbar.title = colorbar_title
        fig.data[0].colorbar.thickness = 12
        fig.data[0].colorbar.len = 0.78
        fig.data[0].colorbar.tickfont = dict(size=10)
    return fig


def _configure_publication_mpl_style() -> None:
    """Apply the exact publication matplotlib style before drawing fig-style panels."""
    pub_main.configure_style()


def _render_safe_window(ax, safe_mask: np.ndarray) -> None:
    """Render the refined safe-window overlay exactly like the released heatmaps."""
    if not np.any(safe_mask):
        return
    x_safe, y_safe, smooth_mask = pub_main.build_refined_safe_window(
        safe_mask,
        sigma=0.75,
        refine=8,
    )
    ax.contourf(
        x_safe,
        y_safe,
        smooth_mask,
        levels=[0.5, 1.1],
        colors=[pub_main.SAFE_GREEN],
        alpha=pub_main.SAFE_ZONE_ALPHA,
        zorder=5,
    )
    ax.contour(
        x_safe,
        y_safe,
        smooth_mask,
        levels=[0.5],
        colors=[pub_main.SAFE_ZONE_EDGE],
        linewidths=1.0,
        linestyles=["-"],
        alpha=pub_main.SAFE_ZONE_EDGE_ALPHA,
        zorder=6,
    )
    pub_main.draw_safe_window_boundary_closure(
        ax,
        x_safe,
        y_safe,
        smooth_mask,
        level=0.5,
        color=pub_main.SAFE_ZONE_EDGE,
        linewidth=1.0,
        alpha=pub_main.SAFE_ZONE_EDGE_ALPHA,
    )


def build_fig4_heatmap_mpl(
    scene: dict[str, Any],
    *,
    title: str,
    map_kind: str = "cv",
) -> plt.Figure:
    """
    Build a Fig.4-style single heatmap panel using the released styling.

    `map_kind="cv"` matches the released panel semantics.
    `map_kind="retention"` reuses the same panel style for the retention view.
    """
    _configure_publication_mpl_style()

    n_grid, hh_grid = pub_main.log2_mesh()
    fig, ax = plt.subplots(figsize=(3.65, 3.35))
    fig.subplots_adjust(left=0.18, right=0.96, top=0.84, bottom=0.24)

    if map_kind == "cv":
        z_values = np.asarray(scene["cv_map"], dtype=float) * 100.0
        cmap = pub_main.FIG4_CV_CMAP
        vmin = 0.0
        vmax = float(scene["cv_vmax"])
        cbar_label = "CV (%)"
    elif map_kind == "retention":
        z_values = np.asarray(scene["worst_ratio_map"], dtype=float) * 100.0
        cmap = "viridis"
        vmin = 0.0
        vmax = 100.0
        cbar_label = "Worst-case retention (%)"
    else:
        raise ValueError(f"Unsupported map_kind: {map_kind}")

    im = ax.pcolormesh(
        n_grid,
        hh_grid,
        z_values,
        cmap=cmap,
        shading="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    _render_safe_window(ax, np.asarray(scene["safe_mask"], dtype=bool))
    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        pad=pub_main.FIG_TITLE_PAD,
        fontsize=8,
    )
    pub_main.configure_design_axes(ax, show_xlabel=False, show_ylabel=True)

    cax = fig.add_axes([0.18, 0.10, 0.78, 0.035])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    pub_main.style_colorbar(cbar, cbar_label, fontsize=7.3, labelpad=1.5)
    cbar.ax.tick_params(pad=1.5)
    return fig


def build_fig4_decision_map_mpl(
    df_filtered: pd.DataFrame,
    *,
    recommended_row: pd.Series | None,
    foms_threshold: float,
) -> plt.Figure:
    """Build the exact Fig.4 decision-map style from the released script."""
    _configure_publication_mpl_style()

    fig, ax = plt.subplots(figsize=(4.7, 3.55))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.84, bottom=0.30)

    ashby_df = df_filtered.rename(
        columns={
            "CV_pct": "cv_pct",
            "log10_FOMS": "log10_foms",
        }
    ).copy()
    safe_df = ashby_df[ashby_df["safe_zone"]].copy()
    core_recommend = None
    if recommended_row is not None:
        core_recommend = recommended_row.rename(
            {
                "CV_pct": "cv_pct",
                "log10_FOMS": "log10_foms",
            }
        )

    log_foms_threshold = (
        np.log10(max(float(foms_threshold), 1e-30))
        if np.isfinite(foms_threshold)
        else float(np.nanmin(ashby_df["log10_foms"])) if len(ashby_df) > 0 else -2.0
    )
    pub_main._draw_ashby_panel(
        ax,
        ashby_df,
        safe_df,
        core_recommend,
        log_foms_threshold,
    )
    legend_handles, legend_labels = pub_main._build_ashby_legend()
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.05),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=pub_main.FIG_CALLOUT_SMALL_SIZE,
        columnspacing=1.0,
        handlelength=1.1,
    )
    return fig


def build_performance_figure(
    scene: dict[str, Any],
    *,
    title: str,
    highlight_slice: bool = True,
) -> go.Figure:
    """Render the Plotly counterpart of the Fig.3 performance landscape."""
    design = scene["design"]
    phase_point = scene["phase_point"]
    log_foms = np.log10(np.maximum(design["foms"], 1e-30))
    fig = go.Figure(
        data=go.Heatmap(
            x=pub_main.N_VALUES,
            y=pub_main.HH_VALUES,
            z=log_foms,
            colorscale="Viridis",
            zmin=scene["perf_vmin"],
            zmax=scene["perf_vmax"],
            hovertemplate="n=%{x}<br>h/R=%{y:.4f}<br>log10(FOMS)=%{z:.2f}<extra></extra>",
        )
    )
    add_foms_contours(
        fig,
        pub_main.N_VALUES,
        pub_main.HH_VALUES,
        design["foms"],
        line_color="rgba(255,255,255,0.88)",
    )
    if highlight_slice:
        fig.add_hline(
            y=scene["transition_hh"],
            line_color="white",
            line_width=1.2,
            line_dash="dash",
        )
        add_point_marker(
            fig,
            phase_point,
            "phase point",
            fill_color="white",
            line_color="#111827",
            size=11,
        )
    return apply_design_layout(fig, title, "log10(FOMS)")


def build_mechanism_figure(
    scene: dict[str, Any],
    *,
    title: str,
    highlight_slice: bool = True,
) -> go.Figure:
    """Render the Plotly counterpart of the Fig.3 mechanism map."""
    design = scene["design"]
    fig = go.Figure(
        data=go.Heatmap(
            x=pub_main.N_VALUES,
            y=pub_main.HH_VALUES,
            z=design["f_charge"],
            colorscale=DOMINANCE_COLORSCALE,
            zmin=0.0,
            zmax=1.0,
            hovertemplate="n=%{x}<br>h/R=%{y:.4f}<br>f_charge=%{z:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Contour(
            x=pub_main.N_VALUES,
            y=pub_main.HH_VALUES,
            z=design["f_charge"],
            contours=dict(coloring="none", start=0.5, end=0.5, size=1.0),
            line=dict(color="white", width=1.1, dash="dash"),
            showscale=False,
            hoverinfo="skip",
        )
    )
    if highlight_slice:
        fig.add_hline(
            y=scene["transition_hh"],
            line_color="#111827",
            line_width=1.0,
            line_dash="dash",
        )
    return apply_design_layout(fig, title, "f_charge")


def build_cv_figure(scene: dict[str, Any], *, title: str) -> go.Figure:
    """Render a Plotly Fig.4-style CV heatmap with detailed hover."""
    customdata = np.empty(np.asarray(scene["cv_map"]).shape + (5,), dtype=object)
    customdata[..., 0] = float(scene["E_fixed"])
    customdata[..., 1] = float(scene["dd_fixed"])
    customdata[..., 2] = np.asarray(scene["foms_map"], dtype=float)
    customdata[..., 3] = np.asarray(scene["worst_ratio_map"], dtype=float) * 100.0
    customdata[..., 4] = np.where(np.asarray(scene["safe_mask"], dtype=bool), "Yes", "No")
    fig = go.Figure(
        data=go.Heatmap(
            x=pub_main.N_VALUES,
            y=pub_main.HH_VALUES,
            z=scene["cv_map"] * 100.0,
            colorscale=FIG4_CV_COLORSCALE,
            zmin=0.0,
            zmax=scene["cv_vmax"],
            customdata=customdata,
            hovertemplate=(
                "n=%{x:.0f}<br>E=%{customdata[0]:.2f}<br>d/R=%{customdata[1]:.4f}<br>h/R=%{y:.4f}"
                "<br>CV=%{z:.2f}%<br>FOMS=%{customdata[2]:.3e}"
                "<br>Retention=%{customdata[3]:.1f}%<br>safe-zone=%{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )
    _add_refined_safe_region_overlay(fig, np.asarray(scene["safe_mask"], dtype=bool))
    return _apply_fig4_heatmap_layout(fig, title=title, colorbar_title="CV (%)")


def build_retention_figure(scene: dict[str, Any], *, title: str) -> go.Figure:
    """Render a Plotly heatmap using the Fig.4 panel style for retention diagnostics."""
    customdata = np.empty(np.asarray(scene["worst_ratio_map"]).shape + (5,), dtype=object)
    customdata[..., 0] = float(scene["E_fixed"])
    customdata[..., 1] = float(scene["dd_fixed"])
    customdata[..., 2] = np.asarray(scene["foms_map"], dtype=float)
    customdata[..., 3] = np.asarray(scene["cv_map"], dtype=float) * 100.0
    customdata[..., 4] = np.where(np.asarray(scene["safe_mask"], dtype=bool), "Yes", "No")
    fig = go.Figure(
        data=go.Heatmap(
            x=pub_main.N_VALUES,
            y=pub_main.HH_VALUES,
            z=scene["worst_ratio_map"] * 100.0,
            colorscale="Viridis",
            zmin=0.0,
            zmax=100.0,
            customdata=customdata,
            hovertemplate=(
                "n=%{x:.0f}<br>E=%{customdata[0]:.2f}<br>d/R=%{customdata[1]:.4f}<br>h/R=%{y:.4f}"
                "<br>Retention=%{z:.1f}%<br>FOMS=%{customdata[2]:.3e}"
                "<br>CV=%{customdata[3]:.2f}%<br>safe-zone=%{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )
    _add_refined_safe_region_overlay(fig, np.asarray(scene["safe_mask"], dtype=bool))
    return _apply_fig4_heatmap_layout(
        fig,
        title=title,
        colorbar_title="Worst-case retention (%)",
    )


def build_transition_slice_figure(scene: dict[str, Any]) -> go.Figure:
    """Render the Fig.3 transition slice along n."""
    design = scene["design"]
    row = scene["transition_row"]
    f_charge_row = np.asarray(design["f_charge"][row, :], dtype=float)
    q2_norm = normalize_log_series(design["qsc"][row, :] ** 2)
    invc_norm = normalize_log_series(design["invc"][row, :])
    foms_norm = normalize_log_series(design["foms"][row, :])
    bounds = pub_main.segment_bounds(pub_main.N_VALUES)

    fig = go.Figure()
    for idx, f_value in enumerate(f_charge_row):
        if f_value < 0.4:
            fill_color = pub_main.REGIME_COLORS[pub_main.REGIME_CAPACITANCE]
        elif f_value > 0.6:
            fill_color = pub_main.REGIME_COLORS[pub_main.REGIME_CHARGE]
        else:
            fill_color = pub_main.REGIME_COLORS[pub_main.REGIME_MIXED]
        fig.add_vrect(
            x0=bounds[idx],
            x1=bounds[idx + 1],
            fillcolor=fill_color,
            opacity=0.12,
            line_width=0,
            layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=pub_main.N_VALUES,
            y=q2_norm,
            mode="lines+markers",
            name="Qsc²",
            line=dict(color=pub_main.LINE_COLORS["q2"], width=2.0),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pub_main.N_VALUES,
            y=invc_norm,
            mode="lines+markers",
            name="1/Csum",
            line=dict(color=pub_main.LINE_COLORS["invc"], width=2.0, dash="dash"),
            marker=dict(symbol="square", size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pub_main.N_VALUES,
            y=foms_norm,
            mode="lines+markers",
            name="FOMS",
            line=dict(color=pub_main.LINE_COLORS["foms"], width=2.2, dash="dot"),
            marker=dict(symbol="diamond", size=7),
        )
    )
    fig.update_layout(
        title="mechanism transition along n",
        xaxis=dict(
            title="n (log scale)",
            type="log",
            tickmode="array",
            tickvals=pub_main.N_VALUES,
            ticktext=[str(int(value)) for value in pub_main.N_VALUES],
            range=[
                np.log10(pub_main.N_VALUES[0] / 1.3),
                np.log10(pub_main.N_VALUES[-1] * 1.3),
            ],
        ),
        yaxis=dict(title="normalized value", range=[-0.02, 1.14]),
        height=430,
        margin=dict(l=70, r=30, t=60, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        annotations=[
            dict(
                x=item["x"],
                y=item["y"],
                text=item["text"],
                showarrow=False,
                font=dict(color=item["color"], size=11),
                xanchor=item["ha"],
            )
            for item in pub_main.FIG3G_REGION_LABELS
        ]
        + [
            dict(
                x=14.5,
                y=0.75,
                text="Qsc²",
                showarrow=False,
                font=dict(color=pub_main.LINE_COLORS["q2"], size=11),
            ),
            dict(
                x=6.3,
                y=0.43,
                text="1/Csum",
                showarrow=False,
                font=dict(color=pub_main.LINE_COLORS["invc"], size=11),
            ),
            dict(
                x=24.0,
                y=0.92,
                text="FOMS",
                showarrow=False,
                font=dict(color=pub_main.LINE_COLORS["foms"], size=11),
            ),
            dict(
                x=0.02,
                y=0.03,
                xref="paper",
                yref="paper",
                text=f"slice at h/R={scene['transition_hh']:.3g}",
                showarrow=False,
                font=dict(color="#475467", size=11),
            ),
        ],
    )
    return fig


def build_si_s6_heatmap(
    df: pd.DataFrame,
    *,
    k_values: list[int],
    threshold_range: tuple[float, float],
) -> go.Figure:
    """Build the Fig.S6 charge/cap ratio heatmap with custom filters."""
    filtered = df[
        df["k_neighbors"].isin(k_values)
        & (df["dominance_threshold"] >= threshold_range[0])
        & (df["dominance_threshold"] <= threshold_range[1])
    ].copy()
    pivot = filtered.pivot(
        index="k_neighbors",
        columns="dominance_threshold",
        values="charge_cap_ratio",
    ).sort_index()
    fig = go.Figure(
        data=go.Heatmap(
            x=[f"{float(value):.2f}" for value in pivot.columns],
            y=[str(int(value)) for value in pivot.index],
            z=pivot.values,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}",
            colorscale=SI_CHARGE_RATIO_COLORSCALE,
            colorbar=dict(title="Charge/cap"),
            hovertemplate=(
                "k=%{y}<br>threshold=%{x}<br>charge/cap=%{z:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Fig.S6a · charge-dominance exceeds capacitance",
        xaxis_title="Dominance threshold",
        yaxis_title="k (neighbors)",
        height=330,
        margin=dict(l=60, r=30, t=55, b=45),
    )
    return fig


def build_si_s6_line_plot(
    df: pd.DataFrame,
    *,
    k_values: list[int],
    threshold_range: tuple[float, float],
    reference_k: int,
    reference_threshold: float,
) -> go.Figure:
    """Build the Fig.S6 charge-dominant percentage line plot."""
    filtered = df[
        df["k_neighbors"].isin(k_values)
        & (df["dominance_threshold"] >= threshold_range[0])
        & (df["dominance_threshold"] <= threshold_range[1])
    ].copy()
    fig = go.Figure()
    palette = [
        pub_si.TASK_COLORS["qsc"],
        pub_si.TASK_COLORS["foms_direct"],
        pub_si.TASK_COLORS["invc"],
        pub_si.TASK_COLORS["foms_phys"],
        "#E69F00",
    ]
    line_styles = ["solid", "dash", "dot", "dashdot", "longdash"]
    markers = ["circle", "square", "triangle-up", "diamond", "triangle-down"]

    for idx, k_value in enumerate(sorted(filtered["k_neighbors"].unique())):
        subset = filtered[filtered["k_neighbors"] == k_value].sort_values(
            "dominance_threshold"
        )
        fig.add_trace(
            go.Scatter(
                x=subset["dominance_threshold"],
                y=subset["charge_pct"],
                mode="lines+markers",
                name=f"k={int(k_value)}",
                line=dict(
                    color=palette[idx % len(palette)],
                    width=2.0,
                    dash=line_styles[idx % len(line_styles)],
                ),
                marker=dict(symbol=markers[idx % len(markers)], size=8),
            )
        )

    reference_row = filtered[
        (filtered["k_neighbors"] == reference_k)
        & np.isclose(filtered["dominance_threshold"], reference_threshold)
    ]
    if not reference_row.empty:
        fig.add_vline(
            x=reference_threshold,
            line_color="#C6CDD4",
            line_width=1.0,
            line_dash="dot",
        )
        fig.add_trace(
            go.Scatter(
                x=[reference_threshold],
                y=[reference_row["charge_pct"].iloc[0]],
                mode="markers",
                name="reference",
                marker=dict(symbol="star", size=14, color="red"),
            )
        )

    fig.update_layout(
        title="Fig.S6b · reference setting lies in a stable zone",
        xaxis_title="Dominance threshold",
        yaxis_title="Charge-dominant (%)",
        height=330,
        margin=dict(l=60, r=30, t=55, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    return fig


def build_si_s7_fraction_plot(
    df: pd.DataFrame,
    *,
    resolutions: list[int],
) -> go.Figure:
    """Build the Fig.S7 class-fraction stability bar chart."""
    filtered = df[df["hh_n_points"].isin(resolutions)].copy().sort_values("hh_n_points")
    labels = [f"hh={int(value)}" for value in filtered["hh_n_points"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=filtered["charge_pct"],
            name="Charge-dom.",
            marker_color=pub_si.REGIME_COLORS["charge"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=filtered["cap_pct"],
            name="Cap-dom.",
            marker_color=pub_si.REGIME_COLORS["cap"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=filtered["mixed_pct"],
            name="Mixed",
            marker_color=pub_si.REGIME_COLORS["mixed"],
        )
    )
    fig.update_layout(
        title="Fig.S7a · class fractions change minimally",
        xaxis_title="Grid resolution",
        yaxis_title="Percentage (%)",
        barmode="group",
        height=330,
        margin=dict(l=60, r=30, t=55, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    return fig


def build_si_s7_agreement_plot(
    metrics: dict[str, Any] | None,
    *,
    resolutions: list[int],
) -> go.Figure | None:
    """Build the Fig.S7 pairwise spatial-agreement heatmap."""
    if not metrics:
        return None

    selected = sorted(resolutions)
    agreement = np.eye(len(selected))
    iou = np.eye(len(selected))
    index = {value: idx for idx, value in enumerate(selected)}

    for item in metrics.get("pairwise_metrics", []):
        left, right = item["pair"]
        if left in index and right in index:
            i = index[left]
            j = index[right]
            agreement[i, j] = agreement[j, i] = item["agreement"]
            iou[i, j] = iou[j, i] = item["mean_iou"]

    text = [
        [f"{agreement[i, j] * 100:.1f}%<br>IoU {iou[i, j]:.2f}" for j in range(len(selected))]
        for i in range(len(selected))
    ]
    fig = go.Figure(
        data=go.Heatmap(
            x=[f"hh={value}" for value in selected],
            y=[f"hh={value}" for value in selected],
            z=agreement,
            zmin=0.90,
            zmax=1.0,
            colorscale=SI_AGREEMENT_COLORSCALE,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="Pixel-wise agreement"),
            hovertemplate="%{x} vs %{y}<br>agreement=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Fig.S7b · spatial agreement stays high",
        height=330,
        margin=dict(l=60, r=30, t=55, b=45),
    )
    return fig
