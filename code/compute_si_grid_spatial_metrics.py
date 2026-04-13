#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "figures_publication" / "src" / "si"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

from experiment_sensitivity import (  # noqa: E402
    CONFIG,
    REGIME_CAPACITANCE,
    REGIME_CHARGE,
    REGIME_MIXED,
    run_t92_grid_resolution,
)


def main():
    cfg = dict(CONFIG)
    cfg["checkpoint_dir"] = str(ROOT / "checkpoints_multitask_physics")
    cfg["artifact_dir"] = str(ROOT / "artifacts_multitask_physics")
    cfg["output_dir"] = str(ROOT / "outputs" / "sensitivity")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    df, regime_maps, hh_configs = run_t92_grid_resolution(cfg, device)

    labels = [int(row["hh_n_points"]) for _, row in df.iterrows()]
    target_hh = max(labels)
    resampled = []
    for result in regime_maps:
        regime = np.asarray(result["regime"])
        factor = target_hh // regime.shape[0]
        resampled.append(np.repeat(regime, factor, axis=0))

    pairs = []
    class_defs = [
        (REGIME_CHARGE, "charge"),
        (REGIME_CAPACITANCE, "cap"),
        (REGIME_MIXED, "mixed"),
    ]

    for i in range(len(resampled)):
        for j in range(i + 1, len(resampled)):
            a = resampled[i]
            b = resampled[j]
            valid = np.isfinite(a) & np.isfinite(b)
            agreement = float((a[valid] == b[valid]).mean())

            ious = {}
            mean_iou_vals = []
            for code, name in class_defs:
                am = a[valid] == code
                bm = b[valid] == code
                union = int((am | bm).sum())
                inter = int((am & bm).sum())
                val = (inter / union) if union else None
                ious[name] = val
                if val is not None:
                    mean_iou_vals.append(val)

            pairs.append({
                "pair": [labels[i], labels[j]],
                "agreement": agreement,
                "mean_iou": float(np.mean(mean_iou_vals)),
                "ious": ious,
            })

    payload = {
        "fractions": df.to_dict(orient="records"),
        "pairwise_metrics": pairs,
        "target_hh_rows": target_hh,
        "n_values": cfg["t92_n_values"],
        "maps": [
            {
                "hh_n_points": int(row["hh_n_points"]),
                "regime": np.asarray(result["regime"]).astype(int).tolist(),
                "hh_values": np.asarray(hh_values, dtype=float).tolist(),
            }
            for row, result, hh_values in zip(df.to_dict(orient="records"), regime_maps, hh_configs)
        ],
    }
    out_path = SOURCE_DIR / "grid_resolution_spatial_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
