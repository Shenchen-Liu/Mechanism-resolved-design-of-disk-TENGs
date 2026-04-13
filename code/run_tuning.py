#!/usr/bin/env python3
"""
Hyperparameter tuning wrapper for train_multitask_physics.py.
Runs training with custom output directories to avoid overwriting.

Usage:
    python run_tuning.py --exp_name A --embed_dim 256 --num_layers 3 ...
"""

import os
import sys
import importlib

# Parse --exp_name first, then pass rest to train script
exp_name = None
remaining_args = []
args_iter = iter(sys.argv[1:])
for arg in args_iter:
    if arg == "--exp_name":
        exp_name = next(args_iter)
    else:
        remaining_args.append(arg)

if not exp_name:
    print("ERROR: --exp_name is required")
    sys.exit(1)

# Override sys.argv for the train script
sys.argv = [sys.argv[0]] + remaining_args

# Patch output directories before importing train module
import train_multitask_physics as train_mod
train_mod.CHECKPOINT_DIR = f"../checkpoints_multitask_physics/tuning_{exp_name}"
train_mod.ARTIFACT_DIR = f"../artifacts_multitask_physics/tuning_{exp_name}"
train_mod.OUTPUT_DIR = f"../outputs_multitask_physics/tuning_{exp_name}"

# Run
train_mod.main()
