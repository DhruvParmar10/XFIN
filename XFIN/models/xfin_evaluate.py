#!/usr/bin/env python3
"""
xfin_evaluate.py

Evaluate the trained model's performance against the baseline (sector proxy).

This script reads the model's test metrics from the training report
and compares them to the baseline metrics computed from the test set.

It assumes that 'test.npz' contains:
 - 'y' (the true labels)
 - 'y_proxy' (the baseline sector_esg_proxy values for the test set)
"""
import os
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Helper for backward-compatible RMSE ---
def rmse(a, b):
    """Calculate RMSE, compatible with all sklearn versions."""
    return np.sqrt(mean_squared_error(a, b))
# -----------------------------------------

def evaluate_model(processed_dir, artifacts_dir):
    """
    Load model and baseline metrics and print a comparison.
    """
    processed_dir = Path(processed_dir)
    artifacts_dir = Path(artifacts_dir)

    # --- 1. Load Model Performance ---
    report_path = artifacts_dir / 'training_report.json'
    if not report_path.exists():
        print(f"Error: Training report not found at {report_path}")
        sys.exit(1)
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    try:
        model_metrics = report['test_metrics']
        model_test_mae = model_metrics['mae']
        model_test_rmse = model_metrics['rmse']
        n_test = report['n_test']
    except KeyError:
        print("Error: training_report.json is missing 'test_metrics' or 'n_test'.")
        sys.exit(1)

    # --- 2. Load Baseline Data ---
    test_path = processed_dir / 'test.npz'
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        sys.exit(1)
        
    npz = np.load(test_path, allow_pickle=True)
    
    if 'y' not in npz.files:
        print("Error: 'y' (true labels) not found in test.npz.")
        sys.exit(1)
    y_test = npz['y']

    # This is the critical dependency:
    if 'y_proxy' not in npz.files:
        print("Error: 'y_proxy' (baseline values) not found in test.npz.")
        print("Please update 'xfin_preprocess_split.py' to save 'y_proxy' in test.npz and re-run it.")
        sys.exit(1)
    
    y_proxy = npz['y_proxy']

    if len(y_test) != len(y_proxy):
        print("Error: Mismatch between y_test and y_proxy lengths.")
        sys.exit(1)

    # --- 3. Calculate Baseline Performance ---
    baseline_mae = mean_absolute_error(y_test, y_proxy)
    baseline_rmse = rmse(y_test, y_proxy)

    # --- 4. Print Comparison Report ---
    print("=" * 60)
    print(f"XFIN ESG Model Evaluation (Test Set, n={n_test})")
    print("=" * 60)
    
    print(f"\nModel: {report.get('model_name', 'LGBM Regressor')}")
    print(f"Best Iteration: {report.get('best_iteration', 'N/A')}")
    print(f"Trained in: {report.get('training_time_seconds', 0.0):.2f}s")
    
    print("\n--- Performance Comparison ---")
    
    # Simple table formatting
    header = f"| {'Metric':<6} | {'Model (LGBM)':<14} | {'Baseline (Proxy)':<18} |"
    print(header)
    print(f"|{'-'*8}|{'-'*16}|{'-'*20}|")
    print(f"| {'MAE':<6} | {model_test_mae:<14.4f} | {baseline_mae:<18.4f} |")
    print(f"| {'RMSE':<6} | {model_test_rmse:<14.4f} | {baseline_rmse:<18.4f} |")

    print("\n--- Verdict ---")
    if model_test_mae < baseline_mae:
        mae_diff = baseline_mae - model_test_mae
        print(f"✅ Success: Model's MAE is {mae_diff:.4f} points better than the baseline.")
    else:
        mae_diff = model_test_mae - baseline_mae
        print(f"❌ Warning: Model's MAE is {mae_diff:.4f} points worse than the baseline.")

    if model_test_rmse < baseline_rmse:
        rmse_diff = baseline_rmse - model_test_rmse
        print(f"✅ Success: Model's RMSE is {rmse_diff:.4f} points better than the baseline.")
    else:
        rmse_diff = model_test_rmse - baseline_rmse
        print(f"❌ Warning: Model's RMSE is {rmse_diff:.4f} points worse than the baseline.")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate model vs. baseline")
    p.add_argument("--processed_dir", "-p", default="./processed",
                   help="Directory containing test.npz")
    p.add_argument("--artifacts_dir", "-a", default="./artifacts",
                   help="Directory containing training_report.json")
    args = p.parse_args()

    evaluate_model(
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir
    )