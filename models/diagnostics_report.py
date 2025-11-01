#!/usr/bin/env python3
"""
diagnostics_report.py

Quick diagnostics to find why many predictions are equal (e.g. 50/100).
Outputs:
 - diagnostics_report.json
 - diagnostics_summary.csv

Run:
  python diagnostics_report.py --processed ./processed --artifacts ./artifacts --model_name xfin_esg_model_v1
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sys
import math

def load_npz(path):
    npz = np.load(path, allow_pickle=True)
    return npz

def try_load_model(artifacts_dir, model_name):
    models_dir = Path(artifacts_dir) / 'models'
    job = models_dir / f"{model_name}.joblib"
    booster = models_dir / f"{model_name}_booster.txt"
    model = None
    booster_obj = None
    if job.exists():
        try:
            model = joblib.load(job)
            return model, 'joblib'
        except Exception as e:
            print("joblib load failed:", e)
    if booster.exists():
        try:
            import lightgbm as lgb
            booster_obj = lgb.Booster(model_file=str(booster))
            return booster_obj, 'booster'
        except Exception as e:
            print("booster load failed:", e)
    return None, None

def predict_with(model_obj, X):
    # handle sklearn LGBM or lightgbm.Booster
    try:
        if hasattr(model_obj, 'predict'):
            # sklearn LGBM or joblib'd Booster
            try:
                preds = model_obj.predict(X)
            except TypeError:
                # some wrappers require num_iteration kw
                preds = model_obj.predict(X, num_iteration=getattr(model_obj, 'best_iteration_', None))
            return np.asarray(preds, dtype=float)
        else:
            # fall back
            return np.asarray(model_obj.predict(X), dtype=float)
    except Exception as e:
        raise

def main(args):
    proc = Path(args.processed)
    art = Path(args.artifacts)
    model_name = args.model_name

    # check processed files exist
    val_path = proc / 'val.npz'
    train_path = proc / 'train.npz'
    test_path = proc / 'test.npz'
    for p in [val_path, train_path, test_path]:
        if not p.exists():
            print(f"Missing {p}; ensure preprocessing was run. Exiting.")
            sys.exit(1)

    # load val
    val = np.load(val_path, allow_pickle=True)
    X_val = val['X']
    y_val = val['y'] if 'y' in val.files else None
    y_proxy = val['y_proxy'] if 'y_proxy' in val.files else None
    sw_val = val['sample_weight'] if 'sample_weight' in val.files else None

    model_obj, model_type = try_load_model(art, model_name)
    if model_obj is None:
        print("No model found in artifacts/models. Exiting.")
        sys.exit(1)

    preds = predict_with(model_obj, X_val)
    preds = np.asarray(preds).reshape(-1)

    # diagnostics
    u = np.unique(np.round(preds, 6))
    frac_close_50 = float((np.isclose(preds, 50.0, atol=args.tol)).mean())
    n_equal_50 = int((np.isclose(preds, 50.0, atol=args.tol)).sum())

    # feature variance check
    variances = np.var(X_val, axis=0)
    zero_var_count = int((variances == 0).sum())
    low_var_count = int((variances < args.low_var_thresh).sum())

    # correlation with proxy
    corr_proxy = None
    if y_proxy is not None:
        try:
            corr = np.corrcoef(preds, y_proxy)[0,1]
            corr_proxy = None if math.isnan(corr) else float(corr)
        except Exception:
            corr_proxy = None

    report = {
        'n_val': int(X_val.shape[0]),
        'pred_mean': float(np.mean(preds)),
        'pred_std': float(np.std(preds)),
        'unique_pred_count': int(len(u)),
        'fraction_pred_near_50': frac_close_50,
        'count_pred_near_50': n_equal_50,
        'zero_variance_feature_count': zero_var_count,
        'low_variance_feature_count': low_var_count,
        'corr_with_y_proxy': corr_proxy,
        'model_type_loaded': model_type
    }

    # Summarize top offending features (by variance)
    feat_var_series = pd.Series(variances)
    feat_var_series.index = [f"f{i}" for i in range(len(variances))]
    feat_var_summary = feat_var_series.sort_values().head(30).to_dict()

    report['lowest_variance_features_preview'] = feat_var_summary

    # write outputs
    out_json = Path(args.out_dir) / 'diagnostics_report.json'
    out_csv = Path(args.out_dir) / 'diagnostics_summary.csv'
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)

    # CSV: rowwise summary (single row) and some arrays
    df = pd.DataFrame([report])
    df.to_csv(out_csv, index=False)

    print("Diagnostics saved:", out_json, out_csv)
    print("Quick summary:")
    print(f"  predictions mean={report['pred_mean']:.3f} std={report['pred_std']:.3f} unique={report['unique_pred_count']}")
    print(f"  fraction near 50 = {report['fraction_pred_near_50']:.3%} (count {report['count_pred_near_50']})")
    if report['corr_with_y_proxy'] is not None:
        print(f"  corr(preds, y_proxy) = {report['corr_with_y_proxy']:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="./processed")
    p.add_argument("--artifacts", default="./artifacts")
    p.add_argument("--model_name", default="xfin_esg_model_v1")
    p.add_argument("--out_dir", default="./diagnostics_out")
    p.add_argument("--tol", type=float, default=1e-6, help="tolerance for 'equal to 50'")
    p.add_argument("--low_var_thresh", type=float, default=1e-12, help="threshold for low variance detection")
    args = p.parse_args()
    main(args)
