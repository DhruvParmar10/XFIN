#!/usr/bin/env python3
"""
xfin_explain_esg.py

Generate SHAP explanations for the trained XFIN ESG model.

Modes:
 - Explain a sample from processed set (train/val/test) by index
 - Explain a single raw-row CSV (single-row) by transforming with preprocessor

Outputs:
 - JSON file with base value and list of {feature, value, shap}
 - Optional waterfall PNG plot

Usage examples:
# explain validation sample idx 0, save JSON + plot
python xfin_explain_esg.py --processed_dir ./processed --artifacts_dir ./artifacts \
  --set val --index 0 --out_json ./explain_sample0.json --out_plot ./explain_sample0.png

# explain using a raw CSV (single-row CSV with original feature columns)
python xfin_explain_esg.py --processed_dir ./processed --artifacts_dir ./artifacts \
  --raw_csv ./single_row_raw.csv --out_json ./explain_from_raw.json --out_plot ./explain_raw.png
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------
def load_processed_npz(processed_dir, split_name):
    p = Path(processed_dir) / f"{split_name}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Processed split file not found: {p}")
    npz = np.load(p, allow_pickle=True)
    X = npz['X']
    y = npz['y'] if 'y' in npz.files else None
    # optionally y_proxy may exist for baseline checks
    extras = {k: npz[k] for k in npz.files if k not in ('X','y')}
    return X, y, extras

def find_model_and_explainer(artifacts_dir, model_name):
    art = Path(artifacts_dir)
    models_dir = art / 'models'
    # prefer explainer saved by training
    expl_paths = list(models_dir.glob(f"{model_name}*_shap_explainer.*"))
    expl = None
    if expl_paths:
        try:
            expl = joblib.load(expl_paths[0])
        except Exception:
            expl = None

    # try joblib model file first
    model_file = models_dir / f"{model_name}.joblib"
    booster_txt = models_dir / f"{model_name}_booster.txt"
    model_obj = None
    booster_obj = None
    if model_file.exists():
        try:
            model_obj = joblib.load(model_file)
            # if this is a booster (lgb Booster) loaded via joblib it can be used as-is
        except Exception:
            model_obj = None
    if booster_txt.exists():
        try:
            import lightgbm as lgb
            booster_obj = lgb.Booster(model_file=str(booster_txt))
        except Exception:
            booster_obj = None

    # if no joblib model, try to detect any other file
    if model_obj is None and booster_obj is None:
        # try any joblib in models_dir
        candidates = list(models_dir.glob("*.joblib"))
        for c in candidates:
            try:
                loaded = joblib.load(c)
                # check for Booster or sklearn estimator
                if hasattr(loaded, 'predict') or (str(type(loaded)).lower().find('booster')>=0):
                    model_obj = loaded
                    break
            except Exception:
                continue

    return model_obj, booster_obj, expl

def build_explainer(model_obj, booster_obj):
    # prefer booster
    target = booster_obj if booster_obj is not None else model_obj
    if target is None:
        raise RuntimeError("No model/booster available to build explainer.")
    explainer = shap.TreeExplainer(target)
    return explainer

def make_feature_dataframe(X_row, feature_names):
    # X_row: 1D numpy array (transformed features)
    if feature_names is None:
        # create generic names
        feature_names = [f"f{i}" for i in range(X_row.shape[0])]
    return pd.DataFrame([X_row], columns=feature_names)

def save_waterfall_plot(exp_obj, out_path, show=False):
    # exp_obj: shap.Explanation (single sample or array)
    plt.figure(figsize=(8,6))
    try:
        # shap.plots.waterfall accepts an Explanation or raw values
        shap.plots.waterfall(exp_obj[0] if isinstance(exp_obj, list) or (hasattr(exp_obj,'values') and exp_obj.values.shape[0]>1) else exp_obj)
    except Exception:
        # fallback: create Explanation manually
        shap.plots.waterfall(exp_obj)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

# -------------------------
# Main
# -------------------------
def main(args):
    processed_dir = Path(args.processed_dir)
    artifacts_dir = Path(args.artifacts_dir)
    metadata_path = processed_dir / 'metadata.json'
    preproc_path = processed_dir / 'preprocessor.joblib'

    if not metadata_path.exists():
        print(f"metadata.json not found in {processed_dir}. Exiting.")
        sys.exit(1)
    metadata = json.load(open(metadata_path, 'r'))
    feature_names = metadata.get('feature_names', None)

    # load preprocessor if raw csv mode used
    preprocessor = None
    if args.raw_csv:
        if not preproc_path.exists():
            print(f"preprocessor.joblib not found at {preproc_path} — cannot transform raw CSV. Exiting.")
            sys.exit(1)
        preprocessor = joblib.load(preproc_path)

    # load model / booster / explainer
    model_obj, booster_obj, saved_expl = find_model_and_explainer(artifacts_dir, args.model_name)
    explainer = None
    if saved_expl is not None:
        explainer = saved_expl
    else:
        explainer = build_explainer(model_obj, booster_obj)

    # Prepare the sample to explain:
    if args.raw_csv:
        # raw CSV path must be a single-row CSV (or first row used)
        raw_df = pd.read_csv(args.raw_csv)
        if raw_df.shape[0] < 1:
            print("Raw CSV contains no rows. Exiting.")
            sys.exit(1)
        row = raw_df.iloc[[0]]  # keep DataFrame shape (1, ncols)
        # transform with preprocessor (the same pipeline used during training)
        X_trans = preprocessor.transform(row)
        # X_trans is numpy array shape (1, n_features_transformed)
        X_sample = np.asarray(X_trans)
        sample_source = f"raw_csv:{args.raw_csv}"
    else:
        # pick from processed split (train/val/test)
        split = args.set.lower()
        if split not in ('train','val','test'):
            print("--set must be one of train|val|test when not using --raw_csv")
            sys.exit(1)
        X_all, y_all, extras = load_processed_npz(processed_dir, split)
        idx = int(args.index)
        if idx < 0 or idx >= X_all.shape[0]:
            print(f"Index out of range for {split} (0..{X_all.shape[0]-1}). Exiting.")
            sys.exit(1)
        X_sample = X_all[idx:idx+1]
        sample_source = f"{split}[{idx}]"
        # try to find true label if available
        y_val = y_all[idx] if y_all is not None else None

    # compute SHAP values
    print(f"Using explainer = {type(explainer)} ; explaining sample source = {sample_source}")
    try:
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        # SHAP sometimes returns list (for multiclass) — still handle
        print("explainer.shap_values error:", e)
        shap_values = explainer.shap_values(X_sample)

    # shap_values may be a list (multiclass) or array; for regression typically an array (n_samples, n_features)
    if isinstance(shap_values, list):
        # pick first output if regression-like single-output wrapped in list
        shap_arr = shap_values[0]
    else:
        shap_arr = shap_values

    # single-sample array
    shap_arr = np.asarray(shap_arr)
    if shap_arr.ndim == 2 and shap_arr.shape[0] == 1:
        shap_sample = shap_arr[0]
    elif shap_arr.ndim == 1:
        shap_sample = shap_arr
    else:
        # if multiple outputs shape (n_outputs, n_features) try to flatten first output
        shap_sample = shap_arr[0]

    # get base value (expected_value)
    base_value = None
    try:
        base_value = explainer.expected_value
    except Exception:
        try:
            base_value = explainer.expected_value[0]
        except Exception:
            base_value = None

    # Prepare feature names and values
    if feature_names is None:
        # fallback feature names as f0..fN
        feature_names_local = [f"f{i}" for i in range(shap_sample.shape[0])]
    else:
        feature_names_local = feature_names
    if len(feature_names_local) != shap_sample.shape[0]:
        # mismatch — attempt to trim/pad
        print("Warning: feature_names length != shap vector length. Attempting to align.")
        min_len = min(len(feature_names_local), shap_sample.shape[0])
        feature_names_local = feature_names_local[:min_len]
        shap_sample = shap_sample[:min_len]

    # Get feature values — if we explained from raw -> X_sample are transformed numeric features
    feat_values = X_sample.flatten()[:len(feature_names_local)]

    # Build JSON output structure
    features = []
    for name, val, sv in zip(feature_names_local, feat_values, shap_sample):
        features.append({
            'feature': name,
            'value': float(np.round(float(val), 6)) if np.isfinite(val) else None,
            'shap': float(np.round(float(sv), 8)) if np.isfinite(sv) else None,
            'abs_shap': float(abs(sv))
        })

    # sort by absolute shap descending
    features_sorted = sorted(features, key=lambda x: x['abs_shap'], reverse=True)

    out_json = {
        'sample_source': sample_source,
        'base': float(base_value) if (base_value is not None and np.isfinite(base_value)) else None,
        'features': features_sorted
    }

    # Write JSON
    out_json_path = Path(args.out_json) if args.out_json else (Path.cwd() / 'shap_explanation.json')
    with open(out_json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"Wrote explanation JSON to {out_json_path}")

    # Optional waterfall plot
    if args.out_plot:
        # create shap.Explanation object for plotting
        try:
            explanation = shap.Explanation(values=shap_sample.reshape(1, -1),
                                           base_values=np.array([base_value]) if base_value is not None else None,
                                           data=feat_values.reshape(1,-1),
                                           feature_names=feature_names_local)
            save_path = Path(args.out_plot)
            # generate waterfall
            shap.plots.waterfall(explanation[0], show=False)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Wrote waterfall plot to {save_path}")
        except Exception as e:
            print("Could not create waterfall plot:", e)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate SHAP explanation for an ESG model prediction.")
    p.add_argument("--processed_dir", "-p", default="./processed", help="Directory with train/val/test .npz and metadata.json and preprocessor.joblib")
    p.add_argument("--artifacts_dir", "-a", default="./artifacts", help="Directory containing trained model under models/")
    p.add_argument("--model_name", default="xfin_esg_model_v1", help="Base model name used when training")
    p.add_argument("--set", default="val", help="Which processed split to pick from (train/val/test) - used when not using --raw_csv")
    p.add_argument("--index", type=int, default=0, help="Index in the chosen split to explain")
    p.add_argument("--raw_csv", default=None, help="Single-row raw CSV file path (explain this row instead of indexed sample)")
    p.add_argument("--out_json", default="./shap_explanation.json", help="Output JSON explanation path")
    p.add_argument("--out_plot", default=None, help="Optional PNG waterfall plot path")
    args = p.parse_args()
    main(args)
