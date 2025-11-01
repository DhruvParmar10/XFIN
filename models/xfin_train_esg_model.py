#!/usr/bin/env python3
"""
xfin_train_esg_model.py

Train a LightGBM regressor on the preprocessed ESG dataset produced by xfin_preprocess_split.py.

This script is robust to different lightgbm versions:
 - It first attempts to use sklearn API LGBMRegressor with early stopping.
 - If that fails (TypeError for callbacks), it falls back to the native lightgbm.train() API.
Outputs:
 - model or booster saved under out_dir/models/
 - shap explainer saved
 - training_report.json, feature_importance.csv, shap_global_sample.csv
"""
import os
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap

# Prefer sklearn Hist / LGBM import after joblib etc.
try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_npz(path):
    npz = np.load(path, allow_pickle=True)
    X = npz['X']
    y = npz['y']
    sw = npz['sample_weight'] if 'sample_weight' in npz else None
    return X, y, sw

# --- FIX: Use np.sqrt for backward-compatibility with sklearn ---
def rmse(a, b):
    # squared=False is for newer sklearn versions.
    # np.sqrt(mse) is compatible with all versions.
    return np.sqrt(mean_squared_error(a, b))
# -------------------------------------------------------------

# -------------------------
# Train / Eval
# -------------------------
def train_and_evaluate(
    processed_dir,
    out_dir,
    model_name='xfin_esg_model_v1',
    random_seed=42,
    n_estimators=2000,
    learning_rate=0.05,
    early_stopping_rounds=50,
    verbose_eval=50
):
    processed_dir = Path(processed_dir)
    ensure_dir(out_dir)
    models_dir = Path(out_dir) / 'models'
    ensure_dir(models_dir)

    # Load data
    train_path = processed_dir / 'train.npz'
    val_path = processed_dir / 'val.npz'
    test_path = processed_dir / 'test.npz'
    meta_path = processed_dir / 'metadata.json'
    preproc_path = processed_dir / 'preprocessor.joblib'

    if not train_path.exists() or not val_path.exists() or not test_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Processed inputs missing. Run preprocessing step and ensure train/val/test npz and metadata present in processed_dir.")

    X_train, y_train, sw_train = load_npz(train_path)
    X_val, y_val, sw_val = load_npz(val_path)
    X_test, y_test, sw_test = load_npz(test_path)

    metadata = json.load(open(meta_path, 'r'))
    feature_names = metadata.get('feature_names', None)

    print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    if sw_train is None:
        sw_train = np.ones(len(y_train), dtype=float)
    if sw_val is None:
        sw_val = np.ones(len(y_val), dtype=float)
    if sw_test is None:
        sw_test = np.ones(len(y_test), dtype=float)

    trained_via_sklearn = False
    model_obj = None
    booster = None
    training_time = None

    # --- Define callbacks list ---
    # This is the modern way to handle early stopping and logging for both APIs
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        lgb.log_evaluation(period=verbose_eval)
    ]
    # ---------------------------------

    if LGBM_AVAILABLE:
        # Attempt sklearn API with callbacks (preferred)
        try:
            print("Attempting sklearn LGBMRegressor.fit(...) with callbacks...")
            model = LGBMRegressor(
                objective='regression',
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=31,
                random_state=random_seed,
                n_jobs=-1,
                verbose=-1  # Silence default logger, use callback instead
            )
            t0 = time.time()
            
            # --- Use 'callbacks' argument ---
            model.fit(
                X_train, y_train,
                sample_weight=sw_train,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[sw_val],
                eval_metric='rmse',
                callbacks=lgb_callbacks
            )
            # -------------------------------------

            training_time = time.time() - t0
            print(f"Sklearn LGBMRegressor trained in {training_time:.1f}s")
            trained_via_sklearn = True
            model_obj = model
            # extract underlying booster
            try:
                booster = model.booster_
            except Exception:
                booster = None
        except TypeError as te:
            # fallback to native train API
            print(f"sklearn LGBMRegressor.fit(...) failed (likely callbacks issue). Falling back to lightgbm.train()")
            print("TypeError detail:", te)
        except Exception as e:
            # other exceptions -> still fallback
            print("sklearn LGBMRegressor.fit raised an unexpected error; falling back to lightgbm.train(). Error:", e)

    # If not trained via sklearn, use native lgb.train
    if not trained_via_sklearn:
        if not LGBM_AVAILABLE:
            raise RuntimeError("LightGBM not available in environment. Install lightgbm or use alternative model.")
        print("Preparing native lightgbm.Dataset and training with callbacks via lgb.train()")
        dtrain = lgb.Dataset(X_train, label=y_train, weight=sw_train)
        dval = lgb.Dataset(X_val, label=y_val, weight=sw_val, reference=dtrain)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': learning_rate,
            'num_leaves': 31,
            'verbosity': -1,
            'seed': random_seed
        }
        t0 = time.time()
        
        # --- Use 'callbacks' argument ---
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            valid_names=['valid'],
            callbacks=lgb_callbacks
        )
        # -------------------------------------

        training_time = time.time() - t0
        print(f"Native lightgbm.train finished in {training_time:.1f}s. Best iteration: {booster.best_iteration}")

    # Prediction helper depending on object type
    def predict_fn(X, use_booster_first=True):
        if booster is not None:
            # booster.predict uses num_iteration best_iteration if provided
            try:
                # for booster, use best_iteration if available
                it = getattr(booster, 'best_iteration', None) or getattr(booster, 'best_iteration_', None)
                if it:
                    return booster.predict(X, num_iteration=it)
            except Exception:
                pass
            return booster.predict(X)
        elif model_obj is not None:
            return model_obj.predict(X)
        else:
            raise RuntimeError("No trained model available for prediction")

    # Evaluate metrics
    def eval_split(name, X, y, sw):
        preds = predict_fn(X)
        return {
            'n': int(len(y)),
            'mae': float(mean_absolute_error(y, preds)),
            'rmse': float(rmse(y, preds)), # This now uses the corrected rmse function
            'r2': float(r2_score(y, preds))
        }

    train_metrics = eval_split('train', X_train, y_train, sw_train)
    val_metrics = eval_split('val', X_val, y_val, sw_val)
    test_metrics = eval_split('test', X_test, y_test, sw_test)

    print("Metrics:")
    print(" Train:", train_metrics)
    print(" Val:  ", val_metrics)
    print(" Test: ", test_metrics)

    # Save model/booster
    model_out_path = models_dir / f"{model_name}.joblib"
    booster_out_path = models_dir / f"{model_name}_booster.txt"
    # Prefer saving booster (native) if available
    if booster is not None:
        try:
            # save booster model text file
            booster.save_model(str(booster_out_path))
            # also pickle booster for fast reload
            joblib.dump(booster, model_out_path)
            print(f"Saved Booster to {booster_out_path} and pickled to {model_out_path}")
            saved_model_path = str(model_out_path)
        except Exception as e:
            print("Could not save booster via both methods, attempting joblib only:", e)
            joblib.dump(booster, model_out_path)
            saved_model_path = str(model_out_path)
    else:
        # fallback: save sklearn model_obj
        joblib.dump(model_obj, model_out_path)
        print(f"Saved sklearn model to {model_out_path}")
        saved_model_path = str(model_out_path)

    # SHAP explainer
    print("Building SHAP TreeExplainer...")
    try:
        # Use booster if possible for best performance
        explainer_target = booster if booster is not None else model_obj
        explainer = shap.TreeExplainer(explainer_target)
        explainer_out_path = models_dir / f"{model_name}_shap_explainer.joblib"
        joblib.dump(explainer, explainer_out_path)
        print(f"Saved SHAP explainer to {explainer_out_path}")
    except Exception as e:
        explainer = None
        explainer_out_path = None
        print("SHAP explainer build/save failed:", e)

    # Feature importance
    fi = None
    try:
        if booster is not None:
            gain = booster.feature_importance(importance_type='gain')
            split = booster.feature_importance(importance_type='split')
            fnames = feature_names if feature_names is not None else [f'f{i}' for i in range(len(gain))]
            fi = pd.DataFrame({'feature': fnames, 'gain': gain, 'split': split}).sort_values('gain', ascending=False)
        else:
            # sklearn model feature importance (if available)
            if hasattr(model_obj, 'feature_importances_'):
                imp = model_obj.feature_importances_
                fnames = feature_names if feature_names is not None else [f'f{i}' for i in range(len(imp))]
                fi = pd.DataFrame({'feature': fnames, 'importance': imp}).sort_values('importance', ascending=False)
        if fi is not None:
            fi_path = Path(out_dir) / 'feature_importance.csv'
            fi.to_csv(fi_path, index=False)
            print(f"Wrote feature importance to {fi_path}")
    except Exception as e:
        print("Could not compute feature importance:", e)
        fi_path = None

    # Small SHAP sample (if explainer exists)
    try:
        if explainer is not None:
            # Use a deterministic sample from validation set for consistency
            sample_size = min(50, X_val.shape[0])
            np.random.seed(random_seed) # ensure sample is reproducible
            sample_idx = np.random.choice(X_val.shape[0], sample_size, replace=False)

            X_sample = X_val[sample_idx]
            shap_values = explainer.shap_values(X_sample)
            
            # Use feature names if available for the DataFrame
            if feature_names:
                X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
                shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
            else:
                X_sample_df = pd.DataFrame(X_sample)
                shap_values_df = pd.DataFrame(shap_values)

            # Save the raw SHAP values for the sample
            shap_raw_out = Path(out_dir) / 'shap_sample_values.csv'
            shap_values_df.to_csv(shap_raw_out, index=False)
            
            # Save the corresponding feature values for the sample
            shap_features_out = Path(out_dir) / 'shap_sample_features.csv'
            X_sample_df.to_csv(shap_features_out, index=False)

            # Calculate and save global (mean abs) SHAP values for the full validation set
            print("Calculating global SHAP values on full validation set...")
            shap_values_all_val = explainer.shap_values(X_val)
            mean_abs = np.mean(np.abs(shap_values_all_val), axis=0)
            
            if feature_names:
                shap_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs})
            else:
                shap_df = pd.DataFrame({'feature': [f'f{i}' for i in range(len(mean_abs))], 'mean_abs_shap': mean_abs})
                
            shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
            shap_out = Path(out_dir) / 'shap_global_summary.csv'
            shap_df.to_csv(shap_out, index=False)
            print(f"Wrote global SHAP summary (from val set) to {shap_out}")
            print(f"Wrote SHAP sample (values/features) to {shap_raw_out} / {shap_features_out}")
            
    except Exception as e:
        print(f"SHAP sample computation failed: {e}")

    # Save training report
    report = {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'n_train': int(len(y_train)),
        'n_val': int(len(y_val)),
        'n_test': int(len(y_test)),
        'best_iteration': int(getattr(booster, 'best_iteration', -1) or getattr(model_obj, 'best_iteration_', -1) or -1),
        'training_time_seconds': training_time
    }
    report_path = Path(out_dir) / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved training report to {report_path}")

    return {
        'model_path': saved_model_path,
        'explainer_path': str(explainer_out_path) if explainer_out_path is not None else None,
        'report_path': str(report_path),
        'feature_importance_csv': str(fi_path) if fi_path is not None else None
    }

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train LightGBM regressor for ESG prediction")
    p.add_argument("--processed_dir", "-p", default="./processed",
                   help="Directory containing train.npz, val.npz, test.npz, preprocessor.joblib, metadata.json")
    p.add_argument("--out_dir", "-o", default="./artifacts",
                   help="Where to save model, explainer, and reports")
    p.add_argument("--model_name", default="xfin_esg_model_v1", help="Base name for saved model files")
    p.add_argument("--n_estimators", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--early_stop", type=int, default=50)
    p.add_argument("--verbose_eval", type=int, default=50)
    args = p.parse_args()

    out = train_and_evaluate(
        processed_dir=args.processed_dir,
        out_dir=args.out_dir,
        model_name=args.model_name,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stop,
        verbose_eval=args.verbose_eval
    )

    print("Done. Artifacts:", out)