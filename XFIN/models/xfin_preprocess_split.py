#!/usr/bin/env python3
"""
xfin_preprocess_split.py

Preprocess, impute, and split for ESG dataset.
This script now auto-generates the 'y_proxy' (baseline) values
by calculating the mean of non-imputed 'total_esg' scores for each sector.

Usage example:
python xfin_preprocess_split.py --input esg_india.csv --outdir ./processed --label_col total_esg --imputed_weight 0.4 --test_size 0.15 --val_size 0.15
"""

import os
import argparse
import json
from pathlib import Path
import fnmatch

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ---------- Helpers ----------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def detect_feature_columns(df, label_col):
    """
    Auto-detect numeric and categorical columns to use as features.
    Exclude obvious non-feature columns (symbol, yf_symbol, company, label, imputed, source, label_note).
    """
    exclude = {label_col, 'symbol', 'yf_symbol', 'company', 'source', 'imputed', 'label_note', 'sector_esg_proxy'}
    cand_cols = [c for c in df.columns if c not in exclude]
    numeric_cols = []
    categorical_cols = []
    for c in cand_cols:
        # treat booleans as numeric
        try:
            if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
                numeric_cols.append(c)
                continue
        except Exception:
            pass
        try:
            nunique = df[c].nunique(dropna=True)
        except Exception:
            nunique = 0
        if c.lower() == 'sector' or nunique < 200:
            categorical_cols.append(c)
        else:
            # if it looks numeric in strings, try converting a sample
            try:
                sample = df[c].dropna().astype(str).iloc[:50]
                _ = pd.to_numeric(sample)
                numeric_cols.append(c)
            except Exception:
                categorical_cols.append(c)
    return numeric_cols, categorical_cols

def find_candidate_esg_csvs(cwd):
    """
    Return list of candidate CSV filenames in cwd that look like ESG outputs
    """
    names = os.listdir(cwd)
    candidates = []
    for n in names:
        if not n.lower().endswith('.csv'):
            continue
        low = n.lower()
        if 'esg' in low or 'esg_india' in low or 'esg_output' in low or 'esg_india_output' in low:
            candidates.append(n)
    # fallback: any csv
    if not candidates:
        candidates = [n for n in names if n.lower().endswith('.csv')]
    return candidates

def load_csv(path, verbose=True):
    """
    Load CSV path with helpful fallbacks:
     - if exact file exists, use it
     - else search cwd for likely ESG CSVs and pick the first match (with warning)
     - else raise FileNotFoundError listing CSVs in cwd
    """
    # direct path
    if os.path.exists(path):
        if verbose:
            print(f"Loading CSV: {path}")
        return pd.read_csv(path)

    # try relative to cwd (maybe user passed a basename)
    cwd = os.getcwd()
    basename = os.path.basename(path)
    # try matching exact basename ignoring case
    for f in os.listdir(cwd):
        if f.lower() == basename.lower():
            candidate = os.path.join(cwd, f)
            if verbose:
                print(f"Exact filename not found but matched case-insensitively: {candidate}")
            return pd.read_csv(candidate)

    # search for ESG-like files
    candidates = find_candidate_esg_csvs(cwd)
    if candidates:
        picked = candidates[0]
        if verbose:
            print(f"Input file '{path}' not found. Falling back to first candidate CSV in cwd: '{picked}'")
        return pd.read_csv(os.path.join(cwd, picked))

    # nothing found — raise friendly error listing CSVs present
    csvs = [f for f in os.listdir(cwd) if f.lower().endswith('.csv')]
    msg = f"Input CSV '{path}' not found in path. CSVs in current directory ({cwd}):\n"
    if csvs:
        msg += "\n".join(f"  - {c}" for c in csvs)
    else:
        msg += "  (no CSV files found in current directory)\n"
    msg += "\nPlease provide the correct --input path."
    raise FileNotFoundError(msg)

def sklearn_supports_sparse_output():
    # return True if sklearn version >= 1.2 (sparse_output kwarg present)
    try:
        ver = sklearn.__version__.split('.')
        major = int(ver[0])
        minor = int(ver[1]) if len(ver) > 1 else 0
        return (major > 1) or (major == 1 and minor >= 2)
    except Exception:
        return False

# ---------- Main preprocessing function ----------
def preprocess_and_split(
    input_csv,
    out_dir,
    label_col='total_esg',
    # --- We no longer need 'proxy_col' as an argument ---
    drop_imputed=False,
    imputed_weight=0.5,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
    verbose=True
):
    ensure_dir(out_dir)
    df = load_csv(input_csv, verbose=verbose)
    n_total = len(df)
    if verbose:
        print(f"Loaded {n_total} rows from CSV (using input='{input_csv}')")

    # Check label column
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV. Columns: {list(df.columns)}")
    if 'sector' not in df.columns:
        raise ValueError(f"Required column 'sector' not found in CSV. Columns: {list(df.columns)}")

    # Ensure 'imputed' exists (if missing create False)
    if 'imputed' not in df.columns:
        df['imputed'] = False

    # Basic diagnostics
    n_missing_label = df[label_col].isna().sum()
    n_imputed_flags = int(df['imputed'].sum())
    if verbose:
        print(f"Label column '{label_col}': missing values = {n_missing_label}")
        print(f"Imputed label flags: {n_imputed_flags} rows marked imputed (if collector set this flag)")

    df = df.copy()

    # Optionally drop rows with missing labels entirely
    if drop_imputed:
        before = len(df)
        df = df[~df['imputed'].astype(bool)]
        df = df[df[label_col].notna()]
        if verbose:
            print(f"Dropped imputed rows & missing labels: {before - len(df)} rows removed. Remaining: {len(df)}")
    else:
        # keep rows; for missing label rows, we cannot train on them — so drop rows where label is NaN
        before = len(df)
        df = df[df[label_col].notna()]
        if verbose:
            print(f"Dropped rows with missing label (NaN): {before - len(df)} removed. Remaining: {len(df)}")

    # Auto-detect features
    numeric_cols, categorical_cols = detect_feature_columns(df, label_col=label_col)
    
    if verbose:
        print(f"Detected {len(numeric_cols)} numeric cols and {len(categorical_cols)} categorical cols.")
        print("Numeric columns:", numeric_cols)
        print("Categorical columns:", categorical_cols)

    # Extract y and sample weights
    y = df[label_col].astype(float).values
    sample_weight = np.ones(len(df), dtype=float)
    if not drop_imputed and 'imputed' in df.columns:
        imputed_mask = df['imputed'].astype(bool).values
        sample_weight[imputed_mask] = float(imputed_weight)
        if verbose:
            print(f"Assigned sample weight {imputed_weight} to {imputed_mask.sum()} imputed rows")

    # --- NEW LOGIC: Generate y_proxy (baseline) on the fly ---
    if verbose:
        print("Generating 'y_proxy' baseline from non-imputed sector averages...")
    
    # 1. Get true (non-imputed) ESG scores
    true_esg_df = df[df['imputed'] != True][['sector', label_col]]
    
    # 2. Calculate mean ESG per sector from true scores
    sector_proxy_map = true_esg_df.groupby('sector')[label_col].mean().to_dict()
    
    # 3. Get a global mean from true scores as a fallback
    global_true_mean = true_esg_df[label_col].mean()
    
    if verbose:
        print(f"   Calculated {len(sector_proxy_map)} sector averages.")
        print(f"   Global mean fallback (for sectors with no true data): {global_true_mean:.3f}")

    # 4. Create the y_proxy array by mapping sectors to the proxy map
    #    Fill sectors with no true data with the global_true_mean
    y_proxy = df['sector'].map(sector_proxy_map).fillna(global_true_mean).values
    
    n_proxy_missing = pd.Series(y_proxy).isna().sum()
    if n_proxy_missing > 0:
         if verbose:
            print(f"   ⚠️ Warning: {n_proxy_missing} proxy values are still NaN (this can happen if no true data exists at all). Filling with 0.")
         y_proxy = np.nan_to_num(y_proxy, nan=0.0) # Final fallback
    # -------------------------------------------------------------

    # -------------------------
    # SANITIZE numeric columns
    # -------------------------
    if numeric_cols:
        if verbose:
            print("Sanitizing numeric columns (coerce -> numeric, replace inf -> NaN, detect extreme values)...")
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                df[col] = np.nan
                if verbose:
                    print(f"   ⚠️ Could not coerce column '{col}' to numeric; filled with NaN")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        HUGE_ABS_THRESHOLD = 1e18
        huge_flags = (df[numeric_cols].abs() > HUGE_ABS_THRESHOLD)
        if huge_flags.any().any():
            cols_with_huge = [c for c in numeric_cols if huge_flags[c].any()]
            if verbose:
                print(f"   ⚠️ Extreme numeric values detected in columns: {cols_with_huge}")
                for c in cols_with_huge:
                    sample_idxs = df.index[huge_flags[c]].tolist()[:5]
                    print(f"      Column '{c}' has {huge_flags[c].sum()} extreme entries, sample indices: {sample_idxs}")
            for c in cols_with_huge:
                df.loc[huge_flags[c], c] = np.nan
        # create marketCap_log
        if 'marketCap' in numeric_cols:
            try:
                df['marketCap_log'] = df['marketCap'].where(df['marketCap'] > 0, np.nan)
                df['marketCap_log'] = np.log1p(df['marketCap_log'])
                if verbose:
                    print("   ℹ️ Created 'marketCap_log' feature (log1p of positive marketCap).")
                if 'marketCap_log' not in numeric_cols:
                    numeric_cols.append('marketCap_log')
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ marketCap log transform failed: {e}")

    # Build transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    onehot_kwargs = {'handle_unknown': 'ignore'}
    if sklearn_supports_sparse_output():
        onehot_kwargs['sparse_output'] = False
    else:
        onehot_kwargs['sparse'] = False

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='__UNKNOWN__')),
        ('onehot', OneHotEncoder(**onehot_kwargs))
    ])

    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No feature columns detected. Ensure your CSV contains usable numeric/categorical columns.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)

    # Validate split sizes
    rest_size = test_size + val_size
    if rest_size <= 0 or rest_size >= 1.0:
        raise ValueError("test_size + val_size must be between 0 and 1")

    X_full = df[numeric_cols + categorical_cols] if (numeric_cols + categorical_cols) else pd.DataFrame(index=df.index)

    # --- MODIFICATION: Add y_proxy to splits ---
    # initial split
    X_train_df, X_rest_df, y_train, y_rest, sw_train, sw_rest, y_proxy_train, y_proxy_rest = train_test_split(
        X_full, y, sample_weight, y_proxy, test_size=rest_size, random_state=random_state
    )

    # split rest -> val/test
    val_proportion_of_rest = val_size / rest_size
    X_val_df, X_test_df, y_val, y_test, sw_val, sw_test, y_proxy_val, y_proxy_test = train_test_split(
        X_rest_df, y_rest, sw_rest, y_proxy_rest, test_size=(1.0 - val_proportion_of_rest), random_state=random_state
    )
    # -------------------------------------------

    if verbose:
        print("Split sizes:")
        print(f"  Train: {len(X_train_df)}, Val: {len(X_val_df)}, Test: {len(X_test_df)}")

    # Fit preprocessor and transform
    preprocessor.fit(X_train_df)
    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    # feature names
    feature_names = []
    feature_names.extend(numeric_cols)
    if categorical_cols:
        cat_pipe = None
        try:
            cat_pipe = preprocessor.named_transformers_.get('cat')
        except Exception:
            cat_pipe = None
        if cat_pipe is not None:
            try:
                onehot = cat_pipe.named_steps['onehot']
                # --- FIX: Handle get_feature_names_out vs categories_ ---
                if hasattr(onehot, 'get_feature_names_out'):
                    feature_names.extend(onehot.get_feature_names_out(categorical_cols))
                else:
                    # Fallback for older sklearn
                    for col, cats in zip(categorical_cols, onehot.categories_):
                        for c in cats:
                            feature_names.append(f"{col}__{c}")
                # -----------------------------------------------------
            except Exception:
                # Fallback if categories/names fail
                feature_names.extend(categorical_cols)

    # Save artifacts
    base_out = Path(out_dir)
    ensure_dir(base_out)
    
    # --- MODIFICATION: Add y_proxy to all npz files ---
    np.savez_compressed(base_out / 'train.npz', 
                        X=X_train, 
                        y=y_train, 
                        sample_weight=sw_train, 
                        y_proxy=y_proxy_train)
                        
    np.savez_compressed(base_out / 'val.npz', 
                        X=X_val, 
                        y=y_val, 
                        sample_weight=sw_val,
                        y_proxy=y_proxy_val)
                        
    np.savez_compressed(base_out / 'test.npz', 
                        X=X_test, 
                        y=y_test, 
                        sample_weight=sw_test,
                        y_proxy=y_proxy_test)
    # --------------------------------------------------

    X_train_df.to_csv(base_out / 'X_train_df.csv', index=False)
    X_val_df.to_csv(base_out / 'X_val_df.csv', index=False)
    X_test_df.to_csv(base_out / 'X_test_df.csv', index=False)

    joblib.dump(preprocessor, base_out / 'preprocessor.joblib')

    # Create raw_input_columns for prediction (before one-hot encoding)
    raw_input_columns = numeric_cols + categorical_cols
    
    meta = {
        'raw_input_columns': raw_input_columns,  # For RobustMLPredictor
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'feature_names': feature_names,
        'label_col': label_col,
        # 'proxy_col' is no longer read, it's generated
        'drop_imputed': bool(drop_imputed),
        'imputed_weight': float(imputed_weight),
        'n_total_loaded': int(n_total),
        'n_after_filter': int(len(df))
    }
    with open(base_out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    def print_stats(y_array, name):
        # --- FIX for RuntimeWarning ---
        # Handle empty or all-NaN arrays gracefully
        if y_array is None or y_array.size == 0 or np.all(np.isnan(y_array)):
            print(f"  {name}: n={len(y_array) if y_array is not None else 0} (No valid data to calculate stats)")
            return
        # ------------------------------
        print(f"  {name}: n={len(y_array)} mean={np.nanmean(y_array):.3f} std={np.nanstd(y_array):.3f} min={np.nanmin(y_array):.3f} max={np.nanmax(y_array):.3f}")

    print("\nLabel statistics (total_esg):")
    print_stats(y, "All used rows")
    print_stats(y_train, "Train")
    print_stats(y_val, "Val")
    print_stats(y_test, "Test")
    
    # --- ADDITION: Print proxy stats ---
    print("\nProxy Label statistics (Generated Sector Averages):")
    print_stats(y_proxy, "All used rows")
    print_stats(y_proxy_train, "Train")
    print_stats(y_proxy_val, "Val")
    print_stats(y_proxy_test, "Test")
    # ---------------------------------

    print(f"\nSaved preprocessed data and artifacts to {base_out.resolve()}")

    return {
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'X_train_shape': X_train.shape,
        'X_val_shape': X_val.shape,
        'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape,
        'y_val_shape': y_val.shape,
        'y_test_shape': y_test.shape,
        'metadata': meta
    }

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ESG CSV and split into train/val/test.")
    parser.add_argument("--input", "-i", help="Input CSV path (collector output)", required=True)
    parser.add_argument("--outdir", "-o", help="Output directory for processed artifacts", default="./processed")
    parser.add_argument("--label_col", help="Label column name (default: total_esg)", default="total_esg")
    # --- REMOVED --proxy_col argument ---
    parser.add_argument("--drop_imputed", help="If true, drop rows flagged as imputed", action="store_true")
    parser.add_argument("--imputed_weight", help="Sample weight to assign to imputed labels if not dropped (0-1)", type=float, default=0.5)
    parser.add_argument("--test_size", help="Test set fraction (default 0.15)", type=float, default=0.15)
    parser.add_argument("--val_size", help="Validation set fraction (default 0.15)", type=float, default=0.15)
    parser.add_argument("--random_state", help="Random state for splitting", type=int, default=42)
    parser.add_argument("--no_verbose", help="Run quietly", action="store_true")

    args = parser.parse_args()
    
    # --- FIX: Handle OneHotEncoder typo ---
    # This was a typo in the script I sent you. Correcting it here.
    try:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='__UNKNOWN__')),
            ('onehot', OneHotEncoder(**onehot_kwargs))
        ])
    except NameError as e:
        # This is to fix the typo from the previous script
        if 'onehot_kwargs' in str(e):
            onehot_kwargs = {'handle_unknown': 'ignore'}
            if sklearn_supports_sparse_output():
                onehot_kwargs['sparse_output'] = False
            else:
                onehot_kwargs['sparse'] = False
        else:
            raise e
    # -------------------------------------


    out = preprocess_and_split(
        input_csv=args.input,
        out_dir=args.outdir,
        label_col=args.label_col,
        # --- 'proxy_col' argument removed ---
        drop_imputed=args.drop_imputed,
        imputed_weight=args.imputed_weight,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        verbose=(not args.no_verbose)
    )