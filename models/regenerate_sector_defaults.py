#!/usr/bin/env python3
"""
regenerate_sector_defaults.py

Reads esg_india.csv (collector output) and generates processed/sector_feature_defaults.json
The script computes medians for numeric columns (including E, S, G if present).
Run:
  python regenerate_sector_defaults.py --input esg_india.csv --outdir ./processed
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def infer_numeric_cols(df):
    # choose a conservative default set if present
    candidates = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object and df[c].str.replace('.','',1).str.isnumeric().all():
            candidates.append(c)
    return candidates

def main(args):
    inp = Path(args.input)
    outdir = Path(args.outdir)
    if not inp.exists():
        print("Input CSV not found:", inp)
        return
    df = pd.read_csv(inp)
    if 'sector' not in df.columns:
        print("Warning: 'sector' column not found. Using 'Other' only.")
        df['sector'] = 'Other'
    # numeric columns we care about (common)
    prefer = ['marketCap','trailingPE','beta','revenue','environment_score','social_score','governance_score','total_esg']
    numeric = [c for c in prefer if c in df.columns]
    # fallback: auto-detect
    if not numeric:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # compute medians per sector
    grouped = df.groupby('sector')[numeric].median(numeric_only=True)
    out = {}
    for sector, row in grouped.iterrows():
        d = {}
        for col in numeric:
            v = row.get(col, np.nan)
            d[col] = None if pd.isna(v) else float(v)
        out[sector if pd.notna(sector) else 'Other'] = d
    # global median fallback
    global_med = df[numeric].median(numeric_only=True).to_dict()
    out['Other'] = {k:(None if pd.isna(v) else float(v)) for k,v in global_med.items()}
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'sector_feature_defaults.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("Wrote sector defaults:", out_path)
    print("Sample sectors:", list(out.keys())[:10])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="esg_india.csv")
    p.add_argument("--outdir", default="./processed")
    args = p.parse_args()
    main(args)
