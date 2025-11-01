"""
Generate sector_feature_defaults.json for robust ML fallback.
This computes per-sector median values for all numeric raw features.
"""

import pandas as pd
import json
from pathlib import Path

def generate_sector_defaults():
    print("="*70)
    print("GENERATING SECTOR FEATURE DEFAULTS")
    print("="*70)
    
    # Load training data
    data_path = Path(__file__).parent / 'esg_india.csv'
    df = pd.read_csv(data_path)
    
    print(f"\n✅ Loaded {len(df)} records from esg_india.csv")
    print(f"   Sectors: {df['sector'].nunique()} unique")
    
    # Define raw numeric features (inputs to model, including ESG components)
    raw_numeric_features = [
        'marketCap', 'trailingPE', 'beta', 'vol_30d', 'ret_1m',
        'environment_score', 'social_score', 'governance_score'
    ]
    
    # Compute per-sector medians
    sector_defaults = {}
    
    for sector in df['sector'].unique():
        sector_df = df[df['sector'] == sector]
        defaults = {}
        
        for col in raw_numeric_features:
            if col in sector_df.columns:
                # Use median (robust to outliers)
                median_val = sector_df[col].median()
                defaults[col] = float(median_val) if pd.notna(median_val) else None
        
        # Add sector name itself
        defaults['sector'] = sector
        
        sector_defaults[sector] = defaults
        print(f"   {sector}: {len(sector_df)} stocks, medians computed")
    
    # Add global 'Other' fallback (median across ALL stocks)
    global_defaults = {}
    for col in raw_numeric_features:
        if col in df.columns:
            median_val = df[col].median()
            global_defaults[col] = float(median_val) if pd.notna(median_val) else None
    global_defaults['sector'] = 'Other'
    sector_defaults['Other'] = global_defaults
    
    print(f"\n✅ Computed defaults for {len(sector_defaults)} sectors + 'Other' global")
    
    # Save to processed/sector_feature_defaults.json
    output_path = Path(__file__).parent / 'processed' / 'sector_feature_defaults.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(sector_defaults, f, indent=2)
    
    print(f"✅ Saved to: {output_path}")
    
    # Show sample
    print(f"\n📊 Sample defaults for 'Technology' sector:")
    if 'Technology' in sector_defaults:
        for k, v in sector_defaults['Technology'].items():
            print(f"   {k}: {v}")
    
    print("\n" + "="*70)
    print("✅ SECTOR DEFAULTS GENERATION COMPLETE")
    print("="*70)
    
    return sector_defaults

if __name__ == '__main__':
    generate_sector_defaults()
