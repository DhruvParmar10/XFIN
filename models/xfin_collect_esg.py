#!/usr/bin/env python3
"""
xfin_collect_esg.py

Discover Indian tickers (NSE), fetch ESG labels & basic features via yfinance (and Finnhub fallback),
impute missing ESG using sector-proxy, and save results to CSV.

Outputs columns: symbol, yf_symbol, company, total_esg, environment_score, social_score, governance_score,
source, imputed (bool), marketCap, trailingPE, beta, vol_30d, ret_1m, sector, label_note
"""

import os
import time
import argparse
import math
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from dateutil.parser import parse as dateparse

# Optional import - used only if web CSV fetch fails
try:
    from nsetools import Nse
    NSETOOLS_AVAILABLE = True
except Exception:
    NSETOOLS_AVAILABLE = False

# ------------------------------
# Config
# ------------------------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", None)
# Rate-limit pause between yfinance / finnhub calls to be polite (seconds)
SLEEP_SECONDS = 0.8

# Sector proxy baseline (used to impute missing ESG)
SECTOR_ESG_PROXY = {
    'IT Services': {'E': 70, 'S': 65, 'G': 70},
    'Banking': {'E': 55, 'S': 60, 'G': 65},
    'FMCG': {'E': 60, 'S': 65, 'G': 65},
    'Pharmaceuticals': {'E': 50, 'S': 55, 'G': 60},
    'Telecommunications': {'E': 55, 'S': 60, 'G': 60},
    'Automobiles': {'E': 45, 'S': 50, 'G': 55},
    'Oil & Gas': {'E': 35, 'S': 45, 'G': 55},
    'Power': {'E': 40, 'S': 50, 'G': 55},
    'Metals & Mining': {'E': 35, 'S': 45, 'G': 50},
    'Chemicals': {'E': 40, 'S': 50, 'G': 55},
    'Cement': {'E': 40, 'S': 50, 'G': 55},
    'Real Estate': {'E': 45, 'S': 50, 'G': 55},
    'Infrastructure': {'E': 45, 'S': 50, 'G': 55},
    'Textiles': {'E': 45, 'S': 50, 'G': 50},
    'Financial Services': {'E': 50, 'S': 55, 'G': 60},
    'Retail': {'E': 50, 'S': 55, 'G': 55},
    'Media & Entertainment': {'E': 55, 'S': 60, 'G': 60},
    'Other': {'E': 50, 'S': 50, 'G': 50}
}

# ------------------------------
# Utilities
# ------------------------------
def download_nse_equity_list_csv():
    """
    Try to download NSE's equity list CSV from known endpoints.
    Returns pandas DataFrame or None on failure.
    """
    urls = [
        "https://www1.nseindia.com/content/equities/EQUITY_L.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://www.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; XFINBot/1.0; +https://example.com)"
    }

    for u in urls:
        try:
            print(f"Trying to download NSE CSV from: {u}")
            resp = requests.get(u, headers=headers, timeout=15)
            if resp.status_code == 200 and resp.text.strip():
                from io import StringIO
                df = pd.read_csv(StringIO(resp.text))
                print(f"Downloaded {len(df)} rows from {u}")
                return df
            else:
                print(f"  → HTTP {resp.status_code} or empty response")
        except Exception as e:
            print(f"  → error fetching {u}: {e}")
    return None


def get_nse_tickers_via_nsetools():
    """
    If nsetools available, use it to list securities.
    Returns list of symbols (strings).
    """
    if not NSETOOLS_AVAILABLE:
        return None
    try:
        nse = Nse()
        codes = nse.get_stock_codes()  # dict: symbol->name
        # remove header row key 'SYMBOL'
        codes = {k: v for k, v in codes.items() if k != 'SYMBOL'}
        print(f"nsetools returned {len(codes)} symbols")
        return list(codes.keys())
    except Exception as e:
        print("nsetools failed:", e)
        return None


def build_indian_ticker_list(limit=None):
    """
    Return a list of yfinance-ready ticker symbols to try (format: RELIANCE.NS).
    Steps:
      1. Try downloading NSE EQUITY csv
      2. Fallback to nsetools if available
      3. else return a small hardcoded sample (developer fallback)
    """
    df = download_nse_equity_list_csv()
    tickers = []
    if df is not None:
        # In EQUITY_L.csv there is typically a column 'SYMBOL' or 'Symbol'
        col = None
        for candidate in ['SYMBOL', 'Symbol', 'symbol']:
            if candidate in df.columns:
                col = candidate
                break
        if not col:
            # attempt heuristics
            possible = [c for c in df.columns if 'symbol' in c.lower()]
            col = possible[0] if possible else None

        if col:
            symbols = df[col].astype(str).str.strip().unique().tolist()
            tickers = [s + ".NS" for s in symbols if s and len(s) <= 10]
            print(f"Found {len(tickers)} tickers from NSE CSV")
        else:
            print("Could not find SYMBOL column in NSE CSV; falling back")
            tickers = []

    if not tickers:
        from_nsetools = get_nse_tickers_via_nsetools()
        if from_nsetools:
            tickers = [s + ".NS" for s in from_nsetools]
    
    if not tickers:
        # fallback sample list (developer convenience)
        print("Falling back to built-in sample list (small). Install nsetools or allow CSV download for full list.")
        tickers = [
            "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","LT.NS","SBIN.NS",
            "BHARTIARTL.NS","HINDUNILVR.NS","ITC.NS","ONGC.NS","AXISBANK.NS"
        ]
    if limit:
        return tickers[:limit]
    return tickers


# ------------------------------
# ESG fetch helpers
# ------------------------------
def extract_esg_from_yfinance(symbol):
    """
    Attempt to extract ESG values from yfinance Ticker.sustainability DataFrame.
    Returns dict or None.
    """
    try:
        t = yf.Ticker(symbol)
        s = t.sustainability
        if s is None or s.empty:
            return None
        # s is a DataFrame with index as metric names and single column 'Value'
        # common index keys seen: 'environmentScore', 'socialScore', 'governanceScore', 'totalEsg', 'esgPerformance'
        out = {}
        # index can be strings; guard presence
        for key in ['totalEsg', 'environmentScore', 'socialScore', 'governanceScore', 'esgPerformance', 'percentile']:
            try:
                if key in s.index:
                    val = s.loc[key].values[0]
                    out[key] = val if (not pd.isna(val)) else None
                else:
                    out[key] = None
            except Exception:
                out[key] = None
        # map to standard key names
        if any(out.get(k) is not None for k in ['totalEsg','environmentScore','socialScore','governanceScore']):
            return {
                'total_esg': float(out.get('totalEsg')) if out.get('totalEsg') is not None else None,
                'environment_score': float(out.get('environmentScore')) if out.get('environmentScore') is not None else None,
                'social_score': float(out.get('socialScore')) if out.get('socialScore') is not None else None,
                'governance_score': float(out.get('governanceScore')) if out.get('governanceScore') is not None else None,
                'esg_performance': out.get('esgPerformance'),
                'percentile': out.get('percentile'),
                'source': 'yahoo'
            }
        return None
    except Exception as e:
        # yfinance sometimes raises odd errors; just return None
        # print("yfinance ESG error for", symbol, e)
        return None


def extract_esg_from_finnhub(symbol):
    """
    Query Finnhub stock ESG endpoint.
    Returns dict or None.
    Uses FINNHUB_API_KEY from env.
    """
    if not FINNHUB_API_KEY:
        return None
    url = f"https://finnhub.io/api/v1/stock/esg?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        # Finnhub's returned keys can vary across accounts; check typical keys
        # We'll try a few possibilities and map them
        # Example possible keys: 'esgScore', 'environmentScore', 'socialScore', 'governanceScore'
        # Map to our schema if possible:
        mapped = {}
        # prioritise exact esgScore key if present
        if 'esgScore' in data:
            mapped['total_esg'] = float(data.get('esgScore') or 0.0)
        elif 'totalEsg' in data:
            mapped['total_esg'] = float(data.get('totalEsg') or 0.0)
        else:
            mapped['total_esg'] = None
        mapped['environment_score'] = float(data.get('environmentScore')) if data.get('environmentScore') else None
        mapped['social_score'] = float(data.get('socialScore')) if data.get('socialScore') else None
        mapped['governance_score'] = float(data.get('governanceScore')) if data.get('governanceScore') else None
        # if at least one useful field present return
        if any(v is not None for v in mapped.values()):
            mapped['source'] = 'finnhub'
            return mapped
        return None
    except Exception:
        return None


# ------------------------------
# Feature extraction from yfinance
# ------------------------------
def extract_basic_features(symbol):
    """
    Use yfinance to extract basic numeric & categorical features for a symbol.
    Returns dict (features may contain NaN).
    """
    out = {
        'symbol': symbol,
        'company': None,
        'marketCap': None,
        'trailingPE': None,
        'beta': None,
        'vol_30d': None,
        'ret_1m': None,
        'sector': None
    }
    try:
        t = yf.Ticker(symbol)
        info = t.info if hasattr(t, 'info') else {}
        out['company'] = info.get('longName') or info.get('shortName') or None
        out['marketCap'] = info.get('marketCap', None)
        out['trailingPE'] = info.get('trailingPE', None)
        out['beta'] = info.get('beta', None)
        out['sector'] = info.get('sector', None) or info.get('industry', None) or None

        # historical returns & volatility
        hist = t.history(period='1y', interval='1d', auto_adjust=False)
        if hist is not None and not hist.empty:
            hist = hist.dropna(subset=['Close'])
            hist['ret'] = hist['Close'].pct_change()
            # 30-day vol (annualized not necessary)
            if len(hist) >= 30:
                out['vol_30d'] = float(hist['ret'].rolling(window=30).std().iloc[-1])
            else:
                out['vol_30d'] = float(hist['ret'].std())
            if len(hist) >= 22:
                out['ret_1m'] = float(hist['Close'].iloc[-1] / hist['Close'].iloc[-22] - 1)
            else:
                out['ret_1m'] = float(hist['ret'].tail(22).sum()) if len(hist) > 1 else None
        else:
            out['vol_30d'] = None
            out['ret_1m'] = None

    except Exception as e:
        # being defensive; don't crash the batch
        # print("yfinance feature error", symbol, e)
        pass

    return out


# ------------------------------
# Sector proxy imputation
# ------------------------------
def impute_esg_by_sector(sector, market_cap=None):
    """
    Simple sector-proxy imputation with optional market cap adjustment.
    Returns dict: total_esg, environment_score, social_score, governance_score, source='sector_proxy'
    """
    base = SECTOR_ESG_PROXY.get(sector, SECTOR_ESG_PROXY['Other'])
    env = base['E']; soc = base['S']; gov = base['G']
    adjustment = 0
    if market_cap:
        # market_cap is in native currency units (INR). Apply simple adjustments.
        if market_cap > 50_000_000_000:
            adjustment = 10
        elif market_cap < 5_000_000_000:
            adjustment = -10
    env = max(0, min(100, env + adjustment))
    soc = max(0, min(100, soc + adjustment))
    gov = max(0, min(100, gov + adjustment))
    total = round((env + soc + gov) / 3.0, 1)
    return {
        'total_esg': total,
        'environment_score': env,
        'social_score': soc,
        'governance_score': gov,
        'source': 'sector_proxy'
    }


# ------------------------------
# Main orchestration
# ------------------------------
def collect(esg_out_csv: str, limit: int = None, verbosity: int = 1):
    tickers = build_indian_ticker_list(limit)
    print(f"Total tickers to process: {len(tickers)} (limit={limit})")
    rows = []
    n = len(tickers)
    for idx, yf_symbol in enumerate(tickers, start=1):
        try:
            if verbosity:
                print(f"[{idx}/{n}] Processing {yf_symbol} ...", end=" ")

            # 1) get basic features
            feats = extract_basic_features(yf_symbol)
            company = feats.get('company') or yf_symbol

            # 2) try yfinance ESG
            esg = extract_esg_from_yfinance(yf_symbol)
            source = None; imputed = False

            if esg:
                source = esg.get('source', 'yahoo')
                total_esg = esg.get('total_esg')
                env = esg.get('environment_score'); soc = esg.get('social_score'); gov = esg.get('governance_score')
                label_note = 'yahoo'
            else:
                # 3) try finnhub (if configured)
                esg = extract_esg_from_finnhub(yf_symbol)
                if esg:
                    source = esg.get('source', 'finnhub')
                    total_esg = esg.get('total_esg')
                    env = esg.get('environment_score'); soc = esg.get('social_score'); gov = esg.get('governance_score')
                    label_note = 'finnhub'
                else:
                    # 4) impute from sector proxy
                    sector_name = feats.get('sector') or 'Other'
                    proxy = impute_esg_by_sector(sector_name, market_cap=feats.get('marketCap'))
                    source = proxy['source']
                    total_esg = proxy['total_esg']; env = proxy['environment_score']; soc = proxy['social_score']; gov = proxy['governance_score']
                    imputed = True
                    label_note = f"imputed_from_sector({sector_name})"

            # 5) record row
            row = {
                'symbol': yf_symbol.replace('.NS','.NS'),  # keep canonical .NS
                'yf_symbol': yf_symbol,
                'company': company,
                'total_esg': total_esg,
                'environment_score': env,
                'social_score': soc,
                'governance_score': gov,
                'source': source,
                'imputed': bool(imputed),
                'marketCap': feats.get('marketCap'),
                'trailingPE': feats.get('trailingPE'),
                'beta': feats.get('beta'),
                'vol_30d': feats.get('vol_30d'),
                'ret_1m': feats.get('ret_1m'),
                'sector': feats.get('sector'),
                'label_note': label_note
            }
            rows.append(row)

            if verbosity:
                print(f"ESG source={source} imputed={imputed} total_esg={total_esg}")

            # polite pause
            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print(f"\nError processing {yf_symbol}: {e}")
            # keep going
            continue

    df_out = pd.DataFrame(rows)
    # ensure column order
    cols = [
        'symbol','yf_symbol','company','sector','marketCap','trailingPE','beta','vol_30d','ret_1m',
        'total_esg','environment_score','social_score','governance_score','source','imputed','label_note'
    ]
    cols = [c for c in cols if c in df_out.columns] + [c for c in df_out.columns if c not in cols]
    df_out = df_out[cols]
    df_out.to_csv(esg_out_csv, index=False)
    print(f"\nSaved output to {esg_out_csv} : {len(df_out)} rows")
    return df_out


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Indian tickers' ESG & basic features (yfinance + finnhub fallback)")
    parser.add_argument("--out", "-o", help="Output CSV path", default="esg_india_output.csv")
    parser.add_argument("--limit", "-n", help="Limit number of tickers (for testing)", type=int, default=None)
    parser.add_argument("--verbose", "-v", help="Verbosity (0 quiet, 1 default)", type=int, default=1)
    args = parser.parse_args()
    collect(esg_out_csv=args.out, limit=args.limit, verbosity=args.verbose)
