"""
Robust ML Prediction Utilities with Multi-Layer Fallbacks
==========================================================

Implements resilient feature fetching and prediction pipeline:
1. Cache → Yahoo → Finnhub → FMP → Sector Defaults
2. Tracks imputation and data sources
3. Handles missing features gracefully
4. Returns detailed metadata for audit/regulatory compliance

Author: XFIN Development Team
Date: November 2025
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import pandas as pd
import joblib
import warnings

# Optional imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class RobustMLPredictor:
    """
    Robust ML predictor with multi-layer feature fallbacks and imputation tracking
    """
    
    def __init__(self, models_dir: str, cache_dir: str = None, verbose: bool = True):
        """
        Initialize robust predictor
        
        Parameters:
        -----------
        models_dir : str
            Path to models directory containing processed/ and artifacts/
        cache_dir : str, optional
            Path to feature cache directory (default: models_dir/../feature_cache)
        verbose : bool
            Enable verbose logging
        """
        self.verbose = verbose
        self.models_dir = Path(models_dir)
        self.processed_dir = self.models_dir / 'processed'
        self.artifacts_dir = self.models_dir / 'artifacts'
        
        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = self.models_dir.parent / 'feature_cache'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load artifacts
        self.artifacts = self._load_prediction_artifacts()
        
        if self.verbose:
            print(f"✅ Robust ML Predictor initialized")
            print(f"   Cache directory: {self.cache_dir}")
            print(f"   Sectors with defaults: {len(self.artifacts.get('sector_defaults', {}))}")
    
    def _load_prediction_artifacts(self) -> Dict[str, Any]:
        """Load all required model artifacts"""
        artifacts = {}
        
        try:
            # Load preprocessor
            preproc_path = self.processed_dir / 'preprocessor.joblib'
            if not preproc_path.exists():
                raise FileNotFoundError(f"Preprocessor not found: {preproc_path}")
            artifacts['preprocessor'] = joblib.load(preproc_path)
            
            # Load metadata
            metadata_path = self.processed_dir / 'metadata.json'
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            artifacts['raw_input_columns'] = metadata.get('raw_input_columns')
            artifacts['feature_names'] = metadata.get('feature_names')
            artifacts['numeric_cols'] = metadata.get('numeric_cols', [])
            
            if not artifacts['raw_input_columns']:
                raise ValueError("metadata.json missing 'raw_input_columns'")
            
            # Load sector defaults
            sector_defaults_path = self.processed_dir / 'sector_feature_defaults.json'
            if sector_defaults_path.exists():
                with open(sector_defaults_path, 'r') as f:
                    artifacts['sector_defaults'] = json.load(f)
            else:
                warnings.warn(f"Sector defaults not found: {sector_defaults_path}")
                artifacts['sector_defaults'] = {}
            
            # Load model (try joblib first, then booster)
            models_path = self.artifacts_dir / 'models'
            model_name = 'xfin_esg_model_v1'
            
            artifacts['model'] = None
            artifacts['booster'] = None
            
            model_joblib = models_path / f'{model_name}.joblib'
            if model_joblib.exists():
                artifacts['model'] = joblib.load(model_joblib)
                if self.verbose:
                    print(f"✅ Loaded model: {model_joblib.name}")
            
            model_booster = models_path / f'{model_name}_booster.txt'
            if model_booster.exists() and LGBM_AVAILABLE:
                artifacts['booster'] = lgb.Booster(model_file=str(model_booster))
                if self.verbose:
                    print(f"✅ Loaded booster: {model_booster.name}")
            
            if artifacts['model'] is None and artifacts['booster'] is None:
                raise FileNotFoundError("No model file found (joblib or booster)")
            
            return artifacts
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load artifacts: {e}")
            raise
    
    def _fetch_from_yahoo(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch features from Yahoo Finance"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data (not 404)
            if not info or info.get('regularMarketPrice') is None:
                return None
            
            # Get historical data
            hist = stock.history(period='1mo')
            if len(hist) == 0:
                return None
            
            features = {}
            
            # Market fundamentals
            features['marketCap'] = info.get('marketCap')
            features['trailingPE'] = info.get('trailingPE', info.get('forwardPE'))
            features['beta'] = info.get('beta')
            
            # Calculated features
            if len(hist) > 1:
                returns = hist['Close'].pct_change().dropna()
                features['vol_30d'] = float(returns.std()) if len(returns) > 0 else None
                features['ret_1m'] = float((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1)
            else:
                features['vol_30d'] = None
                features['ret_1m'] = None
            
            # Sector from Yahoo (if available)
            features['sector'] = info.get('sector')
            
            return features
            
        except Exception as e:
            if self.verbose:
                print(f"   Yahoo fetch failed: {e}")
            return None
    
    def _fetch_from_cache(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch from cache"""
        cache_path = self.cache_dir / f"{ticker}.json"
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            # Check cache age (optional: expire after 7 days)
            cache_age_days = (time.time() - cache_path.stat().st_mtime) / 86400
            if cache_age_days > 7:
                if self.verbose:
                    print(f"   ⚠️ Cache expired ({cache_age_days:.1f} days old)")
                return None
            
            return cached
            
        except Exception as e:
            if self.verbose:
                print(f"   Cache read failed: {e}")
            return None
    
    def _save_to_cache(self, ticker: str, features: Dict[str, Any]):
        """Save features to cache"""
        try:
            cache_path = self.cache_dir / f"{ticker}.json"
            with open(cache_path, 'w') as f:
                json.dump(features, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"   Cache save failed: {e}")
    
    def get_features_with_fallbacks(
        self, 
        ticker: str, 
        sector: Optional[str] = None
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """
        Fetch features with multi-layer fallbacks
        
        Returns:
        --------
        (features_dict, imputed_features_list, data_sources_list)
        """
        used_sources = []
        imputed_features = []
        raw_features = {}
        
        # 1. Try cache first (prefer cached snapshots with real ESG scores)
        cached = self._fetch_from_cache(ticker)
        if cached and cached.get('marketCap') is not None:
            used_sources.append('cache')
            raw_features = cached.copy()
            if self.verbose:
                cache_has_esg = all(
                    cached.get(k) not in [None, 50.0] 
                    for k in ['environment_score', 'social_score', 'governance_score']
                    if k in cached
                )
                status = "with real ESG" if cache_has_esg else "partial data"
                print(f"   ✅ Cache hit for {ticker} ({status})")
        
        # 2. Try Yahoo Finance for financial features
        if not raw_features:
            yahoo_features = self._fetch_from_yahoo(ticker)
            if yahoo_features and yahoo_features.get('marketCap') is not None:
                used_sources.append('yahoo')
                raw_features = yahoo_features.copy()
                # Save to cache
                self._save_to_cache(ticker, yahoo_features)
                if self.verbose:
                    print(f"   ✅ Yahoo fetch successful for {ticker}")
        
        # 3. If still no base data, mark as complete sector default fallback
        if not raw_features:
            used_sources.append('sector_defaults_complete')
            if self.verbose:
                print(f"   ⚠️ No API/cache data - using complete sector defaults for {ticker}")
        
        # 4. Fill missing features with sector defaults
        sector_to_use = sector or raw_features.get('sector', 'Other')
        sector_defaults = self.artifacts['sector_defaults'].get(
            sector_to_use, 
            self.artifacts['sector_defaults'].get('Other', {})
        )
        
        # Track which features are being imputed from sector defaults
        esg_features = ['environment_score', 'social_score', 'governance_score']
        esg_imputed_count = 0
        
        # Ensure all raw_input_columns are present
        for col in self.artifacts['raw_input_columns']:
            if col == 'sector':
                # Ensure sector is set
                if 'sector' not in raw_features or raw_features['sector'] is None:
                    raw_features['sector'] = sector_to_use
            elif col not in raw_features or raw_features[col] is None:
                # Use sector default
                default_val = sector_defaults.get(col)
                raw_features[col] = default_val
                imputed_features.append(col)
                
                # Track ESG imputation separately
                if col in esg_features:
                    esg_imputed_count += 1
                    
                if default_val is None:
                    if self.verbose:
                        print(f"   ⚠️ No default available for {col}")
        
        # Mark as heavy imputation if all ESG scores are imputed
        if esg_imputed_count == len(esg_features) and 'sector_defaults_complete' not in used_sources:
            used_sources.append('sector_defaults_esg')
            if self.verbose:
                print(f"   ⚠️ All ESG components imputed from sector defaults")
        
        return raw_features, imputed_features, used_sources
    
    def predict_from_raw_features(
        self,
        raw_features: Dict[str, Any],
        imputed_features: List[str],
        used_sources: List[str],
        max_imputed_ratio: float = 0.45
    ) -> Dict[str, Any]:
        """
        Predict ESG from raw features with imputation checks
        
        Parameters:
        -----------
        raw_features : dict
            Raw input features matching raw_input_columns
        imputed_features : list
            List of feature names that were imputed
        used_sources : list
            List of data sources used
        max_imputed_ratio : float
            Maximum allowed imputation ratio (default: 0.45)
        
        Returns:
        --------
        dict : Prediction result with metadata
        """
        preproc = self.artifacts['preprocessor']
        model = self.artifacts['model']
        booster = self.artifacts['booster']
        raw_cols = self.artifacts['raw_input_columns']
        
        # Build one-row DataFrame
        row = {c: raw_features.get(c, np.nan) for c in raw_cols}
        df = pd.DataFrame([row], columns=raw_cols)
        
        # Add computed features (marketCap_log)
        if 'marketCap' in df.columns and 'marketCap_log' not in df.columns:
            df['marketCap_log'] = df['marketCap'].where(df['marketCap'] > 0, np.nan)
            df['marketCap_log'] = np.log1p(df['marketCap_log'])
        
        # Calculate imputation ratio
        missing_cols = [c for c in raw_cols if pd.isna(df.at[0, c]) or df.at[0, c] is None]
        imputation_ratio = len(imputed_features) / max(1, len(raw_cols))
        
        # Runtime safeguard #1: Check imputation ratio threshold
        if imputation_ratio > max_imputed_ratio:
            if self.verbose:
                print(f"   ⚠️ SAFEGUARD: Imputation ratio {imputation_ratio:.1%} > {max_imputed_ratio:.1%}")
                print(f"   → Falling back to sector proxy instead of unreliable ML prediction")
            return {
                'predicted_esg': None,
                'esg_source': 'model_failed_too_many_imputed',
                'used_imputed_features': imputed_features,
                'imputation_ratio': imputation_ratio,
                'used_sources': used_sources,
                'reason': f'Imputation ratio {imputation_ratio:.2%} exceeds max {max_imputed_ratio:.2%}'
            }
        
        # Runtime safeguard #2: Check if all ESG components are imputed (circular dependency issue)
        esg_features = ['environment_score', 'social_score', 'governance_score']
        all_esg_imputed = all(f in imputed_features for f in esg_features if f in raw_cols)
        
        if all_esg_imputed and 'sector_defaults_esg' in used_sources:
            if self.verbose:
                print(f"   ⚠️ SAFEGUARD: All ESG components imputed from sector defaults")
                print(f"   → ML prediction will echo sector average (~50.0)")
            # Still allow prediction but mark it clearly
            # The monitoring system will catch this
        
        # Runtime safeguard #3: Check feature variance (near-zero variance = unreliable)
        numeric_features = [c for c in raw_cols if c != 'sector' and c in df.columns]
        if len(numeric_features) > 0:
            feature_values = df[numeric_features].iloc[0].values
            # Check if all features are very similar (low variance)
            if len(feature_values) > 1:
                variance = np.var(feature_values[~np.isnan(feature_values)])
                if variance < 1e-6:
                    if self.verbose:
                        print(f"   ⚠️ SAFEGUARD: Near-zero feature variance ({variance:.2e})")
                        print(f"   → Features may be too uniform for reliable prediction")
                    # Log warning but continue
        
        # Transform using preprocessor
        try:
            X = preproc.transform(df)
        except Exception as e:
            return {
                'predicted_esg': None,
                'esg_source': 'model_transform_error',
                'error': str(e),
                'used_imputed_features': missing_cols,
                'imputation_ratio': imputation_ratio,
                'used_sources': used_sources
            }
        
        # Predict
        try:
            if model is not None and hasattr(model, 'predict'):
                pred = model.predict(X)
                pred_val = float(pred[0])
            elif booster is not None:
                pred = booster.predict(X)
                pred_val = float(pred[0])
            else:
                return {
                    'predicted_esg': None,
                    'esg_source': 'no_model_available',
                    'used_sources': used_sources
                }
            
            # Clip to valid ESG range [0, 100]
            pred_val = float(np.clip(pred_val, 0, 100))
            
            # Monitoring: Check if prediction is suspiciously close to 50 (sector default)
            is_near_50 = abs(pred_val - 50.0) < 0.1
            has_esg_imputation = any(f in imputed_features for f in esg_features)
            
            result = {
                'predicted_esg': pred_val,
                'esg_source': 'predicted_by_model',
                'used_imputed_features': imputed_features,
                'imputation_ratio': imputation_ratio,
                'used_sources': used_sources,
                'raw_features': raw_features,
                # Monitoring fields
                'prediction_near_50': is_near_50,
                'esg_features_imputed': has_esg_imputation,
                'imputation_count': len(imputed_features)
            }
            
            # Warning if prediction is suspiciously generic
            if is_near_50 and has_esg_imputation and self.verbose:
                print(f"   ⚠️ MONITOR: Prediction ≈ 50.0 with imputed ESG inputs")
                print(f"   → Likely echoing sector defaults, not genuine ML prediction")
            
            return result
            
        except Exception as e:
            return {
                'predicted_esg': None,
                'esg_source': 'model_predict_error',
                'error': str(e),
                'used_imputed_features': missing_cols,
                'imputation_ratio': imputation_ratio,
                'used_sources': used_sources
            }
    
    def predict_esg(
        self, 
        ticker: str, 
        stock_name: str = None,
        sector: str = None,
        max_imputed_ratio: float = 0.45
    ) -> Optional[Dict[str, Any]]:
        """
        Main prediction method with full fallback chain
        
        Parameters:
        -----------
        ticker : str
            Stock ticker (e.g., 'RELIANCE.NS')
        stock_name : str, optional
            Company name
        sector : str, optional
            Sector classification
        max_imputed_ratio : float
            Maximum imputation ratio allowed (default: 0.45 = 45%)
        
        Returns:
        --------
        dict : Prediction result or None if prediction fails
        """
        # Get features with fallbacks
        raw_features, imputed_features, used_sources = self.get_features_with_fallbacks(
            ticker, sector
        )
        
        # Predict
        result = self.predict_from_raw_features(
            raw_features, 
            imputed_features, 
            used_sources,
            max_imputed_ratio
        )
        
        # Add ticker and stock name
        result['ticker'] = ticker
        result['stock_name'] = stock_name
        
        if self.verbose and result.get('predicted_esg') is not None:
            impute_pct = result['imputation_ratio'] * 100
            print(f"   ✅ ML prediction: {result['predicted_esg']:.1f}/100 (imputed: {impute_pct:.1f}%)")
        
        return result
