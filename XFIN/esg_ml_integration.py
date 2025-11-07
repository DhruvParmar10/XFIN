"""
ESG ML Model Integration Module
================================

Integrates the trained LightGBM ESG prediction model with SHAP explainability
into the XFIN portfolio ESG scoring system.

Features:
- ML-based ESG prediction for stocks missing API data  
- SHAP-based feature importance and explanations
- Portfolio-level aggregated SHAP waterfall
- Seamless fallback chain: API → ML Model → Sector Proxy
- Robust multi-layer feature fetching with imputation tracking

Author: XFIN Development Team
Date: November 2025
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
import warnings

# Import robust predictor
from .robust_ml_predictor import RobustMLPredictor

# Optional imports with graceful degradation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available. Install with: pip install yfinance")


class ESGMLPredictor:
    """
    ML-based ESG predictor using trained LightGBM model with SHAP explainability.
    
    Fallback chain: API → ML Model → Sector Proxy
    """
    
    def __init__(self, models_dir: Optional[str] = None, verbose: bool = True):
        """
        Initialize ML predictor
        
        Parameters:
        -----------
        models_dir : str, optional
            Path to models directory (default: XFIN/models/)
        verbose : bool
            Print loading information
        """
        self.verbose = verbose
        
        # Set models directory
        if models_dir is None:
            # Default: XFIN/models/
            xfin_dir = Path(__file__).parent
            models_dir = xfin_dir / 'models'
        else:
            models_dir = Path(models_dir)
        
        self.models_dir = models_dir
        self.processed_dir = models_dir / 'processed'
        self.artifacts_dir = models_dir / 'artifacts'
        self.model_name = 'xfin_esg_model_v1'
        
        # Initialize robust predictor (handles all fallbacks)
        try:
            self.robust_predictor = RobustMLPredictor(
                models_dir=str(models_dir),
                verbose=verbose
            )
            self.model_available = True
        except Exception as e:
            if verbose:
                print(f"⚠️ Could not initialize robust predictor: {e}")
                print("   Will fall back to sector proxy")
            self.robust_predictor = None
            self.model_available = False
        
        # Check if models exist (legacy check)
        if self.model_available:
            self.model_available = self._check_model_availability()
        
        if self.model_available:
            self._load_model_artifacts()
        elif verbose:
            print("⚠️ ML model not available - will use sector proxy fallback")
    
    def _check_model_availability(self) -> bool:
        """Check if required model files exist"""
        required_files = [
            self.processed_dir / 'preprocessor.joblib',
            self.processed_dir / 'metadata.json',
        ]
        
        # Need at least one model file
        model_exists = (
            (self.artifacts_dir / 'models' / f'{self.model_name}.joblib').exists() or
            (self.artifacts_dir / 'models' / f'{self.model_name}_booster.txt').exists()
        )
        
        all_exist = all(f.exists() for f in required_files) and model_exists
        
        if not all_exist and self.verbose:
            missing = [str(f) for f in required_files if not f.exists()]
            if not model_exists:
                missing.append("Model file (joblib or booster.txt)")
            print(f"⚠️ Missing model files: {missing}")
        
        return all_exist
    
    def _load_model_artifacts(self):
        """Load preprocessor, model, and SHAP explainer"""
        try:
            # Load preprocessor
            self.preprocessor = joblib.load(self.processed_dir / 'preprocessor.joblib')
            
            # Load metadata
            with open(self.processed_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            self.feature_names = metadata.get('feature_names', None)
            
            # Load model (try joblib first, then booster)
            models_path = self.artifacts_dir / 'models'
            model_joblib = models_path / f'{self.model_name}.joblib'
            model_booster = models_path / f'{self.model_name}_booster.txt'
            
            self.model = None
            self.booster = None
            
            if model_joblib.exists():
                self.model = joblib.load(model_joblib)
                if self.verbose:
                    print(f"✅ Loaded ML model: {model_joblib.name}")
            elif model_booster.exists() and LGBM_AVAILABLE:
                self.booster = lgb.Booster(model_file=str(model_booster))
                if self.verbose:
                    print(f"✅ Loaded booster: {model_booster.name}")
            else:
                raise FileNotFoundError("No model file found")
            
            # Load or create SHAP explainer
            self.explainer = None
            if SHAP_AVAILABLE:
                explainer_paths = list(models_path.glob(f"{self.model_name}*_shap_explainer.*"))
                if explainer_paths:
                    try:
                        self.explainer = joblib.load(explainer_paths[0])
                        if self.verbose:
                            print(f"✅ Loaded SHAP explainer: {explainer_paths[0].name}")
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠️ Failed to load saved explainer: {e}")
                        self.explainer = None
                
                # Create explainer if not loaded
                if self.explainer is None:
                    target = self.booster if self.booster is not None else self.model
                    if target is not None:
                        try:
                            self.explainer = shap.TreeExplainer(target)
                            if self.verbose:
                                print("✅ Created new SHAP TreeExplainer")
                        except Exception as e:
                            if self.verbose:
                                print(f"⚠️ Could not create SHAP explainer: {e}")
            
            # Runtime validation: Verify preprocessor/model pairing
            if self.verbose:
                print("\n🔍 Validating preprocessor/model compatibility...")
            self._validate_preprocessor_model_pairing()
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to load model artifacts: {e}")
            raise
    
    def _validate_preprocessor_model_pairing(self):
        """
        Verify that preprocessor output shape matches model input shape.
        This catches issues where preprocessor and model were trained on different data.
        """
        try:
            # Get expected feature count from metadata
            expected_features = len(self.feature_names) if self.feature_names else None
            
            # Get model input shape
            model_n_features = None
            if self.model is not None and hasattr(self.model, 'n_features_in_'):
                model_n_features = self.model.n_features_in_
            elif self.booster is not None and hasattr(self.booster, 'num_feature'):
                model_n_features = self.booster.num_feature()
            
            # Validate
            if expected_features and model_n_features:
                if expected_features != model_n_features:
                    error_msg = (
                        f"⚠️ MISMATCH: Preprocessor outputs {expected_features} features "
                        f"but model expects {model_n_features} features!\n"
                        f"   → Preprocessor and model were trained on different data.\n"
                        f"   → Please retrain the model with matching preprocessor."
                    )
                    if self.verbose:
                        print(error_msg)
                    raise ValueError(error_msg)
                else:
                    if self.verbose:
                        print(f"   ✅ Feature count matches: {expected_features} features")
            
            # Check feature names if available
            if self.feature_names and self.verbose:
                print(f"   ✅ Feature names loaded: {len(self.feature_names)} features")
                print(f"      Sample: {self.feature_names[:5]}")
                
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Validation check failed: {e}")
            # Don't raise - validation is informational
    
    def predict_esg(
        self, 
        ticker: str = None, 
        stock_name: str = None, 
        sector: str = None
    ) -> Dict[str, Any]:
        """
        Predict ESG score using robust ML predictor
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        stock_name : str
            Company name
        sector : str
            Sector classification
        
        Returns:
        --------
        dict with prediction results and metadata
        """
        if not self.model_available or self.robust_predictor is None:
            return None
        
        return self.robust_predictor.predict_esg(
            ticker=ticker,
            stock_name=stock_name,
            sector=sector
        )
    
    def fetch_features_for_ticker(self, ticker: str, stock_name: str = None, 
                                   sector: str = None) -> Optional[Dict[str, Any]]:
        """
        Fetch raw features required for ML model prediction
        
        Parameters:
        -----------
        ticker : str
            Stock ticker (e.g., 'RELIANCE.NS')
        stock_name : str, optional
            Company name
        sector : str, optional
            Sector classification
        
        Returns:
        --------
        dict : Raw features or None if fetch fails
        """
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            # Fetch from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get historical data for volatility/returns
            hist = stock.history(period='1mo')
            
            # Calculate features
            features = {}
            
            # Sector
            features['sector'] = sector or info.get('sector', 'Other')
            
            # Market cap
            features['marketCap'] = info.get('marketCap', 0)
            
            # Trailing PE
            features['trailingPE'] = info.get('trailingPE', info.get('forwardPE', np.nan))
            
            # Beta
            features['beta'] = info.get('beta', np.nan)
            
            # 30-day volatility
            if len(hist) > 1:
                returns = hist['Close'].pct_change().dropna()
                features['vol_30d'] = float(returns.std()) if len(returns) > 0 else np.nan
            else:
                features['vol_30d'] = np.nan
            
            # 1-month return
            if len(hist) > 1:
                features['ret_1m'] = float((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1)
            else:
                features['ret_1m'] = np.nan
            
            # ESG component scores (if available from API - will be None if missing)
            features['environment_score'] = info.get('environmentScore', np.nan)
            features['social_score'] = info.get('socialScore', np.nan)
            features['governance_score'] = info.get('governanceScore', np.nan)
            
            return features
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Failed to fetch features for {ticker}: {e}")
            return None
    
    def predict_esg(self, ticker: str, stock_name: str = None, 
                    sector: str = None, max_imputed_ratio: float = 0.45) -> Optional[Dict[str, Any]]:
        """
        Predict ESG score using ML model with robust fallbacks
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        stock_name : str, optional
            Company name
        sector : str, optional
            Sector classification
        max_imputed_ratio : float
            Maximum allowed imputation ratio (default: 0.45 = 45%)
        
        Returns:
        --------
        dict : Prediction result with metadata or None if prediction fails
        """
        if not self.model_available or self.robust_predictor is None:
            return None
        
        # Use robust predictor (handles all fallbacks internally)
        result = self.robust_predictor.predict_esg(
            ticker=ticker,
            stock_name=stock_name,
            sector=sector,
            max_imputed_ratio=max_imputed_ratio
        )
        
        if result is None or result.get('predicted_esg') is None:
            return None
        
        # Add SHAP explanation if available
        if self.explainer is not None and SHAP_AVAILABLE:
            try:
                # Get raw features
                raw_features = result.get('raw_features', {})
                if raw_features:
                    # Create DataFrame
                    raw_cols = self.robust_predictor.artifacts['raw_input_columns']
                    row = {c: raw_features.get(c, np.nan) for c in raw_cols}
                    df = pd.DataFrame([row], columns=raw_cols)
                    
                    # Add computed features (marketCap_log) - CRITICAL for preprocessor
                    if 'marketCap' in df.columns and 'marketCap_log' not in df.columns:
                        df['marketCap_log'] = df['marketCap'].where(df['marketCap'] > 0, np.nan)
                        df['marketCap_log'] = np.log1p(df['marketCap_log'])
                    
                    # Transform
                    X_transformed = self.preprocessor.transform(df)
                    X_flat = np.asarray(X_transformed).reshape(-1)
                    
                    # Get SHAP values
                    shap_vals = self.explainer.shap_values(X_transformed)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[0]
                    shap_vals = np.asarray(shap_vals).reshape(-1)
                    
                    # Get expected value
                    expected_value = float(
                        self.explainer.expected_value if not isinstance(
                            self.explainer.expected_value, np.ndarray
                        ) else self.explainer.expected_value[0]
                    )
                    
                    # Build feature contributions
                    feature_contributions = []
                    for fname, fval, sval in zip(self.feature_names, X_flat, shap_vals):
                        feature_contributions.append({
                            'feature': fname,
                            'value': float(fval) if np.isfinite(fval) else None,
                            'shap': float(sval),
                            'abs_shap': float(abs(sval))
                        })
                    
                    # Sort by absolute SHAP value
                    feature_contributions = sorted(
                        feature_contributions,
                        key=lambda x: x['abs_shap'],
                        reverse=True
                    )
                    
                    # Add to result
                    result['expected_value'] = expected_value
                    result['feature_contributions'] = feature_contributions
                    
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ SHAP calculation failed: {e}")
        
        # Set data_source field
        result['data_source'] = 'ml_model'
        result['model_version'] = self.model_name
        
        return result
    
    def aggregate_portfolio_shap(self, holdings_explained: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate SHAP values across portfolio holdings (weighted by portfolio value)
        
        Parameters:
        -----------
        holdings_explained : list
            List of holdings with ML explanations and weights
        
        Returns:
        --------
        dict : Portfolio-level aggregated SHAP analysis
        """
        if not SHAP_AVAILABLE or not self.feature_names:
            return {'error': 'SHAP not available or feature names missing'}
        
        n_features = len(self.feature_names)
        agg_shap = np.zeros(n_features, dtype=float)
        base_values = []
        total_weight_with_ml = 0.0
        holdings_with_ml = []
        
        # Aggregate SHAP contributions weighted by portfolio allocation
        for holding in holdings_explained:
            weight_pct = holding.get('weight', 0.0)  # This is stored as percentage (0-100)
            weight = weight_pct / 100.0  # Convert to decimal (0-1)
            ml_explanation = holding.get('ml_explanation')
            
            if ml_explanation is None or ml_explanation.get('feature_contributions') is None:
                continue
            
            holdings_with_ml.append(holding['stock_name'])
            total_weight_with_ml += weight
            
            # Get SHAP values aligned to feature_names
            feature_contribs = ml_explanation['feature_contributions']
            shap_dict = {fc['feature']: fc['shap'] for fc in feature_contribs}
            
            # Build aligned SHAP vector
            shap_vector = np.array([shap_dict.get(fn, 0.0) for fn in self.feature_names])
            
            # Weight by portfolio allocation
            agg_shap += shap_vector * weight
            
            # Add weighted base value
            if ml_explanation.get('expected_value') is not None:
                base_values.append(ml_explanation['expected_value'] * weight)
        
        # Calculate portfolio metrics
        portfolio_base = float(np.sum(base_values)) if base_values else 0.0
        final_portfolio_prediction = portfolio_base + float(np.sum(agg_shap))
        
        # Build per-feature contributions
        feature_contributions = []
        for i, fname in enumerate(self.feature_names):
            feature_contributions.append({
                'feature': fname,
                'contribution': float(agg_shap[i]),
                'abs_contribution': float(abs(agg_shap[i]))
            })
        
        # Sort by absolute contribution
        feature_contributions_sorted = sorted(
            feature_contributions,
            key=lambda x: x['abs_contribution'],
            reverse=True
        )
        
        # Group features by category (E/S/G, Sector, Fundamentals)
        grouped_contributions = self._group_feature_contributions(feature_contributions)
        
        return {
            'portfolio_base': portfolio_base,
            'portfolio_prediction': final_portfolio_prediction,
            'total_shap_contribution': float(np.sum(agg_shap)),
            'feature_contributions': feature_contributions_sorted[:20],  # Top 20
            'grouped_contributions': grouped_contributions,
            'holdings_with_ml': holdings_with_ml,
            'coverage_weight': total_weight_with_ml,
            'n_features': n_features
        }
    
    def _group_feature_contributions(self, feature_contributions: List[Dict]) -> List[Dict]:
        """Group features into meaningful categories for UI display"""
        groups = {
            'ESG Components': 0.0,
            'Sector': 0.0,
            'Market Cap': 0.0,
            'Financial Metrics': 0.0,
            'Market Behavior': 0.0,
            'Other': 0.0
        }
        
        for fc in feature_contributions:
            fname = fc['feature'].lower()
            contrib = fc['contribution']
            
            if any(x in fname for x in ['environment', 'social', 'governance', 'esg']):
                groups['ESG Components'] += contrib
            elif 'sector' in fname:
                groups['Sector'] += contrib
            elif 'marketcap' in fname or 'market_cap' in fname:
                groups['Market Cap'] += contrib
            elif any(x in fname for x in ['pe', 'trailingpe', 'revenue', 'ebitda']):
                groups['Financial Metrics'] += contrib
            elif any(x in fname for x in ['beta', 'vol', 'ret', 'volatility', 'return']):
                groups['Market Behavior'] += contrib
            else:
                groups['Other'] += contrib
        
        # Convert to sorted list
        grouped = [{'group': k, 'contribution': v} for k, v in groups.items()]
        grouped = sorted(grouped, key=lambda x: abs(x['contribution']), reverse=True)
        
        return grouped


def create_esg_ml_predictor(models_dir: Optional[str] = None, verbose: bool = True) -> ESGMLPredictor:
    """
    Factory function to create ESG ML predictor
    
    Parameters:
    -----------
    models_dir : str, optional
        Path to models directory
    verbose : bool
        Print loading information
    
    Returns:
    --------
    ESGMLPredictor : ML predictor instance
    """
    return ESGMLPredictor(models_dir=models_dir, verbose=verbose)
