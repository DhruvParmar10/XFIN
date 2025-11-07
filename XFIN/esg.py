"""
ESG Scoring Module for XFIN

Provides Environmental, Social, and Governance (ESG) scoring capabilities
for individual securities and portfolios with AI-powered explanations.

Follows SEBI BRSR framework suitable for Indian/Asian markets.
Implements 5-star rating system with sector-based proxy fallback.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from .utils import get_llm_explanation

# Import from consolidated data_utils
from .data_utils import (
    extract_holding_from_row,
    detect_date_price_columns,
    find_column_case_insensitive,
    safe_float_conversion,
    create_diagnostics_report,
    get_sector as get_sector_unified,
    get_ticker
)

# Try new multi-source fetcher first, fallback to old one
try:
    from .esg_data_sources_multi import MultiSourceESGFetcher
    USE_MULTI_SOURCE = True
except ImportError:
    from .esg_data_sources import ESGDataFetcher
    USE_MULTI_SOURCE = False

# Try to import ML integration
try:
    from .esg_ml_integration import ESGMLPredictor
    ML_MODEL_AVAILABLE = True
except ImportError:
    ML_MODEL_AVAILABLE = False


class ESGScoringEngine:
    """
    ESG (Environmental, Social, Governance) scoring engine for portfolio analysis.
    
    Features:
    - 5-star rating system (⭐ to ⭐⭐⭐⭐⭐)
    - Sector-specific weighting
    - Proxy fallback for missing data
    - Integration with real ESG data sources
    - SEBI BRSR framework compliance
    """
    
    def __init__(self, api_key: Optional[str] = None, scoring_methodology: str = "sector_adjusted"):
        """
        Initialize ESG Scoring Engine
        
        Parameters:
        -----------
        api_key : str, optional
            OpenRouter API key for AI-powered explanations
        scoring_methodology : str, default "sector_adjusted"
            Scoring methodology: "weighted", "equal", "sector_adjusted"
        """
        self.api_key = api_key
        self.scoring_methodology = scoring_methodology
        
        # Initialize data fetcher (multi-source or single-source)
        if USE_MULTI_SOURCE:
            self.data_fetcher = MultiSourceESGFetcher()
            print("✅ Using multi-source ESG fetcher (Finnhub + Yahoo + RapidAPI + FMP)")
        else:
            self.data_fetcher = ESGDataFetcher()
            print("⚠️ Using single-source ESG fetcher (Yahoo only)")
        
        # Initialize ML predictor for enhanced ESG estimation
        self.ml_predictor = None
        if ML_MODEL_AVAILABLE:
            try:
                self.ml_predictor = ESGMLPredictor(verbose=False)
                if self.ml_predictor.model_available:
                    print("✅ ML ESG predictor loaded (LightGBM + SHAP)")
                else:
                    self.ml_predictor = None
            except Exception as e:
                print(f"⚠️ ML predictor initialization failed: {e}")
                self.ml_predictor = None
        
        # Load sector proxy database
        self.sector_proxy_df = self._load_sector_proxy_data()
        
        # Default ESG weights (overridden by sector-specific weights)
        self.default_esg_weights = {
            'environmental': 0.40,
            'social': 0.30, 
            'governance': 0.30
        }
        
        # Sector-specific weights (following BRSR framework)
        self.sector_weights = {
            'Energy': {'environmental': 0.50, 'social': 0.25, 'governance': 0.25},
            'Oil & Gas': {'environmental': 0.50, 'social': 0.25, 'governance': 0.25},
            'Manufacturing': {'environmental': 0.50, 'social': 0.25, 'governance': 0.25},
            'Automobiles': {'environmental': 0.50, 'social': 0.25, 'governance': 0.25},
            'Power': {'environmental': 0.50, 'social': 0.25, 'governance': 0.25},
            'Metals & Mining': {'environmental': 0.50, 'social': 0.25, 'governance': 0.25},
            'Chemicals': {'environmental': 0.50, 'social': 0.25, 'governance': 0.25},
            
            'Banking': {'environmental': 0.20, 'social': 0.30, 'governance': 0.50},
            'Financial Services': {'environmental': 0.20, 'social': 0.30, 'governance': 0.50},
            
            'IT Services': {'environmental': 0.30, 'social': 0.40, 'governance': 0.30},
            'Technology': {'environmental': 0.30, 'social': 0.40, 'governance': 0.30},
            'Telecom': {'environmental': 0.30, 'social': 0.40, 'governance': 0.30},
        }
        
        # Industry-specific ESG risk factors
        self.industry_risk_mapping = self._initialize_industry_mapping()
    
    def _load_sector_proxy_data(self) -> pd.DataFrame:
        """Load sector proxy ESG scores"""
        proxy_path = Path(__file__).parent / 'data' / 'sector_esg_proxy.csv'
        if proxy_path.exists():
            return pd.read_csv(proxy_path)
        return pd.DataFrame()  # Empty if not found
    
    def fetch_esg_data(self, ticker: str, company_name: str, isin: Optional[str] = None) -> Optional[Dict]:
        """
        Fetch ESG data using data fetcher with fallback
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        company_name : str
            Company name
        isin : str, optional
            ISIN code
        
        Returns:
        --------
        dict or None
            ESG data if available
        """
        return self.data_fetcher.fetch_esg_data(ticker, company_name, isin)
    
    def calculate_esg_score(self, esg_data: Optional[Dict], sector: str, market_cap: float) -> Dict:
        """
        Calculate ESG score with 5-star rating
        
        Parameters:
        -----------
        esg_data : dict or None
            Raw ESG data from data sources (or None to use proxy)
        sector : str
            Company sector
        market_cap : float
            Market capitalization
        
        Returns:
        --------
        dict
            ESG scores, star rating, and metadata
        """
        if esg_data is None:
            # Use sector proxy
            return self.apply_sector_proxy(sector, market_cap)
        
        # Extract scores (already converted to 0-100 scale by data fetcher)
        env_score = esg_data.get('environmental_score', 0)
        social_score = esg_data.get('social_score', 0)
        gov_score = esg_data.get('governance_score', 0)
        
        # Get sector-specific weights
        weights = self.sector_weights.get(sector, self.default_esg_weights)
        
        # Calculate overall ESG score (0-100) using our sector-specific weights
        # Don't use Yahoo's totalEsg as it's a combined risk score with different methodology
        overall_esg = (
            env_score * weights['environmental'] +
            social_score * weights['social'] +
            gov_score * weights['governance']
        )
        
        # Convert to 5-star rating
        star_rating = self._score_to_stars(overall_esg)
        
        return {
            'environmental_score': round(env_score, 2),
            'social_score': round(social_score, 2),
            'governance_score': round(gov_score, 2),
            'overall_esg_score': round(overall_esg, 2),
            'star_rating': star_rating,
            'star_rating_text': '⭐' * star_rating,
            'rating_label': self._get_rating_label(star_rating),
            'data_source': esg_data.get('data_source', 'Unknown'),
            'is_proxy': False,
            'sector': sector,
            'weights_used': weights
        }
    
    def apply_sector_proxy(self, sector: str, market_cap: float) -> Dict:
        """
        Apply sector-based proxy ESG scores when actual data unavailable
        Uses intelligent sector-based estimates when no real data exists
        
        Parameters:
        -----------
        sector : str
            Company sector
        market_cap : float
            Market capitalization
        
        Returns:
        --------
        dict
            Proxy ESG scores with metadata
        """
        # Sector-based default ESG scores (0-100 scale)
        # Based on typical ESG performance by sector in Indian markets
        sector_defaults = {
            'Banking': {'E': 55, 'S': 60, 'G': 65},
            'Financial Services': {'E': 55, 'S': 60, 'G': 65},
            'IT Services': {'E': 70, 'S': 65, 'G': 70},
            'Technology': {'E': 70, 'S': 65, 'G': 70},
            'Pharmaceuticals': {'E': 50, 'S': 55, 'G': 60},
            'Pharma': {'E': 50, 'S': 55, 'G': 60},
            'Oil & Gas': {'E': 35, 'S': 45, 'G': 55},
            'Power': {'E': 40, 'S': 50, 'G': 55},
            'FMCG': {'E': 60, 'S': 65, 'G': 65},
            'Automobiles': {'E': 45, 'S': 55, 'G': 60},
            'Auto': {'E': 45, 'S': 55, 'G': 60},
            'Metals & Mining': {'E': 35, 'S': 45, 'G': 50},
            'Infrastructure': {'E': 40, 'S': 50, 'G': 55},
            'Cement': {'E': 40, 'S': 50, 'G': 55},
            'Real Estate': {'E': 45, 'S': 50, 'G': 50},
            'Telecom': {'E': 55, 'S': 60, 'G': 60},
            'Media': {'E': 50, 'S': 55, 'G': 55},
            'Other': {'E': 50, 'S': 50, 'G': 50}
        }
        
        # Determine market cap range
        if market_cap >= 50000000000:  # 50B+ = Large cap
            cap_range = 'Large'
            cap_multiplier = 1.1  # Large caps tend to have better ESG
        elif market_cap >= 5000000000:  # 5B-50B = Mid cap
            cap_range = 'Mid'
            cap_multiplier = 1.0
        else:  # < 5B = Small cap
            cap_range = 'Small'
            cap_multiplier = 0.9  # Small caps tend to have weaker ESG
        
        # Try to lookup in proxy database first
        if not self.sector_proxy_df.empty:
            proxy_match = self.sector_proxy_df[
                (self.sector_proxy_df['sector'] == sector) &
                (self.sector_proxy_df['market_cap_range'] == cap_range)
            ]
            
            if proxy_match.empty:
                # Fallback to 'Other' sector
                proxy_match = self.sector_proxy_df[
                    (self.sector_proxy_df['sector'] == 'Other') &
                    (self.sector_proxy_df['market_cap_range'] == cap_range)
                ]
            
            if not proxy_match.empty:
                proxy = proxy_match.iloc[0]
                env_score = proxy['environmental_score']
                social_score = proxy['social_score']
                gov_score = proxy['governance_score']
                overall_esg = proxy['overall_esg']
                data_source = proxy['data_source']
                print(f"   📊 Using proxy database: {sector} ({cap_range} cap)")
            else:
                # Use sector defaults
                defaults = sector_defaults.get(sector, sector_defaults['Other'])
                env_score = defaults['E'] * cap_multiplier
                social_score = defaults['S'] * cap_multiplier
                gov_score = defaults['G'] * cap_multiplier
                overall_esg = (env_score + social_score + gov_score) / 3
                data_source = f'Sector Average ({sector})'
                print(f"   📊 Using sector defaults: {sector} → E:{env_score:.0f} S:{social_score:.0f} G:{gov_score:.0f}")
        else:
            # Use sector defaults (no proxy database)
            defaults = sector_defaults.get(sector, sector_defaults['Other'])
            env_score = defaults['E'] * cap_multiplier
            social_score = defaults['S'] * cap_multiplier
            gov_score = defaults['G'] * cap_multiplier
            overall_esg = (env_score + social_score + gov_score) / 3
            data_source = f'Sector Average ({sector})'
            print(f"   📊 Using sector defaults: {sector} → E:{env_score:.0f} S:{social_score:.0f} G:{gov_score:.0f}")
        
        # Ensure scores are within 0-100 range
        env_score = max(0, min(100, env_score))
        social_score = max(0, min(100, social_score))
        gov_score = max(0, min(100, gov_score))
        overall_esg = max(0, min(100, overall_esg))
        
        star_rating = self._score_to_stars(overall_esg)
        
        return {
            'environmental_score': round(env_score, 2),
            'social_score': round(social_score, 2),
            'governance_score': round(gov_score, 2),
            'overall_esg_score': round(overall_esg, 2),
            'star_rating': star_rating,
            'star_rating_text': '⭐' * star_rating,
            'rating_label': self._get_rating_label(star_rating),
            'data_source': f'{data_source} (Proxy)',
            'is_proxy': True,
            'sector': sector,
            'market_cap_range': cap_range
        }
    
    def _score_to_stars(self, score: float) -> int:
        """
        Convert ESG score (0-100) to star rating (1-5)
        
        ⭐⭐⭐⭐⭐ Leader: 80-100
        ⭐⭐⭐⭐ Strong: 60-79
        ⭐⭐⭐ Average: 40-59
        ⭐⭐ Below Average: 20-39
        ⭐ Laggard: 0-19
        """
        if score >= 80:
            return 5
        elif score >= 60:
            return 4
        elif score >= 40:
            return 3
        elif score >= 20:
            return 2
        else:
            return 1
    
    def _get_rating_label(self, stars: int) -> str:
        """Get text label for star rating"""
        labels = {
            5: 'Leader',
            4: 'Strong',
            3: 'Average',
            2: 'Below Average',
            1: 'Laggard'
        }
        return labels.get(stars, 'Unrated')
    
    def get_esg_risk_multiplier(self, esg_score: float) -> float:
        """
        Get risk adjustment factor for stress testing integration
        
        Parameters:
        -----------
        esg_score : float
            Overall ESG score (0-100)
        
        Returns:
        --------
        float
            Risk multiplier (0.8 for good ESG, 1.3 for poor ESG)
        """
        if esg_score >= 80:  # 5 stars
            return 0.80  # 20% risk reduction
        elif esg_score >= 60:  # 4 stars
            return 0.90  # 10% risk reduction
        elif esg_score >= 40:  # 3 stars
            return 1.00  # Neutral
        elif esg_score >= 20:  # 2 stars
            return 1.15  # 15% risk increase
        else:  # 1 star
            return 1.30  # 30% risk increase
    
    def calculate_portfolio_esg(self, portfolio_df: pd.DataFrame) -> Dict:
        """
        Calculate portfolio-level ESG with coverage tracking
        
        Parameters:
        -----------
        portfolio_df : pd.DataFrame
            Portfolio holdings data
        
        Returns:
        --------
        dict
            Portfolio ESG scores, coverage %, sector breakdown, holdings detail
        """
        if portfolio_df.empty:
            return self._default_portfolio_esg()
        
        holdings_esg = []
        total_value = 0
        rated_value = 0
        unrated_count = 0
        
        print(f"\n🔍 Portfolio ESG Analysis Started:")
        print(f"   Total holdings to analyze: {len(portfolio_df)}")
        print(f"   Available columns: {list(portfolio_df.columns)}")
        
        # Debug: Show first row column mapping (simplified)
        if len(portfolio_df) > 0:
            print(f"\n   Sample row (first holding) - Key columns:")
            first_row = portfolio_df.iloc[0]
            # Only show relevant columns, not all data
            col_map_lower = {col.strip().lstrip('\ufeff').lower(): col for col in first_row.index}
            
            for pattern in ['company', 'stock name', 'isin', 'symbol', 'quantity', 'qty', 
                           'current value', 'closing value', 'invested value', 'market value']:
                actual_col = col_map_lower.get(pattern)
                if actual_col and pd.notna(first_row[actual_col]):
                    print(f"      {actual_col}: {first_row[actual_col]}")
        
        # Detect date columns once for the entire DataFrame
        date_price_columns = detect_date_price_columns(portfolio_df)
        if date_price_columns:
            print(f"\n   📅 Detected {len(date_price_columns)} date/prev-close price columns:")
            for col in date_price_columns[:5]:  # Show first 5
                print(f"      • {col}")
        
        # Track skipped rows for diagnostics
        skipped_rows = []
        
        for idx, row in portfolio_df.iterrows():
            # Create case-insensitive column lookup (also strip BOM and whitespace)
            col_map_lower = {col.strip().lstrip('\ufeff').lower(): col for col in row.index}
            
            # Extract holding with salvage logic
            holding_info = extract_holding_from_row(row, portfolio_df, col_map_lower)
            
            if not holding_info:
                reason = "No valid name/symbol/ISIN and no meaningful numeric data"
                print(f"   ⚠️ Skipping row {idx}: {reason}")
                skipped_rows.append((idx, reason))
                continue
            
            stock_name = holding_info['stock_name']
            symbol = holding_info.get('symbol')
            isin = holding_info.get('isin')
            salvage_reason = holding_info.get('salvage_reason')
            
            print(f"\n   📝 Processing: {stock_name}")
            if salvage_reason:
                print(f"   💡 Salvaged using: {salvage_reason}")
            
            # Get sector from CSV or infer
            sector = None
            for pattern in ['sector', 'industry', 'sector name']:
                actual_col = col_map_lower.get(pattern)
                if actual_col and pd.notna(row[actual_col]):
                    sector = row[actual_col]
                    break
            
            # If no sector in CSV, infer it
            if not sector:
                # Get ticker for API lookup (already imported from data_utils)
                ticker_for_api = get_ticker(stock_name=stock_name, isin=isin, symbol=symbol)
                
                # Use unified sector detection (always available from data_utils)
                sector = get_sector_unified(stock_name, ticker=ticker_for_api, isin=isin, prefer_api=True)
                if ticker_for_api:
                    print(f"   📍 Sector: {stock_name} → {sector}")
            
            # ENHANCED VALUE CALCULATION with date column support
            current_value, value_source = self._calculate_current_value_enhanced(
                row, portfolio_df, col_map_lower, date_price_columns
            )
            
            # Final validation
            if current_value == 0:
                reason = f"Zero value after trying all {len(date_price_columns) + 6} calculation methods"
                print(f"   ⚠️ WARNING: {stock_name} has ZERO value!")
                print(f"   ℹ️ Skipping {stock_name} from portfolio calculation")
                skipped_rows.append((idx, reason))
                continue
            
            print(f"   ✅ {stock_name}: Value = ₹{current_value:,.2f} via {value_source}")
            
            total_value += current_value
            
            # Get ticker using consolidated data_utils (already imported)
            ticker = get_ticker(stock_name=stock_name, isin=isin, symbol=symbol)
            
            # ENHANCED ESG DATA FETCHING WITH 3-TIER FALLBACK
            # Priority: API Data → ML Prediction → Sector Proxy
            
            # TIER 1: Try to fetch from API sources (Yahoo, BRSR, etc.)
            esg_data = self.fetch_esg_data(ticker, stock_name, isin)
            ml_explanation = None
            ml_attempted = False
            
            # TIER 2: If no API data, try ML model prediction (with robust fallbacks)
            if esg_data is None and self.ml_predictor is not None:
                ml_attempted = True
                try:
                    print(f"   🤖 Tier 2: No API ESG data - trying ML prediction...")
                    ml_result = self.ml_predictor.predict_esg(
                        ticker=ticker, 
                        stock_name=stock_name, 
                        sector=sector,
                        max_imputed_ratio=0.45  # Allow up to 45% feature imputation (updated from 30%)
                    )
                    
                    if ml_result is not None and ml_result.get('predicted_esg') is not None:
                        # Convert ML prediction to esg_data format
                        esg_data = {
                            'overall_esg_score': ml_result['predicted_esg'],
                            'environmental_score': ml_result.get('raw_features', {}).get('environment_score'),
                            'social_score': ml_result.get('raw_features', {}).get('social_score'),
                            'governance_score': ml_result.get('raw_features', {}).get('governance_score'),
                            'data_source': ml_result.get('esg_source', 'ml_model'),
                            'model_version': ml_result.get('model_version', 'v1'),
                            # NEW: Add imputation metadata
                            'used_imputed_features': ml_result.get('used_imputed_features', []),
                            'imputation_ratio': ml_result.get('imputation_ratio', 0.0),
                            'data_sources_used': ml_result.get('used_sources', [])
                        }
                        ml_explanation = ml_result  # Save full explanation
                        
                        # Log with imputation info
                        impute_pct = ml_result.get('imputation_ratio', 0.0) * 100
                        sources_str = ', '.join(ml_result.get('used_sources', []))
                        print(f"   ✅ ML prediction successful: {ml_result['predicted_esg']:.1f}/100")
                        print(f"      Sources: {sources_str} | Imputed: {impute_pct:.1f}%")
                    else:
                        # Check reason for failure
                        reason = ml_result.get('reason', 'unknown') if ml_result else 'no_result'
                        print(f"   ⚠️ ML prediction failed: {reason}")
                except Exception as e:
                    print(f"   ⚠️ ML prediction failed: {e}")
            
            # TIER 3: Calculate score (with SECTOR PROXY as final fallback)
            if esg_data is None and ml_attempted:
                print(f"   📊 Tier 3: Falling back to sector proxy...")
            
            esg_score = self.calculate_esg_score(esg_data, sector, current_value)
            
            # Track proxy usage and data source
            data_source_label = esg_score.get('data_source', 'unknown')
            
            # Check if it's an ML prediction (look for ml_explanation presence)
            if ml_explanation is not None:
                # ML prediction - count as rated but track separately
                rated_value += current_value
                print(f"   🎯 Data source: ML model prediction (SHAP explainable)")
            elif not esg_score['is_proxy']:
                rated_value += current_value
                print(f"   🎯 Data source: Real ESG from {data_source_label}")
            else:
                unrated_count += 1
                print(f"   🎯 Data source: Sector proxy fallback (tier 3) - {sector}")
            
            holdings_esg.append({
                'stock_name': stock_name,
                'ticker': ticker,
                'sector': sector,
                'current_value': current_value,
                'weight': 0,  # Will calculate after
                'ml_explanation': ml_explanation,  # Store ML SHAP explanation
                **esg_score
            })
        
        # Check if we have any valid holdings after filtering
        if not holdings_esg or total_value == 0:
            print(f"\n⚠️ WARNING: No valid holdings found (all stocks had zero value)")
            print(f"   Returning empty portfolio ESG result")
            return self._default_portfolio_esg()
        
        # Calculate weights and weighted scores
        weighted_env = 0
        weighted_social = 0
        weighted_gov = 0
        weighted_overall = 0
        
        print(f"\n📊 Portfolio Totals:")
        print(f"   Total Portfolio Value: ₹{total_value:,.2f}")
        print(f"   Calculating weights...")
        
        for holding in holdings_esg:
            weight = holding['current_value'] / total_value if total_value > 0 else 0
            holding['weight'] = round(weight * 100, 2)  # As percentage
            
            weighted_env += holding['environmental_score'] * weight
            weighted_social += holding['social_score'] * weight
            weighted_gov += holding['governance_score'] * weight
            weighted_overall += holding['overall_esg_score'] * weight
            
            print(f"   {holding['stock_name']}: {holding['weight']:.2f}% (ESG: {holding['overall_esg_score']:.2f})")
        
        print(f"\n✅ Weighted Portfolio ESG Score: {weighted_overall:.2f}/100")
        
        # Portfolio star rating
        portfolio_stars = self._score_to_stars(weighted_overall)
        
        # Coverage percentage - with safety checks for NaN
        import numpy as np
        if total_value > 0 and not np.isnan(rated_value) and not np.isnan(total_value):
            coverage_pct = (rated_value / total_value * 100)
        else:
            coverage_pct = 0.0
        
        # Ensure coverage_pct is valid
        if np.isnan(coverage_pct) or np.isinf(coverage_pct):
            coverage_pct = 0.0
        
        # Sanitize counts to ensure they're valid integers
        rated_holdings_count = max(0, len(holdings_esg) - unrated_count)
        unrated_holdings_count = max(0, unrated_count)
        
        # Track data sources for transparency
        data_sources = {
            'api_data': 0,
            'ml_predictions': 0,
            'sector_proxy': 0
        }
        
        for holding in holdings_esg:
            source = holding.get('data_source', 'unknown').lower()
            is_proxy = holding.get('is_proxy', False)
            
            if holding.get('ml_explanation') is not None:
                # Has ML explanation means it was ML predicted
                data_sources['ml_predictions'] += 1
            elif is_proxy:
                # Explicitly marked as proxy
                data_sources['sector_proxy'] += 1
            elif any(api_name in source for api_name in ['yahoo', 'finance', 'brsr', 'finnhub', 'rapidapi', 'fmp', 'msci', 'esg']):
                # Contains API-related keywords
                data_sources['api_data'] += 1
            else:
                # Fallback - if not ML and not clearly API, assume proxy
                data_sources['sector_proxy'] += 1
        
        print(f"\n📊 Data Sources Summary:")
        print(f"   🌐 API Data: {data_sources['api_data']} stocks")
        print(f"   🤖 ML Predictions: {data_sources['ml_predictions']} stocks")
        print(f"   📊 Sector Proxy: {data_sources['sector_proxy']} stocks")
        
        # Monitoring: ML prediction quality check
        if data_sources['ml_predictions'] > 0:
            ml_holdings = [h for h in holdings_esg if h.get('ml_explanation') is not None]
            predictions_near_50 = sum(1 for h in ml_holdings if abs(h.get('esg_score', 0) - 50.0) < 0.1)
            fraction_near_50 = predictions_near_50 / len(ml_holdings) if ml_holdings else 0
            
            print(f"\n🔍 ML Prediction Quality Monitor:")
            print(f"   • Predictions ≈ 50.0: {predictions_near_50}/{len(ml_holdings)} ({fraction_near_50*100:.1f}%)")
            
            # Alert if too many predictions are generic (threshold: 30%)
            if fraction_near_50 > 0.3:
                print(f"   ⚠️ WARNING: {fraction_near_50*100:.1f}% of ML predictions are near 50.0")
                print(f"   → This suggests ESG features are being imputed from sector defaults")
                print(f"   → Model may be echoing sector averages instead of learning from data")
                print(f"   → Consider: (1) Collecting more real ESG data, or (2) Retraining without E/S/G inputs")
            
            # Check imputation stats
            high_imputation = sum(1 for h in ml_holdings if h.get('ml_explanation', {}).get('imputation_ratio', 0) > 0.2)
            if high_imputation > 0:
                print(f"   • High imputation (>20%): {high_imputation}/{len(ml_holdings)} stocks")
        
        # Sector breakdown
        sector_breakdown = self._calculate_sector_breakdown(holdings_esg)
        
        # PORTFOLIO-LEVEL SHAP AGGREGATION (ML Enhancement)
        portfolio_shap_analysis = None
        if self.ml_predictor is not None:
            try:
                # Check if any holdings have ML explanations
                has_ml_explanations = any(h.get('ml_explanation') is not None for h in holdings_esg)
                
                if has_ml_explanations:
                    print(f"\n🔬 Generating portfolio-level SHAP analysis...")
                    portfolio_shap_analysis = self.ml_predictor.aggregate_portfolio_shap(holdings_esg)
                    
                    if portfolio_shap_analysis:
                        print(f"   ✅ Portfolio SHAP: {portfolio_shap_analysis['coverage_weight']*100:.1f}% of portfolio explained by ML")
                        print(f"   📊 Top feature groups:")
                        for group in portfolio_shap_analysis.get('grouped_contributions', [])[:3]:
                            print(f"      • {group['group']}: {group['contribution']:+.2f} contribution")
            except Exception as e:
                print(f"   ⚠️ SHAP aggregation failed: {e}")
                portfolio_shap_analysis = None
        
        return {
            'portfolio_esg_scores': {
                'environmental': round(weighted_env, 1),
                'social': round(weighted_social, 1),
                'governance': round(weighted_gov, 1),
                'overall': round(weighted_overall, 1)
            },
            'star_rating': portfolio_stars,
            'star_rating_text': '⭐' * portfolio_stars,
            'rating_label': self._get_rating_label(portfolio_stars),
            'coverage_percentage': round(max(0.0, min(100.0, coverage_pct)), 1),  # Clamp to 0-100
            'total_holdings': len(holdings_esg),
            'rated_holdings': rated_holdings_count,
            'unrated_holdings': unrated_holdings_count,
            'sector_breakdown': sector_breakdown,
            'holdings_detail': holdings_esg,
            'total_portfolio_value': total_value,
            'risk_multiplier': self.get_esg_risk_multiplier(weighted_overall),
            'data_sources': data_sources,  # NEW: Track where ESG data came from
            'portfolio_shap_analysis': portfolio_shap_analysis  # NEW: ML explainability
        }
    
    def _create_ticker(self, stock_name: str) -> str:
        """Create ticker symbol from stock name for NSE"""
        # Simple heuristic - extract first word and add .NS
        # In production, use a proper mapping database
        name_clean = stock_name.upper().split()[0]
        
        # Handle common variations
        ticker_map = {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFOSYS': 'INFY.NS',
            'HDFC': 'HDFCBANK.NS',
            'ICICI': 'ICICIBANK.NS',
            'WIPRO': 'WIPRO.NS',
            'ITC': 'ITC.NS',
            'BHARTI': 'BHARTIARTL.NS',
            'AIRTEL': 'BHARTIARTL.NS',
            'MARUTI': 'MARUTI.NS',
            'TATA': 'TATAMOTORS.NS',
        }
        
        for key, value in ticker_map.items():
            if key in stock_name.upper():
                return value
        
        # Default: add .NS
        return f"{name_clean}.NS"
    
    def _calculate_current_value_enhanced(
        self,
        row: pd.Series,
        df: pd.DataFrame,
        col_map: Dict[str, str],
        date_price_columns: List[str]
    ) -> Tuple[float, str]:
        """
        Enhanced value calculation with multiple fallback methods.
        
        Tries multiple calculation methods in priority order:
        1. Direct "Current Value" column
        2. Qty × canonical Price columns
        3. **NEW**: Date columns (newest→oldest) as price source
        4. Invested + P&L
        5. Invested × (1 + Change%)
        6. Qty × Avg Cost (with optional %change)
        7. Last resort: Invested value
        
        Args:
            row: DataFrame row containing holding data
            df: Full portfolio DataFrame (for column access)
            col_map: Lowercase column name mapping
            date_price_columns: List of date-based price columns (sorted newest first)
        
        Returns:
            Tuple of (calculated_value, calculation_method_description)
        """
        
        # Method 1: Direct Current Value
        current_value_col = find_column_case_insensitive(
            col_map, 
            ['current value', 'market value', 'current_value', 'market_value']
        )
        if current_value_col:
            val = safe_float_conversion(row.get(current_value_col, 0))
            if val > 0:
                return (val, "direct_current_value")
        
        # Method 2: Qty × canonical Price columns
        qty_col = find_column_case_insensitive(
            col_map, 
            ['qty', 'quantity', 'shares', 'units']
        )
        price_col = find_column_case_insensitive(
            col_map, 
            ['ltp', 'last traded price', 'cmp', 'current market price', 'current price', 'price']
        )
        
        if qty_col and price_col:
            qty = safe_float_conversion(row.get(qty_col, 0))
            price = safe_float_conversion(row.get(price_col, 0))
            if qty > 0 and price > 0:
                return (qty * price, f"qty_x_price({price_col})")
        
        # Method 3: **NEW** Date-based price columns (newest → oldest)
        if date_price_columns and qty_col:
            qty = safe_float_conversion(row.get(qty_col, 0))
            
            for date_col in date_price_columns:
                date_price = safe_float_conversion(row.get(date_col, 0))
                if date_price > 0:
                    if qty > 0:
                        # Direct multiplication if qty exists
                        return (qty * date_price, f"qty_x_date_price({date_col})")
                    else:
                        # Estimate qty if missing
                        invested_col = find_column_case_insensitive(
                            col_map, 
                            ['invested value', 'invested', 'cost', 'invested_value']
                        )
                        avg_cost_col = find_column_case_insensitive(
                            col_map, 
                            ['avg cost', 'avg. cost', 'average cost', 'avg_cost', 'buy price']
                        )
                        
                        if invested_col:
                            invested = safe_float_conversion(row.get(invested_col, 0))
                            if invested > 0 and avg_cost_col:
                                avg_cost = safe_float_conversion(row.get(avg_cost_col, 0))
                                if avg_cost > 0:
                                    estimated_qty = invested / avg_cost
                                    return (estimated_qty * date_price, f"estimated_qty_x_date_price({date_col})")
        
        # Method 4: Invested + P&L
        invested_col = find_column_case_insensitive(
            col_map, 
            ['invested value', 'invested', 'cost', 'invested_value']
        )
        pnl_col = find_column_case_insensitive(
            col_map, 
            ['p&l', 'pnl', 'profit loss', 'gain loss', 'profit/loss', 
             'unrealized p&l', 'unrealized profit/loss', 'unrealized profit / loss',
             'unrealised profit/loss', 'unrealised p&l', 'realized profit/loss',
             'realised profit/loss', 'profit & loss', 'gain/loss']
        )
        
        if invested_col and pnl_col:
            invested = safe_float_conversion(row.get(invested_col, 0))
            pnl = safe_float_conversion(row.get(pnl_col, 0))
            if invested > 0:
                return (invested + pnl, "invested_plus_pnl")
        
        # Method 5: Invested × (1 + Change%)
        change_col = find_column_case_insensitive(
            col_map, 
            ['%chg', '% chg', 'change%', 'change %', 'day change%', '% change']
        )
        
        if invested_col and change_col:
            invested = safe_float_conversion(row.get(invested_col, 0))
            change_pct = safe_float_conversion(row.get(change_col, 0))
            if invested > 0:
                # Handle both decimal (0.05) and percentage (5.0) formats
                if abs(change_pct) > 1:
                    change_pct = change_pct / 100.0
                return (invested * (1 + change_pct), "invested_x_change_pct")
        
        # Method 6: Qty × Avg Cost (with optional %change)
        avg_cost_col = find_column_case_insensitive(
            col_map, 
            ['avg cost', 'avg. cost', 'average cost', 'avg_cost', 'buy price']
        )
        
        if qty_col and avg_cost_col:
            qty = safe_float_conversion(row.get(qty_col, 0))
            avg_cost = safe_float_conversion(row.get(avg_cost_col, 0))
            
            if qty > 0 and avg_cost > 0:
                base_value = qty * avg_cost
                
                # Apply change% if available
                if change_col:
                    change_pct = safe_float_conversion(row.get(change_col, 0))
                    if abs(change_pct) > 1:
                        change_pct = change_pct / 100.0
                    return (base_value * (1 + change_pct), "qty_x_avg_cost_with_change")
                else:
                    return (base_value, "qty_x_avg_cost")
        
        # Method 7: Last resort - Invested value only
        if invested_col:
            invested = safe_float_conversion(row.get(invested_col, 0))
            if invested > 0:
                return (invested, "invested_fallback")
        
        # Complete failure
        return (0.0, "no_calculation_possible")
    
    def _calculate_sector_breakdown(self, holdings_esg: List[Dict]) -> Dict:
        """Calculate ESG breakdown by sector"""
        sector_data = {}
        
        for holding in holdings_esg:
            sector = holding['sector']
            if sector not in sector_data:
                sector_data[sector] = {
                    'count': 0,
                    'total_weight': 0,
                    'avg_esg_score': 0,
                    'avg_env': 0,
                    'avg_social': 0,
                    'avg_gov': 0,
                    'holdings': []
                }
            
            sector_data[sector]['count'] += 1
            sector_data[sector]['total_weight'] += holding['weight']
            sector_data[sector]['avg_esg_score'] += holding['overall_esg_score']
            sector_data[sector]['avg_env'] += holding['environmental_score']
            sector_data[sector]['avg_social'] += holding['social_score']
            sector_data[sector]['avg_gov'] += holding['governance_score']
            sector_data[sector]['holdings'].append(holding['stock_name'])
        
        # Calculate averages
        for sector in sector_data:
            count = sector_data[sector]['count']
            sector_data[sector]['avg_esg_score'] = round(sector_data[sector]['avg_esg_score'] / count, 1)
            sector_data[sector]['avg_env'] = round(sector_data[sector]['avg_env'] / count, 1)
            sector_data[sector]['avg_social'] = round(sector_data[sector]['avg_social'] / count, 1)
            sector_data[sector]['avg_gov'] = round(sector_data[sector]['avg_gov'] / count, 1)
            sector_data[sector]['total_weight'] = round(sector_data[sector]['total_weight'], 1)
        
        return sector_data
    
    def score_security(self, security_data: Dict) -> Dict:
        """
        Calculate ESG score for an individual security
        
        Parameters:
        -----------
        security_data : dict
            Security information including:
            - name: Company name
            - sector: Sector (optional, will infer)
            - market_cap: Market capitalization
            - ticker: Stock ticker (optional)
            
        Returns:
        --------
        dict
            Comprehensive ESG scoring results with 5-star rating
        """
        security_name = security_data.get('name', 'Unknown')
        sector = security_data.get('sector')
        if not sector:
            sector = self._infer_sector(security_name)
        
        market_cap = security_data.get('market_cap', 0)
        ticker = security_data.get('ticker')
        if not ticker:
            ticker = self._create_ticker(security_name)
        
        # Fetch ESG data
        esg_data = self.fetch_esg_data(ticker, security_name)
        
        # Calculate comprehensive score
        esg_score = self.calculate_esg_score(esg_data, sector, market_cap)
        
        return {
            'security_name': security_name,
            'ticker': ticker,
            'sector': sector,
            'esg_scores': {
                'environmental': esg_score['environmental_score'],
                'social': esg_score['social_score'],
                'governance': esg_score['governance_score'],
                'overall': esg_score['overall_esg_score']
            },
            'star_rating': esg_score['star_rating'],
            'star_rating_text': esg_score['star_rating_text'],
            'rating_label': esg_score['rating_label'],
            'risk_level': self._classify_esg_risk(esg_score['overall_esg_score']),
            'data_source': esg_score['data_source'],
            'is_proxy': esg_score['is_proxy'],
            'risk_multiplier': self.get_esg_risk_multiplier(esg_score['overall_esg_score']),
            'improvement_areas': self._identify_improvement_areas(
                esg_score['environmental_score'],
                esg_score['social_score'],
                esg_score['governance_score']
            )
        }
    
    def score_portfolio(self, portfolio_data: pd.DataFrame) -> Dict:
        """
        Calculate aggregate ESG score for an entire portfolio
        
        Parameters:
        -----------
        portfolio_data : pd.DataFrame
            Portfolio holdings data
            
        Returns:
        --------
        dict
            Portfolio-level ESG analysis with coverage tracking
        """
        if portfolio_data.empty:
            return self._default_portfolio_esg()
        
        # Use comprehensive portfolio ESG calculation
        portfolio_esg = self.calculate_portfolio_esg(portfolio_data)
        
        # Add additional analysis
        holdings_detail = portfolio_esg['holdings_detail']
        
        # Identify top and bottom performers
        sorted_holdings = sorted(holdings_detail, key=lambda x: x['overall_esg_score'], reverse=True)
        top_performers = sorted_holdings[:min(5, len(sorted_holdings))]
        bottom_performers = sorted_holdings[-min(3, len(sorted_holdings)):]
        
        # Generate recommendations
        recommendations = self._generate_comprehensive_recommendations(portfolio_esg)
        
        return {
            'portfolio_esg_scores': portfolio_esg['portfolio_esg_scores'],
            'star_rating': portfolio_esg['star_rating'],
            'star_rating_text': portfolio_esg['star_rating_text'],
            'rating_label': portfolio_esg['rating_label'],
            'risk_level': self._classify_esg_risk(portfolio_esg['portfolio_esg_scores']['overall']),
            'coverage_percentage': portfolio_esg['coverage_percentage'],
            'total_holdings': portfolio_esg['total_holdings'],
            'rated_holdings': portfolio_esg['rated_holdings'],
            'unrated_holdings': portfolio_esg['unrated_holdings'],
            'holdings_detail': holdings_detail,  # Changed from holdings_analysis
            'holdings_analysis': holdings_detail,  # Keep for backwards compatibility
            'sector_breakdown': portfolio_esg['sector_breakdown'],
            'top_esg_performers': top_performers,
            'bottom_esg_performers': bottom_performers,
            'esg_risk_exposures': self._identify_esg_risks_from_holdings(holdings_detail),
            'improvement_recommendations': recommendations,
            'risk_multiplier': portfolio_esg['risk_multiplier'],
            'data_sources': portfolio_esg.get('data_sources', {}),  # NEW: Data source tracking
            'portfolio_shap_analysis': portfolio_esg.get('portfolio_shap_analysis')  # NEW: ML explainability
        }
    
    def _generate_comprehensive_recommendations(self, portfolio_esg: Dict) -> List[str]:
        """Generate ESG improvement recommendations based on comprehensive analysis"""
        recommendations = []
        
        overall_score = portfolio_esg['portfolio_esg_scores']['overall']
        coverage = portfolio_esg['coverage_percentage']
        unrated = portfolio_esg['unrated_holdings']
        
        # Coverage recommendations
        if coverage < 50:
            recommendations.append(f"⚠️ ESG data coverage is only {coverage:.1f}%. Consider focusing on larger, better-documented holdings.")
        elif coverage < 80:
            recommendations.append(f"ℹ️ ESG data available for {coverage:.1f}% of portfolio. Some scores are estimated using sector proxies.")
        
        # Score-based recommendations
        if overall_score < 40:
            recommendations.append("🔴 Portfolio has LOW ESG rating (⭐⭐). Major improvements needed:")
            recommendations.append("   - Divest from bottom-rated ESG holdings")
            recommendations.append("   - Increase allocation to 4★ and 5★ ESG companies")
        elif overall_score < 60:
            recommendations.append("🟡 Portfolio has AVERAGE ESG rating (⭐⭐⭐). Improvement opportunities:")
            recommendations.append("   - Gradually shift allocation toward higher-rated ESG holdings")
            recommendations.append("   - Engage with low-performers on ESG improvements")
        elif overall_score < 80:
            recommendations.append("🟢 Portfolio has STRONG ESG rating (⭐⭐⭐⭐). Maintain and enhance:")
            recommendations.append("   - Monitor for any ESG deterioration in holdings")
            recommendations.append("   - Consider upgrading remaining 3★ holdings to 4★+")
        else:
            recommendations.append("🌟 Portfolio has LEADER ESG rating (⭐⭐⭐⭐⭐). Maintain excellence:")
            recommendations.append("   - Continue ESG leadership and monitoring")
            recommendations.append("   - Share ESG best practices with investees")
        
        # Sector-specific recommendations
        sector_breakdown = portfolio_esg['sector_breakdown']
        low_esg_sectors = [s for s, data in sector_breakdown.items() if data['avg_esg_score'] < 50]
        
        if low_esg_sectors:
            recommendations.append(f"⚠️ Low ESG performance in sectors: {', '.join(low_esg_sectors)}")
            recommendations.append("   - Consider sector rotation toward ESG-friendly sectors")
        
        return recommendations
    
    def _identify_esg_risks_from_holdings(self, holdings: List[Dict]) -> List[str]:
        """Identify ESG risk exposures from holdings"""
        risks = []
        
        total_holdings = len(holdings)
        if total_holdings == 0:
            return risks
        
        # Count by star rating
        star_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        proxy_count = 0
        
        for holding in holdings:
            star_counts[holding['star_rating']] += 1
            if holding['is_proxy']:
                proxy_count += 1
        
        # Risk from low-rated holdings
        low_rated = star_counts[1] + star_counts[2]
        if low_rated / total_holdings > 0.3:
            risks.append(f"High Risk: {low_rated} holdings ({low_rated/total_holdings*100:.0f}%) rated ⭐ or ⭐⭐")
        
        # Risk from proxy usage
        if proxy_count / total_holdings > 0.5:
            risks.append(f"Data Risk: {proxy_count} holdings ({proxy_count/total_holdings*100:.0f}%) using estimated ESG scores")
        
        # Sector concentration risks
        sector_weights = {}
        for holding in holdings:
            sector = holding['sector']
            sector_weights[sector] = sector_weights.get(sector, 0) + holding['weight']
        
        high_impact_sectors = ['Energy', 'Oil & Gas', 'Metals & Mining', 'Chemicals']
        for sector in high_impact_sectors:
            if sector in sector_weights and sector_weights[sector] > 30:
                risks.append(f"Sector Risk: High exposure ({sector_weights[sector]:.1f}%) to environmentally-sensitive {sector} sector")
        
        return risks if risks else ["No significant ESG risk exposures identified"]
    
    def generate_esg_report(self, portfolio_analysis: Dict) -> str:
        """
        Generate comprehensive ESG analysis report using LLM
        
        Parameters:
        -----------
        portfolio_analysis : dict
            Results from score_portfolio method
            
        Returns:
        --------
        str
            Detailed ESG report with recommendations
        """
        if not self.api_key:
            return self._generate_basic_esg_report(portfolio_analysis)
        
        # Prepare ESG analysis context for LLM  
        esg_context = {
            'analysis_type': 'ESG Portfolio Assessment',
            'portfolio_scores': portfolio_analysis['portfolio_esg_scores'],
            'risk_level': portfolio_analysis['risk_level'],
            'top_performers': portfolio_analysis['top_esg_performers'],
            'risk_exposures': portfolio_analysis['esg_risk_exposures'],
            'sector_breakdown': portfolio_analysis['sector_breakdown']
        }
        
        # Use the universal LLM function from utils
        user_input = f"ESG ANALYSIS REQUEST: {esg_context}"
        
        return get_llm_explanation(
            prediction=portfolio_analysis['risk_level'],
            shap_top=str(portfolio_analysis['portfolio_esg_scores']),
            lime_top=str(portfolio_analysis['improvement_recommendations']),
            user_input=user_input,
            api_key=self.api_key
        )
    
    def check_esg_compliance(self, portfolio_analysis: Dict) -> Dict:
        """
        Check ESG compliance against regulatory standards (SFDR, etc.)
        
        Parameters:
        -----------
        portfolio_analysis : dict
            Portfolio ESG analysis results
            
        Returns:
        --------
        dict
            Compliance status and recommendations
        """
        overall_score = portfolio_analysis['portfolio_esg_scores'].get('overall', 
                        portfolio_analysis['portfolio_esg_scores'].get('composite', 0))
        risk_level = portfolio_analysis['risk_level']
        
        # SFDR compliance thresholds (simplified)
        sfdr_thresholds = {
            'article_6': {'min_esg_score': 30},  # Basic ESG integration
            'article_8': {'min_esg_score': 60},  # ESG promotion
            'article_9': {'min_esg_score': 80}   # Sustainable investment
        }
        
        compliance_status = {
            'sfdr_article_6': overall_score >= sfdr_thresholds['article_6']['min_esg_score'],
            'sfdr_article_8': overall_score >= sfdr_thresholds['article_8']['min_esg_score'],
            'sfdr_article_9': overall_score >= sfdr_thresholds['article_9']['min_esg_score']
        }
        
        # Determine highest achievable classification
        if compliance_status['sfdr_article_9']:
            classification = 'Article 9 - Sustainable Investment'
        elif compliance_status['sfdr_article_8']:
            classification = 'Article 8 - ESG Promoting'
        elif compliance_status['sfdr_article_6']:
            classification = 'Article 6 - ESG Integration'
        else:
            classification = 'Non-Compliant'
        
        return {
            'sfdr_classification': classification,
            'compliance_details': compliance_status,
            'improvement_needed': self._calculate_improvement_needed(overall_score, sfdr_thresholds),
            'regulatory_recommendations': self._generate_compliance_recommendations(compliance_status)
        }
    
    # Private helper methods
    
    def _classify_esg_risk(self, composite_score: float) -> str:
        """Classify ESG risk level based on composite score"""
        if composite_score >= 80:
            return 'Low Risk'
        elif composite_score >= 60:
            return 'Medium Risk'
        elif composite_score >= 40:
            return 'High Risk'
        else:
            return 'Extreme Risk'
    
    def _identify_improvement_areas(self, env_score: float, social_score: float, gov_score: float) -> List[str]:
        """Identify areas for ESG improvement"""
        improvements = []
        
        if env_score < 60:
            improvements.append('Environmental practices need improvement')
        if social_score < 60:
            improvements.append('Social responsibility initiatives required')
        if gov_score < 60:
            improvements.append('Governance structures need strengthening')
            
        return improvements if improvements else ['Maintain current ESG standards']
    
    def _fetch_sector_from_api(self, ticker: str) -> Optional[str]:
        """
        Fetch sector information from Yahoo Finance API
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        
        Returns:
        --------
        str or None
            Sector name if found
        """
        try:
            import yfinance as yf
            
            # Try to get stock info
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                return None
            
            # Try to get sector (Yahoo Finance provides this)
            sector = info.get('sector')
            if sector:
                # Map Yahoo Finance sectors to our sector categories
                sector_mapping = {
                    'Financial Services': 'Financial Services',
                    'Banks': 'Banking',
                    'Technology': 'IT Services',
                    'Communication Services': 'Telecom',
                    'Energy': 'Oil & Gas',
                    'Basic Materials': 'Metals & Mining',
                    'Utilities': 'Power',
                    'Industrials': 'Infrastructure',
                    'Consumer Cyclical': 'FMCG',
                    'Consumer Defensive': 'FMCG',
                    'Healthcare': 'Pharmaceuticals',
                    'Real Estate': 'Real Estate'
                }
                
                mapped_sector = sector_mapping.get(sector, sector)
                return mapped_sector
            
            # Try industry as fallback
            industry = info.get('industry')
            if industry:
                # Map common industries
                industry_lower = industry.lower()
                if 'bank' in industry_lower:
                    return 'Banking'
                elif 'software' in industry_lower or 'technology' in industry_lower or 'it' in industry_lower:
                    return 'IT Services'
                elif 'oil' in industry_lower or 'gas' in industry_lower or 'petroleum' in industry_lower:
                    return 'Oil & Gas'
                elif 'pharma' in industry_lower or 'drug' in industry_lower:
                    return 'Pharmaceuticals'
                elif 'auto' in industry_lower:
                    return 'Automobiles'
                elif 'telecom' in industry_lower:
                    return 'Telecom'
                elif 'power' in industry_lower or 'electric' in industry_lower:
                    return 'Power'
                elif 'steel' in industry_lower or 'metal' in industry_lower or 'mining' in industry_lower:
                    return 'Metals & Mining'
                elif 'cement' in industry_lower or 'construction' in industry_lower:
                    return 'Infrastructure'
                
                return industry  # Return raw industry if no mapping
            
            return None
            
        except Exception as e:
            # Silently fail - will use name-based inference
            return None
    
    def _infer_sector(self, security_name: str, ticker: str = None, isin: str = None) -> str:
        """
        Infer sector from security name with API fallback
        
        Parameters:
        -----------
        security_name : str
            Company name
        ticker : str, optional
            Stock ticker for API lookup
        isin : str, optional
            ISIN code
        
        Returns:
        --------
        str
            Sector name
        """
        # First try to get sector from API if ticker is available
        if ticker:
            api_sector = self._fetch_sector_from_api(ticker)
            if api_sector:
                print(f"   📍 Sector from API: {security_name} → {api_sector}")
                return api_sector
        
        # Fallback to name-based inference
        name_upper = security_name.upper()
        
        # Banking & Finance
        if any(term in name_upper for term in ['BANK', 'HDFC', 'ICICI', 'SBI', 'AXIS', 'KOTAK']):
            return 'Banking'
        if any(term in name_upper for term in ['FINANCIAL', 'FINANCE', 'INSURANCE', 'LIC', 'BAJAJ FIN']):
            return 'Financial Services'
        
        # IT & Tech
        if any(term in name_upper for term in ['TCS', 'INFOSYS', 'INFY', 'WIPRO', 'TECH M', 'HCL', 'SOFTWARE', 'IT ', 'COMPUTER']):
            return 'IT Services'
        
        # Pharma & Healthcare
        if any(term in name_upper for term in ['PHARMA', 'SUN PHARMA', 'DR REDDY', 'CIPLA', 'HEALTH', 'MEDICAL', 'HOSPITAL']):
            return 'Pharmaceuticals'
        
        # Oil & Gas
        if any(term in name_upper for term in ['RELIANCE', 'ONGC', 'OIL', 'GAS', 'BPCL', 'HPCL', 'IOC', 'PETROLEUM']):
            return 'Oil & Gas'
        
        # Auto
        if any(term in name_upper for term in ['MARUTI', 'TATA MOTOR', 'MAHINDRA', 'BAJAJ AUTO', 'HERO', 'TVS', 'AUTOMOBILE']):
            return 'Automobiles'
        
        # FMCG
        if any(term in name_upper for term in ['ITC', 'HUL', 'HINDUSTAN', 'BRITANNIA', 'NESTLE', 'DABUR', 'FMCG', 'CONSUMER']):
            return 'FMCG'
        
        # Power & Utilities
        if any(term in name_upper for term in ['POWER', 'NTPC', 'COAL INDIA', 'POWER GRID', 'UTILITY', 'UTILITIES']):
            return 'Power'
        
        # Telecom
        if any(term in name_upper for term in ['BHARTI', 'AIRTEL', 'TELECOM', 'VODAFONE', 'IDEA']):
            return 'Telecom'
        
        # Metals & Mining
        if any(term in name_upper for term in ['TATA STEEL', 'JSW', 'HINDALCO', 'VEDANTA', 'METAL', 'MINING', 'STEEL', 'ALUMINIUM']):
            return 'Metals & Mining'
        
        # Infrastructure
        if any(term in name_upper for term in ['L&T', 'LARSEN', 'INFRA', 'CONSTRUCTION', 'CEMENT', 'UL TRATECH', 'ACC']):
            return 'Infrastructure'
        
        # Cement
        if any(term in name_upper for term in ['CEMENT', 'ULTRATECH', 'ACC', 'SHREE CEM', 'AMBUJA']):
            return 'Cement'
        
        # Real Estate
        if any(term in name_upper for term in ['DLF', 'OBEROI', 'GODREJ PROP', 'REAL', 'ESTATE', 'PROPERTY']):
            return 'Real Estate'
        
        return 'Other'
    
    def _initialize_industry_mapping(self) -> Dict:
        """Initialize industry-specific ESG risk factors"""
        return {
            'Energy': {'environmental_risk': 'High', 'regulatory_risk': 'High', 'transition_risk': 'High'},
            'Oil & Gas': {'environmental_risk': 'High', 'regulatory_risk': 'High', 'transition_risk': 'High'},
            'IT Services': {'data_privacy_risk': 'Medium', 'labor_risk': 'Low', 'environmental_risk': 'Low'},
            'Technology': {'data_privacy_risk': 'Medium', 'labor_risk': 'Low', 'environmental_risk': 'Low'},
            'Banking': {'governance_risk': 'Medium', 'systemic_risk': 'High', 'regulatory_risk': 'High'},
            'Financial Services': {'governance_risk': 'Medium', 'systemic_risk': 'Medium', 'regulatory_risk': 'High'},
            'Pharmaceuticals': {'regulatory_risk': 'High', 'social_impact': 'High', 'environmental_risk': 'Medium'},
            'Automobiles': {'environmental_risk': 'High', 'transition_risk': 'High', 'labor_risk': 'Medium'},
            'Metals & Mining': {'environmental_risk': 'High', 'social_risk': 'High', 'regulatory_risk': 'Medium'},
            'Power': {'environmental_risk': 'High', 'regulatory_risk': 'High', 'transition_risk': 'High'}
        }
    
    def _default_portfolio_esg(self) -> Dict:
        """Default ESG analysis for empty portfolios"""
        return {
            'portfolio_esg_scores': {
                'environmental': 0,
                'social': 0, 
                'governance': 0,
                'overall': 0
            },
            'star_rating': 0,
            'star_rating_text': 'Unrated',
            'rating_label': 'No Data',
            'risk_level': 'Unable to assess',
            'coverage_percentage': 0,
            'total_holdings': 0,
            'rated_holdings': 0,
            'unrated_holdings': 0,
            'holdings_analysis': [],
            'sector_breakdown': {},
            'top_esg_performers': [],
            'bottom_esg_performers': [],
            'esg_risk_exposures': [],
            'improvement_recommendations': ['Add holdings to analyze ESG performance'],
            'risk_multiplier': 1.0
        }
    
    def generate_esg_report(self, portfolio_analysis: Dict) -> str:
        """
        Generate comprehensive ESG analysis report using LLM
        
        Parameters:
        -----------
        portfolio_analysis : dict
            Results from score_portfolio method
            
        Returns:
        --------
        str
            Detailed ESG report with recommendations
        """
        if not self.api_key:
            return self._generate_basic_esg_report(portfolio_analysis)
        
        # Prepare ESG analysis context for LLM  
        scores = portfolio_analysis['portfolio_esg_scores']
        star_rating = portfolio_analysis.get('star_rating', 0)
        coverage = portfolio_analysis.get('coverage_percentage', 0)
        
        esg_context = f"""
ESG Portfolio Analysis:
- Overall ESG Score: {scores.get('overall', 0)}/100 ({star_rating}★)
- Environmental: {scores.get('environmental', 0)}/100
- Social: {scores.get('social', 0)}/100  
- Governance: {scores.get('governance', 0)}/100
- Data Coverage: {coverage}%
- Risk Level: {portfolio_analysis.get('risk_level', 'Unknown')}

Top Performers: {len(portfolio_analysis.get('top_esg_performers', []))} holdings
Bottom Performers: {len(portfolio_analysis.get('bottom_esg_performers', []))} holdings
Sector Breakdown: {len(portfolio_analysis.get('sector_breakdown', {}))} sectors

Provide comprehensive ESG improvement recommendations considering the portfolio's rating and sector exposures.
"""
        
        # Use the universal LLM function from utils
        return get_llm_explanation(
            prediction=f"ESG Rating: {star_rating}★",
            shap_top=str(scores),
            lime_top=str(portfolio_analysis.get('improvement_recommendations', [])),
            user_input=esg_context,
            api_key=self.api_key
        )
    
    def _generate_basic_esg_report(self, portfolio_analysis: Dict) -> str:
        """Generate basic ESG report without LLM"""
        scores = portfolio_analysis['portfolio_esg_scores']
        risk_level = portfolio_analysis['risk_level']
        star_rating = portfolio_analysis.get('star_rating', 0)
        coverage = portfolio_analysis.get('coverage_percentage', 0)
        
        return f"""
📊 **ESG Portfolio Analysis Report**

**Overall ESG Rating:** {'⭐' * star_rating} ({star_rating}/5) - {portfolio_analysis.get('rating_label', 'N/A')}

**ESG Scores:**
- Environmental: {scores.get('environmental', 0)}/100
- Social: {scores.get('social', 0)}/100  
- Governance: {scores.get('governance', 0)}/100
- **Overall Score: {scores.get('overall', 0)}/100**

**Risk Assessment:** {risk_level}
**Data Coverage:** {coverage:.1f}% of portfolio

**Key Recommendations:**
{chr(10).join(['• ' + rec for rec in portfolio_analysis.get('improvement_recommendations', [])])}

*For detailed AI-powered analysis, please provide an OpenRouter API key.*
        """
    
    def _calculate_improvement_needed(self, current_score: float, thresholds: Dict) -> Dict:
        """Calculate improvement needed for compliance"""
        return {
            'article_8': max(0, thresholds['article_8']['min_esg_score'] - current_score),
            'article_9': max(0, thresholds['article_9']['min_esg_score'] - current_score)
        }
    
    def _generate_compliance_recommendations(self, compliance_status: Dict) -> List[str]:
        """Generate compliance-specific recommendations"""
        recommendations = []
        
        if not compliance_status['sfdr_article_6']:
            recommendations.append('Implement basic ESG integration framework')
        if not compliance_status['sfdr_article_8']:
            recommendations.append('Enhance ESG promotion strategies')
        if not compliance_status['sfdr_article_9']:
            recommendations.append('Develop sustainable investment criteria')
            
        return recommendations