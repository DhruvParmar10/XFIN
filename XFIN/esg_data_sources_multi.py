"""
Multi-Source ESG Data Fetcher
Integrates Yahoo Finance, Finnhub, RapidAPI, and FMP for ESG data
"""

import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import sqlite3
from pathlib import Path

# Import ticker mapper from consolidated data_utils
from .data_utils import get_ticker


class MultiSourceESGFetcher:
    """Fetches ESG data from multiple sources with intelligent fallback"""
    
    def __init__(self, cache_dir: Optional[str] = None, cache_expiry_days: int = 90):
        """
        Initialize multi-source ESG data fetcher
        
        Args:
            cache_dir: Directory for cache database
            cache_expiry_days: Days before cache expires
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.cache_db = self.cache_dir / 'esg_cache.db'
        self.cache_expiry_days = cache_expiry_days
        
        # Initialize cache database
        self._init_cache_db()
        
        # API Keys
        self.finnhub_key = os.getenv('FINNHUB_API_KEY', 'd3uatj1r01qvr0dmb8b0d3uatj1r01qvr0dmb8bg')
        self.rapidapi_key = os.getenv('RAPIDAPI_KEY', 'ef37ad4811msh95fb34d426505c9p1784f7jsnbabaa18ce237')
        self.fmp_key = os.getenv('FMP_API_KEY', os.getenv('FMP_KEY', ''))
        
        # Try to import finnhub
        try:
            import finnhub
            self.finnhub_client = finnhub.Client(api_key=self.finnhub_key)
            self.finnhub_available = True
        except ImportError:
            self.finnhub_client = None
            self.finnhub_available = False
            print("⚠️ Finnhub not installed. Run: pip install finnhub-python")
    
    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS esg_cache (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                environmental_score REAL,
                social_score REAL,
                governance_score REAL,
                overall_esg_score REAL,
                total_esg REAL,
                peer_group TEXT,
                percentile REAL,
                data_source TEXT,
                fetch_date TEXT,
                raw_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_esg_data(self, ticker: str, company_name: Optional[str] = None, 
                       isin: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch ESG data with multi-source fallback:
        1. Check cache
        2. Try Finnhub (best for US stocks)
        3. Try Yahoo Finance (good for Indian stocks)
        4. Try RapidAPI (comprehensive)
        5. Try FMP (if key available)
        6. Return None (will use proxy)
        
        Args:
            ticker: Stock ticker
            company_name: Company name
            isin: ISIN code
        
        Returns:
            Dictionary with ESG scores or None
        """
        # Get best ticker using mapper (ISIN > Name > Symbol)
        improved_ticker = get_ticker(stock_name=company_name, isin=isin, symbol=ticker)
        
        # Step 1: Check cache
        cached = self.get_cached_data(improved_ticker)
        if cached:
            return cached
        
        # Extract base ticker for US APIs (remove .NS/.BO suffix)
        us_ticker = improved_ticker.replace('.NS', '').replace('.BO', '')
        
        # Step 2: Try Yahoo Finance FIRST (best coverage at 90%)
        yahoo_data = self.fetch_from_yahoo_finance(improved_ticker)
        if yahoo_data:
            print(f"✅ Yahoo ESG: {company_name or ticker} → {yahoo_data.get('overall_esg_score', 0):.2f}/100")
            self.cache_esg_data(improved_ticker, yahoo_data, company_name)
            return yahoo_data
        
        # Step 3: Try Finnhub (good for US and major international stocks)
        finnhub_data = self.fetch_from_finnhub(us_ticker)
        if finnhub_data:
            print(f"✅ Finnhub ESG: {company_name or ticker} → {finnhub_data.get('overall_esg_score', 0):.2f}/100")
            self.cache_esg_data(improved_ticker, finnhub_data, company_name)
            return finnhub_data
        
        # Step 4: Try RapidAPI
        if company_name:
            rapid_data = self.fetch_from_rapidapi(company_name)
            if rapid_data:
                print(f"✅ RapidAPI ESG: {company_name} → {rapid_data.get('overall_esg_score', 0):.2f}/100")
                self.cache_esg_data(improved_ticker, rapid_data, company_name)
                return rapid_data
        
        # Step 5: Try FMP (if key available)
        if self.fmp_key:
            fmp_data = self.fetch_from_fmp(us_ticker)
            if fmp_data:
                print(f"✅ FMP ESG: {company_name or ticker} → {fmp_data.get('overall_esg_score', 0):.2f}/100")
                self.cache_esg_data(improved_ticker, fmp_data, company_name)
                return fmp_data
        
        # Step 6: No data found
        print(f"⚠️ No ESG data: {company_name or ticker} (will use sector proxy)")
        return None
    
    def fetch_from_finnhub(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch ESG from Finnhub API"""
        if not self.finnhub_available or not self.finnhub_client:
            return None
        
        try:
            esg_data = self.finnhub_client.company_esg_score(ticker)
            
            if not esg_data or 'environmentScore' not in esg_data:
                return None
            
            # Finnhub returns scores 0-100 (higher is better)
            env_score = float(esg_data.get('environmentScore', 0))
            social_score = float(esg_data.get('socialScore', 0))
            gov_score = float(esg_data.get('governanceScore', 0))
            total_esg = float(esg_data.get('totalESG', 0))
            
            return {
                'environmental_score': env_score,
                'social_score': social_score,
                'governance_score': gov_score,
                'overall_esg_score': total_esg,
                'total_esg': total_esg,
                'data_source': 'Finnhub',
                'raw_data': esg_data
            }
        
        except Exception as e:
            return None
    
    def fetch_from_yahoo_finance(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch ESG from Yahoo Finance (MSCI data)"""
        try:
            stock = yf.Ticker(ticker)
            sustainability = stock.sustainability
            
            if sustainability is None or sustainability.empty:
                return None
            
            # Extract scores
            total_esg_raw = None
            env_raw = None
            social_raw = None
            gov_raw = None
            
            if 'totalEsg' in sustainability.index:
                total_esg_raw = float(sustainability.loc['totalEsg'].values[0])
            
            if 'environmentScore' in sustainability.index:
                env_raw = float(sustainability.loc['environmentScore'].values[0])
            
            if 'socialScore' in sustainability.index:
                social_raw = float(sustainability.loc['socialScore'].values[0])
            
            if 'governanceScore' in sustainability.index:
                gov_raw = float(sustainability.loc['governanceScore'].values[0])
            
            # Convert MSCI scores (0-50, lower=better) to 0-100 scale (higher=better)
            def convert_msci_to_100(score):
                if score is None:
                    return None
                # Invert and scale: 0 becomes 100, 50 becomes 0
                converted = 100 - (score * 1.8)
                return max(0, min(100, converted))
            
            env_score = convert_msci_to_100(env_raw) if env_raw is not None else None
            social_score = convert_msci_to_100(social_raw) if social_raw is not None else None
            gov_score = convert_msci_to_100(gov_raw) if gov_raw is not None else None
            overall_score = convert_msci_to_100(total_esg_raw) if total_esg_raw is not None else None
            
            # If we don't have overall but have components, calculate it
            if overall_score is None and all([env_score, social_score, gov_score]):
                overall_score = (env_score + social_score + gov_score) / 3
            
            if overall_score is None:
                return None
            
            return {
                'environmental_score': env_score or 50,
                'social_score': social_score or 50,
                'governance_score': gov_score or 50,
                'overall_esg_score': overall_score,
                'total_esg': overall_score,
                'data_source': 'Yahoo Finance (MSCI)',
                'msci_scores': {
                    'total': total_esg_raw,
                    'environment': env_raw,
                    'social': social_raw,
                    'governance': gov_raw
                }
            }
        
        except Exception as e:
            return None
    
    def fetch_from_rapidapi(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Fetch ESG from RapidAPI"""
        try:
            url = "https://esg-environmental-social-governance-data.p.rapidapi.com/search"
            headers = {
                "x-rapidapi-key": self.rapidapi_key,
                "x-rapidapi-host": "esg-environmental-social-governance-data.p.rapidapi.com"
            }
            params = {"q": company_name}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if not data or not isinstance(data, list) or len(data) == 0:
                return None
            
            # Get first match
            esg_info = data[0]
            
            # Extract ESG scores (adapt based on actual API response structure)
            env_score = float(esg_info.get('environment_score', 0))
            social_score = float(esg_info.get('social_score', 0))
            gov_score = float(esg_info.get('governance_score', 0))
            
            if env_score == 0 and social_score == 0 and gov_score == 0:
                return None
            
            overall = (env_score + social_score + gov_score) / 3
            
            return {
                'environmental_score': env_score,
                'social_score': social_score,
                'governance_score': gov_score,
                'overall_esg_score': overall,
                'total_esg': overall,
                'data_source': 'RapidAPI',
                'raw_data': esg_info
            }
        
        except Exception as e:
            return None
    
    def fetch_from_fmp(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch ESG from Financial Modeling Prep"""
        if not self.fmp_key:
            return None
        
        try:
            url = f"https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data"
            params = {
                'symbol': ticker,
                'apikey': self.fmp_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            if not data or not isinstance(data, list) or len(data) == 0:
                return None
            
            esg_info = data[0]
            
            # Extract scores
            env_score = float(esg_info.get('environmentalScore', 0))
            social_score = float(esg_info.get('socialScore', 0))
            gov_score = float(esg_info.get('governanceScore', 0))
            esg_score = float(esg_info.get('ESGScore', 0))
            
            if esg_score == 0:
                esg_score = (env_score + social_score + gov_score) / 3 if env_score or social_score or gov_score else 0
            
            if esg_score == 0:
                return None
            
            return {
                'environmental_score': env_score,
                'social_score': social_score,
                'governance_score': gov_score,
                'overall_esg_score': esg_score,
                'total_esg': esg_score,
                'data_source': 'Financial Modeling Prep',
                'raw_data': esg_info
            }
        
        except Exception as e:
            return None
    
    def get_cached_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get cached ESG data if not expired"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM esg_cache 
                WHERE ticker = ? 
                AND date(fetch_date) >= date('now', '-' || ? || ' days')
            ''', (ticker, self.cache_expiry_days))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'environmental_score': row[2],
                    'social_score': row[3],
                    'governance_score': row[4],
                    'overall_esg_score': row[5],
                    'total_esg': row[6],
                    'data_source': f"{row[9]} (Cached)",
                }
            
            return None
        
        except Exception as e:
            return None
    
    def cache_esg_data(self, ticker: str, esg_data: Dict, company_name: Optional[str] = None):
        """Cache ESG data to SQLite"""
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO esg_cache 
                (ticker, company_name, environmental_score, social_score, governance_score,
                 overall_esg_score, total_esg, peer_group, percentile, data_source, fetch_date, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                company_name,
                esg_data.get('environmental_score', 0),
                esg_data.get('social_score', 0),
                esg_data.get('governance_score', 0),
                esg_data.get('overall_esg_score', 0),
                esg_data.get('total_esg', 0),
                esg_data.get('peer_group', ''),
                esg_data.get('percentile', 0),
                esg_data.get('data_source', 'Unknown'),
                datetime.now().isoformat(),
                str(esg_data.get('raw_data', {}))
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            pass
