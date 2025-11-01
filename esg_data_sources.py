"""
ESG Data Fetcher Module
Fetches ESG data from multiple sources with intelligent caching
"""

import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import sqlite3
from pathlib import Path

# Import ticker mapper
try:
    from .ticker_mapper import get_ticker
except ImportError:
    # Fallback if ticker_mapper not available
    def get_ticker(stock_name=None, isin=None, symbol=None):
        if symbol and '.NS' in symbol:
            return symbol
        if symbol:
            return f"{symbol}.NS"
        if stock_name:
            return f"{stock_name.split()[0].upper()}.NS"
        return "UNKNOWN.NS"


class ESGDataFetcher:
    """Fetches and caches ESG data from multiple sources"""
    
    def __init__(self, cache_dir: Optional[str] = None, cache_expiry_days: int = 90):
        """
        Initialize ESG data fetcher
        
        Args:
            cache_dir: Directory for cache database (default: XFIN/data)
            cache_expiry_days: Days before cache expires (default: 90)
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.cache_db = self.cache_dir / 'esg_cache.db'
        self.cache_expiry_days = cache_expiry_days
        
        # Initialize cache database
        self._init_cache_db()
        
        # BRSR data (manual/scraped) - placeholder for future implementation
        self.brsr_data_path = self.cache_dir / 'brsr_data.csv'
    
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
        Fetch ESG data with fallback strategy:
        1. Check cache
        2. Try Yahoo Finance with improved ticker mapping
        3. Try BRSR database (if available)
        4. Return None (will use proxy)
        
        Args:
            ticker: Stock ticker (e.g., 'RELIANCE.NS', 'TCS.NS')
            company_name: Company name for BRSR lookup
            isin: ISIN code for alternative lookup
        
        Returns:
            Dictionary with ESG scores or None
        """
        # Get best ticker using mapper (ISIN > Name > Symbol)
        improved_ticker = get_ticker(stock_name=company_name, isin=isin, symbol=ticker)
        
        # Debug: print ticker mapping
        if improved_ticker != ticker:
            print(f"📍 Ticker mapped: {company_name} → {improved_ticker} (was: {ticker})")
        
        # Step 1: Check cache
        cached = self.get_cached_data(improved_ticker)
        if cached:
            return cached
        
        # Step 2: Try Yahoo Finance with improved ticker
        yahoo_data = self.fetch_from_yahoo_finance(improved_ticker)
        if yahoo_data:
            self.cache_esg_data(improved_ticker, yahoo_data, company_name)
            return yahoo_data
        
        # Step 3: If improved ticker failed, try original ticker
        if improved_ticker != ticker:
            cached = self.get_cached_data(ticker)
            if cached:
                return cached
            
            yahoo_data = self.fetch_from_yahoo_finance(ticker)
            if yahoo_data:
                self.cache_esg_data(ticker, yahoo_data, company_name)
                return yahoo_data
        
        # Step 4: Try BRSR database (if available)
        if company_name:
            brsr_data = self.fetch_from_brsr_database(company_name, isin)
            if brsr_data:
                self.cache_esg_data(improved_ticker, brsr_data, company_name)
                return brsr_data
        
        # Step 5: No data found
        return None
    
    def fetch_from_yahoo_finance(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch ESG data from Yahoo Finance
        
        Yahoo Finance uses MSCI ESG ratings where LOWER scores are BETTER.
        We convert to 0-100 scale where HIGHER is BETTER.
        
        Args:
            ticker: Stock ticker (e.g., 'RELIANCE.NS')
        
        Returns:
            Dictionary with ESG scores (0-100 scale) or None
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get sustainability data
            sustainability = stock.sustainability
            
            if sustainability is None or sustainability.empty:
                return None
            
            # Extract ESG scores from DataFrame
            # Structure: DataFrame with index as row names, single column 'esgScores'
            esg_data = {}
            
            # Extract raw scores (MSCI scale: lower is better)
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
            
            # Convert MSCI scores (0-50 scale, lower=better) to our 0-100 scale (higher=better)
            # MSCI: 0-10=Excellent, 10-20=Good, 20-30=Average, 30-40=Poor, 40-50=Very Poor
            # Our scale: 80-100=Leader, 60-80=Strong, 40-60=Average, 20-40=Below Avg, 0-20=Laggard
            
            def convert_msci_to_score(msci_score):
                """Convert MSCI ESG Risk Score to 0-100 scale (inverted)"""
                if msci_score is None:
                    return None
                # Invert: lower MSCI score = higher our score
                # MSCI range: 0-50, Our range: 0-100
                # Formula: 100 - (msci_score * 2)
                # Use scale factor for better distribution
                converted = 100 - (msci_score * 1.8)
                return max(0, min(100, converted))
            
            # Convert scores
            if env_raw is not None:
                esg_data['environmental_score'] = convert_msci_to_score(env_raw)
            
            if social_raw is not None:
                esg_data['social_score'] = convert_msci_to_score(social_raw)
            
            if gov_raw is not None:
                esg_data['governance_score'] = convert_msci_to_score(gov_raw)
            
            if total_esg_raw is not None:
                esg_data['overall_esg_score'] = convert_msci_to_score(total_esg_raw)
            
            # Additional metadata
            if 'peerGroup' in sustainability.index:
                esg_data['peer_group'] = str(sustainability.loc['peerGroup'].values[0])
            
            if 'percentile' in sustainability.index:
                esg_data['percentile'] = float(sustainability.loc['percentile'].values[0])
            
            if 'esgPerformance' in sustainability.index:
                esg_data['esg_performance'] = str(sustainability.loc['esgPerformance'].values[0])
            
            # Store raw scores for reference
            esg_data['raw_scores'] = {
                'total_esg': total_esg_raw,
                'environment': env_raw,
                'social': social_raw,
                'governance': gov_raw
            }
            
            # Only return if we have at least one score
            if any(key in esg_data for key in ['environmental_score', 'social_score', 'governance_score']):
                esg_data['data_source'] = 'Yahoo Finance (MSCI ESG)'
                esg_data['fetch_date'] = datetime.now().isoformat()
                return esg_data
            
            return None
            
        except Exception as e:
            print(f"Yahoo Finance fetch error for {ticker}: {e}")
            return None
    
    def fetch_from_brsr_database(self, company_name: str, 
                                 isin: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch ESG data from local BRSR database
        
        Args:
            company_name: Company name to search
            isin: ISIN code for matching
        
        Returns:
            Dictionary with ESG scores or None
        """
        if not self.brsr_data_path.exists():
            return None
        
        try:
            brsr_df = pd.read_csv(self.brsr_data_path)
            
            # Search by company name (case-insensitive partial match)
            matches = brsr_df[brsr_df['company_name'].str.lower().str.contains(
                company_name.lower(), na=False)]
            
            # If ISIN provided, try that too
            if isin and 'isin' in brsr_df.columns:
                isin_matches = brsr_df[brsr_df['isin'] == isin]
                if not isin_matches.empty:
                    matches = isin_matches
            
            if matches.empty:
                return None
            
            # Take first match
            match = matches.iloc[0]
            
            esg_data = {
                'environmental_score': float(match.get('environmental_score', 0)),
                'social_score': float(match.get('social_score', 0)),
                'governance_score': float(match.get('governance_score', 0)),
                'overall_esg_score': float(match.get('overall_esg', 0)),
                'data_source': 'SEBI BRSR',
                'fetch_date': datetime.now().isoformat()
            }
            
            return esg_data
            
        except Exception as e:
            print(f"BRSR database error: {e}")
            return None
    
    def fetch_from_news_sentiment(self, company_name: str, 
                                  api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch ESG sentiment from news analysis (using LLM)
        This is a placeholder for future implementation
        
        Args:
            company_name: Company name
            api_key: OpenRouter API key
        
        Returns:
            Dictionary with sentiment scores or None
        """
        # Placeholder - to be implemented with LLM integration
        # Would analyze recent news for ESG controversies
        return None
    
    def cache_esg_data(self, ticker: str, esg_data: Dict[str, Any], 
                      company_name: Optional[str] = None):
        """
        Store ESG data in cache
        
        Args:
            ticker: Stock ticker
            esg_data: ESG data dictionary
            company_name: Optional company name
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO esg_cache 
            (ticker, company_name, environmental_score, social_score, 
             governance_score, overall_esg_score, total_esg, peer_group, 
             percentile, data_source, fetch_date, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker,
            company_name or '',
            esg_data.get('environmental_score'),
            esg_data.get('social_score'),
            esg_data.get('governance_score'),
            esg_data.get('overall_esg_score'),
            esg_data.get('total_esg'),
            esg_data.get('peer_group', ''),
            esg_data.get('percentile'),
            esg_data.get('data_source', 'Unknown'),
            esg_data.get('fetch_date', datetime.now().isoformat()),
            json.dumps(esg_data)
        ))
        
        conn.commit()
        conn.close()
    
    def get_cached_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached ESG data if still valid
        
        Args:
            ticker: Stock ticker
        
        Returns:
            Cached ESG data or None if expired/not found
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM esg_cache WHERE ticker = ?
        ''', (ticker,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Check expiry
        fetch_date = datetime.fromisoformat(row[10])  # fetch_date column
        if datetime.now() - fetch_date > timedelta(days=self.cache_expiry_days):
            return None  # Expired
        
        # Reconstruct data
        esg_data = {
            'environmental_score': row[2],
            'social_score': row[3],
            'governance_score': row[4],
            'overall_esg_score': row[5],
            'total_esg': row[6],
            'peer_group': row[7],
            'percentile': row[8],
            'data_source': row[9] + ' (cached)',
            'fetch_date': row[10]
        }
        
        # Remove None values
        esg_data = {k: v for k, v in esg_data.items() if v is not None}
        
        return esg_data
    
    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cache for specific ticker or all
        
        Args:
            ticker: Specific ticker to clear, or None for all
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        if ticker:
            cursor.execute('DELETE FROM esg_cache WHERE ticker = ?', (ticker,))
        else:
            cursor.execute('DELETE FROM esg_cache')
        
        conn.commit()
        conn.close()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM esg_cache')
        total = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT COUNT(*) FROM esg_cache 
            WHERE datetime(fetch_date) > datetime('now', '-' || ? || ' days')
        ''', (self.cache_expiry_days,))
        valid = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_cached': total,
            'valid_entries': valid,
            'expired_entries': total - valid,
            'cache_expiry_days': self.cache_expiry_days
        }


# Standalone test function
if __name__ == "__main__":
    # Test the fetcher
    fetcher = ESGDataFetcher()
    
    print("Testing ESG Data Fetcher")
    print("=" * 60)
    
    # Test with Indian stocks (add .NS for NSE, .BO for BSE)
    test_tickers = [
        ('RELIANCE.NS', 'Reliance Industries'),
        ('TCS.NS', 'Tata Consultancy Services'),
        ('HDFCBANK.NS', 'HDFC Bank'),
        ('INFY.NS', 'Infosys')
    ]
    
    for ticker, name in test_tickers:
        print(f"\nFetching ESG data for {name} ({ticker})...")
        esg_data = fetcher.fetch_esg_data(ticker, name)
        
        if esg_data:
            print(f"✅ Found data from: {esg_data.get('data_source')}")
            print(f"   Environmental: {esg_data.get('environmental_score', 'N/A')}")
            print(f"   Social: {esg_data.get('social_score', 'N/A')}")
            print(f"   Governance: {esg_data.get('governance_score', 'N/A')}")
            if 'total_esg' in esg_data:
                print(f"   Total ESG: {esg_data.get('total_esg', 'N/A')}")
        else:
            print(f"❌ No ESG data found")
    
    # Show cache stats
    print("\n" + "=" * 60)
    stats = fetcher.get_cache_stats()
    print("Cache Statistics:")
    print(f"  Total entries: {stats['total_cached']}")
    print(f"  Valid entries: {stats['valid_entries']}")
    print(f"  Expired entries: {stats['expired_entries']}")
