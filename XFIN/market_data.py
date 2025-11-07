"""
XFIN Market Data Service
========================

Flexible market data integration with multiple API providers and fallbacks.
Supports environment-based configuration for production deployment.

Authors: XFIN Development Team
License: MIT
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Import XFIN configuration
from .config import get_config, require_api_key


class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information including market cap"""
        pass
    
    @abstractmethod
    def get_batch_company_info(self, symbols: List[str]) -> Dict:
        """Get batch company information"""
        pass


class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage API provider - Good for global markets (FREE: 500 calls/day)"""
    
    def __init__(self, api_key: str, is_demo: bool = False):
        self.api_key = api_key
        self.is_demo = is_demo
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit = 5  # 5 calls per minute for free tier
        
    def get_company_info(self, symbol: str) -> Dict:
        """Get company overview from Alpha Vantage"""
        try:
            # Rate limiting
            if self.is_demo:
                current_time = time.time()
                if current_time - self.last_call_time < 12:  # 5 calls per minute = 12 seconds between calls
                    time.sleep(12 - (current_time - self.last_call_time))
                self.last_call_time = time.time()
            
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'MarketCapitalization' in data and data['MarketCapitalization'] != 'None':
                market_cap = float(data['MarketCapitalization'])
                return {
                    'symbol': symbol,
                    'market_cap': market_cap,
                    'market_cap_category': self._classify_market_cap(market_cap),
                    'sector': data.get('Sector', 'Unknown'),
                    'industry': data.get('Industry', 'Unknown'),
                    'provider': 'Alpha Vantage',
                    'data_quality': 'High'
                }
            
            return {'error': 'Market cap data not available', 'provider': 'Alpha Vantage'}
            
        except Exception as e:
            return {'error': f'Alpha Vantage API error: {str(e)}', 'provider': 'Alpha Vantage'}
    
    def get_batch_company_info(self, symbols: List[str]) -> Dict:
        """Get batch company information with rate limiting"""
        results = {}
        
        if self.is_demo and len(symbols) > 5:
            print(f"⚠️ Demo API key limits batch size to 5 stocks. Processing first 5 of {len(symbols)} symbols.")
            symbols = symbols[:5]
        
        for symbol in symbols:
            result = self.get_company_info(symbol)
            results[symbol] = result
            
        return results
    
    def _classify_market_cap(self, market_cap: float) -> str:
        """Classify market cap based on USD values (Alpha Vantage uses USD)"""
        # Convert USD to approximate INR crores for classification
        # $1 ≈ ₹83, 1 crore = 10M
        market_cap_inr_cr = (market_cap * 83) / 10000000
        
        if market_cap_inr_cr >= 20000:  # ≈$2.4B+
            return 'Large Cap'
        elif market_cap_inr_cr >= 5000:  # ≈$600M+
            return 'Mid Cap'
        else:
            return 'Small Cap'


class NSEProvider(MarketDataProvider):
    """NSE India API provider - Best for Indian stocks, no API key required"""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        
    def get_company_info(self, symbol: str) -> Dict:
        """Get company info from NSE"""
        try:
            # Clean symbol for NSE format
            nse_symbol = symbol.replace('.NS', '').replace('.BO', '').upper()
            
            url = f"{self.base_url}/quote-equity?symbol={nse_symbol}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract market cap (NSE provides in crores)
                market_cap_cr = data.get('marketCap', 0)
                
                if market_cap_cr and market_cap_cr > 0:
                    return {
                        'symbol': symbol,
                        'market_cap': market_cap_cr * 10000000,  # Convert crores to actual value
                        'market_cap_category': self._classify_market_cap(market_cap_cr),
                        'sector': self._clean_sector_name(data.get('industry', 'Unknown')),
                        'provider': 'NSE India',
                        'data_quality': 'High'
                    }
            
            return {'error': 'NSE data not available', 'provider': 'NSE India'}
            
        except Exception as e:
            return {'error': f'NSE API error: {str(e)}', 'provider': 'NSE India'}
    
    def get_batch_company_info(self, symbols: List[str]) -> Dict:
        """Get batch company information from NSE"""
        results = {}
        
        for symbol in symbols:
            result = self.get_company_info(symbol)
            results[symbol] = result
            time.sleep(0.5)  # Be respectful to NSE servers
            
        return results
    
    def _classify_market_cap(self, market_cap_cr: float) -> str:
        """Classify based on SEBI/NSE standards (crores)"""
        if market_cap_cr >= 20000:
            return 'Large Cap'
        elif market_cap_cr >= 5000:
            return 'Mid Cap'
        else:
            return 'Small Cap'
    
    def _clean_sector_name(self, sector: str) -> str:
        """Clean and standardize sector names from NSE"""
        sector_mapping = {
            'FINANCIAL SERVICES': 'Banking & Finance',
            'BANKS': 'Banking & Finance',
            'INFORMATION TECHNOLOGY': 'Technology',
            'PHARMACEUTICALS': 'Healthcare & Pharma',
            'OIL GAS & CONSUMABLE FUELS': 'Energy & Oil',
            'AUTOMOBILES': 'Automobiles',
            'CONSUMER GOODS': 'Consumer Goods',
            'CONSTRUCTION': 'Infrastructure',
            'TELECOMMUNICATIONS': 'Telecom'
        }
        
        return sector_mapping.get(sector.upper(), sector)


class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance provider - Free fallback option"""
    
    def __init__(self):
        self.yf_available = self._check_yfinance_availability()
        
    def _check_yfinance_availability(self) -> bool:
        """Check if yfinance package is available"""
        try:
            import yfinance
            return True
        except ImportError:
            return False
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company info from Yahoo Finance"""
        if not self.yf_available:
            return {'error': 'yfinance package not installed', 'provider': 'Yahoo Finance'}
        
        try:
            import yfinance as yf
            
            # Handle different symbol formats
            yahoo_symbol = self._format_symbol_for_yahoo(symbol)
            
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            market_cap = info.get('marketCap', 0)
            
            if market_cap and market_cap > 0:
                return {
                    'symbol': symbol,
                    'market_cap': market_cap,
                    'market_cap_category': self._classify_market_cap(market_cap),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'provider': 'Yahoo Finance',
                    'data_quality': 'Medium'
                }
            
            return {'error': 'Yahoo Finance data not available', 'provider': 'Yahoo Finance'}
            
        except Exception as e:
            return {'error': f'Yahoo Finance error: {str(e)}', 'provider': 'Yahoo Finance'}
    
    def get_batch_company_info(self, symbols: List[str]) -> Dict:
        """Get batch company information from Yahoo Finance"""
        results = {}
        
        for symbol in symbols:
            result = self.get_company_info(symbol)
            results[symbol] = result
            
        return results
    
    def _format_symbol_for_yahoo(self, symbol: str) -> str:
        """Format symbol for Yahoo Finance API"""
        # Add .NS for NSE stocks if not present
        if not ('.' in symbol) and len(symbol) <= 10:
            # Likely an Indian stock
            return f"{symbol}.NS"
        return symbol
    
    def _classify_market_cap(self, market_cap: float) -> str:
        """Classify market cap (Yahoo provides USD values)"""
        # Similar to Alpha Vantage classification
        market_cap_inr_cr = (market_cap * 83) / 10000000
        
        if market_cap_inr_cr >= 20000:
            return 'Large Cap'
        elif market_cap_inr_cr >= 5000:
            return 'Mid Cap'
        else:
            return 'Small Cap'


class FallbackClassifier:
    """Fallback classification when all APIs fail"""
    
    @staticmethod
    def classify_by_keywords(symbol: str) -> Dict:
        """Classify using keyword-based approach"""
        
        # Enhanced sector classification
        sector_keywords = {
            'Banking & Finance': [
                'BANK', 'BANKS', 'BANKING', 'FINANCIAL', 'FINANCE', 'CAPITAL', 'CREDIT',
                'INSURANCE', 'MUTUAL', 'FUND', 'SECURITIES', 'INVESTMENT', 'ASSET',
                'HDFC', 'ICICI', 'SBI', 'AXIS', 'KOTAK', 'BAJAJ', 'LIC', 'IDFC'
            ],
            'Technology': [
                'TECH', 'TECHNOLOGY', 'SOFTWARE', 'IT', 'INFOTECH', 'SYSTEM', 'SYSTEMS',
                'DATA', 'DIGITAL', 'CYBER', 'CLOUD', 'AI', 'AUTOMATION',
                'TCS', 'INFOSYS', 'WIPRO', 'TECHM', 'MINDTREE', 'MPHASIS'
            ],
            'Healthcare & Pharma': [
                'PHARMA', 'PHARMACEUTICAL', 'HEALTH', 'HEALTHCARE', 'MEDICAL', 'MEDICINE',
                'DRUG', 'DRUGS', 'BIO', 'LIFE', 'HOSPITAL', 'CLINIC',
                'CIPLA', 'LUPIN', 'DRLABS', 'SUNPHARMA', 'BIOCON', 'CADILA'
            ],
            'Energy & Oil': [
                'OIL', 'GAS', 'ENERGY', 'POWER', 'PETROLEUM', 'COAL', 'SOLAR',
                'ELECTRIC', 'ELECTRICITY', 'RENEWABLE', 'FUEL',
                'ONGC', 'IOC', 'BPCL', 'HPCL', 'GAIL', 'NTPC', 'POWERGRID'
            ],
            'Automobiles': [
                'AUTO', 'AUTOMOBILE', 'MOTORS', 'MOTOR', 'CAR', 'CARS', 'VEHICLE',
                'TRACTOR', 'BIKE', 'SCOOTER', 'TRUCK', 'BUS',
                'MARUTI', 'TATA', 'MAHINDRA', 'BAJAJ', 'HERO', 'TVS', 'EICHER'
            ],
            'Consumer Goods': [
                'CONSUMER', 'GOODS', 'PRODUCTS', 'FOODS', 'FOOD', 'BEVERAGE',
                'RETAIL', 'STORE', 'MART', 'BRAND', 'LIFESTYLE',
                'HUL', 'ITC', 'BRITANNIA', 'NESTLE', 'GODREJ', 'DABUR'
            ],
            'Infrastructure': [
                'CONSTRUCTION', 'INFRASTRUCTURE', 'BUILDING', 'CEMENT', 'STEEL',
                'REAL', 'ESTATE', 'PROPERTY', 'HOUSING', 'DEVELOPER',
                'L&T', 'DLF', 'ULTRATECH', 'ACC', 'AMBUJA', 'JSW'
            ],
            'Telecom': [
                'TELECOM', 'COMMUNICATION', 'MOBILE', 'NETWORK', 'WIRELESS',
                'BHARTI', 'AIRTEL', 'JIO', 'IDEA', 'VODAFONE'
            ]
        }
        
        symbol_upper = symbol.upper()
        sector = 'Diversified'
        max_matches = 0
        
        for sector_name, keywords in sector_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in symbol_upper)
            if matches > max_matches:
                max_matches = matches
                sector = sector_name
        
        return {
            'symbol': symbol,
            'market_cap': 0,
            'market_cap_category': 'Unknown Cap',
            'sector': sector,
            'provider': 'Fallback Classifier',
            'data_quality': 'Low'
        }


class FinancialModelingPrepProvider(MarketDataProvider):
    """Financial Modeling Prep API - BEST FREE OPTION (250 calls/day FREE)
    
    Get FREE API key from: https://financialmodelingprep.com/developer/docs
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
    def get_company_info(self, symbol: str) -> Dict:
        """Get company info from Financial Modeling Prep using new API endpoints"""
        try:
            # Try new endpoint structure (post Aug 2025)
            url = f"{self.base_url}/market-capitalization/{symbol}?apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    company = data[0]
                    market_cap = company.get('marketCapitalization', 0)
                    
                    # Get additional company info from quote endpoint
                    quote_url = f"{self.base_url}/quote/{symbol}?apikey={self.api_key}"
                    quote_response = requests.get(quote_url, timeout=10)
                    
                    sector = 'Unknown'
                    if quote_response.status_code == 200:
                        quote_data = quote_response.json()
                        if quote_data and len(quote_data) > 0:
                            sector = quote_data[0].get('sector', 'Unknown')
                    
                    return {
                        'symbol': symbol,
                        'market_cap': market_cap,
                        'market_cap_category': self._classify_market_cap(market_cap),
                        'sector': sector,
                        'provider': 'Financial Modeling Prep',
                        'data_quality': 'High'
                    }
            elif response.status_code == 403:
                # Legacy endpoint issue - fall back to basic quote
                quote_url = f"{self.base_url}/quote/{symbol}?apikey={self.api_key}"
                quote_response = requests.get(quote_url, timeout=10)
                
                if quote_response.status_code == 200:
                    quote_data = quote_response.json()
                    if quote_data and len(quote_data) > 0:
                        quote = quote_data[0]
                        market_cap = quote.get('marketCap', 0)
                        
                        return {
                            'symbol': symbol,
                            'market_cap': market_cap,
                            'market_cap_category': self._classify_market_cap(market_cap),
                            'sector': quote.get('sector', 'Unknown'),
                            'provider': 'Financial Modeling Prep (Quote)',
                            'data_quality': 'Medium'
                        }
                        
                return {'error': 'FMP legacy endpoint deprecated - upgrade plan needed', 'provider': 'Financial Modeling Prep'}
            
            return {'error': f'FMP API call failed (HTTP {response.status_code})', 'provider': 'Financial Modeling Prep'}
            
        except Exception as e:
            return {'error': f'FMP API error: {str(e)}', 'provider': 'Financial Modeling Prep'}
    
    def get_batch_company_info(self, symbols: List[str]) -> Dict:
        """Get batch company information from Financial Modeling Prep"""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_company_info(symbol)
        return results
    
    def _classify_market_cap(self, market_cap: float) -> str:
        """Classify market cap (USD-based)"""
        if market_cap >= 10_000_000_000:  # $10B+
            return 'Large Cap'
        elif market_cap >= 2_000_000_000:  # $2B+
            return 'Mid Cap'
        else:
            return 'Small Cap'

class IEXCloudProvider(MarketDataProvider):
    """IEX Cloud API - DEPRECATED (no longer supported)
    
    Note: IEX Cloud has been deprecated
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://cloud.iexapis.com/stable"
        
    def get_company_info(self, symbol: str) -> Dict:
        """Get company info from IEX Cloud"""
        try:
            # Get company stats
            url = f"{self.base_url}/stock/{symbol}/stats?token={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                market_cap = data.get('marketcap', 0)
                
                # Get company info for sector
                company_url = f"{self.base_url}/stock/{symbol}/company?token={self.api_key}"
                company_response = requests.get(company_url, timeout=10)
                company_data = company_response.json() if company_response.status_code == 200 else {}
                
                return {
                    'symbol': symbol,
                    'market_cap': market_cap,
                    'market_cap_category': self._classify_market_cap(market_cap),
                    'sector': company_data.get('sector', 'Unknown'),
                    'industry': company_data.get('industry', 'Unknown'),
                    'provider': 'IEX Cloud',
                    'data_quality': 'High'
                }
            
            return {'error': 'IEX Cloud API call failed', 'provider': 'IEX Cloud'}
            
        except Exception as e:
            return {'error': f'IEX Cloud error: {str(e)}', 'provider': 'IEX Cloud'}
    
    def get_batch_company_info(self, symbols: List[str]) -> Dict:
        """Get batch company information from IEX Cloud"""
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_company_info(symbol)
        return results
    
    def _classify_market_cap(self, market_cap: float) -> str:
        """Classify market cap (USD-based)"""
        if market_cap >= 10_000_000_000:  # $10B+
            return 'Large Cap'
        elif market_cap >= 2_000_000_000:  # $2B+
            return 'Mid Cap'
        else:
            return 'Small Cap'


class PolygonProvider(MarketDataProvider):
    """Polygon.io API - Good free tier (5 calls/minute) for US stocks
    
    Get FREE API key from: https://polygon.io/
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v3"
        
    def get_company_info(self, symbol: str) -> Dict:
        """Get company info from Polygon.io"""
        try:
            # Get ticker details
            url = f"{self.base_url}/reference/tickers/{symbol}?apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'OK' and 'results' in data:
                    ticker_data = data['results']
                    market_cap = ticker_data.get('market_cap', 0)
                    
                    return {
                        'symbol': symbol,
                        'market_cap': market_cap,
                        'market_cap_category': self._classify_market_cap(market_cap),
                        'sector': ticker_data.get('sic_description', 'Unknown'),
                        'provider': 'Polygon.io',
                        'data_quality': 'High'
                    }
            elif response.status_code == 403:
                return {'error': 'Polygon.io: API key required or rate limit exceeded', 'provider': 'Polygon.io'}
            else:
                return {'error': f'Polygon.io API error (HTTP {response.status_code})', 'provider': 'Polygon.io'}
            
        except Exception as e:
            return {'error': f'Polygon.io error: {str(e)}', 'provider': 'Polygon.io'}
    
    def get_batch_company_info(self, symbols: List[str]) -> Dict:
        """Get batch company information from Polygon.io with rate limiting"""
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.get_company_info(symbol)
            time.sleep(12)  # 5 calls/minute = 12 seconds between calls for free tier
            
        return results
    
    def _classify_market_cap(self, market_cap: float) -> str:
        """Classify market cap based on USD values"""
        if market_cap >= 10_000_000_000:  # $10B+
            return 'Large Cap'
        elif market_cap >= 2_000_000_000:  # $2B+
            return 'Mid Cap'
        else:
            return 'Small Cap'


class MarketDataService:
    """
    Main market data service with flexible API key management
    
    Supports:
    - Demo keys for immediate usage
    - User-provided keys for full functionality  
    - Environment variables for easy configuration
    - Multiple provider fallbacks
    - Intelligent caching
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.providers = []
        self.cache = {}
        self.cache_duration = timedelta(hours=4)
        
        # Use XFIN configuration for API keys
        self.env = get_config()
        
        self._initialize_providers()
        
    def _initialize_providers(self):
        """Initialize available providers based on configuration and available keys"""
        
        # Priority 1: Alpha Vantage (500 calls/day free tier available)
        if self.env.has_api_key('alpha_vantage'):
            alpha_key = self.env.get_api_key('alpha_vantage')
            self.providers.append(AlphaVantageProvider(alpha_key, False))
            # Logging removed to prevent infinite loops
        else:
            print("� Configure ALPHA_VANTAGE_KEY for market data access")
            print("   Get your FREE key: https://www.alphavantage.co/support/#api-key")
        
        # Priority 2: Polygon.io (5 calls/minute free tier)
        if self.env.has_api_key('polygon'):
            polygon_key = self.env.get_api_key('polygon')
            self.providers.append(PolygonProvider(polygon_key))
            # Logging removed to prevent infinite loops
        else:
            print("� Configure POLYGON_KEY for additional market data")
            print("   Get your FREE key (5 calls/min): https://polygon.io/")
        
        # Priority 3: IEX Cloud (50K messages/month free)
        if self.env.has_api_key('iex_cloud'):
            iex_key = self.env.get_api_key('iex_cloud')
            self.providers.append(IEXCloudProvider(iex_key))
            # Logging removed to prevent infinite loops
        else:
            print("� Configure IEX_CLOUD_KEY for comprehensive market data")
            print("   Get your FREE key (50K messages/month): https://iexcloud.io/")
        
        # Priority 4: NSE (no API key required, good for Indian stocks)
        self.providers.append(NSEProvider())
        
        # Priority 5: Yahoo Finance (fallback)
        yahoo_provider = YahooFinanceProvider()
        if yahoo_provider.yf_available:
            self.providers.append(yahoo_provider)
        else:
            print("💡 Install yfinance for additional market data: pip install yfinance")
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment configuration"""
        return self.env.get_api_key(provider)
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get market data for symbols with intelligent fallbacks
        
        Returns:
        --------
        Dict mapping symbol to market data or error info
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        results = {}
        
        # Check cache first
        symbols_to_fetch = []
        for symbol in symbols:
            cache_key = f"{symbol}_{datetime.now().date()}"
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if datetime.now() - cache_entry['timestamp'] < self.cache_duration:
                    results[symbol] = cache_entry['data']
                    continue
            symbols_to_fetch.append(symbol)
        
        if not symbols_to_fetch:
            return results
        
        # Try each provider until we get good data
        for provider in self.providers:
            remaining_symbols = [s for s in symbols_to_fetch if s not in results or 'error' in results[s]]
            
            if not remaining_symbols:
                break
            
            try:
                if len(remaining_symbols) == 1:
                    provider_results = {remaining_symbols[0]: provider.get_company_info(remaining_symbols[0])}
                else:
                    provider_results = provider.get_batch_company_info(remaining_symbols)
                
                # Process results
                for symbol, data in provider_results.items():
                    if 'error' not in data:
                        # Cache successful results
                        cache_key = f"{symbol}_{datetime.now().date()}"
                        self.cache[cache_key] = {
                            'data': data,
                            'timestamp': datetime.now()
                        }
                        results[symbol] = data
                    else:
                        results[symbol] = data
                
            except Exception as e:
                continue
        
        # Use fallback classification for any remaining failures
        for symbol in symbols_to_fetch:
            if symbol not in results or 'error' in results[symbol]:
                results[symbol] = FallbackClassifier.classify_by_keywords(symbol)
        
        return results
    
    def get_data_quality_summary(self, results: Dict) -> Dict:
        """Get summary of data quality and sources"""
        quality_summary = {
            'total_symbols': len(results),
            'high_quality': 0,
            'medium_quality': 0,
            'low_quality': 0,
            'providers_used': set(),
            'using_demo_key': False
        }
        
        for symbol, data in results.items():
            if 'data_quality' in data:
                if data['data_quality'] == 'High':
                    quality_summary['high_quality'] += 1
                elif data['data_quality'] == 'Medium':
                    quality_summary['medium_quality'] += 1
                else:
                    quality_summary['low_quality'] += 1
                
                quality_summary['providers_used'].add(data.get('provider', 'Unknown'))
        
        # Check API key configuration status
        quality_summary['api_keys_configured'] = {
            'alpha_vantage': self.env.has_api_key('alpha_vantage'),
            'polygon': self.env.has_api_key('polygon'),
            'iex_cloud': self.env.has_api_key('iex_cloud')
        }
        
        return quality_summary
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company info for a single symbol"""
        results = self.get_market_data([symbol])
        return results.get(symbol, {'error': 'No data available'})
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            'cache_size': len(self.cache),
            'providers_available': len(self.providers),
            'providers': [type(p).__name__ for p in self.providers]
        }
    
    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market cap for a single symbol"""
        data = self.get_company_info(symbol)
        if 'error' not in data and 'market_cap' in data:
            return data['market_cap']
        return None
    
    def get_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a single symbol"""
        data = self.get_company_info(symbol)
        if 'error' not in data and 'sector' in data:
            return data['sector']
        return None


def create_market_data_service(api_keys: Dict = None, custom_config: Dict = None) -> MarketDataService:
    """
    Convenience function to create a market data service
    
    Parameters:
    -----------
    api_keys : Dict, optional
        Dictionary of API keys {'alpha_vantage_key': 'your_key'}
    custom_config : Dict, optional
        Additional configuration options
    
    Returns:
    --------
    MarketDataService instance
    """
    config = {}
    
    if api_keys:
        config.update(api_keys)
    
    if custom_config:
        config.update(custom_config)
    
    return MarketDataService(config)