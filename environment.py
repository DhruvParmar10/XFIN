"""
XFIN Library Environment Configuration
Handles API keys and configuration for the XFIN library
"""

import os
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class XFINEnvironment:
    """
    Centralized environment configuration for XFIN library
    """
    
    def __init__(self):
        self._api_keys = None
        self._config_cache = {}
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """
        Get API keys from environment variables with fallbacks
        Returns None for unavailable keys (no hardcoded fallbacks)
        """
        if self._api_keys is None:
            self._api_keys = {
                'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY'),
                'polygon': os.getenv('POLYGON_KEY'), 
                'fmp': os.getenv('FMP_KEY'),
                'iex_cloud': os.getenv('IEX_CLOUD_KEY'),
                'openrouter': os.getenv('OPENROUTER_API_KEY')
            }
        return self._api_keys
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get specific API key for a provider"""
        return self.get_api_keys().get(provider)
    
    def has_api_key(self, provider: str) -> bool:
        """Check if API key is available for provider"""
        key = self.get_api_key(provider)
        return key is not None and key.strip() != ""
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value from environment
        """
        if key not in self._config_cache:
            env_key = f"XFIN_{key.upper()}"
            self._config_cache[key] = os.getenv(env_key, default)
        
        return self._config_cache[key]
    
    def get_rate_limit(self, provider: str) -> int:
        """Get rate limit for specific provider"""
        rate_limits = {
            'alpha_vantage': int(os.getenv('XFIN_ALPHA_VANTAGE_RATE_LIMIT', '12')),  # 5 calls per minute
            'polygon': int(os.getenv('XFIN_POLYGON_RATE_LIMIT', '12')),  # 5 calls per minute  
            'fmp': int(os.getenv('XFIN_FMP_RATE_LIMIT', '6')),  # 10 calls per second
            'iex_cloud': int(os.getenv('XFIN_IEX_RATE_LIMIT', '6'))
        }
        return rate_limits.get(provider, 12)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv('XFIN_ENVIRONMENT', 'development').lower() == 'production'
    
    def get_debug_level(self) -> str:
        """Get debug level configuration"""
        return os.getenv('XFIN_DEBUG_LEVEL', 'INFO').upper()
    
    def get_cache_ttl(self) -> int:
        """Get cache time-to-live in seconds"""
        return int(os.getenv('XFIN_CACHE_TTL', '3600'))  # 1 hour default
    
    def print_config_status(self):
        """Print current configuration status for debugging"""
        print("🔧 XFIN Library Configuration Status")
        print("=" * 50)
        
        api_keys = self.get_api_keys()
        for provider, key in api_keys.items():
            status = "✅ Available" if key else "❌ Not configured"
            print(f"{provider:12}: {status}")
        
        print(f"\nEnvironment: {os.getenv('XFIN_ENVIRONMENT', 'development')}")
        print(f"Debug Level: {self.get_debug_level()}")
        print(f"Cache TTL:   {self.get_cache_ttl()}s")
        print(f"Production:  {self.is_production()}")


# Global instance for the library to use
xfin_env = XFINEnvironment()


def get_xfin_env() -> XFINEnvironment:
    """Get the global XFIN environment instance"""
    return xfin_env


def setup_xfin_env(api_keys: Optional[Dict[str, str]] = None) -> XFINEnvironment:
    """
    Setup XFIN environment with optional API keys override
    
    Args:
        api_keys: Optional dictionary of API keys to override environment variables
    
    Returns:
        Configured XFINEnvironment instance
    """
    env = XFINEnvironment()
    
    if api_keys:
        # Override with provided keys
        env._api_keys = {
            'alpha_vantage': api_keys.get('alpha_vantage_key', env.get_api_key('alpha_vantage')),
            'polygon': api_keys.get('polygon_key', env.get_api_key('polygon')),
            'fmp': api_keys.get('fmp_key', env.get_api_key('fmp')),
            'iex_cloud': api_keys.get('iex_cloud_key', env.get_api_key('iex_cloud')),
            'openrouter': api_keys.get('openrouter_key', env.get_api_key('openrouter'))
        }
    
    return env


def require_api_key(provider: str) -> str:
    """
    Require an API key for a provider, raise error if not available
    
    Args:
        provider: Provider name (e.g., 'alpha_vantage', 'polygon')
        
    Returns:
        The API key
        
    Raises:
        ValueError: If API key is not configured
    """
    key = xfin_env.get_api_key(provider)
    if not key:
        available_providers = [p for p in xfin_env.get_api_keys().keys() if xfin_env.has_api_key(p)]
        raise ValueError(
            f"API key for '{provider}' is not configured. "
            f"Please set the {provider.upper()}_KEY environment variable. "
            f"Available providers: {', '.join(available_providers) if available_providers else 'none'}"
        )
    return key