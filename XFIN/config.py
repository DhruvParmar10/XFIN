"""
XFIN Configuration Management
============================

Unified configuration system for XFIN library with support for:
- API key management (demo, user, environment)
- Custom risk thresholds 
- Provider preferences
- Enterprise configurations
- Environment variable loading
- Rate limiting
- Caching configuration

Authors: XFIN Development Team
License: MIT
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class XFINConfig:
    """
    Main configuration class for XFIN library
    
    Provides flexible configuration management for:
    - Market data API keys
    - Risk calculation thresholds
    - Provider preferences
    - Caching settings
    - Enterprise customizations
    """
    
    def __init__(self, config_dict: Dict = None):
        """
        Initialize XFIN configuration
        
        Parameters:
        -----------
        config_dict : Dict, optional
            Dictionary of configuration options
        """
        
        # Default configuration
        self.config = {
            # API Keys - FREE options marked with 💚
            'api_keys': {
                'financial_modeling_prep_key': None,  # 💚 FREE: 250 calls/day
                'alpha_vantage_key': None,            # 💚 FREE: 500 calls/day
                'iex_cloud_key': None,                # 💚 FREE: 50K messages/month
                'openrouter_key': None,               # 💚 FREE tier available
                'polygon_key': None,                  # 💚 FREE: 5 calls/minute
                # Premium options (paid)
                'bloomberg_key': None,
                'refinitiv_key': None
            },
            
            # Market Data Settings
            'market_data': {
                'cache_enabled': True,
                'cache_duration_hours': 4,
                'max_cache_entries': 1000,
                'preferred_providers': ['alpha_vantage', 'nse', 'yahoo_finance'],
                'rate_limit_respect': True
            },
            
            # Risk Calculation Thresholds
            'risk_thresholds': {
                # Market Cap Classification (in crores INR)
                'large_cap_min': 20000,
                'mid_cap_min': 5000,
                
                # Concentration Risk
                'concentration_warning': 10,  # %
                'concentration_critical': 15,  # %
                
                # Sector Limits
                'max_sector_exposure': 25,  # %
                'sector_concentration_warning': 20,  # %
                
                # VaR Settings
                'var_confidence_level': 95,  # %
                'var_time_horizon_days': 252,  # 1 year
                
                # Stress Testing
                'stress_scenarios': ['market_correction', 'recession_scenario', 'inflation_spike'],
                'custom_scenario_enabled': True
            },
            
            # Visualization Settings
            'charts': {
                'color_palette': 'professional',  # 'professional', 'colorful', 'minimal'
                'chart_style': 'seaborn',
                'figure_dpi': 100,
                'default_figure_size': (12, 8)
            },
            
            # LLM Integration
            'llm': {
                'default_model': 'x-ai/grok-code-fast-1',
                'max_tokens': 500,
                'temperature': 0.3,
                'enable_caching': True
            },
            
            # Enterprise Features
            'enterprise': {
                'batch_processing_enabled': False,
                'audit_logging': False,
                'compliance_reporting': False,
                'custom_risk_models': False
            }
        }
        
        # Override with user-provided config
        if config_dict:
            self._deep_update(self.config, config_dict)
        
        # Load from environment variables
        self._load_from_environment()
        
        # Load from config file if exists
        self._load_from_config_file()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update nested dictionaries"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # API Keys
        env_mappings = {
            'XFIN_ALPHA_VANTAGE_KEY': ['api_keys', 'alpha_vantage_key'],
            'XFIN_OPENROUTER_KEY': ['api_keys', 'openrouter_key'],
            'XFIN_BLOOMBERG_KEY': ['api_keys', 'bloomberg_key'],
            'XFIN_REFINITIV_KEY': ['api_keys', 'refinitiv_key'],
            
            # Risk Thresholds
            'XFIN_LARGE_CAP_MIN': ['risk_thresholds', 'large_cap_min'],
            'XFIN_MID_CAP_MIN': ['risk_thresholds', 'mid_cap_min'],
            'XFIN_CONCENTRATION_WARNING': ['risk_thresholds', 'concentration_warning'],
            'XFIN_VAR_CONFIDENCE': ['risk_thresholds', 'var_confidence_level'],
            
            # Cache Settings
            'XFIN_CACHE_DURATION': ['market_data', 'cache_duration_hours'],
            'XFIN_MAX_CACHE_ENTRIES': ['market_data', 'max_cache_entries']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                try:
                    # Try to convert to appropriate type
                    if config_path[-1] in ['large_cap_min', 'mid_cap_min', 'concentration_warning', 
                                          'var_confidence_level', 'cache_duration_hours', 'max_cache_entries']:
                        env_value = float(env_value)
                    
                    self._set_nested_value(self.config, config_path, env_value)
                except (ValueError, TypeError):
                    pass  # Skip invalid values
    
    def _load_from_config_file(self):
        """Load configuration from ~/.xfin/config.json if it exists"""
        config_path = Path.home() / '.xfin' / 'config.json'
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                self._deep_update(self.config, file_config)
            except (json.JSONDecodeError, IOError):
                pass  # Skip invalid config files
    
    def _set_nested_value(self, dictionary: Dict, keys: List[str], value: Any):
        """Set a value in a nested dictionary using a list of keys"""
        for key in keys[:-1]:
            dictionary = dictionary.setdefault(key, {})
        dictionary[keys[-1]] = value
    
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a provider"""
        self.config['api_keys'][f'{provider}_key'] = api_key
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider"""
        key_name = f'{provider}_key'
        # First check config dict
        api_key = self.config['api_keys'].get(key_name)
        
        # Fallback to environment variables
        if not api_key:
            env_var_map = {
                'alpha_vantage': 'ALPHA_VANTAGE_KEY',
                'polygon': 'POLYGON_KEY',
                'fmp': 'FMP_KEY',
                'financial_modeling_prep': 'FMP_KEY',
                'iex_cloud': 'IEX_CLOUD_KEY',
                'openrouter': 'OPENROUTER_API_KEY',
                'bloomberg': 'BLOOMBERG_KEY',
                'refinitiv': 'REFINITIV_KEY'
            }
            env_var = env_var_map.get(provider)
            if env_var:
                api_key = os.getenv(env_var)
        
        return api_key
    
    def has_api_key(self, provider: str) -> bool:
        """Check if API key is available for provider"""
        key = self.get_api_key(provider)
        return key is not None and key.strip() != ""
    
    def get_rate_limit(self, provider: str) -> int:
        """Get rate limit for specific provider (requests per second)"""
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
        cache_hours = self.config['market_data'].get('cache_duration_hours', 4)
        return int(os.getenv('XFIN_CACHE_TTL', str(cache_hours * 3600)))
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider"""
        return self.config['api_keys'].get(f'{provider}_key')
    
    def set_risk_threshold(self, threshold_name: str, value: float):
        """Set a risk threshold value"""
        if threshold_name in self.config['risk_thresholds']:
            self.config['risk_thresholds'][threshold_name] = value
    
    def get_risk_threshold(self, threshold_name: str) -> Optional[float]:
        """Get a risk threshold value"""
        return self.config['risk_thresholds'].get(threshold_name)
    
    def enable_enterprise_features(self, features: List[str] = None):
        """Enable enterprise features"""
        if features is None:
            features = ['batch_processing_enabled', 'audit_logging', 'compliance_reporting']
        
        for feature in features:
            if feature in self.config['enterprise']:
                self.config['enterprise'][feature] = True
    
    def get_market_data_config(self) -> Dict:
        """Get market data configuration for MarketDataService"""
        return {
            **self.config['api_keys'],
            **self.config['market_data']
        }
    
    def get_chart_config(self) -> Dict:
        """Get chart configuration for StressPlotGenerator"""
        return self.config['charts']
    
    def get_llm_config(self) -> Dict:
        """Get LLM configuration"""
        return {
            **self.config['llm'],
            'api_key': self.config['api_keys']['openrouter_key']
        }
    
    def save_to_file(self, config_path: str = None):
        """
        Save current configuration to file
        
        Parameters:
        -----------
        config_path : str, optional
            Path to save config file. Defaults to ~/.xfin/config.json
        """
        if config_path is None:
            config_dir = Path.home() / '.xfin'
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / 'config.json'
        
        # Remove sensitive keys before saving
        safe_config = self._get_safe_config()
        
        with open(config_path, 'w') as f:
            json.dump(safe_config, f, indent=2)
    
    def _get_safe_config(self) -> Dict:
        """Get configuration with sensitive data removed"""
        safe_config = json.loads(json.dumps(self.config))  # Deep copy
        
        # Remove API keys for security
        for key in safe_config['api_keys']:
            safe_config['api_keys'][key] = None
        
        return safe_config
    
    def validate_config(self) -> Dict[str, List[str]]:
        """
        Validate current configuration
        
        Returns:
        --------
        Dict with 'warnings' and 'errors' lists
        """
        warnings = []
        errors = []
        
        # Check API keys
        if not any(self.config['api_keys'].values()):
            warnings.append("No API keys configured - using demo/fallback providers only")
        
        # Check risk thresholds
        if self.config['risk_thresholds']['large_cap_min'] <= self.config['risk_thresholds']['mid_cap_min']:
            errors.append("Large cap minimum must be greater than mid cap minimum")
        
        if self.config['risk_thresholds']['concentration_warning'] >= self.config['risk_thresholds']['concentration_critical']:
            errors.append("Concentration warning threshold must be less than critical threshold")
        
        # Check VaR settings
        if not 90 <= self.config['risk_thresholds']['var_confidence_level'] <= 99:
            warnings.append("VaR confidence level outside typical range (90-99%)")
        
        return {'warnings': warnings, 'errors': errors}
    
    def print_summary(self):
        """Print configuration summary"""
        print("🔧 XFIN Configuration Summary")
        print("=" * 60)
        
        # API Keys
        print("\n📊 Market Data APIs:")
        api_providers = {
            'alpha_vantage': 'Alpha Vantage',
            'polygon': 'Polygon.io',
            'fmp': 'Financial Modeling Prep',
            'iex_cloud': 'IEX Cloud',
            'openrouter': 'OpenRouter (LLM)',
            'bloomberg': 'Bloomberg Terminal',
            'refinitiv': 'Refinitiv'
        }
        
        for provider_key, provider_name in api_providers.items():
            has_key = self.has_api_key(provider_key)
            status = "✅ Configured" if has_key else "❌ Not configured"
            print(f"  {provider_name:25}: {status}")
        
        # Risk Thresholds
        print(f"\n⚠️  Risk Thresholds:")
        print(f"  Large Cap: ₹{self.config['risk_thresholds']['large_cap_min']:,} crores+")
        print(f"  Mid Cap: ₹{self.config['risk_thresholds']['mid_cap_min']:,} crores+")
        print(f"  Concentration Warning: {self.config['risk_thresholds']['concentration_warning']}%")
        print(f"  VaR Confidence: {self.config['risk_thresholds']['var_confidence_level']}%")
        
        # Cache Settings
        print(f"\n💾 Cache Settings:")
        print(f"  Enabled: {self.config['market_data']['cache_enabled']}")
        print(f"  Duration: {self.config['market_data']['cache_duration_hours']} hours")
        print(f"  Max Entries: {self.config['market_data']['max_cache_entries']:,}")
        
        # Environment
        print(f"\n🌐 Environment:")
        print(f"  Mode: {os.getenv('XFIN_ENVIRONMENT', 'development')}")
        print(f"  Debug Level: {self.get_debug_level()}")
        print(f"  Production: {self.is_production()}")
        
        # Validation
        validation = self.validate_config()
        if validation['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        if validation['errors']:
            print(f"\n❌ Errors:")
            for error in validation['errors']:
                print(f"  • {error}")


# ============================================================================
# Global Instance & Convenience Functions
# ============================================================================

# Global configuration instance
_global_config = None


def get_config() -> XFINConfig:
    """Get or create the global XFIN configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = XFINConfig()
    return _global_config


def reset_config():
    """Reset the global configuration to default"""
    global _global_config
    _global_config = None


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
    config = get_config()
    key = config.get_api_key(provider)
    if not key:
        raise ValueError(
            f"API key for '{provider}' is not configured. "
            f"Please set the {provider.upper()}_KEY environment variable or "
            f"configure it using config.set_api_key('{provider}', 'your_key')"
        )
    return key


def print_summary(self):
        """Print configuration summary"""
        print("🔧 XFIN Configuration Summary")
        print("=" * 40)
        
        # API Keys
        print("\n📊 Market Data APIs:")
        for provider, key in self.config['api_keys'].items():
            status = "✅ Configured" if key else "❌ Not configured"
            print(f"  {provider}: {status}")
        
        # Risk Thresholds
        print(f"\n⚠️ Risk Thresholds:")
        print(f"  Large Cap: ₹{self.config['risk_thresholds']['large_cap_min']:,} crores+")
        print(f"  Mid Cap: ₹{self.config['risk_thresholds']['mid_cap_min']:,} crores+")
        print(f"  Concentration Warning: {self.config['risk_thresholds']['concentration_warning']}%")
        print(f"  VaR Confidence: {self.config['risk_thresholds']['var_confidence_level']}%")
        
        # Cache Settings
        print(f"\n💾 Cache Settings:")
        print(f"  Enabled: {self.config['market_data']['cache_enabled']}")
        print(f"  Duration: {self.config['market_data']['cache_duration_hours']} hours")
        print(f"  Max Entries: {self.config['market_data']['max_cache_entries']:,}")
        
        # Validation
        validation = self.validate_config()
        if validation['warnings']:
            print(f"\n⚠️ Warnings:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        if validation['errors']:
            print(f"\n❌ Errors:")
            for error in validation['errors']:
                print(f"  • {error}")


# Convenience Functions for Library Users

def setup_xfin(api_keys: Dict = None, 
               custom_thresholds: Dict = None, 
               enterprise_features: List[str] = None) -> XFINConfig:
    """
    Quick setup function for XFIN library
    
    Parameters:
    -----------
    api_keys : Dict, optional
        API keys: {'alpha_vantage_key': 'key', 'openrouter_key': 'key'}
    custom_thresholds : Dict, optional
        Custom risk thresholds: {'large_cap_min': 30000, 'concentration_warning': 8}
    enterprise_features : List[str], optional
        Enterprise features to enable: ['batch_processing_enabled', 'audit_logging']
    
    Returns:
    --------
    XFINConfig instance
    
    Example:
    --------
    >>> import XFIN
    >>> config = XFIN.setup_xfin({
    ...     'alpha_vantage_key': 'your_key_here',
    ...     'openrouter_key': 'your_llm_key'
    ... })
    >>> engine = XFIN.StressTestingEngine(config)
    """
    
    config_dict = {}
    
    if api_keys:
        config_dict['api_keys'] = api_keys
    
    if custom_thresholds:
        config_dict['risk_thresholds'] = custom_thresholds
    
    config = XFINConfig(config_dict)
    
    if enterprise_features:
        config.enable_enterprise_features(enterprise_features)
    
    return config


def setup_demo_config() -> XFINConfig:
    """
    Setup demo configuration with limited functionality
    
    Returns:
    --------
    XFINConfig with demo settings
    """
    return XFINConfig({
        'market_data': {
            'preferred_providers': ['nse', 'yahoo_finance'],  # Free providers first
            'cache_duration_hours': 1  # Shorter cache for demo
        }
    })


def setup_banking_config(bank_name: str = "Generic Bank", 
                        custom_thresholds: Dict = None) -> XFINConfig:
    """
    Setup configuration optimized for banking institutions
    
    Parameters:
    -----------
    bank_name : str
        Name of the banking institution
    custom_thresholds : Dict, optional
        Bank-specific risk thresholds
    
    Returns:
    --------
    XFINConfig optimized for banking use
    """
    
    # Conservative banking defaults
    banking_thresholds = {
        'large_cap_min': 25000,  # More conservative
        'mid_cap_min': 7500,
        'concentration_warning': 8,  # Lower concentration limits
        'concentration_critical': 12,
        'max_sector_exposure': 20,
        'var_confidence_level': 99,  # Higher confidence for banks
        'stress_scenarios': ['market_correction', 'recession_scenario', 'banking_crisis']
    }
    
    if custom_thresholds:
        banking_thresholds.update(custom_thresholds)
    
    config = XFINConfig({
        'risk_thresholds': banking_thresholds,
        'enterprise': {
            'batch_processing_enabled': True,
            'audit_logging': True,
            'compliance_reporting': True,
            'custom_risk_models': True
        },
        'market_data': {
            'preferred_providers': ['bloomberg', 'alpha_vantage', 'refinitiv', 'nse'],
            'cache_duration_hours': 6  # Longer cache for stability
        }
    })
    
    return config


def load_config_from_file(file_path: str) -> XFINConfig:
    """
    Load configuration from JSON file
    
    Parameters:
    -----------
    file_path : str
        Path to configuration file
    
    Returns:
    --------
    XFINConfig instance
    """
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    
    return XFINConfig(config_dict)


def create_config_template(file_path: str = "xfin_config_template.json"):
    """
    Create a configuration template file
    
    Parameters:
    -----------
    file_path : str
        Path where to save the template
    """
    template = XFINConfig()._get_safe_config()
    
    # Add comments as special keys
    template['_comments'] = {
        'api_keys': 'Get free API keys from respective providers',
        'risk_thresholds': 'Adjust based on your risk management policies',
        'market_data': 'Configure caching and provider preferences',
        'enterprise': 'Enable for advanced features (may require license)'
    }
    
    with open(file_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"📋 Configuration template saved to: {file_path}")
    print("💡 Edit the template and load with: XFIN.load_config_from_file('{file_path}')")