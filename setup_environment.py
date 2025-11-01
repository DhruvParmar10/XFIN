#!/usr/bin/env python3
"""
XFIN Environment Setup Utility
==============================

Quick setup script to help users configure their XFIN environment with API keys.
This script creates a properly formatted .env file with all necessary environment variables.

Usage:
------
python setup_environment.py

Or import and use programmatically:
>>> from XFIN.setup_environment import setup_interactive
>>> setup_interactive()
"""

import os
from pathlib import Path
from typing import Dict, Optional


def create_env_file(api_keys: Dict[str, str], overwrite: bool = False) -> bool:
    """
    Create a .env file with the provided API keys
    
    Args:
        api_keys: Dictionary of API provider keys
        overwrite: Whether to overwrite existing .env file
        
    Returns:
        True if .env file was created successfully
    """
    env_path = Path('.env')
    
    if env_path.exists() and not overwrite:
        print(f"⚠️  .env file already exists at {env_path.absolute()}")
        response = input("Overwrite existing .env file? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    env_content = [
        "# XFIN Library Environment Configuration",
        "# Generated automatically - add your API keys below",
        "",
        "# Market Data API Keys (all optional - get free keys from providers)",
        ""
    ]
    
    # Add API key configurations
    for provider, key in api_keys.items():
        env_key = f"{provider.upper()}_KEY"
        if key and key.strip():
            env_content.append(f"{env_key}={key}")
        else:
            env_content.append(f"# {env_key}=your_key_here")
    
    env_content.extend([
        "",
        "# XFIN Configuration Options",
        "# XFIN_ENVIRONMENT=development",
        "# XFIN_DEBUG_LEVEL=INFO", 
        "# XFIN_CACHE_TTL=3600",
        "",
        "# Rate Limiting (requests per minute)",
        "# XFIN_ALPHA_VANTAGE_RATE_LIMIT=12",
        "# XFIN_POLYGON_RATE_LIMIT=12",
        "# XFIN_FMP_RATE_LIMIT=6",
        "# XFIN_IEX_RATE_LIMIT=6",
        ""
    ])
    
    try:
        with open(env_path, 'w') as f:
            f.write('\n'.join(env_content))
        
        print(f"✅ Environment file created at {env_path.absolute()}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False


def get_free_api_keys_info() -> Dict[str, Dict[str, str]]:
    """Return information about free API key sources"""
    return {
        'alpha_vantage': {
            'name': 'Alpha Vantage',
            'free_tier': '500 calls/day',
            'url': 'https://www.alphavantage.co/support/#api-key',
            'description': 'Global stock market data, indices, forex'
        },
        'polygon': {
            'name': 'Polygon.io',
            'free_tier': '5 calls/minute', 
            'url': 'https://polygon.io/',
            'description': 'US stock market data, real-time quotes'
        },
        'iex_cloud': {
            'name': 'IEX Cloud',
            'free_tier': '50,000 messages/month',
            'url': 'https://iexcloud.io/', 
            'description': 'US equities, ETFs, mutual funds'
        },
        'fmp': {
            'name': 'Financial Modeling Prep',
            'free_tier': '250 calls/day',
            'url': 'https://financialmodelingprep.com/developer/docs',
            'description': 'Financial statements, ratios, metrics'
        },
        'openrouter': {
            'name': 'OpenRouter',
            'free_tier': 'Credits available',
            'url': 'https://openrouter.ai/',
            'description': 'AI/LLM API access for explanations'
        }
    }


def setup_interactive():
    """Interactive setup for XFIN environment configuration"""
    
    print("🚀 XFIN Library Environment Setup")
    print("=" * 50)
    print()
    print("This utility helps you configure API keys for market data providers.")
    print("All API keys are OPTIONAL - XFIN works with any combination of providers.")
    print()
    
    # Show available providers
    providers_info = get_free_api_keys_info()
    
    print("📊 Available Market Data Providers:")
    print()
    for provider, info in providers_info.items():
        print(f"  • {info['name']}")
        print(f"    Free Tier: {info['free_tier']}")
        print(f"    URL: {info['url']}")
        print(f"    Data: {info['description']}")
        print()
    
    print("💡 Tip: You can start with just one provider and add more later!")
    print()
    
    # Collect API keys
    api_keys = {}
    
    for provider, info in providers_info.items():
        print(f"🔑 {info['name']} API Key")
        print(f"   Free tier: {info['free_tier']}")
        print(f"   Get key: {info['url']}")
        
        key = input(f"   Enter your {info['name']} API key (or press Enter to skip): ").strip()
        
        if key:
            api_keys[provider] = key
            print(f"   ✅ {info['name']} key added")
        else:
            print(f"   ⏭️  Skipped {info['name']}")
        print()
    
    if not api_keys:
        print("⚠️  No API keys provided. XFIN will use limited functionality.")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    # Create .env file
    success = create_env_file(api_keys, overwrite=True)
    
    if success:
        print()
        print("🎉 Setup Complete!")
        print() 
        print("Next steps:")
        print("1. Test your configuration:")
        print("   >>> from XFIN import xfin_env")
        print("   >>> xfin_env.print_config_status()")
        print()
        print("2. Start using XFIN:")
        print("   >>> from XFIN import StressTestingEngine")
        print("   >>> engine = StressTestingEngine()")
        print()
        if not api_keys:
            print("💡 Add API keys to .env file later for full functionality")
    
    return success


def setup_minimal():
    """Create a minimal .env template"""
    api_keys = {}  # Empty - just create template
    return create_env_file(api_keys)


def check_environment() -> Dict[str, bool]:
    """Check current environment configuration"""
    from .environment import xfin_env
    
    status = {
        'env_file_exists': Path('.env').exists(),
        'api_keys': {}
    }
    
    # Check each provider
    for provider in ['alpha_vantage', 'polygon', 'iex_cloud', 'fmp', 'openrouter']:
        status['api_keys'][provider] = xfin_env.has_api_key(provider)
    
    return status


def print_environment_status():
    """Print current environment configuration status"""
    from .environment import xfin_env
    
    print("🔧 XFIN Environment Status")
    print("=" * 50)
    
    # Check .env file
    env_path = Path('.env')
    if env_path.exists():
        print(f"✅ Environment file: {env_path.absolute()}")
    else:
        print("❌ No .env file found")
        print("   Run setup_interactive() to create one")
    
    print()
    xfin_env.print_config_status()


if __name__ == "__main__":
    # Run interactive setup if called directly
    setup_interactive()