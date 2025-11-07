"""
XFIN - Enterprise Financial Risk Analysis Library
==================================================

Professional-grade stress testing and ESG analysis for banks, NBFCs, and financial institutions.

Quick Start
-----------
>>> import xfin
>>> 
>>> # Stress Testing
>>> analyzer = xfin.StressAnalyzer()
>>> results = analyzer.analyze(portfolio_df, scenario="market_correction")
>>> 
>>> # ESG Scoring
>>> esg = xfin.ESGAnalyzer()
>>> scores = esg.score_portfolio(portfolio_df)

Installation
------------
pip install xfin

Documentation
-------------
https://github.com/DhruvParmar10/XFIN
"""

__version__ = "0.1.0"
__author__ = "Rishabh Bhangale & Dhruv Parmar"
__license__ = "MIT"

# =============================================================================
# PUBLIC API - Core Classes
# =============================================================================

from .stress_testing import StressTestingEngine as StressAnalyzer
from .esg import ESGScoringEngine as ESGAnalyzer
from .credit_risk import CreditRiskModule as CreditAnalyzer
from .compliance import ComplianceEngine as ComplianceChecker

# =============================================================================
# PUBLIC API - Visualization
# =============================================================================

from .stress_plots import StressPlotGenerator as StressPlots
from .esg_plots import ESGPlotGenerator as ESGPlots

# =============================================================================
# PUBLIC API - Configuration
# =============================================================================

from .config import get_config, reset_config, setup_xfin

# =============================================================================
# PUBLIC API - Utilities
# =============================================================================

def configure(alpha_vantage_key=None, openrouter_key=None, **kwargs):
    """
    Configure XFIN with API keys for enhanced functionality.
    
    XFIN works without configuration, but API keys enable:
    - Live market data (Alpha Vantage - free 500 calls/day)
    - AI explanations (OpenRouter - free tier available)
    - Additional data sources (optional)
    
    Parameters
    ----------
    alpha_vantage_key : str, optional
        Alpha Vantage API key for real-time market data
    openrouter_key : str, optional
        OpenRouter API key for AI-powered insights
    **kwargs : dict
        Additional configuration options:
        - polygon_key : Polygon.io API key
        - finnhub_key : Finnhub API key
        - cache_enabled : Enable/disable caching (default: True)
        
    Examples
    --------
    >>> import xfin
    >>> xfin.configure(alpha_vantage_key="YOUR_KEY")
    ✅ Alpha Vantage configured
    
    >>> xfin.configure(
    ...     alpha_vantage_key="KEY1",
    ...     openrouter_key="KEY2"
    ... )
    ✅ Alpha Vantage configured
    ✅ OpenRouter configured
    """
    import os
    
    configured = []
    
    if alpha_vantage_key:
        os.environ['ALPHA_VANTAGE_KEY'] = alpha_vantage_key
        configured.append('Alpha Vantage')
        
    if openrouter_key:
        os.environ['OPENROUTER_API_KEY'] = openrouter_key
        configured.append('OpenRouter')
        
    for key, value in kwargs.items():
        if key == 'polygon_key' and value:
            os.environ['POLYGON_KEY'] = value
            configured.append('Polygon')
        elif key == 'finnhub_key' and value:
            os.environ['FINNHUB_KEY'] = value
            configured.append('Finnhub')
    
    if configured:
        for service in configured:
            print(f"✅ {service} configured")
    else:
        print("ℹ️  No API keys provided. XFIN running in basic mode.")
        print("   Use xfin.configure(alpha_vantage_key='...') to enable live data")


def info():
    """
    Display XFIN library information and configuration status.
    
    Shows:
    - Library version
    - Available features
    - Configuration status
    - Installed components
    
    Examples
    --------
    >>> import xfin
    >>> xfin.info()
    XFIN v0.1.0 - Financial Risk Analysis Library
    ✅ Stress Testing: Available
    ✅ ESG Analysis: Available
    ...
    """
    config = get_config()
    
    print("=" * 60)
    print(f"XFIN v{__version__} - Financial Risk Analysis Library")
    print("=" * 60)
    print()
    print("📊 Core Features:")
    print("  ✅ Stress Testing")
    print("  ✅ ESG Scoring")
    print("  ✅ Credit Risk Analysis")
    print("  ✅ Regulatory Compliance")
    print()
    print("🔧 Configuration:")
    print(f"  {'✅' if config.has_api_key('alpha_vantage') else '⚪'} Live Market Data (Alpha Vantage)")
    print(f"  {'✅' if config.has_api_key('openrouter') else '⚪'} AI Explanations (OpenRouter)")
    print(f"  {'✅' if config.has_api_key('polygon') else '⚪'} Enhanced Data (Polygon)")
    print()
    print("📚 Documentation: https://github.com/DhruvParmar10/XFIN")
    print("=" * 60)


def list_scenarios():
    """
    List all available stress testing scenarios.
    
    Returns
    -------
    list
        List of scenario names
        
    Examples
    --------
    >>> import xfin
    >>> scenarios = xfin.list_scenarios()
    >>> print(scenarios)
    ['market_correction', 'recession_scenario', ...]
    """
    from .stress_testing import ScenarioGenerator
    generator = ScenarioGenerator()
    return generator.list_scenarios()


def show_scenarios():
    """
    Display detailed information about all stress testing scenarios.
    
    Shows scenario name, description, and key parameters for each
    available stress testing scenario.
    
    Examples
    --------
    >>> import xfin
    >>> xfin.show_scenarios()
    Available Stress Testing Scenarios:
    1. Market Correction
       10-15% equity drop, happens every 1-2 years
    ...
    """
    from .stress_testing import ScenarioGenerator
    
    generator = ScenarioGenerator()
    scenarios = generator.list_scenarios()
    
    print("📊 Available Stress Testing Scenarios:")
    print("=" * 60)
    
    for i, scenario_name in enumerate(scenarios, 1):
        scenario = generator.get_scenario(scenario_name)
        print(f"\n{i}. {scenario['name']}")
        print(f"   {scenario['description']}")
    
    print("\n" + "=" * 60)
    print(f"Total: {len(scenarios)} scenarios available")


# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # Version
    '__version__',
    
    # Core Analysis Classes
    'StressAnalyzer',
    'ESGAnalyzer',
    'CreditAnalyzer',
    'ComplianceChecker',
    
    # Visualization
    'StressPlots',
    'ESGPlots',
    
    # Configuration
    'configure',
    'info',
    'get_config',
    'reset_config',
    
    # Utilities
    'list_scenarios',
    'show_scenarios',
]

# =============================================================================
# BACKWARDS COMPATIBILITY (Deprecated - Use new names above)
# =============================================================================

def _deprecation_warning(old_name, new_name):
    """Show deprecation warning"""
    import warnings
    warnings.warn(
        f"{old_name} is deprecated and will be removed in v1.0.0. "
        f"Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Legacy imports with deprecation warnings
class StressTestingEngine(StressAnalyzer):
    """Deprecated: Use StressAnalyzer instead"""
    def __init__(self, *args, **kwargs):
        _deprecation_warning('StressTestingEngine', 'StressAnalyzer')
        super().__init__(*args, **kwargs)

class ESGScoringEngine(ESGAnalyzer):
    """Deprecated: Use ESGAnalyzer instead"""
    def __init__(self, *args, **kwargs):
        _deprecation_warning('ESGScoringEngine', 'ESGAnalyzer')
        super().__init__(*args, **kwargs)

class CreditRiskModule(CreditAnalyzer):
    """Deprecated: Use CreditAnalyzer instead"""
    def __init__(self, *args, **kwargs):
        _deprecation_warning('CreditRiskModule', 'CreditAnalyzer')
        super().__init__(*args, **kwargs)

# =============================================================================
# ADVANCED FEATURES (For power users - not in main __all__)
# =============================================================================

# These are available but not advertised in main API
from .market_data import MarketDataService
from .utils import get_llm_explanation
from .app import launch_streamlit_app
from .stress_app import launch_stress_dashboard

# =============================================================================
# LIBRARY INITIALIZATION
# =============================================================================

# Auto-check configuration on import (silent)
_config = get_config()

# Show friendly message on first import
import sys
if 'xfin' not in sys.modules or len(sys.modules) == 1:
    pass  # Silent import for professional library