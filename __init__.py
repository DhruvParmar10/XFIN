"""
XFIN - Enterprise Financial Risk Analysis Library
================================================

Professional-grade stress testing and risk analysis for banks, NBFCs, and financial institutions.
Ready for production deployment with zero configuration required.

🚀 **Immediate Start - No Setup Required:**
>>> import XFIN
>>> engine = XFIN.StressTestingEngine()
>>> results = engine.analyze_portfolio(portfolio_df, scenario="market_correction")

📊 **Stress Testing Features:**
- Portfolio stress testing with 15+ realistic scenarios
- Sector-wise risk analysis and visualization
- Regulatory compliance (RBI, Basel III ready)
- Real-time market data integration (optional)
- AI-powered risk explanations (optional)
- Enterprise-grade security and privacy

🔧 **Library Modes:**
- **Basic Mode**: Works immediately with no API keys required
- **Enhanced Mode**: Add API keys for live market data and AI insights
- **Enterprise Mode**: Full configuration for production deployment

📦 **Installation:**
```bash
pip install XFIN
```

🏦 **For Banks & NBFCs:**
```python
import XFIN

# Immediate usage - no configuration needed
stress_engine = XFIN.StressTestingEngine()

# Analyze portfolio against regulatory scenarios
results = stress_engine.analyze_portfolio(
    portfolio_data=your_df,
    scenario="recession_scenario"
)

# Generate compliance reports
report = stress_engine.generate_compliance_report(results)
```

⚡ **API Keys (Optional Enhancement):**
Set environment variables for enhanced functionality:
- ALPHA_VANTAGE_KEY: Live market data (free tier: 500 calls/day)
- OPENROUTER_API_KEY: AI explanations (free tier available)
"""

# Core modules
from .credit_risk import CreditRiskModule
from .stress_testing import StressTestingEngine
from .esg import ESGScoringEngine
from .explainer import PrivacyPreservingExplainer
from .compliance import ComplianceEngine
from .stress_plots import StressPlotGenerator
from .utils import get_llm_explanation

# Market Data and Configuration
from .market_data import MarketDataService
from .config import XFINConfig, setup_xfin
from .environment import xfin_env, get_xfin_env, setup_xfin_env, XFINEnvironment

# Application interfaces
from .app import launch_streamlit_app
from .stress_app import launch_stress_dashboard

# Package metadata
__version__ = "0.1.0"
__author__ = "Rishabh Bhangale & Dhruv Parmar"
__email__ = "dhruv.jparmar0@gmail.com"
__description__ = "Privacy-Preserving Explainable AI Library for Financial Services"
__url__ = "https://github.com/your-repo/xfin"
__license__ = "MIT"

# Public API - Primary imports for library users
__all__ = [
    # 🎯 MAIN LIBRARY INTERFACE
    'StressTestingEngine',        # Primary stress testing engine
    'StressPlotGenerator',        # Visualization tools
    
    # 🔧 CONFIGURATION & UTILITIES  
    'configure_api_keys',         # Simple API key setup
    'check_configuration',        # Verify setup
    'list_scenarios',            # Available stress scenarios
    
    # 📊 SUPPORTING MODULES (Advanced Users)
    'MarketDataService',         # Market data integration
    'get_llm_explanation',       # AI explanations
    
    # 🏦 ENTERPRISE FEATURES
    'CreditRiskModule',          # Credit risk analysis
    'ESGScoringEngine',          # ESG scoring
    'ComplianceEngine',          # Regulatory compliance
    
    # 🚀 CONVENIENCE FUNCTIONS
    'create_stress_analyzer',    # Quick setup
    'create_credit_analyzer',    # Quick credit analysis
    'create_esg_analyzer',       # Quick ESG analysis
    
    # 📱 APPLICATION INTERFACES
    'launch_stress_dashboard',   # Interactive dashboard
    'launch_streamlit_app',      # Full application
]

# 🚀 LIBRARY CONVENIENCE FUNCTIONS - Primary Interface for Banks/NBFCs

def configure_api_keys(alpha_vantage_key=None, openrouter_key=None, polygon_key=None):
    """
    🔧 Configure API keys for enhanced functionality
    
    Basic XFIN works without any keys. Add keys for enhanced features:
    - Alpha Vantage: Live market data (free 500 calls/day)
    - OpenRouter: AI explanations (free tier available)  
    - Polygon: Additional market data (free 5 calls/minute)
    
    Parameters:
    -----------
    alpha_vantage_key : str, optional
        Alpha Vantage API key for live market data
    openrouter_key : str, optional
        OpenRouter API key for AI-powered explanations
    polygon_key : str, optional
        Polygon.io API key for additional market data
        
    Example:
    --------
    >>> import XFIN
    >>> XFIN.configure_api_keys(
    ...     alpha_vantage_key="your_key_here",
    ...     openrouter_key="your_openrouter_key"  
    ... )
    >>> # Now enhanced features are available
    """
    import os
    
    if alpha_vantage_key:
        os.environ['ALPHA_VANTAGE_KEY'] = alpha_vantage_key
        print("✅ Alpha Vantage API key configured (live market data enabled)")
        
    if openrouter_key:
        os.environ['OPENROUTER_API_KEY'] = openrouter_key
        print("✅ OpenRouter API key configured (AI explanations enabled)")
        
    if polygon_key:
        os.environ['POLYGON_KEY'] = polygon_key
        print("✅ Polygon API key configured (additional market data enabled)")
        
    if not any([alpha_vantage_key, openrouter_key, polygon_key]):
        print("ℹ️  No API keys provided. XFIN will work in basic mode.")
        print("   Add API keys anytime for enhanced functionality.")


def check_configuration():
    """
    📊 Check current XFIN configuration and available features
    
    Returns:
    --------
    dict
        Configuration status and available features
        
    Example:
    --------
    >>> import XFIN
    >>> status = XFIN.check_configuration()
    >>> print(f"Live market data: {status['market_data_available']}")
    """
    from .environment import xfin_env
    
    status = {
        'market_data_available': xfin_env.has_api_key('alpha_vantage'),
        'ai_explanations_available': xfin_env.has_api_key('openrouter'),
        'additional_providers': xfin_env.has_api_key('polygon'),
        'basic_functionality': True,  # Always available
        'environment': xfin_env.get_config('environment', 'development'),
        'version': __version__
    }
    
    print("🔧 XFIN Library Configuration Status:")
    print("=" * 40)
    print(f"✅ Basic Functionality: Always Available")
    print(f"{'✅' if status['market_data_available'] else '❌'} Live Market Data: {'Available' if status['market_data_available'] else 'Configure ALPHA_VANTAGE_KEY'}")
    print(f"{'✅' if status['ai_explanations_available'] else '❌'} AI Explanations: {'Available' if status['ai_explanations_available'] else 'Configure OPENROUTER_API_KEY'}")
    print(f"{'✅' if status['additional_providers'] else '❌'} Enhanced Data: {'Available' if status['additional_providers'] else 'Configure POLYGON_KEY (optional)'}")
    print(f"📦 Version: {status['version']}")
    
    return status


def list_scenarios():
    """
    📋 List all available stress testing scenarios
    
    Returns:
    --------
    list
        Available scenario names
        
    Example:
    --------
    >>> import XFIN
    >>> scenarios = XFIN.list_scenarios()
    >>> print(f"Available scenarios: {scenarios}")
    """
    from .stress_testing import ScenarioGenerator
    
    generator = ScenarioGenerator()
    scenarios = generator.list_scenarios()
    
    print("📊 Available Stress Testing Scenarios:")
    print("=" * 40)
    for i, scenario_name in enumerate(scenarios, 1):
        scenario = generator.get_scenario(scenario_name)
        print(f"{i:2d}. {scenario['name']}")
        print(f"     {scenario['description']}")
        print()
    
    return scenarios


def create_stress_analyzer():
    """
    🎯 Create a ready-to-use stress testing analyzer
    
    This is the main entry point for banks and NBFCs.
    Works immediately without any configuration.
    
    Returns:
    --------
    StressTestingEngine
        Configured stress testing engine
        
    Example:
    --------
    >>> import XFIN
    >>> engine = XFIN.create_stress_analyzer()
    >>> results = engine.analyze_portfolio(portfolio_df, "market_correction")
    """
    return StressTestingEngine()


def create_credit_analyzer(model_interface):
    """
    🏦 Create a credit risk analyzer for loan assessments
    
    Parameters:
    -----------
    model_interface : object
        Trained model with predict/predict_proba methods
        
    Returns:
    --------
    CreditRiskModule
        Ready-to-use credit risk analyzer
        
    Example:
    --------
    >>> import XFIN
    >>> analyzer = XFIN.create_credit_analyzer(your_trained_model)
    >>> risk_assessment = analyzer.full_analysis(applicant_data)
    """
    return CreditRiskModule(model_interface, domain="credit")


def create_esg_analyzer():
    """
    🌱 Create an ESG scoring analyzer for sustainable finance
    
    Returns:
    --------
    ESGScoringEngine
        Ready-to-use ESG analyzer
        
    Example:
    --------
    >>> import XFIN
    >>> esg_engine = XFIN.create_esg_analyzer()
    >>> esg_scores = esg_engine.score_portfolio(portfolio_df)
    """
    return ESGScoringEngine()