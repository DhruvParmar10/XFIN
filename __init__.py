"""
XFIN - Privacy-Preserving Explainable AI Library for Financial Services

A comprehensive library providing explainable AI solutions for:
- Credit Risk Assessment with SHAP/LIME explanations
- Portfolio Stress Testing with scenario analysis
- ESG Analysis and sustainable finance integration
- Regulatory compliance (ECOA, GDPR, SFDR)
"""

from .credit_risk import CreditRiskModule
from .stress_testing import StressTestingModule, StressTestingEngine
from .explainer import PrivacyPreservingExplainer
from .compliance import ComplianceEngine
from .stress_plots import StressPlotGenerator
from .utils import get_llm_explanation
from .app import launch_streamlit_app
from .stress_app import launch_stress_dashboard

__version__ = "0.1.0"
__author__ = "Rishabh Bhangle & Dhruv Parmar"
__email__ = "dhruv.jparmar0@gmail.com"

__all__ = [
    'CreditRiskModule',
    'StressTestingModule', 
    'StressTestingEngine',
    'PrivacyPreservingExplainer',
    'ComplianceEngine',
    'StressPlotGenerator',
    'get_llm_explanation',
    'launch_streamlit_app',
    'launch_stress_dashboard'
]