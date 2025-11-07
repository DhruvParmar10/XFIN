# XFIN: Comprehensive Financial Risk Analysis & XAI Library

[![PyPI version](https://badge.fury.io/py/xfin-xai.svg)](https://badge.fury.io/py/xfin-xai)
[![Documentation Status](https://readthedocs.org/projects/xfin-xai/badge/?version=latest)](https://xfin-xai.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

XFIN is a comprehensive, open-source Python library for **Financial Risk Analysis**, **Explainable AI (XAI)**, and **Sustainable Investment Analysis**. Built for banks, financial institutions, portfolio managers, and individual investors, XFIN provides professional-grade tools for credit risk assessment, portfolio stress testing, and ESG (Environmental, Social, Governance) analysis.

## 🚀 Why XFIN?

XFIN brings together three critical aspects of modern financial analysis:

- **🔒 Privacy-Preserving XAI**: Explain black-box models without exposing proprietary internals
- **🧪 Professional Stress Testing**: Multi-scenario portfolio resilience analysis
- **🌱 Sustainable Investment**: Comprehensive ESG scoring and SFDR compliance

All in one unified, easy-to-use library with a clean pandas-like API.

## 📦 Installation

```bash
pip install xfin-xai
```

## 🎯 Three Powerful Modules

### 1. 💳 Credit Risk & XAI Module

Generate transparent explanations for credit decisions while maintaining model privacy.

**Key Features:**
- SHAP/LIME-based explanations
- Adverse action notices (ECOA/FCRA compliant)
- Counterfactual recommendations
- LLM-powered natural language explanations
- Regulatory audit trails

**Quick Start:**
```python
from XFIN import CreditRiskModule
import pandas as pd

# Your black-box model (kept private)
class BankModel:
    def predict(self, X): return model.predict(X)
    def predict_proba(self, X): return model.predict_proba(X)

# Generate explanation
explainer = CreditRiskModule(BankModel(), domain="credit_risk")
explanation = explainer.explain_prediction(application_data)
recommendations = explainer.generate_recommendations(application_data)
```

**[📖 Full Credit Risk Documentation →](README_CREDIT_RISK.md)**

---

### 2. 🧪 Stress Testing Module

Analyze portfolio resilience under market stress scenarios with Basel III compliance.

**Key Features:**
- 10+ pre-configured stress scenarios
- Value at Risk (VaR) calculation
- Sector-level impact analysis
- HHI diversification metrics
- Custom scenario builder
- Basel III/RBI compliance

**Quick Start:**
```python
from XFIN.stress_testing import StressTestingEngine, ScenarioGenerator
import pandas as pd

# Your portfolio
portfolio = pd.DataFrame({
    'Ticker': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
    'Quantity': [500, 300, 1000],
    'Current_Price': [2450, 3600, 1650],
    'Sector': ['Energy', 'IT', 'Financials']
})

# Run stress test
engine = StressTestingEngine()
scenario_gen = ScenarioGenerator()
scenario = scenario_gen.get_scenario("market_correction")

# Calculate impact
portfolio['Stressed_Value'] = portfolio['Current_Value'] * (1 + portfolio['Sector_Impact'])
total_impact = (portfolio['Stressed_Value'].sum() - portfolio['Current_Value'].sum()) / portfolio['Current_Value'].sum() * 100

print(f"Market Correction Impact: {total_impact:.2f}%")
```

**[📖 Full Stress Testing Documentation →](README_STRESS_TESTING.md)**

---

### 3. 🌱 ESG Scoring Module

Evaluate sustainability and ESG performance of securities and portfolios.

**Key Features:**
- Multi-source ESG data (Yahoo, Finnhub, RapidAPI)
- ML-powered predictions with SHAP explainability
- SFDR Article 6/8/9 classification
- Sector benchmarking
- Portfolio sustainability metrics
- ESG risk assessment

**Quick Start:**
```python
from XFIN.esg import ESGScoringEngine
import pandas as pd

# Initialize ESG engine
esg_engine = ESGScoringEngine()

# Score a security
score = esg_engine.score_security(
    ticker='RELIANCE.NS',
    company_name='Reliance Industries',
    sector='Energy'
)

print(f"ESG Score: {score['overall_score']:.2f}/100")
print(f"E: {score['environmental_score']:.2f}")
print(f"S: {score['social_score']:.2f}")
print(f"G: {score['governance_score']:.2f}")

# Score entire portfolio
portfolio_score = esg_engine.score_portfolio(portfolio)
print(f"Portfolio ESG: {portfolio_score['overall_score']:.2f}/100")
print(f"Rating: {portfolio_score['rating']}")
```

**[📖 Full ESG Documentation →](README_ESG.md)**

---

## 🎨 Interactive Web Dashboards

Launch professional web dashboards for each module:

```bash
# Credit Risk Dashboard
xfin credit

# Stress Testing Dashboard
xfin stress

# ESG Analysis Dashboard
xfin esg

# Unified Dashboard (All modules)
streamlit run unified_dashboard.py
```

## 📊 Complete Analysis Example

Combine all three modules for comprehensive portfolio analysis:

```python
import XFIN as xfin
import pandas as pd

# Portfolio data
portfolio = pd.DataFrame({
    'Ticker': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS'],
    'Quantity': [500, 300, 1000, 400],
    'Current_Price': [2450, 3600, 1650, 1500],
    'Sector': ['Energy', 'IT', 'Financials', 'IT']
})

# 1. Credit Risk (for loan applications)
credit_explainer = xfin.CreditAnalyzer(model, domain="credit_risk")
credit_result = credit_explainer.explain_prediction(application_data)

# 2. Stress Testing
stress_engine = xfin.StressAnalyzer()
scenario = xfin.ScenarioGenerator().get_scenario("recession")
stress_impact = stress_engine.analyze_portfolio(portfolio, scenario)

# 3. ESG Analysis
esg_engine = xfin.ESGAnalyzer()
esg_score = esg_engine.score_portfolio(portfolio)

# 4. Compliance Check
compliance = xfin.ComplianceChecker()
compliance_report = compliance.check_all(
    credit=credit_result,
    stress=stress_impact,
    esg=esg_score
)

print(f"""
📊 Complete Portfolio Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Portfolio Value: ₹{portfolio['Current_Value'].sum():,.2f}
🧪 Stress Test Impact: {stress_impact['impact_pct']:.2f}%
🌱 ESG Score: {esg_score['overall_score']:.2f}/100
✅ Compliance: {'PASS' if compliance_report['compliant'] else 'FAIL'}
""")
```

## 🎓 Use Cases

### For Financial Institutions
- **Credit Underwriting**: Generate compliant explanations for loan decisions
- **Risk Management**: Stress test loan portfolios and investment books
- **ESG Compliance**: Meet SFDR and sustainable finance regulations
- **Regulatory Reporting**: Basel III, RBI, ECOA, GDPR compliance

### For Portfolio Managers
- **Risk Assessment**: Evaluate portfolio stress resilience
- **ESG Integration**: Build Article 8/9 compliant sustainable funds
- **Client Reporting**: Professional risk and sustainability reports
- **Strategy Optimization**: Test portfolio under multiple scenarios

### For Individual Investors
- **Portfolio Analysis**: Understand your investment risk and sustainability
- **Investment Decisions**: Make informed choices based on ESG scores
- **Risk Awareness**: Know your portfolio's downside risk (VaR)
- **Sustainable Investing**: Align investments with values

### For ESG Analysts
- **Company Evaluation**: Comprehensive ESG scoring
- **Sector Benchmarking**: Compare ESG performance
- **Impact Measurement**: Track sustainability metrics
- **Improvement Identification**: Find ESG enhancement opportunities

## 🔧 Requirements

- **Python**: 3.9 or higher
- **Core Dependencies**: 
  - Data: `pandas`, `numpy`, `scipy`
  - ML: `scikit-learn`, `lightgbm`, `shap`
  - Finance: `yfinance`, `pandas-datareader`
  - Visualization: `matplotlib`, `plotly`, `streamlit`
  - XAI: `lime`, `shap`

See [`requirements.txt`](../requirements.txt) for complete list.

## 📖 Documentation

### Main Documentation
- 📚 [Full API Reference](https://xfin-xai.readthedocs.io/en/latest/)
- 🎓 [Tutorials & Examples](../examples/README.md)
- 🗺️ [Roadmap](https://xfin-xai.readthedocs.io/en/latest/roadmap.html)

### Module-Specific Documentation
- 💳 [Credit Risk Module](README_CREDIT_RISK.md)
- 🧪 [Stress Testing Module](README_STRESS_TESTING.md)
- 🌱 [ESG Module](README_ESG.md)

### Guides
- 🚀 [Quick Start Guide](../docs/QUICK_START.md)
- 🎯 [Custom Portfolio Files](../examples/CUSTOM_FILES_GUIDE.md)
- 🔑 [API Key Setup](../examples/QUICKSTART.md)
- 📊 [Example Gallery](../examples/)

## 💻 Programmatic Examples

All examples support custom portfolio CSV/Excel files and API keys:

```python
# Configuration at the top of any example
CUSTOM_PORTFOLIO_FILE = 'my_portfolio.csv'  # Your portfolio
OPENROUTER_API_KEY = 'sk-or-v1-...'        # Your API key (optional)
```

**Available Examples:**
- `example_use_your_data.py` - ⭐ Start here! Complete analysis
- `example_stress_testing_code.py` - Stress testing deep dive
- `example_esg_analysis_code.py` - ESG analysis deep dive
- `example_complete_analysis_code.py` - All modules together
- `example_basic_usage.py` - Simple introduction

[📂 Browse All Examples →](../examples/)

## 🏗️ Architecture

```
XFIN/
├── __init__.py              # Main API exports
├── credit_risk.py           # Credit Risk & XAI Module
├── stress_testing.py        # Stress Testing Engine
├── esg.py                   # ESG Scoring Engine
├── compliance.py            # Regulatory Compliance
├── stress_plots.py          # Stress visualizations
├── app.py                   # Credit Risk Dashboard
├── stress_app.py            # Stress Testing Dashboard
├── explainer.py             # XAI Core Engine
└── utils.py                 # Shared utilities

examples/
├── example_use_your_data.py           # ⭐ Main example
├── example_stress_testing_code.py     # Stress testing
├── example_esg_analysis_code.py       # ESG analysis
├── example_complete_analysis_code.py  # All modules
├── portfolio_template.csv             # CSV template
└── CUSTOM_FILES_GUIDE.md             # Setup guide

docs/
├── README.md                # Main documentation
├── API_GUIDE.md            # Complete API reference
└── QUICK_START.md          # Getting started
```

## 🎨 Features Comparison

| Feature | Credit Risk | Stress Testing | ESG Analysis |
|---------|------------|----------------|--------------|
| **Primary Use** | Loan decisions | Portfolio risk | Sustainability |
| **Explainability** | ✅ SHAP/LIME | ✅ Scenario-based | ✅ SHAP ML |
| **Compliance** | ECOA/FCRA/GDPR | Basel III/RBI | SFDR/TCFD |
| **Visualizations** | ✅ Feature importance | ✅ Stress charts | ✅ ESG scores |
| **API Integration** | OpenRouter (LLM) | Alpha Vantage | Finnhub/Yahoo |
| **Custom Scenarios** | ❌ N/A | ✅ Yes | ❌ N/A |
| **Portfolio Analysis** | ❌ Individual | ✅ Portfolio-level | ✅ Portfolio-level |
| **Real-time Data** | ❌ User-provided | ✅ Market data | ✅ ESG APIs |

## 🔐 Privacy & Security

XFIN is designed with privacy at its core:

- **Model Privacy**: Your black-box models remain private; only predictions are used
- **Data Privacy**: No data is sent to external servers (unless using optional APIs)
- **Local Computation**: All XAI computations run locally
- **API Keys**: Encrypted storage and environment variable support
- **Audit Trails**: Comprehensive logging for regulatory compliance

## 🚀 Performance

- **Fast Analysis**: Process 1000+ portfolio holdings in seconds
- **Scalable**: Handles portfolios from retail (10 stocks) to institutional (1000+ holdings)
- **Efficient**: Low memory footprint, runs on commodity hardware
- **Batch Processing**: Analyze multiple scenarios simultaneously

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

### Areas for Contribution
- 🧪 New stress scenarios
- 🌍 Additional ESG data sources
- 📊 Enhanced visualizations
- 🌐 Multi-language support
- 📖 Documentation improvements
- 🐛 Bug fixes

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Private use

## 🙏 Acknowledgments

- **SHAP & LIME** teams for explainability frameworks
- **MSCI ESG**, **Finnhub**, **Yahoo Finance** for ESG data
- **Basel Committee** and **RBI** for stress testing frameworks
- **EU SFDR** for sustainable finance standards
- **Open-source community** for tools and support

## 📞 Contact & Support

### Get Help
- 📧 **Email**: [dhruv.jparmar0@gmail.com](mailto:dhruv.jparmar0@gmail.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/dhruvparmar10/XFIN/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/dhruvparmar10/XFIN/discussions)

### Stay Updated
- ⭐ **Star** the repo to show support
- 👀 **Watch** for updates
- 🍴 **Fork** to contribute

## 🗺️ Roadmap

### Version 0.2.0 (Q1 2026)
- [ ] Real-time market data integration
- [ ] Monte Carlo simulation
- [ ] Climate risk scenarios (TCFD)
- [ ] Enhanced ML models

### Version 0.3.0 (Q2 2026)
- [ ] Multi-currency support
- [ ] Derivatives stress testing
- [ ] Carbon footprint analysis
- [ ] API endpoints (REST)

### Version 1.0.0 (Q3 2026)
- [ ] Enterprise features
- [ ] Advanced compliance reporting
- [ ] Real-time dashboards
- [ ] Mobile app

## 📈 Version History

- **v0.1.0** (Nov 2025) - Initial release
  - Credit Risk XAI module
  - Stress Testing engine
  - ESG Scoring module
  - Unified dashboard

---

## 🎯 Quick Links

| Resource | Link |
|----------|------|
| 📦 **PyPI Package** | [xfin-xai](https://pypi.org/project/xfin-xai/) |
| 📚 **Documentation** | [ReadTheDocs](https://xfin-xai.readthedocs.io/) |
| 💻 **GitHub Repo** | [XFIN](https://github.com/dhruvparmar10/XFIN) |
| 🐛 **Issue Tracker** | [GitHub Issues](https://github.com/dhruvparmar10/XFIN/issues) |
| 💬 **Discussions** | [GitHub Discussions](https://github.com/dhruvparmar10/XFIN/discussions) |

---

<div align="center">

**XFIN v0.1.0** - Comprehensive Financial Risk Analysis & XAI

*Built with ❤️ by [Dhruv Parmar](https://github.com/dhruvparmar10) & [Rishabh Bhangale](https://github.com/rishabhbhangale)*

[⭐ Star on GitHub](https://github.com/dhruvparmar10/XFIN) • [📖 Read the Docs](https://xfin-xai.readthedocs.io/) • [🐛 Report Bug](https://github.com/dhruvparmar10/XFIN/issues)

</div>
