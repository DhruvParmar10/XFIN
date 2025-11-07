# XFIN ESG: Sustainable Investment Analysis & Portfolio Scoring

[![PyPI version](https://badge.fury.io/py/xfin-xai.svg)](https://badge.fury.io/py/xfin-xai)
[![Documentation Status](https://readthedocs.org/projects/xfin-xai/badge/?version=latest)](https://xfin-xai.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

XFIN ESG is a comprehensive Python module for ESG (Environmental, Social, and Governance) analysis, sustainable investment scoring, and portfolio sustainability assessment. It enables investors, fund managers, and institutions to evaluate companies and portfolios based on ESG criteria with transparency and explainability.

The module focuses on **multi-source ESG data aggregation**, **ML-powered ESG prediction**, **SFDR compliance**, **sector benchmarking**, and **portfolio-level sustainability metrics**, ensuring alignment with **EU SFDR**, **TCFD**, and **SASB** frameworks.

> **Note**: This module combines real ESG data from multiple providers (Yahoo Finance, Finnhub, RapidAPI) with ML-based predictions for comprehensive coverage, especially for emerging markets and small-cap stocks.

## 🚀 Features

- **🌍 Multi-Source ESG Data**: Aggregates from Yahoo Finance (MSCI), Finnhub, RapidAPI, and FMP
- **🤖 ML-Powered Predictions**: LightGBM model with SHAP explainability for missing ESG data
- **📊 Individual & Portfolio Scoring**: Score individual securities and entire portfolios
- **🎯 SFDR Compliance**: Article 6/8/9 classification for EU regulatory compliance
- **📈 Sector Benchmarking**: Compare against sector ESG averages
- **💡 Improvement Recommendations**: Identify low-ESG holdings and alternatives
- **🔍 Component Analysis**: Breakdown of E, S, G scores with contribution analysis
- **⚡ Explainable AI**: SHAP-based explanations for all ML predictions
- **📋 Risk Assessment**: ESG risk scoring and controversy detection

## 📦 Installation

### Quick Installation

```bash
pip install xfin-xai
```

### Launch the ESG Analysis Dashboard

After installation, launch the interactive ESG dashboard:

```bash
xfin esg
```

Or using the unified dashboard:

```bash
streamlit run unified_dashboard.py
```

### Command Line Options

```bash
# Show help
xfin esg --help

# Launch on custom port
xfin esg --port 8504

# Launch on all interfaces
xfin esg --host 0.0.0.0
```

### Development Installation

For development installation:

```bash
git clone https://github.com/dhruvparmar10/XFIN.git
cd XFIN
pip install -e .
```

## 🔧 Requirements

- **Python**: 3.9+
- **Core Dependencies**: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`, `yfinance`, `requests`
- **Optional**: Finnhub API key, RapidAPI key for enhanced ESG coverage

See [`requirements.txt`](../requirements.txt) for the complete list.

## 🚀 Quick Start

Here's a basic example to get started with ESG analysis:

```python
import pandas as pd
from XFIN.esg import ESGScoringEngine

# Sample portfolio data
portfolio = pd.DataFrame({
    'Ticker': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS'],
    'Company_Name': ['Reliance Industries', 'Tata Consultancy Services', 
                     'HDFC Bank', 'Infosys', 'ITC Limited'],
    'Quantity': [500, 300, 1000, 400, 600],
    'Current_Price': [2450, 3600, 1650, 1500, 450],
    'Sector': ['Energy', 'IT', 'Financials', 'IT', 'FMCG']
})

portfolio['Current_Value'] = portfolio['Quantity'] * portfolio['Current_Price']

# Initialize ESG scoring engine
esg_engine = ESGScoringEngine()

print("🌱 Scoring individual holdings...")

# Score each security
esg_scores = []

for idx, row in portfolio.iterrows():
    score = esg_engine.score_security(
        ticker=row['Ticker'],
        company_name=row['Company_Name'],
        sector=row['Sector']
    )
    
    esg_scores.append({
        'Ticker': row['Ticker'],
        'Company': row['Company_Name'],
        'ESG_Score': score.get('overall_score', 50.0),
        'E_Score': score.get('environmental_score', 50.0),
        'S_Score': score.get('social_score', 50.0),
        'G_Score': score.get('governance_score', 50.0),
        'Data_Source': score.get('data_source', 'Unknown')
    })
    
    print(f"✅ {row['Company_Name']}: {score.get('overall_score', 50.0):.2f}/100")

# Merge with portfolio
esg_df = pd.DataFrame(esg_scores)
portfolio_esg = portfolio.merge(esg_df, on='Ticker')

# Calculate portfolio-level ESG score (weighted average)
portfolio_esg['Weight'] = portfolio_esg['Current_Value'] / portfolio_esg['Current_Value'].sum()
portfolio_esg_score = (portfolio_esg['ESG_Score'] * portfolio_esg['Weight']).sum()

# ESG Rating
def get_esg_rating(score):
    if score >= 80: return 'AAA (Excellent)'
    elif score >= 70: return 'AA (Very Good)'
    elif score >= 60: return 'A (Good)'
    elif score >= 50: return 'BBB (Average)'
    elif score >= 40: return 'BB (Below Average)'
    else: return 'B (Poor)'

print(f"\n📊 Portfolio ESG Summary:")
print(f"Overall Score: {portfolio_esg_score:.2f}/100")
print(f"Rating: {get_esg_rating(portfolio_esg_score)}")

# Component breakdown
e_score = (portfolio_esg['E_Score'] * portfolio_esg['Weight']).sum()
s_score = (portfolio_esg['S_Score'] * portfolio_esg['Weight']).sum()
g_score = (portfolio_esg['G_Score'] * portfolio_esg['Weight']).sum()

print(f"\n🌍 Component Scores:")
print(f"Environmental (E): {e_score:.2f}/100")
print(f"Social (S): {s_score:.2f}/100")
print(f"Governance (G): {g_score:.2f}/100")
```

## 📊 ESG Scoring Methodology

### Data Sources (Tiered Approach)

The module uses a **3-tier fallback system** for maximum coverage:

#### Tier 1: Real ESG Data (Preferred)
```python
# Sources: Yahoo Finance (MSCI ESG), Finnhub, RapidAPI, FMP
# Coverage: Large-cap stocks, internationally listed companies
# Quality: Highest (verified by rating agencies)
```

#### Tier 2: ML Prediction (SHAP Explainable)
```python
# LightGBM model trained on 1000+ Indian companies
# Features: Market cap, sector, financial metrics, governance indicators
# Explainability: SHAP values show feature contributions
# Quality: High (validated on test set)
```

#### Tier 3: Sector Proxy (Fallback)
```python
# Sector-wise ESG averages from proxy database
# Adjusted by market cap (Large/Mid/Small cap)
# Quality: Moderate (indicative only)
```

### Scoring Scale

```
90-100: AAA (Exceptional ESG leader)
80-89:  AA  (Excellent ESG performance)
70-79:  A   (Very good ESG practices)
60-69:  BBB (Good, above average)
50-59:  BB  (Average, meets basic standards)
40-49:  B   (Below average, improvement needed)
0-39:   CCC (Poor, significant ESG risks)
```

## 🎯 Advanced Features

### 1. Portfolio-Level Analysis

```python
from XFIN.esg import ESGScoringEngine

esg_engine = ESGScoringEngine()

# Score entire portfolio at once
portfolio_analysis = esg_engine.score_portfolio(portfolio)

print(f"📊 Portfolio Analysis:")
print(f"Overall ESG Score: {portfolio_analysis['overall_score']:.2f}/100")
print(f"Rating: {portfolio_analysis['rating']}")
print(f"Risk Level: {portfolio_analysis['risk_level']}")

# Sector breakdown
for sector, metrics in portfolio_analysis['sector_breakdown'].items():
    print(f"\n{sector}:")
    print(f"  ESG Score: {metrics['avg_esg']:.2f}")
    print(f"  Weight: {metrics['weight']*100:.2f}%")
```

### 2. SFDR Compliance Classification

```python
# Classify holdings by SFDR Article
def classify_sfdr(esg_score):
    if esg_score >= 70:
        return 'Article 9 (Dark Green - Sustainable Investment)'
    elif esg_score >= 50:
        return 'Article 8 (Light Green - ESG Promoting)'
    else:
        return 'Article 6 (Non-ESG)'

portfolio_esg['SFDR_Article'] = portfolio_esg['ESG_Score'].apply(classify_sfdr)

# Calculate SFDR breakdown
sfdr_breakdown = portfolio_esg.groupby('SFDR_Article').agg({
    'Current_Value': 'sum',
    'Weight': 'sum'
}).round(4)

sfdr_breakdown['Value_%'] = sfdr_breakdown['Weight'] * 100

print("\n🇪🇺 SFDR Compliance Breakdown:")
print(sfdr_breakdown[['Value_%']])

# Check if portfolio qualifies as Article 8 fund (>50% sustainable)
article_8_9_weight = sfdr_breakdown.loc[
    sfdr_breakdown.index.str.contains('Article 8|Article 9'), 'Weight'
].sum()

if article_8_9_weight >= 0.5:
    print(f"\n✅ Portfolio qualifies as Article 8 Fund ({article_8_9_weight*100:.1f}% ESG)")
else:
    print(f"\n❌ Portfolio does not qualify as Article 8 ({article_8_9_weight*100:.1f}% ESG)")
```

### 3. ESG Risk Assessment

```python
# Calculate ESG risk score (inverse of ESG score)
portfolio_esg['ESG_Risk'] = 100 - portfolio_esg['ESG_Score']

# Identify high-risk holdings
high_risk = portfolio_esg[portfolio_esg['ESG_Risk'] > 50].sort_values('ESG_Risk', ascending=False)

print("\n⚠️  High ESG Risk Holdings:")
for idx, row in high_risk.iterrows():
    print(f"  {row['Company']}: Risk Score {row['ESG_Risk']:.2f}")
    print(f"    Weight in portfolio: {row['Weight']*100:.2f}%")
    print(f"    Value at risk: ₹{row['Current_Value']:,.2f}")

# Portfolio risk metrics
total_esg_risk = (portfolio_esg['ESG_Risk'] * portfolio_esg['Weight']).sum()
high_risk_value = high_risk['Current_Value'].sum()
high_risk_pct = (high_risk_value / portfolio_esg['Current_Value'].sum()) * 100

print(f"\n📊 Portfolio ESG Risk Summary:")
print(f"Overall ESG Risk: {total_esg_risk:.2f}/100")
print(f"High-risk holdings: {high_risk_pct:.2f}% of portfolio")
```

### 4. Improvement Opportunities

```python
# Identify below-average holdings
avg_esg = portfolio_esg['ESG_Score'].mean()
improvement_candidates = portfolio_esg[portfolio_esg['ESG_Score'] < avg_esg].sort_values('ESG_Score')

print(f"\n💡 ESG Improvement Opportunities:")
print(f"Portfolio Average ESG: {avg_esg:.2f}")
print(f"\nBelow-average holdings ({len(improvement_candidates)}):")

for idx, row in improvement_candidates.iterrows():
    gap = avg_esg - row['ESG_Score']
    print(f"\n  {row['Company']} ({row['Ticker']})")
    print(f"    Current ESG: {row['ESG_Score']:.2f}")
    print(f"    Gap from avg: -{gap:.2f}")
    print(f"    Weight: {row['Weight']*100:.2f}%")
    print(f"    Recommendation: Consider replacing with higher-ESG {row['Sector']} stock")

# Calculate potential score improvement
potential_score = portfolio_esg_score + (len(improvement_candidates) * 5)  # Assume +5 per replacement
print(f"\n✨ Potential Portfolio ESG: {potential_score:.2f}/100 (+{potential_score-portfolio_esg_score:.2f})")
```

### 5. Sector Benchmarking

```python
# Compare portfolio sectors against benchmarks
sector_benchmarks = {
    'IT': 72.0,
    'Financials': 65.0,
    'Energy': 55.0,
    'FMCG': 68.0,
    'Healthcare': 70.0,
    'Industrials': 60.0
}

sector_comparison = portfolio_esg.groupby('Sector').agg({
    'ESG_Score': 'mean',
    'Weight': 'sum',
    'Current_Value': 'sum'
}).round(2)

sector_comparison['Benchmark'] = sector_comparison.index.map(sector_benchmarks)
sector_comparison['vs_Benchmark'] = sector_comparison['ESG_Score'] - sector_comparison['Benchmark']

print("\n📊 Sector ESG vs. Benchmarks:")
print(sector_comparison[['ESG_Score', 'Benchmark', 'vs_Benchmark', 'Weight']])

# Highlight outperformers and underperformers
outperformers = sector_comparison[sector_comparison['vs_Benchmark'] > 0]
underperformers = sector_comparison[sector_comparison['vs_Benchmark'] < 0]

if not outperformers.empty:
    print(f"\n✅ Outperforming sectors:")
    for sector, row in outperformers.iterrows():
        print(f"  {sector}: +{row['vs_Benchmark']:.2f} above benchmark")

if not underperformers.empty:
    print(f"\n⚠️  Underperforming sectors:")
    for sector, row in underperformers.iterrows():
        print(f"  {sector}: {row['vs_Benchmark']:.2f} below benchmark")
```

### 6. ML Prediction with SHAP Explainability

```python
import shap
import matplotlib.pyplot as plt

# Get ML prediction with SHAP explanation
esg_engine = ESGScoringEngine()

# For a specific stock
result = esg_engine.predict_with_shap(
    ticker='EXAMPLE.NS',
    sector='IT',
    market_cap=50000,  # in crores
    # ... other features
)

if result['success']:
    print(f"\n🤖 ML Prediction:")
    print(f"ESG Score: {result['esg_score']:.2f}/100")
    print(f"Confidence: {result['confidence']:.2f}%")
    
    print(f"\n🔍 SHAP Feature Importance:")
    for feature, value in result['shap_values'].items():
        direction = "↑" if value > 0 else "↓"
        print(f"  {feature}: {value:+.2f} {direction}")
    
    # Visualize SHAP
    shap.plots.waterfall(result['shap_explanation'])
    plt.savefig('shap_explanation.png')
```

### 7. ESG Trend Analysis

```python
import yfinance as yf
from datetime import datetime, timedelta

# Track ESG score changes over time
def track_esg_trends(ticker, periods=4):
    """Track quarterly ESG trends"""
    trends = []
    
    for i in range(periods):
        date = datetime.now() - timedelta(days=90*i)
        # Fetch historical ESG data
        score = esg_engine.score_security(ticker, date=date)
        trends.append({
            'Date': date.strftime('%Y-%m-%d'),
            'ESG_Score': score['overall_score']
        })
    
    return pd.DataFrame(trends)

# Example usage
trends = track_esg_trends('RELIANCE.NS')
print("\n📈 ESG Trend Analysis:")
print(trends)

# Calculate trend direction
recent_change = trends.iloc[0]['ESG_Score'] - trends.iloc[-1]['ESG_Score']
if recent_change > 0:
    print(f"✅ ESG Improving (+{recent_change:.2f} over {len(trends)-1} quarters)")
else:
    print(f"⚠️  ESG Declining ({recent_change:.2f} over {len(trends)-1} quarters)")
```

## 📊 Visualization

The module includes built-in ESG visualization:

```python
from XFIN.esg import ESGPlots
import matplotlib.pyplot as plt

plotter = ESGPlots()

# 1. ESG score distribution
plotter.plot_esg_distribution(
    portfolio_esg=portfolio_esg,
    output_file='esg_distribution.png'
)

# 2. Component breakdown (E/S/G)
plotter.plot_component_breakdown(
    e_score=e_score,
    s_score=s_score,
    g_score=g_score,
    output_file='esg_components.png'
)

# 3. Sector ESG heatmap
plotter.plot_sector_heatmap(
    portfolio_esg=portfolio_esg,
    output_file='sector_heatmap.png'
)

# 4. SFDR classification pie chart
plotter.plot_sfdr_breakdown(
    portfolio_esg=portfolio_esg,
    output_file='sfdr_breakdown.png'
)

# 5. ESG risk gauge
plotter.plot_esg_risk_gauge(
    portfolio_esg_score=portfolio_esg_score,
    output_file='esg_risk_gauge.png'
)

plt.show()
```

## 📋 Compliance & Reporting

### SFDR Compliance Report

```python
from XFIN.compliance import ESGComplianceEngine

compliance = ESGComplianceEngine()

# Generate SFDR compliance report
sfdr_report = compliance.generate_sfdr_report(
    portfolio=portfolio_esg,
    fund_name="My Sustainable Fund"
)

print(sfdr_report['summary'])
# Export to PDF
compliance.export_to_pdf(sfdr_report, 'sfdr_report.pdf')
```

### Export ESG Reports

```python
# Export comprehensive ESG report to Excel
with pd.ExcelWriter('esg_analysis_report.xlsx') as writer:
    # Portfolio with ESG scores
    portfolio_esg.to_excel(writer, sheet_name='Portfolio ESG', index=False)
    
    # Sector analysis
    sector_comparison.to_excel(writer, sheet_name='Sector Analysis')
    
    # SFDR breakdown
    sfdr_breakdown.to_excel(writer, sheet_name='SFDR Compliance')
    
    # Improvement opportunities
    improvement_candidates.to_excel(writer, sheet_name='Improvements', index=False)
    
    # Summary metrics
    summary = pd.DataFrame({
        'Metric': ['Portfolio ESG Score', 'ESG Rating', 'ESG Risk', 
                   'E Score', 'S Score', 'G Score', 'Article 8/9 %'],
        'Value': [portfolio_esg_score, get_esg_rating(portfolio_esg_score), 
                  total_esg_risk, e_score, s_score, g_score, article_8_9_weight*100]
    })
    summary.to_excel(writer, sheet_name='Summary', index=False)

print("✅ ESG report exported: esg_analysis_report.xlsx")
```

## 🎓 Use Cases

### 1. **Sustainable Fund Managers**
- Build Article 8/9 compliant portfolios
- Track ESG performance over time
- Benchmark against peers
- Generate SFDR reports

### 2. **ESG Analysts**
- Evaluate company ESG performance
- Identify ESG risks and opportunities
- Sector-level ESG analysis
- Controversy monitoring

### 3. **Impact Investors**
- Screen investments by ESG criteria
- Measure portfolio sustainability
- Calculate impact metrics
- Align with SDGs (Sustainable Development Goals)

### 4. **Retail Investors**
- Understand ESG scores of holdings
- Build sustainable portfolios
- Compare ESG performance
- Make informed investment decisions

### 5. **Corporate ESG Teams**
- Benchmark against competitors
- Track ESG improvements
- Identify areas for enhancement
- Generate ESG reports

## 📖 Documentation

Full documentation is available at [xfin-xai.readthedocs.io](https://xfin-xai.readthedocs.io/en/latest/).

- 📚 [ESG API Reference](https://xfin-xai.readthedocs.io/en/latest/api/esg.html)
- 🎓 [ESG Tutorials](https://xfin-xai.readthedocs.io/en/latest/tutorials/esg.html)
- 🌍 [ESG Methodology](https://xfin-xai.readthedocs.io/en/latest/esg_methodology.html)
- 🗺️ [Roadmap](https://xfin-xai.readthedocs.io/en/latest/roadmap.html)

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/ESGEnhancement`)
3. **Commit** your changes (`git commit -m 'Add ESG feature'`)
4. **Push** to the branch (`git push origin feature/ESGEnhancement`)
5. **Open** a Pull Request

For bugs or feature requests, please open an issue on GitHub.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **MSCI ESG Research** for ESG rating methodology
- **EU SFDR** for sustainable finance framework
- **TCFD** and **SASB** for ESG disclosure standards
- **Yahoo Finance**, **Finnhub**, **RapidAPI** for ESG data
- **LightGBM** and **SHAP** teams for ML and explainability tools

## 📞 Contact

For questions or support:

- **Email**: [dhruv.jparmar0@gmail.com](mailto:dhruv.jparmar0@gmail.com)
- **Issues**: [GitHub Issues](https://github.com/dhruvparmar10/XFIN/issues)

---

**XFIN ESG** - Sustainable Investment Analysis

*Built with ❤️ by Dhruv Parmar & Rishabh Bhangale*
