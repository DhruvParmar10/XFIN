# XFIN Stress Testing: Portfolio Risk Analysis & Scenario Modeling

[![PyPI version](https://badge.fury.io/py/xfin-xai.svg)](https://badge.fury.io/py/xfin-xai)
[![Documentation Status](https://readthedocs.org/projects/xfin-xai/badge/?version=latest)](https://xfin-xai.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

XFIN Stress Testing is a comprehensive Python module for portfolio risk analysis, stress testing, and scenario modeling for financial institutions. It enables portfolio managers and risk analysts to evaluate portfolio resilience under various market conditions and regulatory stress scenarios.

The module focuses on **multi-scenario stress testing**, **Value at Risk (VaR) calculation**, **diversification metrics**, and **sector-level impact analysis**, ensuring compliance with **Basel III** and **RBI stress testing** guidelines.

> **Note**: This module is designed for institutional and retail portfolio analysis, providing professional-grade risk metrics and scenario modeling capabilities.

## 🚀 Features

- **📊 Multi-Scenario Analysis**: Pre-configured scenarios (market correction, recession, inflation, etc.)
- **💰 Value at Risk (VaR)**: Historical simulation and parametric VaR calculation
- **🎯 Sector Impact Analysis**: Granular sector-by-sector stress impact with beta adjustments
- **📈 Diversification Metrics**: HHI (Herfindahl-Hirschman Index) and concentration analysis
- **🔧 Custom Scenarios**: Build and test your own stress scenarios
- **📋 Compliance Ready**: Basel III and RBI-compliant stress testing framework
- **⚡ Real-time Analysis**: Fast computation on portfolios of any size
- **📊 Visualization**: Professional charts and risk gauge dashboards

## 📦 Installation

### Quick Installation

```bash
pip install xfin-xai
```

### Launch the Stress Testing Dashboard

After installation, launch the interactive stress testing dashboard:

```bash
xfin stress
```

Or using the unified dashboard:

```bash
streamlit run unified_dashboard.py
```

### Command Line Options

```bash
# Show help
xfin stress --help

# Launch on custom port
xfin stress --port 8503

# Launch on all interfaces
xfin stress --host 0.0.0.0
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
- **Core Dependencies**: `pandas`, `numpy`, `matplotlib`, `plotly`, `yfinance`, `scipy`
- **Optional**: Alpha Vantage API key for real-time market data

See [`requirements.txt`](../requirements.txt) for the complete list.

## 🚀 Quick Start

Here's a basic example to get started with stress testing:

```python
import pandas as pd
from XFIN.stress_testing import StressTestingEngine, ScenarioGenerator

# Sample portfolio data
portfolio = pd.DataFrame({
    'Ticker': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS'],
    'Quantity': [500, 300, 1000, 400, 600],
    'Purchase_Price': [2200, 3200, 1500, 1300, 420],
    'Current_Price': [2450, 3600, 1650, 1500, 450],
    'Sector': ['Energy', 'IT', 'Financials', 'IT', 'FMCG']
})

# Calculate portfolio values
portfolio['Current_Value'] = portfolio['Quantity'] * portfolio['Current_Price']
portfolio['Investment'] = portfolio['Quantity'] * portfolio['Purchase_Price']

# Initialize stress testing engine
stress_engine = StressTestingEngine()
scenario_gen = ScenarioGenerator()

# Get available scenarios
scenarios = scenario_gen.list_scenarios()
print("Available scenarios:", scenarios)
# Output: ['market_correction', 'recession', 'inflation_spike', 'tech_crash', ...]

# Run stress test for market correction
scenario = scenario_gen.get_scenario("market_correction")
print(f"\nScenario: {scenario['name']}")
print(f"Description: {scenario['description']}")

# Apply stress to portfolio
portfolio['Sector_Impact'] = portfolio['Sector'].map(
    scenario.get('sector_impacts', {})
).fillna(scenario['equity_impact'])

portfolio['Stressed_Value'] = (
    portfolio['Current_Value'] * (1 + portfolio['Sector_Impact'])
)
portfolio['Loss'] = portfolio['Stressed_Value'] - portfolio['Current_Value']

# Calculate portfolio impact
total_value = portfolio['Current_Value'].sum()
total_loss = portfolio['Loss'].sum()
impact_pct = (total_loss / total_value) * 100

print(f"\n📊 Stress Test Results:")
print(f"Current Portfolio Value: ₹{total_value:,.2f}")
print(f"Stressed Portfolio Value: ₹{portfolio['Stressed_Value'].sum():,.2f}")
print(f"Estimated Loss: ₹{total_loss:,.2f}")
print(f"Impact: {impact_pct:.2f}%")

# Calculate Value at Risk (VaR)
import numpy as np

portfolio['Returns'] = (
    (portfolio['Current_Price'] - portfolio['Purchase_Price']) / 
    portfolio['Purchase_Price']
)
portfolio_volatility = portfolio['Returns'].std() * np.sqrt(len(portfolio))
var_95 = total_value * portfolio_volatility * 1.645  # 95% confidence

print(f"\n💰 Risk Metrics:")
print(f"VaR (95% confidence): ₹{var_95:,.2f}")
print(f"Maximum Expected Loss: {(var_95/total_value)*100:.2f}%")

# Calculate diversification (HHI)
sector_weights = portfolio.groupby('Sector')['Current_Value'].sum() / total_value
hhi = (sector_weights ** 2).sum() * 10000

print(f"\n🎲 Diversification Metrics:")
print(f"HHI Index: {hhi:.0f}")
if hhi < 1500:
    print("Diversification: Excellent (Highly Diversified)")
elif hhi < 2500:
    print("Diversification: Good (Well Diversified)")
else:
    print("Diversification: Moderate (Concentrated)")
```

## 📊 Pre-Configured Scenarios

The module includes 10+ professional stress scenarios:

### Market-Wide Scenarios
- **Market Correction**: 10-15% broad market decline
- **Recession**: Severe economic downturn (15-25% impact)
- **Inflation Spike**: High inflation environment
- **Global Crisis**: Systemic financial crisis

### Sector-Specific Scenarios
- **Tech Crash**: Technology sector downturn
- **Banking Crisis**: Financial sector stress
- **Energy Shock**: Oil/gas price volatility
- **Currency Devaluation**: Rupee depreciation impact

### Regulatory Scenarios
- **Basel III Stress**: Regulatory capital adequacy testing
- **RBI Baseline**: Reserve Bank of India baseline scenario
- **RBI Severe**: RBI severe economic stress scenario

## 🎯 Advanced Features

### 1. Multi-Scenario Analysis

```python
from XFIN.stress_testing import StressTestingEngine, ScenarioGenerator

stress_engine = StressTestingEngine()
scenario_gen = ScenarioGenerator()

# Test multiple scenarios
scenarios_to_test = ['market_correction', 'recession', 'inflation_spike']
results = []

for scenario_name in scenarios_to_test:
    scenario = scenario_gen.get_scenario(scenario_name)
    
    # Apply stress (code from Quick Start)
    # ... calculate impact ...
    
    results.append({
        'Scenario': scenario['name'],
        'Impact_%': impact_pct,
        'Loss': total_loss
    })

# Find worst-case scenario
results_df = pd.DataFrame(results)
worst_case = results_df.loc[results_df['Impact_%'].idxmin()]

print(f"Worst Case: {worst_case['Scenario']} ({worst_case['Impact_%']:.2f}%)")
```

### 2. Custom Scenario Creation

```python
from XFIN.stress_testing import ScenarioGenerator

scenario_gen = ScenarioGenerator()

# Create custom scenario
custom_scenario = {
    'name': 'Tech Bubble Burst',
    'description': 'Major technology sector correction',
    'equity_impact': -0.15,  # -15% general equities
    'bond_impact': 0.02,     # +2% bonds (flight to safety)
    'sector_impacts': {
        'IT': -0.30,         # -30% IT sector
        'Telecom': -0.20,    # -20% Telecom
        'Financials': -0.10, # -10% Financials
        'FMCG': -0.05,       # -5% FMCG (defensive)
        'Pharma': 0.00       # 0% Pharma (neutral)
    }
}

# Add to scenario library
scenario_gen.add_custom_scenario(custom_scenario)

# Use custom scenario
scenario = scenario_gen.get_scenario('Tech Bubble Burst')
```

### 3. Sector Impact Analysis

```python
# Group by sector and calculate impacts
sector_analysis = portfolio.groupby('Sector').agg({
    'Current_Value': 'sum',
    'Loss': 'sum',
    'Sector_Impact': 'first'
}).round(2)

sector_analysis['Weight_%'] = (
    sector_analysis['Current_Value'] / total_value * 100
).round(2)

sector_analysis['Impact_%'] = (
    sector_analysis['Loss'] / sector_analysis['Current_Value'] * 100
).round(2)

print("\n📊 Sector-wise Impact:")
print(sector_analysis)

# Output:
#              Current_Value      Loss  Sector_Impact  Weight_%  Impact_%
# Sector                                                                  
# Energy          1,225,000  -122,500          -0.10     25.00    -10.00
# IT              1,680,000  -168,000          -0.10     34.00    -10.00
# Financials      1,650,000  -247,500          -0.15     33.50    -15.00
# FMCG              270,000   -13,500          -0.05      5.50     -5.00
```

### 4. Concentration Risk

```python
import numpy as np

# Calculate concentration by different dimensions

# 1. Sector Concentration
sector_weights = portfolio.groupby('Sector')['Current_Value'].sum() / total_value
sector_hhi = (sector_weights ** 2).sum() * 10000

# 2. Single Stock Concentration
stock_weights = portfolio['Current_Value'] / total_value
max_single_holding = stock_weights.max() * 100

# 3. Top 5 Holdings Concentration
top_5_concentration = stock_weights.nlargest(5).sum() * 100

print(f"\n⚠️  Concentration Metrics:")
print(f"Sector HHI: {sector_hhi:.0f}")
print(f"Largest Single Holding: {max_single_holding:.2f}%")
print(f"Top 5 Holdings: {top_5_concentration:.2f}%")

# Risk assessment
if max_single_holding > 25:
    print("⚠️  WARNING: High single-stock concentration risk")
if sector_hhi > 2500:
    print("⚠️  WARNING: High sector concentration risk")
```

### 5. Historical VaR Calculation

```python
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# Download historical data
end_date = datetime.now()
start_date = end_date - timedelta(days=252)  # 1 year

portfolio_returns = []

for idx, row in portfolio.iterrows():
    ticker = row['Ticker']
    weight = row['Current_Value'] / total_value
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        returns = data['Close'].pct_change().dropna()
        weighted_returns = returns * weight
        portfolio_returns.append(weighted_returns)
    except:
        pass

# Combine into portfolio returns
if portfolio_returns:
    combined_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
    
    # Calculate VaR using historical simulation
    var_95_hist = np.percentile(combined_returns, 5) * total_value
    var_99_hist = np.percentile(combined_returns, 1) * total_value
    
    print(f"\n📈 Historical VaR:")
    print(f"VaR (95%): ₹{abs(var_95_hist):,.2f}")
    print(f"VaR (99%): ₹{abs(var_99_hist):,.2f}")
```

## 📊 Visualization

The module includes built-in visualization capabilities:

```python
from XFIN.stress_plots import StressPlots
import matplotlib.pyplot as plt

# Create visualizer
plotter = StressPlots()

# 1. Stress impact chart
plotter.plot_stress_impact(
    portfolio=portfolio,
    scenario=scenario,
    output_file='stress_impact.png'
)

# 2. Sector allocation pie chart
plotter.plot_sector_allocation(
    portfolio=portfolio,
    output_file='sector_allocation.png'
)

# 3. Risk gauge
plotter.plot_risk_gauge(
    current_value=total_value,
    stressed_value=portfolio['Stressed_Value'].sum(),
    output_file='risk_gauge.png'
)

# 4. Multi-scenario comparison
plotter.plot_scenario_comparison(
    scenarios=results_df,
    output_file='scenario_comparison.png'
)

plt.show()
```

## 📋 Compliance & Reporting

### Basel III Compliance

```python
from XFIN.compliance import ComplianceEngine

compliance = ComplianceEngine()

# Run Basel III stress test
basel_results = compliance.run_basel_stress_test(
    portfolio=portfolio,
    capital_ratio=0.12  # 12% Tier 1 capital ratio
)

print(f"\n✅ Basel III Compliance:")
print(f"Stress Loss: {basel_results['stress_loss_pct']:.2f}%")
print(f"Post-Stress Capital Ratio: {basel_results['post_stress_capital']:.2f}%")
print(f"Minimum Required: 10.5%")
print(f"Status: {'PASS ✅' if basel_results['compliant'] else 'FAIL ❌'}")
```

### Export Reports

```python
# Export results to Excel
import pandas as pd

with pd.ExcelWriter('stress_test_report.xlsx') as writer:
    # Portfolio summary
    portfolio.to_excel(writer, sheet_name='Portfolio', index=False)
    
    # Scenario results
    results_df.to_excel(writer, sheet_name='Scenarios', index=False)
    
    # Sector analysis
    sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
    
    # Risk metrics
    risk_metrics = pd.DataFrame({
        'Metric': ['VaR (95%)', 'VaR (99%)', 'HHI Index', 'Max Single Holding'],
        'Value': [var_95, var_99_hist, hhi, max_single_holding]
    })
    risk_metrics.to_excel(writer, sheet_name='Risk Metrics', index=False)

print("✅ Report exported: stress_test_report.xlsx")
```

## 🎓 Use Cases

### 1. **Portfolio Managers**
- Test portfolio resilience under market stress
- Identify concentration risks
- Optimize sector allocation
- Calculate downside risk (VaR)

### 2. **Risk Analysts**
- Regulatory stress testing (Basel III, RBI)
- Scenario analysis and modeling
- Risk reporting and dashboards
- Compliance validation

### 3. **Financial Advisors**
- Client portfolio risk assessment
- Investment strategy validation
- Risk-adjusted recommendations
- What-if scenario planning

### 4. **Institutional Investors**
- Large portfolio stress testing
- Multi-scenario Monte Carlo simulation
- Tail risk analysis
- Capital adequacy testing

## 📖 Documentation

Full documentation is available at [xfin-xai.readthedocs.io](https://xfin-xai.readthedocs.io/en/latest/).

- 📚 [Stress Testing API Reference](https://xfin-xai.readthedocs.io/en/latest/api/stress_testing.html)
- 🎓 [Stress Testing Tutorials](https://xfin-xai.readthedocs.io/en/latest/tutorials/stress_testing.html)
- 📊 [Scenario Library](https://xfin-xai.readthedocs.io/en/latest/scenarios.html)
- 🗺️ [Roadmap](https://xfin-xai.readthedocs.io/en/latest/roadmap.html)

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/StressScenario`)
3. **Commit** your changes (`git commit -m 'Add new stress scenario'`)
4. **Push** to the branch (`git push origin feature/StressScenario`)
5. **Open** a Pull Request

For bugs or feature requests, please open an issue on GitHub.

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **Basel Committee** for stress testing frameworks
- **Reserve Bank of India** for regulatory guidelines
- **Open-source community** for financial modeling tools
- **QuantLib** and **PyPortfolioOpt** for inspiration

## 📞 Contact

For questions or support:

- **Email**: [dhruv.jparmar0@gmail.com](mailto:dhruv.jparmar0@gmail.com)
- **Issues**: [GitHub Issues](https://github.com/dhruvparmar10/XFIN/issues)

---

**XFIN Stress Testing** - Professional Portfolio Risk Analysis

*Built with ❤️ by Dhruv Parmar & Rishabh Bhangale*
