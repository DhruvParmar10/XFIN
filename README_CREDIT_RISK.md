# XFIN Documentation Index

Welcome to the XFIN library documentation! This index will help you quickly find what you need.

## 📚 Main Documentation

### Getting Started
- **[Main README](README.md)** - Overview of all three modules
- **[Quick Start Guide](../docs/QUICK_START.md)** - 5-minute introduction
- **[Installation Guide](README.md#-installation)** - Setup instructions

### Module Documentation
1. **[Credit Risk & XAI](../docs/README.md)** - Explainable credit decisions
2. **[Stress Testing](README_STRESS_TESTING.md)** - Portfolio risk analysis
3. **[ESG Analysis](README_ESG.md)** - Sustainable investment scoring

### User Guides
- **[Custom Portfolio Files](../examples/CUSTOM_FILES_GUIDE.md)** - Use your own data
- **[Quick Start (3 steps)](../examples/QUICKSTART.md)** - Fastest way to begin
- **[Visual Guide](../examples/VISUAL_GUIDE.md)** - Flowcharts and diagrams
- **[API Reference](../docs/API_GUIDE.md)** - Complete API documentation

## 🎯 I Want To...

### For First-Time Users
```
I want to analyze my portfolio
└─→ Start with: examples/QUICKSTART.md (3 steps)
    └─→ Then run: examples/example_use_your_data.py
```

### For Credit Risk Analysis
```
I want to explain loan decisions
└─→ Read: docs/README.md (Credit Risk)
    └─→ Run: xfin credit (dashboard)
    └─→ Or: examples/example_credit_risk_code.py
```

### For Portfolio Stress Testing
```
I want to stress test my portfolio
└─→ Read: XFIN/README_STRESS_TESTING.md
    └─→ Run: xfin stress (dashboard)
    └─→ Or: examples/example_stress_testing_code.py
```

### For ESG Analysis
```
I want to check ESG scores
└─→ Read: XFIN/README_ESG.md
    └─→ Run: xfin esg (dashboard)
    └─→ Or: examples/example_esg_analysis_code.py
```

### For Complete Analysis
```
I want credit + stress + ESG together
└─→ Run: streamlit run unified_dashboard.py (web UI)
    └─→ Or: examples/example_complete_analysis_code.py (code)
```

## 📂 Documentation Files

### In `/XFIN/` Folder
| File | Description |
|------|-------------|
| **README.md** | Main library overview (all 3 modules) |
| **README_STRESS_TESTING.md** | Stress testing module guide |
| **README_ESG.md** | ESG analysis module guide |
| **README_INDEX.md** | This file (navigation index) |

### In `/docs/` Folder
| File | Description |
|------|-------------|
| **README.md** | Credit Risk & XAI module guide |
| **API_GUIDE.md** | Complete API reference |
| **QUICK_START.md** | 5-minute getting started |
| **LIBRARY_RESTRUCTURING.md** | Architecture details |

### In `/examples/` Folder
| File | Description |
|------|-------------|
| **QUICKSTART.md** | 3-step setup guide |
| **CUSTOM_FILES_GUIDE.md** | Using your own CSV/Excel files |
| **VISUAL_GUIDE.md** | Visual flowcharts |
| **CUSTOM_SETUP_SUMMARY.md** | Configuration summary |
| **README.md** | Examples overview |

## 🎓 Learning Path

### Beginner (Day 1)
1. Read [QUICKSTART.md](../examples/QUICKSTART.md) (5 min)
2. Create portfolio CSV using template (10 min)
3. Run `example_use_your_data.py` (5 min)
4. Review output and understand metrics (10 min)

**Total: 30 minutes to your first analysis!**

### Intermediate (Day 2-3)
1. Read module-specific READMEs:
   - [Stress Testing](README_STRESS_TESTING.md)
   - [ESG Analysis](README_ESG.md)
2. Try individual module examples
3. Experiment with custom scenarios
4. Generate reports and visualizations

**Total: 2-3 hours to master each module**

### Advanced (Week 1+)
1. Read [API_GUIDE.md](../docs/API_GUIDE.md)
2. Build custom scenarios and models
3. Integrate into your workflow
4. Combine all three modules
5. Create custom dashboards

**Total: Ongoing, as needed**

## 🔍 Quick Reference

### Installation
```bash
pip install xfin-xai
```

### Launch Dashboards
```bash
xfin credit          # Credit Risk Dashboard
xfin stress          # Stress Testing Dashboard
xfin esg             # ESG Analysis Dashboard
streamlit run unified_dashboard.py  # All modules
```

### Import Library
```python
import XFIN as xfin

# Credit Risk
credit = xfin.CreditAnalyzer(model)

# Stress Testing
stress = xfin.StressAnalyzer()

# ESG Analysis
esg = xfin.ESGAnalyzer()
```

### Use Custom Data
```python
# At top of any example file:
CUSTOM_PORTFOLIO_FILE = 'my_portfolio.csv'
OPENROUTER_API_KEY = 'sk-or-v1-...'
```

## 📊 Documentation by Feature

### Credit Risk Features
- **XAI Explanations**: [docs/README.md#features](../docs/README.md#-features)
- **Adverse Notices**: [docs/README.md#quick-start](../docs/README.md#-quick-start)
- **Recommendations**: [docs/README.md#quick-start](../docs/README.md#-quick-start)

### Stress Testing Features
- **Scenarios**: [README_STRESS_TESTING.md#scenarios](README_STRESS_TESTING.md#-pre-configured-scenarios)
- **VaR Calculation**: [README_STRESS_TESTING.md#advanced](README_STRESS_TESTING.md#-advanced-features)
- **Custom Scenarios**: [README_STRESS_TESTING.md#custom](README_STRESS_TESTING.md#2-custom-scenario-creation)

### ESG Features
- **Scoring**: [README_ESG.md#methodology](README_ESG.md#-esg-scoring-methodology)
- **SFDR Compliance**: [README_ESG.md#sfdr](README_ESG.md#2-sfdr-compliance-classification)
- **ML Predictions**: [README_ESG.md#ml](README_ESG.md#6-ml-prediction-with-shap-explainability)

## 🆘 Troubleshooting

### File Issues
**Problem**: "File not found"  
**Solution**: See [CUSTOM_FILES_GUIDE.md](../examples/CUSTOM_FILES_GUIDE.md#troubleshooting)

### API Issues
**Problem**: "Invalid API key"  
**Solution**: Check key format and credits at OpenRouter

### Import Issues
**Problem**: "Module not found"  
**Solution**: Run `pip install -e .` in XFIN directory

### Data Issues
**Problem**: "Missing columns"  
**Solution**: Check [CSV format](../examples/QUICKSTART.md#-example-csv-format)

## 🔗 External Links

- **GitHub Repository**: https://github.com/dhruvparmar10/XFIN
- **PyPI Package**: https://pypi.org/project/xfin-xai/
- **ReadTheDocs**: https://xfin-xai.readthedocs.io/
- **Issue Tracker**: https://github.com/dhruvparmar10/XFIN/issues
- **OpenRouter API**: https://openrouter.ai/keys

## 📞 Get Help

1. **Check Documentation** (you're here! ✓)
2. **Search GitHub Issues**: [Open Issues](https://github.com/dhruvparmar10/XFIN/issues)
3. **Ask Question**: [New Discussion](https://github.com/dhruvparmar10/XFIN/discussions)
4. **Report Bug**: [New Issue](https://github.com/dhruvparmar10/XFIN/issues/new)
5. **Email**: dhruv.jparmar0@gmail.com

## 🎯 Quick Navigation

| I need... | Go to... |
|-----------|----------|
| Overview of XFIN | [README.md](README.md) |
| Credit Risk docs | [docs/README.md](../docs/README.md) |
| Stress Testing docs | [README_STRESS_TESTING.md](README_STRESS_TESTING.md) |
| ESG docs | [README_ESG.md](README_ESG.md) |
| Getting started (fast) | [QUICKSTART.md](../examples/QUICKSTART.md) |
| Custom portfolio guide | [CUSTOM_FILES_GUIDE.md](../examples/CUSTOM_FILES_GUIDE.md) |
| Code examples | [examples/](../examples/) |
| API reference | [API_GUIDE.md](../docs/API_GUIDE.md) |

---

**Still can't find what you need?**  
📧 Email: dhruv.jparmar0@gmail.com  
🐛 Issues: https://github.com/dhruvparmar10/XFIN/issues

---

*Last updated: November 2025*  
*XFIN v0.1.0*
