# XFIN-XAI: Explainable AI Library for Financial Services and Banking Systems

[![PyPI version](https://badge.fury.io/py/xfin-xai.svg)](https://badge.fury.io/py/xfin-xai)
[![Documentation Status](https://readthedocs.org/projects/xfin-xai/badge/?version=latest)](https://xfin-xai.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

XFIN-XAI is an open-source Python library designed for privacy-preserving explainable AI (XAI) in financial services and banking systems. It enables banks and financial institutions to generate transparent explanations for black-box models without exposing proprietary internals. The MVP focuses on credit risk explanations, adverse action notices, and counterfactual recommendations, ensuring compliance with regulations like GDPR and ECOA.

This library is built for educational and research purposes, allowing users to explore XAI in finance while maintaining data privacy.

## Features

- **Privacy-Preserving Explanations**: Integrates SHAP and LIME for local explanations using only model predictions (no access to internals).
- **Credit Risk Module**: Generates feature importances, adverse action notices, and actionable recommendations.
- **Compliance Engine**: Produces regulatory-compliant reports and audit trails.
- **LLM Integration**: Uses Gemini (or similar) for natural language explanations and recommendations.
- **Modular Design**: Easily extensible for other domains (e.g., ESG, stress testing) in future versions.
- **Efficient and Scalable**: Runs on commodity hardware with low resource usage.

## Installation

Install XFIN via pip:

```
pip install xfin-xai
```

For development installation, clone the repository and install dependencies:

```
git clone https://github.com/dhruvparmar10/XFIN.git
cd XFIN
pip install -e .
```

### Requirements

- Python 3.8+
- Dependencies: `pandas`, `numpy`, `shap`, `lime`, `joblib`, `matplotlib`,`streamlit`,(Open router for LLM features)

See `requirements.txt` for the full list.

## Quick Start

Here's a basic example to get started with credit risk explanations:

```
import pandas as pd
import joblib
from xfin_xai import CreditRiskModule

# Load your black-box model (example)

model = joblib.load('path/to/your/model.pkl')

# Define a wrapper for your model (only expose predict/predict_proba)

class BankModel:
def predict(self, X):
return model.predict(X)
def predict_proba(self, X):
return model.predict_proba(X)

# Sample input data (replace with your features)

sample_data = pd.DataFrame({
'Annual_income': ,
'Employed_days': , # Add other features as per your dataset
})

# Initialize the explainer

explainer = CreditRiskModule(BankModel(), domain="credit_risk")

# Generate explanation

explanation = explainer.explain_prediction(sample_data)

# Generate recommendations

recommendations = explainer.generate_recommendations(sample_data)

# Generate compliance notice

compliance = explainer.generate_adverse_action_notice(explanation)

print("Prediction Explanation:", explanation)
print("Recommendations:", recommendations)
print("Adverse Action Notice:", compliance)
```

For more details, see the [examples](./examples) directory.

## Documentation

Full documentation is available at [xfin-xai.readthedocs.io](https://xfin-xai.readthedocs.io/en/latest/).

- [API Reference](https://xfin-xai.readthedocs.io/en/latest/api.html)
- [Tutorials](https://xfin-xai.readthedocs.io/en/latest/tutorials.html)
- [Roadmap](https://xfin-xai.readthedocs.io/en/latest/roadmap.html)

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

For bugs or feature requests, open an issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments

- Major Thanks to Shap and Lime for building great tools.
- Thanks to the open-source community for tools like setuptools and ReadTheDocs.

For questions, contact [dhruv.jparmar0@gmail.com] or open an issue.

```

```
