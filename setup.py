from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_path = os.path.join(os.path.dirname(__file__), 'XFIN', '__init__.py')
    with open(init_path, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'

# Read README for long description
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return """
# XFIN - Enterprise Financial Risk Analysis Library

Professional-grade stress testing and risk analysis for banks, NBFCs, and financial institutions.

## Quick Start

```python
import XFIN

# Create stress testing engine - works immediately
engine = XFIN.create_stress_analyzer()

# Run stress test
results = engine.analyze_portfolio(portfolio_df, "market_correction")
```

Ready for production deployment with zero configuration required.
"""

setup(
    name='XFIN',
    version=get_version(),
    author='Rishabh Bhangale & Dhruv Parmar',
    author_email='dhruv.jparmar0@gmail.com',
    description='Enterprise Financial Risk Analysis Library - Professional stress testing for banks, NBFCs, and financial institutions',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/dhruvparmar10/XFIN",
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry", 
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        
        # Explainability frameworks
        'shap>=0.41.0',
        'lime>=0.2.0',
        
        # Visualization
        'matplotlib>=3.5.0',
        'plotly>=5.0.0',
        
        # Web interface
        'streamlit>=1.20.0',
        
        # API and utilities
        'requests>=2.25.0',
        'python-dotenv>=0.19.0',
        
        # Data processing
        'joblib>=1.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'pre-commit>=2.15.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'myst-parser>=0.15.0',
        ],
        'enterprise': [
            'redis>=4.0.0',
            'celery>=5.2.0',
            'fastapi>=0.75.0',
            'uvicorn>=0.17.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'xfin=XFIN.cli:main',  # Main CLI entry point
            'xfin-dashboard=XFIN.stress_app:launch_stress_dashboard',
            'xfin-example=example_bank_usage:main',
        ],
    },
    keywords=[
        'explainable-ai', 'xai', 'finance', 'fintech', 'banking',
        'credit-risk', 'stress-testing', 'esg-scoring', 'portfolio-analysis',
        'risk-management', 'compliance', 'privacy-preserving', 'shap', 'lime',
        'financial-modeling', 'regulatory-compliance', 'sfdr', 'gdpr', 'ecoa'
    ],
    project_urls={
        "Homepage": "https://github.com/dhruvparmar10/XFIN",
        "Bug Reports": "https://github.com/dhruvparmar10/XFIN/issues",
        "Source": "https://github.com/dhruvparmar10/XFIN",
        "Documentation": "https://github.com/dhruvparmar10/XFIN/blob/main/README.md",
        "Changelog": "https://github.com/dhruvparmar10/XFIN/blob/main/RELEASE_NOTES_v0.1.0-scratch.md",
    },
    license='MIT',
    zip_safe=False,
)
