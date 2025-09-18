from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#Shit! Bhangale forgot to remove the AI comments
setup(
    name='xfin-xai',
    version='0.1.2',
    author='Rishabh Bhangale & Dhruv Parmar',
    author_email='dhruv.jparmar0@gmail.com', 
    description='Privacy-Preserving Explainable AI Library for Financial Services and Banking Systems',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhruvparmar10/XFIN", 
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        'streamlit', 
        'pandas', 
        'joblib', 
        'shap', 
        'lime', 
        'numpy', 
        'matplotlib',
        'requests',
        'python-dotenv',
        'scikit-learn'
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
            'flake8',
        ],
    },
    keywords='explainable-ai, finance, privacy, machine-learning, credit-risk, compliance',
    entry_points={
        'console_scripts': [
            'xfin=XFIN.cli:xfin_cli' # Legacy support
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/dhruvparmar10/XFIN/issues",
        "Source": "https://github.com/dhruvparmar10/XFIN",
        "Documentation": "https://github.com/dhruvparmar10/XFIN/blob/main/README.md",
    },
    license='MIT'
)
