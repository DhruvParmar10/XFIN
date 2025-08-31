from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='xfin-xai',
    version='0.1.0',
    author='Rishabh Bhangle & Dhruv Parmar',
    author_email='dhruv.jparmar0@gmail.com',  # Add your email if desired
    description='Privacy-Preserving Explainable AI Library for Financial Services',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhruvparmar10/XFIN",  # Update with your actual repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
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
    project_urls={
        "Bug Reports": "https://github.com/dhruvparmar10/XFIN/issues",
        "Source": "https://github.com/dhruvparmar10/XFIN",
        "Documentation": "https://github.com/dhruvparmar10/XFIN/blob/main/README.md",
    },
    license='MIT'
)
