from setuptools import setup, find_packages

setup(
    name='xfin-xai',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit', 'pandas', 'joblib', 'shap', 'lime', 'numpy', 'matplotlib'
    ],
    description='Privacy-Preserving Explainable AI Library for Financial Services',
    author='Rishabh Bhangle & Dhruv Parmar',
    license='MIT'
)
