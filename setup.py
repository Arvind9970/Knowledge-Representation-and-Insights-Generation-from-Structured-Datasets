from setuptools import setup, find_packages

setup(
    name="knowledge_insights",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.13.0",
        "nltk>=3.8.0",
        "spacy>=3.5.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
