"""
OpenMedallion: Unified Financial AI Package

Combines time-series forecasting (openmedallion-fints) and 
sentiment analysis (openmedallion-finsentiment) with HuggingFace Hub integration.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="openmedallion",
    version="0.1.0",
    author="OpenMedallion Team",
    author_email="",
    description="Unified financial AI package for time-series forecasting and sentiment analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oyi77/OpenMedallion",
    packages=find_packages(include=["openmedallion", "openmedallion.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        # Core dependencies
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        
        # Time-series forecasting (fints)
        "lightgbm>=3.3.0",
        "torch>=2.0.0",
        "ta>=0.10.0",  # Technical indicators
        
        # Sentiment analysis (finsentiment)
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "peft>=0.4.0",  # QLoRA
        "bitsandbytes>=0.41.0",  # 4-bit quantization
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",  # Qwen tokenizer
        
        # HuggingFace Hub integration
        "huggingface-hub>=0.16.0",
        
        # Data collection
        "yfinance>=0.2.0",
        "requests>=2.28.0",
        
        # Utilities
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "cloud": [
            "modal>=0.50.0",  # Modal.com cloud training
            "wandb>=0.15.0",  # Weights & Biases monitoring
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "openmedallion-train-fints=openmedallion.fints.scripts.train_lgbm:main",
            "openmedallion-train-finsentiment=openmedallion.finsentiment.fine_tune_qwen:main",
            "openmedallion-eval-backtest=openmedallion.fints.scripts.eval_backtest:main",
        ],
    },
    include_package_data=True,
    package_data={
        "openmedallion.hub": ["templates/*.md"],
    },
    zip_safe=False,
)
