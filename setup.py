"""
Setup script for probabilistic-market-engine.
"""

from setuptools import setup, find_packages

setup(
    name="probabilistic-market-engine",
    version="1.0.0",
    description="Production-grade nonlinear probabilistic state-space trading engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Quantitative Systems Team",
    python_requires=">=3.9",
    packages=find_packages(include=["probabilistic_market_engine*"]),
    package_data={
        "probabilistic_market_engine": ["config/*.yaml"],
    },
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.24.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
