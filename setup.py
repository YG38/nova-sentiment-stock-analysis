from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nova-sentiment-stock-analysis",
    version="0.1.0",
    author="Nova Analytics Team",
    author_email="contact@nova-analytics.com",
    description="Financial news sentiment analysis and stock market correlation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nova-sentiment-stock-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "nltk>=3.6.0",
        "textblob>=0.17.1",
        "yfinance>=0.1.70",
        "pandas-ta==0.3.14b0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.5b2",
            "isort>=5.8.0",
            "flake8>=3.9.2",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nova-analyze=scripts.analyze:main",
        ],
    },
)
