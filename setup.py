from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="ztf-lcsim",
    version="0.1.0",
    description="ZTF Light Curve Similarity Search Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    python_requires=">=3.8",
    packages=find_packages(exclude=["scripts*", "tests*"]),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "astropy>=5.0",
        "alerce>=1.0",
        "scikit-learn>=1.0",
        "h5py>=3.4",
        "matplotlib>=3.4",
        "seaborn>=0.12",
        "tqdm>=4.62",
        "pyyaml>=6.0",
        "requests>=2.26",
        "click>=8.0",
        "joblib>=1.1",
        "pyarrow>=7.0",      # parquet cache
    ],
    extras_require={
        "faiss":     ["faiss-cpu>=1.7"],
        "faiss-gpu": ["faiss-gpu>=1.7"],
        "dev":       ["pytest>=7", "jupyter", "ipykernel", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "ztf-build-db=scripts.01_build_database:cli",
            "ztf-build-idx=scripts.02_build_index:cli",
            "ztf-search=scripts.03_search:cli",
        ],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
    ],
)
