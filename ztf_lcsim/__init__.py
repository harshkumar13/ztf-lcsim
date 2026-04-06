"""
ztf_lcsim – ZTF Light Curve Similarity Search Engine
=====================================================

Quick-start
-----------
>>> from ztf_lcsim import AlerceDownloader, FeatureExtractor, FeatureDatabase, SimilarityIndex
>>> dl  = AlerceDownloader()
>>> lc  = dl.get_lightcurve("ZTF25acemaph")
>>> fe  = FeatureExtractor()
>>> vec = fe.extract(lc)                    # shape (51,)
>>> print(fe.feature_names)
"""

__version__ = "0.1.0"

from .downloader import AlerceDownloader
from .features   import FeatureExtractor, FEATURE_NAMES
from .database   import FeatureDatabase
from .index      import SimilarityIndex
from .plots      import plot_lightcurve, plot_results, plot_feature_space

__all__ = [
    "AlerceDownloader",
    "FeatureExtractor",
    "FeatureDatabase",
    "SimilarityIndex",
    "FEATURE_NAMES",
    "plot_lightcurve",
    "plot_results",
    "plot_feature_space",
]
