"""
Similarity index – FAISS (preferred) with sklearn fallback.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── FAISS import with detailed diagnostics ────────────────────────────────────
_HAS_FAISS = False
_FAISS_ERROR: Optional[str] = None

try:
    import faiss

    _HAS_FAISS = True

except ModuleNotFoundError as _e:
    _FAISS_ERROR = str(_e)

    if "numpy._core" in str(_e):
        # faiss-cpu >= 1.8 needs numpy >= 2.0
        import numpy as _np

        _np_ver = _np.__version__
        logger.warning(
            f"\n"
            f"  faiss import failed: {_e}\n"
            f"  You have numpy {_np_ver}, but faiss-cpu >= 1.8 needs numpy >= 2.0\n"
            f"\n"
            f"  Fix (choose one):\n"
            f"    A) Upgrade numpy:  pip install 'numpy>=2.0'\n"
            f"    B) Use conda:      conda install -c conda-forge faiss-cpu\n"
            f"    C) Pin old faiss:  pip install 'faiss-cpu<1.8'\n"
            f"\n"
            f"  Falling back to sklearn NearestNeighbors (fully functional, "
            f"just slower for >500k objects).\n"
        )
    else:
        logger.warning(
            f"faiss not installed ({_e}). "
            f"Install with:  pip install faiss-cpu  "
            f"or:  conda install -c conda-forge faiss-cpu\n"
            f"Falling back to sklearn NearestNeighbors."
        )

except Exception as _e:
    _FAISS_ERROR = str(_e)
    logger.warning(
        f"faiss available but failed to load ({type(_e).__name__}: {_e}). "
        f"Falling back to sklearn NearestNeighbors."
    )


class SimilarityIndex:
    """
    Approximate / exact nearest-neighbour index over feature vectors.

    Parameters
    ----------
    index_type : str
        ``"flat"`` exact, ``"ivf"`` approximate, ``"hnsw"`` approximate.
    metric : str
        ``"cosine"`` or ``"l2"``.
    """

    # ── In SimilarityIndex.__init__, add one parameter ────────────────────────────
    def __init__(
        self,
        index_type: str = "flat",
        metric: str = "cosine",
        ivf_nlist: int = 256,
        ivf_nprobe: int = 32,
        hnsw_m: int = 32,
        use_feature_weights: bool = True,  # NEW — apply domain-knowledge weights
    ):
        self.index_type = index_type
        self.metric = metric
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.hnsw_m = hnsw_m
        self.use_feature_weights = use_feature_weights  # NEW

        self._index = None
        self._oids: List[str] = []
        self._scaler = None
        self._feature_medians = None
        self._feature_weights = None  # NEW
        self._n_features: int = 0
        self._is_built: bool = False
        self._X_norm = None

    # ── Replace build() entirely ──────────────────────────────────────────────────
    def build(self, oids: List[str], features: np.ndarray, verbose: bool = True):
        """Build the index from a feature matrix."""
        from .features import get_feature_weights, N_FEATURES
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(features, dtype=np.float32)
        n, d = X.shape
        self._n_features = d
        self._oids = list(oids)

        if verbose:
            logger.info(f"Building index: {n:,} objects × {d} features")

        # ── impute NaN ────────────────────────────────────────────────────────────
        self._feature_medians = np.nanmedian(X, axis=0).astype(np.float32)
        X = _impute(X, self._feature_medians)

        # ── standardise ───────────────────────────────────────────────────────────
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X).astype(np.float32)

        # ── apply domain-knowledge feature weights ────────────────────────────────
        if self.use_feature_weights and d == N_FEATURES:
            self._feature_weights = get_feature_weights()
            X = (X * self._feature_weights).astype(np.float32)
            if verbose:
                # show top-5 most weighted features
                top5 = np.argsort(self._feature_weights)[::-1][:5]
                from .features import FEATURE_NAMES

                top5_names = [
                    (FEATURE_NAMES[i], self._feature_weights[i]) for i in top5
                ]
                logger.info(f"Top-5 weighted features: {top5_names}")
        else:
            self._feature_weights = None

        # ── L2-normalise for cosine similarity ───────────────────────────────────
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            X = (X / norms).astype(np.float32)

        self._X_norm = X.copy()

        if _HAS_FAISS:
            self._index = self._build_faiss(X, d, n)
        else:
            self._index = self._build_sklearn(X)

        self._is_built = True
        if verbose:
            logger.info("Index built ✓")

    def _build_faiss(self, X: np.ndarray, d: int, n: int):
        if self.index_type == "flat":
            idx = faiss.IndexFlatL2(d)
        elif self.index_type == "ivf":
            nlist = min(self.ivf_nlist, max(1, n // 4))
            quantiser = faiss.IndexFlatL2(d)
            idx = faiss.IndexIVFFlat(quantiser, d, nlist)
            idx.train(X)
            idx.nprobe = self.ivf_nprobe
        elif self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(d, self.hnsw_m)
        else:
            raise ValueError(f"Unknown index type: {self.index_type!r}")
        idx.add(X)
        return idx

    def _build_sklearn(self, X: np.ndarray):
        from sklearn.neighbors import NearestNeighbors

        metric = "cosine" if self.metric == "cosine" else "euclidean"
        nn = NearestNeighbors(metric=metric, algorithm="auto")
        nn.fit(X)
        return nn

    # ── search ────────────────────────────────────────────────────────────────

    # ── Update search() to apply weights to query vector too ─────────────────────
    def search(self, query_features, k=20, exclude_self=True):
        if not self._is_built:
            raise RuntimeError("Index not built.")

        q = np.asarray(query_features, dtype=np.float32).reshape(1, -1)
        q = _impute(q, self._feature_medians)
        q = self._scaler.transform(q).astype(np.float32)

        # apply same weights as during build
        if self._feature_weights is not None:
            q = (q * self._feature_weights).astype(np.float32)

        if self.metric == "cosine":
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm

        k_search = min(k + (1 if exclude_self else 0), len(self._oids))

        if _HAS_FAISS:
            dists, indices = self._index.search(q, k_search)
            dists, indices = dists[0], indices[0]
        else:
            dists, indices = self._index.kneighbors(q, n_neighbors=k_search)
            dists, indices = dists[0], indices[0]

        results = []
        for dist, idx in zip(dists, indices):
            if idx < 0 or idx >= len(self._oids):
                continue
            results.append((self._oids[int(idx)], float(dist)))

        if exclude_self:
            results = [(o, d) for o, d in results if d > 1e-6]

        results = results[:k]
        df = pd.DataFrame(results, columns=["oid", "distance"])
        df.insert(0, "rank", range(1, len(df) + 1))
        if self.metric == "cosine":
            df["similarity"] = (1.0 - df["distance"] / 2.0).clip(0, 1)
        else:
            df["similarity"] = 1.0 / (1.0 + df["distance"])
        return df

    def search_by_oid(self, oid: str, k: int = 20) -> pd.DataFrame:
        """Search using an OID already in the index."""
        if oid not in self._oids:
            raise ValueError(f"OID {oid!r} not in index.")
        idx = self._oids.index(oid)
        if self._X_norm is not None:
            vec = self._X_norm[idx]
        elif _HAS_FAISS:
            vec = self._index.reconstruct(idx)
        else:
            raise RuntimeError("Cannot reconstruct vector without stored X_norm.")
        return self.search(vec, k=k, exclude_self=True)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]):
        """Save the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "oids": self._oids,
            "scaler": self._scaler,
            "feature_medians": self._feature_medians,
            "n_features": self._n_features,
            "index_type": self.index_type,
            "metric": self.metric,
            "is_built": self._is_built,
            "X_norm": self._X_norm,
        }

        if _HAS_FAISS and self._index is not None:
            faiss_path = str(path) + ".faiss"
            faiss.write_index(self._index, faiss_path)
            meta["faiss_path"] = faiss_path
        else:
            meta["sklearn_index"] = self._index

        with open(str(path) + ".meta", "wb") as fh:
            pickle.dump(meta, fh, protocol=4)

        logger.info(f"Index saved to {path}")

    def load(self, path: Union[str, Path]) -> "SimilarityIndex":
        """Load a previously saved index. Returns self."""
        path = Path(path)
        meta_path = str(path) + ".meta"

        if not Path(meta_path).exists():
            raise FileNotFoundError(
                f"Index metadata not found at {meta_path}\n"
                "Run scripts/02_build_index.py first."
            )

        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)

        self._oids = meta["oids"]
        self._scaler = meta["scaler"]
        self._feature_medians = meta["feature_medians"]
        self._n_features = meta["n_features"]
        self.index_type = meta["index_type"]
        self.metric = meta["metric"]
        self._is_built = meta["is_built"]
        self._X_norm = meta.get("X_norm")

        if "faiss_path" in meta and _HAS_FAISS:
            self._index = faiss.read_index(meta["faiss_path"])
            if self.index_type == "ivf":
                self._index.nprobe = self.ivf_nprobe
        else:
            self._index = meta.get("sklearn_index")

        logger.info(f"Index loaded: {len(self._oids):,} objects")
        return self

    @property
    def n_objects(self) -> int:
        return len(self._oids)

    def __len__(self) -> int:
        return self.n_objects


# ── helpers ───────────────────────────────────────────────────────────────────


def _impute(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with column medians."""
    X = X.copy()
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        col_idx = np.where(nan_mask.any(axis=0))[0]
        for c in col_idx:
            rows = nan_mask[:, c]
            fill = medians[c] if np.isfinite(medians[c]) else 0.0
            X[rows, c] = fill
    return X
