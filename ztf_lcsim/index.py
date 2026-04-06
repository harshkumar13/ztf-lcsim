"""
Similarity index – FAISS (preferred) with sklearn fallback.

The index works in L2 space on **L2-normalised** feature vectors, which is
equivalent to cosine similarity.  NaN features are median-imputed before
normalisation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── try to import FAISS ───────────────────────────────────────────────────────
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.warning(
        "faiss not installed – falling back to sklearn NearestNeighbors. "
        "For large datasets install with:  pip install faiss-cpu"
    )


class SimilarityIndex:
    """
    Approximate / exact nearest-neighbour index over feature vectors.

    Parameters
    ----------
    index_type : str
        ``"flat"``  – exact L2 search (IndexFlatL2)
        ``"ivf"``   – IVF approximate search
        ``"hnsw"``  – HNSW approximate search
    metric : str
        ``"cosine"`` or ``"l2"``
    ivf_nlist, ivf_nprobe : int
        IVF hyperparameters.
    hnsw_m : int
        HNSW connections-per-node.

    Notes
    -----
    When FAISS is not available, all ``index_type`` values fall back to
    sklearn's ``NearestNeighbors`` (brute-force, accurate but slower).
    """

    def __init__(
        self,
        index_type: str = "flat",
        metric: str = "cosine",
        ivf_nlist: int = 256,
        ivf_nprobe: int = 32,
        hnsw_m: int = 32,
    ):
        self.index_type = index_type
        self.metric = metric
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.hnsw_m = hnsw_m

        self._index = None              # FAISS index or sklearn NearestNeighbors
        self._oids: List[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._feature_medians: Optional[np.ndarray] = None
        self._n_features: int = 0
        self._is_built: bool = False

    # ══════════════════════════════════════════════════════════════════════════
    # Build
    # ══════════════════════════════════════════════════════════════════════════

    def build(
        self,
        oids: List[str],
        features: np.ndarray,
        verbose: bool = True,
    ):
        """
        Build the index from a feature matrix.

        Parameters
        ----------
        oids : list of str, length N
        features : float32 ndarray, shape (N, D)
        """
        X = np.asarray(features, dtype=np.float32)
        n, d = X.shape
        self._n_features = d
        self._oids = list(oids)

        if verbose:
            logger.info(f"Building index: {n:,} objects, {d} features")

        # ── impute NaN ────────────────────────────────────────────────────────
        self._feature_medians = np.nanmedian(X, axis=0).astype(np.float32)
        X = _impute(X, self._feature_medians)

        # ── standardise ───────────────────────────────────────────────────────
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X).astype(np.float32)

        # ── L2-normalise (cosine) ─────────────────────────────────────────────
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            X = (X / norms).astype(np.float32)

        # ── build index ───────────────────────────────────────────────────────
        if _HAS_FAISS:
            self._index = self._build_faiss(X, d, n)
        else:
            self._index = self._build_sklearn(X)

        self._is_built = True
        if verbose:
            logger.info("Index built successfully.")

    def _build_faiss(self, X: np.ndarray, d: int, n: int):
        if self.index_type == "flat":
            idx = faiss.IndexFlatL2(d)
        elif self.index_type == "ivf":
            nlist = min(self.ivf_nlist, n // 4)
            quantiser = faiss.IndexFlatL2(d)
            idx = faiss.IndexIVFFlat(quantiser, d, nlist)
            idx.train(X)
            idx.nprobe = self.ivf_nprobe
        elif self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(d, self.hnsw_m)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        idx.add(X)
        return idx

    def _build_sklearn(self, X: np.ndarray):
        from sklearn.neighbors import NearestNeighbors
        metric = "cosine" if self.metric == "cosine" else "euclidean"
        nn = NearestNeighbors(metric=metric, algorithm="auto")
        nn.fit(X)
        return nn

    # ══════════════════════════════════════════════════════════════════════════
    # Search
    # ══════════════════════════════════════════════════════════════════════════

    def search(
        self,
        query_features: np.ndarray,
        k: int = 20,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """
        Return the *k* most similar objects.

        Parameters
        ----------
        query_features : ndarray, shape (D,) or (1, D)
        k : int
            Number of nearest neighbours.
        exclude_self : bool
            If the query is in the database, exclude it from results.

        Returns
        -------
        pd.DataFrame with columns: rank, oid, distance, similarity
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build() first.")

        q = np.asarray(query_features, dtype=np.float32).reshape(1, -1)
        q = _impute(q, self._feature_medians)
        q = self._scaler.transform(q).astype(np.float32)

        if self.metric == "cosine":
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm

        k_search = k + (1 if exclude_self else 0)
        k_search = min(k_search, len(self._oids))

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
            oid = self._oids[int(idx)]
            results.append((oid, float(dist)))

        if exclude_self and results:
            # remove perfect match (dist ≈ 0) if present
            results = [(o, d) for o, d in results if d > 1e-6]

        results = results[:k]
        df = pd.DataFrame(results, columns=["oid", "distance"])
        df.insert(0, "rank", range(1, len(df) + 1))

        # cosine similarity = 1 − (L2² / 2) for unit vectors
        if self.metric == "cosine":
            df["similarity"] = 1.0 - df["distance"] / 2.0
        else:
            df["similarity"] = 1.0 / (1.0 + df["distance"])

        return df

    def search_by_oid(self, oid: str, k: int = 20) -> pd.DataFrame:
        """Search using an OID already in the index."""
        if oid not in self._oids:
            raise ValueError(f"OID {oid!r} not in index.")
        idx = self._oids.index(oid)
        if _HAS_FAISS:
            vec = self._index.reconstruct(idx)
        else:
            # for sklearn we need to recover the training vector
            raise RuntimeError(
                "search_by_oid with sklearn backend requires storing vectors – "
                "use search() with the raw feature vector instead."
            )
        return self.search(vec, k=k, exclude_self=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, path: Union[str, Path]):
        """Save the index and all metadata to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "oids":             self._oids,
            "scaler":           self._scaler,
            "feature_medians":  self._feature_medians,
            "n_features":       self._n_features,
            "index_type":       self.index_type,
            "metric":           self.metric,
            "is_built":         self._is_built,
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
        """Load a previously saved index.  Returns *self* for chaining."""
        path = Path(path)
        with open(str(path) + ".meta", "rb") as fh:
            meta = pickle.load(fh)

        self._oids            = meta["oids"]
        self._scaler          = meta["scaler"]
        self._feature_medians = meta["feature_medians"]
        self._n_features      = meta["n_features"]
        self.index_type       = meta["index_type"]
        self.metric           = meta["metric"]
        self._is_built        = meta["is_built"]

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
    """Replace NaN with column medians (in-place copy)."""
    X = X.copy()
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        col_idx = np.where(nan_mask.any(axis=0))[0]
        for c in col_idx:
            rows = nan_mask[:, c]
            X[rows, c] = medians[c] if np.isfinite(medians[c]) else 0.0
    return X
