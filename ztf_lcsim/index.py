"""
Similarity index — FAISS (preferred) with sklearn fallback.
Supports domain-knowledge feature weights and ML probability augmentation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── FAISS import ──────────────────────────────────────────────────────────────
_HAS_FAISS = False
_FAISS_ERROR: Optional[str] = None

try:
    import faiss

    _HAS_FAISS = True
except Exception as _e:
    _FAISS_ERROR = str(_e)
    if "numpy._core" in str(_e):
        import numpy as _np

        logger.warning(
            f"faiss import failed (numpy version mismatch: have {_np.__version__}).\n"
            f"Fix: conda install -c conda-forge faiss-cpu\n"
            f"Falling back to sklearn NearestNeighbors."
        )
    else:
        logger.warning(
            f"faiss not available ({_e}). " f"Falling back to sklearn NearestNeighbors."
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
    use_feature_weights : bool
        Apply domain-knowledge feature weights before similarity computation.
    ml_weight : float
        Extra weight multiplier applied to ML class probability columns
        when an ML augmenter is used.
    """

    def __init__(
        self,
        index_type: str = "flat",
        metric: str = "cosine",
        ivf_nlist: int = 256,
        ivf_nprobe: int = 32,
        hnsw_m: int = 32,
        use_feature_weights: bool = True,
        ml_weight: float = 3.0,
    ):
        self.index_type = index_type
        self.metric = metric
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.hnsw_m = hnsw_m
        self.use_feature_weights = use_feature_weights
        self.ml_weight = ml_weight

        self._index = None
        self._oids: List[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._feature_medians: Optional[np.ndarray] = None
        self._feature_weights: Optional[np.ndarray] = None
        self._n_features: int = 0
        self._is_built: bool = False
        self._X_norm: Optional[np.ndarray] = None
        self._ml_augmenter = None

    # ══════════════════════════════════════════════════════════════════════════
    # Build
    # ══════════════════════════════════════════════════════════════════════════

    def build(
        self,
        oids: List[str],
        features: np.ndarray,
        verbose: bool = True,
        ml_augmenter=None,
    ):
        """
        Build the index from a feature matrix.

        Parameters
        ----------
        oids : list of str, length N
        features : float32 ndarray, shape (N, D)
        ml_augmenter : MLFeatureAugmenter, optional
            If provided, class probabilities are appended to features.
        """
        from .features import get_feature_weights, N_FEATURES

        X = np.asarray(features, dtype=np.float32)
        n, d = X.shape
        self._n_features = d
        self._oids = list(oids)
        self._ml_augmenter = ml_augmenter

        if verbose:
            logger.info(f"Building index: {n:,} objects × {d} features")

        # ── impute NaN / Inf ──────────────────────────────────────────────────
        self._feature_medians = np.nanmedian(X, axis=0).astype(np.float32)
        X = _impute(X, self._feature_medians)

        # ── ML augmentation ────────────────────────────────────────────────────
        if ml_augmenter is not None and ml_augmenter.is_fitted:
            proba = ml_augmenter.predict_proba(X)  # (N, n_classes)
            if verbose:
                logger.info(
                    f"ML augmentation: +{proba.shape[1]} class probability "
                    f"features  classes={ml_augmenter.classes_}"
                )
            X = np.hstack([X, proba]).astype(np.float32)
        else:
            if verbose and ml_augmenter is not None:
                logger.warning("ML augmenter provided but not fitted — skipping.")

        # ── standardise ───────────────────────────────────────────────────────
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X).astype(np.float32)

        # ── domain-knowledge + ML feature weights ─────────────────────────────
        if self.use_feature_weights:
            base_w = get_feature_weights()  # shape (D,)
            total_d = X.shape[1]
            if total_d > len(base_w):
                # ML probability columns
                n_prob = total_d - len(base_w)
                prob_w = np.full(n_prob, self.ml_weight, dtype=np.float32)
                self._feature_weights = np.concatenate([base_w, prob_w])
            else:
                self._feature_weights = base_w[:total_d]

            self._feature_weights = (
                self._feature_weights / self._feature_weights.mean()
            ).astype(np.float32)
            X = (X * self._feature_weights).astype(np.float32)

            if verbose:
                top5 = np.argsort(self._feature_weights)[::-1][:5]
                logger.info(
                    f"Top-5 weight indices: {top5.tolist()} "
                    f"(values: {self._feature_weights[top5].tolist()})"
                )
        else:
            self._feature_weights = None

        # ── L2-normalise for cosine similarity ────────────────────────────────
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            X = (X / norms).astype(np.float32)

        self._X_norm = X.copy()

        # ── build index ───────────────────────────────────────────────────────
        if _HAS_FAISS:
            self._index = self._build_faiss(X, X.shape[1], n)
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
            raise ValueError(f"Unknown index_type: {self.index_type!r}")
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
        Return the k most similar objects.

        Parameters
        ----------
        query_features : ndarray, shape (D,) or (1, D)
        k : int
        exclude_self : bool
            Exclude the query if it is in the index (distance ≈ 0).

        Returns
        -------
        pd.DataFrame with columns: rank, oid, distance, similarity
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build() first.")

        q = np.asarray(query_features, dtype=np.float32).reshape(1, -1)

        # impute using raw-feature medians (before ML / scaling)
        med = self._feature_medians
        q = _impute(q, med[: q.shape[1]] if med is not None else med)

        # ML augmentation (must match build-time transform)
        if self._ml_augmenter is not None and self._ml_augmenter.is_fitted:
            proba = self._ml_augmenter.predict_proba(q)
            q = np.hstack([q, proba]).astype(np.float32)

        # standardise
        q = self._scaler.transform(q).astype(np.float32)

        # feature weights
        if self._feature_weights is not None:
            fw = self._feature_weights
            if q.shape[1] <= len(fw):
                q = (q * fw[: q.shape[1]]).astype(np.float32)

        # L2-normalise
        if self.metric == "cosine":
            norm = float(np.linalg.norm(q))
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
            raise RuntimeError(
                "search_by_oid unavailable with sklearn backend without stored X_norm."
            )
        return self.search(vec, k=k, exclude_self=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, path: Union[str, Path]):
        """Save index and all metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "oids": self._oids,
            "scaler": self._scaler,
            "feature_medians": self._feature_medians,
            "feature_weights": self._feature_weights,
            "n_features": self._n_features,
            "index_type": self.index_type,
            "metric": self.metric,
            "is_built": self._is_built,
            "X_norm": self._X_norm,
            "ml_weight": self.ml_weight,
            "ml_augmenter": self._ml_augmenter,
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
                f"Index not found at {meta_path}\n"
                "Run:  python scripts/02_build_index.py"
            )

        with open(meta_path, "rb") as fh:
            meta = pickle.load(fh)

        self._oids = meta["oids"]
        self._scaler = meta["scaler"]
        self._feature_medians = meta["feature_medians"]
        self._feature_weights = meta.get("feature_weights")
        self._n_features = meta["n_features"]
        self.index_type = meta["index_type"]
        self.metric = meta["metric"]
        self._is_built = meta["is_built"]
        self._X_norm = meta.get("X_norm")
        self.ml_weight = meta.get("ml_weight", 3.0)
        self._ml_augmenter = meta.get("ml_augmenter")

        if "faiss_path" in meta and _HAS_FAISS:
            self._index = faiss.read_index(meta["faiss_path"])
            if self.index_type == "ivf":
                self._index.nprobe = self.ivf_nprobe
        else:
            self._index = meta.get("sklearn_index")

        logger.info(
            f"Index loaded: {len(self._oids):,} objects "
            f"({'FAISS' if _HAS_FAISS else 'sklearn'}) "
            f"ml={'yes' if self._ml_augmenter is not None else 'no'}"
        )
        return self

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def n_objects(self) -> int:
        return len(self._oids)

    def __len__(self) -> int:
        return self.n_objects


# ── helpers ───────────────────────────────────────────────────────────────────
# ── replace _impute() at the bottom of ztf_lcsim/index.py ───────────────────


def _impute(X: np.ndarray, medians: Optional[np.ndarray]) -> np.ndarray:
    """
    Replace NaN / Inf with column medians.
    Columns where ALL values are NaN are filled with 0.
    """
    X = X.copy()
    if medians is None:
        return np.where(np.isfinite(X), X, 0.0).astype(X.dtype)

    nan_mask = ~np.isfinite(X)
    if not nan_mask.any():
        return X

    n_cols = min(X.shape[1], len(medians))
    for c in range(n_cols):
        rows = nan_mask[:, c]
        if not rows.any():
            continue
        fill = float(medians[c]) if np.isfinite(medians[c]) else 0.0
        X[rows, c] = fill

    # any columns beyond the medians array length
    for c in range(n_cols, X.shape[1]):
        rows = nan_mask[:, c]
        if rows.any():
            X[rows, c] = 0.0

    return X
