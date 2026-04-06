"""
HDF5 feature database + SQLite metadata store.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, N_FEATURES

logger = logging.getLogger(__name__)

_HDF_CHUNK        = 4096
_HDF_COMPRESS     = "gzip"
_HDF_COMPRESS_OPT = 4


class FeatureDatabase:
    """
    Persistent store for object feature vectors and metadata.

    Parameters
    ----------
    features_path : str or Path
        Path to the HDF5 feature file.
    metadata_path : str or Path
        Path to the SQLite metadata file.
    """

    def __init__(
        self,
        features_path: Union[str, Path],
        metadata_path: Union[str, Path],
    ):
        self.features_path = Path(features_path)
        self.metadata_path = Path(metadata_path)
        self.features_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()

    # ── write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        oids: List[str],
        features: np.ndarray,
        metadata: Optional[pd.DataFrame] = None,
    ):
        """Append oids and their features to the database."""
        if len(oids) == 0:
            return
        features = np.asarray(features, dtype=np.float32)
        assert features.shape == (len(oids), N_FEATURES), (
            f"Expected ({len(oids)}, {N_FEATURES}), got {features.shape}"
        )
        self._hdf_append(oids, features)
        if metadata is not None:
            self._sqlite_insert(oids, metadata)
        else:
            self._sqlite_insert_bare(oids)

    def add_from_dict(
        self,
        feature_dict: Dict[str, np.ndarray],
        metadata: Optional[pd.DataFrame] = None,
    ):
        """Convenience: pass a {oid: feature_vector} dict."""
        oids = list(feature_dict.keys())
        X = np.vstack([feature_dict[o] for o in oids]).astype(np.float32)
        self.add(oids, X, metadata)

    # ── read ──────────────────────────────────────────────────────────────────

    def load_all(self) -> Tuple[List[str], np.ndarray]:
        """Load the full feature matrix."""
        if not self.features_path.exists():
            return [], np.empty((0, N_FEATURES), dtype=np.float32)

        with h5py.File(self.features_path, "r") as fh:
            raw_oids = fh["oids"][:]
            X = fh["features"][:].astype(np.float32)

        oids = [
            o.decode("utf-8") if isinstance(o, (bytes, np.bytes_)) else str(o)
            for o in raw_oids
        ]
        return oids, X

    def load_features(self, oids: List[str]) -> np.ndarray:
        """Return feature vectors for a specific list of OIDs."""
        all_oids, X = self.load_all()
        idx_map = {o: i for i, o in enumerate(all_oids)}
        rows = [idx_map[o] for o in oids if o in idx_map]
        return X[rows]

    def load_metadata(self, oids: Optional[List[str]] = None) -> pd.DataFrame:
        """Load metadata from SQLite."""
        con = sqlite3.connect(self.metadata_path)
        try:
            if oids is None:
                df = pd.read_sql("SELECT * FROM objects", con)
            else:
                placeholders = ",".join("?" * len(oids))
                df = pd.read_sql(
                    f"SELECT * FROM objects WHERE oid IN ({placeholders})",
                    con,
                    params=oids,
                )
        finally:
            con.close()
        return df

    def __len__(self) -> int:
        if not self.features_path.exists():
            return 0
        with h5py.File(self.features_path, "r") as fh:
            return int(fh["oids"].shape[0])

    def __contains__(self, oid: str) -> bool:
        all_oids, _ = self.load_all()
        return oid in set(all_oids)

    def get_feature_names(self) -> List[str]:
        if not self.features_path.exists():
            return FEATURE_NAMES
        with h5py.File(self.features_path, "r") as fh:
            if "feature_names" in fh:
                return [
                    n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n)
                    for n in fh["feature_names"][:]
                ]
        return FEATURE_NAMES

    def stats(self) -> Dict:
        n = len(self)
        try:
            meta = self.load_metadata()
            cls_counts = (
                meta["cls"].value_counts().to_dict()
                if "cls" in meta.columns
                else {}
            )
        except Exception:
            cls_counts = {}
        return {"n_objects": n, "class_counts": cls_counts}

    # ── internal helpers ──────────────────────────────────────────────────────

    def _hdf_append(self, oids: List[str], features: np.ndarray):
        encoded = [o.encode("utf-8") for o in oids]
        n_new = len(oids)

        mode = "a" if self.features_path.exists() else "w"
        with h5py.File(self.features_path, mode) as fh:
            if "oids" not in fh:
                dt = h5py.string_dtype(encoding="utf-8")
                fh.create_dataset(
                    "oids",
                    data=np.array(encoded, dtype=object),
                    dtype=dt,
                    maxshape=(None,),
                    chunks=(min(_HDF_CHUNK, n_new),),
                    compression=_HDF_COMPRESS,
                    compression_opts=_HDF_COMPRESS_OPT,
                )
                fh.create_dataset(
                    "features",
                    data=features,
                    maxshape=(None, N_FEATURES),
                    chunks=(min(_HDF_CHUNK, n_new), N_FEATURES),
                    compression=_HDF_COMPRESS,
                    compression_opts=_HDF_COMPRESS_OPT,
                )
                fn_enc = [n.encode("utf-8") for n in FEATURE_NAMES]
                fh.create_dataset(
                    "feature_names",
                    data=np.array(fn_enc, dtype=object),
                    dtype=dt,
                )
            else:
                old_n = fh["oids"].shape[0]
                new_n = old_n + n_new
                fh["oids"].resize((new_n,))
                fh["oids"][old_n:] = encoded
                fh["features"].resize((new_n, N_FEATURES))
                fh["features"][old_n:] = features

    def _init_sqlite(self):
        con = sqlite3.connect(self.metadata_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                oid         TEXT PRIMARY KEY,
                ra          REAL,
                dec         REAL,
                cls         TEXT,
                probability REAL,
                n_obs_g     INTEGER,
                n_obs_r     INTEGER
            )
        """)
        con.commit()
        con.close()

    def _sqlite_insert(self, oids: List[str], metadata: pd.DataFrame):
        meta_idx = (
            metadata.set_index("oid")
            if "oid" in metadata.columns
            else metadata
        )
        con = sqlite3.connect(self.metadata_path)
        cur = con.cursor()
        for oid in oids:
            row = (
                meta_idx.loc[oid].to_dict()
                if oid in meta_idx.index
                else {}
            )
            cur.execute(
                """INSERT OR REPLACE INTO objects
                   (oid, ra, dec, cls, probability, n_obs_g, n_obs_r)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    oid,
                    row.get("ra"),
                    row.get("dec"),
                    row.get("cls") or row.get("class_name") or row.get("class"),
                    row.get("probability"),
                    row.get("n_obs_g"),
                    row.get("n_obs_r"),
                ),
            )
        con.commit()
        con.close()

    def _sqlite_insert_bare(self, oids: List[str]):
        con = sqlite3.connect(self.metadata_path)
        con.executemany(
            "INSERT OR IGNORE INTO objects (oid) VALUES (?)",
            [(o,) for o in oids],
        )
        con.commit()
        con.close()
