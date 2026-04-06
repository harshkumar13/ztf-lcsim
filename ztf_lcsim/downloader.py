"""
ALeRCE / ZTF data downloader.
Verified against ALeRCE REST API 2025-01.

Correct API parameters (from GET /classifiers):
  classifier : "lc_classifier"
  class_name : "SNIa" | "SNIbc" | "SNII" | "SLSN" |
               "AGN"  | "QSO"  | "Blazar" | "CV/Nova" | "YSO" |
               "RRL"  | "LPV"  | "E" | "DSCT" | "CEP" | "Periodic-Other"
  probability: float  (0–1)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_ALERCE_API = "https://api.alerce.online/ztf/v1"

# Verified classifier name and class list (GET /classifiers, 2025-01)
_DEFAULT_CLASSIFIER = "lc_classifier"
_KNOWN_CLASSIFIERS = [
    "lc_classifier",
    "lc_classifier_transient",
    "lc_classifier_stochastic",
    "lc_classifier_periodic",
    "lc_classifier_top",
    "stamp_classifier",
]

try:
    from alerce.core import Alerce as _AlerceClient

    _HAS_ALERCE = True
except ImportError:
    _HAS_ALERCE = False
    logger.warning("alerce package not found – using REST API directly.")


class AlerceDownloader:
    """
    Wrapper around the ALeRCE API for ZTF data.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for caching light curves as Parquet files.
    timeout : int
        HTTP timeout in seconds.
    max_retries : int
        Retries on transient failures.
    request_delay : float
        Seconds between requests (polite rate-limiting).
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        request_delay: float = 0.15,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_delay = request_delay

        self._cache_dir: Optional[Path] = None
        if cache_dir:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        if _HAS_ALERCE:
            try:
                self._client = _AlerceClient()
            except Exception as exc:
                logger.warning(f"Could not init alerce client ({exc})")

    # ══════════════════════════════════════════════════════════════════════════
    # Light curve access
    # ══════════════════════════════════════════════════════════════════════════

    def get_lightcurve(
        self, oid: str, use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch detections for *oid*.

        Guaranteed columns: mjd, fid, magpsf, sigmapsf
        Returns None on failure.
        """
        if use_cache and self._cache_dir is not None:
            cached = self._load_from_cache(oid)
            if cached is not None:
                return cached

        lc = self._fetch_with_retry(self._download_lightcurve, oid)
        if lc is None:
            return None

        lc = _clean_lightcurve(lc)
        if lc is None or lc.empty:
            return None

        if use_cache and self._cache_dir is not None:
            self._save_to_cache(oid, lc)

        return lc

    def get_lightcurves_batch(
        self,
        oids: List[str],
        n_workers: int = 4,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Parallel batch download. Returns {oid: DataFrame}."""
        results: Dict[str, pd.DataFrame] = {}

        def _fetch(oid: str):
            time.sleep(self.request_delay)
            return oid, self.get_lightcurve(oid, use_cache=use_cache)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_fetch, oid): oid for oid in oids}
            it = as_completed(futures)
            if show_progress:
                it = tqdm(it, total=len(oids), desc="Downloading light curves")
            for fut in it:
                try:
                    oid, lc = fut.result()
                    if lc is not None:
                        results[oid] = lc
                except Exception as exc:
                    logger.debug(f"Batch fetch error: {exc}")

        return results

    def _download_lightcurve(self, oid: str) -> Optional[pd.DataFrame]:
        """
        Download a light curve for *oid*.

        Strategy:
          1. REST API  — always returns flat DataFrame with correct column names
          2. alerce client — returns nested structure, we unwrap it
        """
        # ── 1. REST API (primary) ─────────────────────────────────────────────────
        try:
            r = requests.get(
                f"{_ALERCE_API}/objects/{oid}/lightcurve",
                timeout=self.timeout,
            )
            if r.ok:
                data = r.json()
                dets = data.get("detections", data) if isinstance(data, dict) else data
                if dets:
                    df = pd.DataFrame(dets)
                    if {"mjd", "magpsf", "sigmapsf", "fid"}.issubset(df.columns):
                        return df
                    logger.debug(
                        f"[{oid}] REST cols OK check failed: {list(df.columns)}"
                    )
            else:
                logger.debug(f"[{oid}] REST status={r.status_code}")
        except Exception as exc:
            logger.debug(f"[{oid}] REST exception: {exc}")

        # ── 2. alerce client (fallback) — unwrap nested structure ─────────────────
        if self._client is not None:
            try:
                raw = self._client.query_lightcurve(oid, format="pandas")
                if raw is not None and not raw.empty:
                    df = _unwrap_alerce_client_lc(raw, oid)
                    if df is not None:
                        return df
            except Exception as exc:
                logger.debug(f"[{oid}] client exception: {exc}")

        return None

    # ══════════════════════════════════════════════════════════════════════════
    # Catalog / object queries
    # ══════════════════════════════════════════════════════════════════════════

    def query_objects(
        self,
        class_name: Optional[str] = None,
        classifier: str = _DEFAULT_CLASSIFIER,
        min_probability: float = 0.6,
        max_objects: Optional[int] = None,
        page_size: int = 1000,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Query the ALeRCE object catalog with pagination.

        Parameters
        ----------
        class_name : str, optional
            One of the verified class names for the chosen classifier.
            For "lc_classifier": SNIa, SNIbc, SNII, SLSN,
                                  AGN, QSO, Blazar, CV/Nova, YSO,
                                  RRL, LPV, E, DSCT, CEP, Periodic-Other
        classifier : str
            Default: "lc_classifier" (verified working 2025-01).
        min_probability : float
            Minimum classification probability [0, 1].
        max_objects : int, optional
            Stop after this many objects total.
        page_size : int
            Objects per API page (max 1000).
        """
        pages: List[pd.DataFrame] = []
        page = 1

        pbar = tqdm(
            desc=f"Querying {class_name or 'all'} [{classifier}]",
            unit=" obj",
            disable=not show_progress,
        )

        while True:
            batch = self._fetch_with_retry(
                self._download_object_page,
                class_name=class_name,
                classifier=classifier,
                min_probability=min_probability,
                page=page,
                page_size=page_size,
            )

            if batch is None or batch.empty:
                break

            pages.append(batch)
            pbar.update(len(batch))

            total = sum(len(p) for p in pages)
            if len(batch) < page_size:
                break
            if max_objects and total >= max_objects:
                break

            page += 1
            time.sleep(self.request_delay)

        pbar.close()

        if not pages:
            logger.warning(
                f"No objects returned for class_name={class_name!r}, "
                f"classifier={classifier!r}, probability>={min_probability}.\n"
                f"  Verify class names with:  python scripts/diag_alerce.py\n"
                f"  lc_classifier classes: SNIa, SNIbc, SNII, SLSN, "
                f"AGN, QSO, Blazar, CV/Nova, YSO, RRL, LPV, E, DSCT, CEP, "
                f"Periodic-Other"
            )
            return pd.DataFrame()

        df = pd.concat(pages, ignore_index=True)
        if max_objects:
            df = df.head(max_objects)
        return df

    def _download_object_page(
        self,
        class_name: Optional[str],
        classifier: str,
        min_probability: float,
        page: int,
        page_size: int,
    ) -> Optional[pd.DataFrame]:
        """Try alerce client first, then REST."""

        # ── alerce Python client ──────────────────────────────────────────────
        if self._client is not None:
            try:
                kw: dict = dict(
                    classifier=classifier,
                    probability=min_probability,
                    page=page,
                    page_size=page_size,
                    format="pandas",
                )
                if class_name:
                    kw["class_name"] = class_name
                result = self._client.query_objects(**kw)
                if result is not None and not result.empty:
                    return result
            except Exception as exc:
                logger.debug(f"Client query_objects (p{page}): {exc}")

        # ── REST fallback ─────────────────────────────────────────────────────
        params: dict = {
            "classifier": classifier,
            "probability": min_probability,
            "page": page,
            "page_size": page_size,
        }
        if class_name:
            params["class_name"] = class_name

        try:
            r = requests.get(
                f"{_ALERCE_API}/objects",
                params=params,
                timeout=self.timeout,
            )
            if not r.ok:
                logger.debug(f"REST /objects status={r.status_code}: {r.text[:200]}")
                return None

            data = r.json()
            # Handle {"items": [...]} or {"objects": [...]} or plain list
            if isinstance(data, dict):
                items = (
                    data.get("items")
                    or data.get("objects")
                    or data.get("results")
                    or []
                )
            elif isinstance(data, list):
                items = data
            else:
                return None

            return pd.DataFrame(items) if items else None

        except Exception as exc:
            logger.debug(f"REST /objects failed (p{page}): {exc}")
            return None

    def get_metadata(self, oid: str) -> Optional[dict]:
        """Fetch object-level metadata dict."""
        if self._client is not None:
            try:
                row = self._client.query_object(oid, format="pandas")
                if row is not None and not row.empty:
                    return row.iloc[0].to_dict()
            except Exception:
                pass
        try:
            r = requests.get(f"{_ALERCE_API}/objects/{oid}", timeout=self.timeout)
            if r.ok:
                return r.json()
        except Exception as exc:
            logger.warning(f"Could not fetch metadata for {oid}: {exc}")
        return None

    def get_probabilities(self, oid: str) -> Optional[pd.DataFrame]:
        """Return all classification probabilities for *oid*."""
        if self._client is not None:
            try:
                return self._client.query_probabilities(oid, format="pandas")
            except Exception:
                pass
        try:
            r = requests.get(
                f"{_ALERCE_API}/objects/{oid}/probabilities",
                timeout=self.timeout,
            )
            if r.ok:
                return pd.DataFrame(r.json())
        except Exception as exc:
            logger.warning(f"Could not fetch probabilities for {oid}: {exc}")
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _fetch_with_retry(self, fn, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_delay * (2**attempt))
                else:
                    logger.warning(
                        f"{fn.__name__} failed after {self.max_retries} "
                        f"attempts: {exc}"
                    )
        return None

    # ── disk cache ────────────────────────────────────────────────────────────

    def _cache_path(self, oid: str) -> Path:
        assert self._cache_dir is not None
        subdir = self._cache_dir / oid[:4]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{oid}.parquet"

    def _load_from_cache(self, oid: str) -> Optional[pd.DataFrame]:
        if self._cache_dir is None:
            return None
        p = self._cache_path(oid)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        return None

    def _save_to_cache(self, oid: str, lc: pd.DataFrame):
        if self._cache_dir is None:
            return
        try:
            lc.to_parquet(self._cache_path(oid), compression="snappy", index=False)
        except Exception as exc:
            logger.debug(f"Cache write failed for {oid}: {exc}")

    # ── module-level helpers ──────────────────────────────────────────────────────


def _clean_lightcurve(lc: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Standardise and quality-filter a raw ALeRCE lightcurve DataFrame."""
    if lc is None or lc.empty:
        return None

    required = {"mjd", "magpsf", "sigmapsf", "fid"}
    if not required.issubset(lc.columns):
        missing = required - set(lc.columns)
        logger.warning(f"Missing columns {missing}. Have: {list(lc.columns)}")
        return None

    lc = lc.copy()

    # ── basic quality filters ─────────────────────────────────────────────────
    n0 = len(lc)
    lc = lc.dropna(subset=["mjd", "magpsf", "sigmapsf", "fid"])
    lc = lc[lc["sigmapsf"] > 0]
    lc = lc[lc["sigmapsf"] < 1.5]
    lc = lc[(lc["magpsf"] > 10) & (lc["magpsf"] < 24)]
    lc["fid"] = lc["fid"].astype(int)
    logger.warning(f"After quality filters: {n0} -> {len(lc)} rows")

    if lc.empty:
        return None

    # ── isdiffpos — PERMISSIVE: skip filter if it would remove everything ─────
    if "isdiffpos" in lc.columns:
        pos_mask = lc["isdiffpos"].isin([1, "1", "t", "T", True, 1.0]) | (
            pd.to_numeric(lc["isdiffpos"], errors="coerce") == 1
        )
        n_pos = pos_mask.sum()
        logger.warning(
            f"isdiffpos: {n_pos}/{len(lc)} positive rows "
            f"(unique={lc['isdiffpos'].unique()[:5]})"
        )
        if n_pos > 0:
            lc = lc[pos_mask]
        else:
            # Don't throw away everything — keep all rows
            logger.warning("isdiffpos would remove ALL rows — skipping filter")

    lc = lc.sort_values("mjd").reset_index(drop=True)
    return lc if not lc.empty else None


def _unwrap_alerce_client_lc(
    raw: pd.DataFrame, oid: str = "?"
) -> Optional[pd.DataFrame]:
    """
    The alerce Python client returns query_lightcurve() as a 1-row DataFrame:
        columns = ['detections', 'non_detections']
        raw['detections'].iloc[0] = list of detection dicts  OR  a DataFrame

    This function unwraps that structure into a flat DataFrame.
    """
    # ── Case A: already flat (has magpsf directly) ────────────────────────────
    if "magpsf" in raw.columns:
        return raw

    # ── Case B: nested {'detections': [...], 'non_detections': [...]} ─────────
    if "detections" in raw.columns:
        inner = raw["detections"].iloc[0]

        # inner is a list of dicts
        if isinstance(inner, list) and inner:
            df = pd.DataFrame(inner)
            logger.debug(f"[{oid}] unwrapped client list → {len(df)} rows")
            return df

        # inner is already a DataFrame
        if isinstance(inner, pd.DataFrame) and not inner.empty:
            logger.debug(f"[{oid}] unwrapped client DataFrame → {len(inner)} rows")
            return inner

        # inner is a dict
        if isinstance(inner, dict):
            df = pd.DataFrame([inner])
            logger.debug(f"[{oid}] unwrapped client dict → {len(df)} rows")
            return df

    logger.debug(
        f"[{oid}] Could not unwrap client response. "
        f"Columns: {list(raw.columns)}, "
        f"shape: {raw.shape}"
    )
    return None
