"""
ALeRCE / ZTF data downloader.

Downloads light curves and object metadata from the ALeRCE broker.
Falls back to direct REST calls if the ``alerce`` package is unavailable.
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

# ── optional alerce client ────────────────────────────────────────────────────
try:
    from alerce.core import Alerce as _AlerceClient
    _HAS_ALERCE = True
except ImportError:
    _HAS_ALERCE = False
    logger.warning(
        "alerce package not found – using REST API directly. "
        "Install with:  pip install alerce"
    )


class AlerceDownloader:
    """
    Thin wrapper around the ALeRCE API for ZTF data.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for caching downloaded light curves as Parquet files.
        Set to None to disable caching.
    timeout : int
        HTTP request timeout in seconds.
    max_retries : int
        Number of retries on transient errors.
    request_delay : float
        Seconds to wait between requests (polite rate limiting).
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        request_delay: float = 0.1,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_delay = request_delay

        # ── cache ─────────────────────────────────────────────────────────────
        self._cache_dir: Optional[Path] = None
        if cache_dir:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # ── alerce client ─────────────────────────────────────────────────────
        self._client = None
        if _HAS_ALERCE:
            try:
                self._client = _AlerceClient()
            except Exception as exc:
                logger.warning(f"Could not init alerce client ({exc}); using REST.")

    # ══════════════════════════════════════════════════════════════════════════
    # Light curve access
    # ══════════════════════════════════════════════════════════════════════════

    def get_lightcurve(self, oid: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch detections for *oid* and return a cleaned DataFrame.

        Columns guaranteed: ``mjd``, ``fid``, ``magpsf``, ``sigmapsf``
        Optional columns  : ``ra``, ``dec``, ``isdiffpos``

        Returns None on failure.
        """
        # ── cache hit ─────────────────────────────────────────────────────────
        if use_cache and self._cache_dir is not None:
            cached = self._load_from_cache(oid)
            if cached is not None:
                return cached

        # ── fetch ─────────────────────────────────────────────────────────────
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
        """Parallel batch download. Returns dict ``{oid: DataFrame}``."""

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
                oid, lc = fut.result()
                if lc is not None:
                    results[oid] = lc

        return results

    def _download_lightcurve(self, oid: str) -> Optional[pd.DataFrame]:
        if self._client is not None:
            try:
                lc = self._client.query_lightcurve(oid, format="pandas")
                return lc if (lc is not None and not lc.empty) else None
            except Exception as exc:
                logger.debug(f"alerce client failed for {oid}: {exc}")

        # fallback – REST
        return self._rest_get(f"/objects/{oid}/lightcurve", result_key="detections")

    # ══════════════════════════════════════════════════════════════════════════
    # Catalog / object queries
    # ══════════════════════════════════════════════════════════════════════════

    def query_objects(
        self,
        class_name: Optional[str] = None,
        classifier: str = "lc_classifier_top",
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
            e.g. ``"RRL"``, ``"LPV"``, ``"SNIa"``
        classifier : str
            Which classifier to filter by.
        min_probability : float
            Minimum classification probability.
        max_objects : int, optional
            Stop after this many objects.
        page_size : int
            Objects per API page.

        Returns
        -------
        pd.DataFrame
        """
        pages: List[pd.DataFrame] = []
        page = 1

        pbar = tqdm(desc=f"Querying {class_name or 'all'}", unit="obj",
                    disable=not show_progress)

        while True:
            batch = self._fetch_with_retry(
                self._download_object_page,
                class_name, classifier, min_probability, page, page_size,
            )
            if batch is None or batch.empty:
                break

            pages.append(batch)
            pbar.update(len(batch))

            total_so_far = sum(len(p) for p in pages)
            if len(batch) < page_size:
                break
            if max_objects and total_so_far >= max_objects:
                break

            page += 1
            time.sleep(self.request_delay)

        pbar.close()

        if not pages:
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
        if self._client is not None:
            try:
                kw = dict(
                    classifier=classifier,
                    probability=min_probability,
                    page=page,
                    page_size=page_size,
                    format="pandas",
                )
                if class_name:
                    kw["class_name"] = class_name
                result = self._client.query_objects(**kw)
                return result if result is not None else pd.DataFrame()
            except Exception as exc:
                logger.debug(f"alerce client query_objects failed: {exc}")

        # REST fallback
        params: dict = {
            "classifier": classifier,
            "probability": min_probability,
            "page": page,
            "page_size": page_size,
        }
        if class_name:
            params["class_name"] = class_name

        url = f"{_ALERCE_API}/objects"
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", data) if isinstance(data, dict) else data
            return pd.DataFrame(items) if items else pd.DataFrame()
        except Exception as exc:
            logger.debug(f"REST query_objects failed: {exc}")
            return None

    def get_metadata(self, oid: str) -> Optional[dict]:
        """Fetch object-level metadata (coordinates, class, etc.)."""
        if self._client is not None:
            try:
                row = self._client.query_object(oid, format="pandas")
                if row is not None and not row.empty:
                    return row.iloc[0].to_dict()
            except Exception:
                pass
        try:
            resp = requests.get(f"{_ALERCE_API}/objects/{oid}",
                                timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning(f"Could not fetch metadata for {oid}: {exc}")
            return None

    def get_probabilities(self, oid: str) -> Optional[pd.DataFrame]:
        """Return classification probabilities for *oid*."""
        if self._client is not None:
            try:
                return self._client.query_probabilities(oid, format="pandas")
            except Exception:
                pass
        try:
            resp = requests.get(f"{_ALERCE_API}/objects/{oid}/probabilities",
                                timeout=self.timeout)
            resp.raise_for_status()
            return pd.DataFrame(resp.json())
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
                    wait = self.request_delay * (2 ** attempt)
                    time.sleep(wait)
                else:
                    logger.warning(f"{fn.__name__} failed after "
                                   f"{self.max_retries} attempts: {exc}")
        return None

    def _rest_get(
        self, endpoint: str, result_key: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        try:
            resp = requests.get(f"{_ALERCE_API}{endpoint}", timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if result_key:
                data = data.get(result_key, [])
            return pd.DataFrame(data) if data else None
        except Exception as exc:
            logger.debug(f"REST {endpoint} failed: {exc}")
            return None

    # ── disk cache ────────────────────────────────────────────────────────────

    def _cache_path(self, oid: str) -> Path:
        assert self._cache_dir is not None
        subdir = self._cache_dir / oid[:4]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{oid}.parquet"

    def _load_from_cache(self, oid: str) -> Optional[pd.DataFrame]:
        p = self._cache_path(oid)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        return None

    def _save_to_cache(self, oid: str, lc: pd.DataFrame):
        try:
            lc.to_parquet(self._cache_path(oid), compression="snappy",
                          index=False)
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
        logger.debug(f"Missing columns: {missing}")
        return None

    lc = lc.copy()
    lc = lc.dropna(subset=["mjd", "magpsf", "sigmapsf", "fid"])
    lc = lc[lc["sigmapsf"] > 0]
    lc = lc[lc["sigmapsf"] < 1.5]
    lc = lc[(lc["magpsf"] > 10) & (lc["magpsf"] < 24)]
    lc["fid"] = lc["fid"].astype(int)

    # keep only positive-subtraction detections when the column exists
    if "isdiffpos" in lc.columns:
        lc = lc[lc["isdiffpos"].isin([1, "1", "t", True])]

    lc = lc.sort_values("mjd").reset_index(drop=True)
    return lc if not lc.empty else None
