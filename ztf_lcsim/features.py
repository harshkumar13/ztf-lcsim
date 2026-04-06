"""
Feature extraction from ZTF light curves.
Computes 51 features per object (24 per band x 2 bands + 3 cross-band).
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy import stats

logger = logging.getLogger(__name__)

# ── feature name lists ────────────────────────────────────────────────────────
_BAND_FEATURES = [
    "n_obs", "timespan", "mean_mag", "std_mag", "mad_mag",
    "skew", "kurtosis", "amplitude", "iqr", "beyond1std",
    "percent_amplitude", "meanvariance", "stetson_k", "eta_e",
    "linear_trend", "max_slope", "rcs", "mbrp",
    "fpr_mid20", "fpr_mid50", "fpr_mid80",
    "log10_period", "period_power", "period_fap_log10",
]

_CROSS_FEATURES = ["gr_mean", "gr_std", "gr_amplitude"]

FEATURE_NAMES: List[str] = (
    [f"g_{f}" for f in _BAND_FEATURES]
    + [f"r_{f}" for f in _BAND_FEATURES]
    + _CROSS_FEATURES
)

N_FEATURES = len(FEATURE_NAMES)   # 51


class FeatureExtractor:
    """
    Extract a fixed-length feature vector from a ZTF multi-band light curve.

    Parameters
    ----------
    bands : list of int
        ZTF filter IDs to process.  1=g, 2=r.
    min_obs : int
        Minimum detections required per band.
    ls_min_period, ls_max_period : float
        Period search range in days.
    ls_samples_per_peak : int
        Oversampling factor for the Lomb-Scargle frequency grid.
    """

    def __init__(
        self,
        bands: List[int] = (1, 2),
        min_obs: int = 5,
        ls_min_period: float = 0.1,
        ls_max_period: float = 500.0,
        ls_samples_per_peak: int = 5,
    ):
        self.bands = list(bands)
        self.min_obs = min_obs
        self.ls_min_period = ls_min_period
        self.ls_max_period = ls_max_period
        self.ls_samples_per_peak = ls_samples_per_peak

    @property
    def feature_names(self) -> List[str]:
        return FEATURE_NAMES

    @property
    def n_features(self) -> int:
        return N_FEATURES

    # ── main public method ────────────────────────────────────────────────────

    def extract(self, lc: pd.DataFrame) -> np.ndarray:
        """
        Compute the feature vector for one object.

        Parameters
        ----------
        lc : pd.DataFrame
            Light curve with columns ``mjd``, ``magpsf``, ``sigmapsf``, ``fid``.

        Returns
        -------
        np.ndarray, shape (51,), dtype float32
        """
        vec = np.full(N_FEATURES, np.nan, dtype=np.float32)

        if lc is None or lc.empty:
            return vec

        # ── per-band features ─────────────────────────────────────────────────
        band_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for i, fid in enumerate(self.bands[:2]):
            sub = lc[lc["fid"] == fid].copy()
            if len(sub) < self.min_obs:
                continue

            t = sub["mjd"].values.astype(float)
            m = sub["magpsf"].values.astype(float)
            e = sub["sigmapsf"].values.astype(float)

            mask = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
            t, m, e = t[mask], m[mask], e[mask]
            if len(t) < self.min_obs:
                continue

            band_data[fid] = (t, m, e)
            band_feats = self._extract_band(t, m, e)

            offset = i * len(_BAND_FEATURES)
            for j, name in enumerate(_BAND_FEATURES):
                vec[offset + j] = np.float32(_safe(band_feats.get(name, np.nan)))

        # ── cross-band features ───────────────────────────────────────────────
        xb = self._extract_crossband(lc, band_data)
        offset_xb = 2 * len(_BAND_FEATURES)
        for j, name in enumerate(_CROSS_FEATURES):
            vec[offset_xb + j] = np.float32(_safe(xb.get(name, np.nan)))

        return vec

    def extract_batch(
        self,
        lcs: Dict[str, pd.DataFrame],
        show_progress: bool = True,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Extract features for a dict of light curves.

        Returns
        -------
        oids : list of str
        features : ndarray, shape (N, 51)
        """
        from tqdm import tqdm

        oids_out: List[str] = []
        rows: List[np.ndarray] = []

        it = lcs.items()
        if show_progress:
            it = tqdm(it, total=len(lcs), desc="Extracting features")

        for oid, lc in it:
            vec = self.extract(lc)
            oids_out.append(oid)
            rows.append(vec)

        if not rows:
            return [], np.empty((0, N_FEATURES), dtype=np.float32)

        return oids_out, np.vstack(rows).astype(np.float32)

    # ── single-band extraction ────────────────────────────────────────────────

    def _extract_band(
        self,
        t: np.ndarray,
        m: np.ndarray,
        e: np.ndarray,
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        n = len(m)

        feats["n_obs"] = float(n)
        feats["timespan"] = float(np.ptp(t)) if n > 1 else 0.0

        mean = float(np.mean(m))
        median = float(np.median(m))
        std = float(np.std(m, ddof=1)) if n > 1 else 0.0
        mad = float(np.median(np.abs(m - median)))

        feats["mean_mag"] = mean
        feats["std_mag"] = std
        feats["mad_mag"] = mad
        feats["skew"] = float(stats.skew(m)) if n >= 3 else 0.0
        feats["kurtosis"] = float(stats.kurtosis(m)) if n >= 4 else 0.0

        feats["amplitude"] = float(np.ptp(m) / 2.0)
        q25, q75 = np.percentile(m, [25, 75])
        feats["iqr"] = float(q75 - q25)
        feats["beyond1std"] = float(np.mean(np.abs(m - mean) > std)) if std > 0 else 0.0

        if abs(median) > 0:
            feats["percent_amplitude"] = float(
                np.max(np.abs(m - median)) / abs(median)
            )
        else:
            feats["percent_amplitude"] = np.nan

        feats["meanvariance"] = float(std / abs(mean)) if abs(mean) > 0 else np.nan
        feats["stetson_k"] = _stetson_k(m, e)
        feats["eta_e"] = _eta_e(m)

        if n >= 3:
            slope, *_ = stats.linregress(t, m)
            feats["linear_trend"] = float(slope)
        else:
            feats["linear_trend"] = np.nan

        dt = np.diff(t)
        dm = np.diff(m)
        valid = dt > 0
        if valid.sum() > 0:
            feats["max_slope"] = float(np.max(np.abs(dm[valid] / dt[valid])))
        else:
            feats["max_slope"] = np.nan

        feats["rcs"] = _rcs(m)
        ptp = np.ptp(m)
        if ptp > 0:
            feats["mbrp"] = float(np.mean(np.abs(m - median) < 0.1 * ptp))
        else:
            feats["mbrp"] = 1.0

        feats.update(_flux_percentile_ratios(m))

        if n >= 10:
            period, power, fap = _lombscargle_period(
                t, m, e,
                min_period=self.ls_min_period,
                max_period=self.ls_max_period,
                samples_per_peak=self.ls_samples_per_peak,
            )
            feats["log10_period"] = (
                float(np.log10(period)) if (period and period > 0) else np.nan
            )
            feats["period_power"] = float(power) if power is not None else np.nan
            feats["period_fap_log10"] = (
                float(-np.log10(fap + 1e-300)) if fap is not None else np.nan
            )
        else:
            feats["log10_period"] = np.nan
            feats["period_power"] = np.nan
            feats["period_fap_log10"] = np.nan

        return feats

    def _extract_crossband(
        self,
        lc: pd.DataFrame,
        band_data: Dict,
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {}

        g_sub = lc[lc["fid"] == 1]
        r_sub = lc[lc["fid"] == 2]

        if g_sub.empty or r_sub.empty:
            return feats

        g_t = g_sub["mjd"].values
        r_t = r_sub["mjd"].values
        r_m = r_sub["magpsf"].values

        gr_colors = []
        for gt, gm in zip(g_t, g_sub["magpsf"].values):
            idx = np.argmin(np.abs(r_t - gt))
            if np.abs(r_t[idx] - gt) <= 1.0:
                gr_colors.append(gm - r_m[idx])

        if len(gr_colors) >= 3:
            gr = np.array(gr_colors)
            feats["gr_mean"] = float(np.mean(gr))
            feats["gr_std"] = float(np.std(gr, ddof=1))
            feats["gr_amplitude"] = float(np.ptp(gr) / 2.0)

        return feats


# ══════════════════════════════════════════════════════════════════════════════
# Variability statistics (module-level helpers)
# ══════════════════════════════════════════════════════════════════════════════

def _stetson_k(m: np.ndarray, e: np.ndarray) -> float:
    n = len(m)
    if n < 4:
        return np.nan
    w = 1.0 / (e ** 2 + 1e-12)
    m_bar = np.sum(w * m) / np.sum(w)
    delta = np.sqrt(float(n) / max(n - 1, 1)) * (m - m_bar) / (e + 1e-12)
    numerator = np.sum(np.abs(delta)) / n
    denominator = np.sqrt(np.sum(delta ** 2) / n)
    return float(numerator / denominator) if denominator > 0 else np.nan


def _eta_e(m: np.ndarray) -> float:
    n = len(m)
    if n < 4:
        return np.nan
    var = np.var(m, ddof=1)
    if var == 0:
        return np.nan
    eta = np.sum(np.diff(m) ** 2) / ((n - 1) * var)
    return float(eta)


def _rcs(m: np.ndarray) -> float:
    n = len(m)
    if n < 3:
        return np.nan
    std = np.std(m, ddof=1)
    if std == 0:
        return np.nan
    s = np.cumsum(m - np.mean(m))
    return float((np.max(s) - np.min(s)) / (n * std))


def _flux_percentile_ratios(m: np.ndarray) -> Dict[str, float]:
    if len(m) < 10:
        return {"fpr_mid20": np.nan, "fpr_mid50": np.nan, "fpr_mid80": np.nan}

    ref = np.median(m)
    flux = 10.0 ** (-0.4 * (m - ref))
    p5, p95 = np.percentile(flux, [5, 95])
    denom = p95 - p5
    if denom <= 0:
        return {"fpr_mid20": np.nan, "fpr_mid50": np.nan, "fpr_mid80": np.nan}

    out: Dict[str, float] = {}
    for (lo, hi), name in [
        ((40, 60), "fpr_mid20"),
        ((25, 75), "fpr_mid50"),
        ((10, 90), "fpr_mid80"),
    ]:
        plo, phi = np.percentile(flux, [lo, hi])
        out[name] = float((phi - plo) / denom)

    return out


def _lombscargle_period(
    t: np.ndarray,
    m: np.ndarray,
    e: np.ndarray,
    min_period: float,
    max_period: float,
    samples_per_peak: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ls = LombScargle(t, m, e, normalization="model")
            freq, power = ls.autopower(
                minimum_frequency=1.0 / max_period,
                maximum_frequency=1.0 / min_period,
                samples_per_peak=samples_per_peak,
            )
        if len(power) == 0:
            return None, None, 1.0
        best_idx = int(np.argmax(power))
        best_freq = float(freq[best_idx])
        best_power = float(power[best_idx])
        period = 1.0 / best_freq if best_freq > 0 else None
        fap = float(ls.false_alarm_probability(best_power, method="baluev"))
        return period, best_power, fap
    except Exception as exc:
        logger.debug(f"Lomb-Scargle failed: {exc}")
        return None, None, 1.0


def _safe(v) -> float:
    """Return float(v) if finite, else NaN."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan
