"""
Feature extraction from ZTF light curves.

Computes 51 features per object (24 per band × 2 bands + 3 cross-band).
All features are finite, float32 numbers; missing values are NaN and
will be median-imputed during index construction.
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

# ── feature names ─────────────────────────────────────────────────────────────
_BAND_FEATURES = [
    "n_obs",
    "timespan",
    "mean_mag",
    "std_mag",
    "mad_mag",
    "skew",
    "kurtosis",
    "amplitude",
    "iqr",
    "beyond1std",
    "percent_amplitude",
    "meanvariance",
    "stetson_k",
    "eta_e",
    "linear_trend",
    "max_slope",
    "rcs",
    "mbrp",
    "fpr_mid20",
    "fpr_mid50",
    "fpr_mid80",
    "log10_period",
    "period_power",
    "period_fap_log10",
]

_CROSS_FEATURES = [
    "gr_mean",
    "gr_std",
    "gr_amplitude",
]

# Per-band features for g (fid=1) and r (fid=2)
FEATURE_NAMES: List[str] = (
    [f"g_{f}" for f in _BAND_FEATURES]
    + [f"r_{f}" for f in _BAND_FEATURES]
    + _CROSS_FEATURES
)

N_FEATURES = len(FEATURE_NAMES)   # 51


# ══════════════════════════════════════════════════════════════════════════════
class FeatureExtractor:
    """
    Extract a fixed-length feature vector from a ZTF multi-band light curve.

    Parameters
    ----------
    bands : list of int
        ZTF filter IDs to process.  1 = g, 2 = r.
    min_obs : int
        Minimum detections required per band; missing band → NaN block.
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
            Feature vector.  Missing values are NaN.
        """
        vec = np.full(N_FEATURES, np.nan, dtype=np.float32)

        if lc is None or lc.empty:
            return vec

        # ── per-band features ─────────────────────────────────────────────────
        band_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for i, fid in enumerate(self.bands[:2]):   # support up to 2 bands
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

            # write into vector (block of 24 per band)
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

        # ─ basic ─────────────────────────────────────────────────────────────
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

        # ─ amplitude / spread ────────────────────────────────────────────────
        feats["amplitude"] = float(np.ptp(m) / 2.0)
        q25, q75 = np.percentile(m, [25, 75])
        feats["iqr"] = float(q75 - q25)
        feats["beyond1std"] = float(np.mean(np.abs(m - mean) > std)) if std > 0 else 0.0

        if median != 0:
            feats["percent_amplitude"] = float(np.max(np.abs(m - median)) / abs(median))
        feats["meanvariance"] = float(std / abs(mean)) if mean != 0 else np.nan

        # ─ Stetson K ─────────────────────────────────────────────────────────
        feats["stetson_k"] = _stetson_k(m, e)

        # ─ von Neumann η ─────────────────────────────────────────────────────
        feats["eta_e"] = _eta_e(m)

        # ─ trend / slope ─────────────────────────────────────────────────────
        if n >= 3:
            slope, _, _, _, se = stats.linregress(t, m)
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

        # ─ range cumulative sum (RCS) ─────────────────────────────────────────
        feats["rcs"] = _rcs(m)

        # ─ median buffer range percentage (MBRP) ─────────────────────────────
        ptp = np.ptp(m)
        if ptp > 0:
            feats["mbrp"] = float(np.mean(np.abs(m - median) < 0.1 * ptp))
        else:
            feats["mbrp"] = 1.0

        # ─ flux percentile ratios ─────────────────────────────────────────────
        _fpr = _flux_percentile_ratios(m)
        feats.update(_fpr)

        # ─ Lomb-Scargle period ────────────────────────────────────────────────
        if n >= 10:
            period, power, fap = _lombscargle_period(
                t, m, e,
                min_period=self.ls_min_period,
                max_period=self.ls_max_period,
                samples_per_peak=self.ls_samples_per_peak,
            )
            feats["log10_period"] = float(np.log10(period)) if period > 0 else np.nan
            feats["period_power"] = float(power)
            feats["period_fap_log10"] = float(-np.log10(fap + 1e-300))
        else:
            feats["log10_period"] = np.nan
            feats["period_power"] = np.nan
            feats["period_fap_log10"] = np.nan

        return feats

    # ── cross-band extraction ─────────────────────────────────────────────────

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

        # Interpolate r onto g epochs (nearest within 1 day)
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
# Variability statistics
# ══════════════════════════════════════════════════════════════════════════════

def _stetson_k(m: np.ndarray, e: np.ndarray) -> float:
    """
    Stetson K statistic (Stetson 1996).

    K = (1/√n) Σ|δᵢ| / √(Σδᵢ²/n)
    where δᵢ = √(n/(n-1)) × (mᵢ − m̄) / σᵢ
    """
    n = len(m)
    if n < 4:
        return np.nan
    w = 1.0 / (e ** 2)
    m_bar = np.sum(w * m) / np.sum(w)
    delta = np.sqrt(float(n) / (n - 1)) * (m - m_bar) / e
    numerator = np.sum(np.abs(delta)) / n
    denominator = np.sqrt(np.sum(delta ** 2) / n)
    return float(numerator / denominator) if denominator > 0 else np.nan


def _eta_e(m: np.ndarray) -> float:
    """von Neumann η ratio (Chen et al. 2020 definition)."""
    n = len(m)
    if n < 4:
        return np.nan
    var = np.var(m, ddof=1)
    if var == 0:
        return np.nan
    eta = np.sum(np.diff(m) ** 2) / ((n - 1) * var)
    return float(eta)


def _rcs(m: np.ndarray) -> float:
    """Range of cumulative sum (Shin et al. 2009)."""
    n = len(m)
    if n < 3:
        return np.nan
    std = np.std(m, ddof=1)
    if std == 0:
        return np.nan
    s = np.cumsum(m - np.mean(m))
    return float((np.max(s) - np.min(s)) / (n * std))


def _flux_percentile_ratios(m: np.ndarray) -> Dict[str, float]:
    """Compute flux-space percentile ratios FPR_mid20/50/80."""
    out: Dict[str, float] = {}
    if len(m) < 10:
        return {k: np.nan for k in ("fpr_mid20", "fpr_mid50", "fpr_mid80")}

    flux = 10.0 ** (-0.4 * (m - np.median(m)))
    p5, p95 = np.percentile(flux, [5, 95])
    denom = p95 - p5
    if denom <= 0:
        return {k: np.nan for k in ("fpr_mid20", "fpr_mid50", "fpr_mid80")}

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
) -> Tuple[float, float, float]:
    """
    Best Lomb-Scargle period, normalised power, and FAP.

    Returns (period_days, power, fap)
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ls = LombScargle(t, m, e, normalization="model")
            freq, power = ls.autopower(
                minimum_frequency=1.0 / max_period,
                maximum_frequency=1.0 / min_period,
                samples_per_peak=samples_per_peak,
            )
        best_idx = np.argmax(power)
        best_freq = freq[best_idx]
        best_power = float(power[best_idx])
        period = float(1.0 / best_freq)
        fap = float(ls.false_alarm_probability(best_power, method="baluev"))
        return period, best_power, fap
    except Exception as exc:
        logger.debug(f"Lomb-Scargle failed: {exc}")
        return np.nan, np.nan, 1.0


# ── misc ──────────────────────────────────────────────────────────────────────

def _safe(v) -> float:
    """Return float(v) if finite, else NaN."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan
