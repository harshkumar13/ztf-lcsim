"""
Feature extraction from ZTF light curves.
66 features per object:
  - 28 per band × 2 bands  = 56  (was 24×2=48)
  - 10 cross-band           = 10  (was 3)
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

# ── per-band feature names ─────────────────────────────────────────────────────
_BAND_FEATURES = [
    # basic statistics
    "n_obs",
    "timespan",
    "mean_mag",
    "std_mag",
    "mad_mag",
    "skew",
    "kurtosis",
    # amplitude / spread
    "amplitude",
    "iqr",
    "beyond1std",
    "percent_amplitude",
    "meanvariance",
    # variability indices
    "stetson_k",
    "eta_e",
    "chi2_flat",          # NEW – χ²/dof vs constant model (high = variable)
    # trends / slopes
    "linear_trend",
    "max_slope",
    "rise_decline_ratio", # NEW – fast-rise/slow-decline asymmetry (SNIa → low)
    # morphology
    "rcs",
    "mbrp",
    # flux percentile ratios
    "fpr_mid20",
    "fpr_mid50",
    "fpr_mid80",
    # periodicity  ← most discriminating block
    "log10_period",
    "period_power",
    "period_fap_log10",
    "n_cycles",           # NEW – timespan/period (periodic: 100s; transient: <1)
    "sf_slope",           # NEW – structure function slope (AGN: 0.3-0.5)
]

# ── cross-band feature names ───────────────────────────────────────────────────
_CROSS_FEATURES = [
    # color level
    "gr_mean",
    "gr_std",
    "gr_amplitude",
    "gr_skew",            # NEW – asymmetric color evolution → transient
    "gr_trend",           # NEW – monotonic color change (SNIa: large positive)
    # period cross-checks  ← strongest discriminators
    "period_consistency", # NEW – |log10(P_g/P_r)|; ~0 periodic, large for noise
    "n_cycles_min",       # NEW – min(n_cycles_g, n_cycles_r)
    # color variability
    "color_osc_power",    # NEW – LS power of g-r(t) time series
    "color_osc_fap_log10",# NEW – significance of color oscillation
    "gr_monotonicity",    # NEW – fraction of time color is monotonically changing
]

FEATURE_NAMES: List[str] = (
    [f"g_{f}" for f in _BAND_FEATURES]
    + [f"r_{f}" for f in _BAND_FEATURES]
    + _CROSS_FEATURES
)

N_FEATURES = len(FEATURE_NAMES)  # 28*2 + 10 = 66

# ── domain-knowledge feature weights ──────────────────────────────────────────
# Applied during similarity search: higher = more important
# Rationale: period features and cross-band period agreement are
# the strongest discriminators between periodic, transient, and stochastic.
_FEATURE_WEIGHTS_BASE = {
    # periodicity block — HIGHEST weight
    "period_fap_log10": 4.0,   # significance of period detection
    "n_cycles":         4.0,   # most powerful: periodic=100s, transient<1
    "period_power":     3.0,
    "log10_period":     3.0,
    # cross-band period agreement — CRITICAL
    "period_consistency": 5.0, # same period in g AND r → periodic variable
    "n_cycles_min":       4.0,
    # color evolution
    "color_osc_fap_log10": 3.0,# oscillating color → periodic
    "color_osc_power":     2.5,
    "gr_trend":            2.5, # monotonic color → transient
    "gr_monotonicity":     2.0,
    # variability strength
    "chi2_flat":           2.0,
    "stetson_k":           1.5,
    "eta_e":               1.5,
    "rise_decline_ratio":  2.0, # asymmetric LC → transient
    "sf_slope":            1.5,
    # amplitude
    "amplitude":           1.2,
    "iqr":                 1.0,
    # basic stats — LOWER weight (not very discriminating)
    "mean_mag":            0.3,
    "std_mag":             0.8,
    "skew":                0.8,
    "kurtosis":            0.8,
    "n_obs":               0.5,
    "timespan":            0.5,
    "linear_trend":        0.5,
    "max_slope":           0.5,
}

def get_feature_weights() -> np.ndarray:
    """
    Return a weight vector aligned with FEATURE_NAMES.
    Used by SimilarityIndex to scale features before distance computation.
    """
    weights = np.ones(N_FEATURES, dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        # strip band prefix (g_ or r_) to look up base name
        base = name[2:] if name.startswith(("g_", "r_")) else name
        weights[i] = _FEATURE_WEIGHTS_BASE.get(base, 1.0)
    # normalise so mean weight = 1
    weights = weights / weights.mean()
    return weights


class FeatureExtractor:
    """
    Extract a fixed-length feature vector from a ZTF multi-band light curve.
    """

    def __init__(
        self,
        bands: List[int] = (1, 2),
        min_obs: int = 5,
        ls_min_period: float = 0.1,
        ls_max_period: float = 500.0,
        ls_samples_per_peak: int = 5,
    ):
        self.bands              = list(bands)
        self.min_obs            = min_obs
        self.ls_min_period      = ls_min_period
        self.ls_max_period      = ls_max_period
        self.ls_samples_per_peak = ls_samples_per_peak

    @property
    def feature_names(self) -> List[str]:
        return FEATURE_NAMES

    @property
    def n_features(self) -> int:
        return N_FEATURES

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self, lc: pd.DataFrame) -> np.ndarray:
        """
        Compute the 66-dim feature vector for one object.
        Missing values are NaN (imputed at index-build time).
        """
        vec = np.full(N_FEATURES, np.nan, dtype=np.float32)
        if lc is None or lc.empty:
            return vec

        # per-band results (store for cross-band use)
        band_results: Dict[int, dict] = {}

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

            feats = self._extract_band(t, m, e)
            band_results[fid] = {"feats": feats, "t": t, "m": m, "e": e}

            offset = i * len(_BAND_FEATURES)
            for j, name in enumerate(_BAND_FEATURES):
                vec[offset + j] = np.float32(_safe(feats.get(name, np.nan)))

        # cross-band features
        xb = self._extract_crossband(lc, band_results)
        offset_xb = 2 * len(_BAND_FEATURES)
        for j, name in enumerate(_CROSS_FEATURES):
            vec[offset_xb + j] = np.float32(_safe(xb.get(name, np.nan)))

        return vec

    def extract_batch(
        self,
        lcs: Dict[str, pd.DataFrame],
        show_progress: bool = True,
    ) -> Tuple[List[str], np.ndarray]:
        from tqdm import tqdm
        oids_out, rows = [], []
        it = lcs.items()
        if show_progress:
            it = tqdm(it, total=len(lcs), desc="Extracting features")
        for oid, lc in it:
            rows.append(self.extract(lc))
            oids_out.append(oid)
        if not rows:
            return [], np.empty((0, N_FEATURES), dtype=np.float32)
        return oids_out, np.vstack(rows).astype(np.float32)

    # ── per-band extraction ───────────────────────────────────────────────────

    def _extract_band(self, t, m, e) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        n = len(m)

        # ── basic statistics ──────────────────────────────────────────────────
        feats["n_obs"]    = float(n)
        feats["timespan"] = float(np.ptp(t)) if n > 1 else 0.0

        mean   = float(np.mean(m))
        median = float(np.median(m))
        std    = float(np.std(m, ddof=1)) if n > 1 else 0.0
        mad    = float(np.median(np.abs(m - median)))

        feats["mean_mag"] = mean
        feats["std_mag"]  = std
        feats["mad_mag"]  = mad
        feats["skew"]     = float(stats.skew(m))    if n >= 3 else 0.0
        feats["kurtosis"] = float(stats.kurtosis(m)) if n >= 4 else 0.0

        # ── amplitude ─────────────────────────────────────────────────────────
        q25, q75 = np.percentile(m, [25, 75])
        feats["amplitude"]         = float(np.ptp(m) / 2.0)
        feats["iqr"]               = float(q75 - q25)
        feats["beyond1std"]        = float(np.mean(np.abs(m - mean) > std)) if std > 0 else 0.0
        feats["percent_amplitude"] = float(np.max(np.abs(m - median)) / abs(median)) if abs(median) > 0 else np.nan
        feats["meanvariance"]      = float(std / abs(mean)) if abs(mean) > 0 else np.nan

        # ── variability indices ───────────────────────────────────────────────
        feats["stetson_k"] = _stetson_k(m, e)
        feats["eta_e"]     = _eta_e(m)
        feats["chi2_flat"] = _chi2_flat(m, e)   # NEW

        # ── trends ────────────────────────────────────────────────────────────
        if n >= 3:
            slope, *_ = stats.linregress(t, m)
            feats["linear_trend"] = float(slope)
        else:
            feats["linear_trend"] = np.nan

        dt, dm = np.diff(t), np.diff(m)
        valid = dt > 0
        if valid.sum() > 0:
            feats["max_slope"] = float(np.max(np.abs(dm[valid] / dt[valid])))
        else:
            feats["max_slope"] = np.nan

        feats["rise_decline_ratio"] = _rise_decline_ratio(t, m)  # NEW

        # ── morphology ────────────────────────────────────────────────────────
        feats["rcs"]  = _rcs(m)
        ptp = np.ptp(m)
        feats["mbrp"] = float(np.mean(np.abs(m - median) < 0.1 * ptp)) if ptp > 0 else 1.0
        feats.update(_flux_percentile_ratios(m))

        # ── periodicity ───────────────────────────────────────────────────────
        if n >= 10:
            period, power, fap = _lombscargle_period(
                t, m, e,
                self.ls_min_period, self.ls_max_period, self.ls_samples_per_peak,
            )
            feats["log10_period"]    = float(np.log10(period)) if period and period > 0 else np.nan
            feats["period_power"]    = float(power)  if power  is not None else np.nan
            feats["period_fap_log10"]= float(-np.log10(fap + 1e-300)) if fap is not None else np.nan

            # n_cycles: how many complete cycles were observed?  ← KEY FEATURE
            if period and period > 0 and feats["timespan"] > 0:
                feats["n_cycles"] = float(feats["timespan"] / period)
            else:
                feats["n_cycles"] = np.nan
        else:
            feats["log10_period"]     = np.nan
            feats["period_power"]     = np.nan
            feats["period_fap_log10"] = np.nan
            feats["n_cycles"]         = np.nan

        # structure function slope  ← KEY FEATURE
        feats["sf_slope"] = _structure_function_slope(t, m)  # NEW

        return feats

    # ── cross-band extraction ─────────────────────────────────────────────────

    def _extract_crossband(self, lc: pd.DataFrame, band_results: Dict) -> Dict[str, float]:
        feats: Dict[str, float] = {}

        g_sub = lc[lc["fid"] == 1]
        r_sub = lc[lc["fid"] == 2]
        if g_sub.empty or r_sub.empty:
            return feats

        g_t = g_sub["mjd"].values
        r_t = r_sub["mjd"].values
        r_m = r_sub["magpsf"].values

        # ── g-r colour time series (nearest within 1 day) ─────────────────────
        gr_times, gr_colors = [], []
        for gt, gm in zip(g_t, g_sub["magpsf"].values):
            idx = int(np.argmin(np.abs(r_t - gt)))
            if np.abs(r_t[idx] - gt) <= 1.0:
                gr_times.append(gt)
                gr_colors.append(gm - r_m[idx])

        if len(gr_colors) >= 3:
            gr   = np.array(gr_colors)
            gr_t = np.array(gr_times)

            feats["gr_mean"]      = float(np.mean(gr))
            feats["gr_std"]       = float(np.std(gr, ddof=1))
            feats["gr_amplitude"] = float(np.ptp(gr) / 2.0)
            feats["gr_skew"]      = float(stats.skew(gr))   # NEW

            # colour trend: positive = getting redder (SNIa-like)
            if len(gr) >= 3:
                slope, *_ = stats.linregress(gr_t, gr)
                feats["gr_trend"] = float(slope)             # NEW

            # fraction of consecutive pairs that are monotonically changing
            feats["gr_monotonicity"] = _monotonicity(gr)     # NEW

            # LS periodicity of the colour curve
            if len(gr) >= 10:
                try:
                    gr_e = np.ones(len(gr)) * np.std(gr, ddof=1) * 0.1
                    cp, cpow, cfap = _lombscargle_period(
                        gr_t, gr, gr_e + 1e-6,
                        self.ls_min_period, self.ls_max_period,
                        self.ls_samples_per_peak,
                    )
                    feats["color_osc_power"]     = float(cpow) if cpow else np.nan   # NEW
                    feats["color_osc_fap_log10"] = float(-np.log10(cfap + 1e-300)) if cfap else np.nan  # NEW
                except Exception:
                    pass

        # ── period consistency between bands  ← MOST IMPORTANT ───────────────
        g_feats = band_results.get(1, {}).get("feats", {})
        r_feats = band_results.get(2, {}).get("feats", {})

        p_g_log = g_feats.get("log10_period")
        p_r_log = r_feats.get("log10_period")
        fap_g   = g_feats.get("period_fap_log10", 0)
        fap_r   = r_feats.get("period_fap_log10", 0)

        if (p_g_log is not None and p_r_log is not None
                and np.isfinite(p_g_log) and np.isfinite(p_r_log)):
            # |log10(P_g) - log10(P_r)| ≈ 0  for periodic (same period both bands)
            # large (>0.5) for noise/transients  ← KEY DISCRIMINATOR
            feats["period_consistency"] = float(abs(p_g_log - p_r_log))   # NEW
        else:
            # if one band has no detected period, set to a large penalty value
            feats["period_consistency"] = 2.0

        # minimum n_cycles across bands
        nc_g = g_feats.get("n_cycles", np.nan)
        nc_r = r_feats.get("n_cycles", np.nan)
        valid_nc = [x for x in [nc_g, nc_r] if x is not None and np.isfinite(x)]
        feats["n_cycles_min"] = float(min(valid_nc)) if valid_nc else np.nan   # NEW

        return feats


# ══════════════════════════════════════════════════════════════════════════════
# NEW feature helpers
# ══════════════════════════════════════════════════════════════════════════════

def _chi2_flat(m: np.ndarray, e: np.ndarray) -> float:
    """
    Reduced chi² against a weighted constant model.
    High value → strongly variable object (not flat).
    Periodic variables and transients both score high;
    but combined with period features, separates them.
    """
    n = len(m)
    if n < 3:
        return np.nan
    w    = 1.0 / (e ** 2 + 1e-12)
    m_w  = np.sum(w * m) / np.sum(w)
    chi2 = np.sum(w * (m - m_w) ** 2)
    return float(chi2 / max(n - 1, 1))


def _rise_decline_ratio(t: np.ndarray, m: np.ndarray) -> float:
    """
    Ratio of time spent brightening vs fading.

    In magnitude space: dm/dt < 0 means getting brighter.
    - SNIa: fast rise, slow decline → small ratio (spend little time brightening)
    - Symmetric periodic: ~1
    - Slow rise fast decline: large ratio
    """
    if len(t) < 4:
        return np.nan
    dt = np.diff(t)
    dm = np.diff(m)
    valid = dt > 0
    if valid.sum() < 3:
        return np.nan
    slopes = dm[valid] / dt[valid]
    # brightening = negative slope in magnitude (mag decreasing = brighter)
    n_rise    = np.sum(slopes < 0)
    n_decline = np.sum(slopes > 0)
    if n_decline == 0:
        return np.nan
    return float(n_rise / n_decline)


def _structure_function_slope(
    t: np.ndarray,
    m: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Slope of log(SF) vs log(Δt) at intermediate timescales.

    SF(Δt) = <(m(t) - m(t+Δt))²>

    AGN:             slope ≈ 0.3–0.5  (slow stochastic variability)
    Periodic vars:   slope ≈ 0–0.2   (periodic structure, not power-law)
    Transients (SNIa): slope ≈ 0.5–1.0 near peak, then flattens
    """
    n = len(t)
    if n < 10:
        return np.nan
    try:
        # compute all pairwise time differences and magnitude differences
        dt_all = []
        dm_all = []
        for i in range(n):
            for j in range(i + 1, n):
                dt_all.append(abs(t[j] - t[i]))
                dm_all.append((m[j] - m[i]) ** 2)

        dt_arr = np.array(dt_all)
        dm_arr = np.array(dm_all)

        # bin by log time-lag
        valid = dt_arr > 0
        if valid.sum() < 5:
            return np.nan

        log_dt = np.log10(dt_arr[valid])
        log_sf = np.log10(dm_arr[valid] + 1e-10)

        # only use middle 50% of time-lag range to avoid edge effects
        lo, hi = np.percentile(log_dt, [25, 75])
        mask   = (log_dt >= lo) & (log_dt <= hi)
        if mask.sum() < 5:
            return np.nan

        slope, *_ = stats.linregress(log_dt[mask], log_sf[mask])
        return float(slope)
    except Exception:
        return np.nan


def _monotonicity(x: np.ndarray) -> float:
    """
    Fraction of consecutive pairs that increase (0 to 1).
    0.5 = symmetric (periodic); near 0 or 1 = monotonic (transient).
    """
    if len(x) < 3:
        return np.nan
    diffs = np.diff(x)
    return float(np.mean(diffs > 0))


# ══════════════════════════════════════════════════════════════════════════════
# Existing helpers (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _stetson_k(m: np.ndarray, e: np.ndarray) -> float:
    n = len(m)
    if n < 4:
        return np.nan
    w    = 1.0 / (e ** 2 + 1e-12)
    m_w  = np.sum(w * m) / np.sum(w)
    delta = np.sqrt(float(n) / max(n - 1, 1)) * (m - m_w) / (e + 1e-12)
    num   = np.sum(np.abs(delta)) / n
    den   = np.sqrt(np.sum(delta ** 2) / n)
    return float(num / den) if den > 0 else np.nan


def _eta_e(m: np.ndarray) -> float:
    n = len(m)
    if n < 4:
        return np.nan
    var = np.var(m, ddof=1)
    if var == 0:
        return np.nan
    return float(np.sum(np.diff(m) ** 2) / ((n - 1) * var))


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
    ref   = np.median(m)
    flux  = 10.0 ** (-0.4 * (m - ref))
    p5, p95 = np.percentile(flux, [5, 95])
    denom = p95 - p5
    if denom <= 0:
        return {"fpr_mid20": np.nan, "fpr_mid50": np.nan, "fpr_mid80": np.nan}
    out: Dict[str, float] = {}
    for (lo, hi), name in [((40,60),"fpr_mid20"),((25,75),"fpr_mid50"),((10,90),"fpr_mid80")]:
        plo, phi = np.percentile(flux, [lo, hi])
        out[name] = float((phi - plo) / denom)
    return out


def _lombscargle_period(
    t, m, e, min_period, max_period, samples_per_peak
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
        best_idx  = int(np.argmax(power))
        best_freq = float(freq[best_idx])
        best_pow  = float(power[best_idx])
        period    = 1.0 / best_freq if best_freq > 0 else None
        fap       = float(ls.false_alarm_probability(best_pow, method="baluev"))
        return period, best_pow, fap
    except Exception as exc:
        logger.debug(f"Lomb-Scargle failed: {exc}")
        return None, None, 1.0


def _safe(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan
