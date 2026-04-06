"""
Comprehensive ZTF light curve feature extraction.
115 features per object:
  45 per-band  × 2 bands  =  90  (was 24×2=48)
  25 cross-band            =  25  (was  3)
  ─────────────────────────────
  Total                      115

Key new additions:
  • Fourier R21/phi21/R31/phi31 — shape fingerprint of folded LC
  • n_local_maxima            — undulation / cycle count from raw LC
  • permutation_entropy       — periodic=low, stochastic=high
  • rise_rate / decline_rate  — asymmetry (SNIa: fast rise slow decline)
  • structure function lags    — distinguishes AGN from periodic
  • gr_at_peak / gr_late      — post-peak reddening (SNIa signature)
  • composite LC features     — band-agnostic morphology
  • phase_offset_gr           — phase lag between bands
  • delta_color_15d           — reddening 15 days after peak
"""

from __future__ import annotations

import logging
import warnings
from itertools import permutations as _perms
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy import stats
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Feature name registry
# ══════════════════════════════════════════════════════════════════════════════

_BAND_FEATURES = [
    # ── observation sampling (2) ──────────────────────────────────────────────
    "n_obs",                    # number of clean detections
    "timespan",                 # last − first epoch (days)
    # ── magnitude statistics (8) ─────────────────────────────────────────────
    "mean_mag",
    "median_mag",
    "std_mag",
    "mad_mag",                  # median absolute deviation
    "skew",
    "kurtosis",
    "beyond1std",               # fraction of points > 1σ from mean
    "beyond2std",               # fraction of points > 2σ from mean
    # ── amplitude / spread (7) ───────────────────────────────────────────────
    "amplitude",                # (max − min) / 2 in magnitudes
    "iqr",                      # interquartile range
    "percent_amplitude",        # max|m − median| / |median|
    "meanvariance",             # σ / |μ|
    "fpr_mid20",                # flux percentile ratio [40,60]
    "fpr_mid50",                # flux percentile ratio [25,75]
    "fpr_mid80",                # flux percentile ratio [10,90]
    # ── variability indices (5) ──────────────────────────────────────────────
    "stetson_k",                # Stetson K (1996)
    "eta_e",                    # von Neumann η
    "chi2_flat",                # reduced χ² vs weighted constant
    "rcs",                      # range of cumulative sum
    "mbrp",                     # median buffer range percentage
    # ── trends (2) ───────────────────────────────────────────────────────────
    "linear_trend",             # slope of linear fit (mag/day)
    "max_slope",                # max |Δm/Δt| between consecutive obs
    # ── morphological shape (7) ──────────────────────────────────────────────
    "peak_mag_norm",            # (peak_mag − mean) / std  — how extreme
    "peak_time_frac",           # normalised time of brightness peak [0,1]
    "rise_rate",                # mean brightening rate (mag/day) pre-peak
    "decline_rate",             # mean fading rate post-peak
    "rise_decline_asymmetry",   # (rise − decline) / (rise + decline)
    "n_local_maxima",           # ← KEY: undulation count (periodic=50+, SNIa=1)
    "fwhm_frac",                # fraction of LC above half-max brightness
    # ── periodicity (5) ──────────────────────────────────────────────────────
    "log10_period",
    "period_power",
    "period_fap_log10",
    "n_cycles",                 # ← KEY: timespan/period (periodic=100+, SNIa<1)
    "period_snr",               # LS peak / local noise floor
    # ── Fourier decomposition (4) ← THE shape fingerprint ───────────────────
    "R21",                      # A₂/A₁ — second harmonic strength
    "phi21",                    # φ₂ − 2φ₁ — phase offset
    "R31",                      # A₃/A₁
    "phi31",                    # φ₃ − 3φ₁
    # ── structure function (3) ───────────────────────────────────────────────
    "sf_slope",                 # log-log SF slope (AGN≈0.4, periodic≈0)
    "sf_1d_norm",               # SF(~1 day) / variance
    "sf_10d_norm",              # SF(~10 days) / variance
    # ── information theory (2) ───────────────────────────────────────────────
    "permutation_entropy",      # ← KEY: periodic=low, stochastic=high
    "sample_entropy",           # regularity / predictability
]

_CROSS_FEATURES = [
    # ── color statistics (6) ─────────────────────────────────────────────────
    "gr_mean",
    "gr_std",
    "gr_amplitude",             # (max − min) / 2 of g−r series
    "gr_skew",
    "gr_min",                   # bluest epoch
    "gr_max",                   # reddest epoch
    # ── color at specific epochs (4) ← diagnostic of SNIa vs periodic ────────
    "gr_at_peak",               # g−r color at epoch of peak brightness
    "gr_early",                 # mean g−r in first 20% of timespan
    "gr_mid",                   # mean g−r in middle 20%
    "gr_late",                  # mean g−r in last 20%
    # ── color evolution (5) ──────────────────────────────────────────────────
    "gr_trend",                 # overall color slope (mag/day)
    "gr_trend_late",            # ← KEY for SNIa: post-peak reddening slope
    "gr_monotonicity",          # fraction of monotonic color changes
    "gr_peak_to_late_delta",    # ← KEY: color(late) − color(at_peak)
    "delta_color_15d",          # ← KEY: Δ(g−r) in 15 days after peak
    # ── color oscillation (3) ────────────────────────────────────────────────
    "color_osc_power",
    "color_osc_fap_log10",
    "color_period_ratio",       # color period / g-band period
    # ── cross-band period / timing (4) ───────────────────────────────────────
    "period_consistency",       # |log10 P_g − log10 P_r| (≈0 periodic)
    "n_cycles_min",
    "amplitude_ratio_gr",       # amplitude_g / amplitude_r
    "phase_offset_gr",          # ← phase lag between bands
    # ── composite LC (3) ← band-agnostic ─────────────────────────────────────
    "composite_n_maxima",       # undulations in combined flux LC
    "composite_chi2",           # variability of combined LC
    "composite_amplitude",      # amplitude in combined flux space
]

# ── ADD these entries to _BAND_FEATURES (after "sample_entropy") ─────────────

_BAND_FEATURES_NEW = [
    # ── peak-normalized shape fingerprint ← brightness-scale invariant ────────
    # dm = m(t_peak + offset) - m_peak  (positive = fainter than peak)
    # SNIa: dm_p15 ≈ +1.0 mag, very characteristic
    # RRL:  dm values oscillate, no monotonic trend
    "dm_m30",        # Δmag 30 days BEFORE peak
    "dm_m20",        # Δmag 20 days before peak
    "dm_m10",        # Δmag 10 days before peak
    "dm_p5",         # Δmag  5 days AFTER peak
    "dm_p10",        # Δmag 10 days after peak
    "dm_p15",        # Δmag 15 days after peak  ← dm15 equivalent
    "dm_p20",        # Δmag 20 days after peak
    "dm_p30",        # Δmag 30 days after peak
    "dm_p50",        # Δmag 50 days after peak
    # ── temporal variability evolution ← how uniqueness evolves over time ──────
    "var_early",     # std in first third of timespan
    "var_mid",       # std in middle third
    "var_late",      # std in last third
    "var_trend",     # (var_late−var_early)/(var_late+var_early)  transient→+1
    "var_n_active",  # number of thirds with significant variability
    # ── pre / post peak undulations ← periodic=equal both, SNIa=only post ──────
    "n_maxima_pre",  # undulations BEFORE peak  (periodic: ~100, SNIa: 0)
    "n_maxima_post", # undulations AFTER peak   (periodic: ~100, SNIa: 0-1)
    "maxima_ratio",  # n_post/(n_pre+n_post)     periodic: 0.5, SNIa: >0.8
]

# Update the master list:
_BAND_FEATURES = _BAND_FEATURES + _BAND_FEATURES_NEW

_CROSS_FEATURES_NEW = [
    # ── color at fixed epochs relative to peak ← KEY SNIa discriminator ───────
    "gr_m30",           # g-r color 30 days BEFORE peak
    "gr_m10",           # g-r color 10 days before peak
    "gr_p5",            # g-r color  5 days AFTER peak
    "gr_p15",           # g-r color 15 days after peak  ← post-peak reddening
    "gr_p30",           # g-r color 30 days after peak
    # ── band shape similarity ─────────────────────────────────────────────────
    "band_xcorr",       # cross-correlation of normalized g and r shapes
    "band_shape_rms",   # RMS difference of normalized shapes
    "peak_flux_ratio",  # peak_flux_g / peak_flux_r  (color at peak)
]

_CROSS_FEATURES = _CROSS_FEATURES + _CROSS_FEATURES_NEW

FEATURE_NAMES: List[str] = (
    [f"g_{f}" for f in _BAND_FEATURES]
    + [f"r_{f}" for f in _BAND_FEATURES]
    + _CROSS_FEATURES
)
N_FEATURES = len(FEATURE_NAMES)   # 45*2 + 25 = 115

# ══════════════════════════════════════════════════════════════════════════════
# Domain-knowledge feature weights
# Higher = more discriminating between object classes
# ══════════════════════════════════════════════════════════════════════════════

_WEIGHTS: Dict[str, float] = {
    # ── THE STRONGEST CLASS DISCRIMINATORS ───────────────────────────────────
    "n_local_maxima":       6.0,  # periodic=50+  SNIa=1   AGN=2-5
    "n_cycles":             6.0,  # periodic=100  SNIa<1   AGN<1
    "period_consistency":   6.0,  # periodic≈0    SNIa≫0
    "n_cycles_min":         5.0,
    "permutation_entropy":  5.0,  # periodic=low  stochastic=high
    "R21":                  5.0,  # Fourier shape fingerprint
    "phi21":                5.0,
    # ── STRONG DISCRIMINATORS ────────────────────────────────────────────────
    "R31":                  4.0,
    "phi31":                4.0,
    "period_fap_log10":     4.0,
    "period_power":         3.5,
    "log10_period":         3.0,
    "period_snr":           3.0,
    # ── COLOR EVOLUTION (key SNIa vs periodic) ───────────────────────────────
    "gr_peak_to_late_delta":4.0,  # SNIa: large positive, periodic: ~0
    "delta_color_15d":      4.0,  # SNIa signature
    "gr_trend_late":        3.5,  # SNIa: strongly positive
    "gr_at_peak":           3.0,
    "gr_late":              3.0,
    "gr_monotonicity":      3.0,
    # ── MORPHOLOGICAL ────────────────────────────────────────────────────────
    "rise_decline_asymmetry":3.5, # SNIa: asymmetric; periodic: ~0
    "rise_rate":            3.0,
    "decline_rate":         3.0,
    "fwhm_frac":            2.5,
    "peak_time_frac":       2.0,
    "composite_n_maxima":   4.0,
    # ── STOCHASTIC vs PERIODIC ───────────────────────────────────────────────
    "sf_slope":             3.0,  # AGN≈0.4, periodic≈0, SNIa: variable
    "sf_1d_norm":           2.0,
    "sf_10d_norm":          2.5,
    "color_osc_fap_log10":  3.0,
    "color_osc_power":      2.5,
    "color_period_ratio":   2.5,
    # ── AMPLITUDE / VARIABILITY ──────────────────────────────────────────────
    "chi2_flat":            2.5,
    "amplitude":            2.0,
    "amplitude_ratio_gr":   2.5,
    "phase_offset_gr":      3.0,
    "stetson_k":            1.5,
    "eta_e":                1.5,
    # ── BASIC STATS (lower weight — not very discriminating alone) ────────────
    "mean_mag":             0.3,
    "median_mag":           0.3,
    "std_mag":              0.8,
    "skew":                 0.8,
    "kurtosis":             0.8,
    "n_obs":                0.4,
    "timespan":             0.4,
    "linear_trend":         0.5,
    "max_slope":            0.6,
}


def get_feature_weights() -> np.ndarray:
    """
    Return weight vector aligned with FEATURE_NAMES.
    Applied in SimilarityIndex to scale features before distance computation.
    """
    w = np.ones(N_FEATURES, dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        base = name[2:] if name.startswith(("g_", "r_")) else name
        w[i] = _WEIGHTS.get(base, 1.0)
    return (w / w.mean()).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Main extractor class
# ══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Extract a 115-dim feature vector from a ZTF multi-band light curve.

    Parameters
    ----------
    bands            : ZTF filter IDs (1=g, 2=r, 3=i)
    min_obs          : minimum detections per band
    ls_min_period    : Lomb-Scargle minimum period (days)
    ls_max_period    : Lomb-Scargle maximum period (days)
    ls_samples_per_peak : LS oversampling factor
    smooth_window    : window for LC smoothing when counting undulations
                       (days; None = auto = timespan/20)
    """

    def __init__(
        self,
        bands: List[int] = (1, 2),
        min_obs: int = 5,
        ls_min_period: float = 0.1,
        ls_max_period: float = 500.0,
        ls_samples_per_peak: int = 5,
        smooth_window: Optional[float] = None,
    ):
        self.bands               = list(bands)
        self.min_obs             = min_obs
        self.ls_min_period       = ls_min_period
        self.ls_max_period       = ls_max_period
        self.ls_samples_per_peak = ls_samples_per_peak
        self.smooth_window       = smooth_window

    @property
    def feature_names(self) -> List[str]:
        return FEATURE_NAMES

    @property
    def n_features(self) -> int:
        return N_FEATURES

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self, lc: pd.DataFrame) -> np.ndarray:
        """
        Compute the 115-dim feature vector.
        NaN values are imputed at index-build time.
        """
        vec = np.full(N_FEATURES, np.nan, dtype=np.float32)
        if lc is None or lc.empty:
            return vec

        band_results: Dict[int, dict] = {}

        for i, fid in enumerate(self.bands[:2]):
            sub = lc[lc["fid"] == fid]
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
            band_results[fid] = {
                "feats": feats, "t": t, "m": m, "e": e,
            }
            offset = i * len(_BAND_FEATURES)
            for j, name in enumerate(_BAND_FEATURES):
                vec[offset + j] = np.float32(_safe(feats.get(name, np.nan)))

        # cross-band
        xb     = self._extract_crossband(lc, band_results)
        offset = 2 * len(_BAND_FEATURES)
        for j, name in enumerate(_CROSS_FEATURES):
            vec[offset + j] = np.float32(_safe(xb.get(name, np.nan)))

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
            try:
                rows.append(self.extract(lc))
                oids_out.append(oid)
            except Exception as exc:
                logger.warning(f"[{oid}] feature extraction failed: "
                               f"{type(exc).__name__}: {exc} — skipping")
        if not rows:
            return [], np.empty((0, N_FEATURES), dtype=np.float32)
        return oids_out, np.vstack(rows).astype(np.float32)

    # ── per-band extraction ───────────────────────────────────────────────────

    def _extract_band(
        self,
        t: np.ndarray,
        m: np.ndarray,
        e: np.ndarray,
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        n = len(m)

        # ── sampling ──────────────────────────────────────────────────────────
        feats["n_obs"]    = float(n)
        feats["timespan"] = float(np.ptp(t)) if n > 1 else 0.0

        # ── magnitude statistics ──────────────────────────────────────────────
        mean   = float(np.mean(m))
        median = float(np.median(m))
        std    = float(np.std(m, ddof=1)) if n > 1 else 0.0
        mad    = float(np.median(np.abs(m - median)))

        feats["mean_mag"]   = mean
        feats["median_mag"] = median
        feats["std_mag"]    = std
        feats["mad_mag"]    = mad
        feats["skew"]       = float(stats.skew(m))     if n >= 3 else 0.0
        feats["kurtosis"]   = float(stats.kurtosis(m)) if n >= 4 else 0.0
        feats["beyond1std"] = float(np.mean(np.abs(m - mean) > std))     if std > 0 else 0.0
        feats["beyond2std"] = float(np.mean(np.abs(m - mean) > 2 * std)) if std > 0 else 0.0

        # ── amplitude / spread ────────────────────────────────────────────────
        q25, q75 = np.percentile(m, [25, 75])
        feats["amplitude"]         = float(np.ptp(m) / 2.0)
        feats["iqr"]               = float(q75 - q25)
        feats["percent_amplitude"] = (
            float(np.max(np.abs(m - median)) / abs(median))
            if abs(median) > 0 else np.nan
        )
        feats["meanvariance"] = (
            float(std / abs(mean)) if abs(mean) > 0 else np.nan
        )
        feats.update(_flux_percentile_ratios(m))

        # ── variability indices ───────────────────────────────────────────────
        feats["stetson_k"] = _stetson_k(m, e)
        feats["eta_e"]     = _eta_e(m)
        feats["chi2_flat"] = _chi2_flat(m, e)
        feats["rcs"]       = _rcs(m)
        ptp = np.ptp(m)
        feats["mbrp"] = float(np.mean(np.abs(m - median) < 0.1 * ptp)) if ptp > 0 else 1.0

        # ── trends ────────────────────────────────────────────────────────────
        if n >= 3 and len(np.unique(t)) >= 2:
            slope, *_ = stats.linregress(t, m)
            feats["linear_trend"] = float(slope)
        dt = np.diff(t); dm = np.diff(m)
        valid = dt > 0
        if valid.sum() > 0:
            feats["max_slope"] = float(np.max(np.abs(dm[valid] / dt[valid])))

        # ── morphological shape ───────────────────────────────────────────────
        # brightness peak: minimum magnitude = maximum flux
        pk_idx = int(np.argmin(m))
        peak_mag = float(m[pk_idx])
        feats["peak_mag_norm"]  = float((peak_mag - mean) / std) if std > 0 else 0.0
        ts = feats["timespan"]
        feats["peak_time_frac"] = (
            float((t[pk_idx] - t.min()) / ts) if ts > 0 else 0.5
        )

        rise_r, decl_r, asym = _rise_decline(t, m, pk_idx)
        feats["rise_rate"]             = rise_r
        feats["decline_rate"]          = decl_r
        feats["rise_decline_asymmetry"]= asym

        # undulation count (KEY feature)
        sw = self.smooth_window or (ts / 20.0 if ts > 0 else None)
        feats["n_local_maxima"] = float(
            _count_local_maxima(t, m, smooth_days=sw)
        )

        # fwhm fraction
        feats["fwhm_frac"] = _fwhm_frac(m)

        # ── periodicity ───────────────────────────────────────────────────────
        if n >= 10:
            period, power, fap, snr = _lombscargle_full(
                t, m, e,
                self.ls_min_period, self.ls_max_period,
                self.ls_samples_per_peak,
            )
            feats["log10_period"]     = float(np.log10(period)) if period and period > 0 else np.nan
            feats["period_power"]     = float(power)  if power  is not None else np.nan
            feats["period_fap_log10"] = float(-np.log10(fap + 1e-300)) if fap is not None else np.nan
            feats["n_cycles"]         = (
                float(ts / period)
                if period and period > 0 and ts > 0 else np.nan
            )
            feats["period_snr"] = float(snr) if snr is not None else np.nan

            # Fourier decomposition — THE shape fingerprint
            if period and period > 0 and n >= 15:
                fc = _fourier_decomposition(t, m, e, period, n_harmonics=3)
                if fc is not None:
                    amps, phis = fc
                    feats["R21"]   = float(amps[1] / amps[0]) if amps[0] > 1e-10 else np.nan
                    feats["phi21"] = float(phis[1] - 2 * phis[0])
                    feats["R31"]   = float(amps[2] / amps[0]) if amps[0] > 1e-10 else np.nan
                    feats["phi31"] = float(phis[2] - 3 * phis[0])
                    # wrap to [-π, π]
                    for key in ("phi21", "phi31"):
                        v = feats.get(key, np.nan)
                        if np.isfinite(v):
                            feats[key] = float((v + np.pi) % (2 * np.pi) - np.pi)
        else:
            for k in ("log10_period","period_power","period_fap_log10",
                      "n_cycles","period_snr","R21","phi21","R31","phi31"):
                feats[k] = np.nan

        # ── structure function ────────────────────────────────────────────────
        sf_slope, sf_1d, sf_10d = _structure_function(t, m)
        var = float(np.var(m)) if n > 1 else 1.0
        feats["sf_slope"]   = sf_slope
        feats["sf_1d_norm"] = (sf_1d / var) if (sf_1d and var > 0) else np.nan
        feats["sf_10d_norm"]= (sf_10d / var) if (sf_10d and var > 0) else np.nan

        # ── information theory ────────────────────────────────────────────────
        feats["permutation_entropy"] = _permutation_entropy(m, order=3)
        feats["sample_entropy"]      = _sample_entropy(m, order=2, r=0.2 * std)

        return feats

    # ── cross-band extraction ─────────────────────────────────────────────────

    def _extract_crossband(
        self,
        lc: pd.DataFrame,
        band_results: Dict,
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {}

        g_sub = lc[lc["fid"] == 1]
        r_sub = lc[lc["fid"] == 2]
        if g_sub.empty or r_sub.empty:
            return feats

        g_t = g_sub["mjd"].values.astype(float)
        g_m = g_sub["magpsf"].values.astype(float)
        r_t = r_sub["mjd"].values.astype(float)
        r_m = r_sub["magpsf"].values.astype(float)

        # ── build g-r color time series (nearest obs within 1 day) ───────────
        gr_t, gr_c = _color_series(g_t, g_m, r_t, r_m, max_sep=1.0)

        if len(gr_c) >= 3:
            gr = np.array(gr_c)
            t_gr = np.array(gr_t)

            # ── color statistics ──────────────────────────────────────────────
            feats["gr_mean"]      = float(np.mean(gr))
            feats["gr_std"]       = float(np.std(gr, ddof=1))
            feats["gr_amplitude"] = float(np.ptp(gr) / 2.0)
            feats["gr_skew"]      = float(stats.skew(gr)) if len(gr) >= 3 else np.nan
            feats["gr_min"]       = float(np.min(gr))
            feats["gr_max"]       = float(np.max(gr))

            # ── color at specific epochs ──────────────────────────────────────
            # find peak brightness epoch across both bands
            all_t = np.concatenate([g_t, r_t])
            all_flux = np.concatenate([
                10 ** (-0.4 * g_m),
                10 ** (-0.4 * r_m),
            ])
            peak_t = float(all_t[np.argmax(all_flux)])

            # color at peak epoch (±2 days)
            near_peak = np.abs(t_gr - peak_t) <= 2.0
            if near_peak.sum() >= 1:
                feats["gr_at_peak"] = float(np.mean(gr[near_peak]))

            # time quintile colors
            ts_full = max(np.ptp(t_gr), 1.0)
            t0 = float(t_gr.min())
            for label, lo, hi in [
                ("gr_early", 0.0, 0.2),
                ("gr_mid",   0.4, 0.6),
                ("gr_late",  0.8, 1.0),
            ]:
                mask = (
                    ((t_gr - t0) / ts_full >= lo) &
                    ((t_gr - t0) / ts_full <= hi)
                )
                if mask.sum() >= 1:
                    feats[label] = float(np.mean(gr[mask]))

            # ── color evolution ───────────────────────────────────────────────
            if len(gr) >= 3 and len(np.unique(t_gr)) >= 2:
                slope, *_ = stats.linregress(t_gr, gr)
                feats["gr_trend"] = float(slope)

            # post-peak color slope (KEY for SNIa)
            post = t_gr > peak_t
            if post.sum() >= 3 and len(np.unique(t_gr[post])) >= 2:
                slope_late, *_ = stats.linregress(t_gr[post], gr[post])
                feats["gr_trend_late"] = float(slope_late)

            # fraction of consecutive pairs that are monotonically increasing
            feats["gr_monotonicity"] = _monotonicity(gr)

            # color change from peak to late phase
            gr_at_pk   = feats.get("gr_at_peak",  np.nan)
            gr_at_late = feats.get("gr_late",      np.nan)
            if np.isfinite(gr_at_pk) and np.isfinite(gr_at_late):
                feats["gr_peak_to_late_delta"] = float(gr_at_late - gr_at_pk)

            # Δ(g-r) in ~15 days after peak
            if np.isfinite(gr_at_pk):
                window_15 = (t_gr >= peak_t) & (t_gr <= peak_t + 15)
                if window_15.sum() >= 1:
                    feats["delta_color_15d"] = float(
                        np.mean(gr[window_15]) - gr_at_pk
                    )

            # ── color periodicity ─────────────────────────────────────────────
            if len(gr) >= 10:
                try:
                    gr_e = np.full(len(gr), max(np.std(gr) * 0.1, 0.01))
                    cp, cpow, cfap, _ = _lombscargle_full(
                        t_gr, gr, gr_e,
                        self.ls_min_period, self.ls_max_period,
                        self.ls_samples_per_peak,
                    )
                    feats["color_osc_power"]     = float(cpow) if cpow else np.nan
                    feats["color_osc_fap_log10"] = (
                        float(-np.log10(cfap + 1e-300)) if cfap else np.nan
                    )
                    # ratio of color period to g-band period
                    g_feats = band_results.get(1, {}).get("feats", {})
                    p_g = 10 ** g_feats["log10_period"] if (
                        "log10_period" in g_feats and
                        np.isfinite(g_feats.get("log10_period", np.nan))
                    ) else None
                    if cp and cp > 0 and p_g and p_g > 0:
                        feats["color_period_ratio"] = float(cp / p_g)
                except Exception:
                    pass

        # ── cross-band period & timing ────────────────────────────────────────
        g_feats = band_results.get(1, {}).get("feats", {})
        r_feats = band_results.get(2, {}).get("feats", {})

        p_g_log = g_feats.get("log10_period", np.nan)
        p_r_log = r_feats.get("log10_period", np.nan)

        if np.isfinite(p_g_log) and np.isfinite(p_r_log):
            feats["period_consistency"] = float(abs(p_g_log - p_r_log))
        else:
            feats["period_consistency"] = 2.0  # penalty for no period

        nc_g = g_feats.get("n_cycles", np.nan)
        nc_r = r_feats.get("n_cycles", np.nan)
        valid_nc = [x for x in [nc_g, nc_r] if np.isfinite(x)]
        feats["n_cycles_min"] = float(min(valid_nc)) if valid_nc else np.nan

        # amplitude ratio
        amp_g = g_feats.get("amplitude", np.nan)
        amp_r = r_feats.get("amplitude", np.nan)
        if np.isfinite(amp_g) and np.isfinite(amp_r) and amp_r > 0:
            feats["amplitude_ratio_gr"] = float(amp_g / amp_r)

        # peak time offset (days)
        ptf_g = g_feats.get("peak_time_frac", np.nan)
        ptf_r = r_feats.get("peak_time_frac", np.nan)
        ts_g  = g_feats.get("timespan", np.nan)
        ts_r  = r_feats.get("timespan", np.nan)
        if all(np.isfinite([ptf_g, ptf_r, ts_g, ts_r])):
            feats["peak_time_offset_gr"] = float(
                abs(ptf_g * ts_g - ptf_r * ts_r)
            )

        # phase offset between bands using best period
        if np.isfinite(p_g_log):
            period_use = 10 ** p_g_log
            gt = band_results.get(1, {}).get("t")
            gm = band_results.get(1, {}).get("m")
            rt = band_results.get(2, {}).get("t")
            rm = band_results.get(2, {}).get("m")
            if gt is not None and rt is not None and period_use > 0:
                feats["phase_offset_gr"] = _phase_offset(
                    gt, gm, rt, rm, period_use
                )

        # ── composite LC features (band-agnostic) ─────────────────────────────
        comp = _composite_lc(lc)
        if comp is not None:
            ct, cf = comp
            feats["composite_n_maxima"] = float(
                _count_local_maxima(ct, -cf)    # -flux so minima = peaks
            )
            if len(cf) >= 3:
                cf_std = float(np.std(cf, ddof=1))
                cf_e   = np.full(len(cf), max(cf_std * 0.05, 1e-6))
                feats["composite_chi2"]      = _chi2_flat(-cf, cf_e)
                feats["composite_amplitude"] = float(np.ptp(cf) / 2.0)

        return feats


# ══════════════════════════════════════════════════════════════════════════════
# Feature computation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _count_local_maxima(
    t: np.ndarray,
    m: np.ndarray,
    smooth_days: Optional[float] = None,
    min_prominence_frac: float = 0.1,
) -> int:
    """
    Count brightness peaks (local minima in magnitude) in a smoothed LC.

    Strategy
    --------
    1. Resample onto a uniform grid by linear interpolation
    2. Smooth with a Gaussian kernel to suppress noise
    3. Use scipy.signal.find_peaks on the flux (inverted magnitude)

    Parameters
    ----------
    min_prominence_frac : float
        A peak must be at least this fraction of the full amplitude to count.
    """
    n = len(t)
    if n < 6:
        return 0

    ts = float(np.ptp(t))
    if ts <= 0:
        return 0

    # ── uniform grid ─────────────────────────────────────────────────────────
    n_grid = min(max(n * 3, 200), 2000)
    t_grid = np.linspace(t.min(), t.max(), n_grid)
    m_grid = np.interp(t_grid, t, m)

    # ── Gaussian smoothing ────────────────────────────────────────────────────
    sigma_days  = smooth_days if smooth_days else max(ts / 25.0, 1.0)
    sigma_pts   = sigma_days / (ts / n_grid)
    sigma_pts   = max(int(sigma_pts), 2)

    from scipy.ndimage import gaussian_filter1d
    m_smooth = gaussian_filter1d(m_grid, sigma=sigma_pts)

    # work in flux space (invert magnitude) so peaks = local maxima
    flux_smooth = 10.0 ** (-0.4 * (m_smooth - np.median(m_smooth)))
    flux_amp    = float(np.ptp(flux_smooth))
    if flux_amp <= 0:
        return 0

    prominence_thresh = min_prominence_frac * flux_amp
    peaks, _ = find_peaks(
        flux_smooth,
        prominence=prominence_thresh,
        distance=max(int(sigma_pts * 1.5), 3),
    )
    return len(peaks)


def _fourier_decomposition(
    t: np.ndarray,
    m: np.ndarray,
    e: np.ndarray,
    period: float,
    n_harmonics: int = 3,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Fit a truncated Fourier series to the phase-folded light curve.

    Model: m(φ) = A₀ + Σₖ [aₖ cos(2πkφ) + bₖ sin(2πkφ)]

    Returns
    -------
    (amplitudes, phases) each of length n_harmonics, or None on failure.
    Rₖ₁ = amplitudes[k-1] / amplitudes[0]
    φₖ₁ = phases[k-1] − k * phases[0]
    """
    if len(t) < 2 * n_harmonics + 3:
        return None
    try:
        phase = (t % period) / period

        # build design matrix
        cols = [np.ones(len(phase))]
        for k in range(1, n_harmonics + 1):
            cols.append(np.cos(2 * np.pi * k * phase))
            cols.append(np.sin(2 * np.pi * k * phase))
        A = np.column_stack(cols)

        # weighted least squares
        W  = 1.0 / (e ** 2 + 1e-12)
        Aw = A * np.sqrt(W[:, None])
        mw = m * np.sqrt(W)
        coeffs, _, _, _ = np.linalg.lstsq(Aw, mw, rcond=None)

        # extract amplitudes and phases
        amps, phis = [], []
        for k in range(n_harmonics):
            a_k = coeffs[1 + 2 * k]
            b_k = coeffs[2 + 2 * k]
            A_k = float(np.sqrt(a_k ** 2 + b_k ** 2))
            # phase convention: A_k * cos(2πkφ + φ_k)
            # → a_k = A_k cos(φ_k), b_k = -A_k sin(φ_k)
            phi_k = float(np.arctan2(-b_k, a_k))
            amps.append(A_k)
            phis.append(phi_k)

        return np.array(amps), np.array(phis)

    except Exception as exc:
        logger.debug(f"Fourier fit failed: {exc}")
        return None


def _lombscargle_full(
    t: np.ndarray,
    m: np.ndarray,
    e: np.ndarray,
    min_period: float,
    max_period: float,
    samples_per_peak: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Lomb-Scargle period search.
    Returns (period, power, fap, snr).
    SNR = peak_power / local_noise (estimated from 5th percentile of power).
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
        if len(power) == 0:
            return None, None, 1.0, None

        best_idx  = int(np.argmax(power))
        best_freq = float(freq[best_idx])
        best_pow  = float(power[best_idx])
        period    = 1.0 / best_freq if best_freq > 0 else None
        fap       = float(ls.false_alarm_probability(best_pow, method="baluev"))

        # SNR vs local noise floor
        noise = float(np.percentile(power, 5))
        snr   = float(best_pow / noise) if noise > 0 else np.nan

        return period, best_pow, fap, snr

    except Exception as exc:
        logger.debug(f"LS failed: {exc}")
        return None, None, 1.0, None


def _structure_function(
    t: np.ndarray,
    m: np.ndarray,
    max_pairs: int = 50000,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Compute the discrete structure function SF(Δt) = ⟨(m(t) − m(t+Δt))²⟩.

    Returns (slope, sf_at_1day, sf_at_10day).
    slope is estimated from the log-log SF in the range [2, 80] days.
    """
    n = len(t)
    if n < 8:
        return np.nan, None, None

    # subsample for speed
    if n * (n - 1) // 2 > max_pairs:
        idx = np.random.choice(n, size=min(n, 320), replace=False)
        t, m = t[idx], m[idx]
        n = len(t)

    # all pairwise
    dt_all, dm2_all = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dt_all.append(abs(t[j] - t[i]))
            dm2_all.append((m[j] - m[i]) ** 2)

    dt_arr  = np.array(dt_all)
    dm2_arr = np.array(dm2_all)

    # log-log slope (use middle 25-75% of lag range)
    valid = dt_arr > 0
    if valid.sum() < 6:
        return np.nan, None, None

    log_dt  = np.log10(dt_arr[valid])
    log_dm2 = np.log10(dm2_arr[valid] + 1e-12)

    lo, hi = np.percentile(log_dt, [25, 75])
    mask   = (log_dt >= lo) & (log_dt <= hi)
    if mask.sum() < 5:
        sf_slope = np.nan
    else:
        sf_slope, *_ = stats.linregress(log_dt[mask], log_dm2[mask])
        sf_slope = float(sf_slope)

    # SF at specific lags (bin ± half-decade)
    def sf_at_lag(target_days: float) -> Optional[float]:
        lo_l = target_days / np.sqrt(10)
        hi_l = target_days * np.sqrt(10)
        sel  = (dt_arr >= lo_l) & (dt_arr <= hi_l)
        return float(np.mean(dm2_arr[sel])) if sel.sum() >= 3 else None

    return sf_slope, sf_at_lag(1.0), sf_at_lag(10.0)


def _permutation_entropy(m: np.ndarray, order: int = 3) -> float:
    """
    Bandt & Pompe (2002) permutation entropy, normalised to [0, 1].
    Low  → regular / periodic (predictable ordering pattern)
    High → stochastic / complex
    """
    import math
    n = len(m)
    if n < order + 2:
        return np.nan

    all_pats = list(_perms(range(order)))
    counts   = {p: 0 for p in all_pats}
    total    = 0

    for i in range(n - order + 1):
        window  = m[i:i + order]
        pattern = tuple(np.argsort(window, kind="stable"))
        if pattern in counts:
            counts[pattern] += 1
            total += 1

    if total == 0:
        return np.nan

    probs = np.array([c / total for c in counts.values()])
    probs = probs[probs > 0]
    H     = float(-np.sum(probs * np.log2(probs)))
    H_max = float(np.log2(math.factorial(order)))
    return float(H / H_max) if H_max > 0 else np.nan




def _sample_entropy(seq: np.ndarray, order: int = 2, r: float = 0.2) -> float:
    """
    Approximate sample entropy (simplified, fast version).
    """
    n = len(seq)
    if n < order + 10:
        return np.nan
    try:
        threshold = r * float(np.std(seq, ddof=1)) if r < 1.0 else r
        if threshold <= 0:
            return np.nan

        def count_matches(m_ord):
            count = 0
            for i in range(n - m_ord):
                for j in range(i + 1, n - m_ord):
                    if np.max(np.abs(seq[i:i+m_ord] - seq[j:j+m_ord])) < threshold:
                        count += 1
            return count

        # Use small subsample for speed
        sub = seq[:min(n, 60)]
        n2  = len(sub)
        A   = count_matches(order + 1) if n2 >= order + 2 else 0
        B   = count_matches(order)     if n2 >= order + 1 else 1

        if B == 0:
            return np.nan
        se = float(-np.log(max(A, 1) / B))
        return se if np.isfinite(se) else np.nan
    except Exception:
        return np.nan


def _rise_decline(
    t: np.ndarray,
    m: np.ndarray,
    pk_idx: int,
) -> Tuple[float, float, float]:
    """
    Compute rise and decline rates around the brightness peak.
    Returns (rise_rate, decline_rate, asymmetry).
    All in mag/day (positive = brightening or fading).
    """
    if len(t) < 4:
        return np.nan, np.nan, np.nan

    pre  = slice(None, pk_idx + 1)
    post = slice(pk_idx, None)

    def rate(t_seg, m_seg):
        if len(t_seg) < 2:
            return np.nan
        dt = float(np.ptp(t_seg))
        dm = float(m_seg[-1] - m_seg[0])
        return abs(dm / dt) if dt > 0 else np.nan

    rise_r = rate(t[pre], m[pre])
    decl_r = rate(t[post], m[post])

    denom = (rise_r + decl_r) if (
        rise_r is not None and decl_r is not None
        and np.isfinite(rise_r) and np.isfinite(decl_r)
        and (rise_r + decl_r) > 0
    ) else None
    asym = float((rise_r - decl_r) / denom) if denom else np.nan

    return (
        float(rise_r) if rise_r and np.isfinite(rise_r) else np.nan,
        float(decl_r) if decl_r and np.isfinite(decl_r) else np.nan,
        asym,
    )


def _fwhm_frac(m: np.ndarray) -> float:
    """
    Fraction of observations above half-maximum brightness (flux space).
    """
    if len(m) < 4:
        return np.nan
    flux  = 10.0 ** (-0.4 * (m - np.median(m)))
    f_max = float(np.max(flux))
    f_min = float(np.min(flux))
    half  = f_min + 0.5 * (f_max - f_min)
    return float(np.mean(flux >= half))


def _color_series(
    g_t: np.ndarray, g_m: np.ndarray,
    r_t: np.ndarray, r_m: np.ndarray,
    max_sep: float = 1.0,
) -> Tuple[List[float], List[float]]:
    """
    Build a g−r color time series by matching nearest observations.
    Only pairs with |Δt| ≤ max_sep days are used.
    """
    times, colors = [], []
    for i, (gt, gm) in enumerate(zip(g_t, g_m)):
        j   = int(np.argmin(np.abs(r_t - gt)))
        sep = abs(r_t[j] - gt)
        if sep <= max_sep:
            times.append(float(gt))
            colors.append(float(gm - r_m[j]))
    return times, colors


def _phase_offset(
    t_g: np.ndarray, m_g: np.ndarray,
    t_r: np.ndarray, m_r: np.ndarray,
    period: float,
) -> float:
    """
    Estimate the phase lag between g and r bands at the given period.
    Returns phase offset in units of period (0 = in phase, 0.5 = anti-phase).
    """
    if period <= 0 or len(t_g) < 5 or len(t_r) < 5:
        return np.nan
    try:
        phi_g = (t_g % period) / period
        phi_r = (t_r % period) / period

        # fit simple sinusoid to each band, extract phase
        def fit_phase(phi, m):
            A = np.column_stack([
                np.cos(2 * np.pi * phi),
                np.sin(2 * np.pi * phi),
                np.ones(len(phi)),
            ])
            c, *_ = np.linalg.lstsq(A, m, rcond=None)
            return float(np.arctan2(c[1], c[0]))

        p_g = fit_phase(phi_g, m_g)
        p_r = fit_phase(phi_r, m_r)
        offset = (p_g - p_r) / (2 * np.pi)
        # wrap to [0, 0.5]
        offset = offset % 1.0
        if offset > 0.5:
            offset = 1.0 - offset
        return float(offset)
    except Exception:
        return np.nan


def _composite_lc(
    lc: pd.DataFrame,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Build a composite flux LC by combining all available bands.
    Each band is normalised to zero median flux before combining.
    Returns (time, normalised_flux) sorted by time, or None.
    """
    parts = []
    for fid, grp in lc.groupby("fid"):
        t = grp["mjd"].values.astype(float)
        m = grp["magpsf"].values.astype(float)
        flux = 10.0 ** (-0.4 * m)
        flux_norm = flux - float(np.median(flux))
        parts.append(np.column_stack([t, flux_norm]))

    if not parts:
        return None

    combined = np.vstack(parts)
    order    = np.argsort(combined[:, 0])
    combined = combined[order]
    return combined[:, 0], combined[:, 1]


def _monotonicity(x: np.ndarray) -> float:
    """Fraction of consecutive steps that increase."""
    if len(x) < 3:
        return np.nan
    return float(np.mean(np.diff(x) > 0))


# ══════════════════════════════════════════════════════════════════════════════
# Standard variability statistics (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _stetson_k(m, e) -> float:
    n = len(m)
    if n < 4: return np.nan
    w    = 1.0 / (e ** 2 + 1e-12)
    m_w  = np.sum(w * m) / np.sum(w)
    delta = np.sqrt(float(n) / max(n - 1, 1)) * (m - m_w) / (e + 1e-12)
    num   = np.sum(np.abs(delta)) / n
    den   = np.sqrt(np.sum(delta ** 2) / n)
    return float(num / den) if den > 0 else np.nan

def _eta_e(m) -> float:
    n = len(m)
    if n < 4: return np.nan
    var = np.var(m, ddof=1)
    if var == 0: return np.nan
    return float(np.sum(np.diff(m) ** 2) / ((n - 1) * var))

def _chi2_flat(m, e) -> float:
    n = len(m)
    if n < 3: return np.nan
    w   = 1.0 / (e ** 2 + 1e-12)
    m_w = np.sum(w * m) / np.sum(w)
    return float(np.sum(w * (m - m_w) ** 2) / max(n - 1, 1))

def _rcs(m) -> float:
    n = len(m)
    if n < 3: return np.nan
    std = np.std(m, ddof=1)
    if std == 0: return np.nan
    s = np.cumsum(m - np.mean(m))
    return float((np.max(s) - np.min(s)) / (n * std))

def _flux_percentile_ratios(m) -> Dict[str, float]:
    if len(m) < 10:
        return {"fpr_mid20": np.nan, "fpr_mid50": np.nan, "fpr_mid80": np.nan}
    flux  = 10.0 ** (-0.4 * (m - np.median(m)))
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

def _safe(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan

# ══════════════════════════════════════════════════════════════════════════════
# NEW HELPERS — append at bottom of features.py
# ══════════════════════════════════════════════════════════════════════════════

def _peak_normalized_shape(
    t: np.ndarray,
    m: np.ndarray,
    offsets: Tuple[float, ...] = (-30, -20, -10, +5, +10, +15, +20, +30, +50),
    window: float = 7.0,
) -> Dict[str, float]:
    """
    Brightness at fixed time offsets from peak, normalized to peak magnitude.

    dm(offset) = m_interpolated(t_peak + offset) - m_peak

    All values are POSITIVE (fainter than peak) for transients post-peak.
    Periodic objects have small, oscillating values.
    Scale-invariant: doesn't depend on absolute brightness.

    Parameters
    ----------
    window : float
        Gaussian averaging window width in days.
        Use a larger value for sparse LCs (e.g. 10 days).
    """
    feats: Dict[str, float] = {}
    if len(t) < 5:
        return feats

    # peak = minimum magnitude (maximum flux)
    pk_idx  = int(np.argmin(m))
    t_peak  = float(t[pk_idx])
    m_peak  = float(m[pk_idx])
    sigma   = window / 2.5  # Gaussian sigma

    for offset in offsets:
        label = "dm_m" + str(abs(int(offset))) if offset < 0 \
                else "dm_p" + str(int(offset))
        t_target = t_peak + offset

        dist     = np.abs(t - t_target)
        in_win   = dist <= window
        if in_win.sum() == 0:
            feats[label] = np.nan
            continue

        # Gaussian-weighted interpolation
        w         = np.exp(-0.5 * (dist[in_win] / sigma) ** 2)
        w_sum     = w.sum()
        if w_sum <= 0:
            feats[label] = np.nan
            continue
        m_interp  = float(np.dot(w, m[in_win]) / w_sum)
        feats[label] = m_interp - m_peak   # positive = fainter than peak

    return feats


def _temporal_variability_evolution(
    t: np.ndarray,
    m: np.ndarray,
    n_thirds: int = 3,
) -> Dict[str, float]:
    """
    Split the LC into thirds and measure variability in each.

    Rationale:
    - Transients (SNIa): high var_early (pre-peak), decreasing → var_trend < 0
    - Periodic objects:  consistent variability throughout → var_trend ≈ 0
    - AGN:               slowly increasing → var_trend slightly positive
    - Fading transients: var_early large, var_late small → var_trend very negative
    """
    feats: Dict[str, float] = {}
    n = len(t)
    if n < n_thirds * 3:
        return feats

    ts = float(np.ptp(t))
    if ts <= 0:
        return feats

    t0     = float(t.min())
    thirds = []
    labels = ["var_early", "var_mid", "var_late"]

    for i in range(n_thirds):
        lo   = t0 + i * ts / n_thirds
        hi   = t0 + (i + 1) * ts / n_thirds
        mask = (t >= lo) & (t <= hi)
        if mask.sum() >= 3:
            thirds.append(float(np.std(m[mask], ddof=1)))
        else:
            thirds.append(np.nan)

    for label, val in zip(labels, thirds):
        feats[label] = val

    # variability trend
    if np.isfinite(thirds[0]) and np.isfinite(thirds[-1]):
        denom = thirds[0] + thirds[-1]
        feats["var_trend"] = (
            float((thirds[-1] - thirds[0]) / denom)
            if denom > 0 else 0.0
        )

    # number of thirds with "significant" variability
    valid  = [s for s in thirds if np.isfinite(s)]
    if valid:
        threshold = max(valid) * 0.2
        feats["var_n_active"] = float(
            sum(s >= threshold for s in valid if np.isfinite(s))
        )

    return feats


def _pre_post_peak_undulations(
    t: np.ndarray,
    m: np.ndarray,
    smooth_days: Optional[float] = None,
) -> Dict[str, float]:
    """
    Count undulations before and after the brightness peak separately.

    This is the key discriminator:
    - Periodic variables (RRL, LPV):  n_pre ≈ n_post ≈ many → maxima_ratio ≈ 0.5
    - SNIa / transients:              n_pre ≈ 0, n_post ≈ 0–1 → maxima_ratio > 0.8
    - Post-peak AGN flare:            n_pre ≈ 0, n_post > 0   → maxima_ratio ≈ 1.0
    """
    feats: Dict[str, float] = {}
    if len(t) < 10:
        feats["n_maxima_pre"]  = 0.0
        feats["n_maxima_post"] = 0.0
        feats["maxima_ratio"]  = 0.5
        return feats

    pk_idx = int(np.argmin(m))
    t_peak = float(t[pk_idx])
    sw     = smooth_days or max(float(np.ptp(t)) / 20.0, 1.0)

    pre_mask  = t <= t_peak
    post_mask = t >= t_peak

    n_pre  = 0
    n_post = 0

    if pre_mask.sum() >= 5:
        n_pre = int(_count_local_maxima(
            t[pre_mask], m[pre_mask], smooth_days=sw
        ))
    if post_mask.sum() >= 5:
        n_post = int(_count_local_maxima(
            t[post_mask], m[post_mask], smooth_days=sw
        ))

    feats["n_maxima_pre"]  = float(n_pre)
    feats["n_maxima_post"] = float(n_post)
    total = n_pre + n_post
    feats["maxima_ratio"]  = float(n_post / total) if total > 0 else 0.5

    return feats


def _color_at_epochs(
    gr_t: np.ndarray,
    gr_c: np.ndarray,
    t_peak: float,
    offsets: Tuple[float, ...] = (-30, -10, +5, +15, +30),
    window: float = 8.0,
) -> Dict[str, float]:
    """
    g−r color at fixed time offsets from the brightness peak.

    Key physical signatures:
    - SNIa at +15d: gr_p15 − gr_at_peak ≈ +0.3 to +1.0 (strongly redder)
    - RRL at +15d:  oscillates, no systematic reddening
    - AGN:          slow monotonic color change or none
    """
    feats: Dict[str, float] = {}
    sigma = window / 2.5

    for offset in offsets:
        label    = "gr_m" + str(abs(int(offset))) if offset < 0 \
                   else "gr_p" + str(int(offset))
        t_target = t_peak + offset
        dist     = np.abs(gr_t - t_target)
        in_win   = dist <= window

        if in_win.sum() == 0:
            feats[label] = np.nan
            continue

        w         = np.exp(-0.5 * (dist[in_win] / sigma) ** 2)
        w_sum     = w.sum()
        feats[label] = (
            float(np.dot(w, gr_c[in_win]) / w_sum) if w_sum > 0 else np.nan
        )

    return feats


def _band_shape_similarity(
    g_t: np.ndarray,
    g_m: np.ndarray,
    r_t: np.ndarray,
    r_m: np.ndarray,
) -> Dict[str, float]:
    """
    Measure morphological similarity between g and r band light curves.

    Both bands are normalized to zero-mean unit-variance before comparison.
    - Periodic: band_xcorr ≈ +1 (same shape, just different amplitude)
    - SNIa:     band_xcorr ≈ +0.8 (similar but slightly different timing)
    - AGN:      band_xcorr ≈ +0.5 to +0.9 (correlated stochastic)
    """
    feats: Dict[str, float] = {}
    if len(g_t) < 5 or len(r_t) < 5:
        return feats

    def normalize(arr: np.ndarray) -> np.ndarray:
        s = float(np.std(arr, ddof=1))
        return (arr - np.mean(arr)) / s if s > 0 else arr - np.mean(arr)

    g_norm = normalize(g_m)

    # interpolate r onto g epochs (only within observed range)
    r_lo, r_hi = float(r_t.min()), float(r_t.max())
    in_range   = (g_t >= r_lo) & (g_t <= r_hi)
    if in_range.sum() < 5:
        return feats

    r_interp = np.interp(g_t[in_range], r_t, normalize(r_m))
    g_sub    = g_norm[in_range]

    if len(g_sub) < 5 or not np.isfinite(r_interp).all():
        return feats

    try:
        corr = float(np.corrcoef(g_sub, r_interp)[0, 1])
        if np.isfinite(corr):
            feats["band_xcorr"] = corr
    except Exception:
        pass

    rms = float(np.sqrt(np.mean((g_sub - r_interp) ** 2)))
    if np.isfinite(rms):
        feats["band_shape_rms"] = rms

    # peak flux ratio (color in linear flux space at peak)
    pf_g = float(10 ** (-0.4 * np.min(g_m)))
    pf_r = float(10 ** (-0.4 * np.min(r_m)))
    if pf_r > 0:
        feats["peak_flux_ratio"] = float(pf_g / pf_r)

    return feats
