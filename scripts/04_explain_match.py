#!/usr/bin/env python
"""
Explain why two objects were matched.
Shows which features are similar and which are different.

Usage:
    python scripts/04_explain_match.py --query ZTF25acemaph --match ZTF18abcdefg
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ztf_lcsim.config import Config
from ztf_lcsim.downloader import AlerceDownloader
from ztf_lcsim.features import FeatureExtractor, FEATURE_NAMES, get_feature_weights
from ztf_lcsim.index import SimilarityIndex


@click.command()
@click.option("--query",  required=True, help="Query OID")
@click.option("--match",  default=None,  help="Specific matched OID to explain")
@click.option("--topk",   default=10,    help="Show top-k matches")
@click.option("--config", default="config.yaml")
@click.option("--save",   default=None,  help="Save explanation plot to file")
def cli(query, match, topk, config, save):
    """Explain why objects were matched by showing feature comparisons."""

    cfg = Config(config)
    dl  = AlerceDownloader(cache_dir=str(cfg.cache_dir))
    fe  = FeatureExtractor(
        bands=cfg.features.bands,
        min_obs=cfg.features.min_obs_per_band,
    )

    # ── get query features ────────────────────────────────────────────────────
    print(f"\nFetching query: {query}")
    lc_q   = dl.get_lightcurve(query)
    vec_q  = fe.extract(lc_q)

    # ── search ────────────────────────────────────────────────────────────────
    idx = SimilarityIndex().load(cfg.index_path)
    results = idx.search(vec_q, k=topk)
    print(f"\nTop-{topk} matches:\n{results.to_string(index=False)}\n")

    # ── pick match to explain ─────────────────────────────────────────────────
    if match is None:
        match = results["oid"].iloc[0]
    print(f"Explaining match: {match}")

    lc_m  = dl.get_lightcurve(match)
    vec_m = fe.extract(lc_m)

    # ── feature comparison table ──────────────────────────────────────────────
    weights = get_feature_weights()
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        vq = float(vec_q[i]) if np.isfinite(vec_q[i]) else None
        vm = float(vec_m[i]) if np.isfinite(vec_m[i]) else None
        if vq is not None and vm is not None:
            diff = abs(vq - vm)
        else:
            diff = np.nan
        rows.append({
            "feature": name,
            "query":   vq,
            "match":   vm,
            "abs_diff": diff,
            "weight":  float(weights[i]),
            "weighted_diff": diff * float(weights[i]) if np.isfinite(diff) else np.nan,
        })

    df_feat = pd.DataFrame(rows).sort_values("weighted_diff", ascending=False)

    print("\n── Most DIFFERENT features (why this is a BAD match) ──")
    print(df_feat.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n── Most SIMILAR features (why it was matched) ──")
    print(df_feat.tail(10).iloc[::-1].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── highlight the key discriminators ─────────────────────────────────────
    KEY = ["g_period_fap_log10", "r_period_fap_log10",
           "g_n_cycles",         "r_n_cycles",
           "period_consistency",  "n_cycles_min",
           "color_osc_fap_log10", "gr_trend",
           "g_chi2_flat",         "r_chi2_flat",
           "g_rise_decline_ratio","r_rise_decline_ratio"]

    print("\n── KEY DISCRIMINATING FEATURES ──────────────────────────")
    print(f"  {'Feature':<30} {'Query':>10} {'Match':>10} {'|Diff|':>10}")
    print("  " + "-"*60)
    for k in KEY:
        if k in FEATURE_NAMES:
            i  = FEATURE_NAMES.index(k)
            vq = f"{vec_q[i]:.3f}" if np.isfinite(vec_q[i]) else "NaN"
            vm = f"{vec_m[i]:.3f}" if np.isfinite(vec_m[i]) else "NaN"
            d  = f"{abs(vec_q[i]-vec_m[i]):.3f}" if (np.isfinite(vec_q[i]) and np.isfinite(vec_m[i])) else "—"
            print(f"  {k:<30} {vq:>10} {vm:>10} {d:>10}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (lc, oid) in zip(axes, [(lc_q, query), (lc_m, match)]):
        for fid, color, label in [(1,"#2ecc71","g"), (2,"#e74c3c","r")]:
            sub = lc[lc["fid"] == fid]
            if not sub.empty:
                ax.errorbar(sub["mjd"], sub["magpsf"], yerr=sub["sigmapsf"],
                            fmt="o", color=color, label=label,
                            ms=3, elinewidth=0.7, alpha=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("MJD")
        ax.set_ylabel("mag")
        ax.legend(title="band")
        ax.grid(alpha=0.3)
        ax.set_title(oid, fontsize=10)

    fig.suptitle(
        f"Query: {query}  |  Match: {match}\n"
        f"period_consistency={float(vec_q[FEATURE_NAMES.index('period_consistency')]):.3f} (query) "
        f"vs {float(vec_m[FEATURE_NAMES.index('period_consistency')]):.3f} (match)\n"
        f"n_cycles_min={float(vec_q[FEATURE_NAMES.index('n_cycles_min')]):.1f} (query) "
        f"vs {float(vec_m[FEATURE_NAMES.index('n_cycles_min')]):.1f} (match)",
        fontsize=9,
    )
    fig.tight_layout()

    out = save or f"explain_{query}_vs_{match}.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    cli()
