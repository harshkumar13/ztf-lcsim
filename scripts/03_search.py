#!/usr/bin/env python
"""
Step 3 – Search for objects similar to a given ZTF OID.

Usage
-----
python scripts/03_search.py --oid ZTF25acemaph --topk 20 --plot
python scripts/03_search.py --oid ZTF25acemaph --topk 20 --plot --save-plot results/out.pdf
python scripts/03_search.py --oid ZTF25acemaph --topk 20 --save-csv results/out.csv
python scripts/03_search.py --oid ZTF25acemaph --embed   # show 2-D embedding
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import click
import pandas as pd

from ztf_lcsim.config     import Config
from ztf_lcsim.downloader import AlerceDownloader
from ztf_lcsim.features   import FeatureExtractor, FEATURE_NAMES
from ztf_lcsim.database   import FeatureDatabase
from ztf_lcsim.index      import SimilarityIndex
from ztf_lcsim.plots      import (
    plot_lightcurve,
    plot_results,
    plot_feature_space,
    plot_object_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("search")


@click.command()
@click.option("--config",           default="config.yaml", show_default=True)
@click.option("--oid",              required=True,
              help="Query object ID (e.g. ZTF25acemaph).")
@click.option("--topk",             default=20, show_default=True,
              help="Number of similar objects to return.")
@click.option("--plot/--no-plot",   default=True,  show_default=True,
              help="Show interactive result-grid plot.")
@click.option("--summary/--no-summary", default=True, show_default=True,
              help="Show 4-panel object-summary plot.")
@click.option("--save-plot",        default=None,
              help="Save result-grid plot to file (PDF/PNG/SVG).")
@click.option("--save-summary",     default=None,
              help="Save summary plot to file.")
@click.option("--save-csv",         default=None,
              help="Save result table as CSV.")
@click.option("--embed/--no-embed", default=False,
              help="Show 2-D feature-space embedding (slow for large DBs).")
@click.option("--embed-method",     default="umap",
              type=click.Choice(["umap", "pca", "tsne"]),
              help="Dimensionality-reduction method for embedding.")
def cli(
    config, oid, topk,
    plot, summary, save_plot, save_summary,
    save_csv, embed, embed_method,
):
    """Search for ZTF objects with light curves similar to a given OID."""

    cfg = Config(config)

    # ── load index ────────────────────────────────────────────────────────────
    meta_file = Path(str(cfg.index_path) + ".meta")
    if not meta_file.exists():
        logger.error(
            f"Index not found at {cfg.index_path}.\n"
            "Run:  python scripts/02_build_index.py  first."
        )
        sys.exit(1)

    logger.info(f"Loading index from {cfg.index_path} …")
    idx = SimilarityIndex(
        index_type=cfg.index.type,
        metric=cfg.index.metric,
        ivf_nprobe=cfg.index.ivf_nprobe,
    ).load(cfg.index_path)
    logger.info(f"Index ready: {idx.n_objects:,} objects")

    # ── load database ─────────────────────────────────────────────────────────
    db = FeatureDatabase(cfg.features_path, cfg.metadata_path)

    # ── downloader ────────────────────────────────────────────────────────────
    downloader = AlerceDownloader(
        cache_dir=str(cfg.cache_dir),
        timeout=cfg.alerce.timeout,
        max_retries=cfg.alerce.max_retries,
        request_delay=cfg.alerce.request_delay,
    )

    # ── fetch query light curve ───────────────────────────────────────────────
    logger.info(f"Fetching light curve for query: {oid}")
    query_lc = downloader.get_lightcurve(oid, use_cache=True)
    if query_lc is None or query_lc.empty:
        logger.error(f"Could not fetch light curve for {oid}.")
        sys.exit(1)

    logger.info(
        f"  Detections: {len(query_lc)}  "
        f"(g={int((query_lc.fid == 1).sum())}, "
        f"r={int((query_lc.fid == 2).sum())})"
    )

    # ── extract features ──────────────────────────────────────────────────────
    fe = FeatureExtractor(
        bands=cfg.features.bands,
        min_obs=cfg.features.min_obs_per_band,
        ls_min_period=cfg.features.ls_min_period,
        ls_max_period=cfg.features.ls_max_period,
        ls_samples_per_peak=cfg.features.ls_samples_per_peak,
    )
    query_vec = fe.extract(query_lc)

    n_nan     = int((~np.isfinite(query_vec)).sum())
    n_total   = len(query_vec)
    logger.info(
        f"  Feature vector: {n_total} dims, "
        f"{n_nan} NaN ({n_nan / n_total:.0%})"
    )

    # ── similarity search ─────────────────────────────────────────────────────
    logger.info(f"Searching top-{topk} …")
    results = idx.search(query_vec, k=topk, exclude_self=True)

    print("\n" + "=" * 65)
    print(f"  Top-{topk} similar objects for  {oid}")
    print("=" * 65)
    print(results.to_string(index=False))
    print("=" * 65 + "\n")

    # ── CSV export ────────────────────────────────────────────────────────────
    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(save_csv, index=False)
        logger.info(f"Results saved to {save_csv}")

    # ── metadata enrichment ───────────────────────────────────────────────────
    result_oids = results["oid"].tolist()
    meta        = db.load_metadata(result_oids)

    if not meta.empty:
        results_enriched = results.merge(meta, on="oid", how="left")
        show_cols = [c for c in
                     ["rank", "oid", "similarity", "cls", "probability"]
                     if c in results_enriched.columns]
        print("Enriched results:")
        print(results_enriched[show_cols].to_string(index=False))
        print()

    # ── ML class probabilities for query ─────────────────────────────────────
    period_val  = None
    ml_proba    = None
    ml_classes  = None

    aug_path = cfg.db_dir / "ml_augmenter.pkl"
    if aug_path.exists():
        try:
            from ztf_lcsim.ml_features import MLFeatureAugmenter
            aug        = MLFeatureAugmenter.load(aug_path)
            ml_proba   = aug.predict_proba(query_vec.reshape(1, -1))[0]
            ml_classes = aug.classes_

            print("── ML class probabilities for query ─────────────────────")
            for cls_name, prob in sorted(
                zip(ml_classes, ml_proba), key=lambda x: -x[1]
            ):
                bar = "█" * int(prob * 30)
                print(f"  {cls_name:<20} {prob:.4f}  {bar}")
            print()
        except Exception as exc:
            logger.warning(f"ML augmenter not available: {exc}")

    # ── period from features ──────────────────────────────────────────────────
    if "g_log10_period" in FEATURE_NAMES:
        p_idx = FEATURE_NAMES.index("g_log10_period")
        if np.isfinite(query_vec[p_idx]):
            period_val = float(10 ** query_vec[p_idx])
            fap_idx    = FEATURE_NAMES.index("g_period_fap_log10") \
                         if "g_period_fap_log10" in FEATURE_NAMES else None
            fap_log    = float(query_vec[fap_idx]) if fap_idx is not None else 0.0
            logger.info(
                f"  Best period: {period_val:.4f} d  "
                f"(FAP log10 = {fap_log:.1f})"
            )

    # ── skip all plotting if nothing requested ────────────────────────────────
    if not any([plot, summary, save_plot, save_summary, embed]):
        return

    # ── download result light curves ─────────────────────────────────────────
    logger.info("Downloading result light curves for plotting …")
    result_lcs = downloader.get_lightcurves_batch(
        result_oids,
        n_workers=cfg.catalog.n_workers,
        use_cache=True,
    )
    logger.info(
        f"  Downloaded {len(result_lcs)}/{len(result_oids)} result LCs"
    )

    import matplotlib.pyplot as plt

    # ── result grid ───────────────────────────────────────────────────────────
    if plot or save_plot:
        fig_grid = plot_results(
            query_lc   = query_lc,
            results    = results,
            result_lcs = result_lcs,
            query_oid  = oid,
            n_cols     = 3,
            panel_width  = 5.5,
            panel_height = 3.2,
            max_results  = min(topk, 12),
            metadata   = meta if not meta.empty else None,
            show       = False,
            save       = save_plot or None,
        )
        if plot:
            plt.figure(fig_grid.number)
            plt.show(block=False)

    # ── object summary ────────────────────────────────────────────────────────
    if summary or save_summary:
        fig_sum = plot_object_summary(
            lc            = query_lc,
            oid           = oid,
            feature_vec   = query_vec,
            feature_names = FEATURE_NAMES,
            period        = period_val,
            ml_proba      = ml_proba,
            ml_classes    = ml_classes,
            show          = False,
            save          = save_summary or None,
        )
        if summary:
            plt.figure(fig_sum.number)
            plt.show(block=False)

    # ── keep windows open ─────────────────────────────────────────────────────
    if plot or summary:
        plt.show(block=True)

    # ── feature-space embedding ───────────────────────────────────────────────
    if embed:
        logger.info(f"Building {embed_method.upper()} embedding …")
        all_oids, all_X = db.load_all()
        all_meta        = db.load_metadata()

        labels = None
        if "cls" in all_meta.columns:
            meta_map = dict(zip(all_meta["oid"], all_meta["cls"]))
            labels   = [meta_map.get(o, "Unknown") for o in all_oids]

        fig_emb = plot_feature_space(
            oids       = all_oids,
            features   = all_X,
            labels     = labels,
            query_oid  = oid,
            query_vec  = query_vec,
            method     = embed_method,
            show       = True,
        )


if __name__ == "__main__":
    cli()
