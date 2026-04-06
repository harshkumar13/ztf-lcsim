#!/usr/bin/env python
"""
Step 3 – Search for objects similar to a given ZTF OID.

Usage
-----
python scripts/03_search.py \\
    --config config.yaml \\
    --oid ZTF25acemaph \\
    --topk 20 \\
    --plot \\
    --save-plot results/ZTF25acemaph_similar.pdf
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click
import pandas as pd

from ztf_lcsim.config import Config
from ztf_lcsim.downloader import AlerceDownloader
from ztf_lcsim.features import FeatureExtractor
from ztf_lcsim.database import FeatureDatabase
from ztf_lcsim.index import SimilarityIndex
from ztf_lcsim.plots import plot_lightcurve, plot_results, plot_feature_space

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("search")


@click.command()
@click.option("--config",         default="config.yaml",  show_default=True)
@click.option("--oid",            required=True, help="Query object ID.")
@click.option("--topk",           default=20,   show_default=True,
              help="Number of similar objects to return.")
@click.option("--plot/--no-plot", default=True, show_default=True,
              help="Show the result grid plot.")
@click.option("--save-plot",      default=None,
              help="Save plot to this file (PDF/PNG/SVG).")
@click.option("--save-csv",       default=None,
              help="Save result table as CSV.")
@click.option("--embed/--no-embed", default=False,
              help="Show 2-D feature-space embedding.")
def cli(config, oid, topk, plot, save_plot, save_csv, embed):
    """Search for ZTF objects with similar light curves."""

    cfg = Config(config)

    # ── load index ────────────────────────────────────────────────────────────
    if not Path(str(cfg.index_path) + ".meta").exists():
        logger.error(
            f"Index not found at {cfg.index_path}. "
            "Run 02_build_index.py first."
        )
        sys.exit(1)

    logger.info(f"Loading index from {cfg.index_path} …")
    idx = SimilarityIndex(
        index_type=cfg.index.type,
        metric=cfg.index.metric,
        ivf_nprobe=cfg.index.ivf_nprobe,
    ).load(cfg.index_path)
    logger.info(f"Index ready: {idx.n_objects:,} objects")

    # ── load database (for metadata) ──────────────────────────────────────────
    db = FeatureDatabase(cfg.features_path, cfg.metadata_path)

    # ── download query light curve ────────────────────────────────────────────
    downloader = AlerceDownloader(
        cache_dir=str(cfg.cache_dir),
        timeout=cfg.alerce.timeout,
        max_retries=cfg.alerce.max_retries,
        request_delay=cfg.alerce.request_delay,
    )

    logger.info(f"Fetching light curve for query: {oid}")
    query_lc = downloader.get_lightcurve(oid, use_cache=True)
    if query_lc is None or query_lc.empty:
        logger.error(f"Could not fetch light curve for {oid}.")
        sys.exit(1)

    logger.info(f"  Detections: {len(query_lc)}  "
                f"(g={int((query_lc.fid==1).sum())}, "
                f"r={int((query_lc.fid==2).sum())})")

    # ── extract features ──────────────────────────────────────────────────────
    extractor = FeatureExtractor(
        bands=cfg.features.bands,
        min_obs=cfg.features.min_obs_per_band,
        ls_min_period=cfg.features.ls_min_period,
        ls_max_period=cfg.features.ls_max_period,
    )
    query_vec = extractor.extract(query_lc)

    nan_frac = float((~__import__("numpy").isfinite(query_vec)).mean())
    logger.info(f"  Feature NaN fraction: {nan_frac:.1%}")

    # ── search ────────────────────────────────────────────────────────────────
    logger.info(f"Searching top-{topk} …")
    results = idx.search(query_vec, k=topk, exclude_self=True)

    print("\n" + "=" * 60)
    print(f"  Top-{topk} similar objects for  {oid}")
    print("=" * 60)
    print(results.to_string(index=False))
    print("=" * 60 + "\n")

    if save_csv:
        results.to_csv(save_csv, index=False)
        logger.info(f"Results saved to {save_csv}")

    # ── metadata enrichment ───────────────────────────────────────────────────
    result_oids = results["oid"].tolist()
    meta = db.load_metadata(result_oids)
    results_enriched = results.merge(meta, on="oid", how="left")
    if not results_enriched.empty:
        print("Enriched results (with class info):")
        print(results_enriched[["rank", "oid", "similarity",
                                 "cls", "probability"]].to_string(index=False))
        print()

    if not (plot or save_plot or embed):
        return

    # ── download result light curves ──────────────────────────────────────────
    logger.info("Downloading result light curves for plotting …")
    result_lcs = downloader.get_lightcurves_batch(
        result_oids, n_workers=cfg.catalog.n_workers, use_cache=True
    )

    # ── plot result grid ──────────────────────────────────────────────────────
    if plot or save_plot:
        fig = plot_results(
            query_lc=query_lc,
            results=results,
            result_lcs=result_lcs,
            query_oid=oid,
            n_cols=4,
            metadata=meta if not meta.empty else None,
            show=False,
        )
        if save_plot:
            Path(save_plot).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_plot, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_plot}")
        if plot:
            import matplotlib.pyplot as plt
            plt.show()

    # ── feature-space embedding ───────────────────────────────────────────────
    if embed:
        logger.info("Loading all features for embedding …")
        all_oids, all_X = db.load_all()
        all_meta = db.load_metadata()

        labels = None
        if "cls" in all_meta.columns:
            meta_map = dict(zip(all_meta["oid"], all_meta["cls"]))
            labels = [meta_map.get(o, "?") for o in all_oids]

        fig_emb = plot_feature_space(
            oids=all_oids,
            features=all_X,
            labels=labels,
            query_oid=oid,
            query_vec=query_vec,
            method="umap",
            show=True,
        )


if __name__ == "__main__":
    cli()
