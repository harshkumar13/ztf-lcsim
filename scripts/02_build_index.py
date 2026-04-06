#!/usr/bin/env python
"""
Step 2 – Build the FAISS similarity index from the feature database.

Usage
-----
python scripts/02_build_index.py --config config.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from ztf_lcsim.config import Config
from ztf_lcsim.database import FeatureDatabase
from ztf_lcsim.index import SimilarityIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


@click.command()
@click.option("--config",      default="config.yaml", show_default=True)
@click.option("--index-type",  default=None,
              type=click.Choice(["flat", "ivf", "hnsw"]),
              help="Override index type from config.")
@click.option("--metric",      default=None,
              type=click.Choice(["cosine", "l2"]),
              help="Override similarity metric from config.")
def cli(config, index_type, metric):
    """Build the similarity index from the feature database."""
    cfg = Config(config)

    itype  = index_type or cfg.index.type
    metric = metric     or cfg.index.metric

    logger.info(f"Loading features from: {cfg.features_path}")
    db = FeatureDatabase(cfg.features_path, cfg.metadata_path)
    oids, X = db.load_all()

    if len(oids) == 0:
        logger.error("Feature database is empty! Run 01_build_database.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(oids):,} objects with {X.shape[1]} features each")

    # ── build index ───────────────────────────────────────────────────────────
    idx = SimilarityIndex(
        index_type=itype,
        metric=metric,
        ivf_nlist=cfg.index.ivf_nlist,
        ivf_nprobe=cfg.index.ivf_nprobe,
        hnsw_m=cfg.index.hnsw_m,
    )
    idx.build(oids, X, verbose=True)

    # ── save ─────────────────────────────────────────────────────────────────
    idx.save(cfg.index_path)
    logger.info(f"Index saved to: {cfg.index_path}")

    # ── quick sanity check ────────────────────────────────────────────────────
    logger.info("Sanity check: querying first object...")
    import numpy as np
    test_vec = X[0]
    results = idx.search(test_vec, k=5, exclude_self=False)
    logger.info(f"Top-5 results for {oids[0]}:\n{results.to_string(index=False)}")


if __name__ == "__main__":
    cli()
