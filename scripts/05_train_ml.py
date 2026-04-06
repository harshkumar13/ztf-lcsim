#!/usr/bin/env python
"""
Step 5 – Train the ML feature augmenter on the labeled database.

Usage:
    python scripts/05_train_ml.py
    python scripts/05_train_ml.py --min-per-class 50 --n-estimators 500
    python scripts/05_train_ml.py --rebuild-index  # retrain AND rebuild index
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click
import numpy as np
import pandas as pd

from ztf_lcsim.config     import Config
from ztf_lcsim.database   import FeatureDatabase
from ztf_lcsim.ml_features import MLFeatureAugmenter
from ztf_lcsim.index      import SimilarityIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ml")


@click.command()
@click.option("--config",         default="config.yaml")
@click.option("--min-per-class",  default=20,  show_default=True,
              help="Min objects per class (smaller classes are merged into 'Other')")
@click.option("--n-estimators",   default=300, show_default=True)
@click.option("--ml-weight",      default=3.0, show_default=True,
              help="Weight multiplier for ML probability features in index.")
@click.option("--rebuild-index",  is_flag=True,
              help="Rebuild similarity index after training.")
def cli(config, min_per_class, n_estimators, ml_weight, rebuild_index):
    """Train ML augmenter and (optionally) rebuild the similarity index."""

    cfg = Config(config)
    db  = FeatureDatabase(cfg.features_path, cfg.metadata_path)

    # ── load data ─────────────────────────────────────────────────────────────
    logger.info("Loading feature database...")
    oids, X = db.load_all()
    if len(oids) == 0:
        logger.error("Database empty. Run 01_build_database.py first.")
        sys.exit(1)

    meta  = db.load_metadata()
    logger.info(f"Loaded {len(oids):,} objects, {X.shape[1]} features")

    # ── build label array ─────────────────────────────────────────────────────
    meta_map = (
        dict(zip(meta["oid"], meta["cls"]))
        if "cls" in meta.columns else {}
    )
    labels_raw = [meta_map.get(o, "Unknown") for o in oids]

    # merge rare classes into "Other"
    label_counts = pd.Series(labels_raw).value_counts()
    rare         = set(label_counts[label_counts < min_per_class].index)
    labels       = [
        "Other" if (l in rare or l == "Unknown" or not l) else l
        for l in labels_raw
    ]

    # remove objects with no class
    valid = [i for i, l in enumerate(labels) if l]
    if len(valid) < 50:
        logger.error(
            f"Only {len(valid)} labeled objects. "
            "Add more classes to the database first."
        )
        sys.exit(1)

    X_fit  = X[valid]
    y_fit  = [labels[i] for i in valid]

    class_counts = pd.Series(y_fit).value_counts()
    logger.info(f"Training on {len(X_fit):,} objects:\n{class_counts.to_string()}")

    # ── train ─────────────────────────────────────────────────────────────────
    aug = MLFeatureAugmenter(
        n_estimators=n_estimators,
        calibrate=True,
        random_state=42,
    )
    aug.fit(X_fit, y_fit, verbose=True)

    # ── save augmenter ────────────────────────────────────────────────────────
    aug_path = cfg.db_dir / "ml_augmenter.pkl"
    aug.save(aug_path)
    logger.info(f"Saved: {aug_path}")

    # ── quick sanity check ────────────────────────────────────────────────────
    logger.info("Sanity check: probabilities for first 3 objects")
    proba = aug.predict_proba(X[:3])
    for i in range(3):
        pred = aug.classes_[int(np.argmax(proba[i]))]
        true = labels[i] if i < len(labels) else "?"
        conf = float(np.max(proba[i]))
        logger.info(
            f"  {oids[i]}  true={true:<15} pred={pred:<15} conf={conf:.3f}"
        )

    # ── rebuild index with ML features ────────────────────────────────────────
    if rebuild_index:
        logger.info("Rebuilding similarity index with ML augmentation...")

        idx = SimilarityIndex(
            index_type=cfg.index.type,
            metric=cfg.index.metric,
            ivf_nlist=cfg.index.ivf_nlist,
            ivf_nprobe=cfg.index.ivf_nprobe,
            hnsw_m=cfg.index.hnsw_m,
            use_feature_weights=True,
            ml_weight=ml_weight,
        )
        idx.build(oids, X, verbose=True, ml_augmenter=aug)
        idx.save(cfg.index_path)
        logger.info(f"Index saved: {cfg.index_path}")
    else:
        logger.info(
            "To rebuild the index with ML features, run:\n"
            "  python scripts/05_train_ml.py --rebuild-index\n"
            "  OR: python scripts/02_build_index.py  "
            "(will use ML augmenter automatically if present)"
        )

    logger.info("Done ✓")


if __name__ == "__main__":
    cli()
