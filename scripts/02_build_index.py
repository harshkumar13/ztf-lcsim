#!/usr/bin/env python
"""
Step 2 – Build the FAISS similarity index from the feature database.

Automatically uses the ML augmenter (scripts/05_train_ml.py output)
if it exists in the data directory.

Usage
-----
python scripts/02_build_index.py
python scripts/02_build_index.py --config config.yaml
python scripts/02_build_index.py --index-type ivf
python scripts/02_build_index.py --no-ml        # skip ML augmentation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click

from ztf_lcsim.config  import Config
from ztf_lcsim.database import FeatureDatabase
from ztf_lcsim.index   import SimilarityIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


@click.command()
@click.option("--config",      default="config.yaml", show_default=True)
@click.option(
    "--index-type", "index_type", default=None,
    type=click.Choice(["flat", "ivf", "hnsw"]),
    help="Override index type from config (flat/ivf/hnsw).",
)
@click.option(
    "--metric", default=None,
    type=click.Choice(["cosine", "l2"]),
    help="Override similarity metric from config.",
)
@click.option(
    "--ml-weight", "ml_weight", default=3.0, show_default=True,
    help="Weight multiplier for ML probability features.",
)
@click.option(
    "--no-ml", "no_ml", is_flag=True,
    help="Skip ML augmentation even if augmenter file exists.",
)
@click.option(
    "--no-weights", "no_weights", is_flag=True,
    help="Disable domain-knowledge feature weighting.",
)
def cli(config, index_type, metric, ml_weight, no_ml, no_weights):
    """Build the similarity index from the feature database."""

    cfg    = Config(config)
    itype  = index_type or cfg.index.type
    metric = metric     or cfg.index.metric

    # ── load features ─────────────────────────────────────────────────────────
    logger.info(f"Loading features from: {cfg.features_path}")
    db = FeatureDatabase(cfg.features_path, cfg.metadata_path)
    oids, X = db.load_all()

    if len(oids) == 0:
        logger.error(
            "Feature database is empty!\n"
            "Run:  python scripts/01_build_database.py  first."
        )
        sys.exit(1)

    logger.info(f"Loaded {len(oids):,} objects × {X.shape[1]} features")

    # ── class distribution ─────────────────────────────────────────────────────
    meta = db.load_metadata()
    if "cls" in meta.columns:
        counts = meta["cls"].value_counts()
        logger.info(f"Class breakdown:\n{counts.to_string()}")

    # ── try to load ML augmenter ───────────────────────────────────────────────
    ml_aug   = None
    aug_path = cfg.db_dir / "ml_augmenter.pkl"

    if no_ml:
        logger.info("ML augmentation disabled (--no-ml).")
    elif aug_path.exists():
        try:
            from ztf_lcsim.ml_features import MLFeatureAugmenter
            ml_aug = MLFeatureAugmenter.load(aug_path)
            logger.info(
                f"ML augmenter loaded: {ml_aug.n_prob_features} class features "
                f"| classes = {ml_aug.classes_}"
            )
            if ml_aug.cv_scores_ is not None:
                cv = ml_aug.cv_scores_
                logger.info(
                    f"  CV balanced accuracy: {cv.mean():.3f} ± {cv.std():.3f}"
                )
        except Exception as exc:
            logger.warning(
                f"Could not load ML augmenter: {exc}\n"
                "Building index without ML features. "
                "Run  scripts/05_train_ml.py  to create it."
            )
    else:
        logger.info(
            f"No ML augmenter found at {aug_path}.\n"
            "  → Run  python scripts/05_train_ml.py  for improved matching.\n"
            "  → Building index with statistical features only."
        )

    # ── build index ────────────────────────────────────────────────────────────
    idx = SimilarityIndex(
        index_type=itype,
        metric=metric,
        ivf_nlist=cfg.index.ivf_nlist,
        ivf_nprobe=cfg.index.ivf_nprobe,
        hnsw_m=cfg.index.hnsw_m,
        use_feature_weights=not no_weights,
        ml_weight=ml_weight,
    )

    idx.build(oids, X, verbose=True, ml_augmenter=ml_aug)

    # ── save ───────────────────────────────────────────────────────────────────
    idx.save(cfg.index_path)
    logger.info(f"Index saved to: {cfg.index_path}")

    # ── sanity check ───────────────────────────────────────────────────────────
    logger.info("Sanity check — querying first object...")
    test_vec = X[0]
    results  = idx.search(test_vec, k=5, exclude_self=False)
    logger.info(
        f"Top-5 for {oids[0]}:\n"
        f"{results[['rank','oid','similarity']].to_string(index=False)}"
    )

    # ── summary ────────────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info(f"  Objects in index : {idx.n_objects:,}")
    logger.info(f"  Feature dims     : {X.shape[1]}")
    if ml_aug is not None:
        logger.info(f"  ML classes       : {ml_aug.classes_}")
    logger.info(f"  Index type       : {itype}")
    logger.info(f"  Metric           : {metric}")
    logger.info(f"  Weights          : {'yes' if not no_weights else 'no'}")
    logger.info("=" * 55)


if __name__ == "__main__":
    cli()
