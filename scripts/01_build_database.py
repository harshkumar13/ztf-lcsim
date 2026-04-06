#!/usr/bin/env python
"""
Step 1 – Download ZTF light curves from ALeRCE and build the feature database.

Usage
-----
python scripts/01_build_database.py \\
    --config config.yaml \\
    --classes RRL LPV EB \\
    --max-per-class 5000 \\
    --n-workers 4
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make sure the package root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from ztf_lcsim.config import Config
from ztf_lcsim.downloader import AlerceDownloader
from ztf_lcsim.features import FeatureExtractor
from ztf_lcsim.database import FeatureDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_database")


@click.command()
@click.option("--config",        default="config.yaml", show_default=True)
@click.option("--classes",       multiple=True,
              help="ZTF object classes to include (e.g. RRL LPV EB). "
                   "Default: from config.")
@click.option("--max-per-class", default=None, type=int,
              help="Override max_per_class from config.")
@click.option("--min-prob",      default=None, type=float,
              help="Override min_probability from config.")
@click.option("--n-workers",     default=None, type=int,
              help="Parallel download workers.")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip objects already in the database.")
@click.option("--dry-run",       is_flag=True,
              help="Only print what would be done; don't download.")
def cli(config, classes, max_per_class, min_prob, n_workers, resume, dry_run):
    """Download ZTF light curves and build the feature database."""

    cfg = Config(config)

    # ── resolve parameters ────────────────────────────────────────────────────
    classes_list  = list(classes) if classes else cfg.catalog.classes
    max_per_class = max_per_class or cfg.catalog.max_per_class
    min_prob      = min_prob      or cfg.catalog.min_probability
    n_workers     = n_workers     or cfg.catalog.n_workers

    cfg.db_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config      : {config}")
    logger.info(f"Classes     : {classes_list}")
    logger.info(f"Max/class   : {max_per_class}")
    logger.info(f"Min prob    : {min_prob}")
    logger.info(f"Workers     : {n_workers}")
    logger.info(f"Output dir  : {cfg.db_dir}")

    if dry_run:
        logger.info("DRY RUN – exiting.")
        return

    # ── initialise components ─────────────────────────────────────────────────
    downloader = AlerceDownloader(
        cache_dir=str(cfg.cache_dir),
        timeout=cfg.alerce.timeout,
        max_retries=cfg.alerce.max_retries,
        request_delay=cfg.alerce.request_delay,
    )

    extractor = FeatureExtractor(
        bands=cfg.features.bands,
        min_obs=cfg.features.min_obs_per_band,
        ls_min_period=cfg.features.ls_min_period,
        ls_max_period=cfg.features.ls_max_period,
        ls_samples_per_peak=cfg.features.ls_samples_per_peak,
    )

    db = FeatureDatabase(
        features_path=cfg.features_path,
        metadata_path=cfg.metadata_path,
    )

    # ── pre-load existing OIDs ────────────────────────────────────────────────
    existing_oids: set[str] = set()
    if resume and cfg.features_path.exists():
        existing, _ = db.load_all()
        existing_oids = set(existing)
        logger.info(f"Database has {len(existing_oids):,} existing objects "
                    "(--resume is on)")

    # ── process each class ────────────────────────────────────────────────────
    total_added = 0

    for cls in classes_list:
        logger.info(f"── Class: {cls} ──────────────────────────────────────")

        # Query catalog
        catalog = downloader.query_objects(
            class_name=cls,
            min_probability=min_prob,
            max_objects=max_per_class,
        )

        if catalog.empty:
            logger.warning(f"  No objects returned for class {cls}")
            continue

        oid_col = _find_oid_col(catalog)
        if oid_col is None:
            logger.warning(f"  Cannot find OID column in catalog; skipping {cls}")
            continue

        oids = catalog[oid_col].tolist()
        logger.info(f"  Found {len(oids):,} objects")

        # Skip already-processed
        if resume:
            oids = [o for o in oids if o not in existing_oids]
            logger.info(f"  {len(oids):,} new objects to process")

        if not oids:
            continue

        # Download in batches
        BATCH = 500
        for batch_start in range(0, len(oids), BATCH):
            batch_oids = oids[batch_start: batch_start + BATCH]
            logger.info(f"  Downloading batch "
                        f"{batch_start // BATCH + 1}/"
                        f"{len(oids) // BATCH + 1} "
                        f"({len(batch_oids)} objects)")

            lcs = downloader.get_lightcurves_batch(
                batch_oids,
                n_workers=n_workers,
                use_cache=True,
            )

            if not lcs:
                continue

            # Extract features
            feat_oids, feat_matrix = extractor.extract_batch(lcs)

            if len(feat_oids) == 0:
                continue

            # Build metadata DataFrame for this batch
            meta_rows = []
            for oid in feat_oids:
                lc = lcs[oid]
                row = {
                    "oid": oid,
                    "cls": cls,
                    "probability": _get_prob(catalog, oid_col, oid),
                    "n_obs_g": int((lc["fid"] == 1).sum()),
                    "n_obs_r": int((lc["fid"] == 2).sum()),
                }
                # add ra/dec if available
                if "ra" in lc.columns:
                    row["ra"] = float(lc["ra"].median())
                if "dec" in lc.columns:
                    row["dec"] = float(lc["dec"].median())
                meta_rows.append(row)

            meta_df = pd.DataFrame(meta_rows)
            db.add(feat_oids, feat_matrix, meta_df)
            existing_oids.update(feat_oids)
            total_added += len(feat_oids)

            logger.info(f"    → Added {len(feat_oids):,} objects "
                        f"(total in DB: {len(existing_oids):,})")

    logger.info(f"Done. Added {total_added:,} objects in this run.")
    stats = db.stats()
    logger.info(f"Database stats: {stats}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_oid_col(df: pd.DataFrame) -> str | None:
    for c in ("oid", "objectId", "object_id", "id"):
        if c in df.columns:
            return c
    return None


def _get_prob(catalog: pd.DataFrame, oid_col: str, oid: str) -> float | None:
    mask = catalog[oid_col] == oid
    if mask.any():
        row = catalog[mask].iloc[0]
        for col in ("probability", "prob", "lc_classifier_top_probability"):
            if col in row.index:
                return float(row[col])
    return None


if __name__ == "__main__":
    cli()
