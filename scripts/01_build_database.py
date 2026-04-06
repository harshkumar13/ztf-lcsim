#!/usr/bin/env python
"""
Step 1 – Download ZTF light curves from ALeRCE and build the feature database.

Usage examples
--------------
# comma-separated (single flag)
python scripts/01_build_database.py --classes RRL,LPV,EB

# repeated flags  
python scripts/01_build_database.py --classes RRL --classes LPV --classes EB

# all default classes from config.yaml
python scripts/01_build_database.py

# with overrides
python scripts/01_build_database.py \\
    --classes RRL,LPV \\
    --max-per-class 5000 \\
    --min-prob 0.7 \\
    --n-workers 8
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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


def _parse_classes(classes_tuple: tuple, config_classes: list) -> list:
    """
    Accept any of these formats and always return a plain list of strings:
        --classes RRL,LPV,EB          → ["RRL", "LPV", "EB"]
        --classes RRL --classes LPV   → ["RRL", "LPV"]
        (nothing given)               → config_classes
    """
    if not classes_tuple:
        return list(config_classes)

    result = []
    for item in classes_tuple:
        # each item may itself be "RRL,LPV" if user used commas
        for part in item.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result


@click.command()
@click.option(
    "--config", default="config.yaml", show_default=True,
    help="Path to config.yaml",
)
@click.option(
    "--classes", multiple=True, metavar="CLASS[,CLASS,...]",
    help=(
        "ZTF object classes to include. "
        "Can be comma-separated ('RRL,LPV') or repeated ('--classes RRL --classes LPV'). "
        "Default: all classes in config.yaml."
    ),
)
@click.option(
    "--max-per-class", "max_per_class", default=None, type=int,
    help="Max objects per class (overrides config).",
)
@click.option(
    "--min-prob", "min_prob", default=None, type=float,
    help="Min classification probability (overrides config).",
)
@click.option(
    "--n-workers", "n_workers", default=None, type=int,
    help="Parallel download workers (overrides config).",
)
@click.option(
    "--resume/--no-resume", default=True, show_default=True,
    help="Skip objects already in the database.",
)
@click.option(
    "--dry-run", is_flag=True,
    help="Print what would be done without downloading anything.",
)
def cli(config, classes, max_per_class, min_prob, n_workers, resume, dry_run):
    """Download ZTF light curves and build the feature database."""

    cfg = Config(config)

    # ── resolve parameters ────────────────────────────────────────────────────
    classes_list  = _parse_classes(classes, cfg.catalog.classes)
    max_per_class = max_per_class or cfg.catalog.max_per_class
    min_prob      = min_prob      or cfg.catalog.min_probability
    n_workers     = n_workers     or cfg.catalog.n_workers

    cfg.db_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  ZTF Light Curve Similarity — Build Database")
    logger.info("=" * 60)
    logger.info(f"  Config file  : {config}")
    logger.info(f"  Classes      : {classes_list}")
    logger.info(f"  Max/class    : {max_per_class:,}")
    logger.info(f"  Min prob     : {min_prob}")
    logger.info(f"  Workers      : {n_workers}")
    logger.info(f"  Resume       : {resume}")
    logger.info(f"  Output dir   : {cfg.db_dir}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN — exiting without downloading.")
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
    existing_oids: set = set()
    if resume and cfg.features_path.exists():
        existing, _ = db.load_all()
        existing_oids = set(existing)
        logger.info(
            f"Resume mode: {len(existing_oids):,} objects already in database"
        )

    # ── process each class ────────────────────────────────────────────────────
    total_added = 0

    for cls in classes_list:
        logger.info("")
        logger.info(f"── Class: {cls} " + "─" * 40)

        # ── query catalog ─────────────────────────────────────────────────────
        catalog = downloader.query_objects(
            class_name=cls,
            min_probability=min_prob,
            max_objects=max_per_class,
        )

        if catalog.empty:
            logger.warning(f"  No objects returned for class '{cls}' — skipping")
            continue

        oid_col = _find_oid_col(catalog)
        if oid_col is None:
            logger.warning(
                f"  Cannot find OID column in catalog "
                f"(columns: {list(catalog.columns)}) — skipping '{cls}'"
            )
            continue

        oids = catalog[oid_col].tolist()
        logger.info(f"  Catalog returned : {len(oids):,} objects")

        # ── skip already-processed ────────────────────────────────────────────
        if resume:
            oids_new = [o for o in oids if o not in existing_oids]
            logger.info(
                f"  Already in DB    : {len(oids) - len(oids_new):,}"
            )
            logger.info(f"  New to process   : {len(oids_new):,}")
            oids = oids_new

        if not oids:
            logger.info("  Nothing new to process — skipping")
            continue

        # ── process in batches ────────────────────────────────────────────────
        BATCH_SIZE = 500
        n_batches  = max(1, (len(oids) + BATCH_SIZE - 1) // BATCH_SIZE)

        for batch_i in range(n_batches):
            batch_oids = oids[batch_i * BATCH_SIZE : (batch_i + 1) * BATCH_SIZE]
            if not batch_oids:
                continue

            logger.info(
                f"  Batch {batch_i + 1}/{n_batches} "
                f"({len(batch_oids)} objects)"
            )

            # download
            lcs = downloader.get_lightcurves_batch(
                batch_oids,
                n_workers=n_workers,
                use_cache=True,
            )

            if not lcs:
                logger.warning("  No light curves retrieved for this batch")
                continue

            logger.info(
                f"  Downloaded : {len(lcs):,}/{len(batch_oids)} successfully"
            )

            # extract features
            feat_oids, feat_matrix = extractor.extract_batch(
                lcs, show_progress=True
            )

            if not feat_oids:
                logger.warning("  No features extracted for this batch")
                continue

            # build metadata
            meta_rows = []
            for oid in feat_oids:
                lc  = lcs[oid]
                row = {
                    "oid":         oid,
                    "cls":         cls,
                    "probability": _get_prob(catalog, oid_col, oid),
                    "n_obs_g":     int((lc["fid"] == 1).sum()),
                    "n_obs_r":     int((lc["fid"] == 2).sum()),
                }
                if "ra" in lc.columns:
                    row["ra"]  = float(lc["ra"].median())
                if "dec" in lc.columns:
                    row["dec"] = float(lc["dec"].median())
                meta_rows.append(row)

            meta_df = pd.DataFrame(meta_rows)
            db.add(feat_oids, feat_matrix, meta_df)
            existing_oids.update(feat_oids)
            total_added += len(feat_oids)

            logger.info(
                f"  ✓ Added {len(feat_oids):,} | "
                f"Total in DB: {len(existing_oids):,}"
            )

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Done. Added {total_added:,} objects this run.")
    stats = db.stats()
    logger.info(f"  Total in DB : {stats['n_objects']:,}")
    if stats["class_counts"]:
        logger.info("  Class breakdown:")
        for cls_name, cnt in sorted(stats["class_counts"].items()):
            logger.info(f"    {cls_name:<12} {cnt:>6,}")
    logger.info("=" * 60)


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_oid_col(df: pd.DataFrame) -> str | None:
    """Find the object-ID column regardless of its exact name."""
    for c in ("oid", "objectId", "object_id", "id", "ZTF_id"):
        if c in df.columns:
            return c
    return None


def _get_prob(
    catalog: pd.DataFrame, oid_col: str, oid: str
) -> float | None:
    """Extract classification probability for a single OID."""
    mask = catalog[oid_col] == oid
    if not mask.any():
        return None
    row = catalog[mask].iloc[0]
    for col in ("probability", "prob", "lc_classifier_top_probability"):
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return None


if __name__ == "__main__":
    cli()
