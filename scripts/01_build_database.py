#!/usr/bin/env python
"""
Step 1 – Download ZTF light curves from ALeRCE and build the feature database.

Usage examples
--------------
python scripts/01_build_database.py --classes RRL,LPV,E
python scripts/01_build_database.py --classes SNIa,AGN,RRL --max-per-class 1000
python scripts/01_build_database.py   # uses all classes in config.yaml
python scripts/01_build_database.py --classes SNIa,AGN --dry-run

ALeRCE lc_classifier class names (verified 2025-01):
  Periodic  : RRL, LPV, E, DSCT, CEP, Periodic-Other
  Stochastic: AGN, QSO, Blazar, CV/Nova, YSO
  Transient : SNIa, SNIbc, SNII, SLSN
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_database")


def _parse_classes(classes_tuple: tuple, config_classes: list) -> list:
    """
    Accept comma-separated or repeated flags:
        --classes RRL,LPV,E       → ["RRL", "LPV", "E"]
        --classes RRL --classes E → ["RRL", "E"]
        (nothing)                 → config_classes
    """
    if not classes_tuple:
        return list(config_classes)
    result = []
    for item in classes_tuple:
        for part in item.split(","):
            part = part.strip()
            if part:
                result.append(part)
    return result


@click.command()
@click.option("--config", default="config.yaml", show_default=True)
@click.option(
    "--classes", multiple=True, metavar="CLASS[,CLASS,...]",
    help="Classes to download. Comma-separated or repeated. "
         "Default: from config.yaml. "
         "Valid (lc_classifier): RRL,LPV,E,DSCT,CEP,Periodic-Other,"
         "AGN,QSO,Blazar,CV/Nova,YSO,SNIa,SNIbc,SNII,SLSN",
)
@click.option(
    "--classifier", default=None,
    help="ALeRCE classifier name. Default: from config.yaml (lc_classifier).",
)
@click.option("--max-per-class", "max_per_class", default=None, type=int)
@click.option("--min-prob",      "min_prob",      default=None, type=float)
@click.option("--n-workers",     "n_workers",     default=None, type=int)
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--dry-run", is_flag=True)
def cli(config, classes, classifier, max_per_class, min_prob,
        n_workers, resume, dry_run):
    """Download ZTF light curves and build the feature database."""

    cfg = Config(config)

    # ── resolve parameters ────────────────────────────────────────────────────
    classes_list  = _parse_classes(classes, cfg.catalog.classes)
    classifier    = classifier    or cfg.catalog.classifier
    max_per_class = max_per_class or cfg.catalog.max_per_class
    min_prob      = min_prob      or cfg.catalog.min_probability
    n_workers     = n_workers     or cfg.catalog.n_workers

    cfg.db_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  ZTF Light Curve Similarity — Build Database")
    logger.info("=" * 60)
    logger.info(f"  Config       : {config}")
    logger.info(f"  Classifier   : {classifier}")
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

    # ── resume: load existing OIDs ────────────────────────────────────────────
    existing_oids: set = set()
    if resume and cfg.features_path.exists():
        existing, _ = db.load_all()
        existing_oids = set(existing)
        logger.info(
            f"Resume mode: {len(existing_oids):,} objects already in database"
        )

    total_added = 0

    # ══════════════════════════════════════════════════════════════════════════
    for cls in classes_list:
        logger.info("")
        logger.info(f"── Class: {cls} " + "─" * 38)

        # ── query catalog ─────────────────────────────────────────────────────
        catalog = downloader.query_objects(
            class_name=cls,
            classifier=classifier,          # ← pass from config / CLI
            min_probability=min_prob,
            max_objects=max_per_class,
        )

        if catalog.empty:
            logger.warning(f"  No objects for class '{cls}' — skipping")
            continue

        # ── find OID column ───────────────────────────────────────────────────
        # ALeRCE catalog always returns 'oid'
        oid_col = _find_oid_col(catalog)
        if oid_col is None:
            logger.warning(
                f"  No OID column found. Columns: {list(catalog.columns)}"
            )
            continue

        oids = catalog[oid_col].tolist()
        logger.info(f"  Catalog : {len(oids):,} objects")

        # Print column names once per class for debugging
        logger.debug(f"  Catalog columns: {list(catalog.columns)}")

        # ── skip already-processed ────────────────────────────────────────────
        if resume:
            oids_new = [o for o in oids if o not in existing_oids]
            n_skip   = len(oids) - len(oids_new)
            if n_skip:
                logger.info(f"  Skipping {n_skip:,} already in DB")
            logger.info(f"  New to process: {len(oids_new):,}")
            oids = oids_new

        if not oids:
            logger.info("  Nothing new — skipping")
            continue

        # ── process in batches ────────────────────────────────────────────────
        BATCH_SIZE = 500
        n_batches  = max(1, (len(oids) + BATCH_SIZE - 1) // BATCH_SIZE)

        for batch_i in range(n_batches):
            batch_oids = oids[batch_i * BATCH_SIZE: (batch_i + 1) * BATCH_SIZE]
            if not batch_oids:
                continue

            logger.info(
                f"  Batch {batch_i + 1}/{n_batches} "
                f"({len(batch_oids)} objects)"
            )

            # download
            lcs = downloader.get_lightcurves_batch(
                batch_oids, n_workers=n_workers, use_cache=True,
            )
            if not lcs:
                logger.warning("  No light curves downloaded")
                continue

            logger.info(f"  Downloaded: {len(lcs):,}/{len(batch_oids)}")

            # extract features
            feat_oids, feat_matrix = extractor.extract_batch(
                lcs, show_progress=True
            )
            if not feat_oids:
                logger.warning("  No features extracted")
                continue

            # ── build metadata ────────────────────────────────────────────────
            # Catalog columns: oid, meanra, meandec, ndet, probability, class, ...
            meta_rows = []
            for oid in feat_oids:
                lc  = lcs[oid]
                row = {
                    "oid":         oid,
                    "cls":         cls,
                    "probability": _get_value(catalog, oid_col, oid, "probability"),
                    "n_obs_g":     int((lc["fid"] == 1).sum()),
                    "n_obs_r":     int((lc["fid"] == 2).sum()),
                }

                # Coordinates: ALeRCE catalog uses 'meanra'/'meandec'
                ra = _get_value(
                    catalog, oid_col, oid,
                    "meanra", "ra", "RA"
                )
                dec = _get_value(
                    catalog, oid_col, oid,
                    "meandec", "dec", "Dec", "DEC"
                )
                if ra  is not None: row["ra"]  = float(ra)
                if dec is not None: row["dec"] = float(dec)

                meta_rows.append(row)

            meta_df = pd.DataFrame(meta_rows)
            db.add(feat_oids, feat_matrix, meta_df)
            existing_oids.update(feat_oids)
            total_added += len(feat_oids)

            logger.info(
                f"  ✓ Added {len(feat_oids):,} | "
                f"Total in DB: {len(existing_oids):,}"
            )

    # ── final summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Done. Added {total_added:,} objects this run.")
    stats = db.stats()
    logger.info(f"  Total in DB  : {stats['n_objects']:,}")
    if stats["class_counts"]:
        logger.info("  Class breakdown:")
        for name, cnt in sorted(stats["class_counts"].items()):
            logger.info(f"    {name:<20} {cnt:>6,}")
    logger.info("=" * 60)


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_oid_col(df: pd.DataFrame) -> str | None:
    for c in ("oid", "objectId", "object_id", "id"):
        if c in df.columns:
            return c
    return None


def _get_value(
    catalog: pd.DataFrame,
    oid_col: str,
    oid: str,
    *col_names: str,
) -> float | None:
    """
    Look up a value for *oid* trying multiple possible column names.
    Returns the first match, or None.
    """
    mask = catalog[oid_col] == oid
    if not mask.any():
        return None
    row = catalog[mask].iloc[0]
    for col in col_names:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return None


if __name__ == "__main__":
    cli()

