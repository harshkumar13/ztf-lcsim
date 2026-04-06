---

## End-to-end example (`examples/demo.py`)

```python
#!/usr/bin/env python
"""
Minimal end-to-end demo:
  1. Download one light curve
  2. Extract features
  3. Search a pre-built index
  4. Plot results
"""

import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from ztf_lcsim import (
    AlerceDownloader,
    FeatureExtractor,
    FeatureDatabase,
    SimilarityIndex,
    plot_lightcurve,
    plot_results,
)
from ztf_lcsim.config import Config

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
cfg = Config("config.yaml")
QUERY_OID = "ZTF25acemaph"      # ← your target object
TOP_K = 20

# ─────────────────────────────────────────────────────────────────────────────
# Download query light curve
# ─────────────────────────────────────────────────────────────────────────────
dl = AlerceDownloader(cache_dir=str(cfg.cache_dir))
lc = dl.get_lightcurve(QUERY_OID)
print(f"\nQuery light curve: {len(lc)} detections\n{lc[['mjd','fid','magpsf','sigmapsf']].head()}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Extract features
# ─────────────────────────────────────────────────────────────────────────────
fe = FeatureExtractor()
vec = fe.extract(lc)
print("Feature vector (first 10):")
for name, val in zip(fe.feature_names[:10], vec[:10]):
    print(f"  {name:30s} {val:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Load index & search
# ─────────────────────────────────────────────────────────────────────────────
idx = SimilarityIndex().load(cfg.index_path)
results = idx.search(vec, k=TOP_K)
print(f"\nTop-{TOP_K} similar objects:\n{results.to_string(index=False)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Download result light curves & plot
# ─────────────────────────────────────────────────────────────────────────────
result_oids = results["oid"].tolist()
result_lcs  = dl.get_lightcurves_batch(result_oids, n_workers=4)

fig = plot_results(
    query_lc   = lc,
    results    = results,
    result_lcs = result_lcs,
    query_oid  = QUERY_OID,
    n_cols     = 4,
    show       = True,
)
fig.savefig(f"{QUERY_OID}_similar.pdf", dpi=150, bbox_inches="tight")
print(f"Plot saved to {QUERY_OID}_similar.pdf")
