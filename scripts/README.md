# ztf_lcsim — ZTF Light Curve Similarity Search Engine

Find ZTF objects with light curves similar to any target using a
**fast feature-based ANN search** backed by [ALeRCE](https://alerce.online).

---

## Quick start

```bash
# 1. Install
git clone https://github.com/yourname/ztf_lcsim
cd ztf_lcsim
pip install -e ".[faiss]"      # includes faiss-cpu for fast search
                                # or:  pip install -e .   (uses sklearn)

# 2. Build the feature database  (~30 min for 50 k objects)
python scripts/01_build_database.py \
    --classes RRL LPV EB DSCT \
    --max-per-class 5000

# 3. Build the similarity index  (~2 min)
python scripts/02_build_index.py

# 4. Search!
python scripts/03_search.py \
    --oid ZTF25acemaph \
    --topk 20 \
    --plot \
    --save-plot results/ZTF25acemaph.pdf
