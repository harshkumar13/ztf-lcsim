PYTHON = python
DATA_DIR = ./ztf_data

.PHONY: help install build-db build-idx search clean

help:
	@echo "Usage:"
	@echo "  make install       Install dependencies"
	@echo "  make build-db      Download & build feature database"
	@echo "  make build-idx     Build similarity index from database"
	@echo "  make search OID=ZTF25acemaph  Search for similar objects"
	@echo "  make clean         Remove generated data"

install:
	pip install -e ".[faiss]"

build-db:
	$(PYTHON) scripts/01_build_database.py --config config.yaml

build-idx:
	$(PYTHON) scripts/02_build_index.py --config config.yaml

search:
	$(PYTHON) scripts/03_search.py --config config.yaml --oid $(OID) --topk 20 --plot

clean:
	rm -rf $(DATA_DIR)
