"""Hierarchical YAML configuration with dot-access."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Optional

# ── default values ────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "database": {
        "dir": "./ztf_data",
        "features_file": "features.h5",
        "metadata_file": "metadata.db",
        "cache_subdir": "lc_cache",
    },
    "index": {
        "file": "similarity.index",
        "type": "flat",
        "metric": "cosine",
        "ivf_nlist": 256,
        "ivf_nprobe": 32,
        "hnsw_m": 32,
    },
    "alerce": {
        "timeout": 30,
        "max_retries": 3,
        "request_delay": 0.1,
    },
    "features": {
        "bands": [1, 2],
        "min_obs_per_band": 5,
        "ls_min_period": 0.1,
        "ls_max_period": 500.0,
        "ls_samples_per_peak": 5,
    },
    "catalog": {
        "classes": ["RRL", "LPV", "EB", "DSCT", "CEP",
                    "SNIa", "SNIbc", "SNII", "QSO", "AGN"],
        "min_probability": 0.6,
        "max_per_class": 10000,
        "n_workers": 4,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class _Node:
    """Internal tree node supporting attribute access."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_d", data)

    def __getattr__(self, name: str) -> Any:
        d = object.__getattribute__(self, "_d")
        if name not in d:
            raise AttributeError(f"No config key '{name}'")
        v = d[name]
        return _Node(v) if isinstance(v, dict) else v

    def __setattr__(self, name: str, value: Any):
        object.__getattribute__(self, "_d")[name] = value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def to_dict(self) -> dict:
        return object.__getattribute__(self, "_d").copy()

    def __repr__(self) -> str:
        return f"ConfigNode({object.__getattribute__(self, '_d')})"


class Config(_Node):
    """Top-level configuration object."""

    def __init__(self, path: Optional[str] = None):
        data = _DEFAULTS
        if path and Path(path).exists():
            with open(path) as fh:
                user = yaml.safe_load(fh) or {}
            data = _deep_merge(_DEFAULTS, user)
        super().__init__(data)

    # ── convenience path properties ──────────────────────────────────────────

    @property
    def db_dir(self) -> Path:
        return Path(self.database.dir)

    @property
    def features_path(self) -> Path:
        return self.db_dir / self.database.features_file

    @property
    def metadata_path(self) -> Path:
        return self.db_dir / self.database.metadata_file

    @property
    def index_path(self) -> Path:
        return self.db_dir / self.index.file

    @property
    def cache_dir(self) -> Path:
        return self.db_dir / self.database.cache_subdir

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            yaml.dump(object.__getattribute__(self, "_d"), fh,
                      default_flow_style=False)
