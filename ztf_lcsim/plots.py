"""
Visualisation utilities for light curves and similarity-search results.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_BAND_COLOUR = {1: "#2ecc71", 2: "#e74c3c", 3: "#3498db"}   # g=green r=red i=blue
_BAND_LABEL  = {1: "g", 2: "r", 3: "i"}


def plot_lightcurve(
    lc: pd.DataFrame,
    oid: str = "",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot a ZTF multi-band light curve.

    Parameters
    ----------
    lc : pd.DataFrame
        Must have columns ``mjd``, ``magpsf``, ``sigmapsf``, ``fid``.
    oid : str
        Object ID (used in title/label).
    ax : matplotlib Axes, optional
    title : str, optional
    show : bool
        If True, call ``plt.show()``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    for fid, grp in lc.groupby("fid"):
        color = _BAND_COLOUR.get(int(fid), "grey")
        label = _BAND_LABEL.get(int(fid), str(fid))
        ax.errorbar(
            grp["mjd"], grp["magpsf"], yerr=grp["sigmapsf"],
            fmt="o", color=color, label=label,
            markersize=3, elinewidth=0.8, alpha=0.8, zorder=3,
        )

    ax.invert_yaxis()
    ax.set_xlabel("MJD", fontsize=11)
    ax.set_ylabel("Magnitude (PSF)", fontsize=11)
    ax.legend(title="Band", fontsize=9)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_title(title or f"Light curve — {oid}", fontsize=12)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_results(
    query_lc: pd.DataFrame,
    results: pd.DataFrame,          # from SimilarityIndex.search()
    result_lcs: Dict[str, pd.DataFrame],
    query_oid: str = "Query",
    n_cols: int = 4,
    figsize_per_panel: Tuple[float, float] = (4.5, 2.8),
    show: bool = False,
    metadata: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    Grid of light-curve panels: query on top, similar objects below.

    Parameters
    ----------
    query_lc : pd.DataFrame
    results  : pd.DataFrame  – output of ``SimilarityIndex.search()``
    result_lcs : dict  – ``{oid: lc_dataframe}``
    """
    n_results = min(len(results), len(result_lcs))
    n_rows = 1 + int(np.ceil(n_results / n_cols))
    total_w = figsize_per_panel[0] * n_cols
    total_h = figsize_per_panel[1] * n_rows

    fig = plt.figure(figsize=(total_w, total_h))
    spec = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.55, wspace=0.35,
    )

    # ── query panel (spans top row) ───────────────────────────────────────────
    ax_query = fig.add_subplot(spec[0, :])
    plot_lightcurve(query_lc, oid=query_oid, ax=ax_query)
    ax_query.set_title(f"QUERY: {query_oid}", fontsize=12, fontweight="bold")

    # ── result panels ─────────────────────────────────────────────────────────
    meta_map: Dict[str, dict] = {}
    if metadata is not None and "oid" in metadata.columns:
        meta_map = metadata.set_index("oid").to_dict("index")

    for i, row in enumerate(results.itertuples(index=False)):
        if i >= n_results:
            break
        r = i // n_cols
        c = i % n_cols
        ax = fig.add_subplot(spec[r + 1, c])

        oid = row.oid
        lc = result_lcs.get(oid)
        if lc is not None:
            plot_lightcurve(lc, oid=oid, ax=ax)

        extra = ""
        m = meta_map.get(oid, {})
        if m.get("cls"):
            extra = f"\n{m['cls']} (p={m.get('probability', 0):.2f})"
        sim = getattr(row, "similarity", None)
        sim_str = f"sim={sim:.3f}" if sim is not None else ""
        ax.set_title(f"#{row.rank} {oid}{extra}\n{sim_str}",
                     fontsize=7.5, pad=2)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Top-{n_results} similar objects for {query_oid}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    if show:
        plt.show()
    return fig


def plot_feature_space(
    oids: List[str],
    features: np.ndarray,
    labels: Optional[List[str]] = None,
    query_oid: Optional[str] = None,
    query_vec: Optional[np.ndarray] = None,
    method: str = "umap",   # "umap" | "pca" | "tsne"
    show: bool = False,
) -> plt.Figure:
    """
    2-D embedding of the feature space with optional query highlight.

    Parameters
    ----------
    method : str
        Dimensionality-reduction method. ``"umap"`` requires the umap-learn
        package; falls back to PCA automatically.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X = np.asarray(features, dtype=float)
    # impute NaN
    col_med = np.nanmedian(X, axis=0)
    for c in range(X.shape[1]):
        mask = ~np.isfinite(X[:, c])
        X[mask, c] = col_med[c] if np.isfinite(col_med[c]) else 0.0

    X = StandardScaler().fit_transform(X)

    # add query if given
    query_idx = None
    if query_vec is not None:
        qv = np.asarray(query_vec, dtype=float)
        for c in range(len(qv)):
            if not np.isfinite(qv[c]):
                qv[c] = col_med[c] if np.isfinite(col_med[c]) else 0.0
        X = np.vstack([X, qv.reshape(1, -1)])
        query_idx = len(X) - 1

    # ── embed ─────────────────────────────────────────────────────────────────
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            Z = reducer.fit_transform(X)
        except ImportError:
            logger.warning("umap-learn not installed, using PCA instead.")
            method = "pca"

    if method == "pca":
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=42).fit_transform(X)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    if labels is not None:
        unique = sorted(set(labels))
        palette = sns.color_palette("tab10", len(unique))
        cmap = {l: palette[i] for i, l in enumerate(unique)}
        Zp = Z[:query_idx] if query_idx is not None else Z
        lp = labels[:query_idx] if query_idx is not None else labels
        for lbl in unique:
            mask = [l == lbl for l in lp]
            ax.scatter(Zp[mask, 0], Zp[mask, 1], s=6, alpha=0.5,
                       color=cmap[lbl], label=lbl, rasterized=True)
        ax.legend(title="Class", fontsize=8, markerscale=2,
                  loc="upper right", bbox_to_anchor=(1.18, 1))
    else:
        ax.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.4, color="steelblue",
                   rasterized=True)

    if query_idx is not None:
        ax.scatter(Z[query_idx, 0], Z[query_idx, 1],
                   s=200, color="gold", edgecolors="black", linewidths=1.5,
                   zorder=5, label=query_oid or "Query")
        ax.legend(fontsize=9)

    ax.set_xlabel(f"{method.upper()} dim 1", fontsize=11)
    ax.set_ylabel(f"{method.upper()} dim 2", fontsize=11)
    ax.set_title("Feature-space embedding", fontsize=13)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_feature_comparison(
    query_vec: np.ndarray,
    result_vecs: np.ndarray,
    feature_names: List[str],
    top_n_features: int = 20,
    show: bool = False,
) -> plt.Figure:
    """Bar chart of the most discriminative features between query and results."""
    diffs = np.abs(query_vec - np.nanmean(result_vecs, axis=0))
    top_idx = np.argsort(diffs)[::-1][:top_n_features]
    top_names = [feature_names[i] for i in top_idx]
    top_diffs = diffs[top_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(top_names)), top_diffs[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("|Query − Mean(results)|", fontsize=10)
    ax.set_title("Most discriminative features", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
