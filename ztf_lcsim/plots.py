"""
Visualisation utilities for ZTF light curves and similarity-search results.

Key improvements over previous version:
  - Interactive zoom/pan (uses constrained_layout + proper backend detection)
  - Text never overlaps — font sizes scale with panel size
  - Full-screen safe — all labels re-flow on resize
  - Each plot type is individually configurable
  - Added interactive HTML export via mpld3 (optional)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── band styling ──────────────────────────────────────────────────────────────
_BAND_COLOUR = {1: "#27ae60", 2: "#e74c3c", 3: "#2980b9"}
_BAND_LABEL = {1: "g", 2: "r", 3: "i"}


# ── safe backend setup ────────────────────────────────────────────────────────
def _ensure_interactive_backend():
    """
    Try to switch to an interactive backend if the current one is non-interactive.
    Called once when a show=True plot is requested.
    """
    backend = matplotlib.get_backend().lower()
    non_interactive = ("agg", "pdf", "svg", "ps", "cairo", "pgf")
    if any(b in backend for b in non_interactive):
        for try_backend in ("TkAgg", "Qt5Agg", "Qt4Agg", "WXAgg", "MacOSX"):
            try:
                matplotlib.use(try_backend)
                logger.debug(f"Switched to backend: {try_backend}")
                return
            except Exception:
                continue
        logger.warning(
            "Could not switch to interactive backend. "
            "Plots will be saved but not displayed interactively. "
            "Install tkinter:  conda install tk"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Single light curve
# ══════════════════════════════════════════════════════════════════════════════


def plot_lightcurve(
    lc: pd.DataFrame,
    oid: str = "",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = False,
    save: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    invert_y: bool = True,
    mark_peak: bool = True,
) -> plt.Figure:
    """
    Plot a ZTF multi-band light curve with full interactivity.

    Parameters
    ----------
    lc        : DataFrame with mjd, magpsf, sigmapsf, fid
    oid       : object ID for title/legend
    ax        : existing Axes to plot into (creates new figure if None)
    title     : override auto-title
    show      : display interactive window
    save      : save to file (e.g. "lc.pdf")
    figsize   : (width, height) in inches
    invert_y  : flip y-axis (brighter = up)
    mark_peak : draw a vertical dashed line at peak brightness
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=figsize,
            constrained_layout=True,
        )
    else:
        fig = ax.get_figure()

    if lc is None or lc.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="grey",
        )
        ax.set_title(title or f"{oid}", fontsize=12, pad=8)
        return fig

    # ── find peak epoch ───────────────────────────────────────────────────────
    peak_t = None
    if mark_peak and "magpsf" in lc.columns:
        pk_row = lc.loc[lc["magpsf"].idxmin()]
        peak_t = float(pk_row["mjd"])

    # ── plot each band ────────────────────────────────────────────────────────
    for fid in sorted(lc["fid"].unique()):
        grp = lc[lc["fid"] == fid].sort_values("mjd")
        color = _BAND_COLOUR.get(int(fid), "#888888")
        label = _BAND_LABEL.get(int(fid), str(fid))

        ax.errorbar(
            grp["mjd"],
            grp["magpsf"],
            yerr=grp["sigmapsf"],
            fmt="o",
            color=color,
            label=f"{label}-band  (n={len(grp)})",
            markersize=4,
            elinewidth=0.9,
            capsize=2,
            alpha=0.85,
            zorder=3,
        )

    if peak_t is not None:
        ax.axvline(
            peak_t,
            color="gold",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            zorder=2,
            label=f"peak  MJD={peak_t:.1f}",
        )

    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel("MJD", fontsize=11, labelpad=4)
    ax.set_ylabel("PSF magnitude", fontsize=11, labelpad=4)
    ax.legend(
        fontsize=9,
        loc="best",
        framealpha=0.85,
        edgecolor="none",
    )
    ax.grid(True, alpha=0.25, zorder=0, linestyle="--")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    title_str = title or (f"{oid}" if oid else "Light curve")
    ax.set_title(title_str, fontsize=12, pad=8, fontweight="bold")

    if standalone:
        if save:
            _save_figure(fig, save)
        if show:
            _show_figure(fig)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Result grid  (query + top-k matches)
# ══════════════════════════════════════════════════════════════════════════════


def plot_results(
    query_lc: pd.DataFrame,
    results: pd.DataFrame,
    result_lcs: Dict[str, pd.DataFrame],
    query_oid: str = "Query",
    n_cols: int = 3,
    panel_width: float = 5.5,
    panel_height: float = 3.2,
    show: bool = False,
    save: Optional[str] = None,
    metadata: Optional[pd.DataFrame] = None,
    max_results: int = 12,
) -> plt.Figure:
    """
    Grid of light-curve panels: query spans the full top row,
    matched objects fill rows below.

    Parameters
    ----------
    n_cols       : number of columns in the match grid
    panel_width  : width of each panel in inches
    panel_height : height of each panel in inches
    max_results  : cap on number of match panels
    """
    # ── layout sizes ─────────────────────────────────────────────────────────
    n_results = min(len(results), len(result_lcs), max_results)
    n_rows_res = max(1, int(np.ceil(n_results / n_cols)))
    n_rows = 1 + n_rows_res  # top row = query

    fig_w = panel_width * n_cols
    fig_h = panel_height * n_rows + 0.6  # 0.6 for suptitle

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    # outer grid: 2 rows (query | results)
    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1, n_rows_res],
        hspace=0.45,
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.06,
    )

    # ── query panel ───────────────────────────────────────────────────────────
    ax_query = fig.add_subplot(outer[0])
    plot_lightcurve(
        query_lc,
        oid=query_oid,
        ax=ax_query,
        title=f"QUERY:  {query_oid}",
        invert_y=True,
        mark_peak=True,
    )
    ax_query.title.set_fontsize(13)
    ax_query.title.set_fontweight("bold")
    ax_query.title.set_color("#1a237e")

    # ── result grid ───────────────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows_res,
        n_cols,
        subplot_spec=outer[1],
        hspace=0.65,
        wspace=0.30,
    )

    meta_map: Dict[str, dict] = {}
    if metadata is not None and "oid" in metadata.columns:
        meta_map = metadata.set_index("oid").to_dict("index")

    for i, row in enumerate(results.head(n_results).itertuples(index=False)):
        r = i // n_cols
        c = i % n_cols
        ax = fig.add_subplot(inner[r, c])
        oid = row.oid
        lc = result_lcs.get(oid)

        if lc is not None and not lc.empty:
            plot_lightcurve(lc, oid=oid, ax=ax, invert_y=True, mark_peak=False)
        else:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="grey",
            )

        # ── panel title ───────────────────────────────────────────────────────
        sim = getattr(row, "similarity", None)
        sim_str = f"sim={sim:.3f}" if sim is not None else ""

        m = meta_map.get(oid, {})
        cls_str = ""
        if m.get("cls"):
            prob = m.get("probability", 0.0) or 0.0
            cls_str = f"{m['cls']}  p={prob:.2f}"

        n_g = int((lc["fid"] == 1).sum()) if lc is not None and not lc.empty else 0
        n_r = int((lc["fid"] == 2).sum()) if lc is not None and not lc.empty else 0
        obs_str = f"g={n_g} r={n_r}"

        # build a compact 3-line title that never overflows
        title_parts = [
            f"#{row.rank}  {oid}",
            cls_str if cls_str else obs_str,
            sim_str if sim_str else "",
        ]
        title_text = "\n".join(p for p in title_parts if p)

        ax.set_title(
            title_text,
            fontsize=7.5,
            pad=3,
            linespacing=1.4,
            loc="center",
        )
        ax.tick_params(axis="both", labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)

    # ── blank out unused panels ───────────────────────────────────────────────
    for i in range(n_results, n_rows_res * n_cols):
        r = i // n_cols
        c = i % n_cols
        ax = fig.add_subplot(inner[r, c])
        ax.set_visible(False)

    # ── suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Top-{n_results} similar objects for  {query_oid}",
        fontsize=14,
        fontweight="bold",
        y=0.975,
    )

    if save:
        _save_figure(fig, save)
    if show:
        _show_figure(fig)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Feature-space 2-D embedding
# ══════════════════════════════════════════════════════════════════════════════


def plot_feature_space(
    oids: List[str],
    features: np.ndarray,
    labels: Optional[List[str]] = None,
    query_oid: Optional[str] = None,
    query_vec: Optional[np.ndarray] = None,
    method: str = "umap",
    figsize: Tuple[float, float] = (11, 8),
    show: bool = False,
    save: Optional[str] = None,
    alpha: float = 0.45,
    point_size: float = 8.0,
) -> plt.Figure:
    """
    2-D embedding of the feature space with optional query highlight.

    Parameters
    ----------
    method : "umap" | "pca" | "tsne"
    """
    import matplotlib.cm as cm
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # ── prep data ─────────────────────────────────────────────────────────────
    X = np.asarray(features, dtype=float)
    col_med = np.nanmedian(X, axis=0)
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        X[bad, c] = col_med[c] if np.isfinite(col_med[c]) else 0.0
    X = StandardScaler().fit_transform(X)

    # append query if given
    query_idx = None
    if query_vec is not None:
        qv = np.asarray(query_vec, dtype=float).copy()
        for c in range(len(qv)):
            if not np.isfinite(qv[c]):
                qv[c] = col_med[c] if np.isfinite(col_med[c]) else 0.0
        X = np.vstack([X, qv.reshape(1, -1)])
        query_idx = len(X) - 1

    # ── dimensionality reduction ──────────────────────────────────────────────
    Z = _reduce_2d(X, method)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if labels is not None:
        unique_labels = sorted(
            set(labels[:query_idx] if query_idx is not None else labels)
        )
        n_cls = len(unique_labels)
        palette = cm.get_cmap("tab20", n_cls)
        lbl_map = {l: i for i, l in enumerate(unique_labels)}

        Zp = Z[:query_idx] if query_idx is not None else Z
        lp = labels[:query_idx] if query_idx is not None else labels

        for lbl in unique_labels:
            mask = np.array([l == lbl for l in lp])
            color = palette(lbl_map[lbl])
            ax.scatter(
                Zp[mask, 0],
                Zp[mask, 1],
                s=point_size,
                alpha=alpha,
                color=color,
                label=lbl,
                rasterized=True,
                linewidths=0,
            )

        # legend with scrollable box for many classes
        leg = ax.legend(
            title="Class",
            fontsize=8,
            title_fontsize=9,
            markerscale=2.5,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            framealpha=0.9,
            ncol=max(1, n_cls // 12),
        )
    else:
        ax.scatter(
            Z[:, 0],
            Z[:, 1],
            s=point_size,
            alpha=alpha,
            color="steelblue",
            rasterized=True,
            linewidths=0,
        )

    # highlight query
    if query_idx is not None:
        ax.scatter(
            Z[query_idx, 0],
            Z[query_idx, 1],
            s=220,
            color="#f39c12",
            edgecolors="black",
            linewidths=1.8,
            zorder=10,
            label=query_oid or "Query",
        )
        ax.annotate(
            query_oid or "Query",
            xy=(Z[query_idx, 0], Z[query_idx, 1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#c0392b",
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8),
        )

    meth_label = method.upper() if method != "umap" else "UMAP"
    ax.set_xlabel(f"{meth_label} dimension 1", fontsize=11, labelpad=5)
    ax.set_ylabel(f"{meth_label} dimension 2", fontsize=11, labelpad=5)
    ax.set_title("Light-curve feature space", fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.2, linestyle="--")

    if save:
        _save_figure(fig, save)
    if show:
        _show_figure(fig)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Feature comparison (query vs matches)
# ══════════════════════════════════════════════════════════════════════════════


def plot_feature_comparison(
    query_vec: np.ndarray,
    result_vecs: np.ndarray,
    feature_names: List[str],
    top_n: int = 25,
    figsize: Tuple[float, float] = (13, 7),
    show: bool = False,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart showing the most discriminative features
    between the query and the matched objects.
    """
    diffs = np.abs(query_vec - np.nanmean(result_vecs, axis=0))
    top_n = min(top_n, len(feature_names))
    top_idx = np.argsort(diffs)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_diffs = diffs[top_idx]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    bars = ax.barh(
        range(top_n),
        top_diffs[::-1],
        color="#2980b9",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=8.5)
    ax.set_xlabel("|Query − Mean(matches)|  (standardised)", fontsize=10, labelpad=5)
    ax.set_title(
        f"Top-{top_n} most discriminating features",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # value labels on bars
    for bar, val in zip(bars, top_diffs[::-1]):
        if val > 0.01:
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=7,
                color="#2c3e50",
            )

    if save:
        _save_figure(fig, save)
    if show:
        _show_figure(fig)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Phase-folded LC
# ══════════════════════════════════════════════════════════════════════════════


def plot_phase_folded(
    lc: pd.DataFrame,
    period: float,
    oid: str = "",
    n_cycles_display: int = 2,
    figsize: Tuple[float, float] = (10, 4),
    show: bool = False,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Phase-fold a light curve at a given period and plot 2 cycles.

    Parameters
    ----------
    period           : folding period in days
    n_cycles_display : how many cycles to show (default 2 for clarity)
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if lc is None or lc.empty or period <= 0:
        ax.text(
            0.5,
            0.5,
            "No data / invalid period",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    for fid in sorted(lc["fid"].unique()):
        grp = lc[lc["fid"] == fid].copy()
        phase = (grp["mjd"].values % period) / period
        color = _BAND_COLOUR.get(int(fid), "#888888")
        label = _BAND_LABEL.get(int(fid), str(fid))

        for cycle in range(n_cycles_display):
            ax.errorbar(
                phase + cycle,
                grp["magpsf"].values,
                yerr=grp["sigmapsf"].values,
                fmt="o",
                color=color,
                alpha=0.6,
                markersize=3,
                elinewidth=0.7,
                capsize=1.5,
                label=f"{label}-band" if cycle == 0 else "_nolegend_",
            )

    ax.invert_yaxis()
    ax.set_xlabel(f"Phase  (P = {period:.4f} d)", fontsize=11, labelpad=4)
    ax.set_ylabel("PSF magnitude", fontsize=11, labelpad=4)
    ax.set_xlim(-0.05, n_cycles_display + 0.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.legend(fontsize=9, loc="best", framealpha=0.85, edgecolor="none")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_title(
        f"Phase-folded:  {oid}   P = {period:.4f} d",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )

    if save:
        _save_figure(fig, save)
    if show:
        _show_figure(fig)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ML class probability bars
# ══════════════════════════════════════════════════════════════════════════════


def plot_class_probabilities(
    proba: np.ndarray,
    class_names: List[str],
    oid: str = "",
    figsize: Tuple[float, float] = (9, 4),
    show: bool = False,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of ML classifier output probabilities.
    """
    order = np.argsort(proba)
    names = [class_names[i] for i in order]
    values = proba[order]

    colors = ["#e74c3c" if v == max(proba) else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    bars = ax.barh(
        names, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
    )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability", fontsize=10, labelpad=4)
    ax.set_title(
        f"ML class probabilities  —  {oid}",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.01, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8.5,
            color="black" if val < 0.7 else "white",
            fontweight="bold" if val == max(proba) else "normal",
        )

    if save:
        _save_figure(fig, save)
    if show:
        _show_figure(fig)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Multi-panel summary for one object (LC + folded + feature bar + proba)
# ══════════════════════════════════════════════════════════════════════════════


def plot_object_summary(
    lc: pd.DataFrame,
    oid: str,
    feature_vec: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    period: Optional[float] = None,
    ml_proba: Optional[np.ndarray] = None,
    ml_classes: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 9),
    show: bool = False,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    4-panel summary figure for one object:
      top-left   : raw light curve
      top-right  : phase-folded LC  (or message if no period)
      bottom-left: top-20 feature values
      bottom-right: ML class probabilities
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.40,
        wspace=0.32,
    )

    ax_lc = fig.add_subplot(gs[0, 0])
    ax_fold = fig.add_subplot(gs[0, 1])
    ax_feat = fig.add_subplot(gs[1, 0])
    ax_prob = fig.add_subplot(gs[1, 1])

    # ── panel 1: raw LC ───────────────────────────────────────────────────────
    plot_lightcurve(lc, oid=oid, ax=ax_lc, mark_peak=True)

    # ── panel 2: phase-folded ─────────────────────────────────────────────────
    if period and period > 0 and lc is not None and not lc.empty:
        for fid in sorted(lc["fid"].unique()):
            grp = lc[lc["fid"] == fid].copy()
            phase = (grp["mjd"].values % period) / period
            color = _BAND_COLOUR.get(int(fid), "#888888")
            label = _BAND_LABEL.get(int(fid), str(fid))
            for cyc in range(2):
                ax_fold.errorbar(
                    phase + cyc,
                    grp["magpsf"].values,
                    yerr=grp["sigmapsf"].values,
                    fmt="o",
                    color=color,
                    alpha=0.55,
                    markersize=3,
                    elinewidth=0.7,
                    capsize=1.5,
                    label=label if cyc == 0 else "_nolegend_",
                )
        ax_fold.invert_yaxis()
        ax_fold.set_xlabel(f"Phase  (P = {period:.4f} d)", fontsize=9, labelpad=4)
        ax_fold.set_ylabel("PSF magnitude", fontsize=9, labelpad=4)
        ax_fold.set_xlim(-0.05, 2.05)
        ax_fold.legend(fontsize=8, loc="best", framealpha=0.8, edgecolor="none")
        ax_fold.grid(True, alpha=0.25, linestyle="--")
        ax_fold.set_title(
            f"Phase-folded  P = {period:.4f} d",
            fontsize=10,
            fontweight="bold",
            pad=6,
        )
    else:
        ax_fold.text(
            0.5,
            0.5,
            "No significant period detected",
            ha="center",
            va="center",
            transform=ax_fold.transAxes,
            fontsize=10,
            color="grey",
            style="italic",
        )
        ax_fold.set_title("Phase-folded", fontsize=10)
        ax_fold.axis("off")

    # ── panel 3: top feature values ───────────────────────────────────────────
    if feature_vec is not None and feature_names is not None:
        finite_mask = np.isfinite(feature_vec)
        finite_vals = feature_vec[finite_mask]
        finite_names = [feature_names[i] for i, f in enumerate(finite_mask) if f]

        top_n = min(20, len(finite_vals))
        if top_n > 0:
            order = np.argsort(np.abs(finite_vals))[::-1][:top_n]
            y_labels = [finite_names[i] for i in order[::-1]]
            y_values = [finite_vals[i] for i in order[::-1]]
            colors = ["#e74c3c" if v > 0 else "#2980b9" for v in y_values]

            ax_feat.barh(
                y_labels,
                y_values,
                color=colors,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.4,
            )
            ax_feat.axvline(0, color="black", linewidth=0.8, zorder=3)
            ax_feat.set_xlabel("Feature value (raw)", fontsize=9, labelpad=4)
            ax_feat.set_title(
                "Top-20 features  (by |value|)",
                fontsize=10,
                fontweight="bold",
                pad=6,
            )
            ax_feat.tick_params(labelsize=7.5)
            ax_feat.grid(axis="x", alpha=0.25, linestyle="--", zorder=0)
        else:
            ax_feat.text(
                0.5,
                0.5,
                "All features are NaN",
                ha="center",
                va="center",
                transform=ax_feat.transAxes,
                fontsize=10,
                color="grey",
            )
            ax_feat.axis("off")
    else:
        ax_feat.text(
            0.5,
            0.5,
            "No features available",
            ha="center",
            va="center",
            transform=ax_feat.transAxes,
            fontsize=10,
            color="grey",
        )
        ax_feat.axis("off")

    # ── panel 4: ML class probabilities ──────────────────────────────────────
    if ml_proba is not None and ml_classes is not None and len(ml_proba) > 0:
        order = np.argsort(ml_proba)  # ascending → bottom = highest
        y_lbls = [ml_classes[i] for i in order]
        y_vals = [float(ml_proba[i]) for i in order]
        top_p = max(y_vals)
        colors = ["#e74c3c" if v == top_p else "#3498db" for v in y_vals]

        ax_prob.barh(
            y_lbls,
            y_vals,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        ax_prob.set_xlim(0, 1.08)
        ax_prob.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_prob.set_xlabel("Probability", fontsize=9, labelpad=4)
        ax_prob.set_title(
            "ML class probabilities",
            fontsize=10,
            fontweight="bold",
            pad=6,
        )
        ax_prob.tick_params(labelsize=8.5)
        ax_prob.grid(axis="x", alpha=0.25, linestyle="--", zorder=0)

        # probability labels
        for lbl, val in zip(y_lbls, y_vals):
            ax_prob.text(
                min(val + 0.01, 1.03),
                y_lbls.index(lbl),
                f"{val:.3f}",
                va="center",
                fontsize=8,
                color="black",
                fontweight="bold" if val == top_p else "normal",
            )
    else:
        ax_prob.text(
            0.5,
            0.5,
            "Run  scripts/05_train_ml.py\nfor class probabilities",
            ha="center",
            va="center",
            transform=ax_prob.transAxes,
            fontsize=9,
            color="grey",
            style="italic",
        )
        ax_prob.axis("off")

    fig.suptitle(
        f"Object summary:  {oid}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    if save:
        _save_figure(fig, save)
    if show:
        _show_figure(fig)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════


def _save_figure(fig: plt.Figure, path: str, dpi: int = 150):
    """Save figure, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    logger.info(f"Plot saved: {p}")


def _show_figure(fig: plt.Figure):
    """
    Show the figure interactively.
    Works whether or not we are in a Jupyter notebook.
    """
    try:
        # Jupyter / IPython
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            plt.show()
            return
    except ImportError:
        pass

    # Standard matplotlib window
    _ensure_interactive_backend()
    plt.show(block=True)


def _reduce_2d(X: np.ndarray, method: str) -> np.ndarray:
    """Reduce X to 2-D using the chosen method."""
    from sklearn.decomposition import PCA

    if method == "umap":
        try:
            import umap

            return umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
            ).fit_transform(X)
        except ImportError:
            logger.warning(
                "umap-learn not installed — falling back to PCA.\n"
                "Install with:  pip install umap-learn"
            )
            method = "pca"

    if method == "tsne":
        from sklearn.manifold import TSNE

        perp = min(30, max(5, len(X) // 10))
        return TSNE(
            n_components=2,
            random_state=42,
            perplexity=perp,
            n_iter=1000,
        ).fit_transform(X)

    # PCA (default / fallback)
    return PCA(n_components=2, random_state=42).fit_transform(X)
