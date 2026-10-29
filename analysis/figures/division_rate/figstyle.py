"""Shared, publication-quality plotting for the division-rate figures.

Imported by both ``analysis/pipelines/division_rate/explore.ipynb`` (in-memory
data) and ``replot.py`` (data from CSV) so the notebook and the off-node
regeneration emit identical figures. Each ``fig_*`` returns a Figure; ``save_fig``
writes a vector PDF (editable text, fonttype 42) plus a 300-dpi PNG.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, RED = "#3266a8", "#b2342c"          # fast / slow, positive / negative
GRID = "0.9"


def apply_style():
    """Publication defaults: despined axes, editable vector text."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.titlepad": 10,
        "pdf.fonttype": 42,               # TrueType -> editable in Illustrator
        "ps.fonttype": 42,
    })


def save_fig(fig, figdir, tag, stem):
    os.makedirs(figdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"divrate_{tag}_{stem}.{ext}"),
                    bbox_inches="tight")


def fig_celltypes(tab, which):
    """Horizontal bar of the top/bottom 20 cell types by predicted division rate."""
    sel = tab.head(20) if which == "top" else tab.tail(20)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(sel.index[::-1], sel["div_rate"][::-1],
            color=BLUE if which == "top" else RED, height=0.72)
    ax.set_xlabel("Predicted division rate (doublings per unit time)")
    ax.set_title(f"{which.capitalize()} 20 cell types by predicted division rate")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.01)
    fig.tight_layout()
    return fig


def fig_genes(gene_corr, k=20):
    """Diverging bar of the k most positively / negatively correlated genes."""
    show = pd.concat([gene_corr.head(k), gene_corr.tail(k)])
    colors = [RED if v < 0 else BLUE for v in show.values]
    fig, ax = plt.subplots(figsize=(6.5, 9))
    ax.barh(show.index, show.values, color=colors, height=0.72)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel(r"Spearman $\rho$ with predicted division rate")
    ax.set_title("Genes most correlated with predicted division rate")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.01)
    for lbl in ax.get_yticklabels():       # gene symbols italic (convention)
        lbl.set_style("italic")
    fig.tight_layout()
    return fig


def fig_calibration(val):
    """Predicted vs observed per-cell-type descendant count, with identity line."""
    r = np.corrcoef(val["obs"], val["pred"])[0, 1]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(val["obs"], val["pred"], s=18, alpha=0.7, color=BLUE,
               edgecolor="white", linewidth=0.3)
    hi = max(val["obs"].max(), val["pred"].max()) * 1.05
    ax.plot([0, hi], [0, hi], ls="--", lw=1, color="0.5")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed mean descendant count")
    ax.set_ylabel("Predicted mean descendant count")
    ax.set_title("Per-cell-type calibration")
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va="top")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
