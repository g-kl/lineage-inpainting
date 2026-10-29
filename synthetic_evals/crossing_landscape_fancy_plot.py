#!/usr/bin/env python
"""Crossing landscape as a single row of per-method heatmaps.

One panel per method (ours / multimarginal FM / GENOT-Sinkhorn / GENOT-FGW), each a
smooth field over the (OT likeness, Felsenstein likeness) percentile plane, coloured
by forward energy error normalised to the per-motif marginal baseline (0 = perfect,
1 = as bad as random draws from the correct pooled marginal, >=1 clipped). All four
share one viridis scale + colorbar, so the methods are directly comparable; the
method-coloured title above each panel gives its identity (the same colours as
crossing_landscape_correlation.py).

Both axes are likeness PERCENTILES within the full candidate set (the space the eval
selects targets in); see crossing_landscape/axis_parameterization.md for the axes.

Pure plotting, no compute (reads what crossing_landscape_eval.py wrote):
    PYTHONPATH=. python synthetic_evals/crossing_landscape_fancy_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

OUT_DIR = Path(__file__).resolve().parent / "crossing_landscape"

# likeness window (mirror the eval / the sibling plot by hand)
OT_FLOOR, FELS_FLOOR, TOP_CLIP = 20, 40, 0

# methods + the crossing palette (matches cell_type_crossing_eval.py)
METHODS = [("ours", "ours"), ("fm", "multimarginal FM"),
           ("gs", "GENOT-Sinkhorn"), ("gf", "GENOT-FGW")]
METHOD_COLORS = {"ours": "#1b7837", "fm": "#d95f02", "gs": "#7570b3", "gf": "#e7298a"}

INK, MUTED = "#222222", "#8a8a8a"
ERR_CMAP = "gist_earth_r"
FS_TITLE, FS_LAB, FS_SUB, FS_TICK = 8.5, 7.5, 5.8, 6.0


def _pctile_mapper(ref):
    s = np.sort(np.asarray(ref, float))
    n = len(s)
    return lambda v: 100.0 * np.searchsorted(s, np.asarray(v, float), side="right") / n


def load():
    land = pd.read_csv(OUT_DIR / "landscape.csv")
    cand = pd.read_csv(OUT_DIR / "candidates.csv")
    ot_pct = _pctile_mapper(cand["ot_coherence"])
    fels_pct = _pctile_mapper(cand["fels_coherence"])
    for df in (cand, land):
        df["ot_q"] = ot_pct(df["ot_coherence"])
        df["fels_q"] = fels_pct(df["fels_coherence"])
    if OT_FLOOR is not None:
        ot_lo, ot_hi = np.percentile(cand["ot_coherence"], [OT_FLOOR, 100 - TOP_CLIP])
        fels_lo, fels_hi = np.percentile(cand["fels_coherence"], [FELS_FLOOR, 100 - TOP_CLIP])
        gate = lambda df: df[df["ot_coherence"].between(ot_lo, ot_hi)
                             & df["fels_coherence"].between(fels_lo, fels_hi)]
        cand, land = gate(cand), gate(land)
    return land, cand


def pivot(land):
    piv = land.pivot_table(index=["cfg", "ot_q", "fels_q"],
                           columns="method", values="energy").reset_index()
    if "marg" in piv.columns:
        marg = piv["marg"].to_numpy()
        return piv, (lambda a: a / marg), 0.0, 1.0
    vmax = float(np.nanmax([piv[m] for m, _ in METHODS if m in piv.columns]))
    return piv, (lambda a: a), 0.0, vmax


def xlabel(ax):
    """Per-panel two-line x label: 'OT likeness' in ink, small grey 'percentile' below."""
    ax.text(0.5, -0.16, "OT likeness", transform=ax.transAxes, ha="center", va="top",
            fontsize=FS_LAB, color=INK)
    ax.text(0.5, -0.25, "percentile", transform=ax.transAxes, ha="center", va="top",
            fontsize=FS_SUB, color=MUTED)


def ylabel(fig, x, y):
    """Shared two-line y label; 'percentile' sits inboard (== 'below' after the 90 rot)."""
    fig.text(x, y, "Felsenstein likeness", ha="center", va="center", rotation=90,
             fontsize=FS_LAB, color=INK)
    fig.text(x + 0.013, y, "percentile", ha="center", va="center", rotation=90,
             fontsize=FS_SUB, color=MUTED)


def main():
    matplotlib.rcParams.update({
        "font.family": "cmss10", "mathtext.fontset": "cm", "axes.unicode_minus": False,
        "svg.fonttype": "path", "figure.dpi": 300, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "lines.solid_capstyle": "round",
    })

    land, cand = load()
    piv, norm, vmin, vmax = pivot(land)
    have = [(m, name) for m, name in METHODS if m in piv.columns]
    x, y = piv["ot_q"].to_numpy(), piv["fels_q"].to_numpy()

    def lims(v, frac=0.04):
        lo, hi = float(v.min()), float(v.max())
        pad = (hi - lo) * frac or 0.02
        return lo - pad, hi + pad
    x0, x1 = lims(cand["ot_q"])
    y0, y1 = lims(cand["fels_q"])

    n = len(have)
    left, right, top, bottom = 0.090, 0.885, 0.86, 0.20
    fig = plt.figure(figsize=(6.5, 2.1))
    gs = fig.add_gridspec(1, n, left=left, right=right, top=top, bottom=bottom,
                          wspace=0.10)
    levels = np.linspace(vmin, vmax, 19)
    for k, (m, name) in enumerate(have):
        ax = fig.add_subplot(gs[0, k])
        z = np.clip(norm(piv[m].to_numpy()), vmin, vmax)
        ax.tricontourf(x, y, z, levels=levels, cmap=ERR_CMAP, extend="max")
        ax.scatter(x, y, s=4, c="white", edgecolors=INK, linewidths=0.3,
                   alpha=0.85, zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([20, 60, 100])
        ax.set_yticks([60, 80, 100])
        ax.tick_params(pad=1.5)
        if k > 0:
            ax.set_yticklabels([])
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.set_title(name, color=METHOD_COLORS[m], fontsize=FS_TITLE, pad=3.5)
        xlabel(ax)                       # repeat the x label under every panel

    # shared y label (two-line, styled) + colorbar
    ylabel(fig, 0.026, (top + bottom) / 2)

    cax = fig.add_axes([0.900, bottom, 0.014, top - bottom])
    cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), ERR_CMAP), cax=cax,
                      extend="max", ticks=[0, 0.5, 1.0])
    cb.ax.set_yticklabels(["0", "0.5", r"$\geq 1$"])
    cb.set_label("normalized energy distance", fontsize=FS_LAB, labelpad=2)
    cb.ax.tick_params(labelsize=FS_TICK, pad=1.5)

    for ext, dpi in [("png", 300), ("svg", None)]:
        out = OUT_DIR / f"crossing_landscape_fancy.{ext}"
        fig.savefig(out, dpi=dpi)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
