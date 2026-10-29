#!/usr/bin/env python
"""panel f: per gene, source specificity at E8.5 against leaf divergence at E9.5, one
scatter per fate. right of the vertical cut a gene is source-specific; above the
horizontal one that specificity survives to the leaves. the three share one pair of
limits and one y axis, and panel g names the two classes.

    python analysis/figures/convergence/plot_divergence.py
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import patheffects

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

HALO = [patheffects.withStroke(linewidth=1.4, foreground="white")]


def scatter(ax, h, spec, xhi, yhi, first):
    for cls in ("other", "erased", "memory"):
        g = h[h.cls == cls]
        ax.scatter(g.x, g.y, s=1.5 if cls == "other" else 3.5, lw=0,
                   color=S.CLASS_COL[cls], alpha=0.45 if cls == "other" else 0.9,
                   rasterized=True, zorder=2 if cls == "other" else 3)
    ax.plot([S.DIV_XCUT] * 2, [0, yhi], color=S.GREY, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.plot([S.DIV_XCUT, xhi], [S.DIV_YCUT] * 2, color=S.GREY, lw=0.5, ls=(0, (3, 2)),
            zorder=1)
    ax.set_xlim(0, xhi)
    ax.set_ylim(0, yhi)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([0, 2, 4])

    placed = []      # a label is a wide box, so the cutoff is wider in x than in y
    for cls, key in (("memory", "y"), ("erased", "x")):
        for r in h[h.cls == cls].nlargest(S.F_LABEL, key).itertuples():
            f = (r.x / xhi, r.y / yhi)
            if any(np.hypot((f[0] - a) / 0.26, (f[1] - b) / 0.09) < 1
                   for a, b in placed):
                continue
            placed.append(f)
            side = -1 if f[0] > 0.72 else 1      # keep a name off the right spine
            ax.annotate(r.gene, (r.x, r.y), xytext=(3 * side, 2),
                        textcoords="offset points", ha="left" if side > 0 else "right",
                        fontsize=S.FS_LEGEND - 0.5, color=S.CLASS_COL[cls], style="italic",
                        zorder=5, path_effects=HALO)
    ax.set_title(spec["target"], fontsize=S.FS_ANNOT, color=S.INK, pad=2)
    if first:
        ax.set_ylabel("origin-dependent difference\n(E9.5 leaf $|\\log_2$fc$|$)",
                      fontsize=S.FS_LEGEND)
    else:
        ax.set_yticklabels([])
    S.despine(ax)


def render():
    G = pd.read_csv(os.path.join(S.NR, "gene_divergence.csv"))
    fig, box = S.figure(*S.F_SIZE)
    w, h = S.F_SIZE
    pw, ph = S.F_PLOT
    xhi, yhi = G.x.max() * 1.06, G.y.max() * 1.10      # one pair of limits for the row
    for j, (x, spec) in enumerate(zip(S.F_X, S.PHASE)):
        ax = box(x, S.F_Y, pw, ph)
        scatter(ax, G[G.panel == spec["panel"]], spec, xhi, yhi, j == 0)
    mid = (S.F_X[0] + S.F_X[-1] + pw) / 2
    fig.text(mid / w, 0.10 / h, "difference at origin (E8.5 source $|\\log_2$fc$|$)",
             ha="center", va="baseline", fontsize=S.FS_LEGEND, color=S.INK)
    return S.save(fig, "divergence")


if __name__ == "__main__":
    render()
