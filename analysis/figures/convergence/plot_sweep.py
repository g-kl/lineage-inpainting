#!/usr/bin/env python
"""panel g: panel f's two classes swept over the most convergent neuron and
endothelial fates. one normalised stacked bar per fate -- of the genes a fate changes,
what share is erased and what share is memory, rows running most erased at the top --
and a second set of bars at the right carrying the size the normalisation throws away,
the number of divergent genes.

    python analysis/figures/convergence/plot_sweep.py
"""
import os
import sys

import pandas as pd
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

BAR_H = 0.7


def rows(ax, n):
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.tick_params(length=1.5, pad=1)
    S.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)


def render():
    H = pd.read_csv(os.path.join(S.NR, "sweep.csv"))
    H["n_div"] = H.n_memory + H.n_erased
    H["erased"] = H.n_erased / H.n_div
    H = H.sort_values("erased", ascending=False).reset_index(drop=True)
    y = range(len(H))

    fig, box = S.figure(*S.G_SIZE)
    pw, ph = S.G_PLOT
    ax = box(S.G_X, S.G_Y, pw, ph)
    ax.barh(y, H.erased, BAR_H, color=S.CLASS_COL["erased"], lw=0)
    ax.barh(y, 1 - H.erased, BAR_H, left=H.erased, color=S.CLASS_COL["memory"], lw=0)
    ax.set_xlim(0, 1)
    ax.set_xticks((0, 1))
    rows(ax, len(H))
    ax.set_yticklabels([S.G_LABEL.get(f, S.trim(f)) for f in H.fate],
                       fontsize=S.FS_LEGEND)
    ax.set_xlabel("proportion", fontsize=S.FS_LEGEND)
    for cls, x, ha in (("erased", 0.0, "left"), ("memory", 1.0, "right")):
        ax.text(x, -0.9, cls, fontsize=S.FS_ANNOT, color=S.CLASS_COL[cls], ha=ha,
                va="baseline")

    bx = box(S.G_N_X, S.G_Y, S.G_N_W, ph)
    bx.barh(y, H.n_div, BAR_H, color=S.P.MUTED, lw=0)
    bx.set_xlim(0, H.n_div.max() * 1.04)
    bx.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    rows(bx, len(H))
    bx.set_yticklabels([])
    bx.set_xlabel("genes", fontsize=S.FS_LEGEND)
    bx.text(0, -0.9, "divergent", fontsize=S.FS_ANNOT, color=S.P.MUTED,
            ha="left", va="baseline")
    return S.save(fig, "sweep")


if __name__ == "__main__":
    render()
