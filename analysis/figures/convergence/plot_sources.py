#!/usr/bin/env python
"""panel d: sources on the umap, each linked to the leaf it reaches, hued by the
positional program instead of by source cell type. each map is cropped to the window
its own sheet fills, and the leaves are hued by destination against a legend.

    python analysis/figures/convergence/plot_sources.py
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import style as S


def umap(ax, cax, L, bg, key, sheet, vmax, win):
    (x0, x1), (y0, y1), _ = win
    c = S.SCORE_CMAP((np.clip(L.x, -vmax, vmax) / (2 * vmax) + 0.5).values)
    m = (bg.umap_x.between(x0, x1) & bg.umap_y.between(y0, y1)).values
    ax.scatter(bg.umap_x[m], bg.umap_y[m], lw=0, rasterized=True, zorder=0, **S.D_BG)
    ax.scatter(L.src_u1, L.src_u2, c=c, lw=0, rasterized=True, zorder=1, **S.D_SRC)
    ax.add_collection(LineCollection(
        np.stack([L[["src_u1", "src_u2"]].values, L[["leaf_u1", "leaf_u2"]].values], 1),
        colors=c, rasterized=True, zorder=2, **S.D_LINK[key]))
    ax.scatter(L.leaf_u1, L.leaf_u2, c=L.panel.map(S.DEST_COL).values, lw=0,
               rasterized=True, zorder=3, **S.D_LEAF)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(sheet["title"], fontsize=S.FS_ANNOT, color=S.INK, pad=2)

    cb = plt.colorbar(matplotlib.cm.ScalarMappable(
        matplotlib.colors.Normalize(-vmax, vmax), S.SCORE_CMAP), cax=cax,
        orientation="horizontal")
    cb.set_label(sheet["xlabel"], fontsize=S.FS_LEGEND, labelpad=1)
    cb.set_ticks([-round(vmax, 1), 0, round(vmax, 1)])
    cb.outline.set_linewidth(0.4)
    cax.tick_params(length=1.5, pad=1, labelsize=S.FS_LEGEND - 0.5)


def legend(ax):
    """the destination key, one entry per program."""
    ax.set_axis_off()
    h = [Line2D([], [], marker="o", ls="none", ms=2.6, mec="none", mfc=S.DEST_COL[k],
                label=S.DEST_LABEL[k]) for k in S.DEST_COL]
    lg = ax.legend(handles=h, title="destination", loc="center left",
                   bbox_to_anchor=(0, 0.5), frameon=False, borderpad=0,
                   borderaxespad=0, handletextpad=0.3, labelspacing=0.5,
                   handlelength=0.8, fontsize=S.FS_LEGEND)
    lg.get_title().set(fontsize=S.FS_LEGEND, color=S.INK)
    lg._legend_box.align = "left"
    for t in lg.get_texts():
        t.set_color(S.INK)


def render():
    L = pd.read_csv(os.path.join(S.NR, "umap_links.csv"))
    v = pd.read_csv(os.path.join(S.NR, "score_norm.csv")).set_index("sheet").vmax
    bg = D.background()

    win = {k: S.umap_window(L[L.sheet == k]) for k in S.SHEETS}
    w = [S.D_UMAP_H * win[k][2] for k in S.SHEETS]
    W = 2 * S.D_MARGIN + sum(w) + len(w) * S.D_GAP + S.D_LEG_W
    fig, box = S.figure(W, S.D_HEIGHT)
    x = S.D_MARGIN
    for uw, (key, sheet) in zip(w, S.SHEETS.items()):
        cw = S.D_CBAR_FRAC * uw
        umap(box(x, S.D_UMAP_Y, uw, S.D_UMAP_H),
             box(x + (uw - cw) / 2, S.D_CBAR_Y, cw, S.D_CBAR_H),
             L[L.sheet == key], bg, key, sheet, v[key], win[key])
        x += uw + S.D_GAP
    legend(box(x, S.D_UMAP_Y, S.D_LEG_W, S.D_UMAP_H))
    return S.save(fig, "sources")


if __name__ == "__main__":
    render()
