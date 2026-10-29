"""panel c: the genes each program is built out of, as a gene x group strip per sheet --
the source cell types, then past the rule the destinations. values are the group's
mean log1p expression on one scale shared by both strips, so any row can be compared
with any other. reads numerical_results/genes.csv.

    python -m analysis.figures.supp_conv.plot_genes
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.supp_conv import style as S

CELL = (0.17, 0.135)      # one strip cell, width x height
GENE_W = 0.52             # gene names left of a strip
BLOCK_W = 0.22            # block labels right of a strip
GAP = 0.10
TOP, BOTTOM = 0.04, 0.70  # the column labels hang below
CBAR_W, CBAR_H, CBAR_TICKS = 0.07, 0.62, 0.26


def strip(ax, H, vmax):
    cols = list(dict.fromkeys(H.column))
    G = H.drop_duplicates("gene")
    M = H.pivot_table(index="gene", columns="column", values="mean", sort=False) \
        .reindex(index=G.gene, columns=cols).values
    im = ax.imshow(M, cmap=S.EXPR_CMAP, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([S.C.trim(c) for c in cols], rotation=90, fontsize=S.FS_ROW)
    ax.set_yticks(range(len(G)))
    ax.set_yticklabels(G.gene, fontsize=S.FS_ROW, style="italic")
    ax.tick_params(length=1.5, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)

    edge = np.flatnonzero(G.block.values[1:] != G.block.values[:-1]) + 0.5
    for e in edge:
        ax.axhline(e, color="w", lw=1.4)
    ax.axvline((H.drop_duplicates("column").kind == "source").sum() - 0.5, color="w",
               lw=1.4)
    for lo, hi, b in zip(np.r_[-0.5, edge], np.r_[edge, len(G) - 0.5],
                         dict.fromkeys(G.block)):
        ax.text(len(cols) - 0.3, (lo + hi) / 2, S.BLOCK_LABEL[b], fontsize=S.FS_ROW,
                va="center", ha="left", rotation=270, clip_on=False)
    return im


def render():
    H = pd.read_csv(os.path.join(S.NR, "genes.csv"))
    vmax = np.ceil(H["mean"].max() * 10) / 10          # one scale for both strips
    shape = {k: (h.column.nunique(), h.gene.nunique()) for k, h in H.groupby("sheet")}
    rows = max(n for _, n in shape.values())
    w = sum(GENE_W + c * CELL[0] + BLOCK_W + GAP for c, _ in shape.values())
    size = (w + CBAR_W + CBAR_TICKS, TOP + rows * CELL[1] + BOTTOM)

    fig, box = S.figure(*size)
    x = 0.0
    for key in S.BLOCKS:
        c, n = shape[key]
        x += GENE_W
        im = strip(box(x, size[1] - TOP - n * CELL[1], c * CELL[0], n * CELL[1]),
                   H[H.sheet == key], vmax)
        x += c * CELL[0] + BLOCK_W + GAP
    top = size[1] - TOP
    cax = box(x, top - (rows * CELL[1] + CBAR_H) / 2, CBAR_W, CBAR_H)
    cb = plt.colorbar(im, cax=cax)
    cb.set_ticks(np.arange(0, vmax + 0.01, 1.0))
    cb.outline.set_linewidth(0.4)
    cax.tick_params(length=1.5, pad=1.5, labelsize=S.FS_ROW)
    cax.set_title("log1p", fontsize=S.FS_ROW, pad=3)
    return S.save_svg(fig, "c_genes")


if __name__ == "__main__":
    print(render())
