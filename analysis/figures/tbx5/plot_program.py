#!/usr/bin/env python
"""row 2 panel 4: the posterior SHF program against the nn distortion the model
predicts.

one point per heart tube reference cell: its standardised LDA score along the
posterior SHF axis (fit on the earlier-stage SHF labels, compute_program.py) against
the predicted log nn distortion, over filled and lined contours of the cells' own
density. the cloud is one colour - no subtype is named until row 3, so nothing here is
hued or averaged by one. under the axis, a column each of the N_PROGRAM
transcription factors with the largest loading on that side of the axis, headed by that
side's name.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
import data

W = S.R2_W[3]


def column(fig, x, ha, head, names):
    """a headed column of gene names, placed in inches under the axis."""
    fig.text(x / W, S.PROG_GENE_Y / S.ROW2_H, head, ha=ha, va="baseline",
             fontsize=S.FS_KEY, color=S.GREY)
    for i, name in enumerate(names):
        fig.text(x / W, (S.PROG_GENE_Y - S.PROG_GENE_DY * (i + 1)) / S.ROW2_H, name,
                 ha=ha, va="baseline", fontsize=S.FS_KEY, color=S.INK)


def render():
    import seaborn as sns

    d = pd.read_csv(f"{S.NR}/program_scores.csv", index_col=0)
    g = pd.read_csv(f"{S.NR}/program_genes.csv")
    g = g[g.gene.isin(data.transcription_factors())]
    fit = pd.read_csv(f"{S.NR}/program_fit.csv").iloc[0]

    fig, box = S.figure(W, S.ROW2_H)
    ax = box(*S.PROG_BOX)
    kde = dict(x=d.score, y=d[S.PRED], ax=ax, levels=S.PROG_LEVELS,
               thresh=S.PROG_THRESH, cut=0)
    sns.kdeplot(**kde, fill=True, cmap=S.PROGRAM, zorder=1)
    sns.kdeplot(**kde, fill=False, color=S.PROGRAM_LINE, linewidths=S.PROGRAM_LW,
                alpha=0.85, zorder=2)
    ax.scatter(d.score, d[S.PRED], s=S.S_PROGRAM, c=S.INK, lw=0, alpha=S.PROG_ALPHA,
               zorder=3, rasterized=True)

    lim = [np.percentile(d[k], [0.5, 99.5]) for k in ("score", S.PRED)]
    pad = [0.06 * (hi - lo) for lo, hi in lim]
    ax.set_xlim(lim[0][0] - pad[0], lim[0][1] + pad[0])
    ax.set_ylim(lim[1][0] - pad[1], lim[1][1] + pad[1])
    ax.set_xlabel("pSHF program score", fontsize=S.FS_LABEL, labelpad=1.5)
    ax.set_ylabel(f"predicted\n{S.DIST_LABEL}", fontsize=S.FS_LABEL, labelpad=2.0)
    S.despine(ax)

    x0, x1 = S.PROG_BOX[0], S.PROG_BOX[0] + S.PROG_BOX[2]
    for pole, x, ha in ((False, x0, "left"), (True, x1, "right")):
        names = (g.nsmallest if pole is False else g.nlargest)(S.N_PROGRAM, "loading")
        column(fig, x, ha, S.POLE_NAME[pole], list(names.gene))

    print(f"  program: {int(fit.cells):,} heart tube cells over {int(fit.genes):,} "
          f"genes, auc {fit.auc:.3f}, r = {fit.pearson:.3f}", flush=True)
    return S.save(fig, "r2_program", (W, S.ROW2_H))


if __name__ == "__main__":
    render()
