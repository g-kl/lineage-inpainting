#!/usr/bin/env python
"""row 3 panel 4: predicted against measured nn distortion, one point per reference
cell.

the scope is the heart tube, the same seven subtypes panels 2 and 3 draw, so the
endocardium is out of it. every heart tube reference cell is a point in its subtype's
colour, with that subtype's mean on top of the cloud; the panel carries no key of its
own and is read with panel 2's. the correlation is over the cells.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S

W = S.R3_W[3]


def render():
    d = pd.read_csv(f"{S.NR}/distortion_cells.csv", index_col=0)
    d = d[d.cell_type.isin(S.HEART_TUBE)].sample(frac=1, random_state=S.SEED)
    a = pd.read_csv(f"{S.NR}/distortion_agreement.csv").set_index("scope")
    m = d.groupby("subtype")[[S.MEAS, S.PRED]].mean()

    fig, box = S.figure(W, S.ROW3_H)
    ax = box(*S.SCATTER_BOX)
    ax.axhline(0, color=S.FAINT, lw=0.5, zorder=1)
    ax.axvline(0, color=S.FAINT, lw=0.5, zorder=1)
    ax.scatter(d[S.MEAS], d[S.PRED], s=S.S_SCATTER,
               c=[S.SUBTYPE[s] for s in d.subtype], lw=0, alpha=0.4, zorder=2,
               rasterized=True)
    c = [S.SUBTYPE[s] for s in m.index]
    ax.scatter(m[S.MEAS], m[S.PRED], s=14, c=c, lw=0.6, ec="white", zorder=4)

    lim = [np.percentile(d[k], [0.5, 99.5]) for k in (S.MEAS, S.PRED)]
    pad = 0.06 * max(hi - lo for lo, hi in lim)
    ax.set_xlim(lim[0][0] - pad, lim[0][1] + pad)
    ax.set_ylim(lim[1][0] - pad, lim[1][1] + pad)
    ax.set_xlabel(f"measured {S.DIST_LABEL}", fontsize=S.FS_LABEL, labelpad=1.5)
    ax.set_ylabel(f"predicted {S.DIST_LABEL}", fontsize=S.FS_LABEL, labelpad=2.0)
    r = a.pearson_cells["heart tube"]
    ax.text(0.97, 0.04, f"$r = {r:.2f}$", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=S.FS_TAG, color=S.INK)
    S.despine(ax)
    print(f"  {len(d):,} heart tube cells over {len(m)} subtypes, r = {r:.3f}",
          flush=True)
    return S.save(fig, "r3_scatter", (W, S.ROW3_H))


if __name__ == "__main__":
    render()
