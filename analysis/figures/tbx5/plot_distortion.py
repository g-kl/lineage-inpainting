#!/usr/bin/env python
"""row 3 panel 3: nn distortion over the heart tube map, predicted and measured.

only reference cells carry a value, so the maps are the reference half of panel 2's,
one point per cell. the values are centred on their median over the heart, so 0 is the
middle of the scope rather than a 1:1 distance ratio. the two share one colourbar,
its limits taken at CLIM_Q over both sides at once, so a colour reads the same on
either map.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
from plot_umaps import heart_map

W = S.R3_W[2]


def one(fig, box, u, xl, yl, v, i, tag, lo, hi):
    ax = box(S.DIST_MAP[0] + i * S.DIST_DX, *S.DIST_MAP[1:])
    sc = ax.scatter(u.x, u.y, c=v, s=S.S_DIST, cmap=S.DISTORTION, vmin=lo, vmax=hi,
                    lw=0, rasterized=True)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    S.bare(ax)
    fig.text((S.DIST_MAP[0] + i * S.DIST_DX + S.DIST_MAP[2] / 2) / W,
             S.DIST_TAG_Y / S.ROW3_H, tag, ha="center", va="baseline",
             fontsize=S.FS_TAG, color=S.INK)
    print(f"  {tag}: {np.percentile(v, [100 - S.CLIM_Q, S.CLIM_Q]).round(3)} at "
          f"q{100 - S.CLIM_Q}, q{S.CLIM_Q}", flush=True)
    return sc


def render():
    u, xl, yl = heart_map()
    u = u[u.dataset == "devmap"]
    e = pd.read_csv(f"{S.NR}/distortion_cells.csv", index_col=0).reindex(u.index)
    keep = e[[S.PRED, S.MEAS]].notna().all(1).values
    u, e = u[keep], e[keep]
    order = np.random.default_rng(S.SEED).permutation(len(u))
    u, e = u.iloc[order], e.iloc[order]
    fig, box = S.figure(W, S.ROW3_H)
    both = e[[S.PRED, S.MEAS]].values.ravel()
    lo, hi = (float(np.percentile(both, q)) for q in (100 - S.CLIM_Q, S.CLIM_Q))
    for i, (src, tag) in enumerate([(S.PRED, "predicted"), (S.MEAS, "measured")]):
        sc = one(fig, box, u, xl, yl, e[src].values, i, tag, lo, hi)
    S.hcbar(fig, box(*S.DIST_CBAR), sc, S.DIST_LABEL, ticks=[lo, 0, hi],
            ticklabels=[f"{lo:+.2f}", "0", f"{hi:+.2f}"])
    print(f"  {len(u):,} heart tube reference cells, shared clim {lo:+.3f} to "
          f"{hi:+.3f}", flush=True)
    return S.save(fig, "r3_distortion", (W, S.ROW3_H))


if __name__ == "__main__":
    render()
