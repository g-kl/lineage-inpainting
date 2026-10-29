#!/usr/bin/env python
"""row 3 panels 1 and 2: the two maps.

1  the integrated latent over both datasets. the reference goes down first as a grey
   cloud and the crispants sit on it hued by lineage, so the panel is where the
   crispants land on the reference. the draw is balanced, so the count under it is of
   the whole fit rather than of the points drawn.
2  the heart tube, both datasets on one umap of the same latent, hued by subtype. its
   window and its key are the ones panels 3 and 4 are read with.

both keys sit under their map, so the map has the height of the panel and the panel
letter has its corner.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S


def heart_map():
    """the heart tube umap and the window panels 2 and 3 share."""
    u = pd.read_parquet(f"{S.NR}/umap_heart.parquet")
    xl, yl = S.window(u.x.values, u.y.values, S.MAP_HEART)
    return u[u.x.between(*xl) & u.y.between(*yl)], xl, yl


def counts():
    return pd.read_csv(f"{S.NR}/cell_counts.csv", index_col=0)


def render_all():
    v = pd.read_parquet(f"{S.NR}/umap_all.parquet").sample(frac=1, random_state=S.SEED)
    w = S.R3_W[0]
    t = (v.dataset == S.TARGET).values
    lins = [l for l in S.LINEAGE_COLORS if (v.lineage[t] == l).any()]
    col = np.array([S.LINEAGE_COLORS.get(l, S.FAINT) for l in v.lineage])
    xl, yl = S.window(v.x.values, v.y.values, S.MAP_ALL)

    fig, box = S.figure(w, S.ROW3_H)
    ax = box(*S.MAP_ALL)
    # the backdrop is drawn a shade larger, so it reads as a halo where the crispants
    # cover it rather than disappearing under them
    ax.scatter(v.x[~t], v.y[~t], s=0.45, color=S.BACKDROP, lw=0, rasterized=True,
               zorder=1)
    ax.scatter(v.x[t], v.y[t], s=0.25, c=col[t], lw=0, rasterized=True, zorder=2)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    S.bare(ax)

    n = counts().cells[S.TARGET]
    S.caption(fig, w, S.ROW3_H, f"{S.TARGET} crispant   $n = {n:,}$", S.COUNT_ALL_Y)
    S.key(box(*S.KEY_ALL), [("reference", S.BACKDROP)]
          + [(S.short(l), S.LINEAGE_COLORS[l]) for l in lins], ncol=3, ms=2.0,
          colgap=0.05, pad=0.035, box=True)
    return S.save(fig, "r3_umap_all", (w, S.ROW3_H))


def render_heart():
    u, xl, yl = heart_map()
    u = u.sample(frac=1, random_state=S.SEED)
    w = S.R3_W[1]
    t = (u.dataset == S.TARGET).values
    col = np.array([S.SUBTYPE.get(s, S.FAINT) for s in u.subtype])

    fig, box = S.figure(w, S.ROW3_H)
    ax = box(*S.MAP_HEART)
    ax.scatter(u.x, u.y, s=np.where(t, S.S_CRI, S.S_REF), c=col, lw=0, rasterized=True)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    S.bare(ax)

    n = counts().heart_tube
    for y, (name, k) in zip(S.COUNT_Y, [("E9.5 reference", "devmap"),
                                        (f"{S.TARGET} crispant", S.TARGET)]):
        S.caption(fig, w, S.ROW3_H, f"{name}   $n = {n[k]:,}$", y)
    order = [s for s in S.SUBTYPE if (u.subtype == s).any()]
    S.key(box(*S.KEY_HEART), [(S.short(s), S.SUBTYPE[s]) for s in order], ncol=2,
          ms=2.0, box=True)
    print(f"  heart tube: {len(u):,} cells over {len(order)} subtypes", flush=True)
    return S.save(fig, "r3_umap_heart", (w, S.ROW3_H))


def render():
    return render_all(), render_heart()


if __name__ == "__main__":
    render()
