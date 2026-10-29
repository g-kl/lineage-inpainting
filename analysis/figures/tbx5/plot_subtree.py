#!/usr/bin/env python
"""row 2 panel 2: one control subtree drawn twice, as a rectangular dendrogram with
time along x and one row per leaf.

the left tree hues its edges by Tbx5, the right one by Tbx5 summed over the nodes
along the trajectory, and the right one marks where each trajectory first crosses TBX5_ON with a
triangle pointing along it. each tree carries its colour bar upright at its right, so
nothing sits under the trees but the time axis and the leaf key. an edge is drawn as a
gradient from the parent's value to the child's and flat across each branch point; its
stroke is log in the number of leaves under it. each leaf carries a dot in the colour
of its fate - the forelimb field or the heart, the only two the pruning leaves.
"""
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S

W = S.R2_W[1]


def load():
    d = pd.read_csv(f"{S.NR}/subtree.csv").set_index("node")
    g = nx.DiGraph()
    g.add_nodes_from(d.index)
    g.add_edges_from((p, n) for n, p in d.parent.items() if isinstance(p, str))
    return d, g


def layout(g):
    """y per node: leaves depth-first, an internal node at its children's mean."""
    root = next(n for n in g if g.in_degree(n) == 0)
    y = {n: float(i) for i, n in enumerate(
        n for n in nx.dfs_preorder_nodes(g, root) if g.out_degree(n) == 0)}
    for n in reversed(list(nx.topological_sort(g))):
        if g.out_degree(n):
            y[n] = np.mean([y[c] for c in g.successors(n)])
    return y


def widths(g):
    """stroke per node, log in the number of leaves under it."""
    k = {}
    for n in reversed(list(nx.topological_sort(g))):
        k[n] = sum(k[c] for c in g.successors(n)) or 1
    lo, hi = S.SUB_LW
    return {n: lo + (hi - lo) * np.log(v) / np.log(max(k.values()))
            for n, v in k.items()}


def segments(g, t, v, y, w):
    """(segment, value, stroke) triples: a gradient along each edge, and a flat
    connector at each branch point."""
    segs, vals, lws = [], [], []
    for p, c in g.edges():
        xs = np.linspace(t[p], t[c], S.SUB_GRAD)
        vs = np.linspace(v[p], v[c], S.SUB_GRAD)
        for i in range(S.SUB_GRAD - 1):
            segs.append([(xs[i], y[c]), (xs[i + 1], y[c])])
            vals.append((vs[i] + vs[i + 1]) / 2)
            lws.append(w[c])
    for n in g:
        ch = [y[c] for c in g.successors(n)]
        if ch:
            segs.append([(t[n], min(ch)), (t[n], max(ch))])
            vals.append(v[n])
            lws.append(w[n])
    return segs, np.array(vals), lws


def tree(fig, box, d, g, y, i, col, cmap, label, onset=False):
    ax = box(S.SUB_BOX[0] + i * S.SUB_DX, *S.SUB_BOX[1:])
    t, v = d.time.to_dict(), d[col].to_dict()
    norm = Normalize(0, np.ceil(d[col].max() * 10) / 10)
    segs, vals, lws = segments(g, t, v, y, widths(g))
    ax.add_collection(LineCollection(segs, array=vals, cmap=cmap, norm=norm, lw=lws))

    leaf = d[d.leaf]
    ax.scatter([t[n] for n in leaf.index], [y[n] for n in leaf.index], s=5.0,
               c=[S.LEAF_GROUP[g] for g in leaf.group], linewidths=0, zorder=3,
               clip_on=False)
    if onset:
        on = d[d.onset.notna()]
        ax.scatter(on.onset, [y[n] for n in on.index], marker=S.ONSET_MARK, s=5,
                   c=S.ONSET, linewidths=0, zorder=4)
    ax.set_xlim(d.time.min() - 0.05, d.time.max() + 0.05)
    ax.set_ylim(-1, max(y.values()) + 1)
    ax.set_xlabel("time (E)", fontsize=S.FS_LABEL, labelpad=1.5)
    ax.set_xticks([8.0, 9.0])
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    cax = box(S.SUB_CBAR[0] + i * S.SUB_DX, *S.SUB_CBAR[1:])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="vertical")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.set_ylabel(label, fontsize=S.FS_KEY, color=S.INK, labelpad=1.5)
    cb.set_ticks([])
    cb.outline.set_visible(False)
    for v, y0, va in ((0, -0.03, "top"), (norm.vmax, 1.03, "bottom")):
        cax.text(0.5, y0, f"{v:g}", transform=cax.transAxes, ha="center", va=va,
                 fontsize=S.FS_KEY, color=S.INK)


def render():
    d, g = load()
    y = layout(g)
    fig, box = S.figure(W, S.ROW2_H)
    tree(fig, box, d, g, y, 0, "tbx5", S.TBX5, f"$\\log\\,${S.TARGET}")
    tree(fig, box, d, g, y, 1, "cum", S.TBX5_CUM,
         f"cumulative $\\log\\,${S.TARGET}", onset=True)

    items = [(g, c) for g, c in S.LEAF_GROUP.items()]
    items.append((f"{S.TARGET} on", S.ONSET, S.ONSET_MARK))
    S.key(box(*S.SUB_KEY), items, ncol=3, ms=2.4, box=True)
    print(f"  {int(d.leaf.sum())} leaves, {len(d)} nodes, "
          f"{int(d.onset.notna().sum())} onsets", flush=True)
    return S.save(fig, "r2_subtree", (W, S.ROW2_H))


if __name__ == "__main__":
    render()
