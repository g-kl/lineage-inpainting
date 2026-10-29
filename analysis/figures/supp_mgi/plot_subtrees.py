"""panel c: one control subtree per example, a rectangular dendrogram with time along x
and one row per readout leaf. reads numerical_results/subtrees.csv.

edges are hued by the gene's log1p CP10K in each node's snapped cell, on one scale per
tree: a gradient from the parent's value to the child's along an edge, flat across each
branch point. the stroke is log in the leaves under the edge. a leaf is dotted by its cell
type, one of the three examples' or any other.

    python -m analysis.figures.supp_mgi.plot_subtrees
"""
import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from analysis.figures.supp_mgi import style as S

LEAF_S = 7   # leaf marker area, pt^2


def graph(d):
    g = nx.DiGraph()
    g.add_nodes_from(d.index)
    g.add_edges_from((p, n) for n, p in d["parent"].items() if isinstance(p, str))
    return g


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
    return {n: lo + (hi - lo) * np.log(v) / np.log(max(k.values())) for n, v in k.items()}


def segments(g, t, v, y, w):
    """(segment, value, stroke) lists: a gradient along each edge, and a flat connector
    at each branch point."""
    segs, vals, lws = [], [], []
    for p, c in g.edges():
        xs, vs = np.linspace(t[p], t[c], S.SUB_GRAD), np.linspace(v[p], v[c], S.SUB_GRAD)
        segs += [[(xs[i], y[c]), (xs[i + 1], y[c])] for i in range(S.SUB_GRAD - 1)]
        vals += list((vs[:-1] + vs[1:]) / 2)
        lws += [w[c]] * (S.SUB_GRAD - 1)
    for n in g:
        ch = [y[c] for c in g.successors(n)]
        if ch:
            segs.append([(t[n], min(ch)), (t[n], max(ch))])
            vals.append(v[n])
            lws.append(w[n])
    return segs, np.array(vals), lws


def tree(ax, d, title, norm, xlim):
    g = graph(d)
    y = layout(g)
    t, v = d["time"].to_dict(), d["expression"].to_dict()
    segs, vals, lws = segments(g, t, v, y, widths(g))
    ax.add_collection(LineCollection(segs, array=vals, cmap=S.EXPR_CMAP, norm=norm,
                                     lw=lws, capstyle="projecting", zorder=1))
    leaf = d[d["leaf"]]
    ax.scatter(leaf["time"], [y[n] for n in leaf.index], s=LEAF_S, lw=0, zorder=3,
               c=[S.LEAF_COLORS.get(ct, S.OTHER_LEAF) for ct in leaf["node_type"]],
               clip_on=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(max(y.values()) + 0.6, -0.6)
    ax.set_xticks([7, 8, 9], ["E7", "E8", "E9"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(0, 1.03, title, transform=ax.transAxes, ha="left", va="bottom")


def render():
    d = pd.read_csv(os.path.join(S.NR, "subtrees.csv"))
    xlim = (np.floor(d["time"].min() * 10) / 10 - 0.05, S.RING_TARGET_TIME + 0.15)

    fig, box = S.figure(S.PAGE_W, S.SUBTREE_H)
    h = S.SUBTREE_H - S.SUB_BOTTOM - S.SUB_TOP
    for i, (gene, ct) in enumerate(S.SUBTREE_EXAMPLES):
        g = d[d["gene"] == gene].set_index("node")
        norm = Normalize(0, np.ceil(g["expression"].max() * 10) / 10)
        x0 = S.SUB_LEFT + i * S.SUB_PITCH
        tree(box(x0, S.SUB_BOTTOM, S.SUB_W, h), g, f"{gene}, {ct}", norm, xlim)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=S.EXPR_CMAP),
                          cax=box(x0 + S.SUB_W + S.SUB_CBAR[0], S.SUB_BOTTOM, S.SUB_CBAR[1], h))
        cb.outline.set_visible(False)
        cb.set_ticks([0, norm.vmax], labels=["0", f"{norm.vmax:g}"])
    cb.set_label("expression, log1p CP10K")   # the last bar names all three

    handles = [Line2D([], [], ls="none", marker="o", ms=4, mew=0, color=c, label=ct)
               for ct, c in [*S.LEAF_COLORS.items(), ("other", S.OTHER_LEAF)]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=0.8, handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0.1)
    return S.save_svg(fig, "c_subtrees")


if __name__ == "__main__":
    print(render())
