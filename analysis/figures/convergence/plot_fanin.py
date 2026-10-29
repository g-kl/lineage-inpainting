"""fan-in plots: where a chosen E9.5 fate comes from (reads only numerical_results/).

per target in style.FANIN, three pieces side by side:
  - UMAP zoom on the E9.5 focus leaves + their inferred ancestors, each leaf
    linked to its ancestor and coloured by source group (focus leaves in black)
  - source-group composition bar chart (kept groups + "other")
  - example subtrees each dominated by one source group, focus leaf in black

writes imgs/fanin_{key}.{svg,png}.

    python analysis/figures/convergence/plot_fanin.py [key ...]
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
import style as S

ROW_H_IN = 2.15
COMPACT_W_IN = 4.95        # compact layout: narrower fan-in (shrunk subtrees)
                           # leaving row-2 room for the conv-vs-OT scatter on the
                           # far left
COMPACT_H_IN = 1.60


def _trunc(s, n=17):
    return s if len(s) <= n else s[:n - 1] + "."


def _umap(ax, link, bg, comp, title):
    """full UMAP (matching row 1), leaves linked to their inferred ancestors."""
    color = dict(zip(comp["src_group"], comp["color"]))
    # slightly darker backdrop than the row-1 maps: here it is exposed (only the
    # sparse fan-in links sit on top), so it needs the visual weight to read.
    ax.scatter(bg["umap_x"], bg["umap_y"], s=0.6, c="#d7d7dd", lw=0,
               rasterized=True, zorder=0)
    segs, cols = [], []
    for _, r in link.iterrows():
        segs.append([(r["anc_x"], r["anc_y"]), (r["leaf_x"], r["leaf_y"])])
        cols.append(color.get(r["src_group"], S.GREY))
    ax.add_collection(LineCollection(segs, colors=cols, lw=0.2, alpha=0.13,
                                     rasterized=True, zorder=1))
    ax.scatter(link["anc_x"], link["anc_y"], s=2.2, c=cols, lw=0, alpha=0.65,
               rasterized=True, zorder=2)
    ax.scatter(link["leaf_x"], link["leaf_y"], s=2.2, c=S.INK, lw=0, alpha=0.55,
               rasterized=True, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=S.FS_TITLE, color=S.INK, pad=3)


def _bars(ax, comp, hierarchy):
    d = comp.iloc[::-1]
    y = np.arange(len(d))
    ax.barh(y, d["frac"], color=d["color"], height=0.5)
    for yi, f in zip(y, d["frac"]):
        ax.text(f + 0.008, yi, f"{f*100:.0f}%", va="center", ha="left",
                fontsize=S.FS_ANNOT - 0.5, color=S.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([_trunc(g) for g in d["src_group"]], fontsize=S.FS_TICK)
    ax.set_xlim(0, d["frac"].max() * 1.22)
    ax.set_xlabel("fraction of leaves", fontsize=S.FS_LABEL)
    ax.set_title(f"source {hierarchy.replace('_', ' ')}", fontsize=S.FS_TITLE,
                 color=S.INK, pad=3)
    S.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(0, d["frac"].max() + 0.01, 0.2))


def _tree(ax, ex):
    """square-dendrogram subtree, every node hued by its source group."""
    nodes = {n["node"]: n for n in ex["nodes"]}
    children = {}
    for u, v in ex["edges"]:
        children.setdefault(u, []).append(v)
    segs = []
    for p, kids in children.items():
        xp = nodes[p]["x"]
        kid_y = [nodes[c]["y"] for c in kids]
        segs.append([(xp, min(kid_y)), (xp, max(kid_y))])       # vertical riser
        for c in kids:
            segs.append([(xp, nodes[c]["y"]), (nodes[c]["x"], nodes[c]["y"])])
    ax.add_collection(LineCollection(segs, colors="#c6c6cc", lw=0.5, zorder=1))
    ax.scatter([n["x"] for n in ex["nodes"]], [n["y"] for n in ex["nodes"]],
               s=9, c=[n["color"] for n in ex["nodes"]], lw=0.2,
               edgecolors="white", zorder=2)
    root = nodes[ex["root"]]
    ax.scatter([root["x"]], [root["y"]], s=20, marker="s", c=root["color"],
               lw=0.4, edgecolors="white", zorder=3)
    ax.set_title(_trunc(ex["src_group"], 20), fontsize=S.FS_ANNOT, color=S.INK,
                 pad=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def render(key, out_dir=None, compact=False):
    out_dir = out_dir or S.OUT
    cfg = S.FANIN[key]
    S.apply_rc()
    link = pd.read_pickle(os.path.join(S.NR, f"{key}_leaves.pkl"))
    bg = pd.read_pickle(os.path.join(S.NR, "umap_bg.pkl"))
    comp = pd.read_csv(os.path.join(S.NR, f"{key}_composition.csv"))
    with open(os.path.join(S.NR, f"{key}_subtrees.pkl"), "rb") as f:
        examples = pickle.load(f)

    # explicit placement: wide gap left of the bars (for the group labels), small
    # gap between bars and trees. `compact` is the layout the figure uses -- a
    # narrower figure with the subtree grid squeezed so the conv-vs-OT scatter fits
    # to its left.
    if compact:
        w, h, umap_ax = COMPACT_W_IN, COMPACT_H_IN, [0.00, 0.055, 0.26, 0.891]
        bars_ax, sub_lr = [0.40, 0.235, 0.135, 0.625], (0.57, 0.995)
        sub_tb = (0.862, 0.084)
        name = f"fanin_{key}_compact"
    else:
        w, h, umap_ax = S.PAGE_W_IN, ROW_H_IN, [0.00, 0.02, 0.215, 0.88]
        bars_ax, sub_lr = [0.36, 0.28, 0.12, 0.46], (0.55, 0.99)
        sub_tb = (0.84, 0.26)
        name = f"fanin_{key}"

    fig = plt.figure(figsize=(w, h))
    _umap(fig.add_axes(umap_ax), link, bg, comp, "")
    _bars(fig.add_axes(bars_ax), comp, cfg["hierarchy"])
    sub = GridSpec(2, 2, figure=fig, left=sub_lr[0], right=sub_lr[1],
                   top=sub_tb[0], bottom=sub_tb[1], wspace=0.14, hspace=0.42)
    for k, ex in enumerate(examples[:4]):
        _tree(fig.add_subplot(sub[k // 2, k % 2]), ex)

    svg = os.path.join(out_dir, f"{name}.svg")
    fig.savefig(svg)
    fig.savefig(svg.replace(".svg", ".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {svg} (+ .png)")
    return svg, (w, h)


if __name__ == "__main__":
    for key in (sys.argv[1:] or list(S.FANIN)):
        render(key)
