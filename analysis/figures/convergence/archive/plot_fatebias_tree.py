"""row-3 right: the fate-bias tree illustration (reads only numerical_results/).

one example EC-generating subtree drawn twice, side by side -- coloured by cell type
and by EC fate bias (same topology) so the fate bias can be read against the
underlying identities. trees are rotated to grow downward (root at top, leaves along
the bottom), giving a short-wide panel that tiles with the deg strip. writes
imgs/panels/fatebias_ec_tree.{svg,png}.

    python analysis/figures/convergence/plot_fatebias_tree.py
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import style as S

KEY = "ec"
PANELS = os.path.join(S.OUT, "panels")


def _abbr(s):
    return s.replace(" mesoderm", " meso.")


def _segments(nodes, edges, pos):
    """square-dendrogram segments, time on the (downward) Y axis."""
    children = {}
    for u, v in edges:
        children.setdefault(u, []).append(v)
    segs = []
    for p, kids in children.items():
        yp = pos[p][1]
        kx = [pos[c][0] for c in kids]
        segs.append([(min(kx), yp), (max(kx), yp)])            # horizontal riser
        for c in kids:
            segs.append([(pos[c][0], yp), (pos[c][0], pos[c][1])])  # vertical branch
    return segs


def _draw(ax, nodes, edges, pos, colors):
    ax.add_collection(LineCollection(_segments(nodes, edges, pos), colors="#c6c6cc",
                                     lw=0.5, zorder=1))
    ax.scatter([pos[n["node"]][0] for n in nodes], [pos[n["node"]][1] for n in nodes],
               s=[15 if n["is_leaf"] else 6 for n in nodes], c=colors, lw=0.3,
               edgecolors="#9a9a9a", zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0.03, y=0.10)
    for sp in ax.spines.values():
        sp.set_visible(False)


def render(out_dir=PANELS):
    S.apply_rc()
    with open(os.path.join(S.NR, f"fatebias_{KEY}_tree.pkl"), "rb") as fh:
        T = pickle.load(fh)
    nodes, edges, family = T["nodes"], T["edges"], set(T["family"])

    # rotate: leaf spread on X, developmental time growing downward on Y.
    pos = {n["node"]: (n["y"], -n["x"]) for n in nodes}

    others = sorted({n["cell_type"] for n in nodes
                     if n["cell_type"] and n["cell_type"] not in family})
    tcol = {t: S.SOURCE_TYPE_COLORS.get(t, S.MUTED) for t in others}
    type_color = [S.GREY if n["cell_type"] is None else
                  (S.ACCENT if n["is_ec"] else tcol[n["cell_type"]]) for n in nodes]

    fb = np.array([n["fate_bias"] for n in nodes], float)
    fb_norm = Normalize(0, 1)
    fb_color = [S.FATEBIAS_CMAP(fb_norm(v)) if np.isfinite(v) else S.GREY for v in fb]

    fig = plt.figure(figsize=(2.5, 1.0))
    gs = fig.add_gridspec(1, 2, left=0.26, right=0.90, top=0.84, bottom=0.06,
                          wspace=0.10)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _draw(ax0, nodes, edges, pos, type_color)
    _draw(ax1, nodes, edges, pos, fb_color)
    ax0.set_title("cell type", fontsize=S.FS_ANNOT, pad=2)
    ax1.set_title("EC fate bias", fontsize=S.FS_ANNOT, pad=2)

    handles = [Line2D([0], [0], marker="o", ls="", ms=4, mfc=S.ACCENT, mec="none",
                      label="endothelial")]
    handles += [Line2D([0], [0], marker="o", ls="", ms=4, mfc=tcol[t], mec="none",
                        label=_abbr(t)) for t in others]
    ax0.legend(handles=handles, loc="center right", bbox_to_anchor=(-0.06, 0.5),
               frameon=False, fontsize=S.FS_LEGEND, handletextpad=0.25,
               labelspacing=0.3, borderpad=0.0)

    sm = ScalarMappable(norm=fb_norm, cmap=S.FATEBIAS_CMAP)
    sm.set_array([])
    cax = ax1.inset_axes([1.06, 0.05, 0.05, 0.9])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=S.FS_TICK, length=2)
    cb.outline.set_visible(False)

    svg = os.path.join(out_dir, f"fatebias_{KEY}_tree.svg")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(svg.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {svg} (+ .png)")


if __name__ == "__main__":
    os.makedirs(PANELS, exist_ok=True)
    render()
