"""row 1 (panel a): cell-type transition map, rendered from the cached graph.

reads numerical_results/transition_map.pkl (no dataset / model load) and draws a
germ-layer-coloured alluvial, TRANSPOSED so the page-wide axis carries the cell
types: developmental time flows top->bottom over its 6 levels, the N_TOP_TYPES
most abundant types spread left->right (grouped into contiguous germ-layer blocks, ordered
within/between blocks to minimise weighted edge crossings) with rotated 6 pt
labels, ribbons = the kept transitions, width and opacity by transition count.
writes panels/A_transition_map.svg.

reuses the layout helpers from analysis.lib.celltype_map.sankey
(_minimize_edge_distance, _make_ribbon); everything cosmetic (Latin Modern Sans, tight
margins, no title) is controlled here.
"""
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from analysis.figures.summary_qc import style as S
from analysis.lib.celltype_map.sankey import _minimize_edge_distance, _make_ribbon

PANEL_W_IN = S.PAGE_W_IN
PANEL_H_IN = 3.8
LABEL_H_IN = 1.18          # bottom gutter for rotated cell-type names
LEFT_IN = 0.46             # left gutter for the time axis
TOP_IN = 0.30              # top gutter for the germ-layer key
ROW_PITCH = 1.0            # vertical spacing between adjacent timepoints


def _germ_layer_map(name_to_labels):
    """type name -> germ layer, resolving coarse lineage / germ-layer names
    (e.g. 'neural ectoderm', 'mesoderm') that never appear as a cell_type."""
    m = {v["cell_type"]: v["germ_layer"] for v in name_to_labels.values()}
    for v in name_to_labels.values():          # lineage / germ-layer fallbacks
        m.setdefault(v["lineage"], v["germ_layer"])
        m.setdefault(v["germ_layer"], v["germ_layer"])
    return m


def _top_type_subgraph(G, k):
    tot = defaultdict(int)
    for n, d in G.nodes(data=True):
        tot[n[0]] += d.get("observed_count", 0)
    keep = {ct for ct, _ in sorted(tot.items(), key=lambda kv: -kv[1])[:k]}
    return G.subgraph([n for n in G.nodes() if n[0] in keep]).copy()


def render(out_dir):
    S.apply_rc()
    with open(os.path.join(S.NR, "transition_map.pkl"), "rb") as f:
        cache = pickle.load(f)
    G, color_dict = cache["graph"], cache["color_dict"]
    G = _top_type_subgraph(G, S.N_TOP_TYPES)

    ct_to_gl = _germ_layer_map(cache["name_to_labels"])
    types = _minimize_edge_distance(G, sorted({ct for ct, _ in G.nodes()}),
                                    groups=ct_to_gl)
    times = sorted({t for _, t in G.nodes()})
    col_x = {ct: i for i, ct in enumerate(types)}                  # type -> x
    row_y = {t: -i * ROW_PITCH for i, t in enumerate(times)}       # time -> y (down)
    pos = {n: np.array([float(col_x[n[0]]), float(row_y[n[1]])]) for n in G.nodes()}
    colors = {ct: color_dict.get(ct, S.INK) for ct in types}

    fig = plt.figure(figsize=(PANEL_W_IN, PANEL_H_IN))
    left = LEFT_IN / PANEL_W_IN
    bottom = LABEL_H_IN / PANEL_H_IN
    ax = fig.add_axes([left, bottom, 1 - left - 0.008,
                       1 - bottom - TOP_IN / PANEL_H_IN])

    # ribbons (vertical flow): width / opacity grow with transition count
    max_w = max((d["count"] for *_, d in G.edges(data=True)), default=1)
    for u, v, d in sorted(G.edges(data=True), key=lambda e: e[2]["count"]):
        w = d["count"] / max_w
        (x0, y0), (x1, y1) = pos[u], pos[v]
        dy = y1 - y0
        p0 = np.array([x0, y0]);             p1 = np.array([x0, y0 + 0.42 * dy])
        p2 = np.array([x1, y0 + 0.58 * dy]); p3 = np.array([x1, y1])
        verts = _make_ribbon(p0, p1, p2, p3, width=0.34 * (0.24 + 0.76 * w))
        ax.add_patch(Polygon(verts, closed=True, ec="none", zorder=1,
                             fc=to_rgba(colors[u[0]], 0.28 + 0.45 * w)))

    # nodes: area ~ observed_count (sqrt), faint halo + solid core
    counts = {n: max(d.get("observed_count", 0), 1) for n, d in G.nodes(data=True)}
    cmax = max(counts.values())
    for n, c in counts.items():
        x, y = pos[n]
        s = 5 + 42 * np.sqrt(c / cmax)
        ax.scatter(x, y, s=s * 1.9, color=to_rgba(colors[n[0]], 0.13), zorder=2)
        ax.scatter(x, y, s=s, color=colors[n[0]], zorder=3,
                   edgecolors="white", linewidths=0.4)

    ax.set_xlim(-0.7, len(types) - 0.3)
    ax.set_ylim(min(row_y.values()) - 0.55, 0.55)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=90, fontsize=S.FS_TYPE, color=S.INK)
    ax.set_yticks([row_y[t] for t in times])
    ax.set_yticklabels([f"E{t:.1f}" for t in times], color=S.INK, fontsize=S.FS_LABEL)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)

    _germ_legend(fig, color_dict, ct_to_gl, types)
    out = os.path.join(out_dir, "A_transition_map.svg")
    fig.savefig(out)
    plt.close(fig)
    return out, (PANEL_W_IN, PANEL_H_IN)


def _germ_legend(fig, color_dict, ct_to_gl, types):
    """horizontal germ-layer key (only layers shown) along the top of the panel."""
    shown = {ct_to_gl[ct] for ct in types if ct in ct_to_gl}
    items = [(gl, color_dict[gl]) for gl in sorted(shown) if gl in color_dict]
    handles = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="white",
                      mew=0.5, ms=7, label=l) for l, c in items]
    if handles:
        fig.legend(handles=handles, loc="upper left", ncol=len(handles),
                   bbox_to_anchor=(0.075, 0.998), frameon=False,
                   handletextpad=0.25, columnspacing=1.1, borderpad=0.0,
                   fontsize=S.FS_ANNOT)


if __name__ == "__main__":
    os.makedirs(os.path.join(S.OUT, "panels"), exist_ok=True)
    print(render(os.path.join(S.OUT, "panels")))
