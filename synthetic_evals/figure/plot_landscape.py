"""the real-cell coherence row: motif construction + example motifs over the clustermap.

reads numerical_results/landscape_{heat,examples,meta}.csv. canvas + cartoon primitives
come from glyphs.py. see concept.md.

    python -m synthetic_evals.figure.plot_landscape
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable

from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Circle, Rectangle

from synthetic_evals.figure import glyphs as G
from synthetic_evals.figure import style as S
from synthetic_evals.figure.glyphs import (
    BRACE, GRID, GRID_LW, LETTER_COLORS, NODE_ROOT, U, W, brace_h as _braceH, curve as _curve,
    edges as _edges, leaf_xy as _leaf_xy, node_text as _node_text, parents as _parents,
    tint as _tint,
)

H = 157.0
PANEL_H = H / U                 # inches

INK = S.INK
MUTED = S.MUTED

FLAG_YES, FLAG_NO = BRACE, "#e8eaee"   # annotation rows (metadata, grey)

# --- per-cell-type palette (example motifs) ---------------------------------------
# one hue per cell type, grouped by tissue class so the wiring reads at a glance:
# neural types are TEAL, cranial-mesenchyme types are PURPLE, ectoderm a spare purple.
# keyed by prefix (subtype-blind).
TYPE_COLORS = {
    "hindbrain": "#128079",             # neural (dark teal)
    "spinal cord": "#72c7bf",           # neural (light teal)
    "midbrain": "#2f9e94",              # neural (teal, spare)
    "telencephalon": "#9fd6cf",         # neural (pale teal, spare)
    "pharyngeal arch": "#6a4c93",       # cranial mesenchyme (dark purple)
    "frontonasal": "#b39ddb",           # cranial mesenchyme (light purple)
    "epidermal progenitor": "#8c6bb1",  # surface ectoderm (purple, spare)
}

# --- cell-type names, parentheticals dropped and kept to one line -----------------
ABBR = {
    "telencephalon (dorsal)": "telencephalon",
    "midbrain": "midbrain",
    "hindbrain (r1)": "hindbrain",
    "hindbrain (r2-r4)": "hindbrain",
    "hindbrain (r5-r8)": "hindbrain",
    "spinal cord (cervical)": "spinal cord",
    "spinal cord (thoracic)": "spinal cord",
    "frontonasal mesenchyme": "frontonasal mes.",
    "pharyngeal arch mesenchyme": "phar. arch mes.",
    "pharyngeal arch mesenchyme (p1-p2)": "phar. arch mes.",
    "epidermal progenitor (posterior)": "epidermal prog.",
}

# --- layout (canvas units) --------------------------------------------------------
# --- top block: construction cartoon + one example motif per category -------------
HEAD_Y = 0.0                       # headings
LEFT_CX = 67.0                     # construction cartoon centre
COL_CX = (215.0, 385.0, 555.0)     # example motif centres
TREE_ROOT_Y, TREE_PARENT_Y = 15.0, 29.0
TREE_LEAF_HI, TREE_LEAF_LO = 49.0, 66.0
SEP_P, SEP_L = 27.0, 15.0          # construction tree half-separations
SEP_P_EX, SEP_L_EX = 42.0, 27.0    # example tree half-separations
NODE_R = 2.6

# --- bottom block: the clustermap -------------------------------------------------
HEAT_X0, HEAT_X1 = 74.0, 560.0     # clustermap column span
ANN_Y, ANN_H = 87.0, 7.0           # annotation rows: OT-like, Felsenstein-like
ROW_Y, ROW_H = 104.0, 9.0          # one row per method
TICK_EVERY, TICK_LEN = 4, 2.5      # motif ticks under the clustermap
XLAB_Y = 145.0                     # x label under the ticks
GUTTER_X, SWATCH = 570.0, 9.0      # right gutter: colorbar + the yes/no key
CBAR_LABEL_X = 616.0


def _abbr(name):
    return ABBR.get(name, name)


def _type_color(name):
    for prefix, hexc in TYPE_COLORS.items():
        if name.startswith(prefix):
            return hexc
    return MUTED




# --- clustermap (one column per motif, one row per method) -------------------------
def _rect(fig, x, y, w, h):
    return G.rect(fig, x, y, w, h, height=H)


def _strip(ax, n):
    ax.set_xlim(0, n)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_heat(fig, heat, n, methods, *, height=H, row_y=ROW_Y,
               heat_x0=HEAT_X0, heat_x1=HEAT_X1):
    ax = G.rect(fig, heat_x0, row_y, heat_x1 - heat_x0, ROW_H * len(methods),
                height=height)
    _strip(ax, n)
    ax.pcolormesh(np.arange(n + 1), np.arange(len(methods) + 1),
                  heat[methods].to_numpy().T, cmap=S.ACC_CMAP,
                  vmin=S.ACC_VMIN, vmax=S.ACC_VMAX,
                  edgecolors=GRID, linewidth=GRID_LW)
    ax.set_ylim(len(methods), 0)


def _draw_motif_ticks(ax, n, methods, *, row_y=ROW_Y, heat_x0=HEAT_X0, heat_x1=HEAT_X1):
    """short ticks under the clustermap marking every TICK_EVERY-th motif column."""
    y0 = row_y + ROW_H * len(methods)
    step = (heat_x1 - heat_x0) / n
    for i in range(0, n + 1, TICK_EVERY):
        x = heat_x0 + i * step
        ax.plot([x, x], [y0, y0 + TICK_LEN], color=BRACE, lw=0.5, solid_capstyle="butt",
                clip_on=False, zorder=6)


def _draw_flags(fig, heat, n, *, height=H, ann_y=ANN_Y, heat_x0=HEAT_X0, heat_x1=HEAT_X1):
    ax = G.rect(fig, heat_x0, ann_y, heat_x1 - heat_x0, ANN_H * 2, height=height)
    _strip(ax, n)
    flags = heat[["ot_like", "fels_like"]].to_numpy().T.astype(float)
    ax.imshow(flags, aspect="auto", cmap=ListedColormap([FLAG_NO, FLAG_YES]),
              vmin=0, vmax=1, interpolation="nearest", extent=(0, n, 2, 0))
    ax.set_ylim(2, 0)


def _draw_row_labels(ax, methods, *, row_y=ROW_Y, ann_y=ANN_Y, heat_x0=HEAT_X0):
    for i, m in enumerate(methods):
        ax.text(heat_x0 - 5, row_y + (i + 0.5) * ROW_H, S.METHOD_LABELS[m],
                ha="right", va="center", fontsize=S.FS_SUB,
                color=S.METHOD_COLORS[m], clip_on=False)
    for i, lab in enumerate(("OT-like", "Felsenstein-like")):
        ax.text(heat_x0 - 5, ann_y + (i + 0.5) * ANN_H, lab, ha="right", va="center",
                fontsize=S.FS_SUB, color=BRACE, clip_on=False)


def _draw_colorbar(fig, ax, methods, *, height=H, row_y=ROW_Y, ann_y=ANN_Y):
    cax = G.rect(fig, GUTTER_X, row_y, SWATCH, ROW_H * len(methods), height=height)
    cb = fig.colorbar(ScalarMappable(Normalize(S.ACC_VMIN, S.ACC_VMAX), S.ACC_CMAP),
                      cax=cax, extend="min", ticks=[S.ACC_VMIN, S.ACC_VMAX])
    cb.ax.set_yticklabels([r"$\leq$" + f"{S.ACC_VMIN:.1f}", f"{S.ACC_VMAX:.1f}"])
    cb.ax.tick_params(labelsize=S.FS_TICK, length=1.5, pad=1.2)
    cb.outline.set_visible(False)
    ax.text(CBAR_LABEL_X, row_y + ROW_H * len(methods) / 2, "type\naccuracy",
            ha="center", va="center", rotation=270, linespacing=1.15,
            fontsize=S.FS_LABEL, color=INK, clip_on=False)
    for i, (lab, fc) in enumerate((("yes", FLAG_YES), ("no", FLAG_NO))):
        y = ann_y + (i + 0.5) * ANN_H
        ax.add_patch(Rectangle((GUTTER_X, y - ANN_H / 2 + 0.6), SWATCH, ANN_H - 1.2,
                               fc=fc, ec="none", clip_on=False))
        ax.text(GUTTER_X + SWATCH + 3, y, lab, ha="left", va="center",
                fontsize=S.FS_SUB, color=INK, clip_on=False)


# --- top row: the construction, then one example motif per category ---------------
def _draw_construction(ax, cx, n_motifs):
    ax.text(cx, HEAD_Y, f"{n_motifs} cell type motifs", ha="center", va="top",
            fontsize=S.FS_LABEL, color=INK, clip_on=False)
    par = _parents(cx, TREE_PARENT_Y, SEP_P)
    leaves = _leaf_xy(cx, [TREE_LEAF_HI] * 4, SEP_P, SEP_L)
    _curve(ax, (cx, TREE_ROOT_Y), par[0]); _curve(ax, (cx, TREE_ROOT_Y), par[1])
    ax.add_patch(Circle((cx, TREE_ROOT_Y), NODE_R, fc=NODE_ROOT, ec=NODE_ROOT, lw=0, zorder=5))
    _edges(ax, par, leaves)
    for (x, y), lab in zip(par + leaves, "ABCDEF"):
        _node_text(ax, x, y, lab, "white", LETTER_COLORS[lab])

    _braceH(ax, leaves[0][0] - 8, leaves[3][0] + 8, TREE_LEAF_HI + 9, 6, 1)
    ax.text(cx, TREE_LEAF_HI + 23, "randomly sampled cell types", ha="center", va="top",
            fontsize=S.FS_SUB, color=INK, clip_on=False)


def _draw_example(ax, cx, motif):
    par = _parents(cx, TREE_PARENT_Y, SEP_P_EX)
    leaves = _leaf_xy(cx, [TREE_LEAF_HI, TREE_LEAF_LO, TREE_LEAF_HI, TREE_LEAF_LO],
                      SEP_P_EX, SEP_L_EX)
    _curve(ax, (cx, TREE_ROOT_Y), par[0]); _curve(ax, (cx, TREE_ROOT_Y), par[1])
    ax.add_patch(Circle((cx, TREE_ROOT_Y), NODE_R, fc=NODE_ROOT, ec=NODE_ROOT, lw=0, zorder=5))
    _edges(ax, par, leaves)
    for (x, y), key in zip(par + leaves, ["pA", "pB", "anchorA", "A2", "B1", "anchorB"]):
        col = _type_color(motif[key])
        _node_text(ax, x, y, _abbr(motif[key]), _tint(col, 0.82), col)


def _draw_heading(ax, cx, text):
    ax.text(cx, HEAD_Y, text, ha="center", va="top", fontsize=S.FS_LABEL, color=INK,
            clip_on=False)




def render(out_dir, name="landscape.svg"):
    S.apply_rc()
    heat = pd.read_csv(os.path.join(S.NR, "landscape_heat.csv"))
    examples = pd.read_csv(os.path.join(S.NR, "landscape_examples.csv")).set_index("category")
    n = len(heat)
    methods = S.methods_present(heat.columns, S.MAIN_METHODS)

    fig = plt.figure(figsize=(S.PAGE_W_IN, PANEL_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0)   # y-down
    ax.set_aspect("equal"); ax.set_axis_off()

    _draw_construction(ax, LEFT_CX, n)
    for cat, cx in zip(S.CATEGORIES, COL_CX):
        _draw_heading(ax, cx, S.CAT_LABELS[cat])
        _draw_example(ax, cx, examples.loc[cat])

    _draw_heat(fig, heat, n, methods)
    _draw_flags(fig, heat, n)
    _draw_motif_ticks(ax, n, methods)
    _draw_row_labels(ax, methods)
    _draw_colorbar(fig, ax, methods)
    ax.text((HEAT_X0 + HEAT_X1) / 2, XLAB_Y,
            f"one column per cell-type motif ({n} random draws)",
            ha="center", va="top", fontsize=S.FS_LABEL, color=INK, clip_on=False)

    G.assert_within_page(fig, ax, PANEL_H)
    out = os.path.join(out_dir, name)
    fig.savefig(out)
    fig.savefig(out.replace(".svg", ".png"), dpi=300)
    plt.close(fig)
    return out, (S.PAGE_W_IN, PANEL_H)


if __name__ == "__main__":
    print(render(S.IMGS))
