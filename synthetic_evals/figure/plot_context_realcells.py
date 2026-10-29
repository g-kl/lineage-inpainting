"""the real-cell context row: the two-motif construction + three accuracy heatmaps.

reads numerical_results/context_realcells.csv. canvas + cartoon primitives come from
glyphs.py, so both motif constructions are drawn in one visual language. see concept.md.

    python -m synthetic_evals.figure.plot_context_realcells
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from synthetic_evals.figure import glyphs as G
from synthetic_evals.figure import style as S
from synthetic_evals.figure.glyphs import (
    BRACE, GRID, GRID_LW, LETTER_COLORS, NODE_ROOT, U, W, curve as _curve, edges as _edges,
    leaf_xy as _leaf_xy, node_text as _node_text, parents as _parents,
)

H = 66.0
PANEL_H = H / U                 # inches

INK = S.INK

# --- cartoon layout (canvas units) ------------------------------------------------
TREE_CX = (46.0, 156.0)
ROOT_Y, PROG_Y, LEAF_Y = 6.0, 28.0, 50.0
SEP_P, SEP_L = 22.0, 11.0

# --- heatmap block layout ---------------------------------------------------------
LAB_X = 262.0                   # right edge of the row labels
HM_X0, HM_W, HM_GAP = 266.0, 96.0, 11.0
TITLE_Y = 12.0                  # baseline the titles sit above
ROW_Y, ROW_H = 15.0, 9.0        # one row per method
GUTTER_X, SWATCH = 584.0, 8.0
CBAR_LABEL_X = 632.0
XLAB_Y = 54.0                   # x label under the heatmaps

SPLITS = (
    ("", "all progenitors"),
    ("_unambiguous", "within-type"),
    ("_ambiguous", "between-type"),
)


def _rect(fig, x, y, w, h):
    return G.rect(fig, x, y, w, h, height=H)


def _hm_x0(i, hm_x0=HM_X0, hm_w=HM_W):
    return hm_x0 + i * (hm_w + HM_GAP)


# --- the construction cartoon -----------------------------------------------------
def _draw_motif(ax, cx, prog_type, prog_col):
    """one tree: hidden root -> two same-type progenitors -> two children each, the
    right-hand progenitor differentiating into the shared type B."""
    par = _parents(cx, PROG_Y, SEP_P)
    leaves = _leaf_xy(cx, [LEAF_Y] * 4, SEP_P, SEP_L)
    _curve(ax, (cx, ROOT_Y), par[0])
    _curve(ax, (cx, ROOT_Y), par[1])
    _edges(ax, par, leaves)

    ax.add_patch(Circle((cx, ROOT_Y), 2.6, fc=NODE_ROOT, ec=NODE_ROOT, lw=0, zorder=5))
    for x, y in par:
        _node_text(ax, x, y, prog_type, "white", prog_col)
    for i, (x, y) in enumerate(leaves):
        lab = prog_type if i < 2 else "B"
        _node_text(ax, x, y, lab, "white", LETTER_COLORS[lab])


def _draw_cartoon(ax):
    for cx, t in zip(TREE_CX, "AC"):
        _draw_motif(ax, cx, t, LETTER_COLORS[t])
    ax.text(np.mean(TREE_CX), PROG_Y, "OR", ha="center", va="center",
            fontsize=S.FS_SUB, color=INK, clip_on=False)


# --- the three accuracy heatmaps --------------------------------------------------
def _mesh(ax, M, cmap):
    ax.pcolormesh(np.arange(M.shape[1] + 1), np.arange(M.shape[0] + 1), M,
                  cmap=cmap, vmin=S.ACC_VMIN, vmax=S.ACC_VMAX, edgecolors=GRID,
                  linewidth=GRID_LW)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(M.shape[0], 0)
    for s in ax.spines.values():
        s.set_color(GRID); s.set_linewidth(0.5)


def _draw_split(fig, i, acc, split, methods, *, height=H, row_y=ROW_Y, hm_x0=HM_X0,
                hm_w=HM_W):
    _mesh(G.rect(fig, _hm_x0(i, hm_x0, hm_w), row_y, hm_w, ROW_H * len(methods),
                 height=height),
          acc.loc[methods, f"type_acc{split}"].to_numpy(), S.ACC_CMAP)


def _draw_titles(ax, *, title_y=TITLE_Y, hm_x0=HM_X0, hm_w=HM_W):
    for i, (_split, head) in enumerate(SPLITS):
        ax.text(_hm_x0(i, hm_x0, hm_w) + hm_w / 2, title_y, head, ha="center", va="bottom",
                fontsize=S.FS_LABEL, color=INK, clip_on=False)


def _draw_row_labels(ax, methods, *, row_y=ROW_Y, lab_x=LAB_X):
    for i, m in enumerate(methods):
        ax.text(lab_x, row_y + (i + 0.5) * ROW_H, S.METHOD_LABELS[m], ha="right",
                va="center", fontsize=S.FS_SUB, color=S.METHOD_COLORS[m], clip_on=False)


def _draw_colorbar(fig, ax, methods, *, height=H, row_y=ROW_Y, gutter_x=GUTTER_X):
    cax = G.rect(fig, gutter_x, row_y, SWATCH, ROW_H * len(methods), height=height)
    cb = fig.colorbar(ScalarMappable(Normalize(S.ACC_VMIN, S.ACC_VMAX), S.ACC_CMAP),
                      cax=cax, extend="min", ticks=[S.ACC_VMIN, S.ACC_VMAX])
    cb.ax.set_yticklabels([r"$\leq$" + f"{S.ACC_VMIN:.1f}", f"{S.ACC_VMAX:.1f}"])
    cb.ax.tick_params(labelsize=S.FS_TICK, length=1.5, pad=1.2)
    cb.outline.set_visible(False)
    ax.text(CBAR_LABEL_X + gutter_x - GUTTER_X, row_y + ROW_H * len(methods) / 2,
            "progenitor\ntype accuracy", ha="center", va="center", rotation=270,
            linespacing=1.15, fontsize=S.FS_LABEL, color=INK, clip_on=False)


def render(out_dir, name="context_realcells.svg"):
    S.apply_rc()
    df = pd.read_csv(os.path.join(S.NR, "context_realcells.csv"))
    cols = [f"type_acc{s}" for s, _h in SPLITS]
    acc = df.pivot_table(index="method", columns="cfg", values=cols, sort=False)
    n = len(df["cfg"].unique())
    methods = S.methods_present(acc.index, S.MAIN_METHODS)

    fig = plt.figure(figsize=(S.PAGE_W_IN, PANEL_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0)   # y-down
    ax.set_aspect("equal"); ax.set_axis_off()

    _draw_cartoon(ax)
    for i, (split, _h) in enumerate(SPLITS):
        _draw_split(fig, i, acc, split, methods)
    _draw_titles(ax)
    _draw_row_labels(ax, methods)
    _draw_colorbar(fig, ax, methods)
    ax.text((HM_X0 + _hm_x0(len(SPLITS) - 1) + HM_W) / 2, XLAB_Y,
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
