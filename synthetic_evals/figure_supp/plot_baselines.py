"""baselines row: the main figure's row-c clustermap, widened to every method.

the main page draws four generative methods (style.MAIN_METHODS); the eval also scores the
two published discrete-OT couplings (Waddington-OT, moslin) and the two pooler ablations.
same cache, same colour scale, same column order -- only the row set changes, so this
reads as row c with the extra rows folded in. the construction cartoon and the example
motifs are not repeated; they are in the main text.

the two OT rows are boxed and labelled "non-generative": they fit no model, they solve a
coupling over the cells they are then scored on, so they are the exception to what every
other row is doing rather than another entry in the same list.

reads the main figure's numerical_results/landscape_{heat,meta}.csv.

    python -m synthetic_evals.figure_supp.plot_baselines
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from synthetic_evals.figure import glyphs as G
from synthetic_evals.figure import plot_landscape as L
from synthetic_evals.figure_supp import style as S

# canvas units (glyphs.U = 100 per inch), y-down.
H = 124.0
PANEL_H = H / G.U

ANN_Y = 4.0                              # the two likeness annotation rows
ROW_Y = ANN_Y + L.ANN_H * 2 + 4.0        # method rows start below them
XLAB_Y_PAD = 6.0                         # x label below the motif ticks
# the ablation row labels ("ours, NN-snapped pooler") are wider than row c's left margin,
# so the column span starts further right; and it ends earlier to leave the right-hand
# room the group box's label needs. the colorbar and yes/no key stay where row c puts them.
HEAT_X0 = L.HEAT_X0 + 32.0
HEAT_X1 = L.HEAT_X1 - 70.0

# the rows that fit no model, boxed off from the rest
NON_GENERATIVE = ("wot", "moslin")
BOX_PAD = 1.2                            # box inset from the row block, canvas units
BOX_LW = 0.9
BOX_LABEL_DX = 5.0                       # label offset right of the box


def _draw_group_box(ax, methods):
    """box the contiguous non-generative rows and label them to the right."""
    idx = [i for i, m in enumerate(methods) if m in NON_GENERATIVE]
    if not idx:
        return
    y0 = ROW_Y + min(idx) * L.ROW_H - BOX_PAD
    y1 = ROW_Y + (max(idx) + 1) * L.ROW_H + BOX_PAD
    ax.add_patch(Rectangle((HEAT_X0 - BOX_PAD, y0),
                           (HEAT_X1 - HEAT_X0) + 2 * BOX_PAD, y1 - y0,
                           fc="none", ec=S.INK, lw=BOX_LW, clip_on=False, zorder=7))
    ax.text(HEAT_X1 + BOX_PAD + BOX_LABEL_DX, (y0 + y1) / 2, "non-generative",
            ha="left", va="center", fontsize=S.FS_SUB, color=S.INK, clip_on=False)


def render(out_dir, name="supp_baselines.svg"):
    S.apply_rc()
    heat = pd.read_csv(os.path.join(S.MAIN_NR, "landscape_heat.csv"))
    meta = pd.read_csv(os.path.join(S.MAIN_NR, "landscape_meta.csv")).iloc[0]
    n = len(heat)
    methods = S.methods_present(heat.columns)      # METHODS + ABLATIONS, canonical order

    fig = plt.figure(figsize=(S.PAGE_W_IN, PANEL_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, G.W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.set_axis_off()

    kw = dict(heat_x0=HEAT_X0, heat_x1=HEAT_X1)
    L._draw_heat(fig, heat, n, methods, height=H, row_y=ROW_Y, **kw)
    L._draw_flags(fig, heat, n, height=H, ann_y=ANN_Y, **kw)
    L._draw_motif_ticks(ax, n, methods, row_y=ROW_Y, **kw)
    L._draw_row_labels(ax, methods, row_y=ROW_Y, ann_y=ANN_Y, heat_x0=HEAT_X0)
    L._draw_colorbar(fig, ax, methods, height=H, row_y=ROW_Y, ann_y=ANN_Y)
    _draw_group_box(ax, methods)

    ax.text((HEAT_X0 + HEAT_X1) / 2,
            ROW_Y + L.ROW_H * len(methods) + L.TICK_LEN + XLAB_Y_PAD,
            f"one column per cell-type motif ({n} random draws "
            f"of {int(meta['n_candidates'])})",
            ha="center", va="top", fontsize=S.FS_LABEL, color=S.INK, clip_on=False)

    G.assert_within_page(fig, ax, PANEL_H)
    out = os.path.join(out_dir, name)
    fig.savefig(out)
    fig.savefig(out.replace(".svg", ".png"), dpi=300)
    plt.close(fig)
    return out, (S.PAGE_W_IN, PANEL_H)


if __name__ == "__main__":
    print(render(S.IMGS))
