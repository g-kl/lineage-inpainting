"""context row: the main figure's row-f accuracy heatmaps, widened to every method.

the semi-synthetic sibling of the clustermap row above. main row f draws four generative
methods (style.MAIN_METHODS); the eval also scores the two published discrete-OT couplings
(Waddington-OT, moslin) and the two pooler ablations. same cache, same 0-1 accuracy scale,
same three splits -- only the row set changes. the two-motif construction cartoon is not
repeated; it is in the main text, so the blocks take the clustermap row's column span,
label column and colorbar gutter, and the two rows line up.

the two OT rows are boxed and labelled "non-generative", matching the clustermap row: they
fit no model, they solve a coupling over the cells they are then scored on.

reads the main figure's numerical_results/context_realcells.csv.

    python -m synthetic_evals.figure_supp.plot_context_realcells
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from synthetic_evals.figure import glyphs as G
from synthetic_evals.figure import plot_context_realcells as F
from synthetic_evals.figure import plot_landscape as L
from synthetic_evals.figure_supp import style as S
from synthetic_evals.figure_supp.plot_baselines import (
    BOX_LABEL_DX, BOX_LW, BOX_PAD, HEAT_X0, HEAT_X1, NON_GENERATIVE,
)

H = 107.0
PANEL_H = H / G.U

TITLE_Y = 13.0
ROW_Y = 16.0
XLAB_PAD = 7.0
# the three blocks fill the clustermap row's column span
N_SPLITS = len(F.SPLITS)
HM_W = (HEAT_X1 - HEAT_X0 - (N_SPLITS - 1) * F.HM_GAP) / N_SPLITS


def _draw_group_box(ax, methods):
    """box the contiguous non-generative rows in each block, and name them to the right.

    one box per block rather than one spanning box: a single rectangle would draw its
    top and bottom edges across the gaps between the blocks.
    """
    idx = [i for i, m in enumerate(methods) if m in NON_GENERATIVE]
    if not idx:
        return
    y0 = ROW_Y + min(idx) * F.ROW_H - BOX_PAD
    y1 = ROW_Y + (max(idx) + 1) * F.ROW_H + BOX_PAD
    for i in range(N_SPLITS):
        ax.add_patch(Rectangle((F._hm_x0(i, HEAT_X0, HM_W) - BOX_PAD, y0),
                               HM_W + 2 * BOX_PAD, y1 - y0,
                               fc="none", ec=S.INK, lw=BOX_LW, clip_on=False, zorder=7))
    ax.text(HEAT_X1 + BOX_PAD + BOX_LABEL_DX, (y0 + y1) / 2, "non-generative",
            ha="left", va="center", fontsize=S.FS_SUB, color=S.INK, clip_on=False)


def render(out_dir, name="context_realcells_supp.svg"):
    S.apply_rc()
    df = pd.read_csv(os.path.join(S.MAIN_NR, "context_realcells.csv"))
    cols = [f"type_acc{s}" for s, _h in F.SPLITS]
    acc = df.pivot_table(index="method", columns="cfg", values=cols, sort=False)
    n = len(df["cfg"].unique())
    # METHODS + ABLATIONS in canonical order; the oracle / marg reference rows stay out,
    # as on the main page -- this row is about methods, and 0.5 is chance on the colorbar.
    methods = S.methods_present(acc.index, S.METHODS + S.ABLATIONS)

    fig = plt.figure(figsize=(S.PAGE_W_IN, PANEL_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, G.W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.set_axis_off()

    kw = dict(hm_x0=HEAT_X0, hm_w=HM_W)
    for i, (split, _h) in enumerate(F.SPLITS):
        F._draw_split(fig, i, acc, split, methods, height=H, row_y=ROW_Y, **kw)
    F._draw_titles(ax, title_y=TITLE_Y, **kw)
    F._draw_row_labels(ax, methods, row_y=ROW_Y, lab_x=HEAT_X0 - 5)
    F._draw_colorbar(fig, ax, methods, height=H, row_y=ROW_Y, gutter_x=L.GUTTER_X)
    _draw_group_box(ax, methods)

    ax.text((HEAT_X0 + HEAT_X1) / 2,
            ROW_Y + F.ROW_H * len(methods) + XLAB_PAD,
            f"one column per cell-type motif ({n} random draws)",
            ha="center", va="top", fontsize=S.FS_LABEL, color=S.INK, clip_on=False)

    G.assert_within_page(fig, ax, PANEL_H)
    out = os.path.join(out_dir, name)
    fig.savefig(out)
    fig.savefig(out.replace(".svg", ".png"), dpi=300)
    plt.close(fig)
    return out, (S.PAGE_W_IN, PANEL_H)


if __name__ == "__main__":
    print(render(S.IMGS))
