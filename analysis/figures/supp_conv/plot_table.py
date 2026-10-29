"""panel a: the N_TOP most and least convergent cell types, with their OT distortion,
as one ranked table with the middle of the ranking cut out. laid out in inches so each
column is as wide as its longest entry. reads numerical_results/celltype_scores.csv.

    python -m analysis.figures.supp_conv.plot_table
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.figures.supp_conv import style as S

PITCH = 0.13        # row height
HEAD = 0.20         # column headers
BREAK = 0.10        # the cut between the top and the bottom of the ranking
COL_GAP = 0.10
DOT = 0.10          # room for the family dot left of the cell type
# header, column, format, alignment
COLS = (("", "rank_conv", "{:d}", "right"),
        ("cell type", "cell_type", "{}", "left"),
        ("convergence", "conv", "{:.2f}", "right"),
        ("OT distortion", "ot", "{:.2f}", "right"),
        ("OT rank", "rank_ot", "{:d}", "right"),
        ("leaves", "n", "{:,}", "right"))


def _width(fig, s):
    t = fig.text(0, 0, s, fontsize=S.FS_ROW)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def render():
    S.apply_rc()
    T = pd.read_csv(os.path.join(S.NR, "celltype_scores.csv"))
    rows = pd.concat([T.head(S.N_TOP), T.tail(S.N_TOP)])

    probe = plt.figure()
    widths = [max([_width(probe, h)] + [_width(probe, f.format(v)) for v in rows[c]])
              for h, c, f, _ in COLS]
    plt.close(probe)
    edges, x = [], 0.0
    for i, w in enumerate(widths):
        x += DOT if i == 1 else 0.0
        edges.append((x, x + w))
        x += w + COL_GAP
    size = (x - COL_GAP, HEAD + PITCH * len(rows) + BREAK)

    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.axis("off")
    txt = dict(fontsize=S.FS_ROW, va="center")
    for (h, _, _, al), (lo, hi) in zip(COLS, edges):
        ax.text(hi if al == "right" else lo, HEAD / 2 - 0.02, h, ha=al, **txt)
    ax.plot([0, size[0]], [HEAD - 0.02] * 2, color=S.MUTED, lw=0.5)
    cut = HEAD + PITCH * S.N_TOP + BREAK / 2
    ax.plot([0, size[0]], [cut] * 2, color=S.MUTED, lw=0.5, ls=(0, (1, 2)))
    for i, r in enumerate(rows.itertuples()):
        y = HEAD + PITCH * (i + 0.5) + (BREAK if i >= S.N_TOP else 0.0)
        for (_, c, f, al), (lo, hi) in zip(COLS, edges):
            ax.text(hi if al == "right" else lo, y, f.format(getattr(r, c)), ha=al, **txt)
        if r.family != "other":
            ax.scatter(edges[1][0] - DOT / 2, y, s=9, lw=0, color=S.FAMILY_COL[r.family])
    return S.save_svg(fig, "a_table")


if __name__ == "__main__":
    print(render())
