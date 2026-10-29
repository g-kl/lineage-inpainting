"""panels c-f: the N_TOP heaviest transitions of one non-match category, one row each with
its cells. laid out in inches so each panel is as wide as its longest label. reads
numerical_results/transitions.csv."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.figures.supp_summary import style as S

PITCH = 0.125       # row height
HEAD = 0.2          # room for the header row
GAP = 0.06
ARROW = r"$\rightarrow$"


def _width(fig, s):
    t = fig.text(0, 0, s, fontsize=S.FS_ROW)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def _layout(rows, label):
    """figure and size for (source, target, cells) rows under a header row: the
    category label on the left, `cells` over the count column on the right."""
    probe = plt.figure()
    ws = max(_width(probe, r[0]) for r in rows)
    wt = max(_width(probe, r[1]) for r in rows)
    wc = max(_width(probe, f"{r[2]:,}") for r in rows)
    wa = _width(probe, ARROW)
    plt.close(probe)
    x_src = ws
    x_arrow = x_src + GAP + wa / 2
    x_tgt = x_arrow + wa / 2 + GAP
    x_cnt = x_tgt + wt + 3 * GAP + wc    # right edge of the count column
    size = (x_cnt, HEAD + PITCH * len(rows))
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, size[0])
    ax.set_ylim(size[1], 0)
    ax.axis("off")
    ax.text(0, HEAD - GAP, label, ha="left", va="bottom", fontsize=S.FS_ROW,
            fontweight="bold")
    ax.text(x_cnt, HEAD - GAP, "cells", ha="right", va="bottom", fontsize=S.FS_ROW)
    for i, (src, tgt, n) in enumerate(rows):
        y = HEAD + PITCH * (i + 0.5)
        ax.text(x_src, y, src, ha="right", va="center", fontsize=S.FS_ROW)
        ax.text(x_arrow, y, ARROW, ha="center", va="center", fontsize=S.FS_ROW)
        ax.text(x_tgt, y, tgt, ha="left", va="center", fontsize=S.FS_ROW)
        ax.text(x_cnt, y, f"{n:,}", ha="right", va="center", fontsize=S.FS_ROW)
    return fig, size


def render(category, name):
    S.apply_rc()
    t = pd.read_csv(os.path.join(S.NR, "transitions.csv"))
    t = t[t.category == category].nlargest(S.N_TOP, "count")
    fig, size = _layout(list(zip(t.source_type, t.target_type, t["count"])),
                        S.CATEGORY_LABELS[category])
    out = os.path.join(S.PANELS, name)
    fig.savefig(out)
    plt.close(fig)
    return out, size
