"""crossing transitions row: the main page's fate-accuracy vs ambient-dimension line plot
(rows a/b, second split only) at every transition. rows coherent / crossing, one column
per transition; the 7->8 column is the main-text pair, 5->6 is the first split (the same
in both variants), 6->7 and 8->9 are within-fate steps.

reads the main figure's numerical_results/crossing_fate.csv.

    python -m synthetic_evals.figure_supp.plot_crossing_transitions
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from synthetic_evals.figure import plot_crossing as pc
from synthetic_evals.figure_supp import style as S

PANEL_H = 2.3


def render(out_dir, name="supp_crossing_transitions.svg"):
    S.apply_rc()
    df = pd.read_csv(os.path.join(S.MAIN_NR, "crossing_fate.csv"))
    pts = sorted(set(zip(df["t_src"], df["t_tgt"])))
    fig = plt.figure(figsize=(S.PAGE_W_IN, PANEL_H))
    gs = fig.add_gridspec(2, len(pts), left=0.09, right=0.84, top=0.91, bottom=0.17,
                          wspace=0.12, hspace=0.15)
    for r, v in enumerate(("noncrossing", "crossing")):
        for c, (ts, tt) in enumerate(pts):
            ax = fig.add_subplot(gs[r, c])
            pc._dim_panel(ax, df[(df["variant"] == v) & (df["t_tgt"] == tt)],
                          xticklabels=r == 1, xlabel=r == 1)
            if r == 0:
                ax.set_title(fr"${ts:g}\!\to\!{tt:g}$", fontsize=S.FS_TITLE)
            if c == 0:
                ax.set_ylabel(f"{pc.VARIANT_LABEL[v]}\nfate accuracy")
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
    h = [Line2D([], [], color=S.METHOD_COLORS[m], marker=S.METHOD_MARKERS[m], ms=3.6, lw=1.3,
                label=S.METHOD_LABELS[m]) for m in S.MAIN_METHODS]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=(0.855, 0.5), frameon=False,
               handlelength=1.4, fontsize=S.FS_SUB)
    return S.save_panel(fig, out_dir, name, (S.PAGE_W_IN, PANEL_H))


if __name__ == "__main__":
    print(render(S.IMGS))
