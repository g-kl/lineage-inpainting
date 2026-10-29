"""panel b: MGI pairs in the knockout's lowest E9.0 expression third, each at its rank
among the knockout's cell types by E9.0 expression and by the model's depletion. reads
numerical_results/examples.csv, pairs.csv.

    python -m analysis.figures.supp_mgi.plot_examples
"""
import os

import pandas as pd

from analysis.figures.supp_mgi import style as S

LEFT, RIGHT = 2.05, 0.1


def render():
    e = pd.read_csv(os.path.join(S.NR, "examples.csv"))
    n_ct = pd.read_csv(os.path.join(S.NR, "pairs.csv"))["cell_type"].nunique()

    fig, box = S.figure(S.EXAMPLES_W, S.TOP_H)
    ax = box(LEFT, S.AX_BOTTOM, S.EXAMPLES_W - LEFT - RIGHT, S.TOP_H - S.AX_BOTTOM - S.AX_TOP)
    y = range(len(e))
    ax.hlines(y, e["model_rank"], e["expr_rank"], color=S.GREY, lw=1.0, zorder=1)
    for s, col in (("expression", "expr_rank"), ("model", "model_rank")):
        ax.scatter(e[col], y, s=22, color=S.SCORE_COLORS[s], zorder=2, lw=0)
    # the two scores named once, over the top row's dots, each flush with its dot's
    # outer side so neither runs past the axes
    for s, col, ha, dx in (("expression", "expr_rank", "right", 0.8),
                           ("model", "model_rank", "left", -0.8)):
        ax.text(e[col].iloc[0] + dx, -0.6, s, ha=ha, va="bottom")
    ax.set_yticks(list(y), [f"{r.gene}, {r.cell_type}" for r in e.itertuples()])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(e) - 0.5, -1.3)
    ax.set_xlim(0, n_ct + 1)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_xlabel("rank")
    ax.annotate(f"over {n_ct} MGI cell types", xy=(0.5, 0), xycoords=ax.xaxis.label,
                xytext=(0, -1), textcoords="offset points", ha="center", va="top",
                fontsize=S.FS_SMALL)
    return S.save_svg(fig, "b_examples")


if __name__ == "__main__":
    print(render())
