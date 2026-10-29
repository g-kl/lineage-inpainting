"""panel a: every MGI pair, the gene's E9.0 mean log1p expression in the cell type
against the model's depletion, the y axis cut at [-1, 1]. reads
numerical_results/pairs.csv.

    python -m analysis.figures.supp_mgi.plot_scatter
"""
import os

import pandas as pd
from scipy.stats import spearmanr

from analysis.figures.supp_mgi import style as S

LEFT, RIGHT = 0.6, 0.1


def render():
    d = pd.read_csv(os.path.join(S.NR, "pairs.csv"))
    m = d[d["is_mgi"]]
    rho = spearmanr(m["expression"], m["depletion"]).statistic

    fig, box = S.figure(S.SCATTER_W, S.TOP_H)
    ax = box(LEFT, S.AX_BOTTOM, S.SCATTER_W - LEFT - RIGHT, S.TOP_H - S.AX_BOTTOM - S.AX_TOP)
    ax.axhline(0, color=S.GREY, lw=0.8, ls="--", zorder=0)
    ax.scatter(m["expression"], m["depletion"], s=5, color=S.INK, alpha=0.5, lw=0)
    ax.set_xlim(-0.15, m["expression"].max() + 0.2)
    ax.set_ylim(-1, 1.05)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("E9.0 expression, log1p CP10K")
    ax.set_ylabel("predicted depletion")
    ax.text(0.97, 0.04, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes, ha="right",
            va="bottom")
    return S.save_svg(fig, "a_scatter")


if __name__ == "__main__":
    print(render())
