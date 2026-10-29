"""panel b: fig 4 panel b on the raw scores, before the division by the leaf's local
spacing. one point per cell type, area by dual-scored leaves; endothelial and neuron
types picked out as in fig 4. reads numerical_results/celltype_scores.csv.

    python -m analysis.figures.supp_conv.plot_raw
"""
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr

from analysis.figures.supp_conv import style as S

SIZE = (2.30, 2.75)
AX = (0.55, 0.50, 1.67, 2.19)     # left, bottom, width, height


def render():
    T = pd.read_csv(os.path.join(S.NR, "celltype_scores.csv"))
    fig, box = S.figure(*SIZE)
    ax = box(*AX)
    sizes = 5 + 45 * np.sqrt(T.n / T.n.max())
    for fam in ("other", "neuron", "endothelial"):          # highlights on top
        m = T.family == fam
        hi = fam != "other"
        ax.scatter(T.ot_raw[m], T.conv_raw[m], s=sizes[m], c=S.FAMILY_COL[fam],
                   alpha=0.9 if hi else 0.5, lw=0.4 if hi else 0, edgecolors="white",
                   zorder=3 if hi else 1)
    rho = spearmanr(T.ot_raw, T.conv_raw)[0]
    ax.text(0.97, 0.04, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=S.FS_LEGEND, color=S.MUTED)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=4, mfc=S.FAMILY_COL[f],
                              mec="none", label=f) for f in ("endothelial", "neuron")],
              loc="upper left", frameon=False, handletextpad=0.2, labelspacing=0.25,
              borderpad=0.0, borderaxespad=0.1)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel("OT distortion, raw")
    ax.set_ylabel("convergence score, raw")
    S.despine(ax)
    return S.save_svg(fig, "b_raw")


if __name__ == "__main__":
    print(render())
