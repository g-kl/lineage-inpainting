"""panel d: fig 4 panel f's erased share of divergent genes against each cutoff, the
other held at fig 4's value (the dashed line). one line per fate. reads
numerical_results/cutoffs.csv.

    python -m analysis.figures.supp_conv.plot_cutoffs
"""
import os

import pandas as pd

from analysis.figures.supp_conv import style as S

SIZE = (2.50, 2.96)
AX_X, AX_W, AX_H = 0.50, 1.20, 0.95
AX_Y = (1.95, 0.45)            # top plot, bottom plot
XLABEL = {"ycut": "E9.5 cutoff, $|\\log_2$ fc$|$", "xcut": "E8.5 cutoff, $|\\log_2$ fc$|$"}


def render():
    D = pd.read_csv(os.path.join(S.NR, "cutoffs.csv"))
    fig, box = S.figure(*SIZE)
    # vary one cutoff, hold the other at fig 4's
    for j, (vary, held, at) in enumerate((("ycut", "xcut", S.C.DIV_XCUT),
                                          ("xcut", "ycut", S.C.DIV_YCUT))):
        ax = box(AX_X, AX_Y[j], AX_W, AX_H)
        d = D[D[held] == at]
        for p in S.FATES:
            g = d[d.panel == p].sort_values(vary)
            ax.plot(g[vary], g.erased, color=S.C.DEST_COL[p], lw=1.3, marker="o",
                    ms=2.6, mec="none")
            if j == 1:
                ax.text(g[vary].iloc[-1] + 0.08, g.erased.iloc[-1], S.FATE_LABEL[p],
                        fontsize=S.FS_LEGEND, va="center", ha="left")
        ax.axvline(S.C.DIV_YCUT if vary == "ycut" else S.C.DIV_XCUT, color=S.MUTED,
                   lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.5, 1))
        ax.set_xlabel(XLABEL[vary], labelpad=2)
        ax.set_ylabel("erased fraction")
        S.despine(ax)
    return S.save_svg(fig, "d_cutoffs")


if __name__ == "__main__":
    print(render())
