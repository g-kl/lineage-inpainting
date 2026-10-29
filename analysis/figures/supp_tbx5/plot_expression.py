"""c and d: Tbx5 expression alone against the model. reads
numerical_results/expression_{cells,r}.csv.

c  each heart tube reference cell's raw Tbx5 against its measured nn distortion, in its
   subtype's colour with the subtype means on top: the Tbx5 figure's panel l with the
   model swapped for expression. the correlation is over the cells.
d  the correlation with the measured distortion for both predictors, a pair of bars per
   crispant embryo against that embryo's own measurement, and a last pair against the
   mean over all of them.

    python -m analysis.figures.supp_tbx5.plot_expression
"""
import os

import numpy as np
import pandas as pd

from analysis.figures.supp_tbx5 import style as S

TBX5 = "tbx5"
BAR_W = 0.38            # one bar of a pair, in embryo slots
ALL_GAP = 0.6           # extra room before the all-embryo pair


def render_scatter():
    d = pd.read_csv(os.path.join(S.NR, "expression_cells.csv"), index_col=0)
    d = d.sample(frac=1, random_state=S.SEED)
    r = pd.read_csv(os.path.join(S.NR, "expression_r.csv")).set_index(["embryo", "predictor"])
    m = d.groupby("subtype")[[S.MEAS, TBX5]].mean()

    fig, box = S.figure(S.SCATTER_W, S.ROW_H[1])
    ax = box(*S.SCATTER_BOX)
    ax.scatter(d[S.MEAS], d[TBX5], s=1.6, c=[S.SUBTYPE[s] for s in d.subtype], lw=0,
               alpha=0.4, zorder=2, rasterized=True)
    ax.scatter(m[S.MEAS], m[TBX5], s=20, c=[S.SUBTYPE[s] for s in m.index], lw=0.7,
               ec="white", zorder=4)
    lo, hi = np.percentile(d[S.MEAS], [0.5, 99.5])
    pad = 0.06 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.1, d[TBX5].quantile(0.998) * 1.05)
    ax.set_xlabel(f"measured {S.DIST_LABEL}")
    ax.set_ylabel(f"{S.TARGET} (log1p CP10K)")
    ax.text(0.97, 0.97, f"$r = {r.pearson['all', TBX5]:.2f}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=S.FS_TICK)
    return S.save_svg(fig, "c_scatter")


def render_r():
    r = pd.read_csv(os.path.join(S.NR, "expression_r.csv"))
    rows = [*S.EMBRYO, "all"]
    w = r.pivot(index="embryo", columns="predictor", values="pearson").loc[rows]
    x = np.arange(len(rows)) + np.r_[np.zeros(len(S.EMBRYO)), ALL_GAP]

    fig, box = S.figure(S.R_W, S.ROW_H[1])
    ax = box(*S.R_BOX)
    for sign, k in ((-1, TBX5), (1, S.PRED)):
        ax.bar(x + sign * BAR_W / 2, w[k], width=BAR_W, color=S.PREDICTOR[k], lw=0)
    ax.set_xticks(x, [e.replace(f"{S.TARGET}_e", "") for e in rows])
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("embryo")
    ax.set_ylabel("$r$ with measured")
    S.key(box(*S.R_KEY), [(S.PREDICTOR_NAME[k], S.PREDICTOR[k], "s") for k in (TBX5, S.PRED)],
          ncol=2, fs=S.FS_KEY, ms=4.0, gap=0.04, colgap=0.14)
    return S.save_svg(fig, "d_r")


def render():
    return render_scatter(), render_r()


if __name__ == "__main__":
    print(render())
