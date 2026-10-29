"""the held-out metrics panel: one column per fig 3 metric, one row per method, the mean
and its 95% interval. open dots are the paper's models (trained on every tree).
reads numerical_results/metrics.csv, writes imgs/panels/metrics.svg.

    python -m analysis.figures.supp_holdout.plot_metrics
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from analysis.figures.supp_holdout import style as S


def row_y():
    """y of each method, a gap between families, first row on top."""
    y, fam, gap = {}, None, 0.0
    for i, (m, _, f) in enumerate(S.ROWS):
        if fam is not None and f != fam:
            gap += S.FAMILY_GAP
        y[m], fam = -(i + gap), f
    return y


def render():
    S.apply_rc()
    df = pd.read_csv(os.path.join(S.NR, "metrics.csv"))
    metrics = list(dict.fromkeys(df.metric))
    y = row_y()
    span = max(y.values()) - min(y.values()) + 1.2
    ax_h = span * S.ROW_H
    w, h = S.PAGE_W_IN, S.BOTTOM + ax_h + S.TOP
    col_w = (w - S.LEFT - S.RIGHT) / (len(metrics) + (len(metrics) - 1) * S.WSPACE)

    fig = plt.figure(figsize=(w, h))
    for j, metric in enumerate(metrics):
        x0 = S.LEFT + j * col_w * (1 + S.WSPACE)
        ax = fig.add_axes([x0 / w, S.BOTTOM / h, col_w / w, ax_h / h])
        sub = df[df.metric == metric].set_index("method")
        for m, _, fam in S.ROWS:
            v, lo, hi = sub.loc[m, ["value", "lo", "hi"]]
            c = S.METHOD_COLORS[fam]
            ax.plot([lo, hi], [y[m], y[m]], color=c, lw=1.0, solid_capstyle="butt")
            ax.plot(v, y[m], "o", ms=4.5, mec=c, mew=1.0,
                    mfc="white" if m.endswith("_all") else c, zorder=3)
        ax.set_ylim(min(y.values()) - 0.6, 0.6)
        ax.set_xlim(0, S.XPAD * sub.loc[S.METHODS, ["value", "hi"]].to_numpy().max())
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
        ax.tick_params(labelsize=S.FS_TICK)
        ax.set_xlabel(sub.label.iloc[0], fontsize=S.FS_LABEL)
        ax.set_yticks(list(y.values()))
        ax.set_yticklabels([lab for _, lab, _ in S.ROWS] if j == 0 else [],
                           fontsize=S.FS_TICK)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    out = os.path.join(S.PANELS, "metrics.svg")
    fig.savefig(out, format="svg", transparent=True)
    plt.close(fig)
    return out, (w, h)


if __name__ == "__main__":
    print(render())
