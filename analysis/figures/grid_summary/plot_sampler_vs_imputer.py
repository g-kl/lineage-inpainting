"""the sampling model against the imputing model as the coupler: for every run, the
marginal distortion and its cell-type TV with the splitter doing the pool-back
against the same samples pooled back by the pool model applied once, and each
assignment metric scored on the sampling-side coupling (the splitter used as the
pooler, the main-text figure's coupling) against the same metric on the
imputing-side coupling (the pool model applied once). one point per run over the
main pool's full-aligner runs, one square panel per metric with both axes from 0,
the diagonal as the reference: below it the sampling model's coupling is the better
one. the dots are the strips' key: open ink one-step, tint CFM, teal VP-SDE.

reads numerical_results/<step>/metrics_wide.csv; writes one panel svg.

    python -m analysis.figures.grid_summary.plot_sampler_vs_imputer [--step 5000]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.figures.grid_summary import style as S

PANEL_IN = 0.80          # square panel side (six across the page)
LEFT_IN, RIGHT_IN = 0.42, 0.08
HEAD_IN, FOOT_IN = 0.56, 0.40
GAP_IN = 0.24


def render(out_dir, step, wide=None):
    S.apply_rc()
    if wide is None:
        wide = pd.read_csv(os.path.join(S.nr_dir(step), "metrics_wide.csv"))
    wide = wide[(wide["aligner"] == "full") & (wide["pool"] == S.MAIN_POOL)
                & wide["pairing"].isin(S.PAIRINGS_SHOWN)]
    metrics = S.PAIRED_METRICS
    w_in = S.PAGE_W_IN
    h_in = HEAD_IN + PANEL_IN + FOOT_IN
    fig = plt.figure(figsize=(w_in, h_in))
    n = len(metrics)
    span = w_in - LEFT_IN - RIGHT_IN
    gs = fig.add_gridspec(1, n, left=LEFT_IN / w_in, right=1 - RIGHT_IN / w_in,
                          top=1 - HEAD_IN / h_in, bottom=FOOT_IN / h_in,
                          wspace=GAP_IN / ((span - (n - 1) * GAP_IN) / n))

    for j, m in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        x, y = wide[f"cpl_{m}"].to_numpy(), wide[m].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        both = np.r_[x[ok], y[ok]]          # one range for both axes: the diagonal is the reference
        lim = S.scale_axis(ax, "x", both, overhang_pt=6)
        S.scale_axis(ax, "y", both)
        ax.plot(lim, lim, color=S.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for bridge in S.BRIDGES:
            sel = ok & (wide["bridge"] == bridge).to_numpy()
            ax.plot(x[sel], y[sel], ls="none", zorder=3, **S.dot(S.VARIANT_COLORS.get(bridge)))
        ax.set_box_aspect(1)
        ax.set_title(S.metric_label(m), fontsize=S.FS_COL, color=S.INK, pad=4)
        if j == 0:
            ax.set_ylabel("sampling model", fontsize=S.FS_LABEL, labelpad=3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text((LEFT_IN + span / 2) / w_in, 0.05 / h_in, "imputing model", ha="center",
             va="bottom", fontsize=S.FS_LABEL, color=S.INK)

    handles = [Line2D([], [], ls="none", label=S.disp(b), **S.dot(S.VARIANT_COLORS.get(b)))
               for b in S.BRIDGES]
    fig.legend(handles=handles, loc="upper left", ncol=3,
               bbox_to_anchor=(LEFT_IN / w_in, 1 - S.LEGEND_Y_IN / h_in), handletextpad=0.4,
               columnspacing=1.2, borderaxespad=0)
    S.title(fig, "sampling vs imputing model as the coupler", left_in=LEFT_IN, w_in=w_in, h_in=h_in)
    return S.save_panel(fig, out_dir, "sampler_vs_imputer.svg", (w_in, h_in))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=S.STEP)
    args = ap.parse_args()
    out = os.path.join(S.imgs_dir(args.step), "panels")
    os.makedirs(out, exist_ok=True)
    print(render(out, args.step))


if __name__ == "__main__":
    main()
