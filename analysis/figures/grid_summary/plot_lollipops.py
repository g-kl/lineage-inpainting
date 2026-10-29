"""the aligner-feature ablation: the main-text run with the StratifiedFeatureAligner's
feature set ablated -- marginal only, + path, + common ancestor, + both (the main-text
run) -- at its raw metric values as horizontal lollipops, one small panel per metric,
exactly as the main-text metrics row draws them
(analysis/figures/summary_qc/plot_metrics). the value is annotated above each dot;
every metric is lower = better, so the shorter lollipop wins.

reads numerical_results/<step>/metrics_wide.csv; writes one panel svg.

    python -m analysis.figures.grid_summary.plot_lollipops [--step 5000]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.grid_summary import style as S

LOLLIPOPS = {
    "aligner_ablation": dict(
        title="aligner feature ablation",
        rows=[(a, {**S.HERO, "aligner": a}) for a in S.ALIGNERS],
        colors=S.ALIGNER_COLORS, labels={a: S.disp(a) for a in S.ALIGNERS}),
}

LABEL_W_IN = 0.95      # "+ Path + CA (ours)"
RIGHT_IN = 0.06
HEAD_IN = 0.66         # title + group overbars + column titles
GROUP_Y_IN = 0.30


def _value(wide, where, metric):
    m = np.ones(len(wide), bool)
    for k, v in where.items():
        m &= (wide[k] == v).to_numpy()
    hit = wide.loc[m, metric]
    return float(hit.iloc[0]) if len(hit) else np.nan


def render(out_dir, name, step, wide=None):
    """one lollipop strip -> (svg path, (w_in, h_in))."""
    S.apply_rc()
    spec = LOLLIPOPS[name]
    if wide is None:
        wide = pd.read_csv(os.path.join(S.nr_dir(step), "metrics_wide.csv"))
    metrics = S.METRICS
    rows = spec["rows"]
    n_rows, n_cols = len(rows), len(metrics)
    w_in = S.PAGE_W_IN
    h_in = HEAD_IN + n_rows * S.LOLLI_ROW_H_IN + 0.14
    left_in = 0.06 + LABEL_W_IN
    fig = plt.figure(figsize=(w_in, h_in))
    gs = fig.add_gridspec(1, n_cols, left=left_in / w_in, right=1 - RIGHT_IN / w_in,
                          top=1 - HEAD_IN / h_in, bottom=0.14 / h_in, wspace=0.30)

    axes = []
    for j, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        vals = [_value(wide, where, metric) for _, where in rows]
        finite = [v for v in vals if np.isfinite(v)]
        vmax = max(finite) if finite else 1.0
        xmax = vmax * 1.35
        for i, ((key, _), v) in enumerate(zip(rows, vals)):
            c = spec["colors"][key]
            if np.isfinite(v):
                ax.hlines(i, 0, v, color=c, lw=1.5, alpha=0.85, zorder=2)
                ax.plot(v, i, "o", ms=4.2, mfc=c, mec="white", mew=0.6, zorder=4)
                # values near the axis are left-aligned so the label clears the spine
                near = v < 0.22 * xmax
                ax.annotate(f"{v:.3g}", (v, i), textcoords="offset points",
                            xytext=(-3 if near else 0, 4), ha="left" if near else "center",
                            va="bottom", fontsize=S.FS_SUB, color=S.INK)
            else:
                ax.text(0.5, i, "n/a", transform=ax.get_yaxis_transform(), ha="center",
                        va="center", fontsize=S.FS_SUB, color=S.GREY, style="italic")
        ax.set_xlim(0, xmax)
        ax.set_ylim(n_rows - 0.4, -0.7)
        ax.set_xticks([])
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([spec["labels"][k] for k, _ in rows] if j == 0 else [])
        ax.set_title(S.metric_label(metric), fontsize=S.FS_COL, color=S.INK, pad=4)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(S.INK)
        ax.tick_params(axis="both", length=0)

    S.group_bars(fig, axes, metrics, y_in=GROUP_Y_IN, h_in=h_in)
    S.title(fig, spec["title"], left_in=left_in, w_in=w_in, h_in=h_in)
    return S.save_panel(fig, out_dir, f"{name}.svg", (w_in, h_in))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=S.STEP)
    args = ap.parse_args()
    out = os.path.join(S.imgs_dir(args.step), "panels")
    os.makedirs(out, exist_ok=True)
    for name in LOLLIPOPS:
        print(render(out, name, args.step))


if __name__ == "__main__":
    main()
