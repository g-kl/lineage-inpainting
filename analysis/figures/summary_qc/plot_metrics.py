"""row 2 (panel b): six coupling-quality metrics, lineage inpainting vs the two
GENOT baselines, as horizontal lollipops (one small panel per metric).

reads (at plot time, so it tracks the live caches):
  numerical_results/clonal_distortion.csv     -> clonal distortion (cycle consist.)
  numerical_results/marginal_distortion.csv   -> marginal distortion
  numerical_results/assignment_metrics.csv    -> oscillation / linkage / germ crossing
  numerical_results/hox_consistency.csv       -> Hox-code distortion

every metric is "lower = better", so the shorter lollipop is the better method;
lineage inpainting (teal) sits on top of each panel. method names are
the y labels of the leftmost panel only -- shared vertical order, no legend box.
writes panels/B_metrics.svg.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.summary_qc import style as S

PANEL_W_IN = S.PAGE_W_IN
PANEL_H_IN = 1.45
CI = 1.96
FS_B = 8.5     # row-2 text runs small: lots of labels in a short strip


def _stats(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    m = v.mean()
    if v.size < 2:
        return m, m, m
    half = CI * v.std(ddof=1) / np.sqrt(v.size)
    return m, m - half, m + half


def _metric_values(nr):
    """{metric_key: (label, {method: (point, lo, hi)})} in display order."""
    clo = pd.read_csv(os.path.join(nr, "clonal_distortion.csv"))
    mar = pd.read_csv(os.path.join(nr, "marginal_distortion.csv"))
    hox = pd.read_csv(os.path.join(nr, "hox_consistency.csv"))
    am = pd.read_csv(os.path.join(nr, "assignment_metrics.csv"))
    am["tag"] = am["tag"].fillna("")

    def from_df(df, col):
        return {m: _stats(g[col].to_numpy()) for m, g in df.groupby("method")}

    def from_am(metric, tagged, drop_tags=()):
        sub = am[am["metric"] == metric]
        out = {}
        for m, g in sub.groupby("method"):
            rows = g[g["tag"] != ""] if tagged else g[g["tag"] == ""]
            rows = rows[~rows["tag"].isin(drop_tags)]
            out[m] = _stats(rows["value"].to_numpy())
        return out

    return [
        ("clonal", "clonal\ndistortion", from_df(clo, "energy_distance")),
        ("marginal", "marginal\ndistortion", from_df(mar, "energy_distance")),
        ("oscillation", "lineage\noscillation", from_am("lineage_oscillation_fraction", False)),
        # only the two later transitions are kept: at the earlier cuts the
        # real-real linkage baseline is near zero, so the normalised distortion
        # is ill-conditioned.
        ("linkage", "linkage\ndistortion", from_am("linkage_distortion", True,
                                                    drop_tags=("t8.5_to_t8.0",
                                                               "t8.0_to_t7.5"))),
        # the pooled row (tag "") -- one rate over every parent->child link. a mean of
        # the bin_* rows would weigh a one-link bin like a 200k-link one.
        ("germ", "germ-layer\ncrossing", from_am("germ_layer_crossing", False)),
        # Hox-code consistency: ancestor/descendant correlation of a statically
        # expressed Hox gene, scored per (cell type, gene); distortion = 1 - r so
        # it keeps the "shorter = better" convention with the other panels.
        ("hox", "Hox-code\ndistortion", from_df(hox, "distortion")),
    ]


def _fmt(v):
    return f"{v:.3g}"


def render(out_dir):
    S.apply_rc()
    metrics = _metric_values(S.NR)
    methods = S.METHODS
    y = {m: len(methods) - 1 - i for i, m in enumerate(methods)}   # model on top

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(PANEL_W_IN, PANEL_H_IN))
    fig.subplots_adjust(left=0.17, right=0.99, top=0.80, bottom=0.07, wspace=0.30)

    for ax, (key, label, data) in zip(axes, metrics):
        vpt = max(pt for pt, _, _ in data.values() if np.isfinite(pt))
        for m in methods:
            pt, lo, hi = data[m]
            c = S.METHOD_COLORS[m]
            yy = y[m]
            ax.hlines(yy, 0, pt, color=c, lw=1.5, alpha=0.85, zorder=2)
            ax.plot(pt, yy, "o", ms=5, mfc=c, mec="white", mew=0.6, zorder=4)
            ax.annotate(_fmt(pt), (pt, yy), textcoords="offset points",
                        xytext=(0, 6), va="bottom", ha="center",
                        fontsize=FS_B, color=S.INK)

        ax.set_title(label, fontsize=FS_B + 0.5, color=S.INK, pad=4)
        ax.set_xlim(0, vpt * 1.35)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.set_xticks([])
        ax.set_yticks(list(y.values()))
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color(S.INK)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0)

    axes[0].set_yticklabels([S.METHOD_LABELS[m] for m in methods],
                            fontsize=FS_B, color=S.INK)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.text(0.995, 0.005, r"shorter $=$ better (every metric)", ha="right",
             va="bottom", fontsize=FS_B - 1, color="#777", style="italic")

    out = os.path.join(out_dir, "B_metrics.svg")
    fig.savefig(out)
    plt.close(fig)
    return out, (PANEL_W_IN, PANEL_H_IN)


if __name__ == "__main__":
    os.makedirs(os.path.join(S.OUT, "panels"), exist_ok=True)
    print(render(os.path.join(S.OUT, "panels")))
