"""the ablation strips: one row per grid run, one column per metric, one dot per
level of the ablated axis. the baseline level is an open ink dot, the main-text run's
level a teal dot and any other learned level a tint of it; every metric is lower =
better, so left = better, on a log axis from a round floor, scaled to the rows the
strip shows, with 1-2-5 gridlines (style.scale_axis). see concept.md.

reads numerical_results/<step>/metrics_wide.csv; writes one panel svg per strip.

    python -m analysis.figures.grid_summary.plot_ablations [--step 5000]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from analysis.figures.grid_summary import style as S

# one entry per strip: the axis that varies (`variant`), its levels in legend order
# with the baseline FIRST, the axes that index the rows (outermost first) and the
# runs that take part. the innermost row axis makes the tick label, the rest become
# group brackets (outermost furthest left). every strip but the additive-baseline
# one shows the main pool alone (style.MAIN_POOL). the stop-gradient strip derives its
# axis from the pairing (id / id_sg -> the id family with the gradient flowing /
# stopped, the same for nn; the alternating Id+NN leaf run is its own family).
ABLATIONS = {
    "sampling_family": dict(
        title="sampling model: one-step vs bridges",
        variant="bridge", levels=["one_step", "cfm", "vpsde"], rows=["pairing"],
        query="aligner == 'full' and pool == @main and pairing in @shown"),
    "delta_vs_tree": dict(
        title="additive baseline (in the sampling and the imputing model)",
        variant="pool", levels=["nondelta", "dtfm"], rows=["bridge", "pairing"],
        query="aligner == 'full' and bridge in ['cfm', 'vpsde'] "
              "and pool in ['dtfm', 'nondelta'] and pairing in @shown"),
    "stopgrad": dict(
        title="stop-gradient",
        variant="grad", levels=["sg", "flow"], rows=["bridge", "pair_family"],
        query="aligner == 'full' and pool == @main"),
}
# the aligner-feature ablation is its own figure (plot_lollipops), never rows here
_STOPGRAD = {"id": ("id", "flow"), "id_sg": ("id", "sg"),
             "nn": ("nn", "flow"), "nn_sg": ("nn", "sg"),
             "idnn_leaf": ("idnn", "flow")}

CHAR_W_IN = 0.05       # ~one character of a 7 pt row label
RIGHT_IN = 0.16        # room for the last tick label
WSPACE = 0.28          # gap between metric columns, as a fraction of a column
STAGGER = 0.22         # vertical offset between the dots of one row


def _rows(wide, spec):
    df = wide.query(spec["query"], local_dict={"main": S.MAIN_POOL, "shown": S.PAIRINGS_SHOWN}).copy()
    if spec["variant"] == "grad":
        df["pair_family"] = df["pairing"].map(lambda p: _STOPGRAD[p][0])
        df["grad"] = df["pairing"].map(lambda p: _STOPGRAD[p][1])
    df["level"] = df[spec["variant"]]
    return df


def _runs(seq):
    """(first, last, value) for each run of equal consecutive values."""
    runs, start = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            runs.append((start, i - 1, seq[start]))
            start = i
    return runs


def _brackets(fig, ax, keys, n_lvl, x_inner_in, w_in):
    """row-group brackets left of the tick labels: one level per row axis that is not
    part of the tick label, outermost furthest left."""
    trans = blended_transform_factory(fig.transFigure, ax.transData)
    for lvl in range(n_lvl):
        x_in = x_inner_in - (n_lvl - 1 - lvl) * S.BRACKET_IN
        prefix = [tuple(k[:lvl + 1]) for k in keys]
        for a, b, val in _runs(prefix):
            ax.plot([x_in / w_in] * 2, [a - 0.35, b + 0.35], transform=trans,
                    color=S.INK, lw=0.7, clip_on=False, zorder=1)
            ax.text((x_in - 0.06) / w_in, (a + b) / 2, S.disp(val[-1]), rotation=90,
                    ha="center", va="center", fontsize=S.FS_LABEL, color=S.INK,
                    transform=trans, clip_on=False)


def render(out_dir, name, step, wide=None):
    """one ablation strip -> (svg path, (w_in, h_in))."""
    S.apply_rc()
    spec = ABLATIONS[name]
    if wide is None:
        wide = pd.read_csv(os.path.join(S.nr_dir(step), "metrics_wide.csv"))
    metrics = S.METRICS
    df = _rows(wide, spec)
    axes, levels = spec["rows"], spec["levels"]
    colors = {l: S.VARIANT_COLORS[l] for l in levels[1:]}     # the baseline has none
    keys = [tuple(r) for r in S.sort_rows(df[axes].drop_duplicates(), axes).to_numpy()]
    row_of = [tuple(r[a] for a in axes) for _, r in df.iterrows()]
    labels = [S.disp(k[-1]) for k in keys]
    label_w_in = max(len(l) for l in labels) * CHAR_W_IN + 0.06

    n_rows, n_cols, n_lvl = len(keys), len(metrics), len(axes) - 1
    w_in = S.PAGE_W_IN
    h_in = S.HEAD_IN + n_rows * S.ROW_H_IN + S.FOOT_IN
    left_in = 0.06 + n_lvl * S.BRACKET_IN + label_w_in
    fig = plt.figure(figsize=(w_in, h_in))
    gs = fig.add_gridspec(1, n_cols, left=left_in / w_in, right=1 - RIGHT_IN / w_in,
                          top=1 - S.HEAD_IN / h_in, bottom=S.FOOT_IN / h_in, wspace=WSPACE)
    # a top tick label may reach into the gap to the next column (whose floor label
    # starts at its own spine, so nothing comes the other way)
    gap_in = WSPACE * (w_in - left_in - RIGHT_IN) / (n_cols + (n_cols - 1) * WSPACE)
    overhang_pt = gap_in * S.PT_PER_IN - 2
    offsets = np.linspace(-STAGGER, STAGGER, len(levels)) if len(levels) > 1 else [0.0]

    axes = []
    for j, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        val = {(key, lvl): v for key, lvl, v in zip(row_of, df["level"], df[metric])
               if pd.notna(v)}
        lim = S.scale_axis(ax, "x", list(val.values()), log=True, overhang_pt=overhang_pt)
        for i, key in enumerate(keys):
            present = [k for k, lvl in enumerate(levels) if (key, lvl) in val]
            for k in present:          # a lone dot sits on the row centre
                y = i if len(present) == 1 else i + offsets[k]
                # a value at or below the floor (a U-statistic <= 0) sits on the spine
                ax.plot(max(val[key, levels[k]], lim[0]), y, ls="none", zorder=4 + (k == 0),
                        clip_on=False, **S.dot(colors.get(levels[k])))
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels if j == 0 else [])
        ax.set_title(S.metric_label(metric), fontsize=S.FS_COL, color=S.INK, pad=4)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(S.INK)
        if j == 0:
            _brackets(fig, ax, keys, n_lvl, left_in - label_w_in - 0.04, w_in)

    S.group_bars(fig, axes, metrics, y_in=S.GROUP_Y_IN, h_in=h_in)
    handles = [Line2D([], [], ls="none", label=S.disp(l), **S.dot(colors.get(l)))
               for l in levels]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles),
               bbox_to_anchor=(left_in / w_in, 1 - S.LEGEND_Y_IN / h_in), handlelength=1.5,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0)
    S.title(fig, spec["title"], left_in=left_in, w_in=w_in, h_in=h_in)
    fig.text(1 - RIGHT_IN / w_in, 0.03 / h_in, S.LEFT_CAPTION, ha="right", va="bottom",
             fontsize=S.FS_TICK, color=S.INK)
    return S.save_panel(fig, out_dir, f"{name}.svg", (w_in, h_in))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=S.STEP)
    args = ap.parse_args()
    out = os.path.join(S.imgs_dir(args.step), "panels")
    os.makedirs(out, exist_ok=True)
    for name in ABLATIONS:
        print(render(out, name, args.step))


if __name__ == "__main__":
    main()
