#!/usr/bin/env python
"""panel e: the two programs against time, per fate. a column per fate, identity on the
top row and position on the bottom, a dot every MARK time-units, and the last point of
a curve being that group's own E9.5 leaves, joined by a dashed segment across the gap
between the grid's end and E9.5.

the endothelial column keeps its two annotation groups. the two neuron columns draw
N_BINS lines across the positional score instead, hued by the same map panel d's
colorbar carries -- which is the key, since d sits next to this in the same row.

    python analysis/figures/convergence/plot_phase.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S


def track(cur, end, panel, group):
    """the group's curve on the time grid, and its E9.5 leaf mean as a separate point.
    `x` is the positional score and `y` the identity one, as computed."""
    h = cur[(cur.panel == panel) & (cur.group == group)].sort_values("time")
    e = end[(end.panel == panel) & (end.group == group)]
    return h.reset_index(drop=True), e.iloc[0]


def bin_colors(B, vmax):
    """a bin's hue is its mean positional score read off panel d's colour scale."""
    return {r.group: S.SCORE_CMAP(np.clip(r.x_mean, -vmax, vmax) / (2 * vmax) + 0.5)
            for r in B.itertuples()}


def series(ax, cur, end, spec, colors, key, ylabel, bottom):
    for g, c in colors.items():
        h, e = track(cur, end, spec["panel"], g)
        ax.plot(h.time, h[key], color=c, lw=1.3, zorder=3, label=g,
                solid_capstyle="round")
        d = h[np.isclose(h.time % S.MARK, 0, atol=1e-6)]
        ax.scatter(d.time, d[key], s=5, color=c, zorder=4, lw=0)
        ax.plot([h.time.iloc[-1], e.time], [h[key].iloc[-1], e[key]], color=c,
                lw=1.0, ls=(0, (1.5, 1.2)), zorder=3)
        ax.scatter([e.time], [e[key]], s=11, color=c, zorder=5, lw=0)
    ax.set_xlim(*S.E_XLIM)
    ax.set_xticks(S.E_XTICKS)
    ax.margins(y=0.16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=S.FS_LEGEND)
    if bottom:
        ax.set_xlabel(S.E_XLABEL, fontsize=S.FS_LEGEND)
    else:
        ax.set_xticklabels([])
        ax.set_title(spec["target"], fontsize=S.FS_ANNOT, color=S.INK, pad=2)
    if not bottom and not spec["binned"]:
        ax.legend(frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.0,
                  handlelength=1.0, labelspacing=0.25, fontsize=S.FS_LEGEND - 0.5)
    S.despine(ax)


def render():
    cur = pd.read_csv(os.path.join(S.NR, "phase_curves.csv"))
    end = pd.read_csv(os.path.join(S.NR, "phase_endpoints.csv"))
    B = pd.read_csv(os.path.join(S.NR, "phase_bins.csv"))
    vmax = pd.read_csv(os.path.join(S.NR, "score_norm.csv")).set_index("sheet").vmax
    binc = bin_colors(B, vmax["neuro"])

    fig, box = S.figure(*S.E_SIZE)
    pw, ph = S.E_PLOT
    labelled = set()
    for x, spec in zip(S.E_X, S.PHASE):
        colors = (binc if spec["binned"]
                  else {g: S.BIN_COL[g] for g in S.PROGRAMS[spec["panel"]]["x"]})
        ylabels = (("", "") if spec["sheet"] in labelled
                   else S.E_YLABEL[spec["sheet"]])
        labelled.add(spec["sheet"])
        for y, key, ylabel in zip(S.E_Y, S.E_ROWS, ylabels):
            series(box(x, y, pw, ph), cur, end, spec, colors, key, ylabel,
                   y == S.E_Y[-1])
    return S.save(fig, "phase")


if __name__ == "__main__":
    render()
