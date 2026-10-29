#!/usr/bin/env python
"""row 2 panels 1 and 3: the shift curves and the split panel.

1  when the knockout has shifted a readout trajectory, per lateral plate subtype. the
   N_SHIFT most shifted at the readout are coloured and named; the rest of the lineage
   goes behind in grey.
3  over the fate trajectories of the clones that make both fates: the share biased to
   each fate by each time, against the share carrying Tbx5 by then, so one curve per
   (fate, event). a colour each, the two Tbx5 curves dashed as well.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S


def top_subtypes():
    """the subtypes panel 1 colours, most shifted at the readout first."""
    d = pd.read_csv(f"{S.NR}/shift_readout.csv", index_col=0)
    return list(d.index[:S.N_SHIFT])


def render_shift(top):
    p = pd.read_csv(f"{S.NR}/shift_curves.csv")
    p = p[p.leaves >= S.MIN_LEAVES]
    fig, box = S.figure(S.R2_W[0], S.ROW2_H)
    ax = box(*S.CURVE_BOX)
    for s, g in p.groupby("subtype"):
        if s not in top:
            ax.plot(g.time, g.frac_shifted, color=S.FAINT, lw=0.5, zorder=1)
    for s in top:
        g = p[p.subtype == s]
        ax.plot(g.time, g.frac_shifted, color=S.SUBTYPE[s], lw=1.0, zorder=2)
    ax.set_xlabel("time (E)", fontsize=S.FS_LABEL, labelpad=1.5)
    ax.set_xlim(p.time.min(), p.time.max())
    ax.set_xticks([7.5, 8.0, 8.5, 9.0])
    ax.set_ylabel("trajectories shifted", fontsize=S.FS_LABEL, labelpad=2.0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    S.despine(ax)
    S.key(box(*S.CURVE_KEY), [(S.short(s), S.SUBTYPE[s]) for s in top], ncol=2,
          ms=2.2, box=True)
    return S.save(fig, "r2_shift", (S.R2_W[0], S.ROW2_H))


def ecdf(v, grid):
    """the share of every trajectory whose event has happened by each point of grid.

    the divisor is every trajectory, the ones the event never reaches included, so a
    curve plateaus under 1 by exactly the share it never reaches.
    """
    return np.searchsorted(np.sort(v[~np.isnan(v)]), grid, "right") / len(v)


def render_split():
    d = pd.read_csv(f"{S.NR}/split_onset.csv")
    grid = np.linspace(*S.SPLIT_X, S.SPLIT_GRID)
    w = S.R2_W[2]
    fig, box = S.figure(w, S.ROW2_H)

    ax = box(*S.SPLIT_BOX)
    for (fate, col), (_, c, ls) in S.SPLIT.items():
        v = d[col][d.fate == fate].to_numpy()
        ax.plot(grid, ecdf(v, grid), color=c, lw=1.0, ls=ls, zorder=2)
    ax.set_xlabel("time (E)", fontsize=S.FS_LABEL, labelpad=1.5)
    ax.set_ylabel("share of trajectories", fontsize=S.FS_LABEL, labelpad=2.0)
    ax.set_xlim(*S.SPLIT_X)
    ax.set_xticks([7.5, 8.0, 8.5, 9.0])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    S.despine(ax)

    # a colour each names a curve on its own, so the key is four line samples in one
    # column, carrying each curve's dash and centred under the axes rather than the panel
    S.key(box(*S.SPLIT_KEY), list(S.SPLIT.values()), ncol=1, line=S.KEY_LINE, box=True)

    print(f"  split: {len(d):,} fate trajectories over {d.clone.nunique()} clones",
          flush=True)
    for fate, g in d.groupby("fate"):
        a, b = [grid[np.searchsorted(ecdf(g[c].to_numpy(), grid), S.SPLIT_AT)]
                for c in ("t_split", "t_on")]
        print(f"    {fate}: {len(g):,}, gap {b - a:.2f} at {S.SPLIT_AT:g}, "
              f"{g.t_on.isna().mean():.1%} never Tbx5+", flush=True)
    return S.save(fig, "r2_split", (w, S.ROW2_H))


def render():
    top = top_subtypes()
    print(f"  coloured: {', '.join(top)}", flush=True)
    return render_shift(top), render_split()


if __name__ == "__main__":
    render()
