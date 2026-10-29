"""cell-type transition-graph similarity across the grid: the run x run cosine
similarity of each run's (type, t) -> (type, t + dt) transition graph under the
sampling model's coupling, over every run some grid figure shows. rows / columns
are grouped by sampling model (outer brackets) and imputing model (inner); the
pairing is on the ticks. the aligner ablation is a trailing group of its own, the
aligner feature set on its ticks. the matrix is symmetric with a unit diagonal, so
only its strictly lower triangle is drawn; the column labels sit along the staircase
in the empty half, one per row, and the colourbar in the empty corner.

reads numerical_results/<step>/{similarity,configs}.csv; writes one panel svg.

    python -m analysis.figures.grid_summary.plot_similarity [--step 5000]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from analysis.figures.grid_summary import style as S
from analysis.figures.grid_summary.plot_ablations import _runs

LEFT_IN, RIGHT_IN = 1.32, 0.96     # brackets + row labels | the column labels past the staircase
TOP_IN, BOT_IN = 0.40, 0.10        # title | nothing (the column labels live in the empty half)
TICK_IN = 0.86                     # the longest label ("nearest neighbor (sg)") measures ~0.78 in
BR_INNER_IN, BR_OUTER_IN = 0.06, 0.24   # bracket offsets past the row labels
CBAR_BOX = (0.74, 0.66, 0.035, 0.30)    # the colourbar, in heatmap-axes fractions (the empty upper right)


def render(out_dir, step):
    S.apply_rc()
    nr = S.nr_dir(step)
    sim = pd.read_csv(os.path.join(nr, "similarity.csv"), index_col=0)
    configs = pd.read_csv(os.path.join(nr, "configs.csv")).set_index("cfg")
    ax_of = configs.loc[sim.index]
    n = len(sim)
    C = sim.to_numpy()

    w_in = S.PAGE_W_IN
    heat_in = w_in - LEFT_IN - RIGHT_IN
    h_in = TOP_IN + heat_in + BOT_IN
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([LEFT_IN / w_in, BOT_IN / h_in, heat_in / w_in, heat_in / h_in])
    # colour by the log of the dissimilarity 1 - cosine: the plain Tree Tx runs sit
    # at 0.2-0.5 against everything else and would otherwise squeeze every pair of
    # main-pool runs (0.97-1) into one shade; the colourbar reads in similarity
    off = C[np.triu_indices(n, k=1)]
    D = -np.log10(np.clip(1.0 - C, 1e-6, None))
    d_lo, d_hi = -np.log10(1.0 - off.min()), -np.log10(1.0 - off.max())
    # the matrix is symmetric with a unit diagonal: draw the strictly lower
    # triangle, runs 1..n-1 as rows against runs 0..n-2 as columns
    tri = np.full_like(D, np.nan)
    low = np.tril_indices(n, k=-1)
    tri[low] = D[low]
    tri = tri[1:, :-1]
    m = n - 1
    cmap = S.SIM_CMAP.copy()
    cmap.set_bad(alpha=0)
    im = ax.imshow(np.ma.masked_invalid(tri), cmap=cmap, vmin=d_lo, vmax=d_hi,
                   aspect="equal", interpolation="nearest")
    ablation = (ax_of["aligner"] != "full").to_numpy()
    labels = [S.disp(al if a else pr) for a, al, pr in zip(ablation, ax_of["aligner"], ax_of["pairing"])]
    outer = [("ablation" if a else b) for a, b in zip(ablation, ax_of["bridge"])]
    inner = [(("ablation", None) if a else (b, p))
             for a, b, p in zip(ablation, ax_of["bridge"], ax_of["pool"])]
    ax.set_xticks([])
    ax.set_yticks(range(m))
    ax.set_yticklabels(labels[1:], fontsize=S.FS_TICK)
    ax.tick_params(length=0, pad=2)
    # column labels along the staircase: column j's topmost cell is row j, so its
    # label sits just right of it, one label per row, no rotation
    for j in range(m):
        ax.text(j + 0.65, j, labels[j], ha="left", va="center", fontsize=S.FS_TICK,
                color=S.INK, clip_on=False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # group separators over the filled triangle only. the boundary between runs
    # i-1 and i: a horizontal line at trimmed row i-1's top edge out to the
    # diagonal, a vertical line at column i-1's right edge down to the bottom
    for i in range(1, n):
        if outer[i] != outer[i - 1]:
            style = dict(color=S.INK, lw=0.8)
        elif inner[i] != inner[i - 1]:
            style = dict(color="white", lw=0.5, ls=(0, (1.5, 1.5)))
        else:
            continue
        ax.plot([-0.5, i - 0.5], [i - 1.5, i - 1.5], clip_on=False, **style)
        ax.plot([i - 0.5, i - 0.5], [i - 1.5, m - 0.5], clip_on=False, **style)

    # group brackets on the rows (the column groups are read off the separators);
    # sampling model outer, imputing model inner (none for the ablation group)
    t_left = blended_transform_factory(fig.transFigure, ax.transData)
    display = {"ablation": "aligner ablation"}
    for level, off_in, fs in ((inner, BR_INNER_IN, S.FS_TICK), (outer, BR_OUTER_IN, S.FS_LABEL)):
        x_in = LEFT_IN - TICK_IN - off_in            # left of the row labels
        for a, b, val in _runs(level[1:]):
            key = val[-1] if isinstance(val, tuple) else val
            if key is None:
                continue
            ax.plot([x_in / w_in] * 2, [a - 0.3, b + 0.3], transform=t_left, color=S.INK,
                    lw=0.8, clip_on=False)
            ax.text((x_in - 0.05) / w_in, (a + b) / 2, display.get(key, S.disp(key)), rotation=90,
                    ha="center", va="center", fontsize=fs, color=S.INK, transform=t_left,
                    clip_on=False)

    cax = ax.inset_axes(CBAR_BOX)
    cb = fig.colorbar(im, cax=cax)
    sims = [t for t in (0.2, 0.5, 0.9, 0.99, 0.999) if off.min() <= t <= off.max()]
    cb.set_ticks([-np.log10(1.0 - t) for t in sims])
    cb.set_ticklabels([f"{t:g}" for t in sims])
    cb.set_label("cosine similarity", fontsize=S.FS_LABEL, color=S.INK)
    cb.ax.tick_params(labelsize=S.FS_TICK, width=0.6, length=2)
    cb.outline.set_linewidth(0.6)

    S.title(fig, "transition-graph similarity", left_in=0.06, w_in=w_in, h_in=h_in)
    return S.save_panel(fig, out_dir, "ct_similarity.svg", (w_in, h_in))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=S.STEP)
    args = ap.parse_args()
    out = os.path.join(S.imgs_dir(args.step), "panels")
    os.makedirs(out, exist_ok=True)
    print(render(out, args.step))


if __name__ == "__main__":
    main()
