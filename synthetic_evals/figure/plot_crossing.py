"""render the crossing panels (top block) from the numerical_results caches.

  panel a  recon row, coherent variant  [truth | ours | GS | GF | FM] + fate-vs-d
  panel b  recon row, crossing variant   [truth | ours | GS | GF | FM] + fate-vs-d

each row pairs its reconstruction panels (compressed left) with the fate-accuracy
(at the second split) vs ambient-dimension line plot for that variant (right); the
coherent/crossing row label doubles as the line-plot title, and the method key is
dropped as redundant.

recon panels reuse the shared drawers in synthetic_evals.recon_figure (states are
already projected to 1-d in the cache, so _PROJ_W stays None). writes panel svgs
into imgs/. run via assemble.py.
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic_evals import recon_figure as rf
from synthetic_evals import viz
from synthetic_evals.figure import style as S

RECON_AXES_H = 0.60           # shared axes-box height so rows a and b match vertically
TITLE_PAD, XLABEL_PAD, EDGE_PAD = 0.20, 0.30, 0.05
VARIANT_LABEL = {"noncrossing": "coherent", "crossing": "crossing"}

# recon panels compress into the left block; the fate-vs-dimension line plot for
# the row's variant sits to the right (its y-label + ticks live in the gap between).
RECON_GROUP_RIGHT = 0.735
DIM_LEFT, DIM_RIGHT = 0.80, 0.99


def _recon():
    with open(os.path.join(S.NR, "crossing_recon.pkl"), "rb") as f:
        return pickle.load(f)




def render_recon(out_dir, variant, name, *, titles, xlabel):
    """one variant's row: reconstruction panels [truth | per method] (compressed
    left) beside the fate-vs-ambient-dimension line plot for the same variant."""
    S.apply_rc()
    rf._PROJ_W = None
    ev = _recon()[variant]
    obs, branch = list(ev["obs_times"]), rf._branch_rows(ev)
    order = S.MAIN_METHODS
    nM = len(order)
    top_pad = TITLE_PAD if titles else EDGE_PAD
    bot_pad = XLABEL_PAD if xlabel else EDGE_PAD
    H = RECON_AXES_H + top_pad + bot_pad
    top, bottom = 1.0 - top_pad / H, bot_pad / H
    fig = plt.figure(figsize=(S.PAGE_W_IN, H))
    gs = fig.add_gridspec(1, 1 + nM, width_ratios=[S.DYN_WIDTH] + [S.METHOD_WIDTH] * nM,
                          left=S.RECON_LEFT, right=RECON_GROUP_RIGHT, top=top, bottom=bottom,
                          wspace=0.14)
    axd = fig.add_subplot(gs[0, 0])
    rf._plot_dynamics(axd, ev["trajectories"], obs, ev["pair_times"], branch)
    axd.set_ylabel(f"{VARIANT_LABEL[variant]}\ntime", fontsize=S.FS_LABEL)
    axd.tick_params(labelbottom=xlabel)
    if titles:
        axd.set_title("ground truth", color=viz.TRUTH, fontsize=S.FS_TITLE)
    if xlabel:
        axd.set_xlabel("state")
    for c, m in enumerate(order):
        ax = fig.add_subplot(gs[0, 1 + c])
        rf._recon_panel(ax, ev, S.RECON_CACHE_KEY[m], S.METHOD_COLORS[m],
                        mode="splitter", obs=obs, branch=branch)
        ax.tick_params(labelleft=False, labelbottom=xlabel)
        if titles:
            ax.set_title({"ours": "LI"}.get(m, S.METHOD_LABELS[m]), color=S.METHOD_COLORS[m],
                         fontsize=S.FS_TITLE)
        if xlabel:
            ax.set_xlabel("state")

    fate = pd.read_csv(os.path.join(S.NR, "crossing_fate.csv"))
    fate = fate[fate["t_tgt"] == S.CROSS_SPLIT_PAIR[1]]
    gd = fig.add_gridspec(1, 1, left=DIM_LEFT, right=DIM_RIGHT, top=top, bottom=bottom)
    axm = fig.add_subplot(gd[0, 0])
    _dim_panel(axm, fate[fate["variant"] == variant], xticklabels=xlabel, xlabel=xlabel)
    return S.save_panel(fig, out_dir, name, (S.PAGE_W_IN, H))


def _dim_panel(ax, dfv, *, xticklabels, xlabel):
    """fate accuracy vs ambient dimension for one variant x transition (all four methods)."""
    dims = sorted(dfv["dim"].unique())
    x = np.log2(np.asarray(dims, float))
    for m in S.MAIN_METHODS:
        d = dfv[dfv["method"] == m].set_index("dim").reindex(dims)
        ax.plot(x, d["acc"].to_numpy(), color=S.METHOD_COLORS[m],
                marker=S.METHOD_MARKERS[m], ms=3.6, lw=1.3)
    S.thin_yaxis(ax)
    ax.set_ylim(0.0, S.ACC_VMAX + 0.03)   # from 0, not row e's 0.4: FM drops below it here
    ax.set_ylabel("fate accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims] if xticklabels else [])
    ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
    if xlabel:
        ax.set_xlabel(r"ambient dimension $d$")
    ax.grid(axis="y", which="both", color="0.92", lw=0.6, zorder=0)


if __name__ == "__main__":
    print(render_recon(S.IMGS, "noncrossing", "a_recon_coherent.svg", titles=True, xlabel=False))
    print(render_recon(S.IMGS, "crossing", "b_recon_crossing.svg", titles=False, xlabel=True))
