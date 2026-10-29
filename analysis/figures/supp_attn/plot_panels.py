#!/usr/bin/env python
"""the three panels of the attention supplemental figure, from numerical_results/.

    a  the learned bias scales gamma, ranked
    b  share of attention on the observed leaves, over the rollout
    c  one subtree, one head, four interpolant times

each render_* writes one svg into imgs/panels/ and returns (path, (w_in, h_in)).

    python -m analysis.figures.supp_attn.plot_panels
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures.supp_attn import style as S


def panel_axes(w, h, margins):
    """a (w, h) inch figure holding one axes inset by (left, bottom, right, top) inches."""
    left, bottom, right, top = margins
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - bottom - top) / h])
    return fig, ax


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def render_gamma(scales):
    fig, ax = panel_axes(S.TOP_W, S.TOP_H, S.TOP_AX)
    s = scales.sort_values("scale").reset_index(drop=True)
    x = np.arange(len(s))
    colors = [S.LAYER[i] for i in s.layer]
    ax.vlines(x, 0, s.scale, color=colors, lw=1.4)
    ax.scatter(x, s.scale, s=30, c=colors, lw=0, zorder=3)
    ax.axhline(0, color=S.INK, lw=0.7)
    ax.set_xticks([])
    ax.set_xlabel("attention head")
    ax.set_ylabel("$\\gamma$  (bias per day)")
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=0, marker="o", ms=4,
                                  label=f"layer {i}") for i, c in enumerate(S.LAYER)],
              frameon=False, loc="upper left", handlelength=0.6, ncol=2,
              columnspacing=0.8, labelspacing=0.2, borderpad=0, handletextpad=0.4)
    despine(ax)
    return S.save_svg(fig, "gamma")


def render_share(r):
    fig, ax = panel_axes(S.TOP_W, S.TOP_H, S.TOP_AX)
    for (layer, head), g in r.groupby(["layer", "head"]):
        ax.plot(g.s, g.on_observed, color=S.LAYER[layer], lw=1.1, alpha=0.85,
                marker="o", ms=2.5)
    uniform = r.uniform.iloc[0]
    ax.axhline(uniform, color=S.INK, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.02, uniform + 0.03, "uniform", transform=ax.get_yaxis_transform(),
            fontsize=S.FS_TICK, color=S.MUTED, bbox=dict(fc="white", ec="none", pad=0.8))
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    ax.set_xlabel("interpolant time $s$")
    ax.set_ylabel("attention on observed cells")
    despine(ax)
    return S.save_svg(fig, "share")


def render_example(ex, layer, head):
    w, h = S.PAGE_W, S.C_TOP + S.C_HM + S.C_BOTTOM
    fig = plt.figure(figsize=(w, h))
    order = np.lexsort((np.arange(len(ex["node_times"])), ex["node_times"]))
    obs = ex["observed"][order]
    n = len(order)
    attn = ex["attn"][:, layer, head][:, order][:, :, order]

    axes = []
    for i, k in enumerate(S.STEPS):
        x0 = S.C_LEFT + i * (S.C_HM + S.C_GAP)
        ax = fig.add_axes([x0 / w, S.C_BOTTOM / h, S.C_HM / w, S.C_HM / h])
        im = ax.imshow(np.log2(attn[k] * n + 1e-6), cmap=S.DIVERGE, vmin=-S.LOG2_LIM,
                       vmax=S.LOG2_LIM, interpolation="nearest", aspect="auto")
        for keep, color in ((~obs, S.MASKED), (obs, S.OBS)):
            ax.scatter(np.where(keep)[0], np.full(keep.sum(), 1.03), marker="|", s=9,
                       lw=0.7, color=color, clip_on=False,
                       transform=ax.get_xaxis_transform())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"$s = {ex['s'][k]:.2f}$", labelpad=4)
        for sp in ax.spines.values():
            sp.set_color(S.MUTED)
        axes.append(ax)

    axes[0].set_ylabel("query node")
    strip = axes[0].get_xaxis_transform()
    axes[0].text(-0.5, 1.07, "masked", transform=strip, fontsize=S.FS_SMALL,
                 color=S.MASKED, ha="left", va="bottom")
    axes[0].text(np.where(obs)[0].mean(), 1.07, "observed", transform=strip,
                 fontsize=S.FS_SMALL, color=S.OBS, ha="center", va="bottom")
    fig.text(0.34 / w, 1 - 0.21 / h, f"layer {layer}, head {head}", fontsize=S.FS_TICK,
             va="baseline")
    block = len(S.STEPS) * S.C_HM + (len(S.STEPS) - 1) * S.C_GAP
    fig.text((S.C_LEFT + block / 2) / w, 0.04 / h, "key node", ha="center", va="bottom")

    cax = fig.add_axes([(S.C_LEFT + block + S.C_CB_GAP) / w, S.C_BOTTOM / h,
                        S.C_CB_W / w, S.C_HM / h])
    cb = fig.colorbar(im, cax=cax, ticks=[-S.LOG2_LIM, 0, S.LOG2_LIM])
    cb.set_label("attention / uniform ($\\log_2$)", fontsize=S.FS_TICK)
    cb.ax.tick_params(labelsize=S.FS_SMALL)
    cb.outline.set_visible(False)
    return S.save_svg(fig, "example")


def render():
    S.apply_rc()
    scales = pd.read_csv(os.path.join(S.NR, "scales.csv"))
    r = pd.read_csv(os.path.join(S.NR, "rollout.csv"))
    ex = np.load(os.path.join(S.NR, "example.npz"))
    return [render_gamma(scales), render_share(r), render_example(ex, *S.EXAMPLE_HEAD)]


if __name__ == "__main__":
    for path, (w, h) in render():
        print(f"wrote {path}  ({w:.2f} x {h:.2f} in)")
