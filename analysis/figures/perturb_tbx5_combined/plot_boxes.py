"""panels b and c: the perturb figure's MGI and gnomad boxes, at this sheet's sizes.

same caches, groups and colours as perturb/plot_mgi_box.py and perturb/plot_gnomad_box.py,
drawn in the tbx5 type so they match the tbx5 scatters on the same row. the n line drops
its parentheses to fit the narrower slots.
"""
import os

import numpy as np
import pandas as pd

from analysis.figures.perturb.plot_gnomad_box import BINS
from analysis.figures.perturb.plot_mgi_box import LABELS, MIN_N, fmt_n
from analysis.figures.perturb_tbx5_combined import style as S


def save(fig, name, size):
    import matplotlib.pyplot as plt
    p = os.path.join(S.PANELS, f"{name}.svg")
    fig.savefig(p, transparent=True)
    plt.close(fig)
    print(f"wrote {p}", flush=True)
    return p, size


def boxes(ax, data, colors, widths, lw, alpha):
    bp = ax.boxplot(data, widths=widths, showfliers=False, patch_artist=True,
                    medianprops=dict(color=S.INK, linewidth=lw + 0.3),
                    whiskerprops=dict(color=S.INK, linewidth=lw),
                    capprops=dict(color=S.INK, linewidth=lw),
                    boxprops=dict(linewidth=lw))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(alpha)
        patch.set_edgecolor(S.INK)


def axes_style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=S.FS_LABEL, labelpad=2.0)
    ax.tick_params(axis="y", labelsize=S.FS_TICK, length=2)
    ax.tick_params(axis="x", length=0, labelsize=S.FS_TICK)
    S.despine(ax)


def render_mgi():
    df = pd.read_csv(os.path.join(S.PERTURB_NR, "mgi_box.csv"))
    rand_dep = np.load(os.path.join(S.PERTURB_NR, "mgi_box.npz"))["rand_dep"]

    coarse_kept = [c for c in df["coarse"].unique() if (df["coarse"] == c).sum() > MIN_N]
    frac = lambda c: df.loc[df["coarse"] == c, "perturbed_fraction"].to_numpy()
    coarse_kept.sort(key=lambda c: np.median(frac(c)), reverse=True)
    groups = ["random", "all"] + coarse_kept
    data = {"random": rand_dep, "all": df["perturbed_fraction"].to_numpy(),
            **{c: frac(c) for c in coarse_kept}}
    colors = {"random": S.MGI_REF_COLORS["random"], "all": S.MGI_REF_COLORS["all"],
              **{c: S.MGI_COARSE_COLORS.get(c, S.MUTED) for c in coarse_kept}}

    fig, box = S.figure(S.MGI_W, S.ROW1_H)
    ax = box(*S.MGI_BOX)
    boxes(ax, [data[g] for g in groups], [colors[g] for g in groups], 0.66, 0.6, 0.62)
    jit = np.random.default_rng(S.seed)
    for i, g in enumerate(groups, start=1):
        v = jit.choice(data[g], 250, replace=False) if len(data[g]) > 250 else data[g]
        ax.scatter(i + jit.uniform(-0.17, 0.17, len(v)), v, s=1.2,
                   color=S.INK, alpha=0.25, linewidths=0, zorder=3)
    ax.axhline(0, color=S.MUTED, lw=0.6, zorder=0)

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{LABELS.get(g, g)}\nn={fmt_n(len(data[g]))}" for g in groups],
                       ha="center")
    axes_style(ax, "perturbed fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    return save(fig, "mgi_box", (S.MGI_W, S.ROW1_H))


def render_gnomad():
    df = pd.read_csv(os.path.join(S.PERTURB_NR, "gnomad_box.csv"))
    data = [df.loc[df["bin"] == b, "effect_norm"].dropna().to_numpy() for b in BINS]

    fig, box = S.figure(S.GNOMAD_W, S.ROW1_H)
    ax = box(*S.GNOMAD_BOX)
    boxes(ax, data, [S.GNOMAD_COLORS[b] for b in BINS], 0.62, 0.7, 0.6)
    jit = np.random.default_rng(S.seed)
    for i, v in enumerate(data, start=1):
        ax.scatter(i + jit.uniform(-0.16, 0.16, len(v)), v, s=2.0,
                   color=S.INK, alpha=0.35, linewidths=0, zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{S.GNOMAD_LABELS[b]}\nn={len(v)}" for b, v in zip(BINS, data)])
    axes_style(ax, "perturbation effect")
    ax.set_ylim(0, max(np.percentile(v, 98) for v in data) * 1.06)
    return save(fig, "gnomad_box", (S.GNOMAD_W, S.ROW1_H))


def render():
    return render_mgi(), render_gnomad()


if __name__ == "__main__":
    render()
