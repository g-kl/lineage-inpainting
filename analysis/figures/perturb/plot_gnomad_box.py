"""panel c: gnomad LoF-constraint box (reads numerical_results/gnomad_box.csv).

box + jittered points of perturbation effect for the LoF-intolerant vs LoF-tolerant
gene bins.  writes imgs/gnomad_box.svg.  the mann-whitney p is in
numerical_results/gnomad_box_medians.md, not on the panel.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.figures.perturb import style as S

BINS = ["intolerant", "tolerant"]


def main():
    S.apply_rc()
    df = pd.read_csv(os.path.join(S.NUMERICAL, "gnomad_box.csv"))
    vals = {b: df.loc[df["bin"] == b, "effect_norm"].dropna().to_numpy() for b in BINS}

    fig, ax = plt.subplots(figsize=(S.GNOMAD_W_IN, S.BOX_H_IN))
    fig.subplots_adjust(left=0.25, right=0.97, top=0.95, bottom=0.27)

    data = [vals[b] for b in BINS]
    bp = ax.boxplot(data, widths=0.62, showfliers=False, patch_artist=True,
                    medianprops=dict(color=S.INK, linewidth=1.0),
                    whiskerprops=dict(color=S.INK, linewidth=0.7),
                    capprops=dict(color=S.INK, linewidth=0.7),
                    boxprops=dict(linewidth=0.7))
    for patch, b in zip(bp["boxes"], BINS):
        patch.set_facecolor(S.GNOMAD_COLORS[b])
        patch.set_alpha(0.6)
        patch.set_edgecolor(S.INK)
    jit = np.random.default_rng(S.seed)
    for i, v in enumerate(data, start=1):
        ax.scatter(i + jit.uniform(-0.16, 0.16, len(v)), v, s=3.0,
                   color=S.INK, alpha=0.35, linewidths=0, zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{S.GNOMAD_LABELS[b]}\n(n={len(v)})" for b, v in zip(BINS, data)],
                       fontsize=6.5)
    ax.set_ylabel("perturbation effect", fontsize=7, labelpad=2)
    ax.tick_params(axis="y", labelsize=6, length=2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(np.percentile(v, 98) for v in data) * 1.06)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    out = os.path.join(S.IMGS, "gnomad_box.svg")
    fig.savefig(out)
    fig.savefig(out.replace(".svg", ".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
