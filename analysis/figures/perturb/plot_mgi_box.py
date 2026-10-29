"""panel b: MGI perturbed-fraction box (reads numerical_results/mgi_box.{csv,npz}).

box + jittered points of the KO's perturbed fraction: random pairs and all annotated
pairs as references, then the coarse anatomical contexts with more than MIN_N
associations (ordered by median perturbed fraction).  writes imgs/mgi_box.svg.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.figures.perturb import style as S

MIN_N = 100
LABELS = {"random": "random\npairs", "all": "all\nannot.", "CNS / brain": "brain"}


def fmt_n(n):
    """compact count: round powers of ten >= 10000 as 10^k, else the integer."""
    if n >= 10000:
        e = int(round(np.log10(n)))
        if 10 ** e == n:
            return rf"$10^{{{e}}}$"
    return str(n)


def main():
    S.apply_rc()
    df = pd.read_csv(os.path.join(S.NUMERICAL, "mgi_box.csv"))
    rand_dep = np.load(os.path.join(S.NUMERICAL, "mgi_box.npz"))["rand_dep"]

    coarse_kept = [c for c in df["coarse"].unique() if (df["coarse"] == c).sum() > MIN_N]
    frac = lambda c: df.loc[df["coarse"] == c, "perturbed_fraction"].to_numpy()
    coarse_kept.sort(key=lambda c: np.median(frac(c)), reverse=True)
    groups = ["random", "all"] + coarse_kept
    data = {"random": rand_dep, "all": df["perturbed_fraction"].to_numpy(),
            **{c: frac(c) for c in coarse_kept}}
    colors = {"random": S.MGI_REF_COLORS["random"], "all": S.MGI_REF_COLORS["all"],
              **{c: S.MGI_COARSE_COLORS.get(c, S.MUTED) for c in coarse_kept}}

    fig, ax = plt.subplots(figsize=(S.MGI_W_IN, S.BOX_H_IN))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.27)

    box_data = [data[g] for g in groups]
    bp = ax.boxplot(box_data, widths=0.66, showfliers=False, patch_artist=True,
                    medianprops=dict(color=S.INK, linewidth=0.9),
                    whiskerprops=dict(color=S.INK, linewidth=0.6),
                    capprops=dict(color=S.INK, linewidth=0.6),
                    boxprops=dict(linewidth=0.6))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.62)
        patch.set_edgecolor(S.INK)
    jit = np.random.default_rng(S.seed)
    for i, g in enumerate(groups, start=1):
        v = jit.choice(data[g], 250, replace=False) if len(data[g]) > 250 else data[g]
        ax.scatter(i + jit.uniform(-0.17, 0.17, len(v)), v, s=1.6,
                   color=S.INK, alpha=0.25, linewidths=0, zorder=3)
    ax.axhline(0, color=S.MUTED, lw=0.6, zorder=0)

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{LABELS.get(g, g)}\n(n={fmt_n(len(data[g]))})" for g in groups],
                       fontsize=6, ha="center")
    ax.set_ylabel("perturbed fraction", fontsize=7, labelpad=2)
    ax.tick_params(axis="y", labelsize=6, length=2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-0.05, 1.05)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    out = os.path.join(S.IMGS, "mgi_box.svg")
    fig.savefig(out)
    fig.savefig(out.replace(".svg", ".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
