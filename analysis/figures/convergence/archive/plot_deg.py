"""EC endpoint DEG heatmap (row 3 left; reads only numerical_results/).

wide source x gene strip of the top DEG_N_PER_SOURCE-per-source genes, z-scored
mean expression -- the "memory" of the origin tissue in E9.5 ECs.

    python analysis/figures/convergence/plot_deg.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import style as S

KEY = "ec"
PANELS = os.path.join(S.OUT, "panels")


def _abbr(s):
    return s.replace(" mesoderm", " meso.")


def render(out_dir=PANELS):
    S.apply_rc()
    Z = pd.read_csv(os.path.join(S.NR, f"fatebias_{KEY}_deg.csv"), index_col=0).T
    sources, genes = list(Z.index), list(Z.columns)     # sources x genes: wide strip

    fig, ax = plt.subplots(figsize=(0.088 * len(genes) + 0.65, 0.96))
    im = ax.imshow(Z.values, aspect="auto", cmap=S.DEG_CMAP, vmin=-S.DEG_CLIP,
                   vmax=S.DEG_CLIP)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=6.5, fontstyle="italic")
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([_abbr(s) for s in sources], fontsize=S.FS_TICK - 0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012, aspect=8,
                      ticks=[-S.DEG_CLIP, 0, S.DEG_CLIP])
    cb.set_label("expr z-score", fontsize=S.FS_ANNOT)
    cb.ax.tick_params(labelsize=S.FS_TICK - 1)
    cb.outline.set_visible(False)

    svg = os.path.join(out_dir, f"fatebias_{KEY}_deg.svg")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(svg.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {svg} (+ .png)")


if __name__ == "__main__":
    os.makedirs(PANELS, exist_ok=True)
    render()
