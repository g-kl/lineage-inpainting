"""row 1: three UMAP panels (reads only numerical_results/), left to right:

  1  the scored leaves hued by lineage (titled "coarse type")
  2  OT-vs-tree ancestor discrepancy (how far the OT ancestor lands from the tree
     ancestor) -- where the couplings disagree
  3  convergence score (spread of a leaf's neighbourhood's inferred ancestors)

2 and 3 are normalized by the leaf's local cell spacing and KNN-smoothed. each
panel is the full UMAP as a light grey backdrop with the scored E9.5 leaves drawn
on top. writes imgs/row1_umaps.{svg,png}.

    python analysis/figures/convergence/plot_umap_scores.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import style as S

PANEL_W_IN = S.PAGE_W_IN / 3
PANEL_H_IN = 1.55
DISPLAY_N = 160_000     # scored leaves drawn per scalar panel (subsampled)


def _clean(ax, title):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=S.FS_TITLE, color=S.INK, pad=3)
    ax.set_aspect("equal")
    ax.set_anchor("N")          # top-align the equal-aspect box so titles line up


def _backdrop(ax, bg):
    ax.scatter(bg["umap_x"], bg["umap_y"], s=0.6, c=S.UMAP_BG, lw=0,
               rasterized=True, zorder=0)


def _scalar_panel(ax, df, bg, col, cmap, cbar_label, clip_q):
    _backdrop(ax, bg)
    d = df.dropna(subset=[col])
    if len(d) > DISPLAY_N:
        d = d.sample(n=DISPLAY_N, random_state=int(S.SEED))
    v = d[col].to_numpy()
    lo, hi = np.percentile(v, clip_q)
    # shuffle draw order so neither extreme systematically overplots the other.
    order = np.random.default_rng(int(S.SEED)).permutation(len(v))
    sc = ax.scatter(d["umap_x"].to_numpy()[order], d["umap_y"].to_numpy()[order],
                    c=np.clip(v[order], lo, hi), cmap=cmap, s=0.8, lw=0,
                    vmin=lo, vmax=hi, alpha=0.85, rasterized=True, zorder=1)
    cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, aspect=22)
    cb.set_label(cbar_label, fontsize=S.FS_ANNOT, color=S.INK)
    cb.ax.tick_params(labelsize=S.FS_TICK - 1, colors=S.INK)
    cb.outline.set_visible(False)


def _trunc(s, n=19):
    return s if len(s) <= n else s[:n - 1] + "."


def _lineage_panel(ax, df, bg):
    _backdrop(ax, bg)
    lins = sorted(df["lineage"].dropna().unique())
    cmap = {ln: S.LINEAGE_COLORS.get(ln, S.GREY) for ln in lins}
    d = df.dropna(subset=["lineage"])
    order = np.random.default_rng(int(S.SEED)).permutation(len(d))
    ax.scatter(d["umap_x"].to_numpy()[order], d["umap_y"].to_numpy()[order],
               c=[cmap[l] for l in d["lineage"].to_numpy()[order]], s=0.8, lw=0,
               alpha=0.85, rasterized=True, zorder=1)
    handles = [Line2D([0], [0], marker="o", ls="", ms=3.0, mfc=cmap[ln], mec="none",
                      label=_trunc(ln)) for ln in lins]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=S.FS_LEGEND, handletextpad=0.3,
              labelspacing=0.28, borderpad=0.0)


def render(out_dir=None):
    out_dir = out_dir or S.OUT
    S.apply_rc()
    df = pd.read_pickle(os.path.join(S.NR, "umap_scores.pkl"))
    bg = pd.read_pickle(os.path.join(S.NR, "umap_bg.pkl"))

    fig = plt.figure(figsize=(S.PAGE_W_IN, PANEL_H_IN))
    # lineage leftmost (its legend column hangs into the gap before the OT panel);
    # the two scalar panels are wider to absorb their colorbars.
    axes = [fig.add_axes([0.00, 0.02, 0.175, 0.88]),
            fig.add_axes([0.40, 0.02, 0.215, 0.88]),
            fig.add_axes([0.70, 0.02, 0.215, 0.88])]

    _lineage_panel(axes[0], df, bg)
    _clean(axes[0], "coarse type")
    _scalar_panel(axes[1], df, bg, "ot_smooth", S.OT_CMAP, "normalized ancestor L2",
                  S.OT_CLIP_Q)
    _clean(axes[1], "OT discrepancy")
    _scalar_panel(axes[2], df, bg, "conv_smooth", S.CONV_CMAP, "normalized ancestor spread",
                  S.CLIP_Q)
    _clean(axes[2], "convergence score")

    svg = os.path.join(out_dir, "row1_umaps.svg")
    fig.savefig(svg)
    fig.savefig(svg.replace(".svg", ".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {svg} (+ .png)  ({S.PAGE_W_IN:.2f} x {PANEL_H_IN:.2f} in)")
    return svg, (S.PAGE_W_IN, PANEL_H_IN)


if __name__ == "__main__":
    render()
