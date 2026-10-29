"""row-2 left (panel b): per-cell-type convergence vs OT discrepancy (reads only
numerical_results/celltype_conv_vs_ot.csv).

each point is one E9.5 cell type, x = mean OT-vs-tree ancestor discrepancy and
y = mean convergence score over the leaves carrying both, each normalized by the
leaf's local cell spacing. rather than labelling the crowded corner, two fate
families are picked out against a grey backdrop -- neurons (any type with "neuron"
in the name) and the endothelial types (style.EC_CELL_TYPES) -- since these are the
most convergent, OT-divergent fates. blood-lineage types are dropped. writes
imgs/panels/celltype_conv_vs_ot.{svg,png}.

    python analysis/figures/convergence/plot_celltype_conv_vs_ot.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import style as S

PANELS = os.path.join(S.OUT, "panels")
FIG_W_IN = 1.55
FIG_H_IN = 1.5


def _family(ct):
    if ct in S.EC_CELL_TYPES:
        return "endothelial"
    if isinstance(ct, str) and "neuron" in ct:
        return "neuron"
    return "other"


def render(out_dir=PANELS):
    S.apply_rc()
    g = pd.read_csv(os.path.join(S.NR, "celltype_conv_vs_ot.csv"))
    g = g[~g["lineage"].isin(S.CONV_OT_EXCLUDE_LINEAGES)].reset_index(drop=True)
    g["family"] = g["cell_type"].map(_family)
    cmap = {"endothelial": S.ACCENT, "neuron": S.CONV_OT_NEURON, "other": S.GREY}
    sizes = 5 + 45 * np.sqrt(g["n"] / g["n"].max())            # area ~ dual-scored leaves

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    bg = g["family"] == "other"
    ax.scatter(g["ot"][bg], g["conv"][bg], s=sizes[bg], c=S.GREY, alpha=0.5, lw=0,
               zorder=1)
    for fam in ("neuron", "endothelial"):                      # highlights on top
        m = g["family"] == fam
        ax.scatter(g["ot"][m], g["conv"][m], s=sizes[m], c=cmap[fam], alpha=0.9,
                   lw=0.4, edgecolors="white", zorder=3)

    rho, _ = spearmanr(g["ot"], g["conv"])
    ax.text(0.96, 0.04, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=S.FS_ANNOT, color=S.MUTED)

    handles = [Line2D([0], [0], marker="o", ls="", ms=3.6, mfc=cmap[f], mec="none",
                      label=f) for f in ("endothelial", "neuron")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=S.FS_LEGEND,
              handletextpad=0.2, labelspacing=0.25, borderpad=0.0)

    ax.set_xlabel("OT distortion", fontsize=S.FS_LABEL, color=S.INK)
    ax.set_ylabel("convergence score", fontsize=S.FS_LABEL, color=S.INK)
    S.despine(ax)

    os.makedirs(out_dir, exist_ok=True)
    svg = os.path.join(out_dir, "celltype_conv_vs_ot.svg")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(svg.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {svg} (+ .png)  [{len(g)} types, "
          f"{int((g['family'] != 'other').sum())} highlighted]")
    return svg


if __name__ == "__main__":
    render()
