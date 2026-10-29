#!/usr/bin/env python
"""row 1: the CHD panel screen, four panels.

1  heart against total share of the readout shifted, one point per gene
2  heart specificity against shift time
3  heart shifted at the full knockout against the 50% knockdown
4  the Tbx5 dose response, one curve per subtype, in the same shifted
   trajectories the row 2 curves count

the gene scatters are one grey cloud with Tbx5 picked out; every gene is named.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S

W = S.R1_W[0]
TARGET = S.TARGET.upper()


def panel():
    fig, box = S.figure(W, S.ROW1_H)
    ax = box(*S.R1_BOX)
    S.despine(ax)
    return fig, ax


def genes(ax, d, x, y):
    """one point per gene, Tbx5 picked out, every point named."""
    hi = d.index == TARGET
    c = np.where(hi, S.GENE_HI, S.GENE)
    ax.scatter(d[x], d[y], s=7, c=c, lw=0.3, edgecolors="white", zorder=3)
    S.repel(ax, d[x].to_numpy(), d[y].to_numpy(),
            [g.capitalize() for g in d.index], color=c)


def render_shifted(d):
    fig, ax = panel()
    genes(ax, d, "total_shifted", "heart_shifted")
    ax.set_xlabel("total shifted (%)", fontsize=S.FS_LABEL, labelpad=1.5)
    ax.set_ylabel("heart shifted (%)", fontsize=S.FS_LABEL, labelpad=2.0)
    ax.set_xlim(-6, 112)
    ax.set_ylim(-6, 124)
    ax.set_yticks([0, 50, 100])
    return S.save(fig, "r1_shifted", (W, S.ROW1_H))


def render_specificity(d):
    d = d.dropna(subset=["shift_time"])
    fig, ax = panel()
    genes(ax, d, "shift_time", "specificity")
    ax.set_xlabel("shift time (E)", fontsize=S.FS_LABEL, labelpad=1.5)
    ax.set_ylabel("heart specificity\n$\\log_2($heart $/$ total$)$",
                  fontsize=S.FS_LABEL, labelpad=2.0)
    ax.set_xlim(6.88, 8.42)
    ax.set_ylim(-3.4, 5.1)
    return S.save(fig, "r1_specificity", (W, S.ROW1_H))


def render_dose():
    a = pd.read_csv(f"{S.NR}/dose.csv", index_col=0)
    top = list(a.loc[a.index.min()].sort_values(ascending=False).index[:S.N_DOSE])
    fig, ax = panel()
    for c in a.columns:
        if c not in top:
            ax.plot(a.index, a[c], color=S.DOSE_WISP, lw=0.5, zorder=1)
    for c, fallback in zip(top, S.DOSE):
        ax.plot(a.index, a[c], color=S.SUBTYPE.get(c, fallback), lw=1.1, marker="o",
                ms=1.8, zorder=3, label=S.short(c))
    ax.set_xlabel(f"{S.TARGET} dose (% expression)", fontsize=S.FS_LABEL, labelpad=1.5)
    ax.set_ylabel("trajectories shifted", fontsize=S.FS_LABEL, labelpad=2.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.5, 1])
    leg = ax.legend(handlelength=0.4, handletextpad=0.3, labelspacing=0.18,
                    borderpad=0.35, ncol=1, fontsize=S.FS_KEY, loc="lower left",
                    frameon=True, fancybox=True, framealpha=1.0, facecolor="white",
                    edgecolor=S.KEY_FRAME)
    leg.get_frame().set_linewidth(0.4)
    print(f"  dose: {', '.join(top)}", flush=True)
    return S.save(fig, "r1_dose", (W, S.ROW1_H))


def render_kd(d):
    d = d.dropna(subset=["heart_shifted_50"])
    fig, ax = panel()
    ax.plot([0, 100], [0, 100], color=S.FAINT, lw=0.5, zorder=0)
    genes(ax, d, "heart_shifted_50", "heart_shifted")
    ax.set_xlabel("heart shifted (%), 50% KD", fontsize=S.FS_LABEL, labelpad=1.5)
    ax.set_ylabel("heart shifted (%), full KO", fontsize=S.FS_LABEL, labelpad=2.0)
    ax.set_xlim(-6, 112)
    ax.set_ylim(-6, 124)
    ax.set_yticks([0, 50, 100])
    return S.save(fig, "r1_kd", (W, S.ROW1_H))


def render():
    d = pd.read_csv(f"{S.NR}/metrics.csv", index_col=0)
    return (render_shifted(d), render_specificity(d), render_kd(d), render_dose())


if __name__ == "__main__":
    render()
