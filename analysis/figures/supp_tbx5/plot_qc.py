"""a and b: the crispant qc. reads numerical_results/{snp_umap,qc_cells}.parquet.

a  the SNP demultiplexing umap, every cell with a SNP profile. the kept cells take their
   embryo's hue and everything the demultiplexing or the doublet call threw out goes
   down first in grey. two short bars in the corner name the axes.
b  per embryo: cells recovered, then UMIs and genes per recovered cell. boxes are the
   quartiles, whiskers the 5th and 95th percentiles.

    python -m analysis.figures.supp_tbx5.plot_qc
"""
import os

import numpy as np
import pandas as pd

from analysis.figures.supp_tbx5 import style as S

STRIPS = (("umis", "UMIs per cell"), ("genes", "genes per cell"))


def label(e):
    return e.replace(f"{S.TARGET}_", "")


def axis_bars(ax, name):
    """two short bars off the map's bottom left corner, `name`1 and `name`2 beside them."""
    fig = ax.figure
    pos = ax.get_position()
    w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()
    dx, dy, n = S.SNP_BAR
    x0, y0, lx, ly = dx / w_in, dy / h_in, n / w_in, n / h_in
    kw = dict(transform=ax.transAxes, color=S.INK, lw=0.7, clip_on=False,
              solid_capstyle="butt", zorder=5)
    ax.plot([x0, x0 + lx], [y0, y0], **kw)
    ax.plot([x0, x0], [y0, y0 + ly], **kw)
    kw = dict(transform=ax.transAxes, fontsize=S.FS_KEY, zorder=5)
    ax.text(x0, y0 - 0.02 / h_in, f"{name}1", ha="left", va="top", **kw)
    ax.text(x0 - 0.02 / w_in, y0, f"{name}2", ha="right", va="bottom", rotation=90, **kw)


def render_snp():
    u = pd.read_parquet(os.path.join(S.NR, "snp_umap.parquet"))
    u = u.sample(frac=1, random_state=S.SEED)
    out = (u.embryo == "discarded").values
    xl, yl = S.window(u.x.values, u.y.values, S.SNP_MAP, frac=0.005)

    fig, box = S.figure(S.SNP_W, S.ROW_H[0])
    ax = box(*S.SNP_MAP)
    ax.scatter(u.x[out], u.y[out], s=S.S_UMAP, color=S.GREY, lw=0, rasterized=True)
    ax.scatter(u.x[~out], u.y[~out], s=S.S_UMAP, c=[S.EMBRYO[e] for e in u.embryo[~out]],
               lw=0, rasterized=True)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    S.bare(ax)
    axis_bars(ax, "SNP UMAP")
    S.key(box(*S.SNP_KEY), [(label(e), c) for e, c in S.EMBRYO.items()]
          + [("discarded", S.GREY)], ncol=3, fs=S.FS_KEY, ms=3.0, gap=0.04, colgap=0.10,
          box=True)
    return S.save_svg(fig, "a_snp")


def render_depth():
    q = pd.read_parquet(os.path.join(S.NR, "qc_cells.parquet"))
    names = list(S.EMBRYO)
    col = [S.EMBRYO[n] for n in names]
    x = np.arange(len(names))

    fig, box = S.figure(S.DEPTH_W, S.ROW_H[0])
    ax = box(*S.DEPTH_BOX)
    ax.bar(x, q.embryo.value_counts().reindex(names), width=0.7, color=col, lw=0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(S.FS_TICK)
    ax.set_ylabel("cells recovered")
    axes = [ax]
    for i, (k, name) in enumerate(STRIPS, start=1):
        ax = box(S.DEPTH_BOX[0] + i * S.DEPTH_DX, *S.DEPTH_BOX[1:])
        b = ax.boxplot([q[k][q.embryo == n] for n in names], positions=x, widths=0.6,
                       whis=(5, 95), showfliers=False, patch_artist=True)
        for p, c in zip(b["boxes"], col):
            p.set(facecolor=c, edgecolor=c, lw=0.5)
        for part in ("whiskers", "caps"):
            for ln, c in zip(b[part], np.repeat(col, 2)):
                ln.set(color=c, lw=0.7)
        for ln in b["medians"]:
            ln.set(color="white", lw=0.9)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(name)
        axes.append(ax)
    for ax in axes:
        ax.set_xticks(x, [label(n)[1:] for n in names])
        ax.set_xlabel("embryo")
        ax.set_xlim(-0.6, len(names) - 0.4)
    return S.save_svg(fig, "b_depth")


def render():
    return render_snp(), render_depth()


if __name__ == "__main__":
    print(render())
