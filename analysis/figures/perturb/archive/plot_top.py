"""rows 1-2 of the figure as one matplotlib SVG (read from the cached arrays).

row 1  Nkx2-5 hero at E9.0, four UMAPs sharing one extent + grey backdrop:
       (1) control leaves hued by Nkx2-5 expression
       (2) Nkx2-5 KO leaves coloured shifted / not shifted
       (3) Nkx2-5 KO log-density-ratio
       (4) mean log-density-ratio over the heart-screen (Sierant et al., 2025) genes
row 2  (1) %heart vs %total depletion        (2) heart specificity vs shift time
       (3) Tbx5 dose-response abundance curves (4) heart depletion 100% vs 50% KD
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from analysis.figures.perturb import style as S
from utils.figure_labels import register_fonts, FONT_FAMILY as PANEL_FONT, PANEL_LETTER_FS

S.apply_rc()
register_fonts()
# bold panel letters share the figure-wide renderer's font/size (a & f are placed in
# assemble.py; b-e are placed here, where matplotlib knows each axis's exact position).
PANEL_LETTER_FP = FontProperties(family=PANEL_FONT, weight="bold", size=PANEL_LETTER_FS)


def panel_letter(ax, ch):
    """bold lowercase panel letter at the axis's top-left corner."""
    ax.annotate(ch, xy=(0, 1), xycoords="axes fraction", xytext=(-32, 4),
                textcoords="offset points", fontproperties=PANEL_LETTER_FP,
                color=S.INK, ha="left", va="bottom", annotation_clip=False)


## LABEL DE-OVERLAP ##

def repel_labels(ax, xs, ys, labels, fs=3.8, color=S.INK):
    """place gene labels near their points and nudge them apart (mini-adjustText).

    works in axes-fraction space with estimated text boxes; draws a thin leader
    from each settled label back to its point.
    """
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    wpt, hpt = bb.width, bb.height
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    fx = lambda x: (x - x0) / (x1 - x0)
    fy = lambda y: (y - y0) / (y1 - y0)
    P = np.array([[fx(x), fy(y)] for x, y in zip(xs, ys)])
    ppt = ax.figure.dpi / 72.0                          # px per point (box sizes)
    hw = np.array([(0.55 * fs * len(t) * ppt + 3) / wpt for t in labels]) / 2
    hh = np.full(len(labels), (fs * 1.25 * ppt) / hpt) / 2
    # init: nudge each label radially outward from the cloud centroid
    c = P.mean(0)
    d0 = P - c
    d0 /= np.maximum(np.linalg.norm(d0, axis=1, keepdims=True), 1e-6)
    L = P + d0 * 0.06
    for _ in range(1200):
        for i in range(len(L)):
            f = np.zeros(2)
            for j in range(len(L)):
                if i == j:
                    continue
                d = L[i] - L[j]
                ox = (hw[i] + hw[j] + 0.004) - abs(d[0])
                oy = (hh[i] + hh[j] + 0.004) - abs(d[1])
                if ox > 0 and oy > 0:                       # label boxes overlap
                    s = np.sign(d) if d.any() else np.array([1.0, 1.0])
                    if ox < oy:
                        f[0] += s[0] * ox * 0.7             # resolve along smaller overlap
                    else:
                        f[1] += s[1] * oy * 0.7
            for q in P:                                     # push off the points
                d = L[i] - q
                ox = (hw[i] + 0.012) - abs(d[0])
                oy = (hh[i] + 0.010) - abs(d[1])
                if ox > 0 and oy > 0:
                    f += (np.sign(d) if d.any() else np.array([0.0, 1.0])) * [ox, oy] * 0.5
            f += (P[i] - L[i]) * 0.003                      # weak spring to the point
            L[i] += np.clip(f, -0.05, 0.05)
            L[i, 0] = np.clip(L[i, 0], hw[i], 1 - hw[i])
            L[i, 1] = np.clip(L[i, 1], hh[i], 1 - hh[i])
    # polish: pure label-label separation (no spring) until boxes stop overlapping
    for _ in range(500):
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                d = L[i] - L[j]
                ox = (hw[i] + hw[j] + 0.004) - abs(d[0])
                oy = (hh[i] + hh[j] + 0.004) - abs(d[1])
                if ox > 0 and oy > 0:
                    s = np.sign(d) if d.any() else np.array([1.0, 1.0])
                    if ox < oy:
                        L[i, 0] += s[0] * ox * 0.5; L[j, 0] -= s[0] * ox * 0.5
                    else:
                        L[i, 1] += s[1] * oy * 0.5; L[j, 1] -= s[1] * oy * 0.5
        L[:, 0] = np.clip(L[:, 0], hw, 1 - hw)
        L[:, 1] = np.clip(L[:, 1], hh, 1 - hh)
    for (lx, ly), (px, py), t in zip(L, P, labels):
        if abs(lx - px) > hw.mean() * 0.8 or abs(ly - py) > hh.mean() * 1.5:
            ax.plot([lx, px], [ly, py], color="#bbbbbb", lw=0.3, zorder=2,
                    transform=ax.transAxes)
        ax.text(lx, ly, t, fontsize=fs, color=color, ha="center", va="center",
                zorder=4, transform=ax.transAxes)


## UMAP PANELS ##

def umap_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def backdrop_scatter(ax, backdrop):
    ax.scatter(backdrop[:, 0], backdrop[:, 1], s=1.0, c=S.GREY, linewidths=0,
               rasterized=True, zorder=1)


def value_panel(ax, backdrop, xy, vals, cmap, vmin, vmax, xlim, ylim, by_abs=True):
    """scatter points coloured by a continuous value over the grey backdrop."""
    backdrop_scatter(ax, backdrop)
    o = np.argsort(np.abs(vals) if by_abs else vals)        # strongest drawn on top
    sc = ax.scatter(xy[o, 0], xy[o, 1], s=2.0, c=vals[o], cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=0, rasterized=True, zorder=2)
    umap_axes(ax, xlim, ylim)
    return sc


def slim_cbar(ax, sc, label, ticks, fmt=None):
    cax = ax.inset_axes([0.08, -0.11, 0.84, 0.045])
    cb = ax.figure.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks(ticks)
    if fmt:
        cax.set_xticklabels([fmt(t) for t in ticks])
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cb.set_label(label, fontsize=5.0, labelpad=1)
    cb.outline.set_linewidth(0.4)


## BUILD ##

def draw_hero_row(fig, slot, sh, de, backdrop, xlim, ylim):
    """the Nkx2-5 hero UMAP row (E9.0), four panels.  kept for reuse; hidden from the
    figure by default via S.SHOW_HERO_ROW."""
    r1 = slot.subgridspec(1, 4, wspace=0.08)

    # (1) control leaves hued by Nkx2-5 expression
    ax = fig.add_subplot(r1[0])
    sc = value_panel(ax, backdrop, de["ctrl_leaf_xy"], de["ctrl_leaf_expr"],
                     S.EXPR_CMAP, None, None, xlim, ylim, by_abs=False)
    ax.set_title(f"{S.HERO_GENE} expression", fontsize=7, color=S.INK, pad=2,
                 style="italic")
    slim_cbar(ax, sc, "log1p expression",
              [float(de["ctrl_leaf_expr"].min()), float(de["ctrl_leaf_expr"].max())],
              fmt=lambda t: f"{t:.0f}")

    # (2) KO leaves: shifted vs not shifted
    ax = fig.add_subplot(r1[1])
    backdrop_scatter(ax, backdrop)
    un, shf = sh["unshifted"], sh["shifted"]
    ax.scatter(un[:, 0], un[:, 1], s=1.6, c=S.UNSHIFTED_COL, linewidths=0,
               alpha=0.55, rasterized=True, zorder=2, label="not shifted")
    ax.scatter(shf[:, 0], shf[:, 1], s=1.6, c=S.SHIFTED_COL, linewidths=0,
               alpha=0.65, rasterized=True, zorder=3, label="shifted")
    umap_axes(ax, xlim, ylim)
    ax.set_title(f"{S.HERO_GENE} KO", fontsize=7, color=S.INK, pad=2, style="italic")
    leg = ax.legend(loc="lower left", fontsize=4.8, frameon=False, handletextpad=0.2,
                    borderpad=0.1, labelspacing=0.2, markerscale=2.0)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    # (3) Nkx2-5 KO log-density ratio
    ax = fig.add_subplot(r1[2])
    sc = value_panel(ax, backdrop, de["ref_xy"], de["nkx_ratio"], S.DENSITY_CMAP,
                     -S.VMAX, S.VMAX, xlim, ylim)
    ax.set_title(f"{S.HERO_GENE} KO density", fontsize=7, color=S.INK, pad=2,
                 style="italic")

    # (4) mean log-density ratio over the heart-screen genes
    ax = fig.add_subplot(r1[3])
    sc = value_panel(ax, backdrop, de["ref_xy"], de["mean_ratio"], S.DENSITY_CMAP,
                     -S.VMAX, S.VMAX, xlim, ylim)
    ax.set_title("Sierant et al. genes", fontsize=7, color=S.INK, pad=9)
    ax.text(0.5, 1.012, f"(mean over {len(de['genes'])} genes)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=4.6, color="#9aa0ab")
    slim_cbar(ax, sc, "log density ratio (KO / control)", [-S.VMAX, 0, S.VMAX])


def main():
    sh = np.load(os.path.join(S.NUMERICAL, "shift.npz"))
    de = np.load(os.path.join(S.NUMERICAL, "density.npz"), allow_pickle=True)
    do = np.load(os.path.join(S.NUMERICAL, "dose.npz"), allow_pickle=True)
    m = pd.read_csv(os.path.join(S.NUMERICAL, "metrics.csv"))

    backdrop = sh["backdrop"]
    pad = 0.03 * (backdrop.max(0) - backdrop.min(0))
    xlim = (backdrop[:, 0].min() - pad[0], backdrop[:, 0].max() + pad[0])
    ylim = (backdrop[:, 1].min() - pad[1], backdrop[:, 1].max() + pad[1])

    nrows = 2 if S.SHOW_HERO_ROW else 1
    fig = plt.figure(figsize=(S.PAGE_W_IN,
                              S.TOP_H_IN if S.SHOW_HERO_ROW else S.METRICS_H_IN))
    outer = fig.add_gridspec(nrows, 1, left=0.065, right=0.985, top=0.93, bottom=0.06,
                             hspace=0.42, height_ratios=[1] * nrows)

    if S.SHOW_HERO_ROW:
        draw_hero_row(fig, outer[0], sh, de, backdrop, xlim, ylim)

    # ---------------- quantitative panels ----------------
    r2 = outer[nrows - 1].subgridspec(1, 4, wspace=0.42)

    # (1) %heart vs %total depletion
    ax = fig.add_subplot(r2[0])
    ax.set_box_aspect(1)
    panel_letter(ax, "d")
    ax.scatter(m["total_depl"], m["heart_depl"], s=10, c=S.MUTED, linewidths=0.3,
               edgecolors="white", zorder=3)
    ax.set_xlabel("% total depletion", fontsize=7, color=S.INK, labelpad=2)
    ax.set_ylabel("% heart depletion", fontsize=7, color=S.INK, labelpad=2)
    ax.tick_params(labelsize=6, length=2)
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 108)
    repel_labels(ax, m["total_depl"].to_numpy(), m["heart_depl"].to_numpy(),
                 list(m["gene"]))

    # (2) heart specificity vs shift time
    ax = fig.add_subplot(r2[1])
    ax.set_box_aspect(1)
    panel_letter(ax, "e")
    s = m.dropna(subset=["shift_time"]).copy()
    s["spec"] = np.log2(np.maximum(s["heart_depl"], 1.0) / np.maximum(s["total_depl"], 1.0))
    ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
    ax.scatter(s["shift_time"], s["spec"], s=10, c=S.MUTED, linewidths=0.3,
               edgecolors="white", zorder=3)
    ax.set_xlabel("shift time", fontsize=7, color=S.INK, labelpad=2)
    ax.set_ylabel("heart specificity", fontsize=7, color=S.INK, labelpad=2)
    ax.text(0.03, 0.975,
            r"$\log_2\!\left(\frac{\%\,\mathrm{heart\ depl.}}{\%\,\mathrm{total\ depl.}}\right)$",
            transform=ax.transAxes, fontsize=5.2, color="#6b7280", va="top")
    ax.tick_params(labelsize=6, length=2)
    repel_labels(ax, s["shift_time"].to_numpy(), s["spec"].to_numpy(), list(s["gene"]))

    # (3) Tbx5 dose-response abundance curves
    ax = fig.add_subplot(r2[2])
    ax.set_box_aspect(1)
    panel_letter(ax, "f")
    doses, abundance = do["doses"], do["abundance"]
    cell_types, top_idx = list(do["cell_types"]), list(do["top_idx"])
    order = np.argsort(doses)
    for j in range(len(cell_types)):
        if j in top_idx:
            continue
        ax.plot(doses[order], abundance[j][order], color=S.DOSE_WISP, lw=0.6, zorder=1)
    for rank, j in enumerate(top_idx):
        c = S.DOSE_HIGHLIGHT[rank % len(S.DOSE_HIGHLIGHT)]
        ax.plot(doses[order], abundance[j][order], color=c, lw=1.4, marker="o", ms=2.2,
                zorder=3, label=S.DOSE_ABBREV.get(cell_types[j], cell_types[j]))
    ax.set_xlabel(f"{S.DOSE_GENE} dose (% expression)", fontsize=7, color=S.INK, labelpad=2)
    ax.set_ylabel("norm. cell-type abundance", fontsize=7, color=S.INK, labelpad=2)
    ax.tick_params(labelsize=6, length=2)
    ax.legend(loc="lower right", fontsize=4.4, frameon=False, handlelength=1.0,
              handletextpad=0.3, borderpad=0.1, labelspacing=0.2)

    # (4) heart depletion 100% vs 50% knockdown
    ax = fig.add_subplot(r2[3])
    ax.set_box_aspect(1)
    panel_letter(ax, "g")
    d = m.dropna(subset=["heart_depl_50"])
    lim = (-5, 108)
    ax.plot(lim, lim, color="#dddddd", lw=0.6, zorder=0)
    ax.scatter(d["heart_depl_50"], d["heart_depl"], s=10, c=S.MUTED, linewidths=0.3,
               edgecolors="white", zorder=3)
    ax.set_xlabel("% heart depl. (50% KD)", fontsize=7, color=S.INK, labelpad=2)
    ax.set_ylabel("% heart depl. (100% KD)", fontsize=7, color=S.INK, labelpad=2)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.tick_params(labelsize=6, length=2)
    repel_labels(ax, d["heart_depl_50"].to_numpy(), d["heart_depl"].to_numpy(),
                 list(d["gene"]))

    # bbox_inches="tight" expands the canvas to enclose every label (so the row-2
    # x-labels are never clipped at the block's bottom edge) and trims excess
    # whitespace, so the block's height tracks its content and GAP_IN alone controls
    # the spacing to the ring below.  transparent=True so the block's top whitespace
    # (pulled up over the top row by GAP_RING_TOP_IN) does not paint over panel c's
    # x-labels above it.
    svg = os.path.join(S.IMGS, "figure_top.svg")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(os.path.join(S.IMGS, "figure_top.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {svg}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
