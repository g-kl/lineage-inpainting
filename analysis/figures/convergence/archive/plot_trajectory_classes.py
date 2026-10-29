"""rows 4-5 (v2): trajectory-class panels (reads only numerical_results/).

  class_lines  row 4: seven small line panels (grey members + class mean) + the
               source-source similarity heatmap in the 8th grid cell (fraction of
               genes in the same non-flat class).
  source_hued  row 5: individual gene trajectories hued by source, grouped (curly
               braces) by behaviour: shared EC identity / memory erasure / dynamics.

    python analysis/figures/convergence/plot_trajectory_classes.py [class_lines|source_hued ...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch

sys.path.insert(0, os.path.dirname(__file__))
import style as S

PANELS = os.path.join(S.OUT, "panels")
XLAB = "EC fate bias"
BRACE = "#5a626f"                # brace stroke + labels (matches synthetic_evals cartoon)


def _abbr(s):
    return s.replace(" mesoderm", " meso.")


DABBR = {"somatic mesoderm": "somatic", "dermomyotome": "dermomyotome",
         "splanchnic mesoderm": "splanchnic", "head paraxial mesoderm": "head paraxial",
         "allantois": "allantois"}


def _load():
    d = np.load(os.path.join(S.NR, "traj_classes.npz"), allow_pickle=True)
    sources = [str(s) for s in d["sources"]]
    gnames = np.array([str(g) for g in d["gnames"]])
    row_idx = np.full((len(gnames), len(sources)), -1, int)
    row_idx[d["r_gene"], d["r_src"]] = np.arange(len(d["cid"]))
    return d["grid"], d["feat"], d["cid"], sources, gnames, row_idx


def _save(fig, name):
    svg = os.path.join(PANELS, f"{name}.svg")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(svg.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {svg} (+ .png)")


def _similarity(ax, cax):
    """lower-triangle source x source heatmap: column labels on the bottom, row labels on
    the left, both colour coded to the source palette; vertical colorbar into `cax`."""
    C = pd.read_csv(os.path.join(S.NR, "traj_similarity.csv"), index_col=0)
    src = list(C.columns)
    n = len(C)
    M = np.ma.masked_array(C.values, mask=~np.tri(n, k=-1, dtype=bool))
    cmap = S.TRAJ_SIM_CMAP.copy()
    cmap.set_bad("none")
    off = C.values[np.tri(n, k=-1, dtype=bool)]
    vmin, vmax = np.nanmin(off), np.nanmax(off)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)      # square cells
    thr = vmin + 0.6 * (vmax - vmin)
    for i in range(n):
        for j in range(i):
            v = C.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.0,
                    color="white" if v > thr else S.INK)
    ax.set_xticks(range(n - 1))
    ax.set_xticklabels([DABBR[src[j]] for j in range(n - 1)], rotation=40, ha="right",
                       fontsize=6.0)
    ax.set_yticks(range(1, n))
    ax.set_yticklabels([DABBR[src[i]] for i in range(1, n)], fontsize=6.0)
    for t, j in zip(ax.get_xticklabels(), range(n - 1)):
        t.set_color(S.SOURCE_TYPE_COLORS[src[j]])
    for t, i in zip(ax.get_yticklabels(), range(1, n)):
        t.set_color(S.SOURCE_TYPE_COLORS[src[i]])
    ax.set_xlim(-0.5, n - 1.5); ax.set_ylim(n - 0.5, 0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label("same class fraction", fontsize=6.0)
    cb.ax.tick_params(labelsize=6.0)
    cb.outline.set_visible(False)


N_CLASS_EXAMPLES = 4            # somatic-mesoderm gene names inset per class panel


SRC_INSET = {"somatic mesoderm": "somatic", "dermomyotome": "dermo",
             "splanchnic mesoderm": "splanch", "head paraxial mesoderm": "head par",
             "allantois": "allantois"}


def _tf_set():
    """literature TF symbols (one per line, possibly quoted)."""
    with open(S.TF_LIST) as fh:
        return {ln.strip().strip('"') for ln in fh if ln.strip()}


def _class_examples(c, cid, feat, gnames, row_idx, sources, tfs):
    """the N TF (gene, source) trajectories in class `c` closest to the class mean,
    annotated "gene (source)"; deduped by gene so a panel names distinct TFs."""
    mean = feat[np.where(cid == c)[0]].mean(0)
    cands = []
    for g, s in np.argwhere(row_idx >= 0):
        r = row_idx[g, s]
        if cid[r] == c and str(gnames[g]) in tfs:
            cands.append((float(np.sum((feat[r] - mean) ** 2)), int(g), int(s)))
    cands.sort(key=lambda t: t[0])
    out, seen = [], set()
    for _, g, s in cands:
        if g in seen:
            continue
        seen.add(g)
        out.append(f"{gnames[g]} ({SRC_INSET[sources[s]]})")
        if len(out) == N_CLASS_EXAMPLES:
            break
    return out


def class_lines(out_dir=PANELS):
    S.apply_rc()
    grid, feat, cid, sources, gnames, row_idx = _load()
    classes = list(S.TRAJ_CLASS_LABELS)
    lo, hi = feat.min(), feat.max()
    rng = np.random.default_rng(S.SEED)
    tfs = _tf_set()
    pos = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]   # 7th cell -> heatmap
    left = {(0, 0), (1, 0)}                              # y ticklabels on left column only

    fig = plt.figure(figsize=(6.5, 2.55))
    gs = fig.add_gridspec(2, 4, left=0.06, right=0.99, top=0.93, bottom=0.12,
                          hspace=0.55, wspace=0.14)
    base = None
    for i, c in enumerate(classes):
        r, cc = pos[i]
        ax = fig.add_subplot(gs[r, cc], sharey=base)
        base = base or ax
        m = np.where(cid == c)[0]
        col = S.TRAJ_CLASS_PALETTE[c]
        draw = m if len(m) <= S.TRAJ_MAX_WISPY else rng.choice(m, S.TRAJ_MAX_WISPY, replace=False)
        for rr in draw:
            ax.plot(grid, feat[rr], color=S.GREY, lw=0.3, alpha=0.2, zorder=1)
        ax.plot(grid, feat[m].mean(0), color=col, lw=1.8, zorder=5)
        ax.set_xlim(0, 1); ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 2.5])
        ax.set_title(S.TRAJ_CLASS_LABELS[c], fontsize=S.FS_TICK, color=col, pad=2)
        ax.set_xlabel(XLAB, fontsize=S.FS_TICK)
        S.despine(ax)
        names = _class_examples(c, cid, feat, gnames, row_idx, sources, tfs)
        if names:
            ax.text(0.04, 0.97, "\n".join(names), transform=ax.transAxes, ha="left",
                    va="top", fontsize=6.0, fontstyle="italic", color=col, zorder=6,
                    linespacing=1.15,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5))
        if (r, cc) in left:
            ax.set_ylabel("expr (log)", fontsize=S.FS_TICK)
        else:
            ax.tick_params(labelleft=False)

    cell = gs[1, 3].get_position(fig)                   # heatmap in the empty 8th cell,
    W, H = cell.width, cell.height                      # bottom aligned with the line panels
    axH = fig.add_axes((cell.x0 + 0.40 * W, cell.y0, 0.58 * W, H))
    cax = fig.add_axes((cell.x1 + 0.03 * W, cell.y0 + 0.08 * H, 0.05 * W, 0.55 * H))
    _similarity(axH, cax)
    _save(fig, "traj_class_lines")


def _braceH(bax, xL, xR, yBody, dy, cp, sh, q, lw=1.0):
    """upward curly brace over [xL, xR] on the overlay axis (fig fraction), central cusp
    up toward the label; path math ported from synthetic_evals plot_landscape._braceH."""
    xm = 0.5 * (xL + xR)
    yt = yBody + dy
    yc = yt + cp
    verts = [(xL, yBody), (xL, yt), (xL + q, yt), (xm - sh, yt), (xm, yt), (xm, yc),
             (xm, yt), (xm + sh, yt), (xR - q, yt), (xR, yt), (xR, yBody)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    bax.add_patch(PathPatch(Path(verts, codes), fill=False, ec=BRACE, lw=lw,
                            capstyle="round", joinstyle="round", clip_on=False))
    return yc


def source_hued(out_dir=PANELS):
    S.apply_rc()
    grid, feat, cid, sources, gnames, row_idx = _load()
    ex = pd.read_csv(os.path.join(S.NR, "traj_row5_examples.csv"))
    name2g = {g: i for i, g in enumerate(gnames)}
    src_col = [S.SOURCE_TYPE_COLORS[s] for s in sources]

    groups = [(lab, list(ex[ex.group == key].sort_values("order")["gene"]))
              for key, lab in S.TRAJ_ROW5_GROUPS]
    ncol = sum(len(g[1]) for g in groups)
    rows = [row_idx[name2g[g], si] for _, gl in groups for g in gl
            for si in range(len(sources)) if row_idx[name2g[g], si] >= 0]
    lo, hi = feat[rows].min(), feat[rows].max()          # one shared scale for the row

    fig, axes = plt.subplots(1, ncol, figsize=(6.5, 1.78), squeeze=False, sharey=True)
    ci, spans = 0, []
    for _, glist in groups:
        c0 = ci
        for gname in glist:
            ax = axes[0][ci]
            g = name2g[gname]
            for si in range(len(sources)):
                r = row_idx[g, si]
                if r >= 0:
                    ax.plot(grid, feat[r], color=src_col[si], lw=1.3, zorder=4)
            ax.set_xlim(0, 1); ax.set_ylim(lo, hi)
            ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 1, 2, 3])
            ax.set_xlabel(XLAB, fontsize=S.FS_TICK)
            ax.text(0.05, 0.96, gname, transform=ax.transAxes, ha="left", va="top",
                    fontsize=S.FS_TICK, fontstyle="italic", color=S.INK, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5))
            S.despine(ax)                                # keep the y-axis line + ticks
            if ci == 0:
                ax.set_ylabel("expr (log)", fontsize=S.FS_TICK)
            else:
                ax.tick_params(labelleft=False)
            ci += 1
        spans.append((c0, ci - 1))

    handles = [plt.Line2D([0], [0], color=src_col[i], lw=2.0, label=sources[i])
               for i in range(len(sources))]
    fig.legend(handles=handles, loc="lower center", ncol=len(sources), frameon=False,
               fontsize=S.FS_LEGEND, handlelength=1.4, columnspacing=1.1,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 0.84), w_pad=0.4)

    bax = fig.add_axes((0, 0, 1, 1))
    bax.set_axis_off(); bax.set_xlim(0, 1); bax.set_ylim(0, 1)
    W, H = fig.get_size_inches()
    dy, cp, sh = 0.05 / H, 0.05 / H, 0.045 / W
    ytop = axes[0][0].get_position().y1
    for (c0, c1), (label, _) in zip(spans, groups):
        x0 = axes[0][c0].get_position().x0
        x1 = axes[0][c1].get_position().x1
        q = min(0.06 / W, (x1 - x0) * 0.16)
        yc = _braceH(bax, x0, x1, ytop + 0.03, dy, cp, sh, q)
        bax.text((x0 + x1) / 2, yc + 0.02, label, ha="center", va="bottom",
                 fontsize=S.FS_LABEL, color=BRACE)
    _save(fig, "traj_source_hued")


PANEL_FNS = {"class_lines": class_lines, "source_hued": source_hued}

if __name__ == "__main__":
    os.makedirs(PANELS, exist_ok=True)
    for name in (sys.argv[1:] or list(PANEL_FNS)):
        PANEL_FNS[name]()
