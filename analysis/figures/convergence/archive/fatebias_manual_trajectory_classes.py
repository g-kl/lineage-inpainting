#!/usr/bin/env python
"""MANUAL (rule-based) version of fatebias_source_trajectory_clusters.py.

The Ward-clustering sibling groups the per-(gene, source) trajectories by L2 shape, but a
few biologically obvious classes never fall out on their own -- most sharply the TRANSIENT
(cap-shaped) genes, whose interior bump is a rare, low-variance signal that amplitude- and
level-dominated clustering absorbs into the flat / rising groups no matter how many clusters
are cut. Here we skip clustering entirely and assign each trajectory to a HAND-DEFINED class
from three cheap features of its raw log trajectory over the EC-fate-bias grid:
  slope  = linear fit vs fate bias         (sign + magnitude -> rising / falling / stable)
  level  = mean expression over the grid   (high vs low, for the stable genes)
  bump   = interior peak that clears BOTH ends by a fraction of the range (-> transient)

Classes (numbered for the palette / plots):
  1 high, stable      2 low, stable
  3 rising strongly   4 rising mildly
  5 falling strongly  6 falling mildly
  7 transient         (max in the late fate-bias window (0.5, 0.95), clearing both endpoints)

Precedence: transient wins first (it is a shape call, not a slope one), then the strong/mild
rising and falling bands off |slope|, then the remaining near-flat genes split high vs low at a
fixed level (STABLE_HI). Slope bands are set relative to the mean |slope| (MILD_K, STRONG_K);
the transient gate
uses the sibling's calibrated movement floor (TD.COMMIT_RANGE), a fate-bias window on the peak
(TRANS_LO, TRANS_HI) and a small endpoint-clearance margin (TRANS_FRAC).

Trajectory construction (raw log, one row per (gene, source), gaussian-smoothed vs fate bias,
undersampled grid points interpolated, undersampled sources dropped) is IDENTICAL to the Ward
sibling -- we import its build_features so the only thing that changes is class assignment.

Outputs (analysis/results/fatebias_manual_trajectory_classes/):
  manual_trajectory_classes_<tag>.{png,svg}   per-class panels: member (gene, source)
                                              trajectories (grey) + class mean + TF names
  manual_source_hued_<tag>.{png,svg}          example genes per class (one panel each, the
                                              gene's per-source trajectories hued by source)
  manual_class_switch_<tag>.{png,svg}         dermomyotome class (rows) x somatic mesoderm
                                              class (cols), cell = # genes with that pair
  manual_transient_switch_<tag>.{png,svg}     genes transient in >=1 source but rising in >=1
                                              other; per-source trajectories hued by source
  manual_class_assignments_<tag>.csv          gene x source class-label matrix

    python analysis/figures/convergence/fatebias_manual_trajectory_classes.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S                              # noqa: E402
import fatebias as FB                          # noqa: E402
import fatebias_trajectory_distance as TD      # noqa: E402  (scan / constants / TF list)
import fatebias_trajectory_clusters as CL      # noqa: E402  (palette / label helpers)
import fatebias_source_trajectory_clusters as SC  # noqa: E402  (build_features, raw mode)

KEY = "ec"
WIN_LO, WIN_HI = S.BIN_LO, S.BIN_HI
N_SOURCES = TD.N_SOURCES
N_EX = 5                          # example genes shown per class (source-hued panels)
N_TSWITCH = 10                    # genes shown in the transient-switch panel
EPS = CL.EPS

# manual class thresholds
MILD_K = 0.5                      # |slope| >= MILD_K   * mean|slope|  -> mild mover
STRONG_K = 1.5                    # |slope| >= STRONG_K * mean|slope|  -> strong mover
TRANS_LO, TRANS_HI = 0.5, 0.9    # transient peak must sit in this fate-bias window
TRANS_FRAC = 0.10                 # ... clearing BOTH endpoints by this (small) fraction of range
RANGE_MIN = TD.COMMIT_RANGE       # ... on a trajectory that actually moves this much (log)
STABLE_HI = 0.5                   # stable gene is "high" if its mean level >= this (log)

CLASSES = {1: "high, stable", 2: "low, stable", 3: "rising strongly", 4: "rising mildly",
           5: "falling strongly", 6: "falling mildly", 7: "transient"}
N_CLASS = len(CLASSES)

RESULTS = os.path.join(S.base_dir, "analysis", "results",
                       "fatebias_manual_trajectory_classes")
os.makedirs(RESULTS, exist_ok=True)
TAG = "manual"
YLAB = "expr (log)"


def classify(feat, grid):
    """assign each trajectory (raw log, one row) a class 1..7 from slope / level / bump.
    returns (class id, signed slope, mean level, range) per row."""
    n, ng = feat.shape
    gc = grid - grid.mean()
    slope = (feat @ gc) / (gc @ gc + EPS)                  # d(expr)/d(bias)
    level = feat.mean(1)
    rng = feat.max(1) - feat.min(1)
    imax = feat.argmax(1)
    peakpos = grid[imax]                                   # fate bias at the peak
    peak = feat[np.arange(n), imax]
    up, down = peak - feat[:, 0], peak - feat[:, -1]       # rise to / fall from the peak
    transient = ((peakpos > TRANS_LO) & (peakpos < TRANS_HI) & (rng >= RANGE_MIN)
                 & (up >= TRANS_FRAC * rng) & (down >= TRANS_FRAC * rng))

    am = np.abs(slope).mean()
    mild, strong = MILD_K * am, STRONG_K * am

    conds = [transient,
             slope >= strong, (slope >= mild) & (slope < strong),
             slope <= -strong, (slope <= -mild) & (slope > -strong),
             level >= STABLE_HI]
    cid = np.select(conds, [7, 3, 4, 5, 6, 1], default=2)  # default = low, stable
    return cid, slope, level, rng


def load():
    """the Ward sibling's data pipeline: per-(gene, source) raw log trajectories + row keys."""
    family = S.FANIN[KEY]["exclude"]
    print(f"[1/4] {KEY}: internal nodes per source E[{WIN_LO}, {WIN_HI}) ...", flush=True)
    td, e95, assignments, subtype, sub2type = FB.load_trees()
    cells, stypes, bias = TD.scan(td, e95, assignments, subtype, sub2type,
                                  family, WIN_LO, WIN_HI)
    print("[2/4] reading expression ...", flush=True)
    Xp, ok, genes = FB.read_expression(cells)
    stypes, bias = stypes[ok], bias[ok]
    expressed = Xp.mean(0) >= S.MIN_EXPR
    Xe, gnames = Xp[:, expressed], genes[expressed]
    vc = pd.Series(stypes).value_counts()
    sources = [s for s in S.SOURCE_TYPES if s in vc.index][:N_SOURCES]
    print("[3/4] per-(gene, source) trajectories (raw log) ...", flush=True)
    grid = np.linspace(0.0, 1.0, TD.N_GRID)
    feat, r_gene, r_src, _, _, sources = SC.build_features(
        grid, bias, Xe, stypes, sources, Xe.std(0))
    print(f"      {feat.shape[0]:,} trajectories, {len(gnames):,} genes, "
          f"{len(sources)} sources", flush=True)
    return grid, feat, r_gene, r_src, sources, gnames


def pick_examples(cid, r_gene, slope, level, rng, is_tf):
    """for each class, distinct example genes: TFs first, then by the feature that defines
    the class (slope for movers, level for stable, bump range for transient)."""
    out = []
    for c in CLASSES:
        rows = np.where(cid == c)[0]
        if c in (3, 4):
            score = slope[rows]                    # steepest risers first
        elif c in (5, 6):
            score = -slope[rows]                   # steepest fallers first
        elif c == 1:
            score = level[rows]                    # highest plateau first
        elif c == 2:
            score = -level[rows]                   # lowest plateau first
        else:
            score = rng[rows]                      # biggest bump first
        order = rows[np.lexsort((-score, np.where(is_tf[r_gene[rows]], 0, 1)))]
        genes = []
        for r in order:
            g = int(r_gene[r])
            if g not in genes:
                genes.append(g)
            if len(genes) >= N_EX:
                break
        out.append(genes)
    return out


def main():
    grid, feat, r_gene, r_src, sources, gnames = load()
    n_src, G = len(sources), len(gnames)
    cid, slope, level, rng = classify(feat, grid)

    tfs = TD.read_tf_list(S.TF_LIST)
    is_tf = np.array([g in tfs for g in gnames])
    members, tf_top = {}, {}
    print("[4/4] class sizes:", flush=True)
    for c in CLASSES:
        m = np.where(cid == c)[0]
        members[c] = m
        gtf = r_gene[m][is_tf[r_gene[m]]]
        uniq = pd.unique(gtf[np.argsort(-np.abs(slope[m][is_tf[r_gene[m]]]))])
        tf_top[c] = [gnames[g] for g in uniq[:CL.N_TF_LABEL]]
        print(f"      {c} {CLASSES[c]:16s}: rows={len(m):5d}  genes="
              f"{len(np.unique(r_gene[m])):4d}  TFs: {', '.join(tf_top[c])}", flush=True)

    lab = np.zeros((G, n_src), int)
    lab[r_gene, r_src] = cid
    pd.DataFrame(lab, columns=[s.replace(" ", "_") for s in sources]).assign(
        gene=gnames, is_tf=is_tf,
        n_distinct=[len(set(r[r > 0])) for r in lab]).to_csv(
        os.path.join(RESULTS, f"manual_class_assignments_{TAG}.csv"), index=False)

    S.apply_rc()
    rng_rand = np.random.default_rng(S.SEED)
    plot_classes(grid, feat, members, tf_top, rng_rand)
    row_idx = np.full((G, n_src), -1, int)
    row_idx[r_gene, r_src] = np.arange(feat.shape[0])
    examples = pick_examples(cid, r_gene, slope, level, rng, is_tf)
    ngene = {c: len(np.unique(r_gene[members[c]])) for c in CLASSES}
    classes = [(f"{CLASSES[c]}\n(n={ngene[c]})", examples[i])
               for i, c in enumerate(CLASSES)]
    plot_source_hued(grid, feat, row_idx, sources, gnames, classes)
    plot_class_switch(lab, sources)
    plot_transient_switch(grid, feat, row_idx, lab, sources, gnames, rng, is_tf)

    print(f"done ({TAG}). wrote:", flush=True)
    print(f"      manual_trajectory_classes_{TAG}.png/.svg", flush=True)
    print(f"      manual_source_hued_{TAG}.png/.svg", flush=True)
    print(f"      manual_class_switch_{TAG}.png/.svg", flush=True)
    print(f"      manual_transient_switch_{TAG}.png/.svg", flush=True)
    print(f"      manual_class_assignments_{TAG}.csv", flush=True)


def plot_classes(grid, feat, members, tf_top, rng_rand):
    """one panel per class: member (gene, source) trajectories grey + class mean + TFs."""
    ncol = 3
    nrow = int(np.ceil(N_CLASS / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.PAGE_W_IN, 2.0 * nrow + 0.7),
                             squeeze=False)
    ne = [m for m in members.values() if len(m)]
    lo = min(feat[m].min() for m in ne)
    hi = max(feat[m].max() for m in ne)
    for c in CLASSES:
        ax = axes[(c - 1) // ncol][(c - 1) % ncol]
        m = members[c]
        col = CL.CLUSTER_PALETTE[(c - 1) % len(CL.CLUSTER_PALETTE)]
        if len(m):
            draw = m if len(m) <= CL.MAX_WISPY else rng_rand.choice(m, CL.MAX_WISPY,
                                                                    replace=False)
            for r in draw:
                ax.plot(grid, feat[r], color=S.GREY, lw=0.35, alpha=0.22, zorder=1)
            ax.plot(grid, feat[m].mean(0), color=col, lw=2.2, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.5, 1.0])
        S.despine(ax)
        ax.set_title(f"{CLASSES[c]}  ({len(m)} rows)", fontsize=S.FS_LABEL, color=col, pad=4)
        names = tf_top[c] if tf_top[c] else ["(no TFs)"]
        ax.text(0.04, 0.96, "\n".join(names), transform=ax.transAxes, ha="left", va="top",
                fontsize=S.FS_ANNOT, color=S.INK, linespacing=1.3)
        if (c - 1) % ncol == 0:
            ax.set_ylabel(YLAB, fontsize=S.FS_LABEL)
    for k in range(N_CLASS, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.text(0.5, 0.015, "EC fate bias", ha="center", va="bottom",
             fontsize=S.FS_LABEL, color=S.INK)
    fig.text(0.5, 0.001, "per-(gene, source) trajectories assigned by rule; grey = members, "
             "colour = class mean", ha="center", va="bottom", fontsize=S.FS_ANNOT,
             color=S.MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"manual_trajectory_classes_{TAG}.{ext}"), dpi=300)
    plt.close(fig)


def plot_source_hued(grid, feat, row_idx, sources, gnames, classes):
    """rows = class, cols = example genes; each panel overlays a gene's per-source
    trajectories HUED BY SOURCE (y autoscaled per gene)."""
    n_src = len(sources)
    src_col = [S.ROW3_SOURCE_PALETTE[i % len(S.ROW3_SOURCE_PALETTE)] for i in range(n_src)]
    nrow, ncol = len(classes), N_EX
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.PAGE_W_IN, 1.5 * nrow + 1.0),
                             squeeze=False)
    for ri, (title, glist) in enumerate(classes):
        for ci in range(ncol):
            ax = axes[ri][ci]
            if ci >= len(glist):
                ax.axis("off")
                continue
            g = glist[ci]
            for si in range(n_src):
                r = row_idx[g, si]
                if r >= 0:
                    ax.plot(grid, feat[r], color=src_col[si], lw=1.6, zorder=4)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1.0])
            S.despine(ax)
            ax.set_title(gnames[g], fontsize=S.FS_LABEL, color=S.INK, pad=3,
                         fontstyle="italic")
            if ci == 0:
                ax.set_ylabel(title, fontsize=S.FS_LABEL)
    handles = [plt.Line2D([0], [0], color=src_col[i], lw=2.2, label=sources[i])
               for i in range(n_src)]
    fig.legend(handles=handles, loc="lower center", ncol=min(n_src, 5), frameon=False,
               fontsize=S.FS_LEGEND, handlelength=1.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.075, "EC fate bias", ha="center", va="bottom",
             fontsize=S.FS_LABEL, color=S.INK)
    fig.text(0.5, 0.012, f"per-source trajectories hued by source ({YLAB})", ha="center",
             va="bottom", fontsize=S.FS_ANNOT, color=S.MUTED)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"manual_source_hued_{TAG}.{ext}"), dpi=300)
    plt.close(fig)


def plot_class_switch(lab, sources, row_src="dermomyotome", col_src="somatic mesoderm"):
    """class-switch matrix between two sources: rows = the gene's class in the row source
    (dermomyotome), cols = its class in the col source (somatic mesoderm), cell = # genes
    with that (row -> col) class pair, over genes covered in BOTH. the boxed diagonal is
    genes keeping their class across the two paths; off-diagonal cells are switches. colour
    is a sqrt (PowerNorm) ramp so small off-diagonal cells stay visible; the number is the
    raw count."""
    try:
        ri, ci = sources.index(row_src), sources.index(col_src)
    except ValueError:
        print(f"      ! {row_src} / {col_src} not both kept; skipping switch map", flush=True)
        return
    K = N_CLASS
    both = (lab[:, ri] > 0) & (lab[:, ci] > 0)
    M = np.zeros((K, K), int)
    for a, b in zip(lab[both, ri], lab[both, ci]):
        M[a - 1, b - 1] += 1

    fig, ax = plt.subplots(figsize=(0.55 * K + 3.2, 0.55 * K + 2.6))
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=float(M.max()) or 1.0)
    im = ax.imshow(M, cmap=TD.DIST_CMAP, norm=norm)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([CLASSES[c] for c in range(1, K + 1)], fontsize=S.FS_TICK,
                       rotation=35, ha="right")
    ax.set_yticklabels([CLASSES[c] for c in range(1, K + 1)], fontsize=S.FS_TICK)
    ax.set_xlabel(f"{col_src} class", fontsize=S.FS_LABEL)
    ax.set_ylabel(f"{row_src} class", fontsize=S.FS_LABEL)
    for i in range(K):
        for j in range(K):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=S.FS_ANNOT,
                        color="white" if norm(M[i, j]) > 0.55 else S.INK)
    for d in range(K):                                   # box the diagonal (class kept)
        ax.add_patch(plt.Rectangle((d - 0.5, d - 0.5), 1, 1, fill=False,
                                   edgecolor=S.INK, lw=1.1, zorder=5))
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=S.FS_TICK)
    cb.set_label("genes (sqrt scale)", fontsize=S.FS_LABEL)
    cb.outline.set_visible(False)
    n_same = int(np.trace(M))
    ax.set_title(f"class switches: {row_src} to {col_src}\n"
                 f"{n_same}/{int(both.sum())} genes keep their class",
                 fontsize=S.FS_LABEL, color=S.INK, pad=6)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"manual_class_switch_{TAG}.{ext}"), dpi=300)
    plt.close(fig)


def plot_transient_switch(grid, feat, row_idx, lab, sources, gnames, rng, is_tf):
    """genes TRANSIENT in >=1 source but RISING (mild or strong) in >=1 OTHER source, over any
    sources: one panel each, per-source trajectories hued by source, annotated with how many
    sources are transient (T) vs rising (R). shows the transient bump is a path-specific
    alternative to a monotonic rise. TFs first, then most transient sources / biggest bump."""
    n_src = len(sources)
    is_trans = lab == 7
    is_rise = (lab == 3) | (lab == 4)
    n_trans, n_rise = is_trans.sum(1), is_rise.sum(1)
    cand = np.where((n_trans >= 1) & (n_rise >= 1))[0]
    if not len(cand):
        print("      ! no transient-in-some / rising-in-others genes", flush=True)
        return
    bump = np.array([max((rng[row_idx[g, si]] for si in range(n_src) if is_trans[g, si]),
                         default=0.0) for g in cand])       # biggest transient-row range
    n_cand = len(cand)
    cand = cand[np.lexsort((-bump, -n_trans[cand], np.where(is_tf[cand], 0, 1)))][:N_TSWITCH]
    print(f"      transient in some / rising in others: {n_cand} genes; "
          f"showing {', '.join(gnames[g] for g in cand)}", flush=True)

    src_col = [S.ROW3_SOURCE_PALETTE[i % len(S.ROW3_SOURCE_PALETTE)] for i in range(n_src)]
    ncol = 5
    nrow = int(np.ceil(len(cand) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.PAGE_W_IN, 1.7 * nrow + 1.0),
                             squeeze=False)
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= len(cand):
            ax.axis("off")
            continue
        g = cand[k]
        for si in range(n_src):
            r = row_idx[g, si]
            if r >= 0:
                ax.plot(grid, feat[r], color=src_col[si], lw=1.6, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        S.despine(ax)
        ax.set_title(gnames[g], fontsize=S.FS_LABEL, color=S.INK, pad=3, fontstyle="italic")
        ax.text(0.04, 0.96, f"T:{n_trans[g]}  R:{n_rise[g]}", transform=ax.transAxes,
                ha="left", va="top", fontsize=S.FS_ANNOT, color=S.MUTED)
        if k % ncol == 0:
            ax.set_ylabel(YLAB, fontsize=S.FS_LABEL)
    handles = [plt.Line2D([0], [0], color=src_col[i], lw=2.2, label=sources[i])
               for i in range(n_src)]
    fig.legend(handles=handles, loc="lower center", ncol=min(n_src, 5), frameon=False,
               fontsize=S.FS_LEGEND, handlelength=1.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("transient in some sources, rising in others (T = #transient, R = #rising)",
                 fontsize=S.FS_LABEL, color=S.INK, y=0.995)
    fig.text(0.5, 0.08, "EC fate bias", ha="center", va="bottom", fontsize=S.FS_LABEL,
             color=S.INK)
    fig.tight_layout(rect=(0, 0.11, 1, 0.98))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"manual_transient_switch_{TAG}.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
