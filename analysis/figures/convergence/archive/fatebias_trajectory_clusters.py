#!/usr/bin/env python
"""Discover fate-commitment gene classes by CLUSTERING trajectories (unsupervised).

fatebias_trajectory_distance.py hand-labels "commitment-indicator" genes with a
threshold: a gene qualifies if its smoothed trajectory sweeps at least COMMIT_RANGE
(max - min) along the fate-bias axis. That cut is arbitrary. This pilot asks whether
the same genes fall out on their own -- i.e. whether unsupervised clustering of the
trajectories RECOVERS the fate indicators as one cluster, with no threshold at all.

The object clustered is one trajectory PER GENE: expression smoothed as a function of
EC fate bias, POOLED over all source cell types (the same gaussian-kernel smoothing as
the sibling pilots, applied to every node at once). Each gene's trajectory is centred
(subtract its own grid mean) and divided by the gene's expression SD -- exactly the
normalisation the sibling's per-gene L2 uses -- so what remains is the SHAPE and
AMPLITUDE (in gene-SD units) of how the gene moves with commitment, comparable across
genes on very different expression scales. The distance between two genes is then the
same "L2 thing" as the sibling, now taken gene-to-gene rather than source-to-source:
the RMS gap between their (centred, SD-normalised) trajectories over the shared grid.
Euclidean distance on these feature vectors IS that L2 (up to the constant 1/sqrt(grid
points)), so Ward linkage over the features clusters directly on the L2 metric.

No movement pre-filter is applied on purpose: ALL expressed genes go in, so the flat
majority is free to form its own cluster(s) and the moving genes must separate
themselves. Clusters are numbered by descending mean pooled slope, so cluster 1 is the
strongest "rises with commitment" group -- the candidate fate-indicator cluster. As a
check on whether the clustering reproduces the hand cut, the script reports, per
cluster, the share of the sibling's threshold commitment-indicator genes (pooled
trajectory range >= COMMIT_RANGE) that land in it.

Pooled (source-agnostic) trajectories are used here because the question is only "do
the commitment genes cluster", and the pooled line is the best-populated / cleanest
readout of that. Concatenating the per-source trajectories (to also split core from
path-specific) is the natural next step once this recovers the indicators.

Outputs (analysis/results/fatebias_trajectory_clusters/):
  trajectory_clusters.{png,svg}        one panel per cluster: all member trajectories
                                       (grey, wispy) + cluster mean + a few TF names
  cluster_means.{png,svg}              every cluster's mean trajectory on one axis
  trajectory_cluster_assignments.csv   per-gene cluster / slope / range / is_tf

    python analysis/figures/convergence/fatebias_trajectory_clusters.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

FIG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIG)
import style as S                          # noqa: E402
import fatebias as FB                      # noqa: E402
# reuse the sibling's node scan (EC descendants kept), vectorised smoother and TF reader
# so the trajectory definition is IDENTICAL to fatebias_trajectory_distance.py.
import fatebias_trajectory_distance as TD  # noqa: E402

KEY = "ec"
WIN_LO, WIN_HI = S.BIN_LO, S.BIN_HI
N_CLUSTERS = 6                  # number of trajectory "types" to cut the tree into
# feature construction, from raw -> fully standardised:
#   CENTER       subtract each gene's trajectory mean (drop absolute level, keep shape).
#                False = cluster on the raw log trajectory, so overall expression LEVEL
#                also separates genes (a high flat gene sits apart from a low flat one).
#   SD_NORMALIZE divide by the gene's node-level SD (common scale across genes). shrinks a
#                big-swing / noisy gene like Etv2, so leave off to judge genes on raw
#                log movement.
CENTER = True
SD_NORMALIZE = False
COMMIT_RANGE = TD.COMMIT_RANGE  # sibling's hand cut, used ONLY as a recovery diagnostic
N_TF_LABEL = 6                  # TF names printed per cluster panel (strongest movers)
MAX_WISPY = 400                 # grey member lines drawn per panel (subsampled if more)
EPS = 1e-9

# fresh qualitative palette for the cluster mean lines (10 evenly hue-spaced entries, so
# no wrap at N_CLUSTERS=10): not tab10/20 and not reused from the convergence figure's
# row-3 source palette or the sibling's teal distance map. clusters are slope-ordered, so
# this ramps warm (rising) -> cool (falling).
CLUSTER_PALETTE = ["#c44536", "#d98a3d", "#c9b02e", "#7fa03a", "#3f9e5a", "#2f9188",
                   "#3b6ea5", "#5b5fb0", "#8a55b0", "#bd5a8f"]

RESULTS = os.path.join(S.base_dir, "analysis", "results", "fatebias_trajectory_clusters")
os.makedirs(RESULTS, exist_ok=True)

# outputs are tagged by feature mode so the variants coexist for comparison
TAG = (("sdnorm" if CENTER else "sdraw") if SD_NORMALIZE
       else ("logdev" if CENTER else "raw"))
YLAB = (("(expr - traj mean) / gene SD" if CENTER else "expr / gene SD") if SD_NORMALIZE
        else ("expr - traj mean (log)" if CENTER else "expr (log)"))


def direction(slope, thr):
    """coarse label for a cluster's mean pooled slope."""
    if slope > thr:
        return "rises"
    if slope < -thr:
        return "falls"
    return "flat"


def main():
    family = S.FANIN[KEY]["exclude"]
    print(f"[1/5] {KEY}: internal nodes per source E[{WIN_LO}, {WIN_HI}), "
          f"EC descendants kept ...", flush=True)
    td, e95, assignments, subtype, sub2type = FB.load_trees()
    cells, stypes, bias = TD.scan(td, e95, assignments, subtype, sub2type,
                                  family, WIN_LO, WIN_HI)
    print(f"      {len(cells):,} internal nodes (bias {bias.min():.2f}..{bias.max():.2f})",
          flush=True)

    print("[2/5] reading expression ...", flush=True)
    Xp, ok, genes = FB.read_expression(cells)
    bias = bias[ok]
    expressed = Xp.mean(0) >= S.MIN_EXPR
    Xe, gnames = Xp[:, expressed], genes[expressed]
    std_g = Xe.std(0)
    print(f"      {Xe.shape[0]:,} nodes x {Xe.shape[1]:,} expressed genes", flush=True)

    print("[3/5] pooled trajectory per gene + L2 feature matrix ...", flush=True)
    grid = np.linspace(0.0, 1.0, TD.N_GRID)
    pmu, _, pval = TD.smooth_all(grid, bias, Xe.astype(np.float32))  # (n_grid, G)
    Yv = pmu[pval]                                                   # (n_valid, G)
    gi = grid[pval]

    # feature per gene = mean-centred pooled trajectory over the grid points where the
    # smoother is defined; optionally divided by the gene's node-level SD. Euclidean
    # distance on these rows is the per-gene L2 (RMS gap), gene-to-gene, so a flat gene
    # sits near the origin and a strong mover sits far out along its shape.
    feat = Yv.T.copy()                                             # (G, n_valid), raw log
    if CENTER:
        feat = feat - feat.mean(1, keepdims=True)                   # drop level, keep shape
    if SD_NORMALIZE:
        feat = feat / (std_g[:, None] + EPS)                        # common (gene-SD) scale

    # pooled slope / range for ordering, labelling and the recovery diagnostic
    gc = gi - gi.mean()
    slope = (Yv.T @ gc) / (gc @ gc + EPS)                          # d(expr)/d(bias), pooled
    grange = Yv.max(0) - Yv.min(0)                                  # sibling's max - min cut
    commit = grange >= COMMIT_RANGE

    print(f"[4/5] Ward clustering on the L2 feature matrix -> {N_CLUSTERS} clusters ...",
          flush=True)
    Z = linkage(feat, method="ward")                               # euclidean == L2 metric
    raw = fcluster(Z, N_CLUSTERS, criterion="maxclust")
    labels = np.unique(raw)
    # renumber clusters 1..K by descending mean pooled slope: cluster 1 = strongest riser
    order = sorted(labels, key=lambda c: -slope[raw == c].mean())
    remap = {old: new for new, old in enumerate(order, start=1)}
    cid = np.array([remap[c] for c in raw])

    tfs = TD.read_tf_list(S.TF_LIST)
    is_tf = np.array([g in tfs for g in gnames])
    slope_thr = 0.5 * np.abs(slope).mean()                         # dead-zone for "flat"

    members, tf_top, summary = {}, {}, []
    print(f"      {int(commit.sum()):,} threshold commitment genes "
          f"(pooled range >= {COMMIT_RANGE}) to recover", flush=True)
    for c in range(1, N_CLUSTERS + 1):
        m = np.where(cid == c)[0]
        members[c] = m
        tf_m = m[is_tf[m]]
        tf_m = tf_m[np.argsort(-np.abs(slope[tf_m]))]              # strongest movers first
        tf_top[c] = [gnames[g] for g in tf_m[:N_TF_LABEL]]
        n_commit = int(commit[m].sum())
        summary.append((c, len(m), int(is_tf[m].sum()), slope[m].mean(),
                        grange[m].mean(), n_commit))
        print(f"      cluster {c}: n={len(m):4d}  tf={int(is_tf[m].sum()):3d}  "
              f"mean_slope={slope[m].mean():+.3f}  mean_range={grange[m].mean():.3f}  "
              f"commit={n_commit:3d}/{int(commit.sum())}  "
              f"[{direction(slope[m].mean(), slope_thr)}]", flush=True)
        print(f"        TFs: " + ", ".join(tf_top[c]), flush=True)

    # recovery headline: the cluster with the highest mean trajectory range is the
    # discovered "commitment" cluster; report what share of the hand cut it captures.
    disc = max(range(1, N_CLUSTERS + 1), key=lambda c: grange[members[c]].mean())
    if commit.sum():
        recov = commit[members[disc]].sum() / commit.sum()
        print(f"      -> cluster {disc} has the highest mean range; it holds "
              f"{recov:.0%} of the {int(commit.sum())} threshold commitment genes",
              flush=True)
    # spotlight a few canonical EC master TFs (Etv2 especially): which cluster caught them
    for probe in ("Etv2", "Fli1", "Sox18", "Lmo2"):
        w = np.where(gnames == probe)[0]
        if len(w):
            g = w[0]
            print(f"      probe {probe:5s}: cluster {cid[g]}  "
                  f"slope={slope[g]:+.3f}  range={grange[g]:.3f}  "
                  f"nodeSD={std_g[g]:.3f}", flush=True)

    pd.DataFrame({
        "gene": gnames,
        "cluster": cid,
        "pooled_slope": slope,
        "pooled_range": grange,
        "over_commit_range": commit,
        "is_tf": is_tf,
    }).sort_values(["cluster", "pooled_slope"],
                   ascending=[True, False]).to_csv(
        os.path.join(RESULTS, f"trajectory_cluster_assignments_{TAG}.csv"), index=False)

    print("[5/5] plotting ...", flush=True)
    S.apply_rc()
    rng = np.random.default_rng(S.SEED)
    plot_clusters(gi, feat, members, tf_top, slope, grange, is_tf, gnames,
                  slope_thr, disc, rng)
    plot_means(gi, feat, members, slope, slope_thr)

    print(f"done ({TAG}). wrote:", flush=True)
    print(f"      trajectory_clusters_{TAG}.png/.svg, cluster_means_{TAG}.png/.svg",
          flush=True)
    print(f"      trajectory_cluster_assignments_{TAG}.csv", flush=True)


def plot_clusters(gi, feat, members, tf_top, slope, grange, is_tf, gnames,
                  slope_thr, disc, rng):
    """one panel per cluster: every member trajectory as a grey wispy line, the cluster
    mean over the top, and a few of the cluster's TFs (strongest movers) named. the
    discovered commitment cluster (highest mean range) is flagged in its title."""
    K = len(members)
    ncol = 3
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.PAGE_W_IN, 2.0 * nrow + 0.7),
                             squeeze=False)
    lo = min(feat[m].min() for m in members.values() if len(members))
    hi = max(feat[m].max() for m in members.values() if len(members))
    for c in range(1, K + 1):
        ax = axes[(c - 1) // ncol][(c - 1) % ncol]
        m = members[c]
        col = CLUSTER_PALETTE[(c - 1) % len(CLUSTER_PALETTE)]
        draw = m if len(m) <= MAX_WISPY else rng.choice(m, MAX_WISPY, replace=False)
        for g in draw:
            ax.plot(gi, feat[g], color=S.GREY, lw=0.35, alpha=0.25, zorder=1)
        ax.plot(gi, feat[m].mean(0), color=col, lw=2.2, zorder=5)
        ax.axhline(0, color=S.MUTED, lw=0.5, ls=(0, (3, 3)), zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.5, 1.0])
        S.despine(ax)
        tag = "  *commitment*" if c == disc else ""
        ax.set_title(f"cluster {c}  (n={len(m)}, {direction(slope[m].mean(), slope_thr)})"
                     f"{tag}", fontsize=S.FS_LABEL, color=col, pad=4)
        # name a few TFs inside the panel, stacked from the top corner
        names = tf_top[c] if tf_top[c] else ["(no TFs)"]
        ax.text(0.04, 0.96, "\n".join(names), transform=ax.transAxes,
                ha="left", va="top", fontsize=S.FS_ANNOT, color=S.INK,
                linespacing=1.3)
        if (c - 1) % ncol == 0:
            ax.set_ylabel(YLAB, fontsize=S.FS_LABEL)
    for k in range(K, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.text(0.5, 0.015, "EC fate bias", ha="center", va="bottom",
             fontsize=S.FS_LABEL, color=S.INK)
    fig.text(0.5, 0.001, "trajectories clustered by L2 shape (pooled over sources); "
             "grey = member genes, colour = cluster mean", ha="center", va="bottom",
             fontsize=S.FS_ANNOT, color=S.MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"trajectory_clusters_{TAG}.{ext}"), dpi=300)
    plt.close(fig)


def plot_means(gi, feat, members, slope, slope_thr):
    """every cluster's mean trajectory on a single axis -- the distinct commitment
    'types' the clustering pulled apart, side by side."""
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.axhline(0, color=S.MUTED, lw=0.5, ls=(0, (3, 3)), zorder=0)
    for c in sorted(members):
        m = members[c]
        col = CLUSTER_PALETTE[(c - 1) % len(CLUSTER_PALETTE)]
        ax.plot(gi, feat[m].mean(0), color=col, lw=1.8, zorder=5,
                label=f"{c}: {direction(slope[m].mean(), slope_thr)} (n={len(m)})")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("EC fate bias", fontsize=S.FS_LABEL)
    ax.set_ylabel(YLAB, fontsize=S.FS_LABEL)
    S.despine(ax)
    ax.legend(frameon=False, fontsize=S.FS_LEGEND, loc="best", handlelength=1.4,
              labelspacing=0.3)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"cluster_means_{TAG}.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
