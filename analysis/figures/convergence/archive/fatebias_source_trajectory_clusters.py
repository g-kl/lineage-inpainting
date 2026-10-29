#!/usr/bin/env python
"""SOURCE-SPECIFIC version of the trajectory clustering (fatebias_trajectory_clusters.py).

That pilot clusters one POOLED trajectory per gene (expression smoothed vs EC fate bias,
all source cell types thrown together). This one keeps the sources apart: it builds one
trajectory per (gene, source) pair -- gene g's expression vs fate bias among ONLY the
nodes whose progenitor was source s -- and clusters the whole pool of (gene, source)
trajectories TOGETHER. Because every source's trajectories go into one Ward clustering,
the cluster labels mean the same thing in every source, so each gene gets a cluster label
PER SOURCE and we can read off directly how a gene's trajectory type changes between
sources:
  - same cluster in every source        -> the gene follows one shared recipe (core)
  - a rising cluster in one source, flat -> the gene rises to EC only from that source
    or falling in another                   (path-specific / source residue)

Feature construction (raw / centred / SD-normalised) and the cluster count are set LOCALLY
here (CENTER, SD_NORMALIZE, N_CLUSTERS), decoupled from the pooled sibling: this pilot
clusters on the raw log trajectory (no mean baseline, CENTER=False) cut into 6 types, so an
"always off" trajectory forms its own cluster distinct from an "always on" one and switches
between the two are legible. The node scan (TD.scan, EC descendants
kept so each source runs to fate bias = 1) and gaussian smoother (TD.smooth_all) are the
same as the sibling pilots; the SD used for normalisation is the global per-gene node SD
(shared across sources, so a gene sits on the same scale in every source). Per source a
few grid points can be undersampled -> the smoother returns NaN there; each source's block
is linearly interpolated over its own covered range (clamped at the ends) so Ward gets
complete vectors, and a source with < MIN_VALID covered grid points is dropped entirely.

Outputs (analysis/results/fatebias_source_trajectory_clusters/):
  source_trajectory_clusters_<tag>.{png,svg}   per-cluster panels: every member
                                               (gene, source) trajectory + mean + TF names
  source_hued_trajectories_<tag>.{png,svg}     example genes (one panel each, lines hued by
                                               source) for four recipe classes read off the
                                               switch pair (SWITCH_SRC): CORE (rises ->
                                               rises), two PATH-SPECIFIC classes (flat ->
                                               rises, falls -> flat), and MEMORY (hi-flat ->
                                               lo-flat -- residual expression kept on one path)
  cluster_by_source_<tag>.{png,svg}            focus TFs x source, cell = assigned cluster
                                               (does a gene keep its cluster across sources?)
  source_cluster_agreement_<tag>.{png,svg}     source x source: fraction of genes put in
                                               the SAME cluster (a clustering-based twin of
                                               the sibling's source-similarity heatmap)
  cluster_switch_<tag>.{png,svg}               somatic mesoderm cluster (rows) x
                                               dermomyotome cluster (cols), cell = # genes
                                               with that cluster pair -- the most common
                                               switches; diagonal = kept recipe ('memory')
  source_cluster_assignments_<tag>.csv         gene x source cluster label matrix

    python analysis/figures/convergence/fatebias_source_trajectory_clusters.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.interpolate import interp1d

FIG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIG)
import style as S                          # noqa: E402
import fatebias as FB                      # noqa: E402
import fatebias_trajectory_distance as TD  # noqa: E402  (scan / smooth_all / constants)
import fatebias_trajectory_clusters as CL  # noqa: E402  (feature mode / palette / helpers)

KEY = "ec"
WIN_LO, WIN_HI = S.BIN_LO, S.BIN_HI
N_CLUSTERS = 6                   # cut the shared Ward tree into 6 trajectory types
N_SOURCES = TD.N_SOURCES         # canonical EC fan-in sources (S.SOURCE_TYPES order)
MIN_VALID = TD.MIN_OVERLAP       # a source needs this many covered grid points to be kept
N_FOCUS = 40                     # rows in the cluster-by-source TF comparison heatmap
N_EX = 5                         # example genes shown per recipe class (source-hued panels)
SWITCH_SRC = ("somatic mesoderm", "dermomyotome")  # the two paths the switch map / recipe
                                 # classes compare (row source -> col source)
MEMORY_PAIR = (6, 3)             # the "memory" class: (row-source cluster -> col-source
                                 # cluster) on SWITCH_SRC. 6 -> 3 = high plateau on the row
                                 # path dropping to the low-flat majority on the col path
EPS = CL.EPS

# feature mode for THIS pilot, decoupled from the sibling so the source split can cluster
# on the raw log trajectory (no mean baseline) while the pooled pilot keeps its own mode.
#   CENTER=False  -> no mean baseline subtracted: absolute expression LEVEL also separates
#                    genes (a high-flat gene sits apart from a low-flat one), so an "always
#                    off" trajectory forms its own cluster distinct from an "always on" one.
CENTER = False
SD_NORMALIZE = False

_MODE = (("sdnorm" if CENTER else "sdraw") if SD_NORMALIZE
         else ("logdev" if CENTER else "raw"))
TAG = f"bysource_{_MODE}"
YLAB = (("(expr - traj mean) / gene SD" if CENTER else "expr / gene SD") if SD_NORMALIZE
        else ("expr - traj mean (log)" if CENTER else "expr (log)"))

RESULTS = os.path.join(S.base_dir, "analysis", "results",
                       "fatebias_source_trajectory_clusters")
os.makedirs(RESULTS, exist_ok=True)


def build_features(grid, bias, Xe, stypes, sources, std_g):
    """one (gene, source) trajectory per row. per source: gaussian-smooth every gene vs
    fate bias, interpolate the undersampled grid points over the covered range, then apply
    the sibling's feature mode (raw / centre / SD). returns the stacked feature matrix and,
    per row, its gene index, source index, pooled slope and range."""
    gc = grid - grid.mean()
    feats, r_gene, r_src, r_slope, r_range = [], [], [], [], []
    kept = []
    for si, s in enumerate(sources):
        m = stypes == s
        mu, _, valid = TD.smooth_all(grid, bias[m], Xe[m].astype(np.float32))
        if valid.sum() < MIN_VALID:
            print(f"      ! dropping source {s}: only {int(valid.sum())} covered grid "
                  f"points (< {MIN_VALID})", flush=True)
            continue
        gi = grid[valid]
        f = interp1d(gi, mu[valid], axis=0, bounds_error=False,
                     fill_value=(mu[valid][0], mu[valid][-1]))
        traj = f(grid)                                        # (n_grid, G), NaN-free
        feat = traj.T.copy()                                  # (G, n_grid), raw log
        if CENTER:
            feat = feat - feat.mean(1, keepdims=True)
        if SD_NORMALIZE:
            feat = feat / (std_g[:, None] + EPS)
        feats.append(feat)
        r_gene.append(np.arange(Xe.shape[1]))
        r_src.append(np.full(Xe.shape[1], len(kept)))         # index into `kept`, not `sources`
        r_slope.append((traj.T @ gc) / (gc @ gc + EPS))
        r_range.append(traj.max(0) - traj.min(0))
        kept.append(s)
    feat = np.concatenate(feats, 0)
    return (feat, np.concatenate(r_gene), np.concatenate(r_src),
            np.concatenate(r_slope), np.concatenate(r_range), kept)


def main():
    family = S.FANIN[KEY]["exclude"]
    print(f"[1/5] {KEY}: internal nodes per source E[{WIN_LO}, {WIN_HI}), "
          f"EC descendants kept ...", flush=True)
    td, e95, assignments, subtype, sub2type = FB.load_trees()
    cells, stypes, bias = TD.scan(td, e95, assignments, subtype, sub2type,
                                  family, WIN_LO, WIN_HI)
    print(f"      {len(cells):,} internal nodes", flush=True)

    print("[2/5] reading expression ...", flush=True)
    Xp, ok, genes = FB.read_expression(cells)
    stypes, bias = stypes[ok], bias[ok]
    expressed = Xp.mean(0) >= S.MIN_EXPR
    Xe, gnames = Xp[:, expressed], genes[expressed]
    std_g = Xe.std(0)
    G = Xe.shape[1]

    vc = pd.Series(stypes).value_counts()
    sources = [s for s in S.SOURCE_TYPES if s in vc.index][:N_SOURCES]
    print(f"      {Xe.shape[0]:,} nodes x {G:,} expressed genes; sources: "
          + ", ".join(f"{s} ({vc[s]})" for s in sources), flush=True)

    print("[3/5] per-(gene, source) trajectories + feature matrix "
          f"[mode={_MODE}] ...", flush=True)
    grid = np.linspace(0.0, 1.0, TD.N_GRID)
    feat, r_gene, r_src, r_slope, r_range, sources = build_features(
        grid, bias, Xe, stypes, sources, std_g)
    n_src = len(sources)
    print(f"      {feat.shape[0]:,} (gene, source) trajectories over {n_src} sources",
          flush=True)

    print(f"[4/5] Ward clustering on the shared L2 feature matrix -> {N_CLUSTERS} "
          "clusters ...", flush=True)
    Z = linkage(feat, method="ward")
    raw = fcluster(Z, N_CLUSTERS, criterion="maxclust")
    order = sorted(np.unique(raw), key=lambda c: -r_slope[raw == c].mean())
    remap = {old: new for new, old in enumerate(order, start=1)}
    cid = np.array([remap[c] for c in raw])

    tfs = TD.read_tf_list(S.TF_LIST)
    is_tf = np.array([g in tfs for g in gnames])
    slope_thr = 0.5 * np.abs(r_slope).mean()
    cdir = {c: CL.direction(r_slope[cid == c].mean(), slope_thr)
            for c in range(1, N_CLUSTERS + 1)}

    members, tf_top = {}, {}
    for c in range(1, N_CLUSTERS + 1):
        m = np.where(cid == c)[0]
        members[c] = m
        gtf = r_gene[m][is_tf[r_gene[m]]]
        uniq = pd.unique(gtf[np.argsort(-np.abs(r_slope[m][is_tf[r_gene[m]]]))])
        tf_top[c] = [gnames[g] for g in uniq[:CL.N_TF_LABEL]]
        print(f"      cluster {c}: rows={len(m):5d}  genes={len(np.unique(r_gene[m])):4d}"
              f"  mean_slope={r_slope[m].mean():+.3f}  [{cdir[c]}]", flush=True)
        print(f"        TFs: " + ", ".join(tf_top[c]), flush=True)

    # gene x source cluster label matrix (0 = source not covered / gene absent there)
    lab = np.zeros((G, n_src), int)
    lab[r_gene, r_src] = cid

    # (gene, source) -> feature row + per-source slope, for the source-hued example panels
    row_idx = np.full((G, n_src), -1, int)
    row_idx[r_gene, r_src] = np.arange(feat.shape[0])
    slope_mat = np.full((G, n_src), np.nan)
    slope_mat[r_gene, r_src] = r_slope
    level_mat = np.full((G, n_src), np.nan)             # per-(gene, source) mean expr level
    level_mat[r_gene, r_src] = feat.mean(1)             # separates hi- vs lo-flat plateaus

    # concrete text readout for the canonical EC / mesoderm TFs
    print("      per-source cluster of marker TFs (0 = source uncovered):", flush=True)
    for name in ("Fli1", "Sox18", "Lmo2", "Erg", "Etv2", "Tal1", "Gata2", "Kdr",
                 "Prrx1", "Meox1"):
        w = np.where(gnames == name)[0]
        if len(w):
            row = lab[w[0]]
            print(f"        {name:6s}: " + "  ".join(
                f"{s.split()[0]}={row[si]}" for si, s in enumerate(sources)), flush=True)

    pd.DataFrame(lab, columns=[s.replace(" ", "_") for s in sources]).assign(
        gene=gnames, is_tf=is_tf,
        n_distinct=[len(set(r[r > 0])) for r in lab]).to_csv(
        os.path.join(RESULTS, f"source_cluster_assignments_{TAG}.csv"), index=False)

    print("[5/5] plotting ...", flush=True)
    S.apply_rc()
    rng = np.random.default_rng(S.SEED)
    plot_clusters(grid, feat, members, tf_top, r_slope, cdir, rng)
    plot_cluster_by_source(lab, sources, gnames, is_tf, r_slope, cid, r_gene,
                           slope_thr, cdir)
    plot_agreement(lab, sources)
    plot_cluster_switch(lab, sources, cdir)
    classes = select_recipe_classes(lab, cdir, sources, slope_mat, level_mat, is_tf)
    for title, glist in classes:
        label = title.replace(chr(10), " ").replace(r"$\rightarrow$", "->")
        print(f"      {label}: {len(glist)} genes"
              + (" e.g. " + ", ".join(gnames[g] for g in glist[:N_EX]) if glist else ""),
              flush=True)
    plot_source_hued(grid, feat, row_idx, sources, gnames, classes)

    print(f"done ({TAG}). wrote:", flush=True)
    print(f"      source_trajectory_clusters_{TAG}.png/.svg", flush=True)
    print(f"      source_hued_trajectories_{TAG}.png/.svg", flush=True)
    print(f"      cluster_by_source_{TAG}.png/.svg", flush=True)
    print(f"      source_cluster_agreement_{TAG}.png/.svg", flush=True)
    print(f"      cluster_switch_{TAG}.png/.svg", flush=True)
    print(f"      source_cluster_assignments_{TAG}.csv", flush=True)


def plot_clusters(grid, feat, members, tf_top, r_slope, cdir, rng):
    """one panel per cluster: every member (gene, source) trajectory grey + the cluster
    mean + a few TF names. same layout as the pooled sibling, rows are (gene, source)."""
    K = len(members)
    ncol = 3
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.PAGE_W_IN, 2.0 * nrow + 0.7),
                             squeeze=False)
    lo = min(feat[m].min() for m in members.values() if len(m))
    hi = max(feat[m].max() for m in members.values() if len(m))
    for c in range(1, K + 1):
        ax = axes[(c - 1) // ncol][(c - 1) % ncol]
        m = members[c]
        col = CL.CLUSTER_PALETTE[(c - 1) % len(CL.CLUSTER_PALETTE)]
        draw = m if len(m) <= CL.MAX_WISPY else rng.choice(m, CL.MAX_WISPY, replace=False)
        for r in draw:
            ax.plot(grid, feat[r], color=S.GREY, lw=0.35, alpha=0.22, zorder=1)
        ax.plot(grid, feat[m].mean(0), color=col, lw=2.2, zorder=5)
        ax.axhline(0, color=S.MUTED, lw=0.5, ls=(0, (3, 3)), zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(lo, hi)
        ax.set_xticks([0, 0.5, 1.0])
        S.despine(ax)
        ax.set_title(f"cluster {c}  ({len(m)} rows, {cdir[c]})",
                     fontsize=S.FS_LABEL, color=col, pad=4)
        names = tf_top[c] if tf_top[c] else ["(no TFs)"]
        ax.text(0.04, 0.96, "\n".join(names), transform=ax.transAxes,
                ha="left", va="top", fontsize=S.FS_ANNOT, color=S.INK, linespacing=1.3)
        if (c - 1) % ncol == 0:
            ax.set_ylabel(YLAB, fontsize=S.FS_LABEL)
    for k in range(K, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.text(0.5, 0.015, "EC fate bias", ha="center", va="bottom",
             fontsize=S.FS_LABEL, color=S.INK)
    fig.text(0.5, 0.001, "per-(gene, source) trajectories clustered together (shared "
             "labels); grey = members, colour = cluster mean", ha="center", va="bottom",
             fontsize=S.FS_ANNOT, color=S.MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"source_trajectory_clusters_{TAG}.{ext}"),
                    dpi=300)
    plt.close(fig)


def plot_cluster_by_source(lab, sources, gnames, is_tf, r_slope, cid, r_gene,
                           slope_thr, cdir):
    """the direct comparison: focus TFs (rows) x source (cols), each cell coloured by the
    cluster that gene's trajectory falls into in that source. a row of one colour = the
    gene keeps its recipe across sources (core); a row that changes colour = source-
    specific behaviour. focus = TFs that reach a RISING cluster in >=1 source."""
    K = len(cdir)
    rise = {c for c in cdir if cdir[c] == "rises"}
    # rank a gene by how strongly it rises (max rising-row slope), keep TFs hitting a riser
    best = np.full(len(gnames), -np.inf)
    for r in range(len(cid)):
        if cid[r] in rise:
            best[r_gene[r]] = max(best[r_gene[r]], r_slope[r])
    cand = np.where(is_tf & np.isfinite(best) & (best > -np.inf))[0]
    cand = cand[np.argsort(-best[cand])][:N_FOCUS]
    if len(cand) == 0:
        return
    M = lab[cand].astype(float)                              # (n_focus, n_src), 0 = missing
    M[M == 0] = np.nan
    keys = [np.where(np.isnan(M[:, j]), K + 1, M[:, j]) for j in range(M.shape[1])]
    row_order = np.lexsort(keys[::-1])                       # group similar label patterns
    cand, M = cand[row_order], M[row_order]

    cmap = ListedColormap(CL.CLUSTER_PALETTE[:K])
    cmap.set_bad(S.UMAP_BG)
    norm = BoundaryNorm(np.arange(0.5, K + 1.5), K)
    fig, ax = plt.subplots(figsize=(0.5 * len(sources) + 2.2, 0.17 * len(cand) + 1.1))
    ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in sources], fontsize=S.FS_TICK,
                       rotation=35, ha="right")
    ax.set_yticks(range(len(cand)))
    ax.set_yticklabels([gnames[g] for g in cand], fontsize=max(S.FS_TICK - 1.5, 5.0))
    for c in range(len(cand) + 1):
        ax.axhline(c - 0.5, color="white", lw=0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Line2D([0], [0], marker="s", ls="", ms=6,
                          color=CL.CLUSTER_PALETTE[c - 1],
                          label=f"{c}: {cdir[c]}") for c in range(1, K + 1)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=S.FS_LEGEND, handletextpad=0.3, labelspacing=0.25,
              title="cluster", title_fontsize=S.FS_LEGEND)
    ax.set_title("trajectory cluster per source (rising TFs)", fontsize=S.FS_LABEL,
                 color=S.INK, pad=6)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"cluster_by_source_{TAG}.{ext}"), dpi=300)
    plt.close(fig)


def plot_agreement(lab, sources):
    """source x source: among genes covered in BOTH sources, the fraction assigned to the
    same cluster. a clustering-label twin of the sibling distance heatmap's question."""
    n = len(sources)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            both = (lab[:, i] > 0) & (lab[:, j] > 0)
            if both.sum():
                M[i, j] = (lab[both, i] == lab[both, j]).mean()
    fig, ax = plt.subplots(figsize=(0.5 * n + 1.9, 0.5 * n + 1.7))
    off = M[np.isfinite(M)]
    im = ax.imshow(M, cmap=TD.DIST_CMAP, vmin=float(off.min()), vmax=float(off.max()))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    lab_s = [s.replace(" ", "\n") for s in sources]
    ax.set_xticklabels(lab_s, rotation=35, ha="right", fontsize=S.FS_TICK)
    ax.set_yticklabels(lab_s, fontsize=S.FS_TICK)
    thr = float(off.min()) + 0.6 * (float(off.max()) - float(off.min()))
    for i in range(n):
        for j in range(n):
            txt = "-" if i == j else f"{M[i, j] * 100:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=S.FS_ANNOT,
                    color="white" if np.isfinite(M[i, j]) and M[i, j] > thr else S.INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=S.FS_TICK)
    cb.set_label("genes in the same cluster", fontsize=S.FS_LABEL)
    cb.outline.set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"source_cluster_agreement_{TAG}.{ext}"),
                    dpi=300)
    plt.close(fig)


def plot_cluster_switch(lab, sources, cdir, row_src=SWITCH_SRC[0],
                        col_src=SWITCH_SRC[1]):
    """cluster-switch matrix between two chosen sources. rows = the gene's cluster in the
    row source (somatic mesoderm), cols = its cluster in the col source (dermomyotome),
    cell = how many genes have that (row -> col) cluster pair, counting only genes covered
    in BOTH sources. the boxed diagonal is genes that keep their trajectory type across the
    two paths (shared recipe / 'memory'); off-diagonal cells are the common switches -- a
    gene in an 'off' cluster on one path but a rising cluster on the other is memory kept vs
    erased. colour is a sqrt (PowerNorm) ramp so the smaller off-diagonal switch cells stay
    visible next to a large flat-cluster diagonal; the printed number is the raw count."""
    try:
        ri = next(i for i, s in enumerate(sources) if s == row_src)
        ci = next(i for i, s in enumerate(sources) if s == col_src)
    except StopIteration:
        print(f"      ! {row_src} / {col_src} not both kept; skipping switch map",
              flush=True)
        return
    K = len(cdir)
    both = (lab[:, ri] > 0) & (lab[:, ci] > 0)
    M = np.zeros((K, K), int)
    for a, b in zip(lab[both, ri], lab[both, ci]):
        M[a - 1, b - 1] += 1

    fig, ax = plt.subplots(figsize=(0.55 * K + 2.2, 0.55 * K + 2.0))
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=float(M.max()) or 1.0)
    im = ax.imshow(M, cmap=TD.DIST_CMAP, norm=norm)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"{c}\n{cdir[c]}" for c in range(1, K + 1)], fontsize=S.FS_TICK)
    ax.set_yticklabels([f"{c} {cdir[c]}" for c in range(1, K + 1)], fontsize=S.FS_TICK)
    ax.set_xlabel(f"{col_src} cluster", fontsize=S.FS_LABEL)
    ax.set_ylabel(f"{row_src} cluster", fontsize=S.FS_LABEL)
    for i in range(K):
        for j in range(K):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=S.FS_ANNOT,
                        color="white" if norm(M[i, j]) > 0.55 else S.INK)
    # box the diagonal: genes that keep their cluster (shared recipe / retained memory)
    for d in range(K):
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
    ax.set_title(f"cluster switches: {row_src} to {col_src}\n"
                 f"{n_same}/{int(both.sum())} genes keep their cluster",
                 fontsize=S.FS_LABEL, color=S.INK, pad=6)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"cluster_switch_{TAG}.{ext}"), dpi=300)
    plt.close(fig)


def select_recipe_classes(lab, cdir, sources, slope_mat, level_mat, is_tf):
    """read four recipe classes straight off the cluster-switch pair (SWITCH_SRC): the
    gene's trajectory in the row source -> its trajectory in the col source, for genes
    covered in BOTH.
      CORE            -- rises -> rises: a rising recipe shared by both paths.
      path-specific 1 -- flat  -> rises: only the col path turns the gene on.
      path-specific 2 -- falls -> flat : the row path turns it off, the col path never had it.
      MEMORY          -- the explicit cluster transition MEMORY_PAIR (row cluster -> col
                         cluster): a high plateau on the row path dropping to the low-flat
                         majority on the col path -- residual expression carried (remembered)
                         on one path only.
    a clean two-source read (no all-source counting), so the classes match the switch map's
    diagonal / off-diagonal cells directly. each list is ranked TFs first, then by the
    slope (or, for memory, the retained row-source level) in the source that is changing.
    returns a list of (title, gene_indices)."""
    ri, ci = [sources.index(s) if s in sources else -1 for s in SWITCH_SRC]
    core, p1, p2, mem = [], [], [], []
    arr = r"$\rightarrow$"                           # cmss10 mangles a literal '>'
    mrow, mcol = MEMORY_PAIR
    titles = (f"core\n(rises {arr} rises)", f"path-specific\n(flat {arr} rises)",
              f"path-specific\n(falls {arr} flat)",
              f"memory\n(clus {mrow} {arr} {mcol})")
    if ri < 0 or ci < 0:
        return list(zip(titles, (core, p1, p2, mem)))
    for g in range(lab.shape[0]):
        a, b = lab[g, ri], lab[g, ci]
        if a == 0 or b == 0:                        # need both switch sources covered
            continue
        da, db = cdir[a], cdir[b]
        if a == mrow and b == mcol:                 # explicit pair first: it is a special
            mem.append(g)                           # case of falls->flat, pull it out
        elif da == "rises" and db == "rises":
            core.append(g)
        elif da == "flat" and db == "rises":
            p1.append(g)
        elif da == "falls" and db == "flat":
            p2.append(g)

    def rank(glist, mat, src, sign):
        return sorted(glist, key=lambda g: (not is_tf[g], -sign * mat[g, src]))

    ranked = (rank(core, slope_mat, ci, 1),   # steepest co-risers first
              rank(p1, slope_mat, ci, 1),      # steepest col-riser first
              rank(p2, slope_mat, ri, -1),     # steepest row-faller first
              rank(mem, level_mat, ri, 1))     # highest retained row-source plateau first
    return list(zip(titles, ranked))


def plot_source_hued(grid, feat, row_idx, sources, gnames, classes):
    """one panel per example gene, its per-source trajectories overlaid and HUED BY SOURCE,
    laid out as rows = recipe class (core / rising-in-2 / falling-in-2) x cols = example
    genes. y is autoscaled per gene (raw log level differs gene to gene); the point of each
    panel is how the SOURCES separate within one gene -- core genes stack together, path-
    specific genes fan apart. TFs are shown first within each class."""
    n_src = len(sources)
    src_col = [S.ROW3_SOURCE_PALETTE[i % len(S.ROW3_SOURCE_PALETTE)] for i in range(n_src)]
    nrow, ncol = len(classes), N_EX
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.PAGE_W_IN, 1.85 * nrow + 1.0),
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
                ax.set_ylabel(f"{title}  (n={len(glist)})", fontsize=S.FS_LABEL)
    handles = [plt.Line2D([0], [0], color=src_col[i], lw=2.2, label=sources[i])
               for i in range(n_src)]
    fig.legend(handles=handles, loc="lower center", ncol=min(n_src, 5), frameon=False,
               fontsize=S.FS_LEGEND, handlelength=1.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.10, "EC fate bias", ha="center", va="bottom",
             fontsize=S.FS_LABEL, color=S.INK)
    fig.text(0.5, 0.015, f"per-source trajectories hued by source ({YLAB}); recipe class "
             f"= direction {SWITCH_SRC[0]} " + r"$\rightarrow$" + f" {SWITCH_SRC[1]}",
             ha="center", va="bottom", fontsize=S.FS_ANNOT, color=S.MUTED)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"source_hued_trajectories_{TAG}.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
