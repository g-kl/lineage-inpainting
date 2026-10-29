#!/usr/bin/env python
"""Trajectory *distance* between source tissues along the EC fate-bias axis.

fatebias_axis_trajectories.py smooths each gene's expression as a function of EC fate bias,
one line per source cell type ("as a progenitor commits harder to EC, how does gene X move,
and do all sources move the same way?"). It then hand-classifies genes with a per-source
commitment t-statistic.

This pilot keeps the exact same smoothing (gaussian kernel over the fate-bias grid, one
line per source), but reframes everything around ONE object: the distance between two
trajectories, defined literally as the L2 (RMS over the shared grid) between the two
smoothed lines. From that per-gene, per-source-pair distance we extract three things:

  1. source x source similarity, over COMMITMENT-INDICATOR genes only. First keep the genes
     that track fate commitment: those whose smoothed trajectory sweeps at least COMMIT_RANGE
     (max - min across all sources and fate-bias grid points). Then the heatmap entry is the
     FRACTION of those genes whose two trajectories are "similar" (per-gene L2, normalized by
     the gene's expression SD, below SIM_CUTOFF) -- i.e. "of the genes that move with
     commitment, what share follow the same path between these two sources" (1 = all agree).
     The mean-L2 matrix over the same gene set is saved in the npz as a secondary reference.

  2. CORE EC genes -- low L2 between ALL source pairs (every tissue traces the same line)
     AND "interesting": expressed (mean level above MIN_EXPR) with a slope far from 0 (the
     shared line actually moves with commitment). These are the source-agnostic EC-fate
     program: every path converges on the same trajectory, and that trajectory goes
     somewhere. Ranked by |slope| among genes passing the low-divergence / moving filters.

  3. PATH-SPECIFIC genes -- one source sticks out: it is far (trajectory L2) from the others
     while the rest agree. Per gene, per source we take (mean L2 to the rest) - (mean pairwise
     L2 among the rest); the score is the max over sources and that source is responsible. The
     divergent genes are then BINNED by the direction of the responsible source's trajectory
     (ends higher = "gain", up in one fate; ends lower = "loss", identity decay) and the top
     few by L2 are drawn from each bin -- so the up-in-one-fate genes are not buried under the
     more numerous decay genes. A per-source grid then shows each tissue's top rising divergent
     genes (its private up-program) -- a readout of what is different between the sources.

Node scan differs from the sibling pilot in ONE way: we do not drop EC-typed nodes. The
sibling recipe fixes a progenitor's source cell type at the WIN_LO..WIN_HI slice and keeps
only that in-window node. Here we fix the source the same way (nearest in-window non-EC
ancestor) but then follow the lineage DOWN: every descendant node with >=1 EC descendant
leaf -- including the EC-typed nodes and the EC leaves themselves -- stays on its source's
trajectory. So an EC cell that descends from a somatic-mesoderm progenitor is a
somatic-mesoderm point at high fate bias, not a dropped node. Each source's line therefore
runs all the way to fate bias = 1 through its OWN EC descendants, and any convergence at the
top is real (different cells per source) rather than a shared endpoint forced onto them.

Outputs (analysis/results/fatebias_trajectory_distance/):
  source_distance_heatmap.{png,svg}      analysis 1: source x source normalized L2
  core_genes_trajectories.{png,svg}      analysis 2: shared moving genes, all sources drawn
                                         (each line runs into its own EC descendants)
  path_specific_gains.{png,svg}          analysis 3: genes UP in one source only (highlighted)
  path_specific_losses.{png,svg}         analysis 3: genes DOWN in one source only
  path_specific_by_source.{png,svg}      analysis 3: per-source top uniquely-rising genes
  trajectory_distance_scores.csv         per-gene scores (L2 / slope / gain & loss gaps)
  source_distance_matrix.npz             similar-fraction + mean-L2 source matrices

    python analysis/figures/convergence/fatebias_trajectory_distance.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

FIG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FIG)
import style as S                    # noqa: E402
import fatebias as FB                # noqa: E402  (descendant_frac / node_cell / read_expression)

KEY = "ec"
WIN_LO, WIN_HI = S.BIN_LO, S.BIN_HI   # progenitor window (shared style: 8.25..8.75)
N_SOURCES = 5                   # canonical EC fan-in sources kept (S.SOURCE_TYPES order)
BW_BIAS = 0.08                  # gaussian smoothing bandwidth, in fate-bias units
N_GRID = 60
MIN_OVERLAP = 5                 # min shared valid grid points to score a trajectory pair
EPS = 1e-9

# heatmap is computed over COMMITMENT-INDICATOR genes only: genes whose smoothed trajectory
# moves by at least COMMIT_RANGE (max - min taken across all sources AND fate-bias grid
# points, in log-expr units) are correlated with fate commitment; flat genes are dropped so
# the heatmap asks "of the genes that track commitment, how many follow the same path per
# source pair" rather than comparing the whole (mostly static) transcriptome.
COMMIT_RANGE = 0.5
# within that set, a gene's trajectory is "similar" between two sources if its normalized L2
# (RMS gap over the grid, in gene-SD units) is below this; the heatmap shows the fraction of
# commitment-indicator genes that clear it.
SIM_CUTOFF = 0.25

N_CORE = 6                      # core (shared + moving) genes drawn
N_PATH = 9                     # path-specific genes drawn per direction (gains / losses)
PER_SOURCE = 3                 # per-source path-specific genes in the by-source grid
AGREE_PCTL = 40                # core: keep genes whose max pairwise L2 is below this pctl
SLOPE_PCTL = 60                # core: keep genes whose |slope| is above this pctl

# fresh sequential map for the distance heatmap: light -> deep teal-green. Not reused from
# the convergence figure (which uses Blues / Purples / PuOr / a blue->accent diverging map).
DIST_CMAP = LinearSegmentedColormap.from_list("traj_dist",
                                              ["#f4f1ea", "#8fae9b", "#2f5d57"])

RESULTS = os.path.join(S.base_dir, "analysis", "results", "fatebias_trajectory_distance")
os.makedirs(RESULTS, exist_ok=True)


def smooth_all(grid, x, Y):
    """gaussian-kernel mean +/- SEM of every column of Y over the grid (x = fate bias),
    vectorised across genes. Returns (mean, sem, valid) each (n_grid, n_genes) / (n_grid,).
    Same estimator as fatebias_axis_trajectories.smooth, applied to all genes at once."""
    w = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / BW_BIAS) ** 2)   # (n_grid, m)
    den = w.sum(1)
    invden = 1.0 / np.maximum(den, EPS)
    mean = (w @ Y) * invden[:, None]
    ex2 = (w @ (Y ** 2)) * invden[:, None]
    std = np.sqrt(np.maximum(ex2 - mean ** 2, 0.0))
    n_eff = den ** 2 / np.maximum((w ** 2).sum(1), 1e-12)
    sem = std / np.sqrt(np.maximum(n_eff, 1.0))[:, None]
    valid = den >= 1.0
    mean[~valid] = np.nan
    sem[~valid] = np.nan
    return mean, sem, valid


def scan(td, e95, assignments, subtype, sub2type, family, lo, hi):
    """every INTERNAL E9.5 node with >=1 EC descendant leaf, attributed to the source cell
    type of its nearest in-window non-EC ancestor: progenitor identity is fixed at the
    WIN_LO..WIN_HI slice, then the lineage is followed DOWN. Unlike the sibling pilot we do
    NOT drop EC-typed internal nodes -- an EC-assigned progenitor that descends from a
    somatic-mesoderm node stays a somatic-mesoderm point at its own (high) fate bias, so each
    source's line keeps rising toward commitment instead of stopping where its non-EC nodes
    run out. Leaves (the terminal bias=1 EC cells) are deliberately excluded: folding that
    pooled endpoint into every source would just dilute the per-source signal."""
    import networkx as nx
    tgt_subtypes = {s for s, ct in sub2type.items() if ct in family}
    is_tgt = lambda n: subtype.get(n) in tgt_subtypes
    is_lab = lambda n: subtype.get(n) is not None
    cells, stypes, bias = [], [], []
    for tname in e95:
        g = td.obst[tname]
        amap = assignments.get(tname, {})
        tgt_c, tot_c = FB.descendant_frac(g, is_tgt, is_lab)
        source = {}
        for n in nx.topological_sort(g):             # root -> leaves, so parents resolve first
            preds = list(g.predecessors(n))
            src = source[preds[0]] if preds else None
            cell = FB.node_cell(g, amap, n)
            st = (FB.celltype_of(cell, subtype, sub2type)
                  if cell not in (None, FB.UNKNOWN) else None)
            t = float(g.nodes[n].get("time", np.nan))
            if st is not None and st not in family and lo <= t < hi:
                src = st                             # (re)fix source identity in the window
            source[n] = src
            if g.out_degree(n) == 0:                 # internal nodes only (no bias=1 leaves)
                continue
            if src is None or tgt_c[n] == 0:         # need a window source + >=1 EC descendant
                continue
            if cell is None or cell == FB.UNKNOWN:
                continue
            cells.append(cell)
            stypes.append(src)                       # source of the in-window ancestor
            bias.append(tgt_c[n] / tot_c[n])
    return cells, np.array(stypes), np.array(bias, float)


def read_tf_list(path):
    """literature TF symbols, one per line (possibly quoted)."""
    with open(path) as fh:
        return {ln.strip().strip('"') for ln in fh if ln.strip()}


def pairwise_distances(F_mean, F_valid, std_g):
    """(n_genes, n_src, n_src) symmetric per-gene trajectory L2, normalised by gene SD.
    d[g, i, j] = RMS over the grid points where BOTH source i and j are defined of
    (mean_i - mean_j), divided by std_g. NaN where the overlap is too small."""
    n_src = len(F_mean)
    G = F_mean[0].shape[1]
    D = np.full((G, n_src, n_src), np.nan, np.float32)
    scale = std_g + EPS
    for i in range(n_src):
        for j in range(i + 1, n_src):
            v = F_valid[i] & F_valid[j]
            if v.sum() < MIN_OVERLAP:
                continue
            diff = F_mean[i][v] - F_mean[j][v]       # (n_valid, G)
            d = np.sqrt((diff ** 2).mean(0)) / scale
            D[:, i, j] = D[:, j, i] = d
    return D


def main():
    family = S.FANIN[KEY]["exclude"]
    print(f"[1/6] {KEY}: internal nodes per source E[{WIN_LO}, {WIN_HI}), "
          f"EC descendants kept ...", flush=True)
    td, e95, assignments, subtype, sub2type = FB.load_trees()
    cells, stypes, bias = scan(td, e95, assignments, subtype, sub2type,
                               family, WIN_LO, WIN_HI)
    print(f"      {len(cells):,} internal nodes (bias {bias.min():.2f}..{bias.max():.2f})",
          flush=True)

    print("[2/6] reading expression ...", flush=True)
    Xp, ok, genes = FB.read_expression(cells)
    stypes, bias = stypes[ok], bias[ok]

    # pin the canonical EC fan-in sources (S.SOURCE_TYPES, shared with the figure's row 3)
    # rather than the top-N by node count: attributing EC descendants to each source shifts
    # the abundance ranking, so a raw top-N drops allantois in favour of sclerotome.
    vc = pd.Series(stypes).value_counts()
    sources = [s for s in S.SOURCE_TYPES if s in vc.index][:N_SOURCES]
    colors = {s: S.ROW3_SOURCE_PALETTE[i % len(S.ROW3_SOURCE_PALETTE)]
              for i, s in enumerate(sources)}
    print(f"      sources: " + ", ".join(f"{s} ({vc[s]})" for s in sources), flush=True)

    # drop near-silent genes so distances/slopes are not dominated by noise
    expressed = Xp.mean(0) >= S.MIN_EXPR
    Xe, gnames = Xp[:, expressed], genes[expressed]
    std_g = Xe.std(0)                                         # per-gene scale for normalising L2
    print(f"      {Xe.shape[0]:,} nodes x {Xe.shape[1]:,} expressed genes", flush=True)

    print("[3/6] smoothing per-source trajectories + pooled trajectory ...", flush=True)
    grid = np.linspace(0.0, 1.0, N_GRID)
    F_mean, F_sem, F_valid = [], [], []
    for s in sources:
        m = stypes == s
        mu, se, val = smooth_all(grid, bias[m], Xe[m].astype(np.float32))
        F_mean.append(mu); F_sem.append(se); F_valid.append(val)
    # pooled trajectory (all nodes, ignoring source) -> stable per-gene slope/level
    pmu, _, pval = smooth_all(grid, bias, Xe.astype(np.float32))
    gi = grid[pval]
    Yv = pmu[pval]                                            # (n_valid, G)
    gc = gi - gi.mean()
    slope = (gc @ Yv) / (gc @ gc + EPS)                      # d(expr)/d(bias) of pooled line
    mean_level = Yv.mean(0)

    print("[4/6] per-gene trajectory L2 between every source pair ...", flush=True)
    D = pairwise_distances(F_mean, F_valid, std_g)           # (G, n_src, n_src)
    iu = np.triu_indices(len(sources), 1)
    pair_d = D[:, iu[0], iu[1]]                              # (G, n_pairs)
    n_pairs = pair_d.shape[1]
    all_pairs = np.isfinite(pair_d).sum(1) == n_pairs        # gene scored for every pair

    # commitment-indicator genes: the smoothed trajectory sweeps at least COMMIT_RANGE over
    # the whole family (max - min across every source AND fate-bias grid point). Flat genes do
    # not track commitment and are excluded from the heatmap so it compares the sources only
    # on the genes that actually move along the axis.
    import warnings
    Fall = np.stack(F_mean, axis=0)                          # (n_src, n_grid, G)
    with warnings.catch_warnings():                          # some grid points are NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grange = np.nanmax(Fall, axis=(0, 1)) - np.nanmin(Fall, axis=(0, 1))   # (G,)
    commit = all_pairs & (grange >= COMMIT_RANGE)
    print(f"      {int(commit.sum()):,} commitment-indicator genes "
          f"(trajectory range >= {COMMIT_RANGE}) of {Xe.shape[1]:,}", flush=True)

    # heatmap metric: among commitment-indicator genes, the fraction whose trajectories are
    # "similar" between each source pair (normalized L2 < SIM_CUTOFF gene-SD units). self = 1
    # (NaN for display). Also keep the mean-L2 matrix (same gene set) as an npz reference.
    sim = np.isfinite(D) & (D < SIM_CUTOFF)
    Msim = sim[commit].mean(axis=0)                          # (n_src, n_src), in [0, 1]
    np.fill_diagonal(Msim, np.nan)
    with warnings.catch_warnings():                          # diagonal is all-NaN by design
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Mdist = np.nanmean(D[commit], axis=0)               # mean L2 over commitment genes
    np.fill_diagonal(Mdist, 0.0)

    # analysis 2: core = low divergence (max pairwise L2) + expressed + moving (|slope|)
    agree = np.nanmax(pair_d, axis=1)                        # low = every source agrees
    absslope = np.abs(slope)
    cand = all_pairs & (mean_level >= S.MIN_EXPR)
    a_thr = np.nanpercentile(agree[cand], AGREE_PCTL)
    s_thr = np.nanpercentile(absslope[cand], SLOPE_PCTL)
    core_ok = cand & (agree <= a_thr) & (absslope >= s_thr)
    core = np.where(core_ok)[0][np.argsort(-absslope[core_ok])][:N_CORE]
    print(f"      {core_ok.sum()} core candidates (agree<={a_thr:.3f} SD, "
          f"|slope|>={s_thr:.3f}); top {len(core)}: "
          + ", ".join(gnames[g] for g in core), flush=True)

    # analysis 3: path-specific genes, ranked by the SAME L2 stick-out score (one source far
    # from an agreeing rest), but split by the DIRECTION of the responsible source's trajectory
    # so the up-in-one-fate genes are not buried under the identity-decay ones. Per gene, per
    # source: (mean L2 from that source to the rest) - (mean pairwise L2 among the rest); the
    # score is the max over sources and that source is the responsible one. Direction = whether
    # the responsible source's trajectory ends higher (gain) or lower (loss) than it started.
    n_src = len(sources)
    G = Xe.shape[1]
    out = np.full((G, n_src), np.nan, np.float32)            # source -> mean L2 to the rest
    rest = np.full((G, n_src), np.nan, np.float32)           # mean pairwise L2 among the rest
    for si in range(n_src):
        others = [t for t in range(n_src) if t != si]
        with np.errstate(invalid="ignore"):
            out[:, si] = np.nanmean(D[:, si, others], axis=1)
            oi, oj = np.triu_indices(len(others), 1)
            oa, ob = np.array(others)[oi], np.array(others)[oj]
            rest[:, si] = np.nanmean(D[:, oa, ob], axis=1)
    stick_mat = out - rest                                   # high => si diverges, rest agree

    # per-source trajectory direction: end minus start over its valid grid (>0 = rises to
    # commitment). Used ONLY to bin genes as gain/loss, not to rank them.
    delta = np.full((G, n_src), np.nan, np.float32)
    for si in range(n_src):
        v = np.where(F_valid[si])[0]
        if len(v):
            delta[:, si] = F_mean[si][v[-1]] - F_mean[si][v[0]]

    with np.errstate(invalid="ignore"):
        stick = np.nanmax(stick_mat, axis=1)
    resp = np.full(G, -1, int)
    fin = np.isfinite(stick_mat).any(1)
    resp[fin] = np.nanargmax(stick_mat[fin], axis=1)
    dir_resp = np.array([delta[g, resp[g]] if resp[g] >= 0 else np.nan for g in range(G)])

    base = all_pairs & (mean_level >= S.MIN_EXPR) & np.isfinite(stick)
    gains = np.where(base & (dir_resp > 0))[0]               # responsible source rises
    gains = gains[np.argsort(-stick[gains])][:N_PATH]        # ...ranked by L2 divergence
    losses = np.where(base & (dir_resp < 0))[0]              # responsible source falls
    losses = losses[np.argsort(-stick[losses])][:N_PATH]
    print(f"      path-specific GAINS (up in one source): "
          + ", ".join(f"{gnames[g]} ({sources[resp[g]]})" for g in gains), flush=True)
    print(f"      path-specific LOSSES (down in one source): "
          + ", ".join(f"{gnames[g]} ({sources[resp[g]]})" for g in losses), flush=True)

    # per-source view: each source's top L2-divergent genes that RISE in it (its private
    # up-program), ranked by that source's own stick-out score stick_mat[:, si].
    by_source = {}
    idx = np.where(base)[0]
    for si in range(n_src):
        c = idx[(delta[idx, si] > 0)]
        by_source[si] = list(c[np.argsort(-stick_mat[c, si])][:PER_SOURCE])
    print("      per-source top rising & divergent genes:", flush=True)
    for si, s in enumerate(sources):
        print(f"        {s}: " + ", ".join(gnames[g] for g in by_source[si]), flush=True)

    # ---- save per-gene scores + the source-distance matrices ----
    tfs = read_tf_list(S.TF_LIST)
    pd.DataFrame({
        "gene": gnames,
        "mean_expr": mean_level,
        "slope": slope,                     # pooled (all-source) slope
        "max_pairwise_L2": agree,           # low = all sources trace the same line
        "stick_score": stick,               # high = one source diverges from an agreeing rest
        "stick_source": [sources[r] if r >= 0 else "" for r in resp],
        "stick_direction": ["up" if d > 0 else "down" if d < 0 else ""
                            for d in dir_resp],   # resp source rises (up) or falls (down)
        "is_tf": [g in tfs for g in gnames],
    }).sort_values("stick_score", ascending=False).to_csv(
        os.path.join(RESULTS, "trajectory_distance_scores.csv"), index=False)
    np.savez_compressed(os.path.join(RESULTS, "source_distance_matrix.npz"),
                        M_similar=Msim, M_meanL2=Mdist, cutoff=SIM_CUTOFF,
                        commit_range=COMMIT_RANGE, n_commit=int(commit.sum()),
                        sources=np.array(sources))
    off = Msim[~np.eye(len(sources), dtype=bool)]
    print(f"      similar-trajectory fraction (L2<{SIM_CUTOFF} SD, commitment genes): "
          f"{np.nanmin(off):.0%}..{np.nanmax(off):.0%} across source pairs", flush=True)

    print("[5/6] plotting ...", flush=True)
    S.apply_rc()
    plot_heatmap(Msim, sources, SIM_CUTOFF)
    plot_panels(grid, F_mean, F_sem, sources, colors,
                core, gnames, hl=None,
                path="core_genes_trajectories",
                ylabel="expression",
                caption="core EC genes: every source traces the same moving trajectory "
                        "into its own EC descendants")
    plot_panels(grid, F_mean, F_sem, sources, colors,
                gains, gnames, hl=[resp[g] for g in gains],
                path="path_specific_gains",
                ylabel="expression",
                caption="path-specific gains: one source (highlighted) diverges by RISING "
                        "with commitment; ranked by trajectory L2")
    plot_panels(grid, F_mean, F_sem, sources, colors,
                losses, gnames, hl=[resp[g] for g in losses],
                path="path_specific_losses",
                ylabel="expression",
                caption="path-specific losses: one source (highlighted) diverges by FALLING "
                        "with commitment; ranked by trajectory L2")
    plot_by_source(grid, F_mean, F_sem, sources, colors, by_source, gnames)

    print("[6/6] done. wrote:", flush=True)
    for f in ("source_distance_heatmap", "core_genes_trajectories",
              "path_specific_gains", "path_specific_losses", "path_specific_by_source"):
        print(f"      {f}.png/.svg", flush=True)
    print("      trajectory_distance_scores.csv, source_distance_matrix.npz", flush=True)


def plot_heatmap(M, sources, cutoff):
    """source x source fraction of genes with a similar trajectory (normalized L2 < cutoff
    gene-SD units). Higher = more of the transcriptome traces a shared path. Diagonal (self)
    is left blank. Colour range spans the off-diagonal values for contrast."""
    n = len(sources)
    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    off = M[np.isfinite(M)]
    vmin, vmax = float(off.min()), float(off.max())
    im = ax.imshow(M, cmap=DIST_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    lab = [s.replace(" ", "\n") for s in sources]
    ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=S.FS_TICK)
    ax.set_yticklabels(lab, fontsize=S.FS_TICK)
    thr = vmin + 0.6 * (vmax - vmin)
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
    cb.set_label(rf"commitment genes w/ similar trajectory ($L_2 < {cutoff:g}$ SD)",
                 fontsize=S.FS_LABEL)
    cb.outline.set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"source_distance_heatmap.{ext}"), dpi=300)
    plt.close(fig)


def plot_panels(grid, F_mean, F_sem, sources, colors, gene_idx, gnames,
                hl, path, ylabel, caption):
    """grid of per-gene panels: one smoothed line per source (+/- SEM). Each line reaches
    high fate bias through that source's OWN EC-descendant internal nodes (EC-typed nodes are
    not dropped). hl = per-panel source index to emphasise (None = emphasise all)."""
    ncol = 3
    nrow = int(np.ceil(len(gene_idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.PAGE_W_IN, 1.9 * nrow + 1.0),
                             squeeze=False)

    def panel(ax, g, hlt):
        for si, s in enumerate(sources):
            mu, se = F_mean[si][:, g], F_sem[si][:, g]
            emph = hlt is None or si == hlt
            col = colors[s] if emph else S.GREY
            if emph:
                ax.fill_between(grid, mu - se, mu + se, color=col, alpha=0.15, lw=0,
                                zorder=3)
            ax.plot(grid, mu, color=col, lw=1.7 if emph else 0.7,
                    alpha=1.0 if emph else 0.7, zorder=5 if emph else 2)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.margins(y=0.12)
        S.despine(ax)

    for k, g in enumerate(gene_idx):
        ax = axes[k // ncol][k % ncol]
        hlt = None if hl is None else hl[k]
        panel(ax, g, hlt)
        col = S.INK if hlt is None else colors[sources[hlt]]
        title = gnames[g] if hlt is None else f"{gnames[g]}: {sources[hlt]}"
        ax.set_title(title, fontsize=S.FS_LABEL, color=col, pad=4)
        if k % ncol == 0:
            ax.set_ylabel(ylabel, fontsize=S.FS_LABEL)
    for k in range(len(gene_idx), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    handles = [Line2D([0], [0], color=colors[s], lw=1.8, label=s) for s in sources]
    fig.legend(handles=handles, loc="lower center", ncol=len(sources),
               frameon=False, fontsize=S.FS_LEGEND, handlelength=1.6,
               columnspacing=1.4, handletextpad=0.4, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=S.FS_ANNOT,
             color=S.MUTED)
    fig.text(0.5, 0.10, "EC fate bias", ha="center", va="bottom",
             fontsize=S.FS_LABEL, color=S.INK)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"{path}.{ext}"), dpi=300)
    plt.close(fig)


def plot_by_source(grid, F_mean, F_sem, sources, colors, by_source, gnames):
    """one ROW per source, each showing that source's top PER_SOURCE genes that rise more
    uniquely in it than in any other source -- a per-tissue 'private up-program' readout.
    The owning source is highlighted in its colour on every panel of its row."""
    nrow = len(sources)
    ncol = max(len(v) for v in by_source.values())
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.PAGE_W_IN, 1.7 * nrow + 1.0),
                             squeeze=False)
    for si, s in enumerate(sources):
        for ci in range(ncol):
            ax = axes[si][ci]
            genes = by_source[si]
            if ci >= len(genes):
                ax.axis("off")
                continue
            g = genes[ci]
            for ti, t in enumerate(sources):
                mu, se = F_mean[ti][:, g], F_sem[ti][:, g]
                emph = ti == si
                col = colors[t] if emph else S.GREY
                if emph:
                    ax.fill_between(grid, mu - se, mu + se, color=col, alpha=0.15, lw=0,
                                    zorder=3)
                ax.plot(grid, mu, color=col, lw=1.7 if emph else 0.7,
                        alpha=1.0 if emph else 0.7, zorder=5 if emph else 2)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1.0])
            ax.margins(y=0.12)
            S.despine(ax)
            ax.set_title(gnames[g], fontsize=S.FS_LABEL, color=colors[s], pad=3)
        pos = axes[si][0].get_position()
        fig.text(0.015, (pos.y0 + pos.y1) / 2, s.replace(" ", "\n"), rotation=90,
                 va="center", ha="center", fontsize=S.FS_TICK, color=colors[s])
    fig.text(0.5, 0.02, "EC fate bias", ha="center", va="bottom",
             fontsize=S.FS_LABEL, color=S.INK)
    fig.text(0.5, 0.005, "per source: genes that rise with EC commitment in that source "
             "more than any other", ha="center", va="bottom", fontsize=S.FS_ANNOT,
             color=S.MUTED)
    fig.tight_layout(rect=(0.05, 0.05, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(RESULTS, f"path_specific_by_source.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
