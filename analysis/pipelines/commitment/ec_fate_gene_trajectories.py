"""Per-source EC-fate gene trajectories over time.

For each EC source type we take its admissible subtree (source-typed nodes and
their descendants), split it into EC-fated and non-EC nodes, and plot the per-bin
mean expression of each group as a bold line. Columns are the strongest pooled
per-source DEGs: the top N up in EC (EC-fate indicators) and the top N down in EC
(non-EC / anti-EC indicators).

Run from the repo root (needs the built index).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from analysis.pipelines.commitment import commitment as C
from analysis.pipelines.commitment.ec_source_specification import (
    propagate_source, EC_KEYS, EC_HI, EC_LO, K_SOURCES, MAX_PER_GROUP,
    POOL_CAP, MIN_NODES, PADJ)

OUT = "analysis/pipelines/commitment/results/ec_source"
N_GENE = 3          # genes per direction (up in EC, down in EC)
SEED = 0
EC_COL, NEG_COL = "#c0392b", "#1f2a44"


def pooled_degs(idx, cap, mask, fated, neg, cells, n):
    """Top n up-in-EC and top n down-in-EC genes from one pooled DEG."""
    ca = cap(list(cells[mask & fated]))[:POOL_CAP]
    cb = cap(list(cells[mask & neg]))[:POOL_CAP]
    if len(ca) < MIN_NODES or len(cb) < MIN_NODES:
        return [], []
    df = C.deg_cells(idx, ca, cb, "EC", "nonEC")
    sig = df[df.pvals_adj < PADJ]
    up = list(sig[sig.scores > 0].sort_values("scores", ascending=False).names[:n])
    down = list(sig[sig.scores < 0].sort_values("scores").names[:n])
    return up, down


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(SEED)
    idx = C.Index()

    ti = pd.read_csv(C.TYPE_INFO, index_col=0)
    sub = pd.read_csv(C.SUBTYPES, index_col=0)["cell_subtype"]
    own = idx.nodes["cell"].map(sub).map(dict(zip(ti.index, ti["cell_type"]))).values

    cols = idx.cols["cell_type"]
    ec_cols = [c for c in cols if any(k in c.lower() for k in EC_KEYS)]
    ec_idx = [cols.index(c) for c in ec_cols]
    ec_frac = idx.hist["cell_type"][:, ec_idx].sum(1) / idx.nodes["n_leaf"].values
    is_ec_type = pd.Series(own).isin(ec_cols).values
    fated = ec_frac > EC_HI
    neg = ec_frac < EC_LO

    sources = pd.Series(own[(~is_ec_type) & fated]).value_counts(
        ).head(K_SOURCES).index.tolist()
    src = propagate_source(idx.nodes, own, set(sources))

    bins = idx.nodes["bin"].values
    cells = idx.nodes["cell"].values
    node_row = np.array([idx.row_of.get(c, -1) for c in cells])
    gene_col = {g: i for i, g in enumerate(idx.genes)}
    centers = C.BIN_EDGES[:-1] + 0.125

    def cap(names):
        names = [c for c in names if c in idx.row_of]
        if len(names) > MAX_PER_GROUP:
            names = list(rng.choice(names, MAX_PER_GROUP, replace=False))
        return names

    def expr_of(gene):
        col = np.asarray(idx.expr[:, gene_col[gene]].todense()).ravel()
        y = np.full(len(cells), np.nan)
        ok = node_row >= 0
        y[ok] = col[node_row[ok]]
        return y

    ncol = 2 * N_GENE
    fig, axes = plt.subplots(K_SOURCES, ncol, figsize=(16, 13),
                             sharex=True, squeeze=False)
    for r, s in enumerate(sources):
        mask = src == s
        up, down = pooled_degs(idx, cap, mask, fated, neg, cells, N_GENE)
        genes = [(g, EC_COL) for g in up] + [(g, NEG_COL) for g in down]
        for cc in range(ncol):
            ax = axes[r][cc]
            if cc >= len(genes):
                ax.axis("off")
                continue
            gene, tcol = genes[cc]
            y = expr_of(gene)
            for grp, col, lab in [(fated, EC_COL, "EC-fated"),
                                  (neg, NEG_COL, "non-EC")]:
                m = mask & grp
                mean = [np.nanmean(y[m & (bins == b)]) if (m & (bins == b)).any()
                        else np.nan for b in range(C.NBIN)]
                ax.plot(centers, mean, color=col, lw=2.6, label=lab,
                        path_effects=[pe.Stroke(linewidth=4, foreground="white"),
                                      pe.Normal()])
            ax.set_title(gene, fontsize=10, color=tcol)
            ax.set_xlim(7.5, 9.5)
            if cc == 0:
                ax.set_ylabel(s.replace(" mesoderm", " meso."), fontsize=9)
            if r == K_SOURCES - 1:
                ax.set_xlabel("time")

    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.text(0.30, 0.995, "top up in EC (fate indicators)", ha="center",
             va="top", fontsize=11, color=EC_COL, fontweight="bold")
    fig.text(0.74, 0.995, "top down in EC (non-EC indicators)", ha="center",
             va="top", fontsize=11, color=NEG_COL, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(OUT, "ec_fate_gene_trajectories.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
