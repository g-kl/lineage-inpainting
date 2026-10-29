"""row-3 left cache: DEGs of E9.5 EC endpoints by source tissue (origin memory).

traces every E9.5 EC leaf back to its source cell type at t_anc (reusing the
compute_trajectories path tracer), takes each path's endpoint (the E9.5 EC cell),
groups the endpoints by source, and runs a 1-vs-rest Wilcoxon marker test (scanpy).
the top style.DEG_N_PER_SOURCE genes per source are block-ordered and the per-source
mean expression is z-scored per gene across sources.

writes numerical_results/fatebias_ec_deg.csv (block-ordered genes x sources z-scores).

    python analysis/figures/convergence/compute_deg.py
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))
import style as S
import fatebias as FB
import compute_trajectories as CT

KEY = "ec"


def main():
    family = S.FANIN[KEY]["exclude"]
    t0 = S.FANIN[KEY]["t_anc"]
    print(f"[1/4] tracing EC leaves to their source @ E{t0} ...", flush=True)
    td, e95, assignments, subtype, sub2type = FB.load_trees()
    paths, _ = CT.trace_fate(td, e95, assignments, family, subtype, sub2type, t0)

    sources = list(pd.Series([s for s, _ in paths]).value_counts()
                   .index[:S.TOP_N_SOURCES])
    cell2src = {}
    for s, pts in paths:
        if s in sources:
            cell2src.setdefault(pts[-1][1], s)              # endpoint cell -> source
    print(f"      {len(cell2src):,} EC endpoints; sources: {sources}", flush=True)

    print("[2/4] reading endpoint expression ...", flush=True)
    ep_cells = list(cell2src)
    Xn, ok, genes = FB.read_expression(ep_cells)
    ep_cells = [c for c, o in zip(ep_cells, ok) if o]
    ep_src = np.array([cell2src[c] for c in ep_cells])

    print("[3/4] 1-vs-rest Wilcoxon DEGs per source ...", flush=True)
    import anndata as ad
    import scanpy as sc
    adata = ad.AnnData(X=sp.csr_matrix(Xn),
                       obs=pd.DataFrame({"source": pd.Categorical(
                           ep_src, categories=sources)}, index=ep_cells),
                       var=pd.DataFrame(index=genes))
    sc.tl.rank_genes_groups(adata, "source", method="wilcoxon")
    deg = sc.get.rank_genes_groups_df(adata, group=None)

    gene_order, seen = [], set()
    for s in sources:
        top = (deg[deg.group == s].sort_values("scores", ascending=False)
               .names[:S.DEG_N_PER_SOURCE])
        for g in top:
            if g not in seen:
                seen.add(g)
                gene_order.append(g)

    print("[4/4] z-scoring per-source mean expression ...", flush=True)
    gi = {g: i for i, g in enumerate(genes)}
    idx = [gi[g] for g in gene_order]
    M = np.vstack([Xn[ep_src == s][:, idx].mean(0) for s in sources]).T  # genes x src
    Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    pd.DataFrame(Z, index=gene_order, columns=sources).to_csv(
        os.path.join(S.NR, f"fatebias_{KEY}_deg.csv"))
    print(f"wrote fatebias_{KEY}_deg.csv ({len(gene_order)} genes x {len(sources)} "
          f"sources)", flush=True)


if __name__ == "__main__":
    main()
