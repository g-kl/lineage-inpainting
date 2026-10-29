"""panel c: the genes each program is built out of, averaged over the source cell types
and over the destinations they reach.

the programs are fig 4's own axes (`axes.npz`). a gene is ranked by its correlation
with an axis score over the cells that axis was fit on, not the whole lineage: over
all of neural ectoderm the anterior end picks up Otx2, a forebrain and midbrain gene
that no hindbrain or spinal cord cell expresses.

    python -m analysis.figures.supp_conv.compute_genes
"""
import os
import time

import h5py
import numpy as np
import pandas as pd

from analysis.figures.convergence import compute_axes as CA
from analysis.figures.supp_conv import style as S

D = CA.D   # fig 4's loaders: label maps, csr row reader


def correlations(cells, Z, A, genes, rng):
    """per program axis end, the N_POOL genes most correlated with the axis score, on
    up to N_GENE_CELLS of its fit cells, both sides centred within embryo."""
    _, type2lin = D.labels()
    out = []
    for key, p in S.C.PROGRAMS.items():
        m = cells.lineage.isin(p["lineages"]).values
        c, Y = cells[m].reset_index(drop=True), CA.centred(Z[m], cells.embryo.values[m])
        ty = c.cell_type.values
        xc = np.array([p["xcls"](t, type2lin.get(t, "")) for t in ty], dtype=object)
        fit = {"x": np.isin(xc, p["x"]), "y": (ty == p["target"]) | p["yneg"](ty)}
        for j, axis in enumerate("xy"):
            k = np.flatnonzero(fit[axis])
            i = np.sort(rng.choice(k, min(S.N_GENE_CELLS, len(k)), replace=False))
            X = CA.centred(D.read_rows(c.row.values[i], len(genes)).astype(np.float64),
                           c.embryo.values[i])
            s = CA.centred((Y[i] @ A[f"{key}_V"][:, j])[:, None], c.embryo.values[i])[:, 0]
            r = X.T @ s / (np.sqrt(np.maximum((X ** 2).sum(0), 1e-9)) * np.linalg.norm(s))
            o = np.argsort(r)
            for end, idx in (("pos", o[::-1][:S.N_POOL]), ("neg", o[:S.N_POOL])):
                out += [dict(panel=key, axis=axis, end=end, rank=n + 1, gene=genes[g],
                             corr=r[g]) for n, g in enumerate(idx)]
    return pd.DataFrame(out)


def columns(T, sheet):
    """the N_TYPES most abundant source cell types, ordered along the positional axis,
    then one column per destination, in the order of the identity blocks."""
    T = T[~T.src_type.isin(S.DROP_SRC[sheet])]
    n = T.groupby("src_type").agg(n=("traj", "size"), x=("x", "mean"))
    cols = [dict(kind="source", label=t, n=int(r.n), x=r.x,
                 rows=T.src_row[T.src_type == t].values)
            for t, r in n.nlargest(S.N_TYPES, "n").sort_values("x").iterrows()]
    for target in S.DEST_SHORT:
        g = T[T.target == target]
        if len(g) >= S.MIN_DEST:
            cols.append(dict(kind="dest", label=S.DEST_SHORT[target], n=len(g),
                             x=g.x.mean(), rows=g.leaf_row.values))
    return cols


def strip(Cr, T, sheet, genes, rng):
    """one sheet's gene x column means. the candidates are each block's N_POOL genes
    (a gene in two blocks stays in the first); a block keeps its first N_GENES that
    reach MIN_EXPR in some column."""
    P = pd.concat([Cr[(Cr.panel == panel) & (Cr.axis == axis) & (Cr.end == end)]
                   .assign(block=label)
                   for label, panel, axis, end in S.BLOCKS[sheet]]).drop_duplicates("gene")
    cols = columns(T, sheet)
    pick = {c["label"]: (c["rows"] if len(c["rows"]) <= S.N_CELLS else
                         rng.choice(c["rows"], S.N_CELLS, replace=False)) for c in cols}
    rows = np.unique(np.concatenate(list(pick.values())))
    gi = pd.Series(np.arange(len(genes)), index=genes)[P.gene].values
    X = D.read_rows(rows, len(genes))[:, gi]
    at = pd.Series(np.arange(len(rows)), index=rows)
    M = np.stack([X[at[pick[c["label"]]].values].mean(0) for c in cols], 1)

    live = P.assign(peak=M.max(1))
    keep = (live[live.peak >= S.MIN_EXPR].sort_values("rank")
            .groupby("block", sort=False).head(S.N_GENES))
    keep = pd.concat([keep[keep.block == b[0]] for b in S.BLOCKS[sheet]])
    row = pd.Series(np.arange(len(P)), index=P.gene.values)
    return pd.DataFrame([dict(sheet=sheet, block=g.block, gene=g.gene, rank=g.rank,
                              corr=round(g.corr, 3), kind=c["kind"], column=c["label"],
                              n=c["n"], x=c["x"], mean=M[row[g.gene], j])
                         for g in keep.itertuples() for j, c in enumerate(cols)])


def main():
    t0 = time.time()
    tic = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
    rng = np.random.default_rng(S.C.SEED)
    with h5py.File(S.C.H5AD, "r") as f:
        genes = np.array([g.decode() for g in f["var"]["_index"][:]])

    tic("fit population")
    cells, Z = CA.population()
    tic("gene correlations")
    Cr = correlations(cells, Z, D.axes(), genes, rng)
    Cr.to_csv(os.path.join(S.NR, "gene_corr.csv"), index=False)

    T = pd.read_csv(os.path.join(S.C.NR, "sources.csv"))
    H = []
    for sheet in S.BLOCKS:
        H.append(strip(Cr, T[T.sheet == sheet], sheet, genes, rng))
        h = H[-1]
        tic(f"{sheet}: " + "; ".join(f"{b} {', '.join(g.gene.unique())}"
                                    for b, g in h.groupby("block", sort=False)))
        tic("    columns " + ", ".join(f"{c} (n={int(g.n.iloc[0]):,})"
                                       for c, g in h.groupby("column", sort=False)))
    pd.concat(H).to_csv(os.path.join(S.NR, "genes.csv"), index=False)
    tic(f"wrote gene_corr.csv and genes.csv to {S.NR}")


if __name__ == "__main__":
    main()
