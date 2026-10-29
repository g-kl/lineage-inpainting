"""mean log1p CP10K of every screened gene in every cell type over the E9.0 reference
cells -> numerical_results/expression.parquet (cell type x gene) + expression_n.csv.

cell types come from the same cell -> type map the screen readout uses. X is streamed
in row chunks through h5py, keeping only the E9.0 rows.

    python -m analysis.figures.supp_mgi.compute_expression
"""
import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.figures.perturb.readout import name_to_celltype
from analysis.figures.supp_mgi import style as S

CHUNK = 100_000       # rows per read


def main():
    f = h5py.File(S.STATES_H5AD, "r")
    names = f["obs/_index"][:].astype(str)
    time = f["obs/time"][:]
    genes = f["var/_index"][:].astype(str)
    ct = pd.Series(names).map(name_to_celltype())
    at = time == S.RING_TARGET_TIME
    keep = at & ct.notna().to_numpy()
    print(f"{keep.sum():,} E{S.RING_TARGET_TIME:g} cells, "
          f"{ct[at].isna().sum():,} without a cell type", flush=True)

    cts = sorted(ct[keep].unique())
    code = np.full(len(names), -1)
    code[keep] = pd.Categorical(ct[keep], categories=cts).codes

    X = f["X"]
    indptr = X["indptr"][:]
    rows = np.flatnonzero(keep)
    sums = np.zeros((len(cts), len(genes)))
    for lo in range(rows[0], rows[-1] + 1, CHUNK):
        hi = min(lo + CHUNK, rows[-1] + 1)
        c = code[lo:hi]
        if not (c >= 0).any():
            continue
        a, b = indptr[lo], indptr[hi]
        m = sp.csr_matrix((X["data"][a:b], X["indices"][a:b], indptr[lo:hi + 1] - a),
                          shape=(hi - lo, len(genes)))[c >= 0]
        c = c[c >= 0]
        g = sp.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))),
                          shape=(len(cts), len(c)))
        sums += (g @ m).toarray()
        print(f"  rows {hi:,} / {rows[-1] + 1:,}", flush=True)

    n = np.bincount(code[keep], minlength=len(cts))
    pd.DataFrame(sums / n[:, None], index=cts, columns=genes).to_parquet(
        os.path.join(S.NR, "expression.parquet"))
    pd.Series(n, index=cts, name="n_cells").to_csv(os.path.join(S.NR, "expression_n.csv"))
    print(f"wrote {len(cts)} cell types x {len(genes)} genes", flush=True)


if __name__ == "__main__":
    main()
