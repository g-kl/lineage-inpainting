"""c and d: Tbx5 expression alone against the model, as predictors of the measured nn
distortion.

the cells are the Tbx5 figure's panel l cells: the heart tube reference cells of the
pipeline's heart scope. the expression is each cell's own raw log1p CP10K Tbx5 in the
devmap, not smoothed, since the model's prediction is not smoothed either. both are
correlated with the measured distortion averaged over the crispant embryos, and with
each embryo's own. writes numerical_results/expression_{cells,r}.csv.

    python -m analysis.figures.supp_tbx5.compute_expression
"""
import os

import h5py
import numpy as np
import pandas as pd

from analysis.figures.supp_tbx5 import style as S
from analysis.pipelines.tbx5.support_comparison import common as C

TBX5 = "tbx5"


def devmap_counts(gene):
    """log1p CP10K of one devmap gene over every devmap cell, in devmap row order."""
    with h5py.File(C.DEVMAP) as f:
        var = np.array([x.decode() for x in f["var"]["_index"][:]])
        j = int(np.flatnonzero(var == gene)[0])
        total = np.maximum(f["obs"]["total_counts"][:], 1)
        ip = f["X"]["indptr"][:]
        raw = np.zeros(len(total), "float32")
        for s in range(0, len(total), 200_000):
            e = min(s + 200_000, len(total))
            a, b = int(ip[s]), int(ip[e])
            m = f["X"]["indices"][a:b] == j
            rows = np.repeat(np.arange(s, e), np.diff(ip[s:e + 1]))[m]
            raw[rows] = f["X"]["data"][a:b][m]
    return np.log1p(raw / total * 1e4)


def main():
    d = pd.read_csv(C.out(f"per_embryo_{C.slug(S.HEART_SCOPE)}.csv"), index_col=0)
    d = d[d.cell_type.isin(S.HEART_TUBE)].copy()
    tbx5 = pd.Series(devmap_counts(S.TARGET), index=C.devmap_index())
    d[TBX5] = tbx5.reindex(d.index).values
    assert d[TBX5].notna().all()
    d[["subtype", S.MEAS, TBX5]].to_csv(os.path.join(S.NR, "expression_cells.csv"))

    embryos = sorted(c[len("measured_"):] for c in d.columns if c.startswith("measured_"))
    r = pd.DataFrame([{"predictor": p, "embryo": e,
                       "pearson": d[p].corr(d[S.MEAS] if e == "all" else d[f"measured_{e}"])}
                      for e in ["all"] + embryos for p in (S.PRED, TBX5)])
    r.to_csv(os.path.join(S.NR, "expression_r.csv"), index=False)
    print(f"{len(d):,} heart tube cells, {len(embryos)} crispant embryos", flush=True)


if __name__ == "__main__":
    main()
