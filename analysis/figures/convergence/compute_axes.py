#!/usr/bin/env python
"""the program axes: a positional contrast and an identity contrast per program, the
identity one orthogonalised against the positional one. single Fisher (LDA)
directions, fit on E9.5 devmap cells of the programs' lineages.

    python analysis/figures/convergence/compute_axes.py
"""
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import style as S


def population():
    """E9.5 devmap cells of the programs' lineages, capped per (cell type, embryo),
    with a random half held out for the auc."""
    sub2type, type2lin = D.labels()
    with h5py.File(S.H5AD, "r") as f:
        o = f["obs"]
        cat = lambda k: pd.Categorical.from_codes(
            o[k]["codes"][:], [c.decode() for c in o[k]["categories"][:]])
        df = pd.DataFrame({"time": o["time"][:], "dataset": cat("dataset"),
                           "embryo": cat("embryo").astype(str),
                           "subtype": cat("cell_subtype").astype(str)})
        df["row"] = np.arange(len(df))
        df["cell_type"] = df.subtype.str.lower().map(sub2type)
        df["lineage"] = df.cell_type.map(type2lin)
        df = df[(df.dataset == "devmap") & (df.time == S.FIT_TIME)
                & df.lineage.isin(S.LINEAGES)]
        rng = np.random.default_rng(S.SEED)
        keep = [g if len(g) <= S.N_PER_TYPE
                else g.iloc[rng.choice(len(g), S.N_PER_TYPE, replace=False)]
                for _, g in df.groupby(["cell_type", "embryo"], observed=True)]
        cells = pd.concat(keep).sort_values("row").reset_index(drop=True)
        cells["half"] = rng.integers(0, 2, len(cells))
        Z = f["obsm"]["X_scvi"][:][cells.row.values].astype(np.float64)
    return cells[["row", "embryo", "cell_type", "lineage", "half"]], Z


def centred(Z, embryo):
    """within-embryo centring, so no contrast can be an embryo offset."""
    Y = Z.copy()
    for e in np.unique(embryo):
        Y[embryo == e] -= Y[embryo == e].mean(0)
    return Y


def fisher(Y, g, m):
    """the one direction best separating g from not-g inside m: mean difference over
    the pooled within-group covariance, signed so g is positive."""
    a, b = Y[m & g], Y[m & ~g]
    assert min(len(a), len(b)) > len(Y.T), (len(a), len(b))
    d = a.mean(0) - b.mean(0)
    W = (np.cov(a.T) + np.cov(b.T)) / 2
    v = eigh(np.outer(d, d), W + S.RIDGE * np.trace(W) / len(W) * np.eye(len(W)))[1][:, -1]
    v /= np.linalg.norm(v)
    return v if (a @ v).mean() > (b @ v).mean() else -v


def auc(s, g):
    r = rankdata(s)
    return (r[g].sum() - g.sum() * (g.sum() + 1) / 2) / (g.sum() * (~g).sum())


def program_axes(key, p, cells, Z, type2lin):
    m = cells.lineage.isin(p["lineages"]).values
    cells, Y = cells[m].reset_index(drop=True), centred(Z[m], cells.embryo.values[m])
    ty, half = cells.cell_type.values, cells.half.values

    xc = np.array([p["xcls"](t, type2lin.get(t, "")) for t in ty], dtype=object)
    lo, hi = p["x"]
    gx, mx = xc == hi, np.isin(xc, [lo, hi])
    gy = ty == p["target"]
    my = gy | (p["yneg"](ty) & ~gy)

    vx = fisher(Y, gx, mx)
    vy = fisher(Y, gy, my)
    cos = float(abs(vx @ vy))
    vy = vy - (vy @ vx) * vx
    vy /= np.linalg.norm(vy)

    V = {"x": vx, "y": vy}
    held = {"x": fisher(Y, gx, mx & (half == 0)),
            "y": fisher(Y, gy, my & (half == 0))}
    rows = []
    for a, g, sel in (("x", gx, mx), ("y", gy, my)):
        out = sel & (half == 1)
        rows.append(dict(panel=key, axis=a, positive=hi if a == "x" else p["target"],
                         negative=lo if a == "x" else p["yneg_label"],
                         n_pos=int((g & sel).sum()), n_neg=int((~g & sel).sum()),
                         auc=auc(Y[sel] @ V[a], g[sel]),
                         auc_heldout=auc(Y[out] @ held[a], g[out]), cos_xy=cos))
    return rows, V, Z[m].mean(0)


def main():
    t0 = time.time()
    tic = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
    _, type2lin = D.labels()

    tic("population")
    cells, Z = population()
    tic(f"    {len(cells):,} E{S.FIT_TIME} devmap cells, "
        f"{cells.cell_type.nunique()} types, {cells.embryo.nunique()} embryos")

    rows, store = [], {}
    for key, p in S.PROGRAMS.items():
        tic(f"axes: {key}")
        r, V, mu = program_axes(key, p, cells, Z, type2lin)
        rows += r
        store[f"{key}_V"] = np.stack([V["x"], V["y"]], 1)
        store[f"{key}_mu"] = mu

    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(S.NR, "axes.csv"), index=False)
    np.savez(os.path.join(S.NR, "axes.npz"), **store)
    for r in A.itertuples():
        print(f"      {r.panel:<4} {r.axis}  auc {r.auc:.3f} ({r.auc_heldout:.3f} "
              f"held out)  |cos| {r.cos_xy:.2f}  n {r.n_pos:,}/{r.n_neg:,}", flush=True)
    tic(f"wrote axes.csv and axes.npz to {S.NR}")


if __name__ == "__main__":
    main()
