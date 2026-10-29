#!/usr/bin/env python
"""panel g: the memory/erased split over the G_FATES fates.

panel f's two source groups are the two sides of a fitted positional axis. the neuron
one is fit on hindbrain against spinal cord, and midbrain and hypothalamic sources all
land on its anterior side. here a fate's E8.5 sources are pooled by position from the
annotation instead -- paraxial or lateral plate lineage for the endothelial fate,
neural-tube region for the neurons -- and its two biggest groups are compared.
everything downstream -- the two |log2 fc|, the expression floor, the two cuts -- is
panel f's.

    python analysis/figures/convergence/compute_sweep.py
"""
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
import data as D
import trees as TR
from compute_divergence import coords, means

EC_FAMILY = S.SHEETS["ec"]["at_dest"]


def ranked_fates():
    """the G_FATES fates, carrying row b's convergence score so the csv keeps it, most
    convergent first. named rather than walked down the ranking: at E8.5 most of the
    most convergent fates are already their own biggest ancestor, so a walk fills up
    with placodal sensory neurons and leaves out the two the rest of the row is
    about."""
    c = pd.read_csv(os.path.join(S.NR, "celltype_conv_vs_ot.csv"))
    c = c[c.cell_type.isin(S.G_FATES)]
    return c.sort_values("conv", ascending=False)[["cell_type", "conv", "n"]]


def excluded(fate):
    """a source is not allowed to be the fate itself, or -- for the endothelial fates
    -- anything else already endothelial, as in the rest of the figure."""
    return EC_FAMILY | {fate} if fate in EC_FAMILY else {fate}


def anchored(tt, fate):
    """per E9.5 leaf of `fate`, the assigned ancestor nearest T_SRC and its type."""
    out = []
    for t in tt:
        leaves = np.flatnonzero(t.is_leaf & (t.ctype == fate))
        if not len(leaves):
            continue
        A = t.paths(leaves)
        node = np.maximum(A, 0)
        off = np.where((A >= 0) & t.assigned[node]
                       & (np.abs(t.time[node] - S.T_SRC) < S.SRC_TOL),
                       np.abs(t.time[node] - S.T_SRC), np.inf)
        j = off.argmin(1)
        ok = np.isfinite(off[np.arange(len(A)), j])
        src = A[np.arange(len(A)), j][ok]
        out.append(pd.DataFrame(dict(leaf_cell=t.cell[leaves[ok]],
                                     src_cell=t.cell[src], src_type=t.ctype[src])))
    cols = ["leaf_cell", "src_cell", "src_type"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def positional(fate):
    """the classifier a fate's sources are pooled by: lineage for the endothelial fate,
    neural-tube region for the neurons."""
    return S.ec_x if fate in EC_FAMILY else S.tube_x


def groups(d, fate, gl, type2lin, rng):
    """the fate's two biggest positional groups, and the four cell sets they define."""
    d = d[~d.src_type.isin(excluded(fate))
          & (d.src_type.map(gl) == gl.get(fate, ""))]
    cls = positional(fate)
    d = d.assign(pos=[cls(t, type2lin.get(t, "")) for t in d.src_type])
    d = d[d.pos != ""]
    n = d.pos.value_counts()
    pair = n[n >= S.G_MIN_TRAJ].index[:2]
    if len(pair) < 2:
        return None, n
    cap = lambda r: (r if len(r) <= S.DIV_CELLS
                     else np.sort(rng.choice(r, S.DIV_CELLS, replace=False)))
    g, sizes = {}, {}
    for side, t in zip(("lo", "hi"), pair):
        m = d.pos == t
        sizes[side] = (t, int(m.sum()))
        for kind in ("src", "leaf"):
            g[f"{kind}_{side}"] = cap(np.unique(d[f"{kind}_row"][m].values))
    return (g, sizes), n


def main():
    t0 = time.time()
    tic = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
    rng = np.random.default_rng(S.SEED)
    gl = D.germ_layers()
    _, type2lin = D.labels()
    name, _ = D.obs()
    at_row = pd.Index(name)
    with h5py.File(S.H5AD, "r") as f:
        genes = np.array([g.decode() for g in f["var"]["_index"][:]])

    _, tt = TR.load()
    tic(f"{len(ranked_fates())} named fates")

    out, skipped = [], []
    for r in ranked_fates().itertuples():
        d = anchored(tt, r.cell_type)
        if not len(d):
            skipped.append(dict(fate=r.cell_type, conv=r.conv, n_leaf=0,
                                top="no anchored leaf"))
            continue
        for kind in ("src", "leaf"):
            d[f"{kind}_row"] = at_row.get_indexer(d[f"{kind}_cell"])
        d = d[(d.src_row >= 0) & (d.leaf_row >= 0)]
        got, n = groups(d, r.cell_type, gl, type2lin, rng)
        if got is None:
            skipped.append(dict(fate=r.cell_type, conv=r.conv, n_leaf=len(d),
                                top=", ".join(f"{t} ({v})"
                                              for t, v in n.head(2).items())))
            continue
        g, sizes = got
        rows = np.unique(np.concatenate(list(g.values())))
        X = D.read_rows(rows, len(genes))
        lin, _, keep = means(X, pd.Series(np.arange(len(rows)), index=rows), g)
        _, _, cls = coords(lin, keep)
        c = pd.Series(cls).value_counts()
        out.append(dict(
            fate=r.cell_type, conv=r.conv, n_leaf=len(d),
            src_lo=sizes["lo"][0], n_lo=sizes["lo"][1],
            src_hi=sizes["hi"][0], n_hi=sizes["hi"][1],
            n_tested=int(keep.sum()), n_memory=int(c.get("memory", 0)),
            n_erased=int(c.get("erased", 0)),
            pct_memory=100 * c.get("memory", 0) / keep.sum(),
            pct_erased=100 * c.get("erased", 0) / keep.sum()))
        tic(f"    {r.cell_type}: {len(d):,} anchored leaves, "
            f"{sizes['lo'][0]} ({sizes['lo'][1]:,}) vs {sizes['hi'][0]} "
            f"({sizes['hi'][1]:,}), {out[-1]['pct_memory']:.1f}% memory, "
            f"{out[-1]['pct_erased']:.1f}% erased")

    H = pd.DataFrame(out)
    H.to_csv(os.path.join(S.NR, "sweep.csv"), index=False)
    K = pd.DataFrame(skipped)
    K.to_csv(os.path.join(S.NR, "sweep_skipped.csv"), index=False)
    for r in K.itertuples():
        print(f"      passed over {r.fate:<28} conv {r.conv:.1f}  "
              f"{r.n_leaf:>5,} anchored  best two: {r.top}", flush=True)
    tic(f"wrote sweep.csv ({len(H)} fates) and sweep_skipped.csv ({len(K)}) to {S.NR}")


if __name__ == "__main__":
    main()
