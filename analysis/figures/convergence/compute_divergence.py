#!/usr/bin/env python
"""panel f: per gene, how source-specific it is at E8.5 against how much of that
specificity is left in the E9.5 leaves.

the contrast is the two sides of the positional axis -- source score below and above
zero. x is |log2 fc| between the two sides' E8.5 source cells, y is |log2 fc| between
the E9.5 leaves those sides reach. a gene specific at E8.5 and still split at E9.5 is
memory; specific at E8.5 and not at E9.5 is erased.

    python analysis/figures/convergence/compute_divergence.py
"""
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import style as S

SIDES = ("lo", "hi")
KINDS = ("src", "leaf")


def cell_groups(T, panel, rng):
    """the four cell sets of a panel, capped and deduplicated -- a source shared by
    several trajectories is one cell here, not one per descendant."""
    t = T[T.panel == panel]
    out = {}
    for side, m in zip(SIDES, (t.x < 0, t.x >= 0)):
        for kind in KINDS:
            r = np.unique(t[f"{kind}_row"][m].values.astype(int))
            out[f"{kind}_{side}"] = (r if len(r) <= S.DIV_CELLS else
                                     np.sort(rng.choice(r, S.DIV_CELLS, replace=False)))
    return out


def log2fc(hi, lo):
    return np.log2((hi + S.DIV_EPS) / (lo + S.DIV_EPS))


def means(X, at, groups):
    """per group, the mean over cells on the linear scale and on the log1p one. the
    linear means carry the fold change, the log1p ones the expression floor."""
    lin = {k: np.expm1(X[at[v].values]).mean(0) for k, v in groups.items()}
    log = {k: X[at[v].values].mean(0) for k, v in groups.items()}
    return lin, log, np.maximum.reduce(list(log.values())) >= S.DIV_MIN_EXPR


def coords(lin, keep):
    """the two |log2 fc| and the class per kept gene. panel g reuses this on its own
    pair of source groups, so the two rows are the same statistic throughout."""
    x = np.abs(log2fc(lin["src_hi"], lin["src_lo"]))[keep]
    y = np.abs(log2fc(lin["leaf_hi"], lin["leaf_lo"]))[keep]
    return x, y, np.where(x < S.DIV_XCUT, "other",
                          np.where(y >= S.DIV_YCUT, "memory", "erased"))


def main():
    t0 = time.time()
    tic = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
    rng = np.random.default_rng(S.SEED)
    T = pd.read_csv(os.path.join(S.NR, "sources.csv"))
    with h5py.File(S.H5AD, "r") as f:
        genes = np.array([g.decode() for g in f["var"]["_index"][:]])

    groups = {spec["panel"]: cell_groups(T, spec["panel"], rng) for spec in S.PHASE}
    for panel, g in groups.items():
        tic(f"{panel}: " + ", ".join(f"{k} {len(v):,}" for k, v in g.items()))

    rows = np.unique(np.concatenate([v for g in groups.values() for v in g.values()]))
    tic(f"reading {len(rows):,} cells x {len(genes):,} genes")
    X = D.read_rows(rows, len(genes))
    at = pd.Series(np.arange(len(rows)), index=rows)

    out = []
    for panel, g in groups.items():
        lin, _, keep = means(X, at, g)
        x, y, cls = coords(lin, keep)
        out.append(pd.DataFrame(dict(
            panel=panel, gene=genes[keep], x=x, y=y, cls=cls,
            **{f"m_{k}": v[keep] for k, v in lin.items()})))
        n = pd.Series(cls).value_counts()
        tic(f"    {panel}: {keep.sum():,} of {len(genes):,} genes expressed, "
            f"{n.get('memory', 0)} memory, {n.get('erased', 0)} erased")

    G = pd.concat(out, ignore_index=True)
    G.to_csv(os.path.join(S.NR, "gene_divergence.csv"), index=False)
    for panel, h in G.groupby("panel"):
        for c, key in (("memory", "y"), ("erased", "x")):
            top = h[h.cls == c].nlargest(S.F_LABEL, key)
            print(f"      {panel:<4} {c:<7} " + ", ".join(
                f"{r.gene} ({r.x:.1f}, {r.y:.1f})" for r in top.itertuples()),
                flush=True)
    tic(f"wrote gene_divergence.csv to {S.NR}")


if __name__ == "__main__":
    main()
