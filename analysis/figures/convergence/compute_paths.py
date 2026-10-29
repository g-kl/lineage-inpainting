#!/usr/bin/env python
"""the trajectory cache rows d-g read: every assigned node on the ancestor path of
every E9.5 leaf that reaches a program's target, with its h5ad row and latent, so a
state can be scored on either axis by one matrix multiply.

    python analysis/figures/convergence/compute_paths.py
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
import trees as TR


def collect(names, tt, key, p, type2lin, off):
    """one program's target trajectories, one row per (trajectory, node). `group` is
    the trajectory's source class -- the nearest classifiable strict ancestor."""
    out, n = [], 0
    for name, t in zip(names, tt):
        leaf = np.flatnonzero(t.is_leaf)
        ls = leaf[t.ctype[leaf] == p["target"]]
        if not len(ls):
            continue
        d = TR.trajectories(t, ls, p["xcls"], type2lin)
        d["traj"] += off
        d["group"], d["tree"], d["panel"] = d.src.values, name, key
        out.append(d[["panel", "tree", "traj", "group", "time", "cell", "is_leaf"]])
        off, n = off + len(ls), n + len(ls)
    return pd.concat(out, ignore_index=True), off, n


def main():
    t0 = time.time()
    tic = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)
    _, type2lin = D.labels()

    tic("trees")
    names, tt = TR.load()
    tic(f"    {len(names)} E9.5 trees")

    parts, off = [], 0
    for key, p in S.PROGRAMS.items():
        d, off, n = collect(names, tt, key, p, type2lin, off)
        parts.append(d)
        tic(f"    {key}: {n:,} {p['target']} leaves, {len(d):,} nodes")
    d = pd.concat(parts, ignore_index=True)

    tic("h5ad rows and latents")
    with h5py.File(S.H5AD, "r") as f:
        obs = np.array([n.decode() for n in f["obs"]["_index"][:]])
        want = pd.unique(d.cell.values)
        at = pd.Index(obs).get_indexer(want)
        d["row"] = d.cell.map(dict(zip(want, at))).values
        d = d[d.row >= 0].drop(columns="cell").reset_index(drop=True)
        u, inv = np.unique(d.row.values, return_inverse=True)
        Z = f["obsm"]["X_scvi"][:][u][inv].astype(np.float32)

    d = d.astype({"traj": "int32", "row": "int32", "time": "float32",
                  "is_leaf": "int8"})
    np.savez(os.path.join(S.NR, "cache_paths.npz"), Z=Z,
             **{c: d[c].values for c in D.NODE})
    for key, h in d.groupby("panel"):
        src = h.drop_duplicates("traj").group.replace("", "unclassified")
        tic(f"    {key}: " + ", ".join(f"{v:,} {k}" for k, v in
                                       src.value_counts().items()))
    tic(f"wrote cache_paths.npz ({len(d):,} nodes, {Z.shape[1]} latent dims) to {S.NR}")


if __name__ == "__main__":
    main()
