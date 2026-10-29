"""row-3 cache: convergence-path expression features + trajectory samples.

parametrised by style.FANIN (one entry per fate: ec, epi, cmn). for every E9.5
leaf of the fate, trace its ancestor chain and keep the nodes in the t_anc->9.5
window at their true times (t_anc is per-fate; epicardium reaches back to E8); the
node nearest t_anc gives the source cell type (fate-family sources dropped). per
path, per gene we fit the expression-vs-time
OLS slope (cov(t,x)/var(t)) and record the latest-time endpoint level; both are
averaged within each source cell type. a matched background of non-fate paths
(same source, same window) is scored identically so general maturation ramps can
be subtracted downstream.

this script only produces the raw FEATURES (per-source slope / endpoint matrices)
and the per-cell trajectory SAMPLES. the core / path-specific / source-memory gene
DEFINITIONS live in the swappable gene_sets.py and are applied at plot time, so
changing a definition never re-reads expression.

writes per fate key:
  numerical_results/{key}_slope.csv, {key}_bg_slope.csv, {key}_end.csv
  numerical_results/{key}_traj.npz  (X float16 for fate-path cells, source+time)

    python analysis/figures/convergence/compute_trajectories.py [key ...]
"""
import os
import pickle
import sys
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))
import style as S

UNKNOWN = "unknown"


def celltype_of(cell, subtype, sub2type):
    return sub2type.get(subtype.get(cell))


def node_cell(g, amap, node):
    if node in amap:
        return amap[node]
    return node if g.out_degree(node) == 0 else None


def ancestor_chain(g, leaf):
    nodes, cur = [leaf], leaf
    while True:
        preds = list(g.predecessors(cur))
        if not preds:
            break
        cur = preds[0]
        nodes.append(cur)
    return nodes


def trace_path(g, amap, leaf, subtype, sub2type, family, t0):
    """(source cell type @~t0, [(time, cell), ...]) or None; fate sources dropped."""
    nodes = ancestor_chain(g, leaf)
    ts = np.array([float(g.nodes[n].get("time", np.nan)) for n in nodes])
    d = np.abs(ts - t0)
    j = int(np.nanargmin(d))
    if d[j] > S.TIME_TOL:
        return None
    src = celltype_of(node_cell(g, amap, nodes[j]), subtype, sub2type)
    if src is None or src in family:
        return None
    pts = []
    for n, t in zip(nodes, ts):
        if not (t0 - S.TIME_TOL <= t <= S.T1 + S.TIME_TOL):
            continue
        c = node_cell(g, amap, n)
        if c and c != UNKNOWN:
            pts.append((t, c))
    pts.sort()
    return (src, pts) if len(pts) >= S.MIN_PTS else None


def read_rows(h5ad, rows):
    with h5py.File(h5ad, "r") as f:
        X = f["X"]
        data, indices, indptr = X["data"], X["indices"], X["indptr"][:]
        ncol = int(X.attrs["shape"][1])
        d, i, ptr = [], [], [0]
        for r in rows:
            s, e = int(indptr[r]), int(indptr[r + 1])
            d.append(data[s:e]); i.append(indices[s:e]); ptr.append(ptr[-1] + (e - s))
    return sp.csr_matrix((np.concatenate(d), np.concatenate(i), np.array(ptr)),
                         shape=(len(rows), ncol))


def trace_fate(td, e95, assignments, family, subtype, sub2type, t0):
    """fate paths (leaf cell type in family) and matched non-fate background."""
    ct = lambda c: celltype_of(c, subtype, sub2type)
    paths, bg = [], []
    for tname in e95:
        g = td.obst[tname]
        amap = assignments.get(tname, {})
        for leaf in g.nodes():
            if g.out_degree(leaf) > 0:
                continue
            r = trace_path(g, amap, leaf, subtype, sub2type, family, t0)
            if r is not None:
                (paths if ct(leaf) in family else bg).append(r)
    return paths, bg


def score(path_list, sources, X, cell2row, genes):
    """per-source mean OLS slope and mean endpoint level (genes x sources)."""
    G = X.shape[1]
    ssum = {s: np.zeros(G) for s in sources}
    esum = {s: np.zeros(G) for s in sources}
    cnt, ecnt = defaultdict(int), defaultdict(int)
    for src, pts in path_list:
        tt = np.array([t for t, c in pts if c in cell2row])
        idx = [cell2row[c] for _, c in pts if c in cell2row]
        if len(idx) < S.MIN_PTS:
            continue
        esum[src] += X[idx[-1]]
        ecnt[src] += 1
        tc = tt - tt.mean()
        denom = float((tc ** 2).sum())
        if denom <= 0:
            continue
        ssum[src] += (tc @ X[idx]) / denom
        cnt[src] += 1
    ms = pd.DataFrame({s: ssum[s] / max(cnt[s], 1) for s in sources}, index=genes)
    me = pd.DataFrame({s: esum[s] / max(ecnt[s], 1) for s in sources}, index=genes)
    return ms, me


def build(key, sources, paths, X, cell2row, genes):
    ms, me = score(paths[0], sources, X, cell2row, genes)
    bs, _ = score(paths[1], sources, X, cell2row, genes)
    ms.to_csv(os.path.join(S.NR, f"{key}_slope.csv"))
    bs.to_csv(os.path.join(S.NR, f"{key}_bg_slope.csv"))
    me.to_csv(os.path.join(S.NR, f"{key}_end.csv"))

    # per-source (cell, time) samples for smoothing, dedup shared ancestors.
    src_ct = {s: {} for s in sources}
    for src, pts in paths[0]:
        for t, c in pts:
            if c in cell2row:
                src_ct[src].setdefault(c, t)
    cells = sorted({c for s in sources for c in src_ct[s]})
    row = {c: i for i, c in enumerate(cells)}
    Xf = X[[cell2row[c] for c in cells]].astype(np.float16)
    s_row, s_src, s_t = [], [], []
    for si, s in enumerate(sources):
        for c, t in src_ct[s].items():
            s_row.append(row[c]); s_src.append(si); s_t.append(t)
    np.savez_compressed(
        os.path.join(S.NR, f"{key}_traj.npz"),
        X=Xf, genes=genes, sources=np.array(sources),
        sample_row=np.array(s_row), sample_src=np.array(s_src),
        sample_t=np.array(s_t, float))
    print(f"  [{key}] wrote slope/bg_slope/end.csv + traj.npz "
          f"(X {Xf.shape}, {len(s_row):,} samples)", flush=True)


def main(keys=None):
    keys = keys or list(S.FANIN)
    rng = np.random.default_rng(S.SEED)
    sub2type, _, _ = S.load_labels()
    subtype = pd.read_csv(S.SUBTYPES, index_col=0)["cell_subtype"]

    import treedata
    td = treedata.read_h5td(S.TREES)
    e95 = [n for n in td.obst if n.startswith("E9.5")]
    with open(S.ASSIGN_PKL, "rb") as fh:
        assignments = pickle.load(fh)

    # trace every fate first, so expression is read once for the union of cells.
    fates = {}
    for key in keys:
        print(f"[{key}] tracing {S.FANIN[key]['focus']} paths (source @ E"
              f"{S.FANIN[key]['t_anc']}) ...", flush=True)
        paths, bg_all = trace_fate(td, e95, assignments, S.FANIN[key]["exclude"],
                                   subtype, sub2type, S.FANIN[key]["t_anc"])
        sources = list(pd.Series([s for s, _ in paths]).value_counts()
                       .index[:S.K_SOURCES])
        paths = [(s, p) for s, p in paths if s in sources]
        by_src = defaultdict(list)
        for r in bg_all:
            by_src[r[0]].append(r)
        bg = []
        for s in sources:
            grp = by_src[s]
            if len(grp) > S.BG_MAX:
                grp = [grp[i] for i in rng.choice(len(grp), S.BG_MAX, replace=False)]
            bg += grp
        fates[key] = (sources, (paths, bg))
        print(f"  {len(paths):,} paths, {len(bg):,} background; sources: {sources}",
              flush=True)

    print("reading expression ...", flush=True)
    need = {c for _, (p, b) in fates.values() for _, pts in p + b for _, c in pts}
    with h5py.File(S.H5AD, "r") as f:
        obs_names = np.array([n.decode() for n in f["obs"]["_index"][:]])
        genes = np.array([g.decode() for g in f["var"]["_index"][:]])
    name2row = {n: i for i, n in enumerate(obs_names)}
    rows = sorted({name2row[c] for c in need if c in name2row})
    X = read_rows(S.H5AD, rows).toarray().astype(np.float32)
    cell2row = {obs_names[r]: i for i, r in enumerate(rows)}
    print(f"  {X.shape[0]:,} cells x {X.shape[1]:,} genes", flush=True)

    for key, (sources, paths) in fates.items():
        build(key, sources, paths, X, cell2row, genes)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
