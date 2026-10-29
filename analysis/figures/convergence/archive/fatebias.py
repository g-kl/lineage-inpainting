"""shared fate-bias node scan + ON/OFF mean difference (rows 3-4).

a progenitor node's fate bias toward a target = fraction of its labeled descendant
leaves of that type. these helpers score every internal E9.5 node in a time window,
read each node's source cell type, and split a source's nodes ON/OFF per gene for the
difference in mean fate bias. reused by compute_fatebias.py and
compute_fatebias_sweep.py.

the low-level tree / expression helpers (node_cell, celltype_of, read_rows) come
from compute_trajectories so nothing is duplicated.
"""
import pickle

import h5py
import networkx as nx
import numpy as np
import pandas as pd

import style as S
from compute_trajectories import node_cell, celltype_of, read_rows  # noqa: F401

UNKNOWN = "unknown"


def descendant_frac(g, is_target, is_labeled):
    """per node: (# target-type, # labeled) descendant leaves, via a leaves-up pass."""
    tgt, tot = {}, {}
    for n in reversed(list(nx.topological_sort(g))):
        succ = list(g.successors(n))
        if not succ:
            tot[n] = 1 if is_labeled(n) else 0
            tgt[n] = 1 if is_target(n) else 0
        else:
            tot[n] = sum(tot[c] for c in succ)
            tgt[n] = sum(tgt[c] for c in succ)
    return tgt, tot


def descendant_type_counts(g, leaf_type):
    """per node: {type_idx: count} of descendant leaves + total labeled leaves."""
    counts, total = {}, {}
    for n in reversed(list(nx.topological_sort(g))):
        succ = list(g.successors(n))
        if not succ:
            ti = leaf_type(n)
            counts[n] = {ti: 1} if ti is not None else {}
            total[n] = 1 if ti is not None else 0
        else:
            d, t = {}, 0
            for c in succ:
                t += total[c]
                for k, v in counts[c].items():
                    d[k] = d.get(k, 0) + v
            counts[n], total[n] = d, t
    return counts, total


def onoff_diff(bias, X, cutoff, min_group):
    """per gene: mean(bias | on) - mean(bias | off), on = X > cutoff. signed
    (positive = ON nodes are more fate-biased). NaN if either group < min_group
    nodes. vectorised over all genes at once."""
    if len(bias) == 0:
        return np.full(X.shape[1], np.nan)
    on = (X > cutoff).astype(np.float32)
    n1 = on.sum(0)
    n0 = on.shape[0] - n1
    s1, s0 = bias @ on, bias.sum() - bias @ on
    with np.errstate(invalid="ignore", divide="ignore"):
        d = s1 / n1 - s0 / n0
    d[(n1 < min_group) | (n0 < min_group)] = np.nan
    return d


def load_trees():
    """(treedata, E9.5 tree names, node->cell assignments, subtype series, sub2type)."""
    import treedata
    sub2type, _, _ = S.load_labels()
    subtype = pd.read_csv(S.SUBTYPES, index_col=0)["cell_subtype"]
    td = treedata.read_h5td(S.TREES)
    e95 = [n for n in td.obst if n.startswith("E9.5")]
    with open(S.ASSIGN_PKL, "rb") as fh:
        assignments = pickle.load(fh)
    return td, e95, assignments, subtype, sub2type


def scan_target(td, e95, assignments, subtype, sub2type, family):
    """every internal E9.5 node in the window with >=1 target-family descendant leaf:
    its (assigned cell, source cell type, fate bias). nodes already in the target
    family are dropped, so only the fan-in from other tissues remains."""
    tgt_subtypes = {s for s, ct in sub2type.items() if ct in family}
    is_tgt = lambda n: subtype.get(n) in tgt_subtypes
    is_lab = lambda n: subtype.get(n) is not None
    cells, stypes, bias = [], [], []
    for tname in e95:
        g = td.obst[tname]
        amap = assignments.get(tname, {})
        tgt_c, tot_c = descendant_frac(g, is_tgt, is_lab)
        for n in g.nodes():
            if g.out_degree(n) == 0 or tgt_c[n] == 0:
                continue
            if not (S.BIN_LO <= float(g.nodes[n].get("time", np.nan)) < S.BIN_HI):
                continue
            cell = node_cell(g, amap, n)
            if cell is None or cell == UNKNOWN:
                continue
            st = celltype_of(cell, subtype, sub2type)
            if st is None or st in family:
                continue
            cells.append(cell)
            stypes.append(st)
            bias.append(tgt_c[n] / tot_c[n])
    return cells, np.array(stypes), np.array(bias, float)


def read_expression(cells):
    """(expression for cells found in the h5ad, ok-mask into `cells`, gene names)."""
    with h5py.File(S.H5AD, "r") as f:
        obs_names = np.array([n.decode() for n in f["obs"]["_index"][:]])
        genes = np.array([g.decode() for g in f["var"]["_index"][:]])
    name2row = {n: i for i, n in enumerate(obs_names)}
    rows = sorted({name2row[c] for c in cells if c in name2row})
    X = read_rows(S.H5AD, rows).toarray().astype(np.float32)
    cell2row = {obs_names[r]: i for i, r in enumerate(rows)}
    ok = np.array([c in cell2row for c in cells])
    Xn = X[[cell2row[c] for c in cells if c in cell2row]]
    return Xn, ok, genes
