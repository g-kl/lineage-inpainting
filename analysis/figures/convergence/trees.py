"""the E9.5 trees as flat arrays, and the ancestor paths of a set of leaves."""
import os
import pickle
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import style as S


class Tree:
    """one tree in root-first order. `cell` is the model's assigned cell per node
    (the leaf itself for leaves), `ctype` its cell type."""

    def __init__(self, g, am, subtype, sub2type):
        self.nodes = np.array(list(nx.topological_sort(g)), dtype=object)
        i = {n: k for k, n in enumerate(self.nodes)}
        self.parent = np.full(len(self.nodes), -1)
        for u, v in g.edges():
            self.parent[i[v]] = i[u]
        self.is_leaf = np.array([g.out_degree(n) == 0 for n in self.nodes])
        self.time = np.array([S.T_LEAF if lf else g.nodes[n].get("time", np.nan)
                              for n, lf in zip(self.nodes, self.is_leaf)], float)
        cell = [n if lf else am.get(n, S.UNKNOWN)
                for n, lf in zip(self.nodes, self.is_leaf)]
        self.cell = np.array([c if c is not None else S.UNKNOWN for c in cell],
                             dtype=object)
        self.ctype = (pd.Series(self.cell).map(subtype).str.lower().map(sub2type)
                      .fillna("").values)
        self.assigned = (self.cell != S.UNKNOWN) & (self.ctype != "")

    def source(self, cls, type2lin):
        """per node, the group of the nearest ancestor-or-self the classifier gives a
        group to, and its time. one sweep, so it is O(nodes)."""
        own = np.array([cls(t, type2lin.get(t, "")) if a else ""
                        for t, a in zip(self.ctype, self.assigned)], dtype=object)
        near = np.full(len(self.nodes), -1)
        for k in range(len(self.nodes)):          # root-first, so the parent is done
            p = self.parent[k]
            near[k] = k if own[k] else (near[p] if p >= 0 else -1)
        return own, near

    def paths(self, idx):
        """(n, depth) matrix of ancestor node indices for `idx`, -1 past the root.
        column 0 is the node itself."""
        cur, alive, cols = idx.copy(), np.ones(len(idx), bool), []
        while alive.any():
            cols.append(np.where(alive, cur, -1))
            p = self.parent[cur]
            alive = alive & (p >= 0)
            cur = np.where(p >= 0, p, cur)
        return np.stack(cols, 1)


def load():
    """flat trees, cached in numerical_results/; delete cache_trees.pkl to rebuild."""
    cache = os.path.join(S.NR, "cache_trees.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    import treedata
    td = treedata.read_h5td(S.TREES)
    with open(S.ASSIGN_PKL, "rb") as fh:
        amaps = pickle.load(fh)
    subtype = pd.read_csv(S.SUBTYPES, index_col=0)["cell_subtype"]
    sub2type, _ = D.labels()
    names = sorted(n for n in td.obst if n.startswith("E9.5"))
    tt = [Tree(td.obst[n], amaps.get(n, {}), subtype, sub2type) for n in names]
    with open(cache, "wb") as fh:
        pickle.dump((names, tt), fh)
    return names, tt


def trajectories(t, leaves, cls, type2lin):
    """one row per (trajectory, node) for the ancestor paths of `leaves`, with the
    source class of each trajectory carried alongside."""
    own, near = t.source(cls, type2lin)
    up = t.parent[leaves]
    strict = np.where(up >= 0, near[np.maximum(up, 0)], -1)   # strict ancestors only
    src = np.where(strict >= 0, own[np.maximum(strict, 0)], "")

    A = t.paths(leaves)
    ok = np.maximum(A, 0)
    keep = (A >= 0) & t.assigned[ok] & (t.time[ok] >= S.T_MIN)
    r, c = np.nonzero(keep)
    n = A[r, c]
    return pd.DataFrame(dict(traj=r, src=src[r], node=n, time=t.time[n],
                             cell=t.cell[n], is_leaf=(c == 0).astype(int)))
