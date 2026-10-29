#!/usr/bin/env python
"""the shift curves and the split table, from one pass over the control trees.

a readout leaf is scored under the lateral plate subtype its control cell carries, and
a node holds that trajectory from its own time up to its child's, so the leaves under
that child sit in that node's state over that stretch.

shift  control and knockout share a topology tree by tree, so a node is shifted when
       the knockout snapped it somewhere else or took it off the manifold at the
       pipeline's ratio. the curve is the share of a subtype's leaves whose trajectory
       is shifted at that time.
split  per fate trajectory of a clone that makes both fates - a clone being one
       founder's connected component of the forest: the time its two fates part and
       the time Tbx5 comes on along it.

    python compute_curves.py
"""
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
import data
from analysis.pipelines.perturb.perturb_result import off_manifold_descendants
from compute_subtree import accumulate, leaf_groups, prune

GRID = np.round(np.arange(S.SHIFT_T0, data.LEAF_TIME + S.SHIFT_STEP / 2,
                          S.SHIFT_STEP), 3)


def readout_leaves(tree):
    """the target-time leaves of a control tree."""
    return {n for n in tree.nodes()
            if tree.out_degree(n) == 0
            and abs(tree.nodes[n].get("time", -1) - data.LEAF_TIME) < data.LEAF_TOL}


def diverged(ct, pt):
    """nodes the knockout moved: a different snapped cell, or off the manifold."""
    bad = off_manifold_descendants(pt, S.OFF_RATIO)
    return {n for n in ct.nodes()
            if n in bad or pt.nodes[n].get("snapped_name")
            != ct.nodes[n].get("snapped_name")}


def split_onset(name, ct, sub, sub2type, tbx5):
    """per fate trajectory of one control tree: when its two fates part, and when Tbx5
    comes on along it.

    a clone is pruned to its forelimb field and heart leaves, as the subtree panel
    prunes, so the two fates mean there what they mean here. the split is the last node
    on the trajectory still holding both of them; the onset is `accumulate`'s, so a
    trajectory that dips back under the cut and crosses again is not counted twice.
    """
    out = []
    for c in nx.weakly_connected_components(ct):
        whole = ct.subgraph(c)
        grp = leaf_groups(whole, sub, sub2type)
        if len(set(grp.values())) < 2:
            continue
        g = prune(whole, grp)
        order = list(nx.topological_sort(g))
        t = pd.Series([g.nodes[n]["time"] for n in order], index=order)
        e = pd.Series(tbx5.reindex([g.nodes[n]["snapped_name"] for n in order]).values,
                      index=order)
        _, onset = accumulate(g, t, e)
        both = {}                               # the fates under each node
        for x in reversed(order):
            both[x] = ({grp[x]} if x in grp
                       else set().union(*(both[k] for k in g.successors(x))))
        root = order[0]
        for leaf, fate in grp.items():
            path = nx.shortest_path(g, root, leaf)
            out.append((name, root, leaf, fate,
                        max(t[n] for n in path if len(both[n]) == 2),
                        next((onset[n] for n in path if not np.isnan(onset[n])),
                             np.nan)))
    return out


def one_tree(name, ct, pt, sub, subtypes):
    """(per subtype shift curves, per subtype leaf totals) for one tree."""
    order = list(nx.topological_sort(ct))
    idx = {n: i for i, n in enumerate(order)}
    t = np.array([ct.nodes[n].get("time", np.nan) for n in order])

    col = {s: k for k, s in enumerate(subtypes)}
    cnt = np.zeros((len(order), len(subtypes)), np.int64)
    for n in readout_leaves(ct):
        k = col.get(sub.get(ct.nodes[n].get("snapped_name")))
        if k is not None:
            cnt[idx[n], k] = 1
    leaves = cnt.sum(0)
    for n in reversed(order):
        for c in ct.successors(n):
            cnt[idx[n]] += cnt[idx[c]]

    div = diverged(ct, pt)
    shifted = np.zeros((len(GRID), len(subtypes)))
    for n in order:
        i = idx[n]
        if n not in div:
            continue
        for b, c in [(t[idx[c]], cnt[idx[c]]) for c in ct.successors(n)] \
                or [(np.inf, cnt[i])]:
            lo, hi = np.searchsorted(GRID, [t[i], b])
            if hi > lo and c.any():
                shifted[lo:hi] += c

    print(f"[{name}] {leaves.sum():,} scored leaves", flush=True)
    return shifted, leaves


def main():
    sub = data.devmap_subtypes().to_dict()
    subtypes = data.lpm_subtypes()
    sub2type, _ = data.type_map()
    tbx5 = data.tbx5()
    ctrl, pert = data.screen("control"), data.screen(S.TARGET)

    tot = np.zeros((len(GRID), len(subtypes)))
    leaves = np.zeros(len(subtypes), np.int64)
    rows = []
    for name, d in ctrl.items():
        o, l = one_tree(name, d["tree"], pert[name]["tree"], sub, subtypes)
        tot += o
        leaves += l
        rows += split_onset(name, d["tree"], sub, sub2type, tbx5)

    p = pd.DataFrame({"time": np.repeat(GRID, len(subtypes)),
                      "subtype": np.tile(subtypes, len(GRID)),
                      "leaves": np.tile(leaves, len(GRID)),
                      "shifted": tot.ravel()})
    p["frac_shifted"] = p.shifted / p.leaves.where(p.leaves > 0)
    p.to_csv(f"{S.NR}/shift_curves.csv", index=False)

    end = p[p.time == GRID[-1]].set_index("subtype")
    end = end[end.leaves >= S.MIN_LEAVES].sort_values("frac_shifted", ascending=False)
    end[["leaves", "frac_shifted"]].to_csv(f"{S.NR}/shift_readout.csv")

    b = pd.DataFrame(rows, columns=["tree", "clone", "leaf", "fate", "t_split",
                                    "t_on"])
    b.to_csv(f"{S.NR}/split_onset.csv", index=False)

    print(f"\n=== {len(ctrl)} control trees, at the E9.5 readout", flush=True)
    print(end[["leaves", "frac_shifted"]].round(3).to_string(), flush=True)
    print(f"\n=== {b.clone.nunique()} clones make both fates, {len(b):,} fate "
          f"trajectories, {b.t_on.isna().mean():.1%} of them never Tbx5+", flush=True)
    print(b.assign(lag=b.t_on - b.t_split).groupby("fate")
          .agg(trajectories=("leaf", "size"), split=("t_split", "median"),
               on=("t_on", "median"), lag=("lag", "median"),
               after=("lag", lambda x: (x > 0).mean()))
          .round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
