#!/usr/bin/env python
"""one control subtree carrying both forelimb field and heart leaves, with Tbx5 on
every node.

every leaf that is neither the forelimb field nor a heart cell type is pruned away,
along with the internal nodes left holding none, so the drawn tree is the two fates
only and heart means what it means on the fate panel. a control tree is a forest of
rooted subtrees; each is scored by the smaller of its two pruned leaf counts times
how balanced the two are, among those with at most MAX_LEAVES leaves left and a root
below TBX5_ON - most of the control roots are, and a subtree that starts Tbx5+ has no
onset to draw.

    python compute_subtree.py
"""
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
import data


def leaf_groups(g, sub, sub2type):
    """the leaves of a subtree that are the forelimb field or a heart cell type."""
    out = {}
    for n in g:
        if g.out_degree(n) == 0:
            s = sub.get(g.nodes[n]["snapped_name"])
            if s == S.FORELIMB:
                out[n] = S.FORELIMB
            elif sub2type.get(s) in data.HEART_TYPES:
                out[n] = S.HEART
    return out


def prune(g, keep):
    """the subgraph over the nodes carrying a kept leaf under them."""
    alive = set(keep)
    for n in reversed(list(nx.topological_sort(g))):
        if any(c in alive for c in g.successors(n)):
            alive.add(n)
    return g.subgraph(alive)


def accumulate(g, t, e):
    """per node: Tbx5 summed over the nodes from the root down to it, and the time the
    trajectory first carries it up through TBX5_ON.

    the sum counts each node on the path once, with no weighting by how long the branch
    it sits on lasts, so it is a count of Tbx5 seen along the trajectory rather than an
    integral over time.

    once a node is above the cut every node under it counts as on, so a trajectory that
    dips back below and crosses again is not marked twice: at most one onset sits on
    any root-to-leaf path.
    """
    cum, onset, on = {}, {}, {}
    for n in nx.topological_sort(g):
        p = next(iter(g.predecessors(n)), None)
        if p is None:
            cum[n], onset[n] = float(e[n]), t[n] if e[n] >= S.TBX5_ON else np.nan
        else:
            cum[n] = cum[p] + float(e[n])
            onset[n] = (t[p] + (t[n] - t[p]) * (S.TBX5_ON - e[p]) / (e[n] - e[p])
                        if not on[p] and e[p] < S.TBX5_ON <= e[n] else np.nan)
        on[n] = (p is not None and on[p]) or e[n] >= S.TBX5_ON
    return cum, onset


def main():
    sub = data.devmap_subtypes().to_dict()
    sub2type, _ = data.type_map()
    tbx5 = data.tbx5()

    cand, best = [], None
    for name, d in data.screen("control").items():
        t = d["tree"]
        for c in nx.weakly_connected_components(t):
            whole = t.subgraph(c)
            grp = leaf_groups(whole, sub, sub2type)
            counts = pd.Series(grp).value_counts()
            f, h = counts.get(S.FORELIMB, 0), counts.get(S.HEART, 0)
            if min(f, h) < S.MIN_BOTH:
                continue
            g = prune(whole, grp)
            root = next(x for x in g if g.in_degree(x) == 0)
            e0 = float(tbx5[g.nodes[root]["snapped_name"]])
            v = min(f, h) ** 2 / (f + h)
            cand.append({"tree": name, "root": root, "nodes": g.number_of_nodes(),
                         "leaves": f + h, "forelimb": f, "heart": h,
                         "pruned": whole.number_of_nodes() - g.number_of_nodes(),
                         "root_tbx5": e0, "score": v})
            if f + h <= S.MAX_LEAVES and e0 < S.TBX5_ON and (best is None
                                                            or v > best[0]):
                best = (v, name, g)
    pd.DataFrame(cand).sort_values("score", ascending=False).to_csv(
        f"{S.NR}/subtree_scan.csv", index=False)

    _, name, g = best
    nodes = list(nx.topological_sort(g))
    st = pd.Series([sub.get(g.nodes[n]["snapped_name"]) for n in nodes])
    grp = leaf_groups(g, sub, sub2type)
    d = pd.DataFrame({
        "node": nodes, "parent": [next(iter(g.predecessors(n)), "") for n in nodes],
        "time": [g.nodes[n]["time"] for n in nodes],
        "leaf": [g.out_degree(n) == 0 for n in nodes],
        "subtype": st.values, "cell_type": st.map(sub2type).values,
        "group": [grp.get(n) for n in nodes],
        "tbx5": tbx5.reindex([g.nodes[n]["snapped_name"] for n in nodes]).values})
    cum, onset = accumulate(g, d.set_index("node").time, d.set_index("node").tbx5)
    d["cum"] = d.node.map(cum)
    d["onset"] = d.node.map(onset)
    d.to_csv(f"{S.NR}/subtree.csv", index=False)

    c = pd.DataFrame(cand)
    print(f"=== {len(c)} candidate subtrees, {int((c.root_tbx5 < S.TBX5_ON).sum())} of "
          f"them rooted below {S.TBX5_ON}, best under {S.MAX_LEAVES} pruned leaves is "
          f"{name}", flush=True)
    print(f"{len(nodes)} nodes after pruning, {int(d.leaf.sum())} leaves, "
          f"E{d.time.min():.2f} to E{d.time.max():.2f}, root Tbx5 "
          f"{d.tbx5.iloc[0]:.3f}", flush=True)
    print(d[d.leaf].group.value_counts().to_string(), flush=True)
    print(d[d.leaf].subtype.value_counts().to_string(), flush=True)
    print(f"\nTbx5 0 to {d.tbx5.max():.2f}, accumulated {d['cum'].min():.2f} to "
          f"{d['cum'].max():.2f}, "
          f"{int(d.onset.notna().sum())} onsets at Tbx5 >= {S.TBX5_ON}", flush=True)


if __name__ == "__main__":
    main()
