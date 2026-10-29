"""score lineage inpainting's cell type map against the qiu tree -> numerical_results/transitions.csv.
one row per source_type -> target_type over the whole map, identity included. see concept.md.

    python -m analysis.figures.supp_summary.compute_verdicts
"""
import collections
import os
import pickle

import networkx as nx
import pandas as pd

from analysis.figures.supp_summary import style as S


def neighbours(q):
    """qiu state pairs side by side rather than in sequence: spatial continuity in the
    unpruned file, or siblings under one parent."""
    d = pd.read_csv(S.QIU_FULL)
    d = d[d.edge_type == "Spatial continuity"]
    pairs = {frozenset(p) for p in zip(d.source, d.target)}
    for n in q:
        kids = list(q.successors(n))
        pairs |= {frozenset((a, b)) for i, a in enumerate(kids) for b in kids[i + 1:]}
    return pairs


def best(P, A, B):
    """shortest qiu path from any of A to any of B within MAX_HOPS, or None."""
    p = [P[a][b] for a in A if a in P for b in B if b in P[a] and a != b]
    return min(p, key=len) if p else None


def judge(s, t, m, P, nb):
    """(verdict, category, qiu path) for one type transition."""
    A, B = m.get(s, []), m.get(t, [])
    if s == t:
        return "identity", "match", None
    if not (A and B):
        return "unmapped", "unmapped", None
    if set(A) & set(B):
        return "collapsed", "match", None   # qiu holds one state where we hold several
    if (f := best(P, A, B)) is not None:
        return "match", "match", f
    if (r := best(P, B, A)) is not None:
        return "reversed", "reversed", r
    near = any(frozenset((a, b)) in nb for a in A for b in B)
    return "mismatch", "boundary" if near else "other", None


def main():
    with open(S.MAP_PKL, "rb") as f:
        c = pickle.load(f)
    g, labels = c["graph"], c["name_to_labels"].values()
    d = pd.read_csv(S.QIU)
    q = nx.DiGraph(zip(d.source, d.target))
    m = pd.read_csv(S.MAP_CSV, dtype=str).fillna("")
    m = {r.cell_type: [x.strip() for x in r.qiu_types.split(";") if x.strip()]
         for r in m.itertuples()}
    drawn = {ct for ct, _ in g.nodes()}
    assert drawn <= set(m), (f"no row in celltype_map.csv, run propose_map: "
                             f"{sorted(drawn - set(m))}")
    bad = {x for v in m.values() for x in v} - set(q)
    assert not bad, f"celltype_map.csv names no qiu state calls: {sorted(bad)}"

    # drop the EXCLUDE lineages and every type whose cells all sit in them
    keep = {v["cell_type"] for v in labels if v["lineage"] not in S.EXCLUDE}
    out = S.EXCLUDE | ({v["cell_type"] for v in labels} - keep)
    g = g.subgraph(n for n in g if n[0] not in out)

    cells = collections.Counter()
    for (s, _), (t, _), e in g.edges(data=True):
        cells[(s, t)] += int(e.get("count", 0))
    P = {n: nx.single_source_shortest_path(q, n, cutoff=S.MAX_HOPS) for n in q}
    nb = neighbours(q)
    rows = []
    for (s, t), n in cells.items():
        v, cat, path = judge(s, t, m, P, nb)
        rows.append((s, t, n, v, cat, " -> ".join(path) if path else ""))
    df = pd.DataFrame(rows, columns=["source_type", "target_type", "count", "verdict",
                                     "category", "qiu_path"])
    df = df.sort_values("count", ascending=False)
    df.to_csv(os.path.join(S.NR, "transitions.csv"), index=False)

    t = df.groupby("category")["count"].agg(["size", "sum"]).reindex(S.CATEGORIES)
    t.columns = ["edges", "cells"]
    t["cells_pct"] = (100 * t.cells / t.cells.sum()).round(1)
    print(t.to_string())
    print(f"wrote {S.NR}")


if __name__ == "__main__":
    main()
