"""row-3 middle cache: one example EC-generating subtree for the fate-bias illustration.

scans the E9.5 trees for internal nodes near style.TREE_ROOT_TIME whose subtree has a
drawable number of leaves, a mixed root EC fate bias, and a wide spread of EC fate bias
across its internal nodes (some clades commit to EC while siblings do not); among those
it picks the subtree where style.TREE_GENE (Kdr) expression best tracks the fate bias
across nodes. records per node the square-dendrogram layout, cell type, EC fate bias,
and Kdr expression.

writes numerical_results/fatebias_ec_tree.pkl.

    python analysis/figures/convergence/compute_fatebias_tree.py
"""
import os
import pickle
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import style as S
import fatebias as FB
from compute_fanin import _square_layout

KEY = "ec"


def main():
    family = S.FANIN[KEY]["exclude"]
    td, e95, assignments, subtype, sub2type = FB.load_trees()
    tgt_subtypes = {s for s, ct in sub2type.items() if ct in family}
    is_tgt = lambda n: subtype.get(n) in tgt_subtypes
    is_lab = lambda n: subtype.get(n) is not None
    lo, hi = S.TREE_LEAF_RANGE

    print(f"[1/3] collecting candidate subtrees rooted near E{S.TREE_ROOT_TIME} ...",
          flush=True)
    cands = []
    for tname in e95:
        g = td.obst[tname]
        amap = assignments.get(tname, {})
        tgt_c, tot_c = FB.descendant_frac(g, is_tgt, is_lab)
        for n in g.nodes():
            if g.out_degree(n) == 0 or tot_c[n] == 0:
                continue
            t = float(g.nodes[n].get("time", np.nan))
            if abs(t - S.TREE_ROOT_TIME) > S.TREE_ROOT_TIME_TOL:
                continue
            desc = nx.descendants(g, n) | {n}
            leaves = [m for m in desc if g.out_degree(m) == 0]
            if not (lo <= len(leaves) <= hi):
                continue
            if not (S.TREE_ROOT_BIAS_RANGE[0] <= tgt_c[n] / tot_c[n]
                    <= S.TREE_ROOT_BIAS_RANGE[1]):
                continue
            internal = [m for m in desc
                        if g.out_degree(m) > 0 and tot_c[m] >= S.TREE_MIN_INTERNAL_LEAVES]
            if len(internal) < 3 or sum(is_tgt(m) for m in leaves) < 3:
                continue
            biases = [tgt_c[m] / tot_c[m] for m in internal]
            if max(biases) - min(biases) < S.TREE_MIN_SPREAD:
                continue
            nodes = [(m, (tgt_c[m] / tot_c[m]) if tot_c[m] else np.nan,
                      FB.node_cell(g, amap, m)) for m in desc]
            cands.append({"tname": tname, "root": n, "desc": desc, "nodes": nodes,
                          "spread": max(biases) - min(biases), "nleaf": len(leaves)})
    if not cands:
        raise SystemExit("no subtree matched the criteria; relax style.TREE_* knobs")

    print(f"[2/3] scoring {len(cands)} candidates by {S.TREE_GENE} vs fate bias ...",
          flush=True)
    uniq = sorted({c for cd in cands for _, _, c in cd["nodes"] if c})
    Xn, ok, genes = FB.read_expression(uniq)
    gidx = int(np.where(genes == S.TREE_GENE)[0][0])
    cell2g = dict(zip([c for c, o in zip(uniq, ok) if o], Xn[:, gidx]))

    best = None
    for cd in cands:
        fb, kd = [], []
        for _, b, c in cd["nodes"]:
            v = cell2g.get(c, np.nan)
            if np.isfinite(b) and np.isfinite(v):
                fb.append(b)
                kd.append(v)
        if len(fb) < 5 or np.std(kd) == 0:
            continue
        r = float(np.corrcoef(fb, kd)[0, 1])
        if np.isfinite(r) and (best is None or r > best[0]):
            best = (r, cd)
    if best is None:
        raise SystemExit("no candidate had scorable Kdr vs fate bias")
    corr, cd = best
    tname, root, desc = cd["tname"], cd["root"], cd["desc"]
    g = td.obst[tname]
    amap = assignments.get(tname, {})
    tgt_c, tot_c = FB.descendant_frac(g, is_tgt, is_lab)
    sub = g.subgraph(desc)
    pos = _square_layout(sub, root)
    print(f"      {tname} root={root} leaves={cd['nleaf']} spread={cd['spread']:.2f} "
          f"{S.TREE_GENE}-fate-bias r={corr:.2f}", flush=True)

    recs = []
    for m in sub.nodes():
        cell = FB.node_cell(g, amap, m)
        ct = FB.celltype_of(cell, subtype, sub2type) if cell else None
        recs.append({"node": m, "x": pos[m][0], "y": pos[m][1],
                     "is_leaf": g.out_degree(m) == 0, "is_root": m == root,
                     "cell_type": ct, "is_ec": bool(ct in family),
                     "fate_bias": (tgt_c[m] / tot_c[m]) if tot_c[m] else np.nan,
                     "gene_expr": float(cell2g.get(cell, np.nan))})

    out = {"tree": tname, "root": root, "gene": S.TREE_GENE, "n_leaves": cd["nleaf"],
           "family": sorted(family), "nodes": recs, "edges": list(sub.edges())}
    with open(os.path.join(S.NR, f"fatebias_{KEY}_tree.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    print(f"[3/3] wrote fatebias_{KEY}_tree.pkl ({len(recs)} nodes)", flush=True)


if __name__ == "__main__":
    main()
