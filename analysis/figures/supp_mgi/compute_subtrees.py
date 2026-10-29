"""one control subtree for each of panel c's examples, with the gene's expression on
every node -> numerical_results/subtrees.csv, subtrees.md.

a knockout changes a node when the cell it snapped to expresses the gene, and up to its
first change a knockout trajectory is the control one (the same snapped cells). so the
control tree alone shows where a knockout reaches a trajectory: at the first node on it
whose snapped cell expresses the gene.

a control tree is a forest of rooted subtrees, one per start cell. each is scored by its
readout leaves of the example's cell type that have the gene on at a node above them,
among those with at most MAX_LEAVES readout leaves. a readout leaf is a meta row within
RING_TOL of E9.0, as in readout.py. every other leaf is pruned away, along with the
internal nodes left holding none.

expression is the log1p CP10K of the node's snapped cell (the h5ad's X), the value the
knockout reads as on or off.

    python -m analysis.figures.supp_mgi.compute_subtrees
"""
import os
import pickle

import h5py
import networkx as nx
import numpy as np
import pandas as pd

from analysis.figures.perturb.readout import name_to_celltype
from analysis.figures.supp_mgi import style as S


class Expression:
    """one gene's log1p CP10K in named cells, read row by row out of the h5ad's csr X."""

    def __init__(self):
        self.f = h5py.File(S.STATES_H5AD, "r")
        self.row = pd.Index(self.f["obs/_index"][:].astype(str))
        self.var = pd.Index(self.f["var/_index"][:].astype(str))
        self.indptr = self.f["X/indptr"][:]

    def __call__(self, cells, gene):
        j, X = self.var.get_loc(gene), self.f["X"]
        out = {}
        for c in cells:
            r = self.row.get_loc(c)
            a, b = self.indptr[r], self.indptr[r + 1]
            hit = np.flatnonzero(X["indices"][a:b] == j)
            out[c] = float(X["data"][a + hit[0]]) if len(hit) else 0.0
        return out


def ancestors(t, n):
    """the nodes above n, nearest first."""
    out = []
    while (p := next(iter(t.predecessors(n)), None)) is not None:
        out.append(p)
        n = p
    return out


def prune(t, root, keep):
    """the subtree under root over the nodes carrying a kept leaf under them."""
    sub = t.subgraph(nx.descendants(t, root) | {root})
    alive = set(keep)
    for n in reversed(list(nx.topological_sort(sub))):
        if any(c in alive for c in sub.successors(n)):
            alive.add(n)
    return sub.subgraph(alive)


def main():
    name2ct = name_to_celltype()
    expr = Expression()
    ctrl, readout, root_of = {}, {}, {}
    for e in sorted(os.scandir(os.path.join(S.RING_SCREEN, "control")), key=lambda e: e.name):
        d = pickle.load(open(e.path, "rb"))
        tn, t, meta = os.path.splitext(e.name)[0], d["tree"], d["meta"]
        tw = np.abs(meta["time"].to_numpy(float) - S.RING_TARGET_TIME) < S.RING_TOL
        ctrl[tn], readout[tn] = t, set(meta["node_name"].to_numpy()[tw])
        r = root_of[tn] = {}
        for n in nx.topological_sort(t):
            p = next(iter(t.predecessors(n)), None)
            r[n] = n if p is None else r[p]
    n_leaves = pd.Series({(tn, r): k for tn in ctrl for r, k in
                          pd.Series([root_of[tn][n] for n in readout[tn]]).value_counts().items()})
    ex = pd.read_csv(os.path.join(S.NR, "examples.csv")).set_index(["gene", "cell_type"])
    missing = [p for p in S.SUBTREE_EXAMPLES if p not in ex.index]
    assert not missing, f"{missing} are not among b's examples"

    rows, lines = [], []
    for gene, ct in S.SUBTREE_EXAMPLES:
        # every readout leaf of the cell type: the gene in it, and above it
        leaves = []
        for tn, t in ctrl.items():
            for n in readout[tn]:
                if name2ct.get(t.nodes[n]["snapped_name"]) == ct:
                    leaves.append((tn, n, [t.nodes[a]["snapped_name"] for a in ancestors(t, n)]))
        e = expr({c for tn, n, up in leaves for c in [ctrl[tn].nodes[n]["snapped_name"], *up]},
                 gene)
        lv = pd.DataFrame([{"tree": tn, "root": root_of[tn][n],
                            "on": e[ctrl[tn].nodes[n]["snapped_name"]] > 0,
                            "above": any(e[c] > 0 for c in up)} for tn, n, up in leaves])

        # the subtree: the most of them with the gene on above, then the fewest leaves
        cand = lv.groupby(["tree", "root"]).agg(n_type=("above", "size"),
                                                reached=("above", "sum"))
        cand["leaves"] = n_leaves.reindex(cand.index).to_numpy()
        cand = cand[cand["leaves"] <= S.MAX_LEAVES].reset_index()
        best = cand.sort_values(["reached", "leaves", "tree", "root"],
                                ascending=[False, True, True, True]).iloc[0]
        t = ctrl[best["tree"]]
        g = prune(t, best["root"], readout[best["tree"]] & set(nx.descendants(t, best["root"])))
        nodes = list(nx.topological_sort(g))
        e = expr({t.nodes[n]["snapped_name"] for n in nodes}, gene)
        for n in nodes:
            rows.append({"gene": gene, "cell_type": ct, "tree": best["tree"], "node": n,
                         "parent": next(iter(g.predecessors(n)), ""),
                         "time": t.nodes[n]["time"], "leaf": g.out_degree(n) == 0,
                         "node_type": name2ct.get(t.nodes[n]["snapped_name"]),
                         "expression": e[t.nodes[n]["snapped_name"]]})
        lines.append(
            f"| {gene} | {ct} | {len(lv):,} | {lv['on'].mean():.3f} | "
            f"{lv['above'].mean():.3f} | {best['tree']} | {int(best['leaves'])} | "
            f"{int(best['n_type'])} | {int(best['reached'])} |")
        print(f"{gene} {ct}: {len(cand)} candidate subtrees, {best['tree']} "
              f"{best['root']}, {len(nodes)} nodes", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(S.NR, "subtrees.csv"), index=False)

    lines = [
        "# control subtrees (panel c)", "",
        "every control readout leaf of the cell type: the share with the gene on in the leaf's "
        "snapped cell, and at a node above it (where a knockout reaches it). then the subtree "
        "drawn: its readout leaves, those of the cell type, and those reached.", "",
        "| gene | cell type | leaves of the type | on in the leaf | on above | tree | "
        "subtree leaves | of the type | reached |",
        "| --- |" + " --- |" * 8, *lines, "",
    ]
    out = os.path.join(S.NR, "subtrees.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
