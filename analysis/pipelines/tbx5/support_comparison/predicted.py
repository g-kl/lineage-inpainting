#!/usr/bin/env python
"""the Tbx5 knockout leaves the model predicts, resolved to E9.5 devmap cells.

a target-time leaf that stayed on the manifold contributes the cell it snapped to under
either convention. `C.MODE` decides what happens to one that went off it: `prune` drops
it, which is the screen's own convention, and `freeze` contributes the snapped cell of its
last on-manifold ancestor instead. a frozen state sits at E8.0-E9.0, so it is snapped
forward to its nearest E9.5 devmap cell in the devmap latent.

    python -m analysis.pipelines.tbx5.support_comparison.predicted
"""
import networkx as nx
import numpy as np
import pandas as pd

from analysis.pipelines.perturb.perturb_result import off_manifold_descendants
from analysis.pipelines.tbx5.support_comparison import common as C


def last_on_manifold(tree, bad):
    """per node, the snapped cell of its last on-manifold ancestor."""
    rep = {}
    for n in nx.topological_sort(tree):
        if n not in bad:
            rep[n] = tree.nodes[n].get("snapped_name")
        else:
            p = list(tree.predecessors(n))
            rep[n] = rep.get(p[0]) if p else None
    return rep


def readout(trees, mode=C.MODE):
    """(tree, node, cell, frozen) per target-time leaf under one off-manifold convention.

    under `prune` the frozen block is skipped and every row comes back frozen=False.
    """
    rows = []
    for name, d in trees.items():
        meta = d["meta"]
        tw = np.abs(meta.time.to_numpy(float) - C.LEAF_TIME) < C.LEAF_TOL
        bad = off_manifold_descendants(d["tree"], C.OFF_RATIO)
        node, snap = meta.node_name.to_numpy(), meta.snapped_name.to_numpy()
        on = np.fromiter((n not in bad for n in node), bool, len(node))
        rows.append(pd.DataFrame({"tree": name, "node": node[tw & on],
                                  "cell": snap[tw & on], "frozen": False}))
        if mode != "freeze":
            continue
        rep = last_on_manifold(d["tree"], bad)
        f = [(n, rep[n]) for n in node[tw & ~on] if rep.get(n) is not None]
        rows.append(pd.DataFrame({"tree": name, "node": [n for n, _ in f],
                                  "cell": [c for _, c in f], "frozen": True}))
    return pd.concat(rows, ignore_index=True)


def to_e95(cells):
    """pre-E9.5 cells -> their nearest E9.5 devmap cell in the devmap latent."""
    early = pd.Index(sorted({c for c in cells if not c.startswith(C.REF_STAGE)}))
    if not len(early):
        return {}
    late = C.devmap_index()
    late = late[late.str.startswith(C.REF_STAGE)]
    j = C.knn(C.devmap_latent(late), C.devmap_latent(early), 1).ravel()
    print(f"snapped {len(early):,} frozen states onto E9.5 cells", flush=True)
    return dict(zip(early, late[j]))


def main():
    d = readout(C.screen(C.TARGET))
    d["cell"] = d.cell.map(to_e95(d.cell[d.frozen].unique())).fillna(d.cell)
    d.to_parquet(C.out("predicted_leaves.parquet"), index=False)
    print(f"{C.MODE}: {len(d):,} leaves over {d.tree.nunique()} trees, "
          f"{int(d.frozen.sum()):,} frozen, {d.cell.nunique():,} distinct cells",
          flush=True)


def leaves():
    """what main() wrote."""
    return pd.read_parquet(C.out("predicted_leaves.parquet"))


if __name__ == "__main__":
    main()
