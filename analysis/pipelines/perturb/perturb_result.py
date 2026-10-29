"""output data structure for a perturbation run."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx


@dataclass
class PerturbResult:
    """terminal perturbed leaf states.

    ``meta`` : one row per terminal leaf, columns
      lineage_id, node_name, snapped_name, time, parent_lineage, changed,
      proj_dist, local_dist, nn_dist, off_manifold, reason.
    ``states`` : (n_leaves, dim) array row-aligned to ``meta``.
    ``tree`` : the full propagated lineage tree, keyed by ``node_name`` so its
      leaves line up with ``meta['node_name']``.  every node carries ``time``,
      ``snapped_name`` and the projection fields (``changed``, ``proj_dist``,
      ``local_dist``, ``off_manifold``); terminal leaves also carry ``reason``.
    """

    meta: pd.DataFrame
    states: np.ndarray
    tree: nx.DiGraph = None

    def on_manifold(self):
        m = ~self.meta["off_manifold"].values
        return self.meta[m].reset_index(drop=True), self.states[m]

    def off_manifold(self):
        m = self.meta["off_manifold"].values
        return self.meta[m].reset_index(drop=True), self.states[m]


def is_off(changed, proj_dist, local_dist, ratio):
    """off-manifold label: changed and projected farther than ratio x local spacing."""
    return bool(changed and local_dist is not None and not np.isnan(local_dist)
                and proj_dist > ratio * local_dist)


def off_manifold_descendants(tree, ratio):
    """nodes off-manifold at ``ratio`` plus everything descended from one.

    a node that changed and projected too far would not have propagated in a real
    run, so its whole subtree is invalid.  sweep ``ratio`` after the run and
    filter with ``set(tree) - off_manifold_descendants(tree, ratio)``.
    """
    bad = set()
    for n in nx.topological_sort(tree):
        if n in bad:                                  # already under an off-manifold node
            continue
        d = tree.nodes[n]
        if is_off(d.get("changed"), d.get("proj_dist"), d.get("local_dist"), ratio):
            bad.add(n)
            bad.update(nx.descendants(tree, n))       # only the topmost off node traverses
    return bad
