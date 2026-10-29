"""Layer-anchored subtree assembly for the crossing-family evals.

Builds per-lineage subtrees from a held-out simulation (root at ``t_src``,
observed leaves at ``t_tgt``), matching the training distribution, and flattens
them into the pair list the inference helpers consume. Dimension-agnostic:
identical for native 1-D and high-dimensional (projected) runs.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from datasets.tree_record import collate_subtrees  # noqa: F401  (re-exported use)
from synthetic_evals.core.hydra_io import DEVICE


def build_layer_subtrees(sim_res, t_src: float, t_tgt: float):
    """Per-lineage subtrees from the held-out sim, spanning t_src -> t_tgt.

    For each cell alive at ``t_src``, walk the simulation tree forward
    collecting all descendants until the lineage reaches a cell alive at
    ``t_tgt``. Returns a list of dicts with keys ``graph``, ``root``,
    ``leaves`` (list of (subtree_leaf_name, original_sim_node)), ``x_src``,
    ``leaf_states`` ({leaf_name: tensor}), and ``parent`` (original sim node).

    Node ``time`` attributes are overridden so the root has time ``t_src`` and
    leaves have time ``t_tgt``, matching what the model was trained on.
    """
    alive_src = sim_res._alive_at(t_src)
    alive_tgt = set(sim_res._alive_at(t_tgt))
    subs = []
    for parent in alive_src:
        g = nx.DiGraph()
        root_name = f"sub{len(subs)}_root"
        g.add_node(root_name, time=float(t_src))
        node_to_name = {parent: root_name}
        leaves: list[tuple[str, object]] = []
        leaf_states: dict[str, torch.Tensor] = {}
        frontier = [parent]
        while frontier:
            nxt = []
            for n in frontier:
                if n in alive_tgt and n != parent:
                    leaf_name = node_to_name[n]
                    leaves.append((leaf_name, n))
                    x_c = sim_res._interp_state(n, float(t_tgt))
                    leaf_states[leaf_name] = torch.as_tensor(
                        x_c, dtype=torch.float32,
                    )
                    continue
                for c in sim_res.tree.successors(n):
                    if c in alive_tgt:
                        t_c = float(t_tgt)
                    else:
                        t_c = float(sim_res.tree.nodes[c].get("time", t_tgt))
                    c_name = f"sub{len(subs)}_n{len(node_to_name)}"
                    g.add_node(c_name, time=t_c)
                    g.add_edge(node_to_name[n], c_name)
                    node_to_name[c] = c_name
                    nxt.append(c)
            frontier = nxt
        if not leaves:
            continue
        x_src = np.asarray(
            sim_res._interp_state(parent, float(t_src)), dtype=np.float32,
        )
        if x_src.ndim == 0:
            x_src = x_src[None]
        subs.append({
            "graph": g,
            "root": root_name,
            "leaves": leaves,
            "x_src": x_src,
            "leaf_states": leaf_states,
            "parent": parent,
        })
    return subs


def subtrees_to_pairs(subtrees, t_src: float, t_tgt: float):
    """Flatten subtrees into (parent, child, t_src, t_tgt, x_src, x_tgt) tuples.

    Iteration order matches the splitter/pooler inference outputs.
    """
    pairs = []
    for s in subtrees:
        x_src = s["x_src"]
        for leaf_name, orig in s["leaves"]:
            ls = s["leaf_states"][leaf_name]
            x_tgt = ls.cpu().numpy() if hasattr(ls, "cpu") else np.asarray(ls)
            x_tgt = np.asarray(x_tgt, dtype=np.float32)
            if x_tgt.ndim == 0:
                x_tgt = x_tgt[None]
            pairs.append(
                (s["parent"], orig, float(t_src), float(t_tgt), x_src, x_tgt),
            )
    return pairs


def subtree_dicts_and_leaves(subtrees, leaf_state_fn):
    """Build the (subtree_dicts, leaf_states) inputs for ``collate_subtrees``.

    ``leaf_state_fn(s, leaf_name)`` returns the tensor for the given leaf --
    zeros for splitter inference, the true x_tgt for pooler inference.
    """
    subtree_dicts = []
    leaf_states: dict[str, torch.Tensor] = {}
    for s in subtrees:
        subtree_dicts.append({
            "graph": s["graph"],
            "root": s["root"],
            "leaves": [name for name, _ in s["leaves"]],
        })
        for name, _ in s["leaves"]:
            leaf_states[name] = leaf_state_fn(s, name)
    return subtree_dicts, leaf_states


def leaf_positions(batch, subtrees):
    """(tree_idx, node_idx) of every (subtree, leaf) in iteration order.

    Uses ``batch.node_names`` so it doesn't depend on collation order.
    """
    positions = []
    for i, s in enumerate(subtrees):
        names = batch.node_names[i]
        idx_of = {n: j for j, n in enumerate(names)}
        for name, _ in s["leaves"]:
            positions.append((i, idx_of[name]))
    return positions


def collect_trajectories(sim_res):
    """Every cell trajectory + its leaf-family mu, for the plotter.

    Returns a list of (traj_array, leaf_mu) with traj_array shape (steps, 1+dim).
    """
    leaf_mu_cache: dict = {}

    def leaf_mu(node):
        if node in leaf_mu_cache:
            return leaf_mu_cache[node]
        cur = node
        while sim_res.tree.out_degree(cur) > 0:
            cur = next(iter(sim_res.tree.successors(cur)))
        mu = sim_res.tree.nodes[cur]["params"].get("mu", 0.0)
        leaf_mu_cache[node] = mu
        return mu

    trajs = []
    for n in sim_res.tree.nodes():
        t = sim_res.tree.nodes[n].get("trajectory")
        if t is None:
            continue
        trajs.append((np.asarray(t), float(leaf_mu(n))))
    return trajs
