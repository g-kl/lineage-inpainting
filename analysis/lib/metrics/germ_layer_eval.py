import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, deque


def assignment_germ_layer_transitions(trees, assignments, name_to_gl):
    """Germ-layer transition frequencies from assignment-based condensed trees.

    For each tree, edges are condensed so that only pairs of consecutively
    assigned nodes are considered (unassigned intermediates are skipped).
    For every such condensed edge the source and target germ-layer labels
    are looked up and the transition is tallied.

    Parameters
    ----------
    trees : dict[str, nx.DiGraph]
        ``{tree_name: tree}`` with node attribute ``"time"``.
    assignments : dict[str, dict[str, str]]
        ``{tree_name: {node: obs_name}}`` mapping tree nodes to observed
        cell names.
    name_to_gl : dict[str, str]
        ``{obs_name: germ_layer}`` mapping observed cell names to germ-layer
        labels.

    Returns
    -------
    pd.DataFrame
        Square matrix of transition frequencies (rows = source germ layer,
        columns = target germ layer).  Values sum to 1.
    """
    germ_layers = sorted(set(name_to_gl.values()))
    gl_to_idx = {gl: i for i, gl in enumerate(germ_layers)}
    n = len(germ_layers)
    mat = np.zeros((n, n))

    for tree_name, tree in trees.items():
        if tree_name not in assignments:
            continue
        node_assignments = assignments[tree_name]

        # assigned nodes with known germ layers
        assigned_gl = {}
        for node, obs_name in node_assignments.items():
            if node in tree.nodes and obs_name in name_to_gl:
                assigned_gl[node] = name_to_gl[obs_name]

        # For each assigned node, BFS to nearest assigned descendants
        for node in assigned_gl:
            queue = deque(tree.successors(node))
            while queue:
                child = queue.popleft()
                if child in assigned_gl:
                    # condensed edge: node -> child
                    mat[gl_to_idx[assigned_gl[node]],
                        gl_to_idx[assigned_gl[child]]] += 1
                else:
                    # skip unassigned, keep searching
                    queue.extend(tree.successors(child))

    total = mat.sum()
    if total > 0:
        mat /= total

    return pd.DataFrame(mat, index=germ_layers, columns=germ_layers)


def assignment_germ_layer_crossing_by_bin(trees, assignments, name_to_gl, time_bins,
                                          exclude_source="epiblast"):
    """Germ-layer crossing fraction, overall and per source-node time bin.

    A condensed edge connects an assigned node to its nearest assigned
    descendant (unassigned intermediates skipped); it "crosses" when the source
    and target germ layers differ. Edges whose source germ layer is
    ``exclude_source`` are dropped (epiblast is pluripotent, so its outgoing
    transitions are not errors). Each edge is bucketed by the source node's
    nearest observed time bin.

    Returns ``{"all": fraction, bin_time: fraction, ...}``.
    """
    crossings = defaultdict(int)
    totals = defaultdict(int)

    for tree_name, tree in trees.items():
        if tree_name not in assignments:
            continue
        assigned_gl = {
            node: name_to_gl[obs]
            for node, obs in assignments[tree_name].items()
            if node in tree.nodes and obs in name_to_gl
        }
        for node, src_gl in assigned_gl.items():
            if src_gl == exclude_source:
                continue
            bin_t = min(time_bins, key=lambda b: abs(b - tree.nodes[node]["time"]))
            queue = deque(tree.successors(node))
            while queue:
                child = queue.popleft()
                if child in assigned_gl:
                    crossed = int(assigned_gl[child] != src_gl)
                    crossings["all"] += crossed
                    totals["all"] += 1
                    crossings[bin_t] += crossed
                    totals[bin_t] += 1
                else:
                    queue.extend(tree.successors(child))

    return {b: crossings[b] / totals[b] for b in totals}


def ancestor_map_germ_layer_transitions(ancestor_map, name_to_gl):
    """Germ-layer transition frequencies from an ancestor map.

    For each leaf the ancestor-map provides inferred ancestors at various
    timepoints.  Consecutive timepoint pairs along each trajectory are
    scored for germ-layer consistency.

    Parameters
    ----------
    ancestor_map : dict[str, dict[float, str]]
        ``{leaf_node: {time: ancestor_obs_name}}``.
    name_to_gl : dict[str, str]
        ``{obs_name: germ_layer}``.

    Returns
    -------
    pd.DataFrame
        Square matrix of transition frequencies (rows = earlier timepoint
        germ layer, columns = later timepoint germ layer).  Values sum to 1.
    """
    germ_layers = sorted(set(name_to_gl.values()))
    gl_to_idx = {gl: i for i, gl in enumerate(germ_layers)}
    n = len(germ_layers)
    mat = np.zeros((n, n))

    for leaf, time_to_ancestor in ancestor_map.items():
        sorted_times = sorted(time_to_ancestor.keys())
        for t_earlier, t_later in zip(sorted_times[:-1], sorted_times[1:]):
            anc_earlier = time_to_ancestor[t_earlier]
            anc_later = time_to_ancestor[t_later]
            gl_earlier = name_to_gl.get(anc_earlier)
            gl_later = name_to_gl.get(anc_later)
            if gl_earlier is not None and gl_later is not None:
                mat[gl_to_idx[gl_earlier], gl_to_idx[gl_later]] += 1

    total = mat.sum()
    if total > 0:
        mat /= total

    return pd.DataFrame(mat, index=germ_layers, columns=germ_layers)
