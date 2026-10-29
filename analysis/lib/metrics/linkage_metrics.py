import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm.auto import tqdm
import seaborn as sns
import pandas as pd
import pickle
import re
import functools
import operator


def get_leaves(tree):
    return [n for n in tree.nodes() if tree.out_degree(n) == 0]


def get_parent(tree, node):
    preds = list(tree.predecessors(node))
    return preds[0] if preds else None


def parse_tree_name(tname):
    """parse E[time]-R[i]-C[j] format, return time as float"""
    m = re.match(r'E([\d.]+)-', str(tname))
    return float(m.group(1)) if m else None 
 
def parse_replicate(tname):
    """parse E[time]-R[i]-C[j] format, return 'E[time]-R[i]' string"""
    m = re.match(r'(E[\d.]+-R\d+)', str(tname))
    return m.group(1) if m else None
 
def filter_types(name_to_type, types, timepoint, states, min_count=500):
    """keep only types with >= min_count obs at timepoint"""
    time_mask = states.obs['time'] == timepoint
    time_cells = states.obs_names[time_mask]
    counts = defaultdict(int)
    for c in time_cells:
        ct = name_to_type.get(c)
        if ct is not None:
            counts[ct] += 1
    return [t for t in types if counts[t] >= min_count]


def ancestral_linkage(trees, name_to_cell_type, types=None, n_null=10000, seed=42):
    """Compute ancestral linkage distances for all cell-type pairs.

    For each leaf, walks up the tree recording the distance to the first
    ancestor whose subtree contains each other cell type.

    trees: iterable of nx.DiGraph trees
    name_to_cell_type: dict mapping leaf node -> cell type
    types: list of cell types to consider (default: all in name_to_cell_type)
    """
    if types is None:
        types = sorted(set(name_to_cell_type.values()))
    type_to_bit = {t: 1 << i for i, t in enumerate(types)}
    all_bits = (1 << len(types)) - 1

    linkage = defaultdict(lambda: defaultdict(list))
    null_dists = defaultdict(list)

    for tree in trees:
        root = [n for n, d in tree.in_degree() if d == 0][0]
        time = dict(tree.nodes(data="time"))
        parent = {}
        order = [root]
        for u, v in nx.bfs_edges(tree, root):
            parent[v] = u
            order.append(v)

        subtree = {}
        for node in reversed(order):
            children = list(tree.successors(node))
            subtree[node] = type_to_bit.get(name_to_cell_type.get(node), 0) if not children else \
                            functools.reduce(operator.or_, (subtree[c] for c in children), 0)

        for leaf in order:
            if tree.out_degree(leaf) > 0:
                continue
            ct = name_to_cell_type.get(leaf)
            if ct is None or ct not in type_to_bit:
                continue
            seen, cur = subtree[leaf], leaf
            while cur in parent and seen != all_bits:
                p = parent[cur]
                new = subtree[p] & ~seen
                d = time[leaf] - time[p]
                tmp = new
                while tmp:
                    i = (tmp & -tmp).bit_length() - 1
                    linkage[ct][types[i]].append((leaf, d))
                    null_dists[types[i]].append(d)
                    tmp &= tmp - 1
                seen |= subtree[p]
                cur = p

    rng = np.random.default_rng(seed)
    null = {}
    for t in types:
        vals = null_dists[t]
        if len(vals) <= n_null:
            null[t] = np.array(vals)
        else:
            idx = rng.choice(len(vals), n_null, replace=False)
            null[t] = np.array([vals[i] for i in idx])

    return linkage, null, types


def linkage_heatmap(linkage, null, types):
    """Convert ancestral linkage output to a null-normalized heatmap DataFrame."""
    n = len(types)
    mat = np.full((n, n), np.nan)
    for i, t1 in enumerate(types):
        for j, t2 in enumerate(types):
            if linkage[t1][t2]:
                mat[i, j] = np.mean([x[1] for x in linkage[t1][t2]])
                if null[t2].size > 0:
                    mat[i, j] -= np.mean(null[t2])
    return pd.DataFrame(mat, index=types, columns=types)


def _heatmap_corr(h1, h2):
    """Upper-tri Pearson r between two heatmaps."""
    idx = np.triu_indices(len(h1), k=1)
    x, y = h1.values[idx], h2.values[idx]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    r, _ = pearsonr(x, y)
    return r


# --- (1) real linkage ---

def real_linkage(trees, timepoint, name_to_type, types):
    """Linkage on real trees, considering only leaves at the given timepoint."""
    filtered = {}
    tree_list = []
    for tname, tree in trees.items():
        has_leaves = False
        for n in get_leaves(tree):
            if tree.nodes[n].get('time') == timepoint and n in name_to_type:
                filtered[n] = name_to_type[n]
                has_leaves = True
        if has_leaves:
            tree_list.append(tree)
    lnk, null, types_out = ancestral_linkage(tree_list, filtered, types)
    return linkage_heatmap(lnk, null, types_out)


def real_linkage_per_tree(trees, timepoint, name_to_type, types):
    """Linkage per tree separately, returns {tree_name: heatmap}."""
    result = {}
    for tname, tree in trees.items():
        filtered = {}
        for n in get_leaves(tree):
            if tree.nodes[n].get('time') == timepoint and n in name_to_type:
                filtered[n] = name_to_type[n]
        if not filtered:
            continue
        lnk, null, types_out = ancestral_linkage([tree], filtered, types)
        result[tname] = linkage_heatmap(lnk, null, types_out)
    return result


def real_linkage_per_replicate(trees, timepoint, name_to_type, types):
    """Linkage per replicate (pooling clones), returns {replicate: heatmap}."""
    groups = defaultdict(lambda: ([], {}))
    for tname, tree in trees.items():
        rep = parse_replicate(tname)
        if rep is None:
            continue
        has_leaves = False
        for n in get_leaves(tree):
            if tree.nodes[n].get('time') == timepoint and n in name_to_type:
                groups[rep][1][n] = name_to_type[n]
                has_leaves = True
        if has_leaves:
            groups[rep][0].append(tree)
    result = {}
    for rep, (tree_list, filtered) in groups.items():
        if not filtered:
            continue
        lnk, null, types_out = ancestral_linkage(tree_list, filtered, types)
        result[rep] = linkage_heatmap(lnk, null, types_out)
    return result

# --- (2) assignment-based linkage ---

def assignment_linkage(trees, assignments, source_time, target_time,
                       name_to_type, types):
    """Cut each tree at target_time, map cut leaves to obs via assignments."""
    tree_list = []
    filtered = {}
    for tname, tree in trees.items():
        t_parsed = parse_tree_name(tname)
        if t_parsed is None or t_parsed != source_time:
            continue
        if tname not in assignments:
            continue

        cut_nodes = [n for n in tree.nodes() if tree.nodes[n]['time'] <= target_time]
        cut_tree = tree.subgraph(cut_nodes).copy()
        cut_leaves = get_leaves(cut_tree)

        has_leaves = False
        for leaf in cut_leaves:
            if leaf in assignments[tname]:
                obs = assignments[tname][leaf]
                if obs in name_to_type:
                    filtered[leaf] = name_to_type[obs]
                    has_leaves = True
        if has_leaves:
            tree_list.append(cut_tree)

    lnk, null, types_out = ancestral_linkage(tree_list, filtered, types)
    return linkage_heatmap(lnk, null, types_out)


def assignment_linkage_per_tree(trees, assignments, source_time, target_time,
                                name_to_type, types):
    """Assignment linkage per tree separately, returns {tree_name: heatmap}."""
    result = {}
    for tname, tree in trees.items():
        t_parsed = parse_tree_name(tname)
        if t_parsed is None or t_parsed != source_time:
            continue
        if tname not in assignments:
            continue

        cut_nodes = [n for n in tree.nodes() if tree.nodes[n]['time'] <= target_time]
        cut_tree = tree.subgraph(cut_nodes).copy()
        cut_leaves = get_leaves(cut_tree)

        filtered = {}
        for leaf in cut_leaves:
            if leaf in assignments[tname]:
                obs = assignments[tname][leaf]
                if obs in name_to_type:
                    filtered[leaf] = name_to_type[obs]
        if not filtered:
            continue
        lnk, null, types_out = ancestral_linkage([cut_tree], filtered, types)
        result[tname] = linkage_heatmap(lnk, null, types_out)
    return result


def assignment_linkage_per_replicate(trees, assignments, source_time, target_time,
                                     name_to_type, types):
    """Assignment linkage per replicate (pooling clones), returns {replicate: heatmap}."""
    groups = defaultdict(lambda: ([], {}))
    for tname, tree in trees.items():
        t_parsed = parse_tree_name(tname)
        if t_parsed is None or t_parsed != source_time:
            continue
        if tname not in assignments:
            continue
        rep = parse_replicate(tname)
        if rep is None:
            continue
 
        cut_nodes = [n for n in tree.nodes() if tree.nodes[n]['time'] <= target_time]
        cut_tree = tree.subgraph(cut_nodes).copy()
 
        has_leaves = False
        for leaf in get_leaves(cut_tree):
            if leaf in assignments[tname]:
                obs = assignments[tname][leaf]
                if obs in name_to_type:
                    groups[rep][1][leaf] = name_to_type[obs]
                    has_leaves = True
        if has_leaves:
            groups[rep][0].append(cut_tree)
 
    result = {}
    for rep, (tree_list, filtered) in groups.items():
        if not filtered:
            continue
        lnk, null, types_out = ancestral_linkage(tree_list, filtered, types)
        result[rep] = linkage_heatmap(lnk, null, types_out)
    return result



# --- (3) ancestor map linkage ---

def ancestor_map_linkage(trees, ancestor_map, source_time, target_time,
                         name_to_type, types):
    """Linkage among source_time leaves, relabeled by inferred ancestor at target_time."""
    tree_list = []
    filtered = {}
    for tname, tree in trees.items():
        has_leaves = False
        for n in get_leaves(tree):
            if tree.nodes[n].get('time') != source_time:
                continue
            if n in ancestor_map and target_time in ancestor_map[n]:
                anc = ancestor_map[n][target_time]
                if anc in name_to_type:
                    filtered[n] = name_to_type[anc]
                    has_leaves = True
        if has_leaves:
            tree_list.append(tree)
    lnk, null, types_out = ancestral_linkage(tree_list, filtered, types)
    return linkage_heatmap(lnk, null, types_out)


def ancestor_map_linkage_per_tree(trees, ancestor_map, source_time, target_time,
                                  name_to_type, types):
    """Ancestor map linkage per tree separately, returns {tree_name: heatmap}."""
    result = {}
    for tname, tree in trees.items():
        filtered = {}
        for n in get_leaves(tree):
            if tree.nodes[n].get('time') != source_time:
                continue
            if n in ancestor_map and target_time in ancestor_map[n]:
                anc = ancestor_map[n][target_time]
                if anc in name_to_type:
                    filtered[n] = name_to_type[anc]
        if not filtered:
            continue
        lnk, null, types_out = ancestral_linkage([tree], filtered, types)
        result[tname] = linkage_heatmap(lnk, null, types_out)
    return result

def ancestor_map_linkage_per_replicate(trees, ancestor_map, source_time, target_time,
                                       name_to_type, types):
    """Ancestor map linkage per replicate (pooling clones), returns {replicate: heatmap}."""
    groups = defaultdict(lambda: ([], {}))
    for tname, tree in trees.items():
        rep = parse_replicate(tname)
        if rep is None:
            continue
        has_leaves = False
        for n in get_leaves(tree):
            if tree.nodes[n].get('time') != source_time:
                continue
            if n in ancestor_map and target_time in ancestor_map[n]:
                anc = ancestor_map[n][target_time]
                if anc in name_to_type:
                    groups[rep][1][n] = name_to_type[anc]
                    has_leaves = True
        if has_leaves:
            groups[rep][0].append(tree)
 
    result = {}
    for rep, (tree_list, filtered) in groups.items():
        if not filtered:
            continue
        lnk, null, types_out = ancestral_linkage(tree_list, filtered, types)
        result[rep] = linkage_heatmap(lnk, null, types_out)
    return result

# --- comparison ---

def _pairwise_corrs(heatmaps):
    """Average correlation between all distinct pairs from one set."""
    keys = list(heatmaps.keys())
    rs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = _heatmap_corr(heatmaps[keys[i]], heatmaps[keys[j]])
            if np.isfinite(r):
                rs.append(r)
    return rs


def _cross_corrs(heatmaps_a, heatmaps_b):
    """All correlations between pairs drawn from two sets."""
    rs = []
    for i, ha in enumerate(heatmaps_a.values()):
        for j, hb in enumerate(heatmaps_b.values()):
            # inferred and real heatmaps are distinct replicate sets, so every
            # cross pair counts
            r = _heatmap_corr(ha, hb)
            if np.isfinite(r):
                rs.append(r)
    return rs


def compare_linkage(real, inferred, label='inferred'):
    """real and inferred are DataFrames of null-normalized ancestral linkage."""
    types = real.index.tolist()
    idx = np.triu_indices(len(types), k=1)
    x, y = real.values[idx], inferred.values[idx]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    r, p = pearsonr(x, y) if len(x) > 2 else (0, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=10, alpha=0.5)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3)
    ax.set_xlabel('real ancestral linkage')
    ax.set_ylabel(f'{label} ancestral linkage')
    ax.set_title(f'r={r:.3f}, p={p:.2e}')
    plt.tight_layout()
    return fig, r


def compare_linkage_normalized(real_per_tree, inferred_per_tree, label='inferred'):
    """Correlation of inferred vs real, normalized by real-real baseline.

    real_per_tree, inferred_per_tree: dicts of {tree_name: heatmap}
    Returns normalized_r, raw_r, baseline_r
    """
    rr = _pairwise_corrs(real_per_tree)
    ir = _cross_corrs(inferred_per_tree, real_per_tree)
    baseline = np.mean(rr) if rr else np.nan
    raw = np.mean(ir) if ir else np.nan
    normed = raw / baseline if baseline != 0 and np.isfinite(baseline) else np.nan


    return normed, raw, baseline


# --- lineage oscillation ---

def lineage_oscillation_fraction(trees, assignments, name_to_lineage):
    """Fraction of consecutive labelled-node triples that are an A,B,A bounce.

    Each tree is walked once from the root, tracking the last two labelled
    lineage labels on the path (leaves use their own observed label; internal
    nodes use the label of their assigned cell; unlabelled nodes, e.g. those
    before the snap cutoff, are skipped). Every labelled node with two labelled
    ancestors closes one triple (grandparent label, parent label, this label)
    and is counted exactly once. The triple is an A,B,A bounce when the
    grandparent label equals this label but the parent label differs. Triples
    are NOT collapsed over runs, so the denominator (total triples) depends only
    on tree structure and which nodes are labelled -- not on the assigned labels.

    trees: {tree_name: nx.DiGraph}
    assignments: {tree_name: {node: obs_cell_name}} (couple_all_trees output)
    name_to_lineage: {obs_cell_name: lineage_label}
    Returns A,B,A triples / total triples (nan if no node closes a triple).
    """
    n_alternating = 0
    n_triples = 0
    for tname, tree in trees.items():
        assign = assignments.get(tname, {})
        roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
        stack = [(r, ()) for r in roots]  # (node, last two labelled labels above)
        while stack:
            node, seq = stack.pop()
            if tree.out_degree(node) == 0:
                label = name_to_lineage.get(node)
            else:
                obs = assign.get(node)
                label = name_to_lineage.get(obs) if obs is not None else None
            if label is not None:
                if len(seq) == 2:
                    n_triples += 1
                    if seq[0] == label != seq[1]:
                        n_alternating += 1
                seq = (seq + (label,))[-2:]
            for child in tree.successors(node):
                stack.append((child, seq))
    return n_alternating / n_triples if n_triples else float('nan')


def compare_linkage_wmse(real, inferred, name_to_type, label='inferred'):
    """Weighted MSE between real and inferred linkage heatmaps.

    Weight for pair (i, j) = count_i * count_j, so abundant type pairs
    dominate and rare types with few cells are down-weighted.
    """
    types = real.index.tolist()
    counts = defaultdict(int)
    for ct in name_to_type.values():
        if ct in set(types):
            counts[ct] += 1
    count_arr = np.array([counts[t] for t in types], dtype=float)

    idx = np.triu_indices(len(types), k=1)
    x, y = real.values[idx], inferred.values[idx]
    w = (count_arr[:, None] * count_arr[None, :])[idx]

    mask = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[mask], y[mask], w[mask]

    wmse = np.sum(w * (x - y) ** 2) / np.sum(w) if np.sum(w) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=10, alpha=0.5, c=w, cmap='viridis')
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3)
    ax.set_xlabel('real ancestral linkage')
    ax.set_ylabel(f'{label} ancestral linkage')
    ax.set_title(f'wMSE={wmse:.4f}')
    plt.tight_layout()
    return fig, wmse