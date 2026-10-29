from __future__ import annotations

import networkx as nx
import numpy as np
import ot as pot
import pandas as pd
import torch
from tqdm.auto import tqdm

from datasets.tree_record import collate_subtrees, _extract_subtree, _leaf_counts, _node_depths


def _ancestor_at_time(tree, leaf, target_time, tol=0.5):
    """Walk from leaf toward root; return the node closest to target_time."""
    cur = leaf
    best, best_dist = None, float("inf")
    while cur is not None:
        t = tree.nodes[cur].get("time", float("nan"))
        d = abs(t - target_time)
        if d < best_dist:
            best, best_dist = cur, d
        preds = list(tree.predecessors(cur))
        cur = preds[0] if preds else None
    if best_dist > tol:
        return None
    return best


def _descendant_leaves(tree, node):
    if tree.out_degree(node) == 0:
        return [node]
    leaves, stack = [], [node]
    while stack:
        n = stack.pop()
        children = list(tree.successors(n))
        if not children:
            leaves.append(n)
        else:
            stack.extend(children)
    return leaves


def _wasserstein2(x, y):
    a = np.ones(len(x)) / len(x)
    b = np.ones(len(y)) / len(y)
    M = pot.dist(x, y, metric="sqeuclidean")
    return float(np.sqrt(pot.emd2(a, b, M)))


def _get_time_points(trees, tol=0.25):
    """get leaf times"""
    times = set()
    for tree in trees.values():
        for n in tree.nodes():
            if tree.out_degree(n) != 0:
                continue
            t = tree.nodes[n].get("time", float("nan"))
            if np.isfinite(t):
                times.add(round(t / tol) * tol)
    return sorted(times)


def _cells_at_time(trees, leaf_states, target_time, tol=0.05):
    """(names, states) of observed cells at target_time"""
    names, states = [], []
    for tree in trees.values():
        for node in tree.nodes():
            if tree.out_degree(node) != 0:
                continue
            if node not in leaf_states:
                continue
            t = tree.nodes[node].get("time", float("nan"))
            if abs(t - target_time) <= tol:
                names.append(node)
                states.append(leaf_states[node])
    return names, np.stack(states) if states else np.empty((0, 0))


def _pool_reverse(pool, trees, src_names, t_tgt, device, leaf_states,
                  time_tol=0.05, batch_size=256):
    """Pool source leaves back to ancestor states within time_tol of t_tgt."""
    node_to_tree = {}
    for tree_name, tree in trees.items():
        for node in tree.nodes():
            node_to_tree[node] = (tree_name, tree)

    subtree_dicts = []
    cache = {}
    for name in src_names:
        if name not in node_to_tree:
            continue
        tree_name, tree = node_to_tree[name]
        anc = _ancestor_at_time(tree, name, t_tgt, tol=time_tol)
        if anc is None or anc == name:
            continue
        desc_leaves = _descendant_leaves(tree, anc)
        if not all(l in leaf_states for l in desc_leaves):
            continue
        if tree_name not in cache:
            cache[tree_name] = (_leaf_counts(tree), _node_depths(tree))
        sizes, depths = cache[tree_name]
        subtree_dicts.append(_extract_subtree(tree, anc, sizes, depths, tree_name))

    if not subtree_dicts:
        return np.empty((0, 0))

    out = []
    for start in range(0, len(subtree_dicts), batch_size):
        end = min(start + batch_size, len(subtree_dicts))
        batch = collate_subtrees(subtree_dicts[start:end], leaf_states, device=device).to(device)
        pooled = pool.pool_batch(batch)
        B = end - start
        roots = pooled[torch.arange(B, device=device), batch.root_idx]
        out.append(roots.detach().cpu().numpy())

    return np.concatenate(out, axis=0)


def _subsample(rng, names, states, n):
    if len(names) <= n:
        return names, states
    idx = rng.choice(len(names), size=n, replace=False)
    return [names[j] for j in idx], states[idx]


def _subsample_array(rng, states, n):
    if len(states) <= n:
        return states
    idx = rng.choice(len(states), size=n, replace=False)
    return states[idx]


@torch.no_grad()
def marginal_preservation(
    pool,
    trees,
    leaf_states,
    *,
    n_cells=200,
    time_tol=0.25,
    batch_size=256,
    seed=212121,
    device=None,
):
    """W2 between pooled-back distribution and true marginal at each earlier time-point."""
    if device is None:
        device = next(pool.parameters()).device
    pool.eval()
    rng = np.random.default_rng(seed)

    leaf_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in leaf_states.items()}

    time_points = _get_time_points(trees, tol=time_tol)
    if len(time_points) < 2:
        return pd.DataFrame(columns=["t_src", "t_tgt", "n_src", "n_tgt", "w2"])

    rows = []
    for i in tqdm(range(len(time_points) - 1), desc="marginal preservation"):
        t_early, t_late = time_points[i], time_points[i + 1]

        names_early, states_early = _cells_at_time(trees, leaf_np, t_early, tol=time_tol)
        names_late, states_late = _cells_at_time(trees, leaf_np, t_late, tol=time_tol)
        if len(names_early) < 2 or len(names_late) < 2:
            continue

        src_names, _ = _subsample(rng, names_late, states_late, n_cells)
        tgt_states = _subsample_array(rng, states_early, n_cells)

        transported = _pool_reverse(
            pool, trees, src_names, t_early,
            torch.device(device), leaf_states=leaf_states,
            time_tol=time_tol, batch_size=batch_size,
        )
        if len(transported) < 2:
            continue

        rows.append({
            "t_src": t_late, "t_tgt": t_early,
            "n_src": len(src_names), "n_tgt": len(tgt_states),
            "w2": _wasserstein2(transported, tgt_states),
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def interpolant_matching(
    pool,
    trees,
    leaf_states,
    interpolant_time,
    ground_truth_interpolant_states,
    *,
    n_cells=200,
    time_tol=0.25,
    batch_size=256,
    seed=212121,
    device=None,
):
    """W2 between pooled-back distribution and ground-truth states at interpolant_time.

    Only time-points later than interpolant_time are evaluated
    """
    if device is None:
        device = next(pool.parameters()).device
    pool.eval()
    rng = np.random.default_rng(seed)

    leaf_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in leaf_states.items()}

    time_points = _get_time_points(trees, tol=time_tol)
    gt = np.asarray(ground_truth_interpolant_states)
    rows = []

    for t in tqdm(time_points, desc="interpolant matching"):
        if t <= interpolant_time + time_tol:
            continue

        names_t, states_t = _cells_at_time(trees, leaf_np, t, tol=time_tol)
        if len(names_t) < 2 or len(gt) < 2:
            continue

        src_names, _ = _subsample(rng, names_t, states_t, n_cells)
        tgt_states = _subsample_array(rng, gt, n_cells)

        transported = _pool_reverse(
            pool, trees, src_names, interpolant_time,
            torch.device(device), leaf_states=leaf_states,
            time_tol=time_tol, batch_size=batch_size,
        )
        if len(transported) < 2:
            continue

        rows.append({
            "t_src": t, "t_tgt": interpolant_time,
            "n_src": len(src_names), "n_tgt": len(tgt_states),
            "w2": _wasserstein2(transported, tgt_states),
        })

    return pd.DataFrame(rows)