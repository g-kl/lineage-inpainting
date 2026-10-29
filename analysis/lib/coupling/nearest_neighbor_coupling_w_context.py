"""
couple a lineage tree's internal nodes to observed cells by nearest neighbor.

variant of nearest_neighbor_coupling.py that lets the pooler use
additional context. when assigning nodes at time t_earlier, the pooling
subtree for each target node is rooted at its latest ancestor whose time is
<= t_earlier - sample_time_gap. pooled embeddings of the t_earlier targets
are then extracted from those subtrees and snapped to their nearest observed
cell within the target's own time bin and the immediately adjacent bins. nodes
earlier than min_snap_time are left unassigned. sample_time_gap = 0 roots each
subtree at the target itself.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import anndata as ad
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from datasets.tree_record import collate_subtrees
from analysis.lib.coupling.nearest_neighbor_coupling import (
    UNKNOWN,
    _clip_subtree,
    _nearest_neighbor,
    _pool_call,
    _subsample,
)

logger = logging.getLogger(__name__)


def _ancestor_alive_at(tree: nx.DiGraph, node, t_cutoff: float):
    """latest ancestor of node (including node) whose time is <= t_cutoff."""
    cur = node
    while cur is not None:
        if tree.nodes[cur].get("time", float("nan")) <= t_cutoff:
            return cur
        preds = list(tree.predecessors(cur))
        cur = preds[0] if preds else None
    return None


def _nearest_bin(t: float, bins: list[float]) -> float:
    """observed time bin closest to t."""
    return min(bins, key=lambda b: abs(b - t))


@torch.no_grad()
def couple_tree(
    pool: torch.nn.Module,
    tree: nx.DiGraph,
    states: ad.AnnData,
    *,
    obsm_key: str = "X_scvi",
    time_key: str = "time",
    sample_time_gap: float = 0.0,
    min_snap_time: float = 7.0,
    batch_size: int = 256,
    max_nodes: int = 1024,
    max_obs: int | None = None,
    max_dist: float | None = None,
    pool_n_steps: int = 1,
    use_gpu: bool = True,
    seed: int = 0,
) -> dict[str, dict]:
    """couple assignable internal nodes in the tree to observed cells.

    Each internal node at or after min_snap_time is assigned to the layer of its
    nearest observed time bin, pooled, and snapped to its nearest observed cell
    within that bin and the two adjacent bins. Nodes earlier than min_snap_time
    are left unassigned.

    sample_time_gap: how much earlier than the assigned layer the pooling
    subtree should be rooted. For each target node N at t_earlier, the root
    is the latest ancestor of N whose time is <= t_earlier - sample_time_gap.
    A target with no such ancestor (t_root before the tree root) is skipped.

    max_nodes: cap on nodes per pooled subtree; larger subtrees are subsampled
    by dropping leaves (and any target whose descendant leaves are all dropped
    goes unassigned), so this is set high for coupling.

    returns:
        {"assignments": {node -> obs cell name or UNKNOWN},
         "nn_distances": {node -> L2 distance to nearest obs cell}}.
    """
    pool.eval()
    device = next(pool.parameters()).device
    rng = np.random.default_rng(seed)

    time_bins = sorted(
        np.unique(states.obs[time_key].values.astype(float)).tolist()
    )
    if len(time_bins) < 2:
        logger.warning("need at least 2 time bins, got %d", len(time_bins))
        return {"assignments": {}, "nn_distances": {}}

    tree_min_time = min(
        d.get("time", float("inf")) for _, d in tree.nodes(data=True)
    )

    obs_emb_all = np.asarray(states.obsm[obsm_key], dtype=np.float32)
    obs_name_to_idx = {n: i for i, n in enumerate(states.obs_names.values)}

    current_emb: dict[str, np.ndarray] = {}
    for node in tree.nodes():
        if tree.out_degree(node) == 0 and node in obs_name_to_idx:
            current_emb[node] = obs_emb_all[obs_name_to_idx[node]]
    logger.info("initialized %d leaf embeddings", len(current_emb))

    # observed cells grouped by exact time bin. at each layer a node snaps to
    # its own bin plus the immediately adjacent bins (one slice on each side),
    # and never further across time.
    obs_times = states.obs[time_key].values.astype(float)
    obs_names_all = states.obs_names.values
    obs_by_bin: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for t in time_bins:
        mask = obs_times == t
        obs_by_bin[t] = (obs_emb_all[mask], obs_names_all[mask])

    assignments: dict[str, str] = {}
    nn_distances: dict[str, float] = {}
    totals = {
        "candidates": 0,
        "no_ancestor": 0,
        "incomplete_subtree": 0,
        "snapped": 0,
        "unknown": 0,
    }

    n_bins = len(time_bins)
    for bin_idx in range(n_bins - 1, -1, -1):
        t_earlier = time_bins[bin_idx]
        t_root = t_earlier - sample_time_gap

        if t_root < tree_min_time:
            logger.info(
                "bin %d/%d t_earlier=%.3f t_root=%.3f < tree_min %.3f: skipping bin",
                bin_idx, n_bins - 1, t_earlier, t_root, tree_min_time,
            )
            continue

        # snap to this bin plus one adjacent slice on each side (clamped at ends)
        window_bins = time_bins[max(0, bin_idx - 1) : bin_idx + 2]
        obs_emb_cand = np.concatenate([obs_by_bin[b][0] for b in window_bins])
        obs_names_cand = np.concatenate([obs_by_bin[b][1] for b in window_bins])
        obs_emb_cand, obs_names_cand = _subsample(obs_emb_cand, obs_names_cand, max_obs, rng)
        if len(obs_names_cand) == 0:
            logger.info(
                "bin %d/%d t_earlier=%.3f: no observed cells in window, skipping",
                bin_idx, n_bins - 1, t_earlier,
            )
            continue

        target_nodes = [
            n for n in tree.nodes()
            if tree.out_degree(n) > 0
            and tree.nodes[n].get("time", float("nan")) >= min_snap_time
            and _nearest_bin(tree.nodes[n]["time"], time_bins) == t_earlier
        ]
        if not target_nodes:
            continue
        totals["candidates"] += len(target_nodes)

        ancestor_to_targets: dict[str, list[str]] = defaultdict(list)
        n_no_ancestor = 0
        for n in target_nodes:
            a = _ancestor_alive_at(tree, n, t_root)
            if a is None:
                n_no_ancestor += 1
            else:
                ancestor_to_targets[a].append(n)
        totals["no_ancestor"] += n_no_ancestor

        has_emb = set(current_emb)
        subtree_dicts = []
        n_incomplete = 0
        for ancestor, targets in ancestor_to_targets.items():
            sg, leaves = _clip_subtree(tree, ancestor, has_emb)
            if all(l in current_emb for l in leaves):
                subtree_dicts.append({
                    "graph": sg,
                    "root": ancestor,
                    "root_time": tree.nodes[ancestor].get("time", float("nan")),
                    "leaves": leaves,
                    "_targets": set(targets),
                })
            else:
                n_incomplete += len(targets)
        totals["incomplete_subtree"] += n_incomplete

        n_imputable = sum(len(sd["_targets"]) for sd in subtree_dicts)
        logger.info(
            "bin %d/%d t_earlier=%.3f t_root=%.3f: "
            "candidates=%d no_ancestor=%d incomplete=%d imputable=%d (%d subtrees)",
            bin_idx, n_bins - 1, t_earlier, t_root,
            len(target_nodes), n_no_ancestor, n_incomplete,
            n_imputable, len(subtree_dicts),
        )

        if not subtree_dicts:
            continue

        leaf_tensors = {
            name: torch.as_tensor(emb, dtype=torch.float32)
            for name, emb in current_emb.items()
        }
        node_ids: list[str] = []
        embeddings: list[np.ndarray] = []

        for i in tqdm(range(0, len(subtree_dicts), batch_size), desc="pooling", leave=False):
            chunk = subtree_dicts[i : i + batch_size]
            batch = collate_subtrees(chunk, leaf_tensors, max_nodes=max_nodes)
            batch = batch.to(device)
            pooled = _pool_call(pool, batch, pool_n_steps).cpu()

            for j in range(len(chunk)):
                n_real = batch.n_nodes[j].item()
                names_j = batch.node_names[j][:n_real]
                targets = chunk[j]["_targets"]
                for k, name in enumerate(names_j):
                    if name in targets:
                        node_ids.append(name)
                        embeddings.append(pooled[j, k].numpy())

            del batch, pooled
            torch.cuda.empty_cache()

        if not node_ids:
            continue

        int_emb = np.stack(embeddings)

        nn_idx, nn_dist = _nearest_neighbor(int_emb, obs_emb_cand, use_gpu=use_gpu)
        if not np.isfinite(int_emb).all():
            raise RuntimeError(
                f"pool produced non-finite query embeddings: "
                f"NaN={np.isnan(int_emb).any()} Inf={np.isinf(int_emb).any()}"
            )
        if (nn_idx < 0).any() or (nn_idx >= len(obs_names_cand)).any():
            bad = nn_idx[(nn_idx < 0) | (nn_idx >= len(obs_names_cand))][:5]
            raise RuntimeError(
                f"faiss returned out-of-range indices (e.g. {bad.tolist()}) "
                f"against obs_emb shape {obs_emb_cand.shape}, query shape {int_emb.shape}"
            )

        n_unknown_layer = 0
        for k, node_name in enumerate(node_ids):
            nn_distances[node_name] = float(nn_dist[k])
            if max_dist is not None and nn_dist[k] > max_dist:
                assignments[node_name] = UNKNOWN
                current_emb[node_name] = embeddings[k]
                n_unknown_layer += 1
            else:
                matched_name = obs_names_cand[nn_idx[k]]
                assignments[node_name] = matched_name
                current_emb[node_name] = obs_emb_cand[nn_idx[k]]
        totals["snapped"] += len(node_ids) - n_unknown_layer
        totals["unknown"] += n_unknown_layer
        logger.info(
            "bin %d/%d t_earlier=%.3f: snapped=%d unknown=%d",
            bin_idx, n_bins - 1, t_earlier,
            len(node_ids) - n_unknown_layer, n_unknown_layer,
        )

    logger.info(
        "totals: candidates=%d snapped=%d unknown=%d "
        "no_ancestor=%d incomplete_subtree=%d",
        totals["candidates"], totals["snapped"], totals["unknown"],
        totals["no_ancestor"], totals["incomplete_subtree"],
    )
    return {"assignments": assignments, "nn_distances": nn_distances}


@torch.no_grad()
def couple_all_trees(
    pool: torch.nn.Module,
    trees: dict[str, nx.DiGraph],
    states: ad.AnnData,
    **kwargs,
) -> dict:
    """run couple_tree on every tree and aggregate results."""
    result: dict = {}
    nn_distances: dict[str, dict[str, float]] = {}

    for tree_name, tree in tqdm(trees.items(), desc="coupling trees"):
        out = couple_tree(pool=pool, tree=tree, states=states, **kwargs)
        result[tree_name] = out["assignments"]
        nn_distances[tree_name] = out["nn_distances"]

    result["nn_distances"] = nn_distances
    return result
