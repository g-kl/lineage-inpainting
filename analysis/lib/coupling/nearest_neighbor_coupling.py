"""
couple a lineage tree's internal nodes to observed cells by nearest neighbor.

works layer-by-layer from leaves toward the root:
  1. initialize leaf embeddings from observed data.
  2. for each time layer (latest to earliest):
     a. find internal nodes at that time whose descendant leaves all have embeddings.
     b. pool each subtree bottom-up to get a root embedding.
     c. find the nearest observed cell (by L2) at that time.
     d. if the distance is within max_dist, snap: replace the root embedding
        with the matched cell's embedding so it can serve as a leaf for the
        next (earlier) layer.  otherwise mark as "unknown".
"""

from __future__ import annotations

import inspect
import logging

import anndata as ad
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from datasets.tree_record import collate_subtrees


def _pool_call(pool, batch, n_steps: int):
    """Call ``pool.pool_batch`` with ``n_steps`` if the signature accepts it,
    otherwise call without (for pools like TreeFormer, whose pool_batch takes
    no n_steps).
    """
    if "n_steps" in inspect.signature(pool.pool_batch).parameters:
        return pool.pool_batch(batch, n_steps=n_steps)
    return pool.pool_batch(batch)

logger = logging.getLogger(__name__)

# verbose logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

UNKNOWN = "unknown"

def _nearest_neighbor(
    query: np.ndarray,
    targets: np.ndarray,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """For each row in query, return (index, L2 distance) of the nearest row in targets.

    Always validates the returned indices against ``len(targets)``; if the
    GPU path produced garbage we re-run on CPU. Default is CPU because at the
    embedding widths we use here (50-dim scVI / ~1.7M cells) brute-force CPU
    is well under a second.
    """
    import faiss

    query = np.ascontiguousarray(query, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.float32)

    def _cpu_search() -> tuple[np.ndarray, np.ndarray]:
        idx_cpu = faiss.IndexFlatL2(targets.shape[1])
        idx_cpu.add(targets)
        d, i = idx_cpu.search(query, 1)
        return i.ravel(), d.ravel()

    if use_gpu and hasattr(faiss, "index_cpu_to_all_gpus"):
        try:
            index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(targets.shape[1]))
            index.add(targets)
            dists, idx = index.search(query, 1)
            idx = idx.ravel()
            dists = dists.ravel()
            if (idx < 0).any() or (idx >= len(targets)).any():
                logger.warning("faiss GPU returned out-of-range indices; falling back to CPU")
                return _cpu_search()
            return idx, dists
        except Exception as e:
            logger.warning("faiss GPU search failed (%s); falling back to CPU", e)
    return _cpu_search()


# ---------------------------------------------------------------------------
# Subsampling helper
# ---------------------------------------------------------------------------

def _subsample(
    emb: np.ndarray,
    names: np.ndarray,
    max_obs: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample to at most max_obs rows."""
    if max_obs is None or len(names) <= max_obs:
        return emb, names
    idx = rng.choice(len(names), size=max_obs, replace=False)
    idx.sort()  # preserve order for reproducibility
    return emb[idx], names[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _observed_at(
    states: ad.AnnData,
    time: float,
    time_key: str,
    obsm_key: str,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """return (embeddings (m, d), obs_names) for cells near the given time."""
    times = states.obs[time_key].values.astype(float)
    mask = np.abs(times - time) <= tol
    emb = np.asarray(states.obsm[obsm_key][mask], dtype=np.float32)
    names = states.obs_names[mask].values
    return emb, names


def _clip_subtree(
    tree: nx.DiGraph,
    root,
    has_embedding: set,
) -> tuple[nx.DiGraph, list]:
    """extract subtree from root, stopping at leaves or nodes with embeddings."""
    nodes = set()
    leaves = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.add(node)
        children = list(tree.successors(node))
        if not children:
            leaves.append(node)
        else:
            for child in children:
                if child in has_embedding or tree.out_degree(child) == 0:
                    nodes.add(child)
                    leaves.append(child)
                else:
                    stack.append(child)
    return tree.subgraph(nodes).copy(), leaves


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def couple_tree(
    pool: torch.nn.Module,
    tree: nx.DiGraph,
    states: ad.AnnData,
    *,
    obsm_key: str = "X_scvi",
    time_key: str = "time",
    time_tol: float = 0.5,
    batch_size: int = 256,
    max_obs: int | None = None,
    max_dist: float | None = None,
    pool_n_steps: int = 1,
    seed: int = 0,
) -> dict[str, str]:
    """couple every internal node in the tree to an observed cell.

    args:
        pool: bottom-up pooling model with pool_batch(batch) -> (B, N, dim).
        tree: full lineage tree. leaves must match obs_names in states.
              every node must have a 'time' attribute.
        states: anndata with embeddings in obsm[obsm_key] and time in obs[time_key].
        obsm_key: key for embeddings in states.obsm.
        time_key: key for time in states.obs.
        time_tol: tolerance for matching node times to observation times.
        batch_size: max subtrees per pooling forward pass.
        max_obs: if set, subsample observed cells per time layer to this many
                 before NN search.  useful for fast iteration / prototyping.
        max_dist: L2 distance threshold.  if the nearest observed cell is
                  farther than this, the node is assigned UNKNOWN and its
                  pooled embedding is kept (not snapped to the match).
                  None means no threshold.
        seed: random seed for subsampling reproducibility.

    returns:
        dict with keys:
          "assignments": maps each internal node name to an observed cell name
                         (or UNKNOWN if the nearest neighbour exceeded max_dist).
          "nn_distances": maps each internal node name to the L2 distance
                          to its nearest observed cell.
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

    # leaf initialization
    obs_emb_all = np.asarray(states.obsm[obsm_key], dtype=np.float32)
    obs_name_to_idx = {n: i for i, n in enumerate(states.obs_names.values)}

    current_emb: dict[str, np.ndarray] = {}
    for node in tree.nodes():
        if tree.out_degree(node) == 0 and node in obs_name_to_idx:
            current_emb[node] = obs_emb_all[obs_name_to_idx[node]]

    logger.info("initialized %d leaf embeddings", len(current_emb))

    assignments: dict[str, str] = {}
    nn_distances: dict[str, float] = {}
    n_unknown = 0

    for layer_idx in range(len(time_bins) - 1, 0, -1):
        t_earlier = time_bins[layer_idx - 1]

        target_nodes = [
            n for n in tree.nodes()
            if tree.out_degree(n) > 0
            and abs(tree.nodes[n].get("time", float("nan")) - t_earlier) <= time_tol
        ]
        if not target_nodes:
            continue

        logger.info(
            "layer %d/%d: time=%.3f, candidates=%d",
            layer_idx, len(time_bins) - 1, t_earlier, len(target_nodes),
        )

        has_emb = set(current_emb)
        subtree_dicts = []
        for tnode in tqdm(target_nodes, desc="clipping subtrees", leave=False):
            sg, leaves = _clip_subtree(tree, tnode, has_emb)
            if all(l in current_emb for l in leaves):
                subtree_dicts.append({
                    "graph": sg,
                    "root": tnode,
                    "root_time": tree.nodes[tnode].get("time", float("nan")),
                    "leaves": leaves,
                })

        if not subtree_dicts:
            continue

        # pool subtrees
        leaf_tensors = {
            name: torch.as_tensor(emb, dtype=torch.float32)
            for name, emb in current_emb.items()
        }
        node_ids = []
        embeddings = []

        for i in tqdm(range(0, len(subtree_dicts), batch_size), desc="pooling", leave=False):
            chunk = subtree_dicts[i : i + batch_size]
            batch = collate_subtrees(chunk, leaf_tensors)
            batch = batch.to(device)
            pooled = _pool_call(pool, batch, pool_n_steps).cpu()

            for j in range(len(chunk)):
                root_idx = batch.root_idx[j].item()
                node_ids.append(chunk[j]["root"])
                embeddings.append(pooled[j, root_idx].numpy())

            del batch, pooled
            torch.cuda.empty_cache()

        int_emb = np.stack(embeddings)

        # observed cells at this time (optionally subsampled)
        obs_emb, obs_names = _observed_at(states, t_earlier, time_key, obsm_key, time_tol)
        if len(obs_names) == 0:
            continue
        obs_emb, obs_names = _subsample(obs_emb, obs_names, max_obs, rng)

        # fast nearest neighbor
        nn_idx, nn_dist = _nearest_neighbor(int_emb, obs_emb, use_gpu=False)
        if not np.isfinite(int_emb).all():
            raise RuntimeError(
                f"pool produced non-finite query embeddings: "
                f"NaN={np.isnan(int_emb).any()} Inf={np.isinf(int_emb).any()}"
            )
        if (nn_idx < 0).any() or (nn_idx >= len(obs_names)).any():
            bad = nn_idx[(nn_idx < 0) | (nn_idx >= len(obs_names))][:5]
            raise RuntimeError(
                f"CPU faiss returned out-of-range indices (e.g. {bad.tolist()}) "
                f"against obs_emb shape {obs_emb.shape}, query shape {int_emb.shape}, "
                f"query finite={np.isfinite(int_emb).all()}, "
                f"targets finite={np.isfinite(obs_emb).all()}"
            )

        for k, node_name in enumerate(node_ids):
            nn_distances[node_name] = float(nn_dist[k])
            if max_dist is not None and nn_dist[k] > max_dist:
                assignments[node_name] = UNKNOWN
                # keep the pooled embedding so upstream layers can still use it
                current_emb[node_name] = embeddings[k]
                n_unknown += 1
            else:
                matched_name = obs_names[nn_idx[k]]
                assignments[node_name] = matched_name
                current_emb[node_name] = obs_emb[nn_idx[k]]

    logger.info("coupled %d internal nodes (%d unknown)", len(assignments), n_unknown)
    return {"assignments": assignments, "nn_distances": nn_distances}


@torch.no_grad()
def couple_all_trees(
    pool: torch.nn.Module,
    trees: dict[str, nx.DiGraph],
    states: ad.AnnData,
    **kwargs,
) -> dict:
    """Run couple_tree on every tree and aggregate results.

    args:
        pool: bottom-up pooling model.
        trees: mapping from tree name to lineage tree.
        states: anndata with embeddings and time annotations.
        **kwargs: forwarded to couple_tree.

    returns:
        dict where result[tree_name] gives the assignments dict for that tree,
        and result["nn_distances"][tree_name] gives the per-node distances.
    """
    result: dict = {}
    nn_distances: dict[str, dict[str, float]] = {}

    for tree_name, tree in tqdm(trees.items(), desc="coupling trees"):
        out = couple_tree(pool=pool, tree=tree, states=states, **kwargs)
        result[tree_name] = out["assignments"]
        nn_distances[tree_name] = out["nn_distances"]

    result["nn_distances"] = nn_distances
    return result