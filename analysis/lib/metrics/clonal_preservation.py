# measure how well a pooler + splitter preserve clonal structure.
# pick a leaf, pool its subtree up to an ancestor, split back down,
# then compare reconstructed leaf set to true descendants via energy distance.

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


from datasets.tree_record import TreeBatch, collate_subtrees, _extract_subtree, _leaf_counts, _node_depths


def _ancestor_at_time(tree, leaf, target_time, tol=0.5):
    """walk from leaf toward root, return the node closest to target_time."""
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
    """return all leaf descendants of node (including node if it is a leaf)."""
    if tree.out_degree(node) == 0:
        return [node]
    leaves = []
    stack = [node]
    while stack:
        n = stack.pop()
        children = list(tree.successors(n))
        if not children:
            leaves.append(n)
        else:
            stack.extend(children)
    return leaves


from analysis.lib.metrics.energy_distance import energy_distance as _energy_distance


@torch.no_grad()
def clonal_preservation(
    pool: torch.nn.Module,
    split: torch.nn.Module,
    trees: dict[str, nx.DiGraph],
    leaf_states: dict[str, torch.Tensor],
    *,
    n_leaves: int = 100,
    offset: float = 1.0,
    time_tol: float = 0.5,
    batch_size: int = 64,
    seed: int = 212121,
    device: str | None = None,
    split_steps: int = 100,
) -> pd.DataFrame:
    """compute clonal preservation error for sampled leaves.

    for each sampled leaf, find ancestor at leaf_time - offset, extract the
    subtree rooted there, pool up to the ancestor, split back down, and
    compare the reconstructed leaf states to the true leaf states.

    returns a dataframe with columns:
        tree, leaf, leaf_time, internal_time, n_descendants, energy_distance
    """
    # infer device from model params if not specified
    if device is None:
        device = next(pool.parameters()).device
    pool.eval()
    split.eval()
    rng = np.random.default_rng(seed)

    # collect all candidate leaves across trees
    candidates = []
    for tree_name, tree in tqdm(trees.items(), desc="collecting candidates", leave=False):
        for node in tree.nodes():
            if tree.out_degree(node) != 0:
                continue
            if node not in leaf_states:
                continue
            t = tree.nodes[node].get("time", float("nan"))
            anc = _ancestor_at_time(tree, node, t - offset, tol=time_tol)
            if anc is None or anc == node:
                continue
            candidates.append((tree_name, node, t, anc))

    if not candidates:
        return pd.DataFrame(columns=[
            "tree", "leaf", "leaf_time", "internal_time", "n_descendants", "energy_distance"
        ])

    # subsample
    n_sample = min(n_leaves, len(candidates))
    idxs = rng.choice(len(candidates), size=n_sample, replace=False)
    sampled = [candidates[i] for i in idxs]

    # build subtree records for each sampled leaf
    subtree_dicts = []
    meta = []
    _cache = {}  # avoid recomputing leaf_counts/node_depths per tree
    for tree_name, leaf, leaf_time, anc in tqdm(sampled, desc="building subtrees", leave=False):
        tree = trees[tree_name]
        if tree_name not in _cache:
            _cache[tree_name] = (_leaf_counts(tree), _node_depths(tree))
        sizes, depths = _cache[tree_name]
        st = _extract_subtree(tree, anc, sizes, depths, tree_name)
        desc_leaves = _descendant_leaves(tree, anc)
        # skip if any descendant leaf is missing from leaf_states
        if not all(l in leaf_states for l in desc_leaves):
            continue
        subtree_dicts.append(st)
        anc_time = tree.nodes[anc].get("time", float("nan"))
        meta.append({
            "tree": tree_name,
            "leaf": leaf,
            "leaf_time": np.round(leaf_time, decimals=2),
            "internal_time": anc_time,
            "desc_leaves": desc_leaves,
        })

    if not subtree_dicts:
        return pd.DataFrame(columns=[
            "tree", "leaf", "leaf_time", "internal_time", "n_descendants", "energy_distance"
        ])

    # process in batches
    rows = []
    dim = next(iter(leaf_states.values())).shape[-1]

    for start in tqdm(range(0, len(subtree_dicts), batch_size), desc="clonal preservation"):
        end = min(start + batch_size, len(subtree_dicts))
        batch_sts = subtree_dicts[start:end]
        batch_meta = meta[start:end]

        batch = collate_subtrees(batch_sts, leaf_states, device=device)
        batch = batch.to(device)

        # pool: bottom-up to get all node states including root
        pooled_states = pool.pool_batch(batch)  # (B, N, dim)

        # extract root states
        B = len(batch_sts)
        root_states = pooled_states[torch.arange(B, device=device), batch.root_idx]

        # split: top-down from root
        if hasattr(split, "split_tree_batch"):
            recon_states = split.split_tree_batch(batch, root_states, n_steps=split_steps) \
                if "n_steps" in split.split_tree_batch.__code__.co_varnames \
                else split.split_tree_batch(batch, root_states)
        else:
            raise ValueError("splitter has no split_tree_batch method")

        # compare reconstructed leaf states to true leaf states
        recon_np = recon_states.cpu().numpy()
        for i, m in enumerate(batch_meta):
            desc = m["desc_leaves"]
            node_names = batch.node_names[i]
            n_nodes = batch.n_nodes[i].item()
            name_to_idx = {node_names[j]: j for j in range(n_nodes)}

            # true descendant states
            true_states = np.stack([leaf_states[l].cpu().numpy() for l in desc])

            # reconstructed descendant states
            recon_desc = []
            for l in desc:
                if l in name_to_idx:
                    recon_desc.append(recon_np[i, name_to_idx[l]])
            if not recon_desc:
                continue
            recon_desc = np.stack(recon_desc)

            ed = _energy_distance(true_states, recon_desc)
            rows.append({
                "tree": m["tree"],
                "leaf": m["leaf"],
                "leaf_time": m["leaf_time"],
                "internal_time": m["internal_time"],
                "n_descendants": len(desc),
                "energy_distance": ed,
            })

    return pd.DataFrame(rows)


def _load_genot_model(pkl_path, seed=0):
    """Load a saved GENOT model from a .pkl file."""
    import pickle
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from ott.neural.networks.velocity_field.mlp import MLP

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    cfg = saved["model_config"]
    flat_np = saved["flat_params"]
    t_min = saved["t_min"]
    t_max = saved["t_max"]
    t_range = t_max - t_min if t_max != t_min else 1.0

    mdl = MLP(
        dim=cfg["dim"],
        cond_dim=cfg["cond_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        rngs=nnx.Rngs(seed),
    )
    graphdef, state = nnx.split(mdl)
    _, tree_def_local = jax.tree.flatten(state)
    restored_state = jax.tree.unflatten(
        tree_def_local, [jnp.array(x) for x in flat_np]
    )
    mdl = nnx.merge(graphdef, restored_state)

    return mdl, cfg, t_min, t_max, t_range


def _genot_transport(model, source_expr, t_src, t_tgt, t_min, t_range, dim,
                     n_samples=1, seed=0):
    """Transport source expression points through the GENOT velocity field.

    Returns array of shape (n_source, n_samples, dim).
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from ott.neural.methods.flow_matching import evaluate_velocity_field

    def _norm(t):
        return (float(t) - t_min) / t_range

    src = jnp.array(source_expr)
    n = src.shape[0]
    cond = jnp.concatenate([
        src,
        jnp.full((n, 1), _norm(t_src)),
        jnp.full((n, 1), _norm(t_tgt)),
    ], axis=1)

    key = jr.PRNGKey(seed)
    all_samples = []
    for s in range(n_samples):
        key, k_noise = jr.split(key)
        x0 = jr.normal(k_noise, shape=(n, dim))

        def _integrate_one(x_i, cond_i):
            sol = evaluate_velocity_field(
                model, x_i, cond=cond_i, t0=0.0, t1=1.0
            )
            ys = sol.ys
            if isinstance(ys, dict):
                ys = ys["x_t"]
            return jnp.reshape(ys, (-1,))

        transported = jax.vmap(_integrate_one)(x0, cond)
        all_samples.append(np.asarray(transported))

    # (n_source, n_samples, dim)
    return np.stack(all_samples, axis=1)


def clonal_preservation_genot(
    pkl_path: str,
    trees: dict[str, nx.DiGraph],
    leaf_states: dict[str, np.ndarray | torch.Tensor],
    *,
    n_leaves: int = 100,
    offset: float = 1.0,
    time_tol: float = 0.5,
    n_samples: int = 5,
    seed: int = 212121,
) -> pd.DataFrame:
    """Clonal preservation error for a GENOT conditional flow-matching model.

    For each sampled leaf at time t_late:
      1. Find the ancestor node at t_early = t_late - offset.
      2. Transport the leaf backward: t_late -> t_early (1 sample).
      3. Transport the inferred early cell forward: t_early -> t_late
         (n_samples stochastic samples to form a reconstructed descendant set).
      4. Compare reconstructed descendants to true descendants via energy distance.

    Parameters
    ----------
    pkl_path : str
        Path to the saved GENOT .pkl file.
    trees : dict[str, nx.DiGraph]
        Tree structures with node attribute "time".
    leaf_states : dict[str, np.ndarray | torch.Tensor]
        Expression vectors for leaf nodes (keyed by node name).
    n_leaves : int
        Number of leaves to sample.
    offset : float
        How far back in time to go for the ancestor.
    time_tol : float
        Tolerance for ancestor time matching.
    n_samples : int
        Number of stochastic forward samples per inferred ancestor.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: tree, leaf, leaf_time, internal_time, n_descendants, energy_distance
    """
    rng = np.random.default_rng(seed)

    # load model
    model, cfg, t_min, t_max, t_range = _load_genot_model(pkl_path, seed=seed)
    dim = cfg["dim"]

    # convert leaf_states to numpy
    leaf_np = {}
    for k, v in leaf_states.items():
        leaf_np[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    # collect candidates: leaves with a valid ancestor at t - offset
    candidates = []
    for tree_name, tree in tqdm(trees.items(), desc="collecting candidates", leave=False):
        for node in tree.nodes():
            if tree.out_degree(node) != 0:
                continue
            if node not in leaf_np:
                continue
            t = tree.nodes[node].get("time", float("nan"))
            anc = _ancestor_at_time(tree, node, t - offset, tol=time_tol)
            if anc is None or anc == node:
                continue
            # get descendant leaves of that ancestor
            desc_leaves = _descendant_leaves(tree, anc)
            if not all(l in leaf_np for l in desc_leaves):
                continue
            if len(desc_leaves) < 2:
                continue
            anc_time = tree.nodes[anc].get("time", float("nan"))
            candidates.append((tree_name, node, t, anc, anc_time, desc_leaves))

    if not candidates:
        return pd.DataFrame(columns=[
            "tree", "leaf", "leaf_time", "internal_time", "n_descendants", "energy_distance"
        ])

    # subsample
    n_sample = min(n_leaves, len(candidates))
    idxs = rng.choice(len(candidates), size=n_sample, replace=False)
    sampled = [candidates[i] for i in idxs]

    rows = []
    for tree_name, leaf, leaf_time, anc, anc_time, desc_leaves in tqdm(
        sampled, desc="clonal preservation (genot)"
    ):
        leaf_expr = leaf_np[leaf][np.newaxis, :]  # (1, dim)

        # step 1: transport leaf backward to ancestor time (single sample)
        inferred_ancestor = _genot_transport(
            model, leaf_expr, t_src=leaf_time, t_tgt=anc_time,
            t_min=t_min, t_range=t_range, dim=dim,
            n_samples=1, seed=seed,
        )  # (1, 1, dim)
        inferred_ancestor = inferred_ancestor[0, 0, :]  # (dim,)

        # step 2: transport inferred ancestor forward to leaf time (n_samples)
        ancestor_expr = inferred_ancestor[np.newaxis, :]  # (1, dim)
        recon_descendants = _genot_transport(
            model, ancestor_expr, t_src=anc_time, t_tgt=leaf_time,
            t_min=t_min, t_range=t_range, dim=dim,
            n_samples=n_samples, seed=seed + 1,
        )  # (1, n_samples, dim)
        recon_descendants = recon_descendants[0]  # (n_samples, dim)

        # true descendant states
        true_states = np.stack([leaf_np[l] for l in desc_leaves])

        ed = _energy_distance(true_states, recon_descendants)
        rows.append({
            "tree": tree_name,
            "leaf": leaf,
            "leaf_time": np.round(leaf_time, decimals=2),
            "internal_time": anc_time,
            "n_descendants": len(desc_leaves),
            "energy_distance": ed,
        })

    return pd.DataFrame(rows)
