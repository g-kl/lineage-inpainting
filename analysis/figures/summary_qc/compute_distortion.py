"""clonal distortion + marginal distortion

clonal distortion (cycle consistency)
    pick a leaf, find its ancestor at leaf_time - offset, and take that ancestor's
    subtree. round-trip leaf set -> ancestor -> leaf set and measure the energy
    distance between the regenerated leaf cloud and the true descendant leaf cloud.
      - model          : the SPLITTER does both halves -- anchor the observed
                         leaves and predict the subtree root (pool), then anchor
                         ONLY that predicted root, mask all leaves, and regenerate
                         them (split). lossy through the single root, so a true
                         cycle.
      - genot_*        : transport the focal leaf back to the ancestor time, then
                         forward n_samples times to a reconstructed descendant
                         cloud (analysis.lib.metrics.clonal_preservation.clonal_preservation_genot).
    all methods are scored on the SAME sampled (leaf, ancestor, descendant-set)
    tuples, so the only difference is the pool/split operator.

marginal distortion (energy distance between marginals)
    for each adjacent timepoint pair t_late -> t_early, push the t_late cells back
    to t_early and measure the energy distance to the true t_early marginal.
      - model          : splitter pool (anchor the cell's subtree leaves, read the
                         ancestor/root state at t_early).
      - genot_*        : transport the cells t_late -> t_early through the flow.
    same sampled source / target cells for all three methods.

writes numerical_results/clonal_distortion.csv and marginal_distortion.csv, each
with columns method, ..., energy_distance. model rows are written first so partial
progress survives a GENOT failure. run with the lt conda env from the repo root.
"""
import os
import sys

# keep JAX (GENOT) from grabbing the whole GPU -- it shares with torch here.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from analysis.figures.summary_qc import style as S
from analysis.lib.metrics.clonal_preservation import (
    _ancestor_at_time, _descendant_leaves, _energy_distance,
    _genot_transport, _load_genot_model,
)
from analysis.lib.metrics.marginal_preservation import _cells_at_time, _get_time_points, _subsample
from analysis.pipelines.context._context_common import load_models
from datasets.tree_record import (
    collate_subtrees, _extract_subtree, _leaf_counts, _node_depths,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# shared sampling (identical leaf sets across methods)
# --------------------------------------------------------------------------

def collect_clonal_candidates(trees, leaf_states, *, offset, time_tol, n, seed,
                              min_leaves=2):
    """sampled (tree, leaf, leaf_time, ancestor, anc_time, desc_leaves) tuples."""
    rng = np.random.default_rng(seed)
    cands = []
    for tname, tree in trees.items():
        for node in tree.nodes():
            if tree.out_degree(node) != 0 or node not in leaf_states:
                continue
            t = tree.nodes[node].get("time", float("nan"))
            anc = _ancestor_at_time(tree, node, t - offset, tol=time_tol)
            if anc is None or anc == node:
                continue
            desc = _descendant_leaves(tree, anc)
            if len(desc) < min_leaves or not all(l in leaf_states for l in desc):
                continue
            cands.append((tname, node, t, anc,
                          tree.nodes[anc].get("time", float("nan")), desc))
    if not cands:
        return []
    idx = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    return [cands[int(i)] for i in idx]


def collect_marginal_samples(trees, leaf_np, *, time_tol, n_cells, seed):
    """per adjacent (t_late -> t_early): sampled src names + true target states."""
    rng = np.random.default_rng(seed)
    tps = _get_time_points(trees, tol=time_tol)
    out = []
    for i in range(len(tps) - 1):
        t_early, t_late = tps[i], tps[i + 1]
        names_e, states_e = _cells_at_time(trees, leaf_np, t_early, tol=time_tol)
        names_l, states_l = _cells_at_time(trees, leaf_np, t_late, tol=time_tol)
        if len(names_e) < 2 or len(names_l) < 2:
            continue
        src_names, _ = _subsample(rng, names_l, states_l, n_cells)
        tgt_names, tgt_states = _subsample(rng, names_e, states_e, n_cells)
        out.append((t_late, t_early, src_names, tgt_states))
    return out


# --------------------------------------------------------------------------
# model: the splitter as both pooler and splitter
# --------------------------------------------------------------------------

@torch.no_grad()
def _splitter_root_states(splitter, batch, n_steps):
    """pool: anchor observed leaves, return predicted root state per tree (B, dim)."""
    pred = splitter.split_layer_batch(
        batch, batch.observed_states, batch.observed_mask, n_steps=n_steps)
    B = pred.shape[0]
    return pred[torch.arange(B, device=pred.device), batch.root_idx]


# hooks: ``root_state_of(tname, anc)`` gives the root state to split from instead of
# the pooled one (the grid's descendant distortion starts from the ancestor's coupled
# real cell); ``on_split(row, desc_names, recon_leaves)`` may return extra columns
# for that row (the grid's cell-type TV of the same generation)
root_state_of = None
on_split = None


@torch.no_grad()
def model_clonal(splitter, candidates, trees, leaf_states, *, n_steps, max_nodes,
                 batch_size):
    """splitter leaf->root->leaf cycle energy distance for each candidate."""
    cache, sts, meta = {}, [], []
    for tname, leaf, leaf_time, anc, anc_time, desc in candidates:
        if tname not in cache:
            tree = trees[tname]
            cache[tname] = (_leaf_counts(tree), _node_depths(tree))
        sizes, depths = cache[tname]
        sts.append(_extract_subtree(trees[tname], anc, sizes, depths, tname))
        meta.append((tname, leaf, leaf_time, anc, anc_time, desc))

    rows = []
    for start in tqdm(range(0, len(sts), batch_size), desc="model clonal"):
        chunk = sts[start:start + batch_size]
        cmeta = meta[start:start + batch_size]
        batch = collate_subtrees(chunk, leaf_states, device=device,
                                 max_nodes=max_nodes, randomize_child_index=False).to(device)
        B = len(chunk)
        ar = torch.arange(B, device=device)

        if root_state_of is None:
            root_state = _splitter_root_states(splitter, batch, n_steps)    # pool
        else:                                                               # a given root
            root_state = torch.as_tensor(np.stack([root_state_of(m[0], m[3]) for m in cmeta]),
                                         device=device, dtype=batch.observed_states.dtype)
        down_anchor = torch.zeros_like(batch.observed_states)
        down_anchor[ar, batch.root_idx] = root_state
        down_mask = torch.zeros_like(batch.observed_mask)
        down_mask[ar, batch.root_idx] = True
        recon = splitter.split_layer_batch(batch, down_anchor, down_mask,
                                           n_steps=n_steps).cpu().numpy()    # split

        obs = batch.observed_states.cpu().numpy()
        leaf_mask = batch.leaf_mask.cpu().numpy()
        n_nodes = batch.n_nodes.cpu().numpy()
        for b in range(B):
            li = np.where(leaf_mask[b, :n_nodes[b]])[0]
            tname, leaf, leaf_time, anc, anc_time, desc = cmeta[b]
            row = dict(
                method="model", tree=tname, leaf=leaf,
                leaf_time=round(leaf_time, 2), internal_time=anc_time,
                n_descendants=int(len(li)),
                energy_distance=_energy_distance(obs[b, li], recon[b, li]),
            )
            if on_split is not None:
                row.update(on_split(row, desc, recon[b, li]))
            rows.append(row)
    return rows


# hooks on the pool-back: ``pool_back(batch)`` gives the root states (B, dim) in
# place of the splitter's (the grid scores the imputing model on the same samples);
# ``on_transport(t_late, t_early, transported)`` may return extra columns for that
# row (the grid's cell-type TV of the same samples)
pool_back = None
on_transport = None


@torch.no_grad()
def model_marginal(splitter, samples, trees, leaf_states, *, n_steps, max_nodes,
                   batch_size, time_tol):
    """splitter pool-back: t_late cells -> ancestor states at t_early, energy distance."""
    node_to_tree = {n: tname for tname, tree in trees.items() for n in tree.nodes()}
    cache = {}
    rows = []
    for t_late, t_early, src_names, tgt_states in samples:
        sts = []
        for name in src_names:
            tname = node_to_tree.get(name)
            if tname is None:
                continue
            tree = trees[tname]
            anc = _ancestor_at_time(tree, name, t_early, tol=time_tol)
            if anc is None or anc == name:
                continue
            desc = _descendant_leaves(tree, anc)
            if not all(l in leaf_states for l in desc):
                continue
            if tname not in cache:
                cache[tname] = (_leaf_counts(tree), _node_depths(tree))
            sizes, depths = cache[tname]
            sts.append(_extract_subtree(tree, anc, sizes, depths, tname))
        if len(sts) < 2:
            continue
        roots = []
        for start in range(0, len(sts), batch_size):
            batch = collate_subtrees(sts[start:start + batch_size], leaf_states,
                                     device=device, max_nodes=max_nodes,
                                     randomize_child_index=False).to(device)
            root = (pool_back(batch) if pool_back is not None
                    else _splitter_root_states(splitter, batch, n_steps))
            roots.append(root.cpu().numpy())
        transported = np.concatenate(roots, axis=0)
        extra = on_transport(t_late, t_early, transported) if on_transport is not None else {}
        rows.append(dict(
            method="model", t_src=t_late, t_tgt=t_early,
            n_src=len(transported), n_tgt=len(tgt_states),
            energy_distance=_energy_distance(transported, tgt_states), **extra,
        ))
    return rows


# --------------------------------------------------------------------------
# GENOT baselines (same candidates / samples)
# --------------------------------------------------------------------------

def genot_clonal(pkl, method, candidates, leaf_np, *, n_samples, seed):
    model, cfg, t_min, t_max, t_range = _load_genot_model(pkl, seed=seed)
    dim = cfg["dim"]
    rows = []
    for tname, leaf, leaf_time, anc, anc_time, desc in tqdm(
        candidates, desc=f"{method} clonal",
    ):
        inferred = _genot_transport(
            model, leaf_np[leaf][None, :], t_src=leaf_time, t_tgt=anc_time,
            t_min=t_min, t_range=t_range, dim=dim, n_samples=1, seed=seed,
        )[0, 0, :]
        recon = _genot_transport(
            model, inferred[None, :], t_src=anc_time, t_tgt=leaf_time,
            t_min=t_min, t_range=t_range, dim=dim, n_samples=n_samples, seed=seed + 1,
        )[0]
        true = np.stack([leaf_np[l] for l in desc])
        rows.append(dict(
            method=method, tree=tname, leaf=leaf, leaf_time=round(leaf_time, 2),
            internal_time=anc_time, n_descendants=len(desc),
            energy_distance=_energy_distance(true, recon),
        ))
    return rows


def genot_marginal(pkl, method, samples, leaf_np, *, seed):
    model, cfg, t_min, t_max, t_range = _load_genot_model(pkl, seed=seed)
    dim = cfg["dim"]
    rows = []
    for t_late, t_early, src_names, tgt_states in tqdm(samples, desc=f"{method} marginal"):
        src_states = np.stack([leaf_np[n] for n in src_names])
        transported = _genot_transport(
            model, src_states, t_src=t_late, t_tgt=t_early,
            t_min=t_min, t_range=t_range, dim=dim, n_samples=1, seed=seed,
        )[:, 0, :]
        rows.append(dict(
            method=method, t_src=t_late, t_tgt=t_early,
            n_src=len(src_states), n_tgt=len(tgt_states),
            energy_distance=_energy_distance(transported, tgt_states),
        ))
    return rows


# --------------------------------------------------------------------------

def main():
    clonal_csv = os.path.join(S.NR, "clonal_distortion.csv")
    marginal_csv = os.path.join(S.NR, "marginal_distortion.csv")

    _, splitter, dataset = load_models()
    splitter.eval()
    trees, leaf_states = dataset.trees, dataset.leaf_states
    leaf_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in leaf_states.items()}

    candidates = collect_clonal_candidates(
        trees, leaf_states, offset=S.CLONAL_OFFSET, time_tol=S.CLONAL_TIME_TOL,
        n=S.N_SUBTREE_LEAVES, seed=S.SEED, min_leaves=S.MIN_CLONE_LEAVES)
    print(f"clonal: {len(candidates)} sampled subtrees", flush=True)
    samples = collect_marginal_samples(
        trees, leaf_np, time_tol=S.MARGINAL_TIME_TOL, n_cells=S.N_MARGINAL_CELLS,
        seed=S.SEED)
    print(f"marginal: {len(samples)} adjacent-timepoint pairs", flush=True)

    # --- model first (torch), so its rows persist even if GENOT fails ---
    clonal_rows = model_clonal(
        splitter, candidates, trees, leaf_states,
        n_steps=S.SPLIT_STEPS, max_nodes=S.MAX_NODES, batch_size=S.BATCH_SIZE)
    marginal_rows = model_marginal(
        splitter, samples, trees, leaf_states,
        n_steps=S.SPLIT_STEPS, max_nodes=S.MAX_NODES, batch_size=S.BATCH_SIZE,
        time_tol=S.MARGINAL_TIME_TOL)
    pd.DataFrame(clonal_rows).to_csv(clonal_csv, index=False)
    pd.DataFrame(marginal_rows).to_csv(marginal_csv, index=False)
    print(f"wrote model rows -> {clonal_csv}, {marginal_csv}", flush=True)
    del splitter
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- GENOT baselines on the identical samples ---
    for method, pkl in S.GENOT_PKLS.items():
        clonal_rows += genot_clonal(
            pkl, method, candidates, leaf_np, n_samples=S.N_GENOT_SAMPLES, seed=S.SEED)
        marginal_rows += genot_marginal(pkl, method, samples, leaf_np, seed=S.SEED)
        pd.DataFrame(clonal_rows).to_csv(clonal_csv, index=False)
        pd.DataFrame(marginal_rows).to_csv(marginal_csv, index=False)
        print(f"appended {method} -> {clonal_csv}, {marginal_csv}", flush=True)

    print("done.", flush=True)
    for name, csv in (("clonal", clonal_csv), ("marginal", marginal_csv)):
        df = pd.read_csv(csv)
        print(f"\n{name} median energy distance by method:")
        print(df.groupby("method")["energy_distance"].median().to_string())


if __name__ == "__main__":
    main()
