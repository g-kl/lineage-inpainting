"""splitter (forward) and pooler (reverse) inference for the crossing evals.

handles both splitter interfaces (split_tree_batch / split_layer_batch) and draws
n_samples from stochastic models or tiles a deterministic pass. baseline forward and
reverse prediction is driven off baselines.registry.
"""

from __future__ import annotations

import numpy as np
import torch

from datasets.tree_record import collate_subtrees
from synthetic_evals.baselines import genot_fgw, genot_sinkhorn, marginal_fm, registry
from synthetic_evals.core.hydra_io import DEVICE
from synthetic_evals.core.subtrees import (
    leaf_positions,
    subtree_dicts_and_leaves,
)

fm_predict = marginal_fm.predict


# --------------------------------------------------------------------------
#  Splitter (forward) inference
# --------------------------------------------------------------------------


def split_from_root(splitter, batch, root_states, *, n_flow_steps):
    """Integrate a tree from a root-only anchor, for any splitter interface.

    FlowMatchingSplit-style splitters expose ``split_tree_batch(batch,
    root_states, n_steps)`` directly. Bridge models expose the lower-level
    ``split_layer_batch(batch, anchor_states, anchor_mask, n_steps)``, so we
    build the root-only anchor here. Same semantics either way.
    """
    if hasattr(splitter, "split_tree_batch"):
        return splitter.split_tree_batch(batch, root_states, n_steps=n_flow_steps)
    B, N = batch.mask.shape
    b_arange = torch.arange(B, device=root_states.device)
    anchor_mask = torch.zeros(B, N, dtype=torch.bool, device=root_states.device)
    anchor_mask[b_arange, batch.root_idx] = True
    anchor_states = torch.zeros(
        B, N, root_states.shape[-1],
        device=root_states.device, dtype=root_states.dtype,
    )
    anchor_states[b_arange, batch.root_idx] = root_states
    return splitter.split_layer_batch(
        batch, anchor_states, anchor_mask, n_steps=n_flow_steps,
    )


@torch.no_grad()
def splitter_predict_subtrees(splitter, subtrees, *, n_samples: int, n_flow_steps: int):
    """Anchor each root with its true x_src; sample predictions at all leaves.

    Returns (n_samples, total_leaves, dim) in ``subtrees_to_pairs`` order.
    """
    dim = int(getattr(splitter, "dim"))
    if not subtrees:
        return np.zeros((n_samples, 0, dim), dtype=np.float32)

    subtree_dicts, leaf_states = subtree_dicts_and_leaves(
        subtrees, leaf_state_fn=lambda _s, _n: torch.zeros(dim),
    )
    batch = collate_subtrees(
        subtree_dicts, leaf_states, device=DEVICE, randomize_child_index=False,
    )

    B = batch.mask.shape[0]
    root_states = torch.zeros(B, dim, device=DEVICE, dtype=torch.float32)
    for i, s in enumerate(subtrees):
        root_states[i] = torch.as_tensor(
            s["x_src"], dtype=torch.float32, device=DEVICE,
        )

    positions = leaf_positions(batch, subtrees)
    if not positions:
        return np.zeros((n_samples, 0, dim), dtype=np.float32)

    out = np.zeros((n_samples, len(positions), dim), dtype=np.float32)
    for s_idx in range(n_samples):
        pred = split_from_root(splitter, batch, root_states, n_flow_steps=n_flow_steps)
        for k, (i, j) in enumerate(positions):
            out[s_idx, k] = pred[i, j].detach().cpu().numpy()
    return out


# --------------------------------------------------------------------------
#  Pooler (reverse / ancestral) inference
# --------------------------------------------------------------------------


def _children_mean(subtrees):
    cm = np.stack([
        np.mean(
            np.stack([s["leaf_states"][n].cpu().numpy() for n, _ in s["leaves"]]),
            axis=0,
        )
        for s in subtrees
    ], axis=0).astype(np.float32)
    return cm[:, None] if cm.ndim == 1 else cm


def _parent_true(subtrees):
    pt = np.stack([s["x_src"] for s in subtrees], axis=0).astype(np.float32)
    return pt[:, None] if pt.ndim == 1 else pt


def _child_to_parent(subtrees):
    return np.array(
        [i for i, s in enumerate(subtrees) for _ in s["leaves"]], dtype=np.int64,
    )


@torch.no_grad()
def pool_predict_subtrees(pool, subtrees, *, n_samples: int, n_flow_steps: int):
    """Pooler ancestral prediction (children at t_tgt -> ancestor at t_src).

    Draws ``n_samples`` stochastic rollouts per parent (bridge pooler) or
    repeats the deterministic pass -- symmetric with the splitter. Layout:
    ``parent_pred`` is (S, P, dim); ``parent_true`` / ``parent_children_mean``
    are (P, dim); ``child_to_parent`` is (M,).
    """
    dim = int(getattr(pool, "dim"))
    if not subtrees:
        empty = np.zeros((0, dim), dtype=np.float32)
        return {
            "parent_true": empty,
            "parent_pred": np.zeros((n_samples, 0, dim), dtype=np.float32),
            "parent_children_mean": empty,
            "child_to_parent": np.zeros((0,), dtype=np.int64),
        }

    subtree_dicts, leaf_states = subtree_dicts_and_leaves(
        subtrees, leaf_state_fn=lambda s, name: s["leaf_states"][name],
    )
    batch = collate_subtrees(
        subtree_dicts, leaf_states, device=DEVICE, randomize_child_index=False,
    )

    # Stochastic multi-step pooler (bridge) takes n_steps; deterministic
    # poolers (tree-former) don't. Detect from the signature and
    # draw n_samples either way.
    takes_n_steps = accepts_n_steps(pool.pool_batch)
    b_range = torch.arange(len(subtrees), device=DEVICE)
    samples = []
    for _ in range(n_samples):
        pooled = pool.pool_batch(batch, n_steps=n_flow_steps) if takes_n_steps \
            else pool.pool_batch(batch)
        samples.append(pooled[b_range, batch.root_idx].detach().cpu().numpy())
    parent_pred = np.stack(samples, axis=0).astype(np.float32)   # (S, P, dim)

    return {
        "parent_true": _parent_true(subtrees),
        "parent_pred": parent_pred,
        "parent_children_mean": _children_mean(subtrees),
        "child_to_parent": _child_to_parent(subtrees),
    }


def accepts_n_steps(fn) -> bool:
    import inspect
    try:
        return "n_steps" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def reverse_per_parent(bl, pkg, subtrees, pairs, t_src, t_tgt, *,
                       n_samples: int, n_flow_steps: int, seed: int = 0):
    """Ancestral prediction for a registered baseline: integrate each child back to
    t_src, average per parent, draw ``n_samples`` (vary seed if stochastic, else tile).
    Returns ``parent_pred`` of shape (S, P, dim)."""
    dim_out = bl.dim_of(pkg)
    predict_kwargs = registry.predict_kwargs(bl, n_flow_steps=n_flow_steps, seed=seed)

    if not subtrees:
        empty = np.zeros((0, dim_out), dtype=np.float32)
        return {
            "parent_true": empty,
            "parent_pred": np.zeros((n_samples, 0, dim_out), dtype=np.float32),
            "parent_children_mean": empty,
            "child_to_parent": np.zeros((0,), dtype=np.int64),
        }

    x_c = np.stack([p[5] for p in pairs], axis=0).astype(np.float32)
    x_c_t = torch.from_numpy(x_c).to(DEVICE)
    child_to_parent = _child_to_parent(subtrees)
    P = len(subtrees)

    def _one(kw):
        pred_child = bl.predict(pkg, x_c_t, t_tgt, t_src, **kw).cpu().numpy()
        dim = pred_child.shape[-1]
        sums = np.zeros((P, dim), dtype=np.float64)
        counts = np.zeros(P, dtype=np.int64)
        for i, p_idx in enumerate(child_to_parent):
            sums[p_idx] += pred_child[i]
            counts[p_idx] += 1
        return (sums / counts[:, None]).astype(np.float32)

    if bl.stochastic:
        preds = [_one({**predict_kwargs, "seed": seed + s}) for s in range(n_samples)]
    else:
        preds = [_one(predict_kwargs)] * n_samples
    parent_pred = np.stack(preds, axis=0)   # (S, P, dim)

    return {
        "parent_true": _parent_true(subtrees),
        "parent_pred": parent_pred,
        "parent_children_mean": _children_mean(subtrees),
        "child_to_parent": child_to_parent,
    }


def forward_samples(bl, pkg, x_src, t_src, t_tgt, *, n_samples: int,
                    n_flow_steps: int | None):
    """Forward samples for a registered baseline; returns (S, M, dim). Stochastic
    baselines redraw per seed; deterministic ones run once and tile."""
    x_src_t = torch.from_numpy(np.asarray(x_src, dtype=np.float32)).to(DEVICE)
    if not bl.stochastic:
        kw = registry.predict_kwargs(bl, n_flow_steps=n_flow_steps)
        one = bl.predict(pkg, x_src_t, t_src, t_tgt, **kw).cpu().numpy()
        return np.broadcast_to(one[None], (n_samples,) + one.shape).copy()
    samples = [
        bl.predict(pkg, x_src_t, t_src, t_tgt, seed=s).cpu().numpy()
        for s in range(n_samples)
    ]
    return np.stack(samples, axis=0)


# Convenience handles for the eval loop.
gs_predict = genot_sinkhorn.predict
gf_predict = genot_fgw.predict
