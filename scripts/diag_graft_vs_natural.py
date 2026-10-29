"""Compare a trained model's behaviour on (a) natural batches and (b) grafted
batches, to compare grafting with the splitter's second-stage prediction task.

For a chosen checkpoint:
  1. Pull two natural batches (e.g. an early and a late record).
  2. Pull a grafted batch (the early record extended forward to the late
     record's leaf_time).
  3. Run pool_batch on both natural and grafted batches.
  4. Compute the per-feature align loss between the two natural batches AND
     between the natural late vs grafted-early batches.
  5. Visualise the predicted embeddings at grafted positions on top of the
     observed-cell manifold to see if predictions land on the right
     distribution.

Usage:
    EXPERIMENT_DIR=outputs/... STEP=20000 python scripts/diag_graft_vs_natural.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import hydra
from omegaconf import OmegaConf

from datasets.tree_record import TreeRecord, collate_subtrees
from extenders import SubtreeBank
from extenders.graft import GraftExtender
from utils.experiment_utils import get_experiment_info


def _load_two_records():
    """Load just one E8.5 and one E9.0 tree record from cache (memory-light)."""
    cache_dir = "cache"
    early_cache = next(
        os.path.join(cache_dir, fn)
        for fn in sorted(os.listdir(cache_dir))
        if fn.startswith("final_topology.h5td_E8.5-") and fn.endswith(".pkl")
    )
    late_cache = next(
        os.path.join(cache_dir, fn)
        for fn in sorted(os.listdir(cache_dir))
        if fn.startswith("final_topology.h5td_E9.0-") and fn.endswith(".pkl")
    )
    early = TreeRecord.load(early_cache)
    late = TreeRecord.load(late_cache)
    return early, late, early_cache, late_cache


def _load_leaf_states_for(subtree_dicts, h5ad_path):
    """Load only the cell embeddings referenced by these subtrees (light)."""
    import treedata
    referenced = set()
    for sd in subtree_dicts:
        for n in sd["graph"].nodes():
            referenced.add(n)
    states = treedata.read_h5ad(h5ad_path)
    states = states[states.obs["dataset"] == "devmap"]
    obs_names = list(states.obs_names)
    obs_idx = {n: i for i, n in enumerate(obs_names)}
    X = states.obsm["X_scvi"]
    leaf_states = {}
    for name in referenced:
        if name in obs_idx:
            leaf_states[name] = torch.as_tensor(X[obs_idx[name]], dtype=torch.float32)
    # Also keep a small reference table for binding to pooler
    times = torch.as_tensor(states.obs["time"].values.astype(float), dtype=torch.float32)
    emb = torch.as_tensor(X, dtype=torch.float32)
    return leaf_states, emb, times


def main():
    experiment_dir = os.environ["EXPERIMENT_DIR"]
    step = int(os.environ.get("STEP", "20000"))
    print(f"[diag] EXPERIMENT_DIR={experiment_dir} STEP={step}")

    info = get_experiment_info(experiment_dir)
    cfg = info["config"]

    early_rec, late_rec, _, _ = _load_two_records()
    early_name = "E8.5"
    late_name = "E9.0"

    # Load pooler from checkpoint
    pooler = hydra.utils.instantiate(cfg["pooler"]).cuda()
    pool_ckpt = os.path.join(experiment_dir, f"pool_ema_step{step}.pt")
    pooler.load_state_dict(torch.load(pool_ckpt, map_location="cuda"))
    pooler.eval()
    print(f"[diag] pooler loaded: {type(pooler).__name__}")
    print(f"[diag] early={early_name} (t={early_rec.leaf_time}); "
          f"late={late_name} (t={late_rec.leaf_time})")

    rng = np.random.default_rng(0)
    sample_time = 7.5  # roots fall around here
    n_trees = 32

    early_subtrees = early_rec.get_subtrees(time=sample_time, n=n_trees, rng=rng)
    late_subtrees = late_rec.get_subtrees(time=sample_time, n=n_trees, rng=rng)
    print(f"[diag] early subtrees: {len(early_subtrees)}, "
          f"late subtrees: {len(late_subtrees)}")

    # Build a grafted version of the early subtrees. The extender pulls donor
    # topology from the SubtreeBank built over (early_rec, late_rec).
    extender = GraftExtender()
    extender.bank = SubtreeBank({early_name: early_rec, late_name: late_rec})
    extended = extender(
        early_subtrees,
        target_leaf_time=late_rec.leaf_time, rng=rng,
    )

    # Load only the leaf states we actually need
    all_subtrees = list(early_subtrees) + list(late_subtrees) + list(extended)
    leaf_states, emb, times = _load_leaf_states_for(
        all_subtrees, "data/lt/qiu24_integrated_hvgs_and_scvi.h5ad",
    )
    print(f"[diag] loaded {len(leaf_states)} leaf states")

    if hasattr(pooler, "bind_cells"):
        pooler.bind_cells(emb, times)

    # Collate the three sets
    def to_batch(sds):
        b = collate_subtrees(sds, leaf_states, device="cuda")
        return b

    b_early = to_batch(early_subtrees)
    b_late = to_batch(late_subtrees)
    b_extended = to_batch(extended)

    # Run pool_batch (one-step at t=0, observed_mask anchors)
    with torch.no_grad():
        s_early = pooler.pool_batch(b_early)
        s_late = pooler.pool_batch(b_late)
        s_extended = pooler.pool_batch(b_extended)

    print(f"[diag] state shapes: early={tuple(s_early.shape)}, "
          f"late={tuple(s_late.shape)}, extended={tuple(s_extended.shape)}")

    # ----- 1. compare predicted-state distributions at t=9.0 (the target) -----
    #
    # In b_late, all leaves are real observations at t=9.0.
    # In b_extended, "leaves" are grafted positions whose state is the model's
    # prediction. If the model is doing forward extrapolation correctly, these
    # predicted states should land on the same manifold as real cells at t=9.0.

    def states_at_time(batch, states, t_target, tol=0.05, only_unobs=False):
        time_match = (batch.node_times - t_target).abs() < tol
        mask = batch.mask & time_match
        if only_unobs:
            mask = mask & ~batch.observed_mask
        return states[mask], mask.sum().item()

    # Predicted grafted leaves at t=9.0 in the extended batch
    pred_grafted, n_pred = states_at_time(
        b_extended, s_extended, t_target=9.0, only_unobs=True,
    )
    # Observed real cells at t=9.0 in the late batch
    real_at_9, n_real = states_at_time(
        b_late, s_late, t_target=9.0, only_unobs=False,
    )

    print(f"[diag] predicted grafted at t=9.0: {n_pred}")
    print(f"[diag] real observed at t=9.0:    {n_real}")
    print(f"[diag] pred mean norm: {pred_grafted.norm(dim=-1).mean().item():.3f}")
    print(f"[diag] real mean norm: {real_at_9.norm(dim=-1).mean().item():.3f}")
    print(f"[diag] pred std    : {pred_grafted.std(dim=0).mean().item():.3f}")
    print(f"[diag] real std    : {real_at_9.std(dim=0).mean().item():.3f}")

    # Combined PCA
    combined = torch.cat([pred_grafted, real_at_9], dim=0).cpu().numpy()
    mean = combined.mean(axis=0)
    cov = np.cov(combined - mean, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pcs = eigvecs[:, -2:][:, ::-1]  # top 2 PCs
    proj = (combined - mean) @ pcs
    n1 = pred_grafted.shape[0]
    proj_pred = proj[:n1]
    proj_real = proj[n1:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    ax = axes[0]
    ax.scatter(proj_real[:, 0], proj_real[:, 1], s=8, c="#3c4ebf", alpha=0.5,
               label=f"real @ t=9.0 (n={n_real})", edgecolors="none")
    ax.scatter(proj_pred[:, 0], proj_pred[:, 1], s=8, c="#d6543a", alpha=0.5,
               label=f"predicted grafted @ t=9.0 (n={n_pred})", edgecolors="none")
    ax.set_title("PCA of states at t=9.0")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)

    # ----- 2. norm/scale comparison across times -----
    times = [7.5, 8.0, 8.5, 9.0]
    means_real = []
    means_pred = []
    for t in times:
        # Real interior cells in the late batch (pooler predictions in same time bin)
        st_late, n_l = states_at_time(b_late, s_late, t_target=t)
        # Predicted positions in the extended batch
        st_ext, n_e = states_at_time(b_extended, s_extended, t_target=t)
        means_real.append(st_late.norm(dim=-1).mean().item() if n_l > 0 else float("nan"))
        means_pred.append(st_ext.norm(dim=-1).mean().item() if n_e > 0 else float("nan"))

    ax = axes[1]
    ax.plot(times, means_real, marker="o", label="late batch pooled-state mean-norm")
    ax.plot(times, means_pred, marker="o", label="extended batch pooled-state mean-norm")
    ax.set_xlabel("time bin")
    ax.set_ylabel("mean L2 norm of pooled state")
    ax.set_title("scale of pooled states across time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = os.path.join(experiment_dir, f"diag_graft_step{step}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag] wrote {out}")


if __name__ == "__main__":
    main()
