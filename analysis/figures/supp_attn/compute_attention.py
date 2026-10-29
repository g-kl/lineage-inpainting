#!/usr/bin/env python
"""caches for the attention supplemental figure.

the model is the one the main figures use (`analysis/figures/summary_qc/style.py`).
attention is read by making each layer hand back its weights during the ordinary
inference rollout: observed leaves clamped as anchors, everything above them masked.
rollout step k runs the network at interpolant time s = 1 - k / N_STEPS.

writes into numerical_results/:
  scales.csv   the learned bias scale gamma, one per (layer, head).
  rollout.csv  per interpolant time / layer / head, the share of a masked node's
               attention that lands on the observed leaves, plus the share uniform
               attention would give.
  example.npz  one subtree's attention tensor at every rollout step.

    python -m analysis.figures.supp_attn.compute_attention
"""
import os

import hydra
import networkx as nx
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from analysis.figures.summary_qc import style as QC
from analysis.figures.supp_attn import style as S
from datasets.tree_record import _leaf_counts, collate_subtrees

N_SUBTREES = 128
BATCH = 8
TIME_RANGE = (7.0, 9.5)
MIN_LEAVES, MAX_LEAVES = 24, 96
MAX_NODES = 256
N_STEPS = 16          # reverse-SDE steps
EXAMPLE_LEAVES = (40, 64)
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    cfg = OmegaConf.load(os.path.join(QC.EXPERIMENT_DIR, "config.yaml"))
    model = hydra.utils.instantiate(cfg.splitter).to(device).eval()
    model.load_state_dict(torch.load(
        os.path.join(QC.EXPERIMENT_DIR, f"split_ema_step{QC.STEP}.pt"),
        map_location=device))
    print(f"model: {os.path.basename(QC.EXPERIMENT_DIR)} split_ema_step{QC.STEP}")
    print("loading dataset (a minute or so) ...", flush=True)
    return model, hydra.utils.instantiate(cfg.dataset)


def build_subtree_dict(tree, root_node):
    """the subtree under root_node, nodes inserted in dfs order from the root."""
    nodes = list(nx.dfs_preorder_nodes(tree, root_node))
    sg = nx.DiGraph()
    sg.add_nodes_from((n, tree.nodes[n]) for n in nodes)
    sg.add_edges_from((u, v, tree.edges[u, v]) for u in nodes for v in tree.successors(u))
    leaves = [n for n in sg if sg.out_degree(n) == 0]
    return {"graph": sg, "root": root_node, "leaves": leaves, "size": len(leaves)}


def pick_subtrees(trees, n, rng):
    cands = []
    for tname, tree in trees.items():
        sizes = _leaf_counts(tree)
        for node in tree.nodes():
            t = tree.nodes[node].get("time", float("nan"))
            if (tree.out_degree(node) and TIME_RANGE[0] <= t <= TIME_RANGE[1]
                    and MIN_LEAVES <= sizes[node] <= MAX_LEAVES):
                cands.append((tname, node))
    idx = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    return [cands[int(i)] for i in idx]


class capture:
    """collect every layer's (B, H, N, N) attention from each model forward."""

    def __init__(self, model):
        self.mods = [layer.attn for layer in model.encoder_layers]
        self.weights = []

    def __enter__(self):
        self.orig = [m.forward for m in self.mods]
        for m, orig in zip(self.mods, self.orig):
            def fwd(q, k, v, _o=orig, **kw):
                kw.update(need_weights=True, average_attn_weights=False)
                out, w = _o(q, k, v, **kw)
                self.weights.append(w.detach())
                return out, w
            m.forward = fwd
        return self

    def __exit__(self, *exc):
        for m, orig in zip(self.mods, self.orig):
            m.forward = orig


@torch.no_grad()
def rollout(model, batch):
    """attention at each rollout step: list of (L, B, H, N, N), one per step."""
    anchor = batch.observed_mask & batch.mask
    with capture(model) as cap:
        model.split_layer_batch(batch, batch.observed_states, anchor, n_steps=N_STEPS)
    L = len(model.encoder_layers)
    return [torch.stack(cap.weights[k * L:(k + 1) * L]) for k in range(N_STEPS)]


def interpolant_time(k):
    return 1.0 - k / N_STEPS


def collate(chunk, dataset, seed):
    return collate_subtrees(chunk, dataset.leaf_states, device=device,
                            max_nodes=MAX_NODES, rng=np.random.default_rng(seed))


def observed_share(w, batch):
    """per (layer, head), summed over masked query rows: attention on observed nodes."""
    B, N = batch.mask.shape
    obs = (batch.observed_mask & batch.mask).float().view(1, B, 1, 1, N)
    query = ((~batch.observed_mask) & batch.mask).float().view(1, B, 1, N)
    return ((w * obs).sum(-1) * query).sum((1, 3)).cpu().numpy()


def rollout_table(model, dataset, subtrees):
    L, H = len(model.encoder_layers), model.n_heads
    acc = np.zeros((N_STEPS, L, H))
    n_rows, uniform = 0.0, 0.0

    for i in range(0, len(subtrees), BATCH):
        batch = collate(subtrees[i:i + BATCH], dataset, SEED + i)
        obs = batch.observed_mask & batch.mask
        rows = ((~batch.observed_mask) & batch.mask).float().sum(1)
        uniform += float(((obs.sum(1) / batch.mask.sum(1).clamp_min(1)) * rows).sum())
        n_rows += float(rows.sum())
        for k, w in enumerate(rollout(model, batch)):
            acc[k] += observed_share(w, batch)
        print(f"  subtrees {i}-{i + BATCH}", flush=True)

    print(f"averaged over {n_rows:.0f} masked query rows")
    return pd.DataFrame([
        {"s": interpolant_time(k), "layer": l, "head": h,
         "on_observed": acc[k, l, h] / n_rows, "uniform": uniform / n_rows}
        for k in range(N_STEPS) for l in range(L) for h in range(H)
    ])


def example(model, dataset, subtree):
    batch = collate([subtree], dataset, SEED)
    n = int(batch.n_nodes[0])
    attn = np.stack([w[:, 0, :, :n, :n].cpu().numpy().astype(np.float32)
                     for w in rollout(model, batch)])       # (steps, L, H, n, n)
    return {
        "attn": attn,
        "s": interpolant_time(np.arange(N_STEPS)),
        "node_times": batch.node_times[0, :n].cpu().numpy(),
        "observed": (batch.observed_mask & batch.mask)[0, :n].cpu().numpy(),
    }


def main():
    os.chdir(QC.base_dir)     # the dataset's data_dir / cache paths are relative to the root
    os.makedirs(S.NR, exist_ok=True)
    torch.manual_seed(SEED)
    model, dataset = load_model()

    gamma = model.attention_bias_scale.detach().cpu().numpy()
    pd.DataFrame([{"layer": l, "head": h, "scale": float(v)}
                  for (l, h), v in np.ndenumerate(gamma)]
                 ).to_csv(os.path.join(S.NR, "scales.csv"), index=False)
    print(f"gamma: {(gamma < 0).sum()} of {gamma.size} negative, "
          f"range {gamma.min():.2f} to {gamma.max():.2f}")

    picks = pick_subtrees(dataset.trees, N_SUBTREES, np.random.default_rng(SEED))
    subtrees = [build_subtree_dict(dataset.trees[t], r) for t, r in picks]
    print(f"{len(subtrees)} subtrees from {len({t for t, _ in picks})} trees, "
          f"{np.median([s['size'] for s in subtrees]):.0f} leaves (median)")

    rollout_table(model, dataset, subtrees).to_csv(
        os.path.join(S.NR, "rollout.csv"), index=False)

    small = [s for s in subtrees if EXAMPLE_LEAVES[0] <= s["size"] <= EXAMPLE_LEAVES[1]]
    ex = example(model, dataset, (small or subtrees)[0])
    print(f"example subtree: {ex['attn'].shape[-1]} nodes, "
          f"{int(ex['observed'].sum())} observed")
    np.savez_compressed(os.path.join(S.NR, "example.npz"), **ex)
    print(f"wrote {S.NR}")


if __name__ == "__main__":
    main()
