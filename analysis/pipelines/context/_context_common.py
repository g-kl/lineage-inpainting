"""Shared helpers for the context_scaling_* experiments.

Load a bridge-tree-flow splitter, sample subtrees and focal nodes, anchor a set
of nodes (e.g. the observed leaves) and let the flow predict the remaining node
states with a seeded rollout, then optionally nearest-neighbor-annotate a
predicted state with a cell type.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import networkx as nx
import numpy as np
import torch
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[3]  # the repo root (analysis/pipelines/context/ is three levels down)
sys.path.insert(0, str(root))
os.chdir(root)  # so the dataset's relative data_dir / cache paths resolve

from datasets.tree_record import _leaf_counts, collate_subtrees

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A standard bridge-tree-flow (BridgeDeltaTreeFlow) run; override with env vars.
EXPERIMENT_DIR = Path(os.environ.get(
    "EXPERIMENT_DIR",
    str(root / "outputs"
        / "mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09"),
))
STEP = int(os.environ["STEP"]) if os.environ.get("STEP") else None  # None -> latest
FLOW_STEPS = 8  # multi-step flow rollout used for inference


# ------------------------------- loading -----------------------------------


def latest_split_step(exp_dir: Path) -> int:
    prefix, suffix = "split_ema_step", ".pt"
    steps = [
        int(f[len(prefix):-len(suffix)])
        for f in os.listdir(exp_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    if not steps:
        raise FileNotFoundError(f"no {prefix}*.pt in {exp_dir}")
    return max(steps)


def load_models():
    step = STEP if STEP is not None else latest_split_step(EXPERIMENT_DIR)
    print(f"checkpoint: {EXPERIMENT_DIR.name}  split_ema_step{step}  device={device}")
    cfg = OmegaConf.load(EXPERIMENT_DIR / "config.yaml")
    model = hydra.utils.instantiate(cfg.splitter).to(device).eval()
    model.load_state_dict(
        torch.load(EXPERIMENT_DIR / f"split_ema_step{step}.pt", map_location=device)
    )
    print("loading dataset (this can take a minute) ...")
    dataset = hydra.utils.instantiate(cfg.dataset)
    return cfg, model, dataset


# ------------------------------- subtrees ----------------------------------


def build_subtree_dict(tree: nx.DiGraph, root_node) -> dict:
    desc = nx.descendants(tree, root_node) | {root_node}
    sg = tree.subgraph(desc).copy()
    leaves = [n for n in sg if sg.out_degree(n) == 0]
    return {"graph": sg, "root": root_node, "leaves": leaves, "size": len(leaves)}


def pick_subtrees(trees, n, time_range, min_leaves, max_leaves, rng):
    """Random (tree_name, root) pairs whose subtree fits the size/time window."""
    t_lo, t_hi = float(time_range[0]), float(time_range[1])
    cands = []
    for tname, tree in trees.items():
        sizes = _leaf_counts(tree)
        for node in tree.nodes():
            if tree.out_degree(node) == 0:
                continue
            t = tree.nodes[node].get("time", float("nan"))
            if not (t_lo <= t <= t_hi):
                continue
            if min_leaves <= sizes[node] <= max_leaves:
                cands.append((tname, node))
    if not cands:
        return []
    idx = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    return [cands[int(i)] for i in idx]


def pick_focals_by_celltype(trees, name_to_ct, time_range, min_leaves, max_leaves,
                            per_type_target, rng):
    """Stratified focal-leaf selection balanced by *terminal* cell type.

    Like ``pick_subtrees`` we enumerate every (tree, root) whose subtree fits the
    size/time window, but instead of one uniform-random focal leaf per subtree we
    pool *all* candidate (subtree, leaf) pairs, label each by the leaf's ground-
    truth cell type (``name_to_ct``, a cheap dataset lookup -- no model), and then
    draw up to ``per_type_target`` focal leaves *per cell type*. This guarantees
    rare terminal fates (e.g. endothelial) are represented even though they are a
    small fraction of a uniform-random sample.

    Returns ``{(tree_name, root): [focal_leaf_name, ...]}`` -- focals grouped by
    subtree so each subtree batch / root annotation is built once and reused.
    """
    from collections import defaultdict

    t_lo, t_hi = float(time_range[0]), float(time_range[1])
    cand_by_ct = defaultdict(list)        # cell type -> [(tname, root, leaf), ...]
    for tname, tree in trees.items():
        sizes = _leaf_counts(tree)
        for node in tree.nodes():
            if tree.out_degree(node) == 0:
                continue
            t = tree.nodes[node].get("time", float("nan"))
            if not (t_lo <= t <= t_hi):
                continue
            if not (min_leaves <= sizes[node] <= max_leaves):
                continue
            for leaf in nx.descendants(tree, node):
                if tree.out_degree(leaf) != 0:
                    continue
                ct = name_to_ct.get(str(leaf))
                if ct is not None:
                    cand_by_ct[ct].append((tname, node, str(leaf)))

    selected = defaultdict(list)          # (tname, root) -> [focal leaf, ...]
    for ct, cands in cand_by_ct.items():
        k = min(per_type_target, len(cands))
        idx = rng.choice(len(cands), size=k, replace=False)
        for i in idx:
            tname, node, leaf = cands[int(i)]
            selected[(tname, node)].append(leaf)
    return selected


def _dominant_fate(leaf_names, name_to_ct):
    """Most common terminal cell type among ``leaf_names`` (None if all unknown)."""
    fates = [name_to_ct.get(str(l)) for l in leaf_names]
    fates = [f for f in fates if f is not None]
    return Counter(fates).most_common(1)[0][0] if fates else None


def pick_forward_focals(trees, name_to_ct, time_range, min_leaves, max_leaves,
                        min_desc, per_type_target, rng):
    """Stratified internal-node focal selection for the forward experiments.

    Returns ``{(tree_name, R): [v, ...]}`` over the same (tree, R) subtree
    population as the ancestral run; each candidate v is internal, has >= min_desc
    descendant leaves, has >= 1 cousin leaf under R, and is balanced (up to
    per_type_target) by the dominant terminal cell type among its descendants.
    """
    t_lo, t_hi = float(time_range[0]), float(time_range[1])
    cand_by_ct = defaultdict(list)        # desc fate -> [(tname, R, v), ...]
    for tname, tree in trees.items():
        sizes = _leaf_counts(tree)
        for R in tree.nodes():
            if tree.out_degree(R) == 0:
                continue
            t = tree.nodes[R].get("time", float("nan"))
            if not (t_lo <= t <= t_hi):
                continue
            if not (min_leaves <= sizes[R] <= max_leaves):
                continue
            for v in nx.descendants(tree, R):
                if tree.out_degree(v) == 0:          # v must be internal
                    continue
                if sizes[v] < min_desc:              # enough descendants to score
                    continue
                if sizes[R] - sizes[v] < 1:          # at least one cousin
                    continue
                desc_leaves = [n for n in nx.descendants(tree, v)
                               if tree.out_degree(n) == 0]
                ct = _dominant_fate(desc_leaves, name_to_ct)
                if ct is not None:
                    cand_by_ct[ct].append((tname, R, str(v)))

    selected = defaultdict(list)          # (tname, R) -> [v, ...]
    for ct, cands in cand_by_ct.items():
        k = min(per_type_target, len(cands))
        idx = rng.choice(len(cands), size=k, replace=False)
        for i in idx:
            tname, R, v = cands[int(i)]
            selected[(tname, R)].append(v)
    return selected


def make_subtree_batch(dataset, tree_name, root, max_nodes=128):
    """Collate one subtree; return (batch, n_active, names, leaf_mask)."""
    sd = build_subtree_dict(dataset.trees[tree_name], root)
    batch = collate_subtrees(
        [sd], dataset.leaf_states, device=device, max_nodes=max_nodes,
        randomize_child_index=False,
    )
    n = int(batch.n_nodes[0])
    names = [str(x) for x in batch.node_names[0][:n]]
    leaf_mask = batch.leaf_mask[0, :n].cpu().numpy().astype(bool)
    return sd, batch, n, names, leaf_mask


# ------------------------------ prediction ---------------------------------


@torch.no_grad()
def predict_states(model, batch, anchor_states, anchor_mask, seed, n_steps=FLOW_STEPS):
    """Seeded fill-in rollout. Returns predicted states (N, dim) for tree 0.

    ``seed`` fixes the prior noise, so two runs differ only by their anchors.
    """
    torch.manual_seed(seed)
    pred = model.split_layer_batch(batch, anchor_states, anchor_mask, n_steps=n_steps)
    return pred[0].float().cpu().numpy()


def patristic_matrix(batch, n_active):
    """(n, n) patristic distance t_i + t_j - 2 t_mrca(i, j) for tree 0."""
    nt = batch.node_times[0, :n_active].cpu().numpy()
    mrca = batch.mrca_times[0, :n_active, :n_active].cpu().numpy()
    return nt[:, None] + nt[None, :] - 2.0 * mrca


# --------------------------- cell-type annotation --------------------------


class NNAnnotator:
    """Nearest real cell (within a time window) -> cell type / lineage / germ layer."""

    def __init__(self, dataset, time_tol=0.5):
        obs = dataset.states.obs
        if "time" not in obs.columns:
            obs["time"] = [float(x.split("-")[0][1:]) for x in obs["embryo"].values]
        self.emb = np.ascontiguousarray(dataset.reference_features, dtype=np.float32)
        self.times = obs["time"].to_numpy(dtype=float)
        self.cell_type = obs["cell_type"].to_numpy()
        self.lineage = obs["lineage"].to_numpy()
        self.germ_layer = obs["germ_layer"].to_numpy()
        self.tol = float(time_tol)
        self._cache: dict = {}

    def _index(self, t):
        key = round(float(t), 1)
        if key not in self._cache:
            import faiss
            m = np.abs(self.times - key) <= self.tol
            if not m.any():
                self._cache[key] = None
            else:
                idx = faiss.IndexFlatL2(self.emb.shape[1])
                idx.add(self.emb[m])
                self._cache[key] = (idx, np.where(m)[0])
        return self._cache[key]

    def annotate(self, state, t):
        bundle = self._index(t)
        if bundle is None:
            return None, None, None
        idx, rows = bundle
        _, j = idx.search(np.ascontiguousarray(state[None], dtype=np.float32), 1)
        r = int(rows[int(j[0, 0])])
        return str(self.cell_type[r]), str(self.lineage[r]), str(self.germ_layer[r])
