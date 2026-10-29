"""Context scaling, cycle-consistency variant: how well can the model *recover an
entire subtree's leaf set* from a partial leaf context, via a round trip?

We fix a subtree, pick a random focal leaf, and grow the revealed context by
climbing the focal leaf's ancestors (focal leaf, its parent's clade, its
grandparent's clade, ... up to the whole subtree). At each context level we run a
leaf -> root -> leaf *cycle*:

    up:   anchor the revealed leaves, predict the subtree root state.
    down: anchor ONLY the predicted root (leaves masked out), regenerate every leaf.

The recovered leaf cloud X is then the model's regenerated leaves -- no leaf is
copied from the truth, because the down pass anchors only the root (with a
*predicted* state) and masks all leaves. This avoids the trivial "cheat" where
pinned anchor leaves pass straight through the rollout. The compression through
the single root node is lossy, so the energy distance does NOT go to 0 even at
full context; it measures cycle consistency.

To keep the comparison meaningful across context sizes we always score the SAME,
full leaf set: the reference cloud Y is the true states at every leaf, and X is the
regenerated states at every leaf. Only the revealed context (hence the predicted
root, hence X) changes with context size.

The x-axis is the context size = leaf_time - mrca_time(revealed context). The
y-axis is the energy distance E(X, Y).

Output: a dataframe (subtree, focal leaf, context size, energy distance) + a plot.
Run with the ``lt`` conda env from the repo root.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

# _context_common is a sibling module imported by bare name; ensure this script's
# own directory is importable whether run as a script (its dir is already on the
# path) or imported as analysis.pipelines.context.context_scaling_cycle_consistency (it is not).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _context_common import (
    NNAnnotator, load_models, make_subtree_batch, pick_focals_by_celltype,
    predict_states, root,
)
from analysis.lib.celltype_map.eval import build_label_maps
# single source of truth for the energy distance (the unbiased U-statistic), shared
# with clonal_distortion: so the cycle-consistency numbers here are directly
# comparable to analysis/figures/summary_qc/compute_distortion.py.
from analysis.lib.metrics.clonal_preservation import _energy_distance as energy_distance

# config

TIME_RANGE = (7.0, 9.0)
MIN_LEAVES = 3
MAX_LEAVES = 20
MAX_NODES = 128
# Stratified sampling: draw up to this many focal leaves PER terminal cell type
# (the focal leaf's ground-truth fate), so rare fates such as endothelial are
# represented rather than swamped by a uniform-random draw. Types with fewer
# candidate leaves than this contribute all they have.
PER_TYPE_TARGET = 100
MAX_TIME_TO_MRCA = 1.5    # cap on context size (leaf_time - mrca_time); = summary_qc style.CONTEXT_BIN_EDGES[-1]
SEED = 0

OUT_DIR = root / "analysis" / "results" / "context_scaling_cycle_consistency"


def main(out_dir=OUT_DIR, csv_name="context_scaling_cycle_consistency.csv", plots=True):
    """Compute the cycle-consistency cache.

    ``out_dir`` / ``csv_name`` let a caller (e.g. the figure's
    ``compute_context_scaling.py``) write the cache straight into its own
    numerical_results folder instead of the default analysis/results location;
    ``plots=False`` skips the standalone diagnostic PNGs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, model, dataset = load_models()
    nn = NNAnnotator(dataset)

    # cheap dataset lookup: focal leaf name -> ground-truth terminal cell type.
    name_to_labels, _ = build_label_maps(dataset)
    name_to_ct = {n: d.get("cell_type") for n, d in name_to_labels.items()}

    rng = np.random.default_rng(SEED)
    # {(tree, root): [focal leaf, ...]}, balanced to >= PER_TYPE_TARGET focals per
    # terminal cell type (capped by availability), grouped so each subtree's batch
    # and root annotation are built once.
    selected = pick_focals_by_celltype(
        dataset.trees, name_to_ct, TIME_RANGE, MIN_LEAVES, MAX_LEAVES,
        PER_TYPE_TARGET, rng,
    )
    if not selected:
        raise RuntimeError("no subtrees matched; widen TIME_RANGE / leaf bounds")
    picks = list(selected.items())
    n_focals = sum(len(v) for v in selected.values())
    type_counts = Counter(
        name_to_ct.get(f) for fs in selected.values() for f in fs)
    print(f"picked {len(picks)} subtrees, {n_focals} focal leaves across "
          f"{len(type_counts)} terminal cell types (target {PER_TYPE_TARGET}/type)")
    for ct, c in type_counts.most_common():
        flag = "" if c >= PER_TYPE_TARGET else "  (< target: candidate-limited)"
        print(f"    {str(ct):<32s} {c:>5d}{flag}")

    rows = []
    for si, ((tname, rootnode), focal_names) in enumerate(picks):
        sd, batch, n, names, leaf_mask = make_subtree_batch(
            dataset, tname, rootnode, max_nodes=MAX_NODES,
        )
        _, N = batch.mask.shape
        graph = sd["graph"]
        anchor_states = batch.observed_states
        node_times = batch.node_times[0, :n].cpu().numpy()
        mrca = batch.mrca_times[0, :n, :n].cpu().numpy()
        name2idx = {nm: i for i, nm in enumerate(names)}
        leaf_idxs = np.where(leaf_mask)[0]
        leaf_names = [names[i] for i in leaf_idxs]
        r = int(batch.root_idx[0])           # bottleneck node for the cycle
        seed = SEED + si

        # reference leaf cloud Y: true states at every leaf (fixed for this subtree)
        actual_leaves = anchor_states[0, leaf_idxs].float().cpu().numpy()

        if si % 100 == 0:
            print(f"[{si}/{len(picks)}] {tname} root={rootnode} leaves={len(leaf_idxs)}")

        def anchor_mask_for(idxs):
            m = torch.zeros(1, N, dtype=torch.bool, device=batch.mask.device)
            m[0, list(idxs)] = True
            return m

        # cell-type label for the subtree: annotate the full-context root prediction
        p_ref = predict_states(model, batch, anchor_states, anchor_mask_for(leaf_idxs), seed)
        cell_type, _, _ = nn.annotate(p_ref[r], float(node_times[r]))

        for f_name in focal_names:
            focal_idx = int(name2idx[f_name])
            f = names[focal_idx]
            leaf_time = float(node_times[focal_idx])

            # ancestors from the focal leaf up to the root (clade levels)
            ancestors, cur = [], f
            while True:
                ancestors.append(cur)
                if cur == rootnode:
                    break
                preds = list(graph.predecessors(cur))
                if not preds:
                    break
                cur = preds[0]

            for step, A in enumerate(ancestors):
                if A == f:
                    revealed = [focal_idx]
                else:
                    clade = nx.descendants(graph, A)
                    revealed = [name2idx[l] for l in leaf_names if l in clade]
                mrca_time = float(mrca[np.ix_(revealed, revealed)].min())
                # context grows monotonically up the clade; stop once it exceeds cap
                if leaf_time - mrca_time > MAX_TIME_TO_MRCA:
                    break
                # up: predict the root from the revealed leaf context
                p_up = predict_states(
                    model, batch, anchor_states, anchor_mask_for(revealed), seed,
                )
                root_state = torch.from_numpy(p_up[r]).to(anchor_states)
                # down: anchor ONLY the predicted root, mask all leaves, regenerate
                down_anchor = torch.zeros_like(anchor_states)
                down_anchor[0, r] = root_state
                p_down = predict_states(
                    model, batch, down_anchor, anchor_mask_for([r]), seed,
                )
                recovered_leaves = p_down[leaf_idxs]
                rows.append(dict(
                    tree=tname, subtree_idx=si, focal_leaf=f, root_node=rootnode,
                    cell_type=cell_type, leaf_cell_type=name_to_ct.get(f),
                    step=step, n_revealed=len(revealed), n_leaves=len(leaf_idxs),
                    leaf_time=leaf_time, mrca_time=mrca_time,
                    time_to_mrca=leaf_time - mrca_time,
                    energy_distance=energy_distance(recovered_leaves, actual_leaves),
                ))

    df = pd.DataFrame(rows)
    csv_path = out_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}  ({len(df)} rows)")

    if plots:
        plot_box(df, out_dir / "context_scaling_cycle_consistency.png")
        plot_scatter(df, out_dir / "context_scaling_cycle_consistency_scatter.png")
        plot_cell_type(df, out_dir / "context_scaling_cycle_consistency_cell_type.png")
    return df


def plot_box(df, out_path):
    bins = np.linspace(0, MAX_TIME_TO_MRCA, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.6
    df = df.assign(t_bin=pd.cut(df["time_to_mrca"], bins, labels=centers,
                                include_lowest=True).astype(float))

    fig, ax = plt.subplots(figsize=(8, 5))
    keep = [(c, df.loc[df["t_bin"] == c, "energy_distance"].to_numpy()) for c in centers]
    keep = [(c, d) for c, d in keep if len(d)]
    ax.boxplot([d for _, d in keep], positions=[c for c, _ in keep],
               widths=width, showfliers=False)
    medians = [np.median(d) for _, d in keep]
    ax.plot([c for c, _ in keep], medians, color="tab:red", lw=1.5, marker="o")
    ax.set_xlabel("context size = time from focal leaf to MRCA of revealed context")
    ax.set_ylabel("energy distance: recovered vs. true leaf set")
    ax.set_title("subtree recovery (cycle consistency) vs. context size")
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{c:.2f}" for c in centers])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_scatter(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["time_to_mrca"], df["energy_distance"], s=10, alpha=0.3,
               color="tab:blue", edgecolors="none")
    ax.set_xlabel("context size = time from focal leaf to MRCA of revealed context")
    ax.set_ylabel("energy distance: recovered vs. true leaf set")
    ax.set_title("subtree recovery (cycle consistency) vs. context size")
    ax.set_xlim(0, MAX_TIME_TO_MRCA)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_cell_type(df, out_path, min_per_bin=4):
    bins = np.linspace(0, MAX_TIME_TO_MRCA, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    df = df.assign(t_bin=pd.cut(df["time_to_mrca"], bins, labels=centers,
                                include_lowest=True).astype(float))
    g = df.dropna(subset=["cell_type"]).groupby(["cell_type", "t_bin"])["energy_distance"]
    count = g.count().unstack("t_bin").reindex(columns=centers)
    mean = g.mean().unstack("t_bin").reindex(columns=centers)
    sem = (g.std() / np.sqrt(g.count())).unstack("t_bin").reindex(columns=centers)
    # only keep cell types with > min_per_bin instances in EVERY bin (no gaps)
    keep = (count > min_per_bin).all(axis=1)
    mean, sem = mean[keep], sem[keep]

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab20")
    for i, ct in enumerate(mean.index):
        ax.errorbar(centers, mean.loc[ct].to_numpy(), yerr=sem.loc[ct].to_numpy(),
                    marker="o", lw=1.5, capsize=2, color=cmap(i % 20), label=str(ct))
    ax.set_xlabel("context size = time from focal leaf to MRCA of revealed context")
    ax.set_ylabel("mean energy distance: recovered vs. true leaf set")
    ax.set_title(f"subtree recovery (cycle consistency) by cell type (>{min_per_bin}/bin)")
    ax.set_xlim(0, MAX_TIME_TO_MRCA)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
