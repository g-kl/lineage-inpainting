"""Forward context scaling: how much does revealing more cousins help predict a
focal node's descendants?

For a focal internal node v we predict v's descendant leaves by splitting DOWN from
a cross-section frontier (a "layer") above it -- the splitter only ever predicts
downward, never side to side, so context has to enter through the conditioning layer.
The layer is a complete ``cross_section`` antichain (one node per lineage, no holes,
exactly the slices the trainer anchors), so the conditioning is always in
distribution. It crosses v's lineage at exactly one node f_v; the OTHER frontier
nodes are v's same-level cousins. Sweeping the cut from near the root (a coarse
layer, few nodes) to near the leaves (a fine layer, many nodes) varies how many
cousins the layer reveals.

The x-axis is the number of cousins revealed, n_frontier - 1 (0 = f_v alone, the
root-only layer). Energy distance to v's true descendant cloud FALLS as more cousins
are added -- and the layer's position relative to v (v_time - f_time) is ~flat across
the bulk of that range, so the gain is the added cousin context, not the layer
sitting closer to v.

Setup per subtree:
    pool:  anchor all observed leaves -> predict every internal node's state (the
           pooled estimates inference cannot observe directly).
    split: for each distinct frontier, anchor the pooled states at exactly the
           frontier nodes (a complete cross-section), mask the rest, regenerate the
           leaves. Reused across every focal v whose lineage that frontier crosses.

Recovered cloud X = v's regenerated descendant leaves; reference Y = their true
observed states; the score is the energy distance E(X, Y).

Output: a dataframe (subtree, focal node, frontier rep, cousins revealed, energy
distance) + a plot. Run with the ``lt`` conda env from the repo root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

# _context_common is a sibling module imported by bare name; ensure this script's
# own directory is importable whether run as a script or imported as a package module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _context_common import (
    _dominant_fate, load_models, make_subtree_batch, pick_forward_focals,
    predict_states, root,
)
from analysis.lib.celltype_map.eval import build_label_maps
from datasets.tree_record import cross_section
# single source of truth for the energy distance, shared with the ancestral variant.
from analysis.lib.metrics.clonal_preservation import _energy_distance as energy_distance

# config (kept parallel to context_scaling_cycle_consistency)

TIME_RANGE = (7.0, 9.0)   # window for the SUBTREE ROOT R (same population as ancestral)
MIN_LEAVES = 6
MAX_LEAVES = 20
MIN_DESC_LEAVES = 3       # v must have >= this many descendant leaves to score E(X,Y)
MAX_NODES = 128
PER_TYPE_TARGET = 100     # focal NODES per dominant descendant fate (stratified)
MAX_TIME_TO_MRCA = 1.0    # cap on layer position (v_time - frontier-rep time)
N_CUTS = 60               # cut times swept to enumerate distinct frontiers (deduped)
SEED = 0

OUT_DIR = root / "analysis" / "results" / "context_scaling_forward_layer"


def enumerate_frontiers(batch, n):
    """Distinct ``cross_section`` antichains as the cut sweeps the subtree's span.

    Returns ``[frozenset(frontier_idxs), ...]`` deduped by node set, coarse (near
    root) to fine (near leaves).
    """
    nt = batch.node_times[0, :n]
    lo, hi = float(nt.min()), float(nt.max())
    seen, out = set(), []
    for cut in np.linspace(lo, hi, N_CUTS):
        fr = cross_section(batch, float(cut), direction="up")[0, :n].cpu().numpy()
        idxs = frozenset(np.where(fr)[0].tolist())
        if idxs and idxs not in seen:
            seen.add(idxs)
            out.append(idxs)
    return out


def main(out_dir=OUT_DIR, csv_name="context_scaling_forward_layer.csv", plots=True):
    """Compute the forward context-scaling cache.

    ``out_dir`` / ``csv_name`` let a caller write the cache into its own folder;
    ``plots=False`` skips the standalone diagnostic PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, model, dataset = load_models()

    name_to_labels, _ = build_label_maps(dataset)
    name_to_ct = {n: d.get("cell_type") for n, d in name_to_labels.items()}

    rng = np.random.default_rng(SEED)
    selected = pick_forward_focals(
        dataset.trees, name_to_ct, TIME_RANGE, MIN_LEAVES, MAX_LEAVES,
        MIN_DESC_LEAVES, PER_TYPE_TARGET, rng,
    )
    if not selected:
        raise RuntimeError("no focal nodes matched; widen TIME_RANGE / leaf bounds")
    picks = list(selected.items())
    n_focals = sum(len(v) for v in selected.values())
    print(f"picked {len(picks)} subtrees, {n_focals} focal internal nodes "
          f"(target {PER_TYPE_TARGET}/dominant-descendant-fate)")

    rows = []
    for si, ((tname, rootnode), focal_nodes) in enumerate(picks):
        sd, batch, n, names, leaf_mask = make_subtree_batch(
            dataset, tname, rootnode, max_nodes=MAX_NODES,
        )
        _, N = batch.mask.shape
        graph = sd["graph"]
        anchor_states = batch.observed_states
        node_times = batch.node_times[0, :n].cpu().numpy()
        name2idx = {nm: i for i, nm in enumerate(names)}
        leaf_idxs = np.where(leaf_mask)[0]
        leaf_names = [names[i] for i in leaf_idxs]
        seed = SEED + si

        if si % 100 == 0:
            print(f"[{si}/{len(picks)}] {tname} root={rootnode} "
                  f"leaves={len(leaf_idxs)} focals={len(focal_nodes)}")

        def anchor_mask_for(idxs):
            m = torch.zeros(1, N, dtype=torch.bool, device=batch.mask.device)
            m[0, list(idxs)] = True
            return m

        # POOL: anchor all observed leaves, read every internal node's pooled state.
        pooled = predict_states(model, batch, anchor_states,
                                anchor_mask_for(leaf_idxs), seed)

        # SPLIT once per distinct frontier: anchor the pooled states at exactly the
        # frontier nodes (a complete cross-section), mask the rest, regenerate leaves.
        frontier_pred = []   # (frontier_idxs, predicted_states)
        for idxs in enumerate_frontiers(batch, n):
            down_anchor = torch.zeros_like(anchor_states)
            for j in idxs:
                down_anchor[0, j] = torch.from_numpy(pooled[j]).to(anchor_states)
            pred = predict_states(model, batch, down_anchor,
                                  anchor_mask_for(list(idxs)), seed)
            frontier_pred.append((idxs, pred))

        for v_name in focal_nodes:
            if v_name not in name2idx:
                continue
            v_idx = int(name2idx[v_name])
            v_time = float(node_times[v_idx])

            v_clade = nx.descendants(graph, v_name)
            desc_names = [l for l in leaf_names if l in v_clade]
            if len(desc_names) < MIN_DESC_LEAVES:
                continue
            desc_idxs = [name2idx[l] for l in desc_names]
            true_desc = anchor_states[0, desc_idxs].float().cpu().numpy()
            desc_ct = _dominant_fate(desc_names, name_to_ct)

            # strict ancestors of v, parent -> root (candidate frontier reps)
            ancestors, cur = [], v_name
            while cur != rootnode:
                cur = list(graph.predecessors(cur))[0]
                ancestors.append(cur)
            anc_idx = {name2idx[a]: a for a in ancestors}

            for idxs, pred in frontier_pred:
                # the antichain crosses v's lineage at exactly one strict ancestor f_v;
                # every other frontier node is a same-level cousin of v.
                reps = [a_i for a_i in anc_idx if a_i in idxs]
                if not reps:
                    continue
                f_idx = reps[0]
                f_time = float(node_times[f_idx])
                if v_time - f_time > MAX_TIME_TO_MRCA:
                    continue
                rows.append(dict(
                    tree=tname, subtree_idx=si, focal_node=v_name,
                    frontier_rep=anc_idx[f_idx], root_node=rootnode,
                    desc_cell_type=desc_ct, n_frontier=len(idxs),
                    n_cousins=len(idxs) - 1, n_desc=len(desc_idxs),
                    v_time=v_time, mrca_time=f_time, time_to_mrca=v_time - f_time,
                    energy_distance=energy_distance(pred[desc_idxs], true_desc),
                ))

    df = pd.DataFrame(rows)
    csv_path = out_dir / csv_name
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}  ({len(df)} rows)")

    if plots:
        plot_scaling(df, out_dir / "context_scaling_forward_layer.png")
    return df


def plot_scaling(df, out_path, max_cousins=10):
    """Median energy distance (+ IQR) vs. number of cousins the layer reveals."""
    k = df["n_cousins"].clip(upper=max_cousins)
    xs = np.arange(max_cousins + 1)
    med = [np.median(df.loc[k == x, "energy_distance"]) for x in xs]
    q1 = [np.percentile(df.loc[k == x, "energy_distance"], 25) for x in xs]
    q3 = [np.percentile(df.loc[k == x, "energy_distance"], 75) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(xs, q1, q3, color="tab:blue", alpha=0.16, lw=0)
    ax.plot(xs, med, color="tab:blue", lw=1.6, marker="o")
    ax.set_xlabel("cousins revealed by the layer")
    ax.set_ylabel("energy distance: recovered vs. true descendant set")
    ax.set_title("forward prediction: more cousin context lowers energy distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
