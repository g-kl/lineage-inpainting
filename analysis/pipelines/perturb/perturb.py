"""assemble the perturbation pipeline.

per gene, per starting tree:

  1. build the faiss indices (all cells, gene-off cells, expression bins).
  2. start from the pooled tree -- internal node states sampled bottom-up by the
     splitter at the start time.
  3. snap the frontier to the nearest real cell and geometrically project it
     onto the perturbed expression level.
  4. step to the next reference timepoint along the observed tree's topology:
     children of lineages that did not change copy their pooled/observed state.
  5. children of lineages that did change get their states from the splitter,
     anchored at the projected parent -- this is the next frontier.
  6. repeat from 3 until the target time.

the topology is always the observed tree's (``topology="fixed"``).  once a lineage
is perturbed, all of its descendants' states are predicted by the splitter even
where the gene later reads as unchanged.  a control run (mode="control") never
changes anything, so every node copies its observed state and the result is the
observed tree.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import torch
import hydra

from datasets.tree_record import collate_subtrees, _snapshot_at
from utils.experiment_utils import get_experiment_info

from analysis.pipelines.perturb.projection import (
    build_time_indices, build_zero_indices, build_bin_indices,
    snap_to_nearest_batch, project_frontier,
)
from analysis.pipelines.perturb.extend import split_layer, _to_numpy
from analysis.pipelines.perturb.perturb_result import PerturbResult, is_off

OBS_TIMES = (7.5, 8.0, 8.5, 9.0, 9.5, 10.0)
EPS = 1e-6


# ---------------------------------------------------------------------------
# loading + setup
# ---------------------------------------------------------------------------

def load_model(experiment_dir):
    """load the (ema) pooler and splitter from an experiment dir -> (pooler, splitter, config)."""
    d = get_experiment_info(experiment_dir)
    config = d["config"]

    def _latest(tag):
        files = [x for x in os.listdir(d["dir"]) if tag in x and x.endswith(".pt")]
        return max(files, key=lambda x: int(x.split("_")[2][4:-3]))

    splitter = hydra.utils.instantiate(config["splitter"]).cuda()
    splitter.load_state_dict(torch.load(os.path.join(d["dir"], _latest("split_ema"))))
    pooler = hydra.utils.instantiate(config["pooler"]).cuda()
    pooler.load_state_dict(torch.load(os.path.join(d["dir"], _latest("pool_ema"))))
    return pooler, splitter, config


def build_time_indices_for(dataset, expr_layer="counts"):
    """gene-independent per-timepoint faiss indices over all cells (the expensive
    part).  build once and reuse across genes; ``build_zero_indices`` /
    ``build_bin_indices`` then add the cheap gene-specific indices."""
    states = dataset.states
    expr = states.layers[expr_layer] if expr_layer in states.layers else states.X
    return build_time_indices(
        np.asarray(dataset.reference_features, dtype=np.float32),
        states.obs_names.values, states.obs["time"].values.astype(float), expr,
    )


def build_indices(dataset, gene_idx, n_bins=4, expr_layer="counts"):
    """faiss indices for projecting one gene: all cells, gene-off cells, bins."""
    time_indices = build_time_indices_for(dataset, expr_layer)
    return {
        "time": time_indices,
        "zero": build_zero_indices(time_indices, gene_idx),
        "bins": build_bin_indices(time_indices, gene_idx, n_bins=n_bins),
    }


# ---------------------------------------------------------------------------
# bottom-up pooling: impute internal node states of the observed tree
# ---------------------------------------------------------------------------

def _nearest(values, t):
    return min(values, key=lambda x: abs(x - t))


def _clip_subtree(tree, root, has_state):
    """subtree under ``root`` clipped at nodes that already have a state."""
    nodes, leaves, stack = set(), [], [root]
    while stack:
        n = stack.pop()
        nodes.add(n)
        children = list(tree.successors(n))
        if not children:
            leaves.append(n)
        for c in children:
            if c in has_state or tree.out_degree(c) == 0:
                nodes.add(c)
                leaves.append(c)
            else:
                stack.append(c)
    return tree.subgraph(nodes).copy(), leaves


@torch.no_grad()
def impute_tree_states(pooler, tree, leaf_states, batch_size=256,
                       obs_times=OBS_TIMES, time_tol=0.5, n_steps=8, seed=0):
    """pool every internal node bottom-up, latest obs time to earliest -> {node: state}.

    ``pooler`` is the splitter used as the sampler (``n_steps`` bridge steps);
    seeded so every shard draws the same pooled states for a tree."""
    device = next(pooler.parameters()).device
    pooler.eval()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_times = sorted(obs_times)
    node_times = {n: tree.nodes[n].get("time", float("nan")) for n in tree.nodes()}

    by_time = defaultdict(list)
    for n in tree.nodes():
        if tree.out_degree(n) > 0:
            near = _nearest(obs_times, node_times[n])
            if abs(node_times[n] - near) <= time_tol:
                by_time[near].append(n)

    state = {n: leaf_states[n] for n in tree.nodes()
             if tree.out_degree(n) == 0 and n in leaf_states}

    for t in reversed(obs_times):
        roots = by_time.get(t, [])
        if not roots:
            continue
        has = set(state)
        subtrees = []
        for r in roots:
            sg, leaves = _clip_subtree(tree, r, has)
            if all(l in state for l in leaves):
                subtrees.append({"graph": sg, "root": r, "leaves": leaves})
        if not subtrees:
            continue
        leaf_tensors = {n: torch.as_tensor(np.asarray(s), dtype=torch.float32)
                        for n, s in state.items()}
        for i in range(0, len(subtrees), batch_size):
            chunk = subtrees[i:i + batch_size]
            batch = collate_subtrees(chunk, leaf_tensors, rng=rng).to(device)
            pooled = pooler.pool_batch(batch, n_steps=n_steps).cpu()
            for j, sd in enumerate(chunk):
                state[sd["root"]] = pooled[j, batch.root_idx[j].item()].numpy()
    return state


# ---------------------------------------------------------------------------
# observed-tree replay
# ---------------------------------------------------------------------------
#
# the perturbation replays the *observed* tree's topology, snapping every node to
# the nearest real cell at the nearest reference timepoint (the grid of times at
# which snap manifolds actually exist -- ``time_indices`` keys, e.g. 7.5, 8.0,
# 8.5, 8.75, 9.0, ...).  the topology it produces is a pure function of the
# observed tree and does NOT depend on the perturbation: control and knockout
# runs of the same tree have exactly the same nodes and edges.  only the *states*
# differ -- control copies the pooled/observed state and snaps it back to itself,
# a perturbed lineage's states are re-predicted by the splitter and re-projected
# (re-snapped) onto the perturbed expression level at every reference timepoint it
# crosses.  this is the "iterative re-snapping" the projection perturbation needs.

def _clade_from_seed(tree, seed, start_time, eps=EPS):
    """every node reachable from a seed node through descendants strictly past the
    ``start_time`` cut (``time > start_time``); the seed nodes themselves are the
    cross-section floors, so their pre-cut relatives belong to other seed nodes and
    are excluded -- the clades are leaf-disjoint (cf. ``_extract_subtree``)."""
    clade, stack = set(seed), []
    for s in seed:
        stack.extend(c for c in tree.successors(s)
                     if tree.nodes[c].get("time", float("inf")) > start_time + eps)
    while stack:
        n = stack.pop()
        if n in clade:
            continue
        clade.add(n)
        stack.extend(tree.successors(n))
    return clade


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_perturbation(tree, pooled_states, splitter, count_sampler, indices,
                     gene_idx, target_time, start_time=None, mode="knockout",
                     topology="fixed", target_fraction=0.0, ratio=3.0,
                     sample_time_gap=0.5, n_count_samples=32, n_steps=8,
                     split_batch_size=16, max_nodes=256, obs_times=OBS_TIMES,
                     time_tol=0.25, rng=None):
    """propagate the pooled tree forward under a projection perturbation.

    ``indices`` is the dict from :func:`build_indices`.  ``mode`` selects the
    projection:
      "control"  nothing moves -> reproduces the observed tree exactly.
      "knockout" project every expressing cell onto its nearest gene-off cell.
      "dose"     move each cell to ``expr * target_fraction`` and project onto the
                 nearest cell in that quantile bin (dose <1 -> knockdown, >1 -> up,
                 inf -> overexpression: force every cell, incl. gene-off, to the
                 top bin).

    only ``topology="fixed"`` is supported: the observed tree's topology is
    replayed verbatim, so a control run and any perturbed run of the same tree
    share an identical node/edge set (see the module note above).  the frontier
    marches over the reference-time grid (``indices["time"]`` keys) rather than a
    fixed 0.5 grid, and each observed node is snapped once at its nearest reference
    time -- nodes whose observed time falls between two reference times snap to the
    nearer one.  ``sample_time_gap``/``count_sampler``/``n_count_samples`` are
    accepted and ignored.  returns a :class:`PerturbResult`.
    """
    if mode not in ("control", "knockout", "dose"):
        raise ValueError(f"mode must be control/knockout/dose, got {mode!r}")
    if topology != "fixed":
        raise ValueError(
            f"topology must be 'fixed' (the observed topology is replayed), "
            f"got {topology!r}")
    rng = rng or np.random.default_rng()
    device = next(splitter.parameters()).device
    time_indices = indices["time"]
    ref_times = sorted(time_indices.keys())          # grid where snap manifolds exist

    def grid_of(t):
        return _nearest(ref_times, t)

    tt = {n: tree.nodes[n].get("time", float("nan")) for n in tree.nodes()}

    if start_time is None:
        start_time = min(ref_times)
    start_grid = grid_of(start_time)
    target_grid = grid_of(target_time)
    grid_seq = [t for t in ref_times if start_grid - EPS <= t <= target_grid + EPS]

    # seed = cross-section of the observed tree at start_time (each crossing branch
    # represented once by its floor node), restricted to the pooled nodes.
    seed = [n for n in _snapshot_at(tree, start_grid) if n in pooled_states]
    if not seed:
        raise ValueError(f"no pooled nodes at start_time={start_grid}")
    clade = _clade_from_seed(tree, seed, start_grid)

    built = nx.DiGraph()
    node_state = {}                      # name -> projected state
    info = {}                            # name -> {"live":, "changed":}

    def project_and_record(items, t):
        """snap each (name, raw_state) to the nearest real cell at reference time
        ``t``, project onto the perturbed expression level, and write the built
        node's attributes + ``node_state``/``info['changed']``."""
        ti = time_indices[t]
        raw = np.stack([s for _, s in items]).astype(np.float32)
        _, _, _, snapped = snap_to_nearest_batch(raw, ti)      # onto the real manifold
        zero_index = indices["zero"][t] if mode == "knockout" else None
        bin_index = indices["bins"][t] if mode == "dose" else None
        proj = project_frontier(
            np.ascontiguousarray(snapped, dtype=np.float32), ti, gene_idx,
            knockdown=mode != "control", zero_index=zero_index, bin_index=bin_index,
            target_fraction=target_fraction,
        )
        for i, (name, _) in enumerate(items):
            changed = bool(proj["changed"][i])
            ld = proj["local_dist"][i]
            ld = None if np.isnan(ld) else float(ld)
            node_state[name] = proj["kd_state"][i]
            info[name]["changed"] = changed
            built.nodes[name].update(
                snapped_name=proj["snapped"][i], changed=changed,
                proj_dist=float(proj["proj_dist"][i]), local_dist=ld,
                off_manifold=is_off(changed, float(proj["proj_dist"][i]), ld, ratio),
            )

    # --- seed layer ---
    for n in seed:
        built.add_node(n, time=tt[n])
        info[n] = {"live": True, "changed": False}
    project_and_record([(n, _to_numpy(pooled_states[n])) for n in seed], start_grid)

    # pending[parent] = observed clade children not yet attached; a node is "open"
    # while it still has children to attach (possibly across several grid layers).
    pending = {n: [c for c in tree.successors(n) if c in clade] for n in seed}
    open_nodes = {n for n in seed if pending[n]}

    # march the reference grid; each observed node is placed at its own nearest
    # reference time, so a lineage that skips a grid point is simply not re-snapped
    # there (its parent stays open until the grid reaches the child).
    for t in grid_seq:
        while True:                       # sub-steps: same-layer parent->child chains
            batch = []                    # (parent, child, perturbed)
            for parent in list(open_nodes):
                keep = []
                for c in pending[parent]:
                    if abs(grid_of(tt[c]) - t) < EPS:
                        perturbed = not (info[parent]["live"]
                                         and not info[parent]["changed"])
                        batch.append((parent, c, perturbed))
                    else:
                        keep.append(c)
                pending[parent] = keep
                if not keep:
                    open_nodes.discard(parent)
            if not batch:
                break

            # attach the new nodes and seed their bookkeeping / pending lists
            for parent, c, _ in batch:
                built.add_node(c, time=tt[c])
                built.add_edge(parent, c)
                info[c] = {"live": info[parent]["live"] and not info[parent]["changed"],
                           "changed": False}
                pending[c] = [k for k in tree.successors(c) if k in clade]
                if pending[c]:
                    open_nodes.add(c)

            # perturbed children: splitter predicts state from the (snapped) parent
            # anchor; unchanged children copy the pooled/observed state.  both are
            # then snapped + projected at ``t`` by project_and_record.
            to_predict = {p: [] for p, c, pert in batch if pert}
            for parent, c, pert in batch:
                if pert:
                    to_predict[parent].append(c)
            preds = _predict_children(splitter, built, node_state, to_predict, device,
                                      n_steps, split_batch_size, max_nodes)
            items = []
            for parent, c, pert in batch:
                if pert:
                    raw = preds.get(c)
                else:
                    raw = _to_numpy(pooled_states[c]) if c in pooled_states else None
                if raw is None:                    # missing pooled state / splitter
                    raw = node_state[parent]        # carry parent forward (safety net)
                items.append((c, raw))
            project_and_record(items, t)

    return _assemble(built, node_state, target_time)


def _predict_children(splitter, built, node_state, parent_to_children, device,
                      n_steps, split_batch_size, max_nodes):
    """splitter-predict the states of perturbed children, one context per parent:
    the parent is the observed (snapped) anchor and its children are the masked
    targets, exactly the cross-section-anchored split the splitter was trained on."""
    contexts = []
    for parent, children in parent_to_children.items():
        if not children:
            continue
        g = nx.DiGraph()
        g.add_node(parent, time=built.nodes[parent]["time"])
        for c in children:
            g.add_node(c, time=built.nodes[c]["time"])
            g.add_edge(parent, c)
        contexts.append({"graph": g, "root": parent,
                         "anchors": {parent: node_state[parent]}, "targets": children})
    if not contexts:
        return {}
    return split_layer(splitter, contexts, device, n_steps=n_steps,
                       split_batch_size=split_batch_size, max_nodes=max_nodes)


def _assemble(built, node_state, target_time):
    """collect terminal leaves into a PerturbResult."""
    rows, states = [], []
    for n in built.nodes():
        if built.out_degree(n) != 0 or n not in node_state:
            continue
        d = built.nodes[n]
        reason = "reached_target" if d["time"] >= target_time - EPS else "terminal_topology"
        d["reason"] = reason
        preds = list(built.predecessors(n))
        rows.append({
            "lineage_id": n, "node_name": n, "snapped_name": d.get("snapped_name"),
            "time": d["time"], "parent_lineage": preds[0] if preds else None,
            "changed": d.get("changed", False), "proj_dist": d.get("proj_dist"),
            "local_dist": d.get("local_dist"), "nn_dist": d.get("proj_dist"),
            "off_manifold": d.get("off_manifold", False), "reason": reason,
        })
        states.append(node_state[n])
    meta = pd.DataFrame(rows)
    arr = np.stack(states) if states else np.empty((0, 0), np.float32)
    return PerturbResult(meta=meta, states=arr, tree=built)
