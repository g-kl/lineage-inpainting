"""extend a perturbed lineage forward, then fill the new nodes' states.

there are two ways to grow the topology one step (cur -> target), selected by the
``topology`` argument of :func:`extend_layer`:

* "generated" -- the parametric extender.  the count sampler
  (``ConditionalCountSampler``) answers "how many active descendants does this
  cell have at ``time + delta``?"; we build a topology as close to a binary tree
  as possible (:func:`extenders.conditional.attach_skeleton`) and then CUT the
  terminal layer, carrying the penultimate nodes forward
  (:func:`grow_skeleton_cut`) so the boundary division is not counted twice as the
  lineage is extended step after step.
* "fixed" -- copy the true topology of the pooled/control tree, i.e. attach each
  cell's observed children verbatim (no count sampler involved).

either way the splitter then fills the new (masked) nodes' states conditioned on
the perturbed frontier as observed anchors.
"""

import itertools

import numpy as np
import torch

from datasets.tree_record import collate_subtrees
from extenders.conditional import attach_skeleton
from extenders.count import ConditionalCountSampler


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def load_count_sampler(model_path, device="cuda"):
    return ConditionalCountSampler(model_path=model_path, device=device)


def sample_counts(states, time, delta, sampler, rng, n_samples=32):
    """predicted child count per state at horizon ``delta`` (mean draw, >= 1)."""
    states = np.ascontiguousarray(states, dtype=np.float32)
    cond = {"state": states,
            "time": np.full(len(states), time, np.float32),
            "delta": np.full(len(states), delta, np.float32)}
    mean = sampler.sample_batch(cond, rng, n_samples=n_samples).mean(1)
    return np.maximum(1, np.rint(mean).astype(int))


def grow_skeleton(graph, parent, t_parent, t_end, K, name_gen):
    """grow a balanced-binary (+trifurcation) skeleton of ``K`` leaves beneath
    ``parent`` in ``graph``; returns the new leaf names (all at ``t_end``)."""
    return attach_skeleton(graph, parent, K, t_parent, t_end, name_gen)


def grow_skeleton_cut(graph, parent, t_parent, t_end, K, name_gen):
    """grow a ``K``-leaf skeleton beneath ``parent`` then CUT the terminal layer:
    the ``K`` leaves are removed and their parents (the penultimate nodes) are
    promoted to ``t_end`` and returned as the carried frontier.

    the count sampler answers "how many active descendants at ``t+delta``?", which
    counts each lineage at the boundary as a *post-division* cell.  if we then
    extend those leaves again, the boundary division is counted a second time (once
    as the leaf, once as that leaf dividing) -- the lineages compound a phantom
    division every step.  proceeding from the penultimate nodes instead realises
    the boundary division exactly once: the division that produced this step's
    leaves becomes the first division of the next step.

    ``K <= 3`` attaches the leaves directly to ``parent`` (no intermediate), so the
    cut collapses the whole step's division and the lineage carries forward as a
    single continuation node.  returns the carried node names (all at ``t_end``)."""
    leaves = attach_skeleton(graph, parent, K, t_parent, t_end, name_gen)
    if not leaves:
        return []
    penult, seen = [], set()
    for lf in leaves:
        p = next(iter(graph.predecessors(lf)))
        if p not in seen:
            seen.add(p)
            penult.append(p)
    graph.remove_nodes_from(leaves)
    carried = []
    for p in penult:
        if p == parent:                  # K<=3: leaves were direct children of the
            n = next(name_gen)           # frontier cell -> one continuation node
            graph.add_node(n, time=t_end)
            graph.add_edge(parent, n)
            carried.append(n)
        else:                            # promote the penultimate node to the layer
            graph.nodes[p]["time"] = t_end
            carried.append(p)
    return carried


def _copy_observed_children(built, tree, layer, pooled_states, name, target,
                            predict_target, child_pooled, predict):
    """attach ``name``'s observed children at ``target`` (from the control
    ``tree``) to ``built``; returns the new leaf names.

    ``predict`` records, per child, whether the splitter should fill its state
    (``True``) or whether the pooled/control state is copied verbatim
    (``False``, also stashing it in ``child_pooled``)."""
    leaves = []
    for c in tree.successors(name):
        if layer.get(c) == target and c in pooled_states:
            built.add_node(c, time=tree.nodes[c].get("time", target))
            built.add_edge(name, c)
            predict_target[c] = predict
            if not predict:
                child_pooled[c] = _to_numpy(pooled_states[c])
            leaves.append(c)
    return leaves


def extend_layer(built, frontier, tree, layer, pooled_states, cur, target,
                 topology, counter, count_sampler=None, rng=None,
                 n_count_samples=32):
    """attach each frontier cell's children for the step ``cur`` -> ``target``.

    ``topology``:
      "generated"  perturbed lineages (changed or descended from a change) grow a
                   synthetic skeleton sized by the count sampler; unchanged live
                   lineages copy their observed children.
      "fixed"      every lineage copies its observed children from the control
                   ``tree``; a perturbed lineage's children are still splitter-
                   predicted, an unchanged one's are copied from ``pooled_states``.

    returns ``(predict_target, child_pooled, children_of)``:
      ``predict_target[c]``  True if child ``c``'s state is splitter-predicted.
      ``child_pooled[c]``    copied control state for non-predicted children.
      ``children_of[name]``  the new leaf names attached beneath each frontier cell.
    """
    if topology not in ("generated", "fixed"):
        raise ValueError(f"topology must be generated/fixed, got {topology!r}")

    predict_target, child_pooled, children_of = {}, {}, {}

    if topology == "generated":
        synth = [f for f in frontier if f["changed"] or not f["live"]]
        counts = (sample_counts([f["kd_state"] for f in synth], cur, target - cur,
                                count_sampler, rng, n_samples=n_count_samples)
                  if synth else [])
        ci = 0
        for f in frontier:
            name = f["name"]
            if f["changed"] or not f["live"]:
                name_gen = (f"p{next(counter)}" for _ in itertools.count())
                leaves = grow_skeleton_cut(built, name, cur, target, int(counts[ci]), name_gen)
                ci += 1
                for c in leaves:
                    predict_target[c] = True
            else:
                leaves = _copy_observed_children(built, tree, layer, pooled_states,
                                                 name, target, predict_target,
                                                 child_pooled, predict=False)
            children_of[name] = leaves
        return predict_target, child_pooled, children_of

    # topology == "fixed": always copy the control tree's topology
    for f in frontier:
        name = f["name"]
        perturbed = f["changed"] or not f["live"]
        children_of[name] = _copy_observed_children(
            built, tree, layer, pooled_states, name, target,
            predict_target, child_pooled, predict=perturbed)
    return predict_target, child_pooled, children_of


@torch.no_grad()
def split_layer(splitter, contexts, device, n_steps=8, split_batch_size=16,
                max_nodes=256):
    """run the splitter over context subtrees -> {target_name: predicted_state}.

    each context: {"graph", "root", "anchors": {name: state}, "targets": [name]}.
    the anchors are the observed (perturbed) frontier cells; every other node in
    the graph -- reconstructed ancestors and the masked targets -- is filled by
    the flow conditioned on them and on the tree structure.
    """
    out = {}
    for s in range(0, len(contexts), split_batch_size):
        chunk = contexts[s:s + split_batch_size]
        leaf_states, subtrees = {}, []
        for ctx in chunk:
            for name, v in ctx["anchors"].items():
                leaf_states[name] = torch.as_tensor(np.asarray(v), dtype=torch.float32)
            subtrees.append({"graph": ctx["graph"], "root": ctx["root"]})
        batch = collate_subtrees(subtrees, leaf_states, device=device,
                                 max_nodes=max_nodes, randomize_child_index=False)
        pred = splitter.split_layer_batch(
            batch, batch.observed_states, batch.observed_mask, n_steps=n_steps,
        ).cpu().numpy()
        for b, ctx in enumerate(chunk):
            loc = {n: j for j, n in enumerate(batch.node_names[b])}
            for name in ctx["targets"]:
                if name in loc:
                    out[name] = pred[b, loc[name]]
    return out
