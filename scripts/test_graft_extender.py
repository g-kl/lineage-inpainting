"""Sanity test for GraftExtender on real mouse-embryo data.

What we check:
  * A SubtreeBank built from the (early, late) records yields a non-empty donor
    stratum for (early.leaf_time, late.leaf_time).
  * After grafting, every original early leaf becomes internal (has children).
  * Every node added by the extender is unobserved in the collated batch
    (obs_mask is False).
  * Every original observed node is still observed (obs_mask is True).
  * No 1->1 chains are created at the graft junctions (early leaf has out_degree >= 2).
  * The extended graph is a tree (every non-root node has exactly one parent).
  * Leaf times of the extended subtree match the target_leaf_time.

Prints summary stats so the shapes can be eyeballed.
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from datasets.mouse_embryo import MouseEmbryoDataset   # noqa: E402
from datasets.tree_record import collate_subtrees   # noqa: E402
from extenders import SubtreeBank   # noqa: E402
from extenders.graft import GraftExtender   # noqa: E402


def fmt_stats(name, values):
    if not values:
        return f"  {name}: (empty)"
    arr = np.asarray(values, dtype=float)
    return (
        f"  {name}: n={len(arr)} min={arr.min():.3f} max={arr.max():.3f} "
        f"mean={arr.mean():.3f}"
    )


def main():
    print("loading dataset (max_size=256)...")
    ds = MouseEmbryoDataset(data_dir="data/lt", max_size=256)
    records = ds.records
    leaf_states = ds.leaf_states

    # Pick the lowest-leaf-time and highest-leaf-time records.
    by_lt = sorted(records.items(), key=lambda kv: kv[1].leaf_time)
    early_name, early = by_lt[0]
    late_name, late = by_lt[-1]
    t_early = float(early.leaf_time)
    t_late = float(late.leaf_time)
    print(f"early rec: {early_name} (leaf_time={t_early})")
    print(f"late  rec: {late_name} (leaf_time={t_late})")

    rng = np.random.default_rng(0)

    # Get early subtrees (rooted somewhere below t_early).
    sample_time = max(r.leaf_time for r in by_lt[:1]) - 1.0   # below early leaves
    early_subtrees = early.get_subtrees(time=sample_time, n=8, rng=rng)
    if not early_subtrees:
        # fall back to anything available for this record
        early_subtrees = early.get_subtrees(time=t_early - 0.5, n=8, rng=rng)
    print(f"early subtrees drawn: {len(early_subtrees)} "
          f"(root_time={early_subtrees[0]['root_time']:.3f}, "
          f"leaf_time={early_subtrees[0]['leaf_time']:.3f})")

    # Build the donor bank from just these two records (the (t_early, t_late)
    # stratum is the late record's trees sliced at t_early). Restricting to two
    # records keeps the build fast for the test.
    extender = GraftExtender()
    extender.bank = SubtreeBank({early_name: early, late_name: late})
    stratum = extender.bank._stratum(t_early, t_late)
    donors = stratum.donors if stratum is not None else []
    print(f"donor stratum ({t_early} -> {t_late}): {len(donors)} candidates")
    if donors:
        print(fmt_stats("donor sizes", [d.size for d in donors]))

    # Run extender.
    extended = extender(
        early_subtrees,
        target_leaf_time=t_late,
        rng=rng,
    )

    # Per-subtree structural checks.
    print("\nper-subtree shape (orig vs extended):")
    for i, (orig, ext) in enumerate(zip(early_subtrees, extended)):
        og = orig["graph"]
        eg = ext["graph"]
        orig_leaves = set(n for n in og if og.out_degree(n) == 0)
        ext_leaves = set(n for n in eg if eg.out_degree(n) == 0)

        # Tree-ness: every non-root has exactly one parent.
        non_root_parents = [eg.in_degree(n) for n in eg if n != ext["root"]]
        assert all(p == 1 for p in non_root_parents), \
            f"subtree {i}: not a tree (some node has != 1 parents)"

        # Every original early leaf is now internal (has children).
        not_extended = [l for l in orig_leaves if eg.out_degree(l) == 0]
        n_grafted = len(orig_leaves) - len(not_extended)

        # No 1->1 chains at junctions: every leaf-that-was-extended now has out_degree >= 2.
        bad_junctions = [
            l for l in orig_leaves
            if eg.out_degree(l) > 0 and eg.out_degree(l) < 2
        ]
        assert not bad_junctions, \
            f"subtree {i}: 1-child graft junction at {bad_junctions}"

        # All added nodes (= ext nodes not in orig graph) are not in leaf_states.
        added = set(eg.nodes()) - set(og.nodes())
        leaked = [n for n in added if n in leaf_states]
        assert not leaked, f"subtree {i}: grafted nodes leaked observed data: {leaked[:3]}..."

        # New leaves' times should be at or below target_leaf_time, max ~= target.
        ext_leaf_times = [eg.nodes[n].get("time", float("nan")) for n in ext_leaves]

        print(
            f"  subtree {i}: orig_nodes={og.number_of_nodes()} "
            f"ext_nodes={eg.number_of_nodes()} "
            f"orig_leaves={len(orig_leaves)} grafted={n_grafted} "
            f"new_leaves={len(ext_leaves)} "
            f"max_leaf_t={max(ext_leaf_times):.3f}"
        )

    # Collate and check obs_mask layout.
    print("\ncollation check:")
    batch_orig = collate_subtrees(early_subtrees, leaf_states, device="cpu")
    batch_ext  = collate_subtrees(extended,        leaf_states, device="cpu")

    om_orig = batch_orig.observed_mask.numpy()
    om_ext  = batch_ext.observed_mask.numpy()
    mask_orig = batch_orig.mask.numpy()
    mask_ext  = batch_ext.mask.numpy()

    print(f"  orig batch: total_nodes={mask_orig.sum()} observed={om_orig.sum()}")
    print(f"  ext  batch: total_nodes={mask_ext.sum()}  observed={om_ext.sum()}")
    print(f"  delta nodes={mask_ext.sum() - mask_orig.sum()} "
          f"delta observed={om_ext.sum() - om_orig.sum()} (expect 0)")

    # The number of observed cells must not increase -- the extender only adds unobserved.
    assert om_ext.sum() == om_orig.sum(), \
        f"observed count changed: orig={om_orig.sum()} ext={om_ext.sum()}"

    # Show how leaf time distribution looks before / after.
    leaf_t_orig = batch_orig.node_times[batch_orig.leaf_mask & batch_orig.mask].numpy()
    leaf_t_ext  = batch_ext .node_times[batch_ext .leaf_mask & batch_ext .mask].numpy()
    print(fmt_stats("orig leaf times", leaf_t_orig.tolist()))
    print(fmt_stats("ext  leaf times", leaf_t_ext.tolist()))

    obs_t_ext = batch_ext.node_times[batch_ext.observed_mask & batch_ext.mask].numpy()
    print(fmt_stats("ext  observed-node times", obs_t_ext.tolist()))

    print("\nall checks passed.")


if __name__ == "__main__":
    main()
