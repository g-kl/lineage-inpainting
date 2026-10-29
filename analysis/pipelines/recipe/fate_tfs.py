#!/usr/bin/env python
"""fate-specifying TFs per cell type and developmental time, from a projection
knockout screen (control/*.pkl + genes/*.pkl from tf_projection_screen_ko).

for every control leaf and every knockout we find the divergence time -- the
control time of the first node on the root->leaf path the knockout changed
(pruned or re-snapped to a different cell) -- and round it to the nearest 0.5.
so each leaf gets, per time bin, the set of TFs that pushed it off course there.

aggregating over the leaves of a cell type gives, per (cell type, time bin):
  penetrance    fraction of that type's leaves the TF disrupts in that bin
  specificity   log2(penetrance / pooled-background penetrance of that TF+bin),
                i.e. how much more this TF hits this cell type than the average
  variability   mean Jaccard similarity of the TF sets of random pairs of that
                type's leaves disrupted in that bin (low = heterogeneous paths)

writes a long CSV of the TFs clearing a penetrance cutoff, ranked by specificity.

    python analysis/pipelines/recipe/fate_tfs.py
"""

import os
import sys
import glob
import pickle
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../../.."))
os.chdir(os.path.abspath(os.path.dirname(__file__) + "/../../.."))

from analysis.pipelines.recipe.divergence import (
    load_cell_type_map, load_control, first_divergence_time)

EXPERIMENT_DIR = ("outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_"
                  "nosa_std_none_57ab441ec3fcecf81d545755db5ddb09")


def round_half(t):
    return round(t * 2) / 2


def mean_pair_similarity(sets, n_pairs, rng):
    """mean Jaccard similarity over random pairs of TF sets; nan if < 2 sets."""
    if len(sets) < 2:
        return float("nan")
    i = rng.integers(0, len(sets), size=n_pairs)
    j = rng.integers(0, len(sets), size=n_pairs)
    sims = []
    for a, b in zip(i, j):
        if a == b:
            continue
        sa, sb = sets[a], sets[b]
        u = len(sa | sb)
        sims.append(len(sa & sb) / u if u else 1.0)
    return float(np.mean(sims)) if sims else float("nan")


def build(screen_dir, data_dir, level="cell_type"):
    name_to_ct = load_cell_type_map(data_dir, level)
    ctrl = load_control(os.path.join(screen_dir, "control"), name_to_ct)
    tfs = sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(screen_dir, "genes", "*.pkl")))
    print(f"{len(ctrl)} control trees, {len(tfs)} TFs", flush=True)

    # per global leaf (tree, node): {time bin -> set of disrupting TFs}; cell type
    leaf_bins = defaultdict(lambda: defaultdict(set))
    leaf_ct = {}
    ct_total = Counter()
    for tn, c in ctrl.items():
        for leaf, (ct, _) in c["leaves"].items():
            leaf_ct[(tn, leaf)] = ct
            ct_total[ct] += 1

    for i, tf in enumerate(tfs, 1):
        trees = pickle.load(open(os.path.join(screen_dir, "genes", f"{tf}.pkl"), "rb"))
        for tn, c in ctrl.items():
            if tn not in trees:
                continue
            ptree = trees[tn]["tree"]
            snap, tnode = c["snap"], c["time"]
            for leaf, (ct, path) in c["leaves"].items():
                t = first_divergence_time(path, snap, tnode, ptree)
                if t is not None:
                    leaf_bins[(tn, leaf)][round_half(t)].add(tf)
        if i % 50 == 0 or i == len(tfs):
            print(f"  {i}/{len(tfs)} TFs", flush=True)

    # disruption counts per (cell type, bin, TF) and pooled background per (TF, bin)
    counts = defaultdict(lambda: defaultdict(int))
    bg_hits = defaultdict(int)
    sets_by = defaultdict(list)                  # (cell type, bin) -> [TF set, ...]
    total_leaves = sum(ct_total.values())
    for leafid, bins in leaf_bins.items():
        ct = leaf_ct[leafid]
        for b, tfset in bins.items():
            sets_by[(ct, b)].append(tfset)
            for tf in tfset:
                counts[(ct, b)][tf] += 1
                bg_hits[(tf, b)] += 1
    return ct_total, counts, bg_hits, sets_by, total_leaves


def to_rows(ct_total, counts, bg_hits, sets_by, total_leaves,
            min_penetrance, n_pairs, seed):
    rng = np.random.default_rng(seed)
    variability = {key: mean_pair_similarity(sets, n_pairs, rng)
                   for key, sets in sets_by.items()}
    eps = 1.0 / total_leaves
    rows = []
    for (ct, b), tf_counts in counts.items():
        n_type = ct_total[ct]
        for tf, n in tf_counts.items():
            penetrance = n / n_type
            if penetrance < min_penetrance:
                continue
            background = bg_hits[(tf, b)] / total_leaves
            rows.append({
                "cell_type": ct,
                "time": b,
                "tf": tf,
                "penetrance": penetrance,
                "specificity": float(np.log2((penetrance + eps) / (background + eps))),
                "n_disrupted": n,
                "n_leaves": n_type,
                "variability": variability[(ct, b)],
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["cell_type", "time", "specificity"],
                          ascending=[True, True, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-dir",
                    default=os.path.join(EXPERIMENT_DIR, "tf_projection_screen_ko"))
    ap.add_argument("--data-dir", default="../data/embryos")
    ap.add_argument("--min-penetrance", type=float, default=0.05,
                    help="keep TFs disrupting at least this fraction of a cell "
                         "type's leaves in a time bin")
    ap.add_argument("--n-pairs", type=int, default=2000,
                    help="random leaf pairs sampled for the variability score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", default="cell_type",
                    help="grouping level (a final_type_info column, e.g. "
                         "cell_type or lineage); the group is the 'cell_type' column")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = "fate_tfs.csv" if args.level == "cell_type" else f"fate_tfs_{args.level}.csv"
    out = args.out or os.path.join(os.path.abspath(os.path.dirname(__file__)), name)
    built = build(args.screen_dir, args.data_dir, args.level)
    df = to_rows(*built, args.min_penetrance, args.n_pairs, args.seed)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, "
          f"{df['cell_type'].nunique()} {args.level}s", flush=True)
    print(df.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
