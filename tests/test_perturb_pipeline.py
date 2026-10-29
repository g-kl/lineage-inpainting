#!/usr/bin/env python
#SBATCH --job-name=test_perturb_pipeline
#SBATCH --output=logs/test_perturb_pipeline_%j.log
#SBATCH --error=logs/test_perturb_pipeline_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=bates
"""tests for the projection perturbation pipeline (observed-tree replay).

run with sbatch (needs one GPU for the pooler/splitter):
    sbatch tests/test_perturb_pipeline.py
or directly on a GPU node:
    python tests/test_perturb_pipeline.py

the perturbation replays the observed tree's topology and snaps every node to
the nearest real cell at the nearest reference timepoint, so the topology is a
pure function of the observed tree -- a control run and any knockout run of the
same tree have an *identical* node/edge set; only the states differ.

checks:
  1. the knockout tree has exactly the same node count as the control tree
     (topology is mode-independent), and both recover most of the observed
     tree's leaves (no binning dropout).
  2. with the off-manifold ratio at 0, the off-manifold set is exactly the
     changed nodes plus their descendants; and at the start layer the changed
     flag matches the cells that truly express the gene (read from control).
  3. a dose of 0 sends the gene to 0.
  4. a dose of 0.5 sends the gene to roughly 0.5 of its original level on average.
  5. an Nkx2-5 knockout changes mostly heart cells, and applying the off-manifold
     cutoff depletes heart leaves far more than non-heart leaves.

"""

import os
import sys

base_dir = os.path.abspath(os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import numpy as np
import networkx as nx

from datasets.mouse_embryo import MouseEmbryoDataset
from analysis.pipelines.perturb.perturb import (
    load_model, build_time_indices_for, impute_tree_states, run_perturbation)
from analysis.pipelines.perturb.projection import (
    build_zero_indices, build_bin_indices, project_frontier, _gene_column)
from analysis.pipelines.perturb.perturb_result import off_manifold_descendants


## CONFIG ##

experiment_dir = "outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09"
replicate = "R1"
start_time = 7.5
target_time = 9.0
tree_leaf_time = target_time
obs_times = (7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0)
n_steps = 32                 # splitter integration steps per 0.5 step
n_dose_bins = 4
seed = 42

heart_gene = "Nkx2-5"
heart_types = {"first heart field", "second heart field", "juxtacardiac field"}

# tolerances
leaf_recovery_min = 0.9      # built leaves / observed-tree leaves (no binning loss)
off_manifold_ratio = 1.1     # cutoff swept post-hoc to define "depleted" leaves
dose_half = 0.5
dose_half_band = (0.3, 0.7)  # acceptable mean(new/old) for a 0.5 dose
heart_change_rate_min = 0.3  # fraction of heart-cell nodes flagged changed
nonheart_change_rate_max = 0.15
heart_survival_max = 0.7     # <=70% of heart leaves survive the off-manifold cut
heart_depletion_margin = 2.0 # heart depleted >= 2x more than non-heart


## HARNESS ##

_failures = []

def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        _failures.append(name)


## SETUP (loaded once) ##

print("loading dataset...", flush=True)
dataset = MouseEmbryoDataset()
states = dataset.states
var_to_col = {g: i for i, g in enumerate(states.var_names)}
gene_idx = var_to_col[heart_gene]

# true gene counts and cell types per real cell (aligned to states rows)
gene_counts = states.layers["counts"][:, gene_idx]
gene_counts = np.asarray(gene_counts.todense()).ravel() if hasattr(gene_counts, "todense") \
    else np.asarray(gene_counts).ravel()
name_to_idx = {n: i for i, n in enumerate(states.obs_names)}
name_to_type = dict(zip(states.obs_names, states.obs["cell_type"]))

print("loading pooler/splitter...", flush=True)
pooler, splitter, config = load_model(experiment_dir)

print("building per-timepoint FAISS indices...", flush=True)
time_indices = build_time_indices_for(dataset)


## PICK A SMALL STARTING TREE AND POOL IT ##

candidates = sorted(
    [n for n in dataset.trees if f"E{tree_leaf_time}-{replicate}-" in n],
    key=lambda n: dataset.trees[n].number_of_nodes())
if not candidates:
    raise SystemExit(f"no E{tree_leaf_time}-{replicate} starting trees found")


def perturb(tree, pooled, mode, indices, target_fraction=0.0, ratio=1.0):
    return run_perturbation(
        tree, pooled, splitter, None, indices, gene_idx, target_time,
        start_time=start_time, mode=mode, topology="fixed",
        target_fraction=target_fraction, ratio=ratio, obs_times=obs_times,
        n_steps=n_steps, rng=np.random.default_rng(seed))


tree_name = None
for cand in candidates:
    tree = dataset.trees[cand]
    print(f"pooling {cand} ({tree.number_of_nodes()} nodes)...", flush=True)
    pooled = impute_tree_states(splitter, tree, dataset.leaf_states,
                                batch_size=1024, obs_times=obs_times)
    try:
        control = perturb(tree, pooled, "control", {"time": time_indices})
    except ValueError as e:
        print(f"  skipping {cand}: {e}", flush=True)
        continue
    tree_name = cand
    break
if tree_name is None:
    raise SystemExit("no starting tree had a usable start-time layer")
print(f"using starting tree {tree_name}", flush=True)

print(f"running {heart_gene} knockout...", flush=True)
ko_indices = {"time": time_indices, "zero": build_zero_indices(time_indices, gene_idx)}
ko = perturb(tree, pooled, "knockout", ko_indices)


## TEST 1: knockout tree is topologically identical to control ##

print("\ntest 1: knockout tree == control tree topology; leaves recovered")
n_ctrl = control.tree.number_of_nodes()
n_ko = ko.tree.number_of_nodes()
check("knockout has exactly the same node count as control", n_ctrl == n_ko,
      f"control={n_ctrl} knockout={n_ko}")
check("knockout and control share an identical node set",
      set(control.tree.nodes) == set(ko.tree.nodes),
      f"symmetric_difference={len(set(control.tree.nodes) ^ set(ko.tree.nodes))}")

# the replay should keep (almost) every observed leaf -- no binning dropout. the
# observed tree has all leaves at the target time; the replay's leaves are the
# clade below the start-time cross-section, so compare against that.
obs_leaves = {n for n in tree.nodes if tree.out_degree(n) == 0
              and abs(tree.nodes[n].get("time", -1) - target_time) < 0.3}
built_leaves = {n for n in control.tree.nodes if control.tree.out_degree(n) == 0
                and abs(control.tree.nodes[n].get("time", -1) - target_time) < 0.3}
recovered = len(built_leaves & obs_leaves) / max(len(obs_leaves), 1)
check("control recovers most observed leaves", recovered >= leaf_recovery_min,
      f"built={len(built_leaves)} observed={len(obs_leaves)} recovered={recovered:.3f}")


## TEST 2: ratio=0 off-manifold == changed nodes + descendants ##

print("\ntest 2: off-manifold at ratio 0 matches the expressing nodes")
changed = {n for n in ko.tree.nodes
           if ko.tree.nodes[n].get("changed")
           and ko.tree.nodes[n].get("proj_dist", 0.0) > 0.0}
expected = set(changed)
for n in changed:
    expected |= nx.descendants(ko.tree, n)
got = off_manifold_descendants(ko.tree, 0.0)
check("ratio=0 off-manifold == changed nodes + descendants", got == expected,
      f"expected={len(expected)} got={len(got)}")

# cross-check the changed flag against true expression at the seed (the start-time
# cross-section = the tree roots), where knockout and control states are identical
# (both come straight from pooling, snapped to the same real cell).
mismatch = 0
n_start = 0
for n in control.tree.nodes:
    if control.tree.in_degree(n) != 0:        # seed nodes are the replay's roots
        continue
    if n not in ko.tree.nodes:
        continue
    sn = control.tree.nodes[n].get("snapped_name")
    if sn is None or sn not in name_to_idx:
        continue
    n_start += 1
    expresses = gene_counts[name_to_idx[sn]] > 0
    if bool(ko.tree.nodes[n].get("changed")) != bool(expresses):
        mismatch += 1
check("seed changed flag matches true expression", mismatch == 0,
      f"seed_nodes={n_start} mismatches={mismatch}")


## TEST 3 & 4: dose projection at the cell level ##

print("\ntests 3 & 4: dose projection drives expression to the dosed level")
# pick the timepoint with the most cells expressing the gene
t_best = max(time_indices, key=lambda t: int((_gene_column(time_indices[t], gene_idx) > 0).sum()))
ti = time_indices[t_best]
col = _gene_column(ti, gene_idx)
expr_rows = np.where(col > 0)[0]
rng = np.random.default_rng(seed)
sel = rng.choice(expr_rows, min(512, len(expr_rows)), replace=False)
states_q = ti["embeddings"][sel]
old_expr = col[sel]
row_of = {nm: i for i, nm in enumerate(ti["names"])}
bins = build_bin_indices(time_indices, gene_idx, n_bins=n_dose_bins)


def dose_new_expr(target_fraction):
    proj = project_frontier(states_q, ti, gene_idx, knockdown=True,
                            bin_index=bins[t_best], target_fraction=target_fraction)
    return np.array([col[row_of[nm]] for nm in proj["snapped"]], dtype=float)

# test 3: dose 0 -> expression 0
new0 = dose_new_expr(0.0)
check("dose 0 sends gene expression to 0", bool((new0 == 0).all()),
      f"t={t_best} n={len(sel)} max_new={new0.max():.3f}")

# test 4: dose 0.5 -> ~0.5 of original on average
new_half = dose_new_expr(dose_half)
mean_ratio = float(np.mean(new_half / old_expr))
check("dose 0.5 sends gene expression to roughly 0.5 on average",
      dose_half_band[0] <= mean_ratio <= dose_half_band[1],
      f"t={t_best} n={len(sel)} mean(new/old)={mean_ratio:.3f}")


## TEST 5: Nkx2-5 knockout specifically affects the heart ##

print("\ntest 5: Nkx2-5 knockout changes heart cells and little else")
# cell type comes from the control snapped cell (the knockout tree's snapped
# cells are gene-off, so the original identity is read from control).
by_type_total = {}
by_type_changed = {}
for n in control.tree.nodes:
    if n not in ko.tree.nodes:
        continue
    ch = ko.tree.nodes[n].get("changed")
    if ch is None:
        continue
    sn = control.tree.nodes[n].get("snapped_name")
    ct = name_to_type.get(sn)
    if ct is None or (isinstance(ct, float) and np.isnan(ct)):
        continue
    by_type_total[ct] = by_type_total.get(ct, 0) + 1
    by_type_changed[ct] = by_type_changed.get(ct, 0) + (1 if ch else 0)


def rate(types):
    tot = sum(by_type_total.get(t, 0) for t in types)
    chg = sum(by_type_changed.get(t, 0) for t in types)
    return (chg / tot if tot else 0.0), tot

present_heart = [t for t in heart_types if by_type_total.get(t, 0) > 0]
heart_rate, heart_n = rate(present_heart)
nonheart_types = [t for t in by_type_total if t not in heart_types]
nonheart_rate, nonheart_n = rate(nonheart_types)

print(f"  heart types present: {present_heart}")
print(f"  heart change rate    = {heart_rate:.3f} over {heart_n} nodes")
print(f"  nonheart change rate = {nonheart_rate:.3f} over {nonheart_n} nodes")
top = sorted(by_type_changed.items(), key=lambda kv: kv[1], reverse=True)[:8]
print("  top changed cell types:")
for ct, c in top:
    if c:
        print(f"    {ct}: {c}/{by_type_total[ct]}")

check("heart cells are changed", heart_rate >= heart_change_rate_min,
      f"heart_rate={heart_rate:.3f}")
check("non-heart cells are mostly untouched", nonheart_rate <= nonheart_change_rate_max,
      f"nonheart_rate={nonheart_rate:.3f}")
check("heart is changed far more than non-heart", heart_rate > nonheart_rate,
      f"{heart_rate:.3f} vs {nonheart_rate:.3f}")


## TEST 6: applying the off-manifold cutoff depletes the heart ##

print("\ntest 6: the off-manifold cutoff depletes heart leaves, sparing the rest")
# a knocked-out heart cell projects far off-manifold (heart identity needs the
# gene), so it -- and its descendants -- are dropped by the off-manifold filter;
# non-heart cells never express the gene, never move, and survive.  because the
# knockout and control trees are node-identical, we can read each leaf's true
# identity from the control snapped cell and just ask which survive in the KO.
ko_dead = off_manifold_descendants(ko.tree, off_manifold_ratio)
ko_leaves = [n for n in ko.tree.nodes if ko.tree.out_degree(n) == 0]


def survival(types):
    tot = surv = 0
    for n in ko_leaves:
        sn = control.tree.nodes[n].get("snapped_name")
        ct = name_to_type.get(sn)
        if ct is None or (isinstance(ct, float) and np.isnan(ct)) or ct not in types:
            continue
        tot += 1
        surv += (n not in ko_dead)
    return (surv / tot if tot else float("nan")), tot


heart_surv, heart_leaf_n = survival(set(present_heart))
nonheart_surv, nonheart_leaf_n = survival(set(nonheart_types))
heart_depl = 1.0 - heart_surv
nonheart_depl = 1.0 - nonheart_surv
print(f"  off-manifold ratio   = {off_manifold_ratio}")
print(f"  heart    leaves: {heart_leaf_n:>6d}  survival={heart_surv:.3f}  depletion={heart_depl:.3f}")
print(f"  nonheart leaves: {nonheart_leaf_n:>6d}  survival={nonheart_surv:.3f}  depletion={nonheart_depl:.3f}")

check("heart leaves are substantially depleted", heart_surv <= heart_survival_max,
      f"heart_survival={heart_surv:.3f} (max {heart_survival_max})")
check("heart is depleted far more than non-heart",
      heart_depl >= heart_depletion_margin * max(nonheart_depl, 1e-6),
      f"heart_depletion={heart_depl:.3f} nonheart_depletion={nonheart_depl:.3f}")


## SUMMARY ##

print("\n" + "=" * 60)
if _failures:
    print(f"FAILED {len(_failures)} check(s): {_failures}")
    sys.exit(1)
print("all checks passed.")
