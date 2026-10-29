#!/usr/bin/env python
#SBATCH --job-name=compute_coupling
#SBATCH --output=logs/compute_coupling_%j.log
#SBATCH --error=logs/compute_coupling_%j.log
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=bates

import os
import sys

sys.path.insert(
    0,
    os.environ.get(
        "SLURM_SUBMIT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."),
    ),
)

import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis.lib.metrics.germ_layer_eval import assignment_germ_layer_crossing_by_bin
from analysis.lib.metrics.linkage_metrics import (
    assignment_linkage_per_replicate,
    compare_linkage_normalized,
    lineage_oscillation_fraction,
    real_linkage_per_replicate,
)
from analysis.lib.coupling.nearest_neighbor_coupling_w_context import couple_all_trees
from analysis.lib.coupling.sankey import build_transition_graph, plot_transition_graph
from datasets.mouse_embryo import MouseEmbryoDataset
from utils.experiment_utils import get_experiment_info


## CONFIG ##

# Override via env var so we can submit different checkpoints without editing.
experiment_dir = os.environ.get(
    "EXPERIMENT_DIR",
    "outputs/mouse_embryo_grid_bridge_unshared_cfm_id_nosa_std_7f434a38adbc8cdaa649162bdf7679e1",
)
step = (
    int(os.environ["STEP"]) if os.environ.get("STEP") else None
)  # None -> latest pool_ema_step checkpoint

# Unshared models have a separate pool (e.g., DeltaTreeFormer)
# and split (bridge flow) model. The default eval encodes trees with the
# *pool* model. Set USE_SPLITTER_AS_POOLER=1 to instead encode with the
# *splitter* model (split_ema_step*.pt), producing a second set of couplings
# under a `_splitter` suffix so the pooler-side results stay intact.
use_splitter = os.environ.get("USE_SPLITTER_AS_POOLER", "0") == "1"
ckpt_prefix = "split_ema_step" if use_splitter else "pool_ema_step"
splitter_suffix = "_splitter" if use_splitter else ""

# linkage timepoints: list of (source = leaf time, target = cut time) pairs
linkage_time_pairs = [
    (9.5, 9.0),
    (9.0, 8.5),
    (8.5, 8.0),
    (8.0, 7.5)
]

# sankey / transition-graph params
sankey_nn_quantile = 0.99
sankey_min_support = 50
sankey_min_fold = 2.0
sankey_alpha = 0.05


## RESOLVE CHECKPOINT ##

cfg_name = os.path.basename(experiment_dir.rstrip("/"))
pool_ckpts = [x for x in os.listdir(experiment_dir)
              if x.startswith(ckpt_prefix) and x.endswith(".pt")]
if not pool_ckpts:
    raise FileNotFoundError(f"no {ckpt_prefix}*.pt in {experiment_dir}")


def _ema_step(fname):
    return int(fname[len(ckpt_prefix):-len(".pt")])


if step is None:
    pool_ckpt = max(pool_ckpts, key=_ema_step)
    step = _ema_step(pool_ckpt)
else:
    pool_ckpt = f"{ckpt_prefix}{step}.pt"
    if pool_ckpt not in pool_ckpts:
        raise FileNotFoundError(f"{pool_ckpt} not in {experiment_dir}")

print(f"[compute_coupling_and_metrics] cfg = {cfg_name}")
print(f"  use_splitter_as_pooler={use_splitter}")
print(f"  checkpoint: {pool_ckpt}  (step={step})")


## DATASET & ANNOTATIONS ##

n_trees = int(os.environ.get("N_TREES", "512"))
print(f"loading dataset (max_size={n_trees})...")
dataset = MouseEmbryoDataset(max_size=n_trees)
cell_types_df = dataset.cell_types_df
cell_type_info = dataset.type_info_df

subtype_to_type = {st: t for st, t in zip(cell_type_info.index, cell_type_info['cell_type'])}
subtype_to_lineage = {st: lin for st, lin in zip(cell_type_info.index, cell_type_info['lineage'])}
type_to_gl = {t: gl for t, gl in zip(cell_type_info['cell_type'], cell_type_info['germ_layer'])}

name_to_subtype = {n: st for n, st in zip(cell_types_df.index, cell_types_df['cell_subtype'])}
name_to_cell_type = {
    n: subtype_to_type[st]
    for n, st in name_to_subtype.items()
    if st in subtype_to_type
}
name_to_lineage = {
    n: subtype_to_lineage[st]
    for n, st in name_to_subtype.items()
    if st in subtype_to_lineage
}
name_to_gl = {
    n: type_to_gl[subtype_to_type[st]]
    for n, st in name_to_subtype.items()
    if st in subtype_to_type and subtype_to_type[st] in type_to_gl
}
types = np.unique(list(name_to_cell_type.values()))
lineages = np.unique(list(name_to_lineage.values()))


## LOAD POOLER ##

print("loading pooler...")
info = get_experiment_info(experiment_dir)
pool_cfg_key = "splitter" if use_splitter else "pooler"
pooler = hydra.utils.instantiate(info['config'][pool_cfg_key]).cuda()
pooler.load_state_dict(torch.load(
    os.path.join(experiment_dir, pool_ckpt), map_location='cuda',
))
pooler.eval()

## COUPLE ##

# POOL_N_STEPS: multi-step flow integration in the pooler at inference. Only
# applies when the pooler exposes ``n_steps`` (BridgeTreeFlow does; TreeFormer
# does not).
pool_n_steps = int(os.environ.get("POOL_N_STEPS", "4"))
if pool_n_steps != 1:
    print(f"  [info] using pool_n_steps={pool_n_steps} (multi-step pool inference)")

# `_adj` marks the adjacent-bins coupling: each target snaps to observed cells
# in its own time bin and the bins on either side.
out_suffix = f"_adj_n{pool_n_steps}{splitter_suffix}"

assignments_path = os.path.join(
    experiment_dir, f"assignments_nn_{cfg_name}_step{step}{out_suffix}.pkl"
)
print(f"computing couplings -> {assignments_path}")
assignments = couple_all_trees(
    pooler, dataset.trees, dataset.states, pool_n_steps=pool_n_steps,
    sample_time_gap=0.5, min_snap_time=7.0
)
with open(assignments_path, 'wb') as f:
    pickle.dump(assignments, f)
print(f"  wrote {assignments_path}")


## GERM-LAYER CROSSING ##

print("computing germ-layer crossing (epiblast out-edges ignored)...")
time_bins = sorted(dataset.states.obs['time'].astype(float).unique())
gl_crossing = assignment_germ_layer_crossing_by_bin(
    dataset.trees, assignments, name_to_gl, time_bins, exclude_source='epiblast',
)
germ_layer_crossing = gl_crossing['all']
print(f"  germ_layer_crossing = {germ_layer_crossing:.4f}")
for b in time_bins:
    if b in gl_crossing:
        print(f"  germ_layer_crossing[bin {b}] = {gl_crossing[b]:.4f}")


## LINEAGE OSCILLATION ##

print("computing lineage oscillation fraction...")
oscillation_fraction = lineage_oscillation_fraction(
    dataset.trees, assignments, name_to_lineage,
)
print(f"  lineage_oscillation_fraction = {oscillation_fraction:.4f}")


## LINKAGE CORRELATION (per replicate) ##

print("computing linkage correlation (per replicate)...")
linkage_results = {}
for source_time, target_time in linkage_time_pairs:
    real_lk = real_linkage_per_replicate(
        dataset.trees, target_time, name_to_lineage, types=lineages,
    )
    assign_lk = assignment_linkage_per_replicate(
        dataset.trees, assignments,
        source_time=source_time,
        target_time=target_time,
        name_to_type=name_to_lineage, types=lineages,
    )
    normed, raw, baseline = compare_linkage_normalized(real_lk, assign_lk)
    linkage_results[(source_time, target_time)] = (normed, raw, baseline)
    print(f"  t{source_time}->t{target_time}: linkage_correlation = {normed:.4f} "
          f"(raw={raw:.4f}, baseline={baseline:.4f})")


## METRICS CSV ##

rows = [
    ('germ_layer_crossing', '', germ_layer_crossing),
    ('lineage_oscillation_fraction', '', oscillation_fraction),
]
for b in time_bins:
    if b in gl_crossing:
        rows.append(('germ_layer_crossing', f'bin_{b}', gl_crossing[b]))
for (source_time, target_time), (normed, raw, baseline) in linkage_results.items():
    tag = f"t{source_time}_to_t{target_time}"
    rows.append((f'linkage_correlation_normalized', tag, normed))
    rows.append((f'linkage_correlation_raw', tag, raw))
    rows.append((f'linkage_correlation_baseline', tag, baseline))
for tree_name, dists in assignments['nn_distances'].items():
    if dists:
        rows.append(('mean_nn_distance', tree_name, float(np.mean(list(dists.values())))))
all_dists = [d for dists in assignments['nn_distances'].values() for d in dists.values()]
if all_dists:
    rows.append(('mean_nn_distance', 'all', float(np.mean(all_dists))))

metrics_df = pd.DataFrame(rows, columns=['metric', 'tree', 'value'])
metrics_path = os.path.join(
    experiment_dir, f"metrics_{cfg_name}_step{step}{out_suffix}.csv"
)
metrics_df.to_csv(metrics_path, index=False)
print(f"  wrote {metrics_path}")
print("done.")
