#!/usr/bin/env python
#SBATCH --job-name=compute_baseline_metrics
#SBATCH --output=logs/compute_baseline_metrics_%j.log
#SBATCH --error=logs/compute_baseline_metrics_%j.log
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=bates

"""Compute coupling metrics for the GENOT baselines.

Analogous to ``compute_coupling_and_metrics.py`` but, instead of running a
trained pooler to derive couplings, this loads the per-node ``assignments``
already stored in each ``outputs/e_baselines/genot*.pkl`` (see the
``baselines/`` folder for the code that produces them) and computes the same
germ-layer / oscillation / linkage metrics. Results for all baselines are
written to a single CSV in the baselines folder.
"""

import os
import sys

sys.path.insert(
    0,
    os.environ.get(
        "SLURM_SUBMIT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."),
    ),
)

import glob
import pickle

import numpy as np
import pandas as pd

from analysis.lib.metrics.germ_layer_eval import assignment_germ_layer_crossing_by_bin
from analysis.lib.metrics.linkage_metrics import (
    assignment_linkage_per_replicate,
    compare_linkage_normalized,
    lineage_oscillation_fraction,
    real_linkage_per_replicate,
)
from datasets.mouse_embryo import MouseEmbryoDataset


## CONFIG ##

# Where the baseline scripts write their genot*.pkl outputs (see baselines/).
baseline_dir = os.environ.get("BASELINE_DIR", "outputs/e_baselines")

# linkage timepoints: list of (source = leaf time, target = cut time) pairs
linkage_time_pairs = [
    (9.5, 9.0),
    (9.0, 8.5),
    (8.5, 8.0),
    (8.0, 7.5),
]


## DATASET & ANNOTATIONS ##

print("loading dataset...")
dataset = MouseEmbryoDataset()

cell_types_df = dataset.cell_types_df
cell_type_info = dataset.type_info_df

subtype_to_type = {st: t for st, t in zip(cell_type_info.index, cell_type_info['cell_type'])}
subtype_to_lineage = {st: lin for st, lin in zip(cell_type_info.index, cell_type_info['lineage'])}
type_to_gl = {t: gl for t, gl in zip(cell_type_info['cell_type'], cell_type_info['germ_layer'])}

name_to_subtype = {n: st for n, st in zip(cell_types_df.index, cell_types_df['cell_subtype'])}
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
lineages = np.unique(list(name_to_lineage.values()))

time_bins = sorted(dataset.states.obs['time'].astype(float).unique())


## METRICS ##

def compute_metrics(assignments):
    """Compute coupling metrics for one baseline's per-node assignments.

    ``assignments`` is ``{tree_name: {node: obs_cell}}`` (the format produced
    by ``baselines/genot_common.build_assignments``). Returns a list of
    ``(metric, tree, value)`` rows.
    """
    rows = []

    gl_crossing = assignment_germ_layer_crossing_by_bin(
        dataset.trees, assignments, name_to_gl, time_bins, exclude_source='epiblast',
    )
    rows.append(('germ_layer_crossing', '', gl_crossing['all']))
    for b in time_bins:
        if b in gl_crossing:
            rows.append(('germ_layer_crossing', f'bin_{b}', gl_crossing[b]))

    oscillation_fraction = lineage_oscillation_fraction(
        dataset.trees, assignments, name_to_lineage,
    )
    rows.append(('lineage_oscillation_fraction', '', oscillation_fraction))

    for source_time, target_time in linkage_time_pairs:
        real_lk = real_linkage_per_replicate(
            dataset.trees, target_time, name_to_lineage, types=lineages,
        )
        assign_lk = assignment_linkage_per_replicate(
            dataset.trees, assignments,
            source_time=source_time, target_time=target_time,
            name_to_type=name_to_lineage, types=lineages,
        )
        normed, raw, baseline = compare_linkage_normalized(real_lk, assign_lk)
        tag = f"t{source_time}_to_t{target_time}"
        rows.append(('linkage_correlation_normalized', tag, normed))
        rows.append(('linkage_correlation_raw', tag, raw))
        rows.append(('linkage_correlation_baseline', tag, baseline))

    return rows


## RUN ##

pkl_paths = sorted(glob.glob(os.path.join(baseline_dir, "genot*.pkl")))
if not pkl_paths:
    raise FileNotFoundError(f"no genot*.pkl in {baseline_dir}")
print(f"found {len(pkl_paths)} baseline pkl(s) in {baseline_dir}")

all_rows = []
for path in pkl_paths:
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"computing metrics for {name}...")
    with open(path, 'rb') as f:
        assignments = pickle.load(f)['assignments']
    for metric, tree, value in compute_metrics(assignments):
        all_rows.append((name, metric, tree, value))

metrics_df = pd.DataFrame(all_rows, columns=['baseline', 'metric', 'tree', 'value'])
metrics_path = os.path.join(baseline_dir, "baseline_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"wrote {metrics_path}")
print("done.")
