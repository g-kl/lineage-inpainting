"""row-2 cache (3 of 6 panels): the assignment-based coupling metrics, recomputed
from the FULL couplings for all three methods.

    germ-layer crossing   -- fraction of assigned parent->child edges that cross
                             germ layers (lower = better)
    lineage oscillation   -- fraction of lineages that flicker on/off along the
                             assignment path (lower = better)
    linkage distortion    -- 1 - (normalised linkage correlation), i.e. how far
                             the assignment's lineage-linkage structure is from
                             the real clonal linkage (lower = better)

these use the full per-node assignments (no subtree subsampling), exactly the
computation in analysis/lib/coupling/eval but read straight from the saved couplings:
  - model           : the splitter coupling pkl (style.MODEL_ASSIGNMENTS)
  - genot_sinkhorn  : assignments stored inside genot_w2_unified_model.pkl
  - genot_cbfgw     : assignments stored inside genot_fgw_unified_model.pkl

writes numerical_results/assignment_metrics.csv with columns
  method, metric, tag, value
run with the lt conda env from the repo root.
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from analysis.figures.summary_qc import style as S
from analysis.lib.metrics.germ_layer_eval import assignment_germ_layer_crossing_by_bin
from analysis.lib.metrics.linkage_metrics import (
    assignment_linkage_per_replicate,
    compare_linkage_normalized,
    lineage_oscillation_fraction,
    real_linkage_per_replicate,
)
from datasets.mouse_embryo import MouseEmbryoDataset

# linkage timepoints: (source = leaf time, target = cut time) pairs
LINKAGE_TIME_PAIRS = [(9.5, 9.0), (9.0, 8.5), (8.5, 8.0), (8.0, 7.5)]


def load_assignments():
    """{method: {tree_name: {node: observed_cell}}} for the three methods."""
    out = {}
    with open(S.MODEL_ASSIGNMENTS, "rb") as f:
        out["model"] = pickle.load(f)          # has an extra 'nn_distances' key; ignored
    for method, pkl in S.GENOT_PKLS.items():
        with open(pkl, "rb") as f:
            out[method] = pickle.load(f)["assignments"]
    return out


def compute_for(assignments, trees, name_to_gl, name_to_lineage, lineages, time_bins):
    """list of (metric, tag, value) for one method's couplings."""
    rows = []

    gl = assignment_germ_layer_crossing_by_bin(
        trees, assignments, name_to_gl, time_bins, exclude_source="epiblast")
    rows.append(("germ_layer_crossing", "", gl["all"]))
    for b in time_bins:
        if b in gl:
            rows.append(("germ_layer_crossing", f"bin_{b}", gl[b]))

    osc = lineage_oscillation_fraction(trees, assignments, name_to_lineage)
    rows.append(("lineage_oscillation_fraction", "", osc))

    for source_time, target_time in LINKAGE_TIME_PAIRS:
        real_lk = real_linkage_per_replicate(
            trees, target_time, name_to_lineage, types=lineages)
        assign_lk = assignment_linkage_per_replicate(
            trees, assignments, source_time=source_time, target_time=target_time,
            name_to_type=name_to_lineage, types=lineages)
        normed, raw, baseline = compare_linkage_normalized(real_lk, assign_lk)
        tag = f"t{source_time}_to_t{target_time}"
        # linkage *distortion* = how far the normalised correlation is from 1.
        rows.append(("linkage_correlation_normalized", tag, normed))
        rows.append(("linkage_distortion", tag, abs(1.0 - normed)))
    return rows


def main():
    print("loading dataset...", flush=True)
    dataset = MouseEmbryoDataset(max_size=S.N_TREES)

    ti = dataset.type_info_df
    subtype_to_type = dict(zip(ti.index, ti["cell_type"]))
    subtype_to_lineage = dict(zip(ti.index, ti["lineage"]))
    type_to_gl = dict(zip(ti["cell_type"], ti["germ_layer"]))
    name_to_subtype = dict(zip(dataset.cell_types_df.index,
                               dataset.cell_types_df["cell_subtype"]))
    name_to_lineage = {n: subtype_to_lineage[st] for n, st in name_to_subtype.items()
                       if st in subtype_to_lineage}
    name_to_gl = {n: type_to_gl[subtype_to_type[st]] for n, st in name_to_subtype.items()
                  if st in subtype_to_type and subtype_to_type[st] in type_to_gl}
    lineages = np.unique(list(name_to_lineage.values()))
    time_bins = sorted(dataset.states.obs["time"].astype(float).unique())

    assignments = load_assignments()

    all_rows = []
    for method in S.METHODS:
        print(f"computing assignment metrics for {method}...", flush=True)
        for metric, tag, value in compute_for(
            assignments[method], dataset.trees, name_to_gl, name_to_lineage,
            lineages, time_bins,
        ):
            all_rows.append((method, metric, tag, value))
            if tag == "":
                print(f"  {method}  {metric} = {value:.4f}", flush=True)

    df = pd.DataFrame(all_rows, columns=["method", "metric", "tag", "value"])
    out = os.path.join(S.NR, "assignment_metrics.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
