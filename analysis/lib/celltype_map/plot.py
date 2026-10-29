#!/usr/bin/env python
"""Build and render the cell-type transition map.

  load_inputs()        -- load dataset + assignments once (the expensive part).
  compute_pairs_pmi()  -- extract transition pairs and score their continuous
                          PMI (also expensive; done once, reused across cutoffs).
  graph_from_pairs()   -- threshold by PMI, relabel by RESOLUTION_BY_TIME, apply
                          a composition cutoff -> graph (cheap).

Run ``python -m analysis.lib.celltype_map.plot``. Edit RESOLUTION_BY_TIME and the
knobs below; the PMI cutoff is swept and rendered as small multiples.
"""
import os
import sys

sys.path.insert(
    0,
    os.environ.get(
        "SLURM_SUBMIT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
    ),
)

import pickle

import numpy as np
import matplotlib.pyplot as plt

from analysis.lib.celltype_map.sankey import plot_transition_graph
from analysis.lib.celltype_map.transition_graph import (
    compute_pairs_pmi, graph_from_pairs, quantize_time,
)
from datasets.mouse_embryo import MouseEmbryoDataset


## CONFIG ##

EXPERIMENT_DIR = os.environ.get(
    "EXPERIMENT_DIR",
    "outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09",
)
STEP = int(os.environ["STEP"]) if os.environ.get("STEP") else 5000
N_TREES = int(os.environ.get("N_TREES", "512"))
DT = 0.5
PALETTE = ['#d6543a', '#3c4ebf', '#e3a600', '#40ada6', '#edb1a3']

# annotation level per timepoint (set by hand). anything unlisted -> 'cell_type'
RESOLUTION_BY_TIME = {
    7.5: 'lineage', 8.0: 'lineage',
    8.5: 'cell_type', 9.0: 'cell_type', 9.5: 'cell_type', 10.0: 'cell_type',
}

# PMI estimation
PMI_K = 10
N_MAX = 500_000          # subsample cap for the FAISS k-NN density estimates
# discrete edge filter
MIN_COMPOSITION = 0.1
MIN_COUNT = 5
# PMI cutoffs to sweep (quantiles of the PMI distribution; 0 keeps everything)
PMI_QUANTILES = [0.4, 0.50, 0.6]


## INPUTS ##

def load_inputs(experiment_dir=EXPERIMENT_DIR, step=STEP, n_trees=N_TREES):
    cfg_name = os.path.basename(experiment_dir.rstrip("/"))
    assignments_path = os.path.join(
        experiment_dir, f"assignments_nn_{cfg_name}_step{step}_adj_n1.pkl"
    )
    print(f"[load] cfg={cfg_name} step={step} n_trees={n_trees}")
    dataset = MouseEmbryoDataset(max_size=n_trees)
    print(f"[load] assignments {assignments_path}")
    with open(assignments_path, "rb") as f:
        assignments = pickle.load(f)

    ti = dataset.type_info_df
    st_to = {lvl: dict(zip(ti.index, ti[lvl]))
             for lvl in ("cell_type", "lineage", "germ_layer")}
    name_to_subtype = dict(
        zip(dataset.cell_types_df.index, dataset.cell_types_df["cell_subtype"]))
    name_to_labels = {
        name: {"cell_subtype": st, "cell_type": st_to["cell_type"][st],
               "lineage": st_to["lineage"][st], "germ_layer": st_to["germ_layer"][st]}
        for name, st in name_to_subtype.items() if st in st_to["cell_type"]
    }

    # label -> germ_layer for every level, so colours stay stable across schemes
    label_to_gl = {}
    for st in ti.index:
        gl = st_to["germ_layer"][st]
        for lab in (st, st_to["cell_type"][st], st_to["lineage"][st], gl):
            label_to_gl.setdefault(lab, gl)
    germ_layers = sorted(set(st_to["germ_layer"].values()))
    gl_palette = {gl: PALETTE[i % len(PALETTE)] for i, gl in enumerate(germ_layers)}
    color_dict = {lab: gl_palette[gl] for lab, gl in label_to_gl.items()}

    return dict(dataset=dataset, assignments=assignments,
                name_to_idx=dataset.name_to_idx, name_to_labels=name_to_labels,
                color_dict=color_dict, cfg_name=cfg_name, step=step)


## DRIVER ##

def main():
    inp = load_inputs()
    print("[pmi] extracting pairs and scoring PMI...")
    pp = compute_pairs_pmi(
        inp["dataset"].trees, inp["assignments"], inp["dataset"],
        inp["name_to_idx"], inp["name_to_labels"],
        dt=DT, k=PMI_K, n_max=N_MAX)
    pmi = pp["pmi"]
    print(f"[pmi] {len(pmi)} pairs (of {pp['n_total']} total). "
          f"PMI min/med/max = {pmi.min():.2f}/{np.median(pmi):.2f}/{pmi.max():.2f}; "
          f"frac>0 = {(pmi > 0).mean():.2f}")

    out_prefix = os.path.join(
        EXPERIMENT_DIR, f"cell_type_map_{inp['cfg_name']}_step{inp['step']}")

    graphs = []
    for q in PMI_QUANTILES:
        G = graph_from_pairs(
            pp, inp["dataset"].trees, inp["dataset"], inp["name_to_labels"],
            RESOLUTION_BY_TIME, pmi_quantile=q,
            min_composition=MIN_COMPOSITION, min_count=MIN_COUNT, dt=DT)
        n_types = len({ct for ct, _ in G.nodes()})
        print(f"[build] pmi_q={q:.2f}  cutoff={np.quantile(pmi, q):.2f}  "
              f"types={n_types}  edges={G.number_of_edges()}")
        graphs.append((q, G))

    # small multiples over the PMI cutoff
    ncol = len(graphs)
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 18), dpi=140)
    axes = np.atleast_1d(axes).ravel()
    for ax, (q, G) in zip(axes, graphs):
        plot_transition_graph(G, ax=ax, color_dict=inp["color_dict"], ribbon_width=0.2)
        ax.set_title(f"PMI quantile {q:.2f}", fontsize=10)
    grid_path = out_prefix + "_pmi_sweep.png"
    fig.savefig(grid_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {grid_path}")


if __name__ == "__main__":
    main()
