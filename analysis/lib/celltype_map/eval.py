#!/usr/bin/env python
"""Lightweight cell-type-map eval: assignments file -> transition map on disk.

A parameterised driver around the ``transition_graph`` engine and the
``sankey`` renderer. Given an ``assignments_nn_*.pkl`` it builds the
continuous-PMI cell-type transition graph at a single cutoff and writes two
artifacts into the model directory (the assignments file's parent), named to
match the eval convention::

    cell_type_map_<cfg>_step{step}<suffix>.pkl   # the networkx graph
    cell_type_map_<cfg>_step{step}<suffix>.png   # the rendered alluvial map

Use as a library (reuse one loaded dataset across a grid)::

    from analysis.lib.celltype_map.eval import build_cell_type_map
    ds = MouseEmbryoDataset(max_size=512)
    for path in assignment_paths:
        build_cell_type_map(path, dataset=ds)

or as a CLI::

    python -m analysis.lib.celltype_map.eval --assignments outputs/<cfg>/assignments_nn_<cfg>_step20000_adj_n1.pkl
    python -m analysis.lib.celltype_map.eval --experiment-dir outputs/<cfg> --step 20000
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys

sys.path.insert(
    0,
    os.environ.get(
        "SLURM_SUBMIT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
    ),
)

import numpy as np

from analysis.lib.celltype_map.sankey import plot_transition_graph
from analysis.lib.celltype_map.transition_graph import (
    compute_pairs_pmi, graph_from_pairs,
)


# Annotation level per (quantized) timepoint; anything unlisted -> 'cell_type'.
DEFAULT_RESOLUTION_BY_TIME = {
    7.5: "lineage", 8.0: "lineage",
    8.5: "cell_type", 9.0: "cell_type", 9.5: "cell_type", 10.0: "cell_type",
}
DEFAULT_PALETTE = ["#d6543a", "#3c4ebf", "#e3a600", "#40ada6", "#edb1a3"]

_NAME_RE = re.compile(
    r"assignments_nn_(?P<cfg>.+)_step(?P<step>\d+)(?P<suffix>.*)\.pkl$"
)


# --------------------------------------------------------------------------
# dataset-derived label / colour maps
# --------------------------------------------------------------------------

def build_label_maps(dataset, palette=DEFAULT_PALETTE):
    """``(name_to_labels, color_dict)`` for a MouseEmbryo-style dataset.

    ``name_to_labels`` maps each cell name to its label at every resolution
    level; ``color_dict`` colours every label by its germ layer, so colours stay
    stable whichever resolution a timepoint is rendered at.
    """
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

    label_to_gl = {}
    for st in ti.index:
        gl = st_to["germ_layer"][st]
        for lab in (st, st_to["cell_type"][st], st_to["lineage"][st], gl):
            label_to_gl.setdefault(lab, gl)
    germ_layers = sorted(set(st_to["germ_layer"].values()))
    gl_palette = {gl: palette[i % len(palette)] for i, gl in enumerate(germ_layers)}
    color_dict = {lab: gl_palette[gl] for lab, gl in label_to_gl.items()}
    return name_to_labels, color_dict


# --------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------

def _parse_assignments_name(path):
    """``(cfg, step, suffix)`` from an ``assignments_nn_*.pkl`` filename."""
    m = _NAME_RE.match(os.path.basename(path))
    if not m:
        raise ValueError(
            f"cannot parse cfg/step from {os.path.basename(path)!r}; expected "
            f"assignments_nn_<cfg>_step<step><suffix>.pkl"
        )
    return m.group("cfg"), int(m.group("step")), m.group("suffix")


def resolve_assignments_path(experiment_dir, step=None):
    """Pick an ``assignments_nn_*.pkl`` in ``experiment_dir``.

    Prefers the canonical pool-side coupling (``_adj_n1``, non-splitter) and, if
    ``step`` is given, a file at that step; falls back to the first match.
    """
    files = [f for f in os.listdir(experiment_dir)
             if f.startswith("assignments_nn") and f.endswith(".pkl")
             and "_splitter" not in f]
    if not files:
        raise FileNotFoundError(f"no assignments_nn_*.pkl in {experiment_dir}")
    if step is not None:
        files = [f for f in files if f"_step{step}" in f] or files
    files.sort(key=lambda f: (0 if "_adj_n1." in f or "_adj_n1_" in f else 1, f))
    return os.path.join(experiment_dir, files[0])


# --------------------------------------------------------------------------
# the eval
# --------------------------------------------------------------------------

def build_cell_type_map(
    assignments_path,
    *,
    dataset=None,
    n_trees=512,
    out_dir=None,
    resolution_by_time=None,
    pmi_quantile=0.5,
    dt=0.5,
    pmi_k=10,
    n_max=500_000,
    min_composition=0.1,
    min_count=5,
    composition="max",
    nn_quantile=0.8,
    embedding_key="X_scvi",
    palette=DEFAULT_PALETTE,
    ribbon_width=0.2,
    figsize=(5, 18),
    dpi=140,
    save_graph=True,
    save_png=True,
    use_gpu=True,
    seed=0,
):
    """Build and save the cell-type transition map for one assignments file.

    Returns ``dict(graph, graph_path, png_path, metrics_path, n_pairs, n_total,
    pmi_cutoff)``.
    Pass ``dataset`` to reuse a loaded ``MouseEmbryoDataset`` across many calls;
    otherwise one is loaded with ``max_size=n_trees``.
    """
    resolution_by_time = (DEFAULT_RESOLUTION_BY_TIME if resolution_by_time is None
                          else resolution_by_time)
    cfg, step, suffix = _parse_assignments_name(assignments_path)
    out_dir = out_dir or os.path.dirname(os.path.abspath(assignments_path))

    if dataset is None:
        from datasets.mouse_embryo import MouseEmbryoDataset
        print(f"[cell_type_map] loading dataset (max_size={n_trees})")
        dataset = MouseEmbryoDataset(max_size=n_trees)

    with open(assignments_path, "rb") as f:
        assignments = pickle.load(f)
    name_to_labels, color_dict = build_label_maps(dataset, palette)

    print(f"[cell_type_map] {cfg} step{step}{suffix}: scoring transition PMI")
    pp = compute_pairs_pmi(
        dataset.trees, assignments, dataset,
        dataset.name_to_idx, name_to_labels,
        dt=dt, k=pmi_k, n_max=n_max, nn_quantile=nn_quantile,
        embedding_key=embedding_key, seed=seed, use_gpu=use_gpu)
    pmi = pp["pmi"]
    cutoff = float(np.quantile(pmi, pmi_quantile))
    print(f"[cell_type_map] {len(pmi)} pairs (of {pp['n_total']}); "
          f"pmi q{pmi_quantile:.2f} cutoff={cutoff:.2f}, frac>0={(pmi > 0).mean():.2f}")

    G = graph_from_pairs(
        pp, dataset.trees, dataset, name_to_labels, resolution_by_time,
        pmi_quantile=pmi_quantile, min_composition=min_composition,
        min_count=min_count, dt=dt, composition=composition)
    n_types = len({ct for ct, _ in G.nodes()})
    print(f"[cell_type_map] graph: types={n_types} edges={G.number_of_edges()}")

    base = os.path.join(out_dir, f"cell_type_map_{cfg}_step{step}{suffix}")
    graph_path = png_path = None
    if save_graph:
        graph_path = base + ".pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        print(f"[cell_type_map] wrote {graph_path}")
    if save_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plot_transition_graph(G, ax=ax, color_dict=color_dict,
                              ribbon_width=ribbon_width)
        png_path = base + ".png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[cell_type_map] wrote {png_path}")

    # Curated-transition fraction: how many lit-curated transitions does this
    # graph recover? Reads the literature/transitions/*.csv set and folds the
    # answer into the cpl metrics CSV so it shows up next to germ/osc/linkage.
    metrics_path = None
    try:
        from analysis.lib.metrics.curated_transition import curated_transition_fraction
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        curated_dir = os.path.join(repo_root, "literature", "transitions")
        curated_frac = curated_transition_fraction(curated_dir, G)
        n_edges = G.number_of_edges()
        print(f"[cell_type_map] curated_transition_fraction = {curated_frac:.4f} "
              f"(graph has {n_edges} edges)")

        # Append to / create the matching coupling metrics CSV so the existing
        # loader picks the curated row up alongside germ/osc/linkage.
        import pandas as pd
        # `suffix` is e.g. "_adj_n1" or "_adj_n32_splitter". The matching cpl
        # CSV uses the same suffix on `metrics_<cfg>_step{step}{suffix}.csv`.
        metrics_path = os.path.join(out_dir, f"metrics_{cfg}_step{step}{suffix}.csv")
        new_rows = [
            {"metric": "curated_transition_fraction", "tree": "", "value": curated_frac},
            {"metric": "n_transition_graph_edges",     "tree": "", "value": n_edges},
        ]
        if os.path.exists(metrics_path):
            existing = pd.read_csv(metrics_path)
            # Drop any prior rows for these metric names so we can refresh.
            existing = existing[~existing["metric"].isin(
                ["curated_transition_fraction", "n_transition_graph_edges"])]
            df_out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            df_out = pd.DataFrame(new_rows)
        df_out.to_csv(metrics_path, index=False)
        print(f"[cell_type_map] appended curated to {metrics_path}")
    except Exception as e:
        print(f"[cell_type_map] curated fraction failed: {e}")

    return dict(graph=G, graph_path=graph_path, png_path=png_path,
                metrics_path=metrics_path,
                n_pairs=len(pmi), n_total=pp["n_total"], pmi_cutoff=cutoff)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--assignments", help="path to an assignments_nn_*.pkl")
    src.add_argument("--experiment-dir", help="model dir to resolve assignments in")
    ap.add_argument("--step", type=int, default=None,
                    help="step to prefer when resolving --experiment-dir")
    ap.add_argument("--n-trees", type=int, default=512)
    ap.add_argument("--pmi-quantile", type=float, default=0.5)
    ap.add_argument("--min-composition", type=float, default=0.1)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--no-graph", action="store_true", help="skip the .pkl artifact")
    ap.add_argument("--cpu", action="store_true", help="force CPU faiss")
    args = ap.parse_args()

    path = (args.assignments
            if args.assignments
            else resolve_assignments_path(args.experiment_dir, args.step))
    build_cell_type_map(
        path, n_trees=args.n_trees, pmi_quantile=args.pmi_quantile,
        min_composition=args.min_composition, min_count=args.min_count,
        save_graph=not args.no_graph, use_gpu=not args.cpu)


if __name__ == "__main__":
    main()
