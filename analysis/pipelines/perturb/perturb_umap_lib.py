"""shared helpers for the perturbation-delta UMAP scripts.

both ``perturb_umap_compute.py`` (builds the delta UMAP) and
``perturb_umap_plot.py`` (draws it) need to turn a screen's light trees into the
set of target-time leaves they landed on.  that logic lives here so the two
scripts can never disagree about which cells count.

a screen directory looks like::

    <screen_dir>/genes/<gene>.pkl     {tree_name: {"meta": df, "tree": graph}}
    <screen_dir>/control/<tree>.pkl   {"meta": df, "tree": graph}

we only ever read ``meta`` -- one row per terminal leaf with ``snapped_name``
(the observed cell it projected onto), ``time`` and ``off_manifold``.
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd

from analysis.pipelines.perturb.perturb_result import off_manifold_descendants

GERM_LAYERS = ["ectoderm", "mesoderm", "endoderm"]

# lineages grouped by germ layer, with a refined germ-layer-keyed palette:
# ectoderm = greens/teals, mesoderm = purples (+ blood crimson),
# endoderm = oranges, epiblast = neutral, pgc = magenta.  used to hue the
# central map by the lineage each knockout depletes most.
LINEAGE_COLORS = {
    # ectoderm family
    "neural ectoderm":         "#1B9E77",
    "ectoderm":                "#3CB371",
    "surface ectoderm":        "#66C2A5",
    "neural crest":            "#117A65",
    "extraembryonic ectoderm": "#A6D854",
    # mesoderm family
    "mesoderm":                "#7570B3",
    "paraxial mesoderm":       "#5E4FA2",
    "lateral plate mesoderm":  "#8DA0CB",
    "intermediate mesoderm":   "#9970C2",
    "extraembryonic mesoderm": "#BCA8DD",
    "blood":                   "#C1121F",
    # endoderm family
    "endoderm":                "#D95F02",
    "extraembryonic endoderm": "#FDB462",
    # other
    "epiblast":                "#7F7F7F",
    "primordial germ cell":    "#E7298A",
}
LINEAGE_ORDER = list(LINEAGE_COLORS)


def load_cell_categories(data_dir, subtype_filename="final_subtype.csv",
                         type_info_filename="final_type_info.csv"):
    """name -> (cell_type, lineage) maps without touching the 55GB h5ad.

    ``final_subtype.csv`` is one row per observed cell (index = cell name,
    column ``cell_subtype``); ``final_type_info.csv`` maps each subtype to its
    ``cell_type`` / ``lineage`` / ``germ_layer``.  composing the two gives a
    per-cell category for every snapped leaf the screens reference.

    returns ``(name_to_cell_type, name_to_lineage, type_info)`` where the maps
    are pandas Series indexed by cell name and ``type_info`` is the subtype
    table (so callers can read the canonical cell-type / lineage vocabularies).
    """
    type_info = pd.read_csv(os.path.join(data_dir, type_info_filename), index_col=0)
    sub = pd.read_csv(os.path.join(data_dir, subtype_filename), index_col=0)
    subtype = sub["cell_subtype"]
    name_to_cell_type = subtype.map(type_info["cell_type"])
    name_to_lineage = subtype.map(type_info["lineage"])
    return name_to_cell_type, name_to_lineage, type_info


def category_fractions(names, name_to_cat, categories):
    """fraction of ``names`` falling in each of ``categories`` (fixed order).

    ``name_to_cat`` is a pandas Series (cell name -> category).  unknown names
    drop out; the denominator stays ``len(names)`` so fractions are comparable
    across genes with different unmapped-name rates.
    """
    counts = pd.Series(names).map(name_to_cat).value_counts()
    n = max(len(names), 1)
    return np.array([counts.get(c, 0) / n for c in categories], dtype=float)


def _on_manifold_keep(d, target_time, tol, ratio):
    """boolean mask over one light tree's leaves: within the target-time window and
    on-manifold.

    ``ratio=None`` uses the per-leaf ``off_manifold`` flag baked into ``meta`` at the
    screen's own cutoff.  passing a ``ratio`` re-thresholds from the raw projection
    fields on the tree and also drops the *whole subtree* under any node that
    projected too far (via :func:`off_manifold_descendants`) -- a node that changed
    and landed off-manifold never would have propagated, so its descendant leaves are
    invalid even if their own flag reads on-manifold.
    """
    meta = d["meta"]
    t = meta["time"].to_numpy(dtype=float)
    time_ok = np.abs(t - target_time) < tol
    if ratio is None:
        return time_ok & (~meta["off_manifold"].to_numpy(dtype=bool))
    bad = off_manifold_descendants(d["tree"], ratio)
    node_name = meta["node_name"].to_numpy()
    on = np.fromiter((nm not in bad for nm in node_name), bool, len(node_name))
    return time_ok & on


def leaf_names(light_trees, target_time, tol, ratio=None):
    """snapped names of on-manifold target-time leaves (duplicates kept).

    the name analogue of :func:`leaf_rows` -- it skips the row-index mapping so
    callers that only need cell-type / lineage labels (from
    :func:`load_cell_categories`) never have to load the AnnData states.  see
    :func:`_on_manifold_keep` for the ``ratio`` off-manifold semantics.
    """
    names = []
    for d in light_trees.values():
        keep = _on_manifold_keep(d, target_time, tol, ratio)
        names.extend(d["meta"]["snapped_name"].to_numpy()[keep].tolist())
    return names


def leaf_rows(light_trees, target_time, tol, name_to_row, ratio=None):
    """row indices (into ``states``) of on-manifold target-time leaves.

    ``light_trees`` is the ``{tree_name: {"meta", "tree"}}`` dict of one gene (or
    a wrapped control set).  duplicates are kept -- a snapped cell hit by two
    leaves counts twice, so the row multiset reflects the landed population.  see
    :func:`_on_manifold_keep` for the ``ratio`` off-manifold semantics (default
    ``None`` = the baked per-leaf flag; a ratio prunes off-manifold subtrees).
    """
    rows = []
    for d in light_trees.values():
        keep = _on_manifold_keep(d, target_time, tol, ratio)
        for nm in d["meta"]["snapped_name"].to_numpy()[keep]:
            r = name_to_row.get(nm)
            if r is not None:
                rows.append(r)
    return np.asarray(rows, dtype=int)


def load_gene(genes_dir, gene):
    with open(os.path.join(genes_dir, f"{gene}.pkl"), "rb") as fh:
        return pickle.load(fh)


def gene_names(genes_dir):
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(genes_dir, "*.pkl")))


def load_control(control_dir):
    """control trees wrapped as one ``{tree_name: {"meta", "tree"}}`` dict."""
    out = {}
    for p in sorted(glob.glob(os.path.join(control_dir, "*.pkl"))):
        with open(p, "rb") as fh:
            out[os.path.splitext(os.path.basename(p))[0]] = pickle.load(fh)
    return out


def germ_fractions(rows, germ_layer):
    """fraction of ``rows`` falling in each of the three germ layers."""
    labels = germ_layer[rows]
    n = max(len(rows), 1)
    return np.array([(labels == gl).sum() / n for gl in GERM_LAYERS], dtype=float)
