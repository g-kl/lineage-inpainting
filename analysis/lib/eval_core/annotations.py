"""Dataset -> cell annotation maps, in one place.

Every eval that scores an assignment/coupling needs the same handful of maps
derived from a ``MouseEmbryo``-style dataset: cell name -> cell type / lineage /
germ layer, plus the sorted label sets and time bins. ``load_annotations``
builds all of them, plus the multi-resolution labels and germ-layer colours
used by the transition map.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# germ-layer palette for the transition map; kept here so label -> colour is
# derived from the same maps as everything else.
DEFAULT_PALETTE = ["#d6543a", "#3c4ebf", "#e3a600", "#40ada6", "#edb1a3"]


@dataclass
class Annotations:
    """All dataset-derived label maps needed by the assignment/coupling evals.

    ``name_to_*`` key on a cell's name (as it appears in the trees). ``lineages``
    / ``types`` are the sorted unique label vocabularies (as passed to the
    linkage metrics). ``time_bins`` are the sorted unique observed timepoints.

    ``name_to_labels`` / ``color_dict`` support the transition-map renderer: the
    former gives every cell its label at each resolution level, the latter
    colours every label by its germ layer.
    """

    name_to_cell_type: dict
    name_to_lineage: dict
    name_to_gl: dict
    name_to_subtype: dict
    lineages: np.ndarray
    types: np.ndarray
    time_bins: list
    name_to_labels: dict
    color_dict: dict


def load_annotations(dataset, palette=DEFAULT_PALETTE) -> Annotations:
    """Build every label map from a loaded ``MouseEmbryoDataset``.

    Cheap (pure dataframe joins, no model / GPU); pass a dataset that is already
    loaded so this is reused across a grid of couplings.
    """
    ti = dataset.type_info_df
    # subtype -> {cell_type, lineage, germ_layer}
    st_to = {lvl: dict(zip(ti.index, ti[lvl]))
             for lvl in ("cell_type", "lineage", "germ_layer")}
    type_to_gl = dict(zip(ti["cell_type"], ti["germ_layer"]))

    name_to_subtype = dict(
        zip(dataset.cell_types_df.index, dataset.cell_types_df["cell_subtype"]))
    name_to_cell_type = {n: st_to["cell_type"][st]
                         for n, st in name_to_subtype.items()
                         if st in st_to["cell_type"]}
    name_to_lineage = {n: st_to["lineage"][st]
                       for n, st in name_to_subtype.items()
                       if st in st_to["lineage"]}
    name_to_gl = {n: type_to_gl[st_to["cell_type"][st]]
                  for n, st in name_to_subtype.items()
                  if st in st_to["cell_type"] and st_to["cell_type"][st] in type_to_gl}

    lineages = np.unique(list(name_to_lineage.values()))
    types = np.unique(list(name_to_cell_type.values()))
    time_bins = sorted(dataset.states.obs["time"].astype(float).unique())

    # multi-resolution labels + germ-layer colour per label (transition map)
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

    return Annotations(
        name_to_cell_type=name_to_cell_type,
        name_to_lineage=name_to_lineage,
        name_to_gl=name_to_gl,
        name_to_subtype=name_to_subtype,
        lineages=lineages,
        types=types,
        time_bins=time_bins,
        name_to_labels=name_to_labels,
        color_dict=color_dict,
    )
