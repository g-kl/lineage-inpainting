#!/usr/bin/env python
"""compute step for the fate-recipe subway panel.

reads analysis/pipelines/recipe/fate_tfs.csv (+ the lineage-level csv), picks the most
abundant cell types, filters each (group, time bin) to TFs passing the penetrance
and specificity cutoffs and keeps the top few, then orders the cell types into
lineage groups (hued by germ layer) and writes one subway_data_<rank>.json per
ranking for the plotting step.  the inner E7.5 bin is computed at lineage level,
the outer bins at cell-type level, with their own specificity cutoffs.
"""

import os
import json

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE_DIR = os.path.join(HERE, "..", "..", "pipelines", "recipe")
RECIPE_CSV = os.path.join(RECIPE_DIR, "fate_tfs.csv")
RECIPE_LINEAGE_CSV = os.path.join(RECIPE_DIR, "fate_tfs_lineage.csv")
DATA_DIR = os.path.join(HERE, "..", "..", "..", "..", "data", "embryos")

N_TYPES = 48
MIN_PENETRANCE = 0.2
MIN_SPEC_OUTER = 1.5         # specificity cutoff for the outer cell-type bins
MIN_SPEC_INNER = 0.5         # specificity cutoff for the inner lineage bin
TOP_PER_BIN = 3
RANK_BY = ["penetrance", "specificity"]   # one json + one panel per ranking
# the inner (lineage) bin always ranks by specificity: at lineage level penetrance
# is dominated by broad early disruptors (Hmga2, P4hb, ...) that hit every lineage,
# so only specificity surfaces the lineage-specifying TFs.
INNER_RANK_BY = "specificity"
GERM_ORDER = ["ectoderm", "mesoderm", "endoderm", "epiblast", "primordial germ cell"]


def top_tfs(frame, group, t, rank_by, min_spec):
    """top TFs for one group at one time bin, passing both cutoffs, by rank_by.
    (frame's 'cell_type' column holds whatever grouping the frame was built at.)"""
    sel = frame[(frame["cell_type"] == group) & (frame["time"] == t)
                & (frame["penetrance"] >= MIN_PENETRANCE)
                & (frame["specificity"] >= min_spec)]
    return sel.sort_values(rank_by, ascending=False).head(TOP_PER_BIN)["tf"].tolist()


def cell_type_info():
    """cell_type -> (lineage, germ_layer) and cell_type -> abundance."""
    info = pd.read_csv(os.path.join(DATA_DIR, "final_type_info.csv"), index_col=0)
    sub = pd.read_csv(os.path.join(DATA_DIR, "final_subtype.csv"), usecols=["cell_subtype"])
    abundance = sub["cell_subtype"].map(info["cell_type"]).value_counts()
    mode = lambda s: s.mode().iat[0]
    lineage = info.groupby("cell_type")["lineage"].agg(mode)
    germ = info.groupby("cell_type")["germ_layer"].agg(mode)
    return lineage, germ, abundance


def main():
    df = pd.read_csv(RECIPE_CSV)
    df_lin = pd.read_csv(RECIPE_LINEAGE_CSV)
    lineage, germ, abundance = cell_type_info()

    present = [c for c in abundance.index if c in set(df["cell_type"])]
    types = present[:N_TYPES]
    bins = sorted(df["time"].unique())
    inner_bin, outer_bins = bins[0], bins[1:]   # 7.5 is lineage-level, rest cell-type

    # order: lineage groups (sorted by germ layer then group abundance), cell types
    # within a group by abundance.
    rows = pd.DataFrame({"cell_type": types})
    rows["lineage"] = rows["cell_type"].map(lineage)
    rows["germ_layer"] = rows["cell_type"].map(germ)
    rows["abundance"] = rows["cell_type"].map(abundance)
    grp = rows.groupby("lineage")["abundance"].sum()
    germ_rank = {g: i for i, g in enumerate(GERM_ORDER)}
    lin_germ = rows.groupby("lineage")["germ_layer"].first()
    rows["lin_key"] = rows["lineage"].map(
        lambda l: (germ_rank.get(lin_germ[l], 99), -grp[l]))
    rows = rows.sort_values(["lin_key", "abundance"], ascending=[True, False])

    for rank_by in RANK_BY:
        cells = []
        for _, r in rows.iterrows():
            tfs = {str(b): top_tfs(df, r["cell_type"], b, rank_by, MIN_SPEC_OUTER)
                   for b in outer_bins}
            cells.append({"cell_type": r["cell_type"], "lineage": r["lineage"],
                          "germ_layer": r["germ_layer"],
                          "abundance": int(r["abundance"]), "tfs": tfs})
        lineage_tfs = {l: top_tfs(df_lin, l, inner_bin, INNER_RANK_BY, MIN_SPEC_INNER)
                       for l in dict.fromkeys(rows["lineage"])}
        out = os.path.join(HERE, f"subway_data_{rank_by}.json")
        json.dump({"time_bins": bins, "inner_bin": inner_bin, "cells": cells,
                   "lineage_tfs": lineage_tfs, "rank_by": rank_by,
                   "top_per_bin": TOP_PER_BIN},
                  open(out, "w"), indent=1)
        n_tf = sum(len(t) for c in cells for t in c["tfs"].values())
        print(f"wrote {out}: {len(cells)} cell types, {n_tf} outer ticks, "
              f"{sum(len(v) for v in lineage_tfs.values())} lineage ticks")


if __name__ == "__main__":
    main()
