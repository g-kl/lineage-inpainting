"""Regenerate the division-rate figures from the saved CSV outputs.

No torch / dataset / GPU needed -- reads only the small per-figure tables that
``analysis/pipelines/division_rate/explore.ipynb`` writes next to the model, so it
runs without a GPU.
Plotting is shared with the notebook via ``figstyle`` so the figures are identical.

    python analysis/figures/division_rate/replot.py [model_tag]

``model_tag`` defaults to the best run (``extender_count_energy_nd32_wd1e4``); pass
e.g. ``extender_count_energy`` to redraw the baseline. Set ``EXP`` to point at a
different model directory.
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")

from figstyle import (apply_style, save_fig, fig_celltypes, fig_genes,
                      fig_calibration)

# same run the other figures read; repo-relative so it works outside one machine.
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EXP = os.environ.get(
    "EXP",
    os.path.join(BASE_DIR, "outputs",
                 "mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_"
                 "57ab441ec3fcecf81d545755db5ddb09"),
)
TAG = sys.argv[1] if len(sys.argv) > 1 else "extender_count_energy_nd32_wd1e4"
D = 0.5

# data tables live next to the model; figures go to the repo figures subdir
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
apply_style()


def table(name):
    return os.path.join(EXP, f"divrate_{TAG}_{name}")


tab = pd.read_csv(table(f"by_cell_type_d{D}.csv"), index_col=0).sort_values(
    "div_rate", ascending=False)
for which in ("top", "bottom"):
    save_fig(fig_celltypes(tab, which), FIGDIR, TAG, f"celltypes_{which}")

gc = pd.read_csv(table("gene_corr_d0.5.csv"), index_col=0).iloc[:, 0].sort_values()
save_fig(fig_genes(gc), FIGDIR, TAG, "genes")

val = pd.read_csv(table("calibration_d0.5.csv"), index_col=0)
save_fig(fig_calibration(val), FIGDIR, TAG, "calibration")

print(f"wrote 4 figures (pdf+png) for tag '{TAG}' to {FIGDIR}")
