# tbx5

the figure shows that lineage inpainting predicts shifted heart trajectories for many of the sierant et al list of heart genes, and that beyond "heart is affected" it predicts which cells, at what times, and how sensitive to dose. it then tests the predicted Tbx5 shift in the heart against crispants.

row 1: concerning the CHD panel. (a) shows heart shift vs total shift (b) shows shift specificity vs shift time (c) shows haploinsufficiency: 50% knockdown vs 100% knockdown (d) shows dose sensitivity for Tbx5.

row 2: model predictions for tbx5. (e) shows trajectories shifted over time, at the subtype level. (f) shows Tbx5 expression along an example subtree (pruned to forelimb field and heart), raw and accumulated, with the onset marked. (g) shows when a trajectory becomes biased to its fate against when Tbx5 comes on along it. (h) shows the predicted nn distortion against how posterior the heart tube state is.

row 3: the crispants against the prediction. (i) shows the integrated latent over both datasets, crispants hued by lineage. (j) shows the E9.5 heart tube alone, hued by subtype - the window (k) and (l) are read on. (k) shows predicted and measured nn distortion over that window, one colorbar for both. (l) shows predicted against measured, one point per reference cell with the subtype means on top.

# implementation

three rows of four on one page-width sheet; rows 2 and 3 share their column widths in a different order so they line up. the two gpu scripts run under `rapids_singlecell`, everything else and all the plotting under `lt`.

## compute -> numerical_results/

- `compute_metrics.py` -> `metrics.csv` (a, b, c). per CHD gene: the heart and total share of the E9.0 readout shifted, at the full knockout and at 50%, plus the mean divergence onset.
- `compute_dose.py` -> `dose.csv` (d). share shifted per (dose, subtype). every dose is grown over the same trees, so a trajectory keeps its identity across doses, the denominator holds, and the 100% column is zero by construction.
- `compute_tbx5.py` -> `tbx5_devmap.npy`. Tbx5 per devmap cell, averaged over its `SMOOTH_K` devmap neighbours. raw counts of one gene in one cell are mostly zero, so every panel that reads Tbx5 on a node reads this. gpu, ~40 s.
- `compute_curves.py` -> `shift_curves.csv`, `shift_readout.csv`, `split_onset.csv` (e, g). one pass over the control trees, plus the knockout trees for the shift. the split table is one row per fate trajectory of a clone that makes both fates: the time its fates part and the time Tbx5 comes on along it.
- `compute_subtree.py` -> `subtree.csv`, `subtree_scan.csv` (f). picks the illustration subtree over every control tree: balanced forelimb field / heart leaf counts, few enough leaves to draw, root below `TBX5_ON`. without the balance term the pick is a lopsided tree whose Tbx5 comes on above the first branch, and a root that starts on has no onset to show.
- `compute_program.py` -> `program_scores.csv`, `program_genes.csv`, `program_fit.csv` (h). the A-P LDA (sklearn, Ledoit-Wolf shrinkage) on z-scored log1p CP10K over the devmap panel. runs after `compute_distortion.py`, since it is read on that script's heart tube cells, so a point in (h) is a point on row 3's maps. the in-sample auc is 1.0 by construction; the transferred auc is the check.
- `compute_umap.py` -> `umap_all.parquet`, `umap_heart.parquet`, `cell_counts.csv` (i, j). the whole-embryo layout is the pipeline's; the heart tube one is fit here, so it refits whenever `HEART_TUBE` changes. gpu.
- `compute_distortion.py` -> `distortion_{cells,heart,agreement,support}.csv` (k, l). re-cuts the pipeline's per-embryo tables and computes nothing itself.

## inputs

- the CHD screens (a-c): `heart_projection_screen_v2_dose{0,50}_E9.0_R1` under the grid-bridge experiment, scored here in the v2 lower annotation rather than the v1 vocabulary they were run in.
- the Tbx5 dose trees (d): `analysis/results/dose_response_broadsnap/Tbx5_trees`, one pkl per dose over one shared set of E9.0 trees.
- the Tbx5 screen (rows 2, 3): the `tbx5_broadsnap_screen_w1_dose0_E9.5_R*` replicate dirs, 10 trees over 3 reference replicates.
- the devmap counts and latent, for Tbx5 on the trees and for (h)'s features: `data/embryos/embryos_hvg_v5_scvi_devmap.h5ad`.
- the annotation, and the anterior / posterior SHF labels (h) is fit on: `final_subtype_v2_lower.csv`, `final_type_info_v2_lower.csv`. the factors (h) names: `literature/tf_list.txt`.
- everything else in row 3: `analysis/results/tbx5/support_comparison/` - the demultiplexed crispants, the E9.5 reference and its integrated latent, the transferred subtypes, the integrated umap and the per-embryo nn distortion tables. all taken as given.

## plot + assemble

one script per panel group, all reading only `numerical_results/`: `plot_metrics.py` (a-d), `plot_curves.py` (e, g), `plot_subtree.py` (f), `plot_program.py` (h), `plot_umaps.py` (i, j, and the heart window (k) and (l) share), `plot_distortion.py` (k), `plot_scatter.py` (l). `assemble.py` -> `imgs/tbx5_complete.{svg,png,pdf}`, with a to l at each panel's top left; `--no-regen` assembles the cached panel svgs instead.

`config.py` holds the paths, constants, palettes, sizes and the shared helpers (`figure`, `save`, `key`, `caption`, `window`, `hcbar`, `repel`). keys are placed in inches under their panel rather than drawn as matplotlib legends, and `repel` is what names the genes in row 1. `data.py` holds the loaders and asserts the pipeline's `SCREEN` / `OFF_RATIO` / `MODE` are the broadsnap / 1.1 / freeze the sheet claims. one colour per lateral plate subtype in `SUBTYPE`, shared by every panel that names one; the rest of the lineage is grey, and (h) takes none of them since it comes before row 3 names a subtype.

## run order

    cd analysis/figures/tbx5
    python compute_tbx5.py          # gpu
    python compute_umap.py          # gpu
    python compute_metrics.py
    python compute_dose.py
    python compute_curves.py        # after compute_tbx5
    python compute_subtree.py       # after compute_tbx5
    python compute_distortion.py
    python compute_program.py       # after compute_distortion
    python assemble.py              # renders every panel, then imgs/

`numerical_results/` and `imgs/` are both regenerable and neither is committed.
