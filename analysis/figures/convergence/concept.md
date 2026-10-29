# concept (fig 4)

the figure shows that OT-based models and lineage inpainting disagree about convergence events, and that endothelial cells and neurons are "convergent" through transcriptional program reuse: they first diversify along a spatial program, then "converge" by activating a common cell type identity program. across these convergence events it then splits the genes that tell the sources apart into "memory" genes and erased ones.

- **a** (row 1) three umaps of the E9.5 embryo: cells hued by lineage, by the OT-vs-tree ancestor discrepancy, and by the convergence score.
- **b** (row 2, left) per-cell-type convergence against OT discrepancy. one point per E9.5 cell type, both scores averaged over the leaves carrying both. both axes are normalized by the leaf's local spacing (see `compute_umap_scores.py`). blood-lineage types are dropped. two fate families are picked out against a grey backdrop -- neurons (any type with "neuron" in the name) and the endothelial types (`style.EC_CELL_TYPES`).
- **c** (row 2, right) UMAP showing the EC sources, a bar chart of source composition, and example subtrees under a few prominent sources.
- **d** (row 3, left) the same EC source-> target UMAP hued by program instead of by cell type. two maps -- endothelial cell, and spinal cord interneuron + somatic motor neuron pooled, since those two share a positional axis. each map is cropped to the window its own sources and leaves fill (1st to 99th percentile of each coordinate plus a 6% margin).
- **e** (row 3, right) trajectories of program score against time. one column per fate, identity on the top row and position on the bottom, one shared time axis. a dot every 0.25 time-units. the last point of a line is that group's own E9.5 leaves, joined to the end of the grid by a dashed segment. the endothelial column is split by source lineage (paraxial, lateral plate), the two neuron columns by `style.N_BINS` equal-count bins of the trajectory's E8.5 positional score, hued by panel d's color scale.
- **f** (row 4, left) one point per gene, one scatter per fate. x is the $|\log_2$ fc$|$ between the two sides of the positional axis at E8.5, y the same between the E9.5 leaves those sides reach. right of `DIV_XCUT` a gene tells the two sides apart at E8.5; above `DIV_YCUT` it still tells their descendants apart at E9.5. so top right is memory, bottom right is erased. the three columns share one pair of limits and one y axis.
- **g** (row 4, right) the same two classes over the six `style.G_FATES`, one normalised stacked bar per fate, most erased at the top. the two groups are positional, but taken from the annotation, not panel f's axis: the neuron axis is fit on hindbrain against spinal cord and puts nearly every midbrain and hypothalamic source on its anterior side. so a fate's E8.5 sources are pooled by lineage (paraxial, lateral plate) for the endothelial fate and by neural-tube region (forebrain, midbrain, hindbrain, spinal cord; `style.tube_x`) for the neurons, and its two biggest groups are compared -- hindbrain against spinal cord for cranial motor neuron, spinal cord interneuron and somatic motor neuron, forebrain against midbrain for midbrain and hypothalamic neuron. the fate itself (and, for an endothelial fate, everything else already endothelial) is excluded and sources are restricted to the fate's own germ layer. the bar throws the size away, so grey bars at the right carry the number of divergent genes behind each one.

the two program axes are LDA projection directions, fit on E9.5 cells: a positional contrast (lateral plate against paraxial lineage; `spinal cord *` against `hindbrain *` types) and an identity contrast (endothelium against the rest of mesoderm; a neuron type against the other neuron types).

# implementation

compute is split from plotting. `compute_*.py` do the dataframe/expression work and write caches into `numerical_results/`; the `plot_*.py` read only those caches. nothing outside this directory is read but the dataset and the two couplings. `style.py` carries the paths, the program definitions, the palettes, the geometry, `apply_rc`, `figure` and `save`; `data.py` is the shared loader for rows d-g; `trees.py` is the flat-tree representation rows d-g trace on. the scripts use sibling imports (`import style`, `import data`) and are run as plain scripts, not `-m`.

row a writes one full-width row svg into `imgs/`, the fan-in writes into `imgs/`, the rest into `imgs/panels/`. `assemble.py` nests every row/panel svg (each rendered at its true physical size, CM Sans as vector paths) into one master laid out in points, scales each row to the page width, and writes `imgs/fig4.{svg,png,pdf}`. it lays the figure out as four physical rows (a | b c | d e | f g) with per-boundary gaps in `assemble.GAP_AFTER_IN`. bold lowercase panel letters a-g are stamped at each panel's top-left via `utils/figure_labels.py` (Latin Modern Sans bold; faces in `utils/fonts/`). panel figsizes are tuned so each row tiles near scale 1 (fonts stay >= 6 pt).

## compute scripts -> caches

- `compute_scores.py` -> `convergence_df.pkl`, `ot_discrepancy_df.pkl` (a). scored across all tree timepoints. convergence, per time-T cell: KNN over the time-T embeddings, then trace(Cov(.)) of the T_anc = T - `CONV_T_ANC_OFFSET` ancestor embeddings pooled over the cell and its `CONV_K` neighbours. `CONV_K` is 100, the largest k whose neighbourhoods stay mostly inside one cell type; at 500 small types (endocardial cell, hepatoblast, first heart field) pool in their neighbours' ancestors and rise on that alone. OT discrepancy, per time-T leaf: the GENOT-W2 assignments share the model's {tree: {node: cell}} ids, so for each leaf we take its ancestor node nearest T_anc and compute the L2 (in X_scvi) between the model and GENOT cells assigned to it, attributed to the leaf. both scores are raw here; the figure normalizes them (next bullet).
- `compute_umap_scores.py` -> `umap_scores.pkl`, `umap_bg.pkl`, `top_conv_types.csv`, `celltype_conv_vs_ot.csv` (a, b; the last is also what panel g's fate list carries a convergence score from). attaches umap coords + cell_type/lineage from the annotation csvs, divides each leaf's OT L2 by its local spacing and its convergence by the square (mean X_scvi distance to its `SPACING_K` nearest same-stage cells; `ot_norm`, `conv_norm`), KNN-smooths both in X_scvi (`KNN_SMOOTH_K`; the `*_smooth` columns are what panel a displays), and ranks cell types by mean `conv_norm`.
- `compute_fanin.py` -> `{key}_leaves.pkl`, `{key}_composition.csv`, `{key}_subtrees.pkl` (c). parametrised by `style.FANIN`, one entry per fate (`ec`, `epi`, `cmn`). for each fate it traces every E9.5 focus leaf back to the chain node nearest the fate's `t_anc`, bins the source by the fate's `hierarchy`, highlights its source groups, places each leaf + ancestor on the umap, tallies the composition over progenitor nodes in the `BIN_LO`/`BIN_HI` window, and extracts per group the largest subtree that group dominates. `python compute_fanin.py [key ...]` runs all fates, or the given keys.
- `compute_axes.py` -> `axes.npz`, `axes.csv` (d-g). the two program axes per entry in `style.PROGRAMS` (`ec`, `sci`, `smn`): single Fisher (LDA) directions fit on the E9.5 devmap cells of that program's lineages, capped at `N_PER_TYPE` per (cell type, embryo) and centred within embryo so no contrast can be an embryo offset, with `RIDGE` shrinkage on the pooled within-group covariance and the identity axis orthogonalised against the positional one. `axes.csv` carries each axis's auc and a held-out auc, refit on one random half of the cells and scored on the other.
- `compute_paths.py` -> `cache_paths.npz` (d-g). every assigned node on the ancestor path of every E9.5 leaf that reaches a program's target, with its h5ad row and X_scvi latent, so a state scores on either axis by one matrix multiply. nodes below `T_MIN` are dropped, and each trajectory carries its source class (`group`) -- the nearest classifiable strict ancestor. builds `cache_trees.pkl` on the way; delete either cache to rebuild it.
- `compute_sources.py` -> `sources.csv`, `umap_links.csv`, `score_norm.csv`, `funnel.csv`, `src_types.csv` (d, e). one row per trajectory: the assigned ancestor nearest `T_SRC` and the E9.5 leaf it reaches, both with umap coordinates and both scored on the two axes. anchors in the wrong germ layer or already at the destination are dropped and `funnel.csv` counts both. also writes the `N_LINK` subsample panel d draws and the colour limit panels d and e share, so nothing about the figure is decided inside a plot script.
- `compute_phase.py` -> `phase_curves.csv`, `phase_endpoints.csv`, `phase_groups.csv`, `phase_bins.csv` (e). the score bins (equal-count, taken over both neuron panels together so a bin means the same thing in both), gaussian kernel means of both axes on the `GRID` time grid (bandwidth `BW`, grid points under `MIN_ESS` dropped), the E9.5 leaf mean per group, and per group where it starts, where it ends and when it is halfway there. reads `sources.csv`.
- `compute_divergence.py` -> `gene_divergence.csv` (f). the four cell groups per fate (two sides of the positional axis x source/leaf), capped at `DIV_CELLS` unique cells so an ancestor shared by several trajectories counts once, one read of the h5ad for all twelve, and per gene the two $|\log_2$ fc$|$ with its class. group means are over expm1(log1p) with a `DIV_EPS` pseudocount; a gene enters if it reaches a mean log1p of `DIV_MIN_EXPR` in one of a panel's four groups. its `means` and `coords` are the shared statistic, imported by the sweep.
- `compute_sweep.py` -> `sweep.csv`, `sweep_skipped.csv` (g). for each of the six `G_FATES`, traces its E9.5 leaves back to their `T_SRC` ancestor on the cached trees, pools the sources into positional groups (`style.ec_x` or `style.tube_x`), takes the two biggest and runs `compute_divergence`'s statistic on them. carries row b's convergence score through into the csv, and writes any fate that fails the two-group test to `sweep_skipped.csv`. reads `celltype_conv_vs_ot.csv`.

## plotting (read only numerical_results/)

`plot_umap_scores.py` -> `imgs/row1_umaps.{svg,png}` (a)
`plot_celltype_conv_vs_ot.py` -> `imgs/panels/celltype_conv_vs_ot.{svg,png}` (b)
`plot_fanin.py` -> `imgs/fanin_{key}.{svg,png}` (epi, cmn are variants); `compact=True` -> `imgs/fanin_ec_compact.{svg,png}` (c)
`plot_sources.py` -> `imgs/panels/sources.svg` (d)
`plot_phase.py` -> `imgs/panels/phase.svg` (e)
`plot_divergence.py` -> `imgs/panels/divergence.svg` (f)
`plot_sweep.py` -> `imgs/panels/sweep.svg` (g)

`plot_sources.py` is the one panel whose size is not fixed in `style.py`: its crop windows come from `style.umap_window` and the two umap boxes take their widths from those, so the panel width follows the data.

`assemble.py` -> `imgs/fig4.{svg,png,pdf}` (the master). regenerates every panel (each `plot_*.render`), reads their intrinsic point sizes, nests them, and stamps the panel letters; `--no-regen` reuses the panels already in `imgs/`. needs `cairosvg` (in the `lt` env). `imgs/` and `numerical_results/` are gitignored at the repo root.

## source data

one dataset and one model config, both named in `style.py`: the lineage-inpainting coupling `mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_6006f...` step 20k (the adj_n8 splitter, `ASSIGN_PKL`), and the dataset files under `../data/embryos/`.

- `qiu24_integrated_hvgs_and_scvi.h5ad` supplies `obsm/X_scvi` and the obs annotation everywhere, and the expression rows rows f-g read; `final_topology.h5td` the tree topology; `final_umap.csv`, `final_type_info_v2_lower.csv` and `final_subtype_v2_lower.csv` the annotation, lineage/germ-layer maps and umap coordinates.
- `compute_scores.py` additionally reads the GENOT-W2 coupling (the `assignments` key in `outputs/e_baselines/genot_w2_unified_model.pkl`) as the OT baseline. it is pure entropic OT -- the canonical "OT" -- where FGW is structure-aware and agrees too closely with the tree to serve as the OT contrast.
- the program axes and the trajectories rows d-g run on are fit and traced here, by `compute_axes.py` and `compute_paths.py`. nothing is read from another figure or analysis directory.

## run order

    python analysis/figures/convergence/compute_scores.py        # slow, GPU + dataset (a)
    python analysis/figures/convergence/compute_umap_scores.py   # a, b
    python analysis/figures/convergence/compute_fanin.py         # c: ec, epi, cmn
    python analysis/figures/convergence/compute_axes.py          # d-g: the program axes
    python analysis/figures/convergence/compute_paths.py         # d-g: trajectory cache
    python analysis/figures/convergence/compute_sources.py       # d, e
    python analysis/figures/convergence/compute_phase.py         # e; reads sources.csv
    python analysis/figures/convergence/compute_divergence.py    # f; reads h5ad
    python analysis/figures/convergence/compute_sweep.py         # g; reads h5ad + celltype_conv_vs_ot.csv
    python analysis/figures/convergence/assemble.py              # master fig4.{svg,png,pdf}
