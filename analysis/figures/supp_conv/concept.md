# concept (convergence supp)

supporting panels for fig 4

- **a** table of ten most and least convergent cell types.
- **b** repeat of fig4b, but with the unnormalized scores (raw cov trace and ancestor L2)
- **c** all of the linear discriminant gene programs.
- **d** sensitivity of erased fraction to threshold choice

# implementation

compute writes small caches into `numerical_results/`, the plot scripts and `assemble.py` read only those and write into `imgs/` (both gitignored). all cpu, a few seconds each.

`style.py`: paths, `C` (fig 4's style), the panel constants, 10 / 9 pt type with the table and strip rows at 8 pt, `figure`, `save_svg`. the scripts are run with `-m` from the repo root and import fig 4 by package path (`analysis.figures.convergence`).

## compute -> numerical_results/

- `compute_celltypes.py` -> `celltype_scores.csv` (a, b). reads fig 4's `umap_scores.pkl` and runs fig 4's own per-type aggregation (`compute_umap_scores.conv_vs_ot`) twice, once on the normalized columns and once on the raw ones, so both rest on the same dual-scored leaves and the same `MIN_CELLS` cut. drops blood and ranks both.
- `compute_genes.py` -> `gene_corr.csv`, `genes.csv` (c). the axes are fig 4's (`axes.npz`) and so is the fit population (`compute_axes.population()`). a gene is ranked by its correlation with an axis score over up to `N_GENE_CELLS` of the cells that axis was fit on, both centred within embryo, not over the whole lineage: over all of neural ectoderm the anterior end picks up Otx2, which no hindbrain or spinal cord cell expresses. the columns come from fig 4's `sources.csv`. the unresolved `hindbrain` and `spinal cord` calls are dropped from the neuron sources, since their own subdivisions cover them. a block keeps the first `N_GENES` of its `N_POOL` candidates that reach `MIN_EXPR` in some column.
- `compute_cutoffs.py` -> `cutoffs.csv` (d). reads fig 4's `gene_divergence.csv`. per fate and (xcut, ycut), a gene is divergent if `x >= xcut` and erased if also `y < ycut`, the inequalities of `compute_divergence.coords`. asserts that the fig 4 cell reproduces the csv's classes.

## plot + assemble -> imgs/

`plot_table.py` -> a, `plot_raw.py` -> b, `plot_genes.py` -> c, `plot_cutoffs.py` -> d, each into `imgs/panels/`. `assemble.py` -> `imgs/supp_conv.{svg,png,pdf}` through `utils.svg_assembly`: a | b, then c | d, the second panel of each row flush right, 6.5 in wide.

## run order

    PY=python
    $PY -m analysis.figures.supp_conv.compute_celltypes
    $PY -m analysis.figures.supp_conv.compute_genes
    $PY -m analysis.figures.supp_conv.compute_cutoffs
    $PY -m analysis.figures.supp_conv.assemble

fig 4's caches have to exist first (see the run order in `analysis/figures/convergence/concept.md`): `umap_scores.pkl`, `axes.npz`, `sources.csv`, `gene_divergence.csv`.
