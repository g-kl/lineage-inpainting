# concept (synthetic supplement)

companions to the main text synthetic figure. 

# panels

- **crossing**: the main text shows the performance of the models at the 7->8 transition. here we show this broken down across all transitions.

- **semi-synthetic baselines**: main text only shows lineage inpainting and GENOT and MMFM. here we show additional (including non-generative) baselines. we have two heatmaps: the cell type random motifs and the cell type context

- **distributional distance** (`plot_robustness.py`): in row c main text we use fate accuracy, here we plot fate accuracy against energy distance, Sinkhorn divergence and gaussian kernel MMD, each normalised by the motif's marginal.

# implementation

compute is split from plotting, same as the main figure: `compute_*.py` distill upstream eval outputs into `numerical_results/` (committed), `plot_*.py` + `assemble.py` read only those caches and write into `imgs/` (gitignored).

## reused code
- `synthetic_evals.figure.style` -- everything visual: method palette, type sizes, page width, rc, `methods_present`, `save_panel`, `thin_yaxis`. `style.py` here only adds paths and the metric labels, so the supplement cannot drift from the main page.
- `utils.svg_assembly` -- `prefix_ids` / `nest_panel` / `write_master`, shared with the main figure's `assemble.py`.
- `synthetic_evals.figure.glyphs` -- the motif-cartoon primitives, if a supplement panel needs the little trees.
- `synthetic_evals.core.metrics` -- `distances` / `grouped_distances` / `marginal_distances`: energy distance plus the two geomloss metrics. bandwidths use the median-pairwise-distance heuristic, not a fixed blur. `tests/test_metrics_agreement.py` checks that `viz.energy_distance` is the shared U-statistic (`analysis.lib.metrics.energy_distance`).

## compute scripts -> caches
`compute_robustness.py` -> `robust_abs.csv` (per motif x method: raw unclipped energy + that motif's marginal level), `robust_metrics.csv` (long form: one row per motif x method x metric, with the raw value, the marginal-normalised one, and the unclipped fate accuracy from `compute_landscape._type_acc`). reads `crossing_landscape/{landscape.csv,results.pkl}`. the alternate-metric columns only exist when the eval ran with `--extra-metrics`; without them the metric panels are simply omitted and the script says so.

## plotting + assembly
`plot_robustness.py` -> the whole robustness row. `plot_baselines.py` -> the widened clustermap row, `plot_context_realcells.py` -> the widened accuracy blocks, `plot_crossing_transitions.py` -> the crossing row, drawn with the main page's `plot_crossing._dim_panel` (none has a compute step of its own; all read the main figure's caches, which already hold every method the eval scored). `assemble.py` stacks the four rows (a crossing, b clustermap, c context, d robustness) into one master svg, `imgs/synthetic_supp.{svg,pdf,png}`, with a panel letter at each row's top left.

both main-figure heatmap plotters expose their drawers to the supplement: `plot_landscape._draw_{heat,flags,motif_ticks,row_labels,colorbar}` take `height` / `row_y` / `ann_y` / `heat_x0` / `heat_x1`, and `plot_context_realcells._draw_{split,titles,row_labels,colorbar}` take `height` / `row_y` / `title_y` / `hm_x0` / `hm_w` / `lab_x` / `gutter_x`. every one defaults to the main page's constants, so the supplement places the same blocks on its own canvas without a copy and without moving the main page a pixel. both supplement rows drop the cartoon above their blocks (it is in the main text) and spend the freed margin on the longer row labels. `compute_landscape` keeps whatever methods the eval produced (`methods_present`, so `METHODS + ABLATIONS`) but clusters the columns on `METHODS` only -- the ablations must not be able to reorder the columns, so both figures share one column order.

## run order
    PY=python
    PYTHONPATH=. $PY -m synthetic_evals.figure_supp.compute_robustness
    PYTHONPATH=. $PY -m synthetic_evals.figure_supp.assemble

`crossing` needs the main figure's `compute_crossing`. `baselines` needs no compute step here, but it does need the main figure's `compute_landscape` to have run since the eval (that is what puts the `wot` / `moslin` / ablation columns in `landscape_heat.csv`).