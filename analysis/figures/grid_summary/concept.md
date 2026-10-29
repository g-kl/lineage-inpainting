# concept (grid summary)

the mouse-embryo model grid, one figure per question, every figure reporting the
SAMPLING model as the coupler -- the splitter used as the pooler, which is how the
main-text figure couples the main-text model. six page-width (6.5 in) rows drawn with
the shared palette, type and svg assembly of the main-text figures; no panel letters
(supplement).

the grid is the learned-pool runs: sampling model {One-step, CFM, VP-SDE} x imputing
model {Tree Tx, Tree Tx + baseline} (the additive baseline, dropped from or kept in
BOTH the imputing and the sampling model of a run) x pairing {imputer, imputer (sg),
nearest neighbor, nearest neighbor (sg), alternating}, all trained with the full
aligner, plus the aligner-feature ablation on the main-text run (VP-SDE + Tree Tx +
baseline + alternating). the pairing is what the splitter trains on: the imputer's own
nodes (`id`), their nearest real cells (`nn`), or the two on alternate steps
(`idnn_leaf`, the main-text run's); "(sg)" stops the splitter's gradient at the
imputer. the stop-gradient runs appear in the stop-gradient strip only
(`style.PAIRINGS_SHOWN`). the parameter-free pools, LineageOT and the GENOT couplings
are not part of it (the published baselines live in the main-text figure).

columns are the main-text metrics row's six (`analysis/figures/summary_qc/plot_metrics`,
its names) plus three of the grid's own, in two groups that an overbar above the
column titles labels with the main figure's words (its panels c-d and e-f):

    ancestral inference:  marginal distortion | marginal cell-type TV |
                          lineage oscillation | linkage distortion |
                          germ-layer crossing | Hox-code distortion
    descendant inference: clonal distortion | marginal distortion |
                          marginal cell-type TV

every metric is lower = better. the descendant pair repeats the ancestral pair's
titles and the overbar says which side (metrics_wide.csv keeps them apart as
`marginal` / `label_tv` and `spl_energy` / `lab_cell_type_tv`). the descendant pair is
scored per clonal subtree against that subtree's own true leaves, so its "marginal" is
the figure's shorthand for "the generated cloud against its real counterpart", not a
time-marginal. the ancestral group scores the inferred ancestors: the marginal
distortion is the figure's (per adjacent-timepoint pair, the splitter pools sampled
cells back to their ancestors at the earlier time; energy distance against sampled
real cells observed then), the marginal cell-type TV is scored on the SAME pooled-back
states (each snapped to its nearest real cell at that time, as the descendant TV snaps
regenerated descendants; TV between their cell-type composition and that of every tree
leaf observed then; mean over the same pairs), and the four assignment metrics are the
figure's, on the coupling. the descendant group scores the splitter on the figure's
clonal subtrees: the clonal distortion pools a subtree's leaves to the root and splits
back (energy distance against the real leaves, a cycle); the descendant distortion
splits the SAME subtree from the real cell the run's coupling assigned to its root --
the ancestor replaced by a real cell, the coupling's own snap-to-real, so the pooler's
error is out of the picture -- against the real leaves; the descendant cell-type TV is
scored on the same generated leaves (snapped to real cells at the leaf time,
composition against the true leaves'). the two groups are therefore mirror images:
pool real cells up and compare with the real marginal, or start from a real ancestor
and split down and compare with the real descendants, each with an energy distance and
a cell-type TV on the same samples. the strips and the lollipops share the order and
the overbars (`style.METRIC_GROUPS`, `style.group_bars`).

- **ablation strips** (`plot_ablations`): one row per run, one column per metric, one
  dot per level of the ablated axis: the baseline level an open ink dot, the main-text
  run's level a teal dot, any other learned level a tint of it. every metric is lower
  = better, so left = better. each column is log-spaced from the largest 1-2-5 value
  at or below its smallest value (a round floor, close to the data) to just past its
  largest, scaled to the rows the strip shows (`style.scale_axis`): the best runs
  crowd the low end, where a linear axis compresses exactly the differences that
  matter, and a baseline that is 5-10x worse (Tree Tx on the lineage metrics) would
  otherwise squeeze our side into a few points. gridlines at the 1-2-5 positions and
  minor ticks between them make the log spacing visible; a column too narrow for three
  1-2-5 ticks (Hox, the coupling TV: under 2x across the grid) gets a line at every
  integer multiple instead. the floor comes from the positive values, and a value at
  or below it (the one U-statistic energy distance <= 0, an identity run's descendant
  distortion at 20k: its regenerated cloud is indistinguishable from the truth) sits
  on the spine. tick labels sit centred under their ticks, the floor's starting at the
  spine like an origin label (so no label reaches into the gap from the left, and a
  top label may use the gap on the right); a column labels as many 1-2-5 ticks as fit
  (measured), else the floor and the highest tick whose label fits, else the floor
  alone. one short "left = better" caption per strip. row groups are bracketed on the
  left (one bracket level per grouping axis; the stop-gradient strip nests bridge
  inside pool). no arrows, no connectors, no value annotations.
    - `sampling_family`: One-step (baseline), CFM, VP-SDE, over the pairings of the
      main pool.
    - `delta_vs_tree`: Tree Tx (baseline level), Tree Tx + baseline, over bridge x
      pairing: the additive baseline's effect. like every column, scored with the
      sampling model as the coupler, so this is chiefly the sampling model with and
      without its residual frame (the imputing model it trained with changes too).
    - `stopgrad`: the stop-grad run (open dot) and the same pairing with the gradient
      flowing into the imputing model (teal dot), over bridge x pairing family, where
      the families are imputer, nearest neighbor and alternating (the main-text run's,
      gradient flowing only, so one dot on its row).
- **aligner feature ablation** (`plot_lollipops`, `aligner_ablation`): the main-text
  run with the `StratifiedFeatureAligner` feature set ablated (marginal only / + path
  / + common ancestor / + both = the main-text run) at its raw values as horizontal
  lollipops from 0, exactly as the main-text metrics row draws them (hline + dot,
  value above, no x axis); a tint ramp into the model teal as features are added.
  values are printed because the four runs lie within 10-60% of each other.
- **transition-graph similarity** (`plot_similarity`, `ct_similarity`): run x run
  cosine similarity of each run's (cell type, t) -> (cell type, t + dt) transition
  graph under the sampling model's coupling (edge weight = mean of in- and
  out-composition), over every run some other grid figure shows plus the plain Tree Tx
  stop-gradient runs (`style.in_heatmap`: the main pool and the plain Tree Tx pool
  with all five pairings, the aligner ablation), grouped by sampling model (outer
  brackets) and imputing model (inner); the pairing is on the ticks, and the aligner
  ablation is a trailing group of its own with the feature set on its ticks. the
  matrix is symmetric with a unit diagonal, so only the strictly lower triangle is
  drawn: row labels and brackets on the left, the column labels along the staircase in
  the empty half (one per row, unrotated), the colourbar in the empty corner. the
  colour is the log of the dissimilarity (1 - cosine) over the off-diagonal range,
  `crest` sequential, the colourbar ticked in similarity: the plain Tree Tx runs sit
  at 0.2-0.5 against everything else and a linear scale would put every pair of
  main-pool runs (0.97-1) into one shade.
- **sampler vs imputer** (`plot_sampler_vs_imputer`, `sampler_vs_imputer`): the
  marginal distortion and its cell-type TV with the splitter doing the pool-back
  against the same samples pooled back by the pool model applied once
  (`marginal_pool_distortion_<cfg>_step{S}.csv`, `compute_distortion.model_marginal`
  with its `pool_back` hook), and the four assignment metrics scored on the sampling
  model's coupling against the same metric on the imputing model's coupling (the pool
  model applied once); one point per run over the main pool's full-aligner runs,
  square panels with both axes from 0 and the diagonal as the reference: below it the
  sampling model couples better. the dots are the strips' key (open ink one-step, tint
  CFM, teal VP-SDE).

# consistency with the main-text figure

the main-text figure pipeline (`analysis/figures/summary_qc`) is the ground truth for
every number here; the grid never re-implements a metric:

- checkpoint: `summary_qc.style.STEP` (20k), the step the figure scores the main-text
  run at. 5k is rendered alongside as a companion (`--step 5000`).
- coupling: the splitter as the pooler at `style.coupling_n` = 8 integration steps for
  the bridges (`summary_qc.style.MODEL_ASSIGNMENTS` is the main-text run's n = 8 coupling); the
  one-step splitters have no integration (n = 1). 512 trees (`summary_qc.style.N_TREES`).
- clonal + marginal distortion: `compute_distortion.model_clonal` / `model_marginal`
  on the figure's own seeded subtree / cell samples (`collect_clonal_candidates`,
  `collect_marginal_samples` with its style constants), so every run and the figure
  score the same 400 subtrees and the same adjacent-timepoint cells.
- lineage oscillation, linkage distortion, germ-layer crossing:
  `compute_assignment_metrics.compute_for` on the coupling.
- Hox-code distortion: `compute_hox.collect_pairs` on the figure's common support --
  ancestors that the run's coupling AND both GENOT couplings assign -- then the
  figure's per-(cell type, gene) Pearson rule.
- aggregation: as `plot_metrics._metric_values` -- clonal / marginal = mean over the
  samples, germ-layer crossing = pooled over every link, linkage = mean of the
  distortion over the two well-conditioned cuts (t9.5->t9.0, t9.0->t8.5), Hox = mean
  over the (cell type, gene) pairs, oscillation = the single value.
- transition graphs: `analysis.lib.celltype_map.eval.build_cell_type_map` at the
  figure's map rule (`summary_qc.style`: no PMI cut, summed in / out shares >= 0.2,
  per-time label resolution, every pair) on the same coupling.
- the marginal cell-type TV (`label_tv`) is the label analogue of the marginal
  distortion on its own samples: `compute_distortion.model_marginal` hands each pair's
  pooled-back states to a hook (`on_transport`), `eval_core.runner.LabelScorer`
  snaps them to the nearest real cells at the earlier time and writes the TV as the
  `cell_type_tv` column of the marginal distortion CSV; mean over the same pairs.
- the descendant distortion (`descendant`) and its cell-type TV (`descendant_tv`):
  `compute_distortion.model_clonal` on the figure's own clonal candidates (the same
  400 seeded subtrees, the same rollout), with its `root_state_of` hook anchoring
  each root at the real cell the run's sampling-side coupling (n = `coupling_n`)
  assigned to it instead of the pooled-back state, and its `on_split` hook snapping
  the generated leaves to real cells at the leaf time for the TV against the true
  leaves' composition; `descendant_distortion_<cfg>_step{S}.csv`, mean over the
  subtrees (`eval_core.runner.run_model_metrics`). a subtree whose root the coupling
  did not assign is skipped (the job prints the count).

reproducibility: the marginal distortion and every cell-type TV reproduce exactly
under the runner's seed; the clonal and descendant passes do not -- whole batches of
32 subtrees come out different between two runs of the same code and seed (a
batch-level nondeterminism of the splitter's rollout on the GPU), moving the
400-subtree mean by about 1% run to run, far below the gaps between runs.

the main-text run's row therefore reproduces the main-text figure's model numbers,
and `compute_grid` asserts the main-text run (`style.HERO_CFG` =
`summary_qc.style.EXPERIMENT_DIR`) is in the grid with the expected axes.

# implementation

compute is split from plotting. `compute_grid.py` distils the per-run eval artefacts
into small caches under `numerical_results/<step>/` (gitignored); `plot_*.py` +
`assemble.py` read only those and write the master renders into `figures/<step>/`
(committed; the panel svgs go to `imgs/<step>/`, gitignored). paths, axis vocabulary
(directory tag -> display name, canonical orders), palette, type sizes, geometry and rc
live in `style.py`.

## reused code (imported, not copied)
- `analysis.figures.summary_qc.style` -- STEP, the main-text run, the metric constants.
- `utils.palettes` -- the model teal, the similarity colormap, the neutrals.
- `utils.figure_labels` -- `register_fonts` / `FONT_FAMILY` (Latin Modern Sans).
- `utils.svg_assembly` -- `new_master` / `nest_panel` / `write_master` (svg + pdf + png).

## axis vocabulary (run-dir tag -> axis value)
`outputs/mouse_embryo_grid_bridge_unshared_<bridge>_<pairing>_nosa_std<suffix>_<hash>/`,
decoded by `compute_grid.parse_dirname`. bridge: `fullen` = one_step, `cfm`, `vpsde`.
pairing: `id`, `id_sg`, `nn`, `nn_sg`, `idnn_leaf`. suffix: `_none` = Tree Tx + baseline (the
`dtfm` pool: the net predicts the residual over an additive baseline),
`_nondelta` = Tree Tx (all with the full aligner), `_alignmarg` /
`_alignpath` / `_alignca` = the main-text run with the aligner ablated. anything else (graft /
selfgraft runs, the `_ft_none` Flat Tx pool, `_meanpool` / `_fgwpool` / `_w2pool`, the
`_ot_` pairing) is not part of the grid and is skipped.

## compute -> caches
`compute_grid.py --step S` -> `numerical_results/<S>/`
- `configs.csv`: cfg + axes (bridge, pairing, pool, aligner) for every run with a
  checkpoint at S.
- `coverage.csv`: runs scored per (eval kind, n); printed as the sanity table.
- `metrics_wide.csv`: one row per run, the nine `style.METRICS` columns (sampling side)
  and the four `cpl_*` columns (the assignment metrics on the imputing-side coupling).
- `similarity.csv`: the run x run cosine matrix in canonical (bridge, pool, pairing)
  order.

## data sources (per run dir; made by scripts/grid_eval, see its README)
- `assignments_nn_<cfg>_step{S}_adj_n{n}[_splitter].csv` + `metrics_..csv` -- the
  coupling and its assignment metrics (`analysis.lib.eval_core`).
- `{clonal,marginal,descendant,marginal_pool}_distortion_<cfg>_step{S}.csv` -- the
  figure's model rows, the grid's descendant pass and the imputing model's marginal
  pass (`analysis.lib.eval_core.runner.run_model_metrics`); all but the clonal rows
  carry `cell_type_tv`.
- `cell_type_map_<cfg>_step{S}_adj_n{n}_splitter.pkl` -- the transition graph.

## run order
    python scripts/grid_eval/launch_consistency_recompute.py --dry-run   # what is missing / stale
    python scripts/grid_eval/launch_consistency_recompute.py --then-render   # submit it (slurm) + render
    python -m analysis.figures.grid_summary.compute_grid                 # style.STEP = 20000
    python -m analysis.figures.grid_summary.compute_grid --step 5000
    python -m analysis.figures.grid_summary.assemble                     # every figure at style.STEP
    python -m analysis.figures.grid_summary.assemble --step 5000
    python -m analysis.figures.grid_summary.assemble --figure stopgrad   # one figure
