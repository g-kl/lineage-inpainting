# Grid evaluation pipeline

How a trained pool-split run on the mouse-embryo grid gets turned into the
numbers the grid-summary figures plot
([`analysis/figures/grid_summary/`](../../analysis/figures/grid_summary/), see its
`concept.md`). The main-text figure pipeline
([`analysis/figures/summary_qc/`](../../analysis/figures/summary_qc/)) is the ground
truth: every metric here is computed by its code, on the same seeded samples, at the
checkpoint and coupling setting it uses for the main-text model.

## Overview

Each run in the sweep lives at

```
outputs/mouse_embryo_grid_bridge_unshared_<bridge>_<pairing>_nosa_std<suffix>_<hash>/
```

with checkpoints `pool_ema_step{5000,10000,15000,20000}.pt` and
`split_ema_step*.pt`. The evaluation drops its artefacts next to the checkpoint;
the figure's compute step (`compute_grid.py`) keys off the filename patterns below.

The shared driver is [`analysis/lib/eval_core/`](../../analysis/lib/eval_core/):
`runner.py` stages the work (couple -> assignment metrics -> distortion) and owns
the on-disk schema; `metrics.py` borrows the main-text figure's metric code
(`compute_assignment_metrics.compute_for`, `compute_hox.collect_pairs`); the
distortion stage calls `compute_distortion.model_clonal` / `model_marginal`
directly with the figure's style constants, re-using `model_clonal` for the
descendant pass (its `root_state_of` hook anchors the root at the coupled real cell)
and reading the cell-type TVs through their `on_transport` / `on_split` hooks
(`runner.LabelScorer`).

## Artefacts and the scripts that make them

| Kind | Script (this dir) | Entry point | Output (per run dir) |
| ---- | ----------------- | ----------- | -------------------- |
| **cpl** imputing-side coupling (pool model applied once, n = 1) | `eval_coupling.sbatch` | `eval_coupling_one_config.py` (eval_core stages 1 + 2) | `assignments_nn_<cfg>_step{S}_adj_n1.pkl` + `metrics_<cfg>_step{S}_adj_n1.csv` |
| **cplspl** sampling-side coupling (splitter as the pooler, n = `POOL_N_STEPS`) | `eval_coupling_splitter.sbatch` | same, with `USE_SPLITTER_AS_POOLER=1` | `..._adj_n{N}_splitter.pkl` + `metrics_..._adj_n{N}_splitter.csv` |
| **distortion** clonal + marginal + descendant (splitter rollouts), marginal_pool (the imputing model's pool-back) | `eval_distortion_batch.sbatch` | `eval_distortion_batch.py` (stage 3, batched; needs the cplspl coupling; only the missing CSVs unless `OVERWRITE`) | `clonal_distortion_<cfg>_step{S}.csv`, `marginal_distortion_<cfg>_step{S}.csv` (+ `cell_type_tv`), `descendant_distortion_<cfg>_step{S}.csv` (+ `cell_type_tv`), `marginal_pool_distortion_<cfg>_step{S}.csv` (+ `cell_type_tv`) |
| **refresh** re-score existing couplings | `refresh_metrics_batch.sbatch` | `refresh_metrics_batch.py` (stage 2, batched, overwrite) | rewrites the `metrics_*.csv` above |
| **ctmap** cell-type transition graph | `eval_cell_type_map_batch.sbatch` | `eval_cell_type_map_batch.py` (batched) | `cell_type_map_<cfg>_step{S}<suffix>.pkl` (+ curated rows appended to the matching metrics CSV) |

The metrics CSV (columns `metric, tree, value`) carries `germ_layer_crossing` (the
`all` row plus one per time bin), `lineage_oscillation_fraction`,
`linkage_correlation_normalized` / `linkage_distortion` per cut, and
`hox_correlation` / `hox_distortion` per (cell type, Hox gene), scored on the
main-text figure's common support (ancestors that the run's coupling AND both
GENOT couplings assign; `analysis.lib.eval_core.metrics.GENOT_ASSIGNMENTS`).

## The canonical settings (from the main-text figure)

| setting | value | where it is defined |
| --- | --- | --- |
| checkpoint | 20000 (5k as a companion) | `summary_qc.style.STEP` |
| trees | 512 (the data has 56) | `summary_qc.style.N_TREES` |
| coupling n, sampling side | 8 for the bridges, 1 for the one-step splitters | `grid_summary.style.coupling_n`; the main-text model's `summary_qc.style.MODEL_ASSIGNMENTS` is n = 8 |
| coupling n, imputing side | 1 (the pool model is one-shot) | -- |
| distortion | `N_SUBTREE_LEAVES=400`, `CLONAL_OFFSET=1.0`, `CLONAL_TIME_TOL=0.5`, `N_MARGINAL_CELLS=400`, `MARGINAL_TIME_TOL=0.25`, `SPLIT_STEPS=8` (= the coupling's n), `MAX_NODES=256`, `BATCH_SIZE=32`, `SEED=212121` | `summary_qc.style` |
| transition graph | the figure's map rule: `PMI_QUANTILE=0` (no cut), `MIN_COMPOSITION=0.2` on the `sum` of in / out shares, `RESOLUTION_BY_TIME`, every pair | `summary_qc.style`, mirrored by `eval_cell_type_map_batch.py` |
| descendant (grid-only) | the figure's clonal subtrees split from the root's coupled real cell (the sampling-side coupling at its n) instead of the pooled-back root | `eval_core.runner.run_model_metrics` |
| cell-type TVs (grid-only) | each generated cloud (pooled-back ancestors; generated leaves) snapped to the nearest real cells observed at its time, composition vs the real one (the cells observed then; the subtree's true leaves) | `eval_core.runner.LabelScorer` |
| marginal_pool (grid-only) | the marginal distortion's samples pooled back by the pool model applied once (`pool_batch`, as the imputing-side coupling), for the sampler-vs-imputer figure | `eval_core.runner.run_model_metrics` |

### The descendant pass

The grid's descendant columns mirror its ancestral ones. The marginal distortion
pools real cells up and scores the pooled-back ancestors against the real cells at
the earlier time; the descendant distortion takes the figure's clonal subtrees,
anchors each root at the real cell the coupling assigned to it, splits down and
scores the generated leaves against the true leaves. Both get a cell-type TV on the
same generation by snapping it to real cells, the coupling's own snap-to-real.

## Bringing the grid in line: `launch_consistency_recompute.py`

```bash
python scripts/grid_eval/launch_consistency_recompute.py --dry-run   # what is missing / stale
python scripts/grid_eval/launch_consistency_recompute.py --smoke     # one job per kind
python scripts/grid_eval/launch_consistency_recompute.py             # submit everything
```

It derives the work from `outputs/` for every grid run
(`analysis.figures.grid_summary.compute_grid.grid_configs`) and submits it in
dependency order: missing couplings (one job each) -> distortion batches (a clonal
CSV without the figure's `method` column counts as stale) -> refresh batches (after the couplings) -> transition-map batches where the
map is missing (`--force-maps` rebuilds all; after the couplings and the refresh,
since the refresh and the maps write the same metrics CSVs). a marginal CSV without
`cell_type_tv` or a missing descendant CSV is stale too. `--only <kind>` (repeatable) restricts the
kinds, `--after <jid...>` adds dependencies, `--then-render` appends a
`render_grid.sbatch` job (caches + figures at every step) after everything. Batch
jobs load the dataset once (`BATCH` env var, `;`-separated `<dir>|<step>|<n>` tuples
for the distortion, run dirs or coupling pickles otherwise). Every job is idempotent, so a requeue or a re-run of
the launcher only does what is still missing. A JSON manifest of the submitted job
ids lands in `logs/`.

New runs slot in the same way: train them with the grid launcher
(`scripts/experiment_grid.py`, or `scripts/aligner_sweep.sbatch` with
`EXPERIMENT=grid/bridge/<name> PAIRING=<pairing group> SEED=42` and `seed=42`,
`--time` within the partition limit), then chain the launcher on the training jobs:

```bash
sbatch --dependency=afterok:<train jids> --wrap "python scripts/grid_eval/launch_consistency_recompute.py \
  --only coupling --only distortion --only maps --then-render"
```

### The grid's configs are committed

`config/experiment/grid/bridge/` holds one yaml per grid run (the directory is
otherwise gitignored, so a new config is added with `git add -f`). Ten are
generator-owned (`scripts/experiment_grid.py gen` rewrites exactly those: cfm / vpsde
x five pairings on the main-text model's pool); the rest are hand-made beside them: the one-step
(`fullen`) family (512 trees, 2 energy samples), the plain Tree Tx pool (`_nondelta`)
and the aligner ablation (`_align*`). The split loss covers the full tree for every
model (objective default `leaf_reconstruction_only: false`; no config sets it). One
caveat: the aligner-ablation runs trained with `extender: conditional_graft` where the
main-text model has none, so that strip is not a pure aligner ablation.

## Interface (env vars)

| Variable | Used by | Notes |
| --- | --- | --- |
| `EXPERIMENT_DIR` | cpl, cplspl | the run directory |
| `STEP` | all | training step; defaults to the latest checkpoint |
| `POOL_N_STEPS` | cplspl | integration steps of the splitter-as-pooler coupling (8 canonical) |
| `N_TREES` | cpl, cplspl, batches | `MouseEmbryoDataset(max_size=...)`; 512 |
| `OVERWRITE` | cpl, cplspl | `1` recouples / rescores even if the artefacts exist |
| `BATCH`, `STEPS` | batches | see above |

## Manual firing

```bash
D=outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_<hash>
STEP=20000

EXPERIMENT_DIR=$D STEP=$STEP sbatch scripts/grid_eval/eval_coupling.sbatch                        # cpl
EXPERIMENT_DIR=$D STEP=$STEP POOL_N_STEPS=8 sbatch scripts/grid_eval/eval_coupling_splitter.sbatch # cplspl
BATCH="$D|$STEP|8" sbatch scripts/grid_eval/eval_distortion_batch.sbatch                          # distortion (n = the cplspl coupling's)
BATCH="$D" STEPS="$STEP" sbatch scripts/grid_eval/refresh_metrics_batch.sbatch                    # refresh
BATCH="$D/assignments_nn_$(basename $D)_step${STEP}_adj_n8_splitter.pkl" \
  sbatch scripts/grid_eval/eval_cell_type_map_batch.sbatch                                        # ctmap
```

Every sbatch targets `kempner_h100` (lab priority, non-preemptible; cplspl at n = 8
takes ~15 min, a distortion batch of 12 runs ~30 min, a refresh batch ~30 min, a
map batch ~30 min). Override with `sbatch --partition=... --constraint=h100|a100`
if the queue is long; the 20 GB A100 shards on `kempner_requeue` OOM the coupling.

## Watcher

[`watch_grid_milestones.sh`](watch_grid_milestones.sh) polls `outputs/` and fires
the cpl / cplspl matrix for every new `pool_ema_step20000.pt` it has
not fired for yet (a `.eval_20000_submitted` marker per run dir), at n = 1, 8, 32
for the bridges and n = 1 for the one-step models. It submits no distortion / map
batches; use the launcher above for anything that has to be consistent with the
figure.
