# concept (held-out clone supplement)

fig 3's six metrics (clonal, marginal, oscillation, linkage, germ-layer crossing, hox) on clone trees that no model trained on. every fig 3 / grid number is in-sample; this is the out-of-sample version.

- **held-out metrics**: one column per metric, one row per method, the mean and its 95% interval. lower is better in every column. open dots are the paper's models, trained on every tree and scored on the same held-out trees, so the gap to the filled row of the same method is the generalisation gap.

# implementation

the pipeline (`train_*`, `evaluate.py`, `gather.py`) trains the held-out models and scores them into `analysis/results/hold_out_metrics/` (gitignored). `compute_metrics.py` aggregates those into `numerical_results/`, and `plot_metrics.py` + `assemble.py` read only that and write into `imgs/` (both gitignored). `common.py` holds the split, the pipeline's paths and method list, and the held-out dataset. `style.py` holds the figure's paths, row order, labels and geometry.

## the split

11 of the 56 clone trees are held out (`common.HOLDOUT`): two per stage for E7.5-E9.5, from different replicates, and one of the six E10.0 clones. drawn once with `np.random.default_rng(0)`, 23% of all leaves. it is a clone-level split: other clones of the same embryo and the non-clonal cells stay in training, so the claim is "new lineage trees", not "new embryos".

`common.HeldOutMouseEmbryoDataset` removes the held-out clones from everything training reads: records, trees, `leaf_states`, and the real-cell bank (`states` + `reference_features`, which the marginal aligner and the nn pairing bank pull from via `utils/reference.py`). dropping the records alone would leak their cells through those two banks. its constructor asserts the leak is closed.

## methods

| method | row | trained on | from |
|---|---|---|---|
| vpsde | lineage inpainting | train split | `main.py experiment=mouse_embryo_vpsde`, dataset swapped, step 20000 |
| vpsde_all | lineage inpainting, all trees | all 56 | the paper's main-text run (read only) |
| genot_sinkhorn_eps{0.01,0.05,0.2} | GENOT-Sinkhorn, $\varepsilon$ ... | train split | `baselines/genot_sinkhorn.py`, EPSILON swapped |
| genot_cbfgw_fp{0.1,1,10} | GENOT-cbFGW, fused penalty ... | train split | `baselines/genot_fgw_mm.py`, FUSED_PENALTY swapped (alpha = 1 / (1 + fp)) |
| genot_sinkhorn_all, genot_cbfgw_all | GENOT-..., all trees | all 56 | the paper's `outputs/e_baselines/*.pkl` (read only) |

the paper's settings are eps 0.05 and fp 1. the `_all` rows score the same held-out trees with models that saw them, so the gap to the train-split row is the generalisation gap. every genot variant is reported (no selection), which can only help the baselines.

## what is reused

nothing here re-implements a metric or a training loop:

- vpsde trains with `main.py` and the paper's experiment yaml, plus three overrides (`train_vpsde.sbatch`). the clone list is in the config, so it is hashed into the run dir (`outputs/hold_out_vpsde_<hash>/`) and saved in its `config.yaml`.
- genot trains by running the paper's script with three lines swapped (dataset, OUTPUT_PATH, the variant's constant), each asserted to match once (`train_genot.py`).
- scoring calls summary_qc's `compute_distortion` / `compute_assignment_metrics` / `compute_hox.collect_pairs` with its constants and seeds, and `plot_metrics._metric_values` aggregates exactly as fig 3 does. the only restated code is the short hox pearson loop and the annotation maps from `compute_assignment_metrics.main`.

## what changes for held-out scoring

- clonal: candidates are drawn from the held-out trees only.
- marginal: source cells are held-out leaves at t_late; the target stays every tree's leaves at t_early (the population marginal; two held-out clones are too few cells and too fate-biased to be the target).
- germ, oscillation, linkage: the coupling is sliced to the held-out trees; the real linkage reference keeps every tree so it has all replicates.
- hox: common support over all 10 methods, not fig 3's three.

## never call from here

these write next to the paper's models or caches:

- `analysis/lib/eval_core/runner.run_*` / `eval_model` (write into the model's run dir, reuse cached files with no subset tag)
- any summary_qc or grid_summary `main()`
- `baselines/*.py` directly, `baselines/genot_assign.py` (overwrites in place)

the pipeline writes to `analysis/results/hold_out_metrics/`, except the vpsde run dir, which `main.py` puts in `outputs/`. `grid_summary/compute_grid.py` only globs `mouse_embryo_grid_bridge_unshared_*`, so it never sees that dir.

## error bars

mean $\pm$ 1.96 sd / $\sqrt{n}$, fig 3's `summary_qc.plot_metrics._stats`.

| column | mean over | n |
|---|---|---|
| clonal | per-subtree energy distance | 400 subtrees |
| marginal | per stage pair energy distance, E8.0 to E7.5 through E10.0 to E9.5 | 5 |
| oscillation | one pooled fraction | 1, no bar |
| linkage | per transition, E9.5 to E9.0 and E9.0 to E8.5 (fig 3 drops the two earlier ones) | 2 |
| germ | one pooled rate over every parent-child link | 1, no bar |
| hox | 1 - r per (cell type, gene) | 11 |

the interval is over subtrees, stage pairs or genes of one trained model, not over training runs. the 400 subtrees come from 11 trees and can overlap. with n = 2 the linkage bar about spans its two values.

## pipeline -> analysis/results/hold_out_metrics/

four gpus, slurm dependencies (`submit.sh`):

- `train_vpsde.sbatch`: one gpu, 1-3 h.
- `train_genot.sbatch`: six-task array, three at a time (cbFGW ~2 h each, sinkhorn ~17 min).
- `evaluate.sbatch`: one task per method, four at a time. each couples its method on the held-out trees and scores clonal, marginal and the assignment metrics.
- `gather.sbatch`: cpu. the merged caches and hox.

writes:

- `<method>/`: `assignments.pkl` (held-out trees only) and that method's csvs
- `{clonal_distortion,marginal_distortion,assignment_metrics,hox_consistency}.csv`: merged, fig 3's schema
- `genot/<variant>.pkl`: the held-out genot models
- `parity/`: the same for the parity run
- `_snapshot/`: md5 sums and a listing of the paper's artefacts taken before anything ran

## compute -> numerical_results/

`compute_metrics.py` -> `metrics.csv` (metric, label, method, value, lo, hi). `_metric_values` on the merged caches, the rows in `style.ROWS` only. cpu, a second.

## plot + assemble -> imgs/

`plot_metrics.py` -> `imgs/panels/metrics.svg`. `assemble.py` -> `imgs/supp_holdout.{svg,pdf,png}` through `utils.svg_assembly`. one panel, so no letter.

## checks

- **parity.** `evaluate.py --parity` counts every tree as held out, so the paper's three models must reproduce `analysis/figures/summary_qc/numerical_results/`. `gather.py --parity` prints the row-by-row diff. assignment metrics, hox and the genot distortions should match exactly; the model's clonal / marginal rollouts are not deterministic on the gpu, so they move by about 1%.
- **coupling call sites.** for the `_all` methods evaluate.py also redoes the coupling and prints its agreement with the saved one: the genot backward chain should match 100%, the main-text model's rollout coupling mostly.
- **untouched.** compare `_snapshot/before_md5.txt` with a fresh `md5sum` of the same files.

## imports

- `analysis/figures/summary_qc/`: `style` (paths of the paper models, constants, seeds, `METHOD_COLORS`, `apply_rc`, page width), `compute_distortion`, `compute_assignment_metrics.compute_for`, `compute_hox.collect_pairs`, `plot_metrics._metric_values`
- `analysis/lib/`: `coupling.nearest_neighbor_coupling_w_context.couple_all_trees`, `eval_core.models`, `metrics.germ_layer_eval`, `metrics.linkage_metrics`
- `baselines/genot_common.py`, `baselines/genot_sinkhorn.py`, `baselines/genot_fgw_mm.py`
- `utils/svg_assembly.py`: nesting and the svg/pdf/png export

## data sources

- held-out lineage inpainting: `outputs/hold_out_vpsde_6aba556d758689a297cbbb791c225da8/`. its `config.yaml` names the dataset class by a module path that is not in this repo; nothing reads it: `load_splitter` only builds the splitter.
- held-out genot: `analysis/results/hold_out_metrics/genot/<variant>.pkl`
- paper models: `summary_qc.style` `EXPERIMENT_DIR`, `GENOT_SINKHORN_PKL`, `GENOT_CBFGW_PKL`

the hox csv's common support also included a mean-pool baseline not listed above; without it the means move by < 1e-4.

## run order (lt conda env, from the repo root)

    bash analysis/figures/supp_holdout/submit.sh parity   # ~30 min: paper models on every tree vs fig 3
    bash analysis/figures/supp_holdout/submit.sh          # train (~3 h), evaluate (~45 min), gather

    PY=python
    $PY -m analysis.figures.supp_holdout.compute_metrics
    $PY -m analysis.figures.supp_holdout.assemble

any pipeline step runs alone too, e.g. `python -m analysis.figures.supp_holdout.evaluate vpsde`. `python -m analysis.figures.supp_holdout.common` loads the held-out dataset once and runs the leak asserts. only assemble needs rerunning for cosmetics.
