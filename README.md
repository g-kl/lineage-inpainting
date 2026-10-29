# lineage inpainting

code for **learning developmental dynamics from lineage context**.

*Lineage inpainting* is a generative modeling approach for cell states on lineage trees. It is fit on lineage-traced scRNA-seq snapshots. The framework involves two components: (1) an imputation step, which takes trees with states observed at the leaves, and imputes states at the internal nodes (optimized such that imputed states resemble those observed at previous timepoints) and (2) a sampling step, which takes imputed trees, masks all but one layer, and learns to sample the states across the rest of the tree. We implement both steps using tree-structured transformers, and implement the distributional sampler with a stochastic interpolant approach. We train both models end-to-end, simultaneously.

A quick note on terminology: for historical reasons, this codebase refers to the imputer as a "pooler" and the sampler as a "splitter".

This repository contains the model, the training code, the synthetic benchmarks, and the pipelines to generate every result in the paper. The data and the primary trained model weights are on Zenodo: [10.5281/zenodo.23750040](https://doi.org/10.5281/zenodo.23750040).

The model and analysis code was developed by @g-kl and @njwfish. The data and preprocessing code were largely developed by @colganwi and @kijiy.

---

## contents

- [installation](#installation)
- [data](#data)
- [quickstart](#quickstart)
- [reproducing the figures](#reproducing-the-figures)
- [terminology](#terminology)
- [the main model](#the-main-model)
- [repository layout](#repository-layout)
- [data abstractions](#data-abstractions)
- [poolers](#poolers)
- [splitters](#splitters)
- [aligners](#aligners)
- [pairings and extenders](#pairings-and-extenders)
- [record samplers](#record-samplers)
- [trainer](#trainer)
- [configuration](#configuration)
- [running](#running)
- [outputs](#outputs)
- [figures](#figures)
- [analysis layout](#analysis-layout)
- [datasets](#datasets)
- [baselines](#baselines)
- [license](#license)

---

## installation

```bash
git clone https://github.com/g-kl/lineage-inpainting.git
cd lineage-inpainting
conda env create -f environment.yml
conda activate lineage-inpainting
```

[environment.yml](environment.yml) provides versions the paper was run with (python 3.14, torch 2.10 for CUDA 13.0). training and most analyses need an NVIDIA GPU: `device` defaults to `cuda`, and the nearest-neighbor searches use FAISS on the GPU (this will be excruciatingly slow on CPU). a few GPU-UMAP / GPU-preprocessing steps (marked `$GPU` in the figure `concept.md` files) ran in a separate env with `rapids-singlecell`.

run everything from the repository root. the scripts import the repo's own packages (`datasets`, `utils`, `analysis`, ...) from the working directory.

the `.sbatch` / `.sh` files and the `#SBATCH` headers at the top of some scripts are the job files we used on our clusters. they carry our partitions, accounts and paths, so adapt them before submitting.

---
## data

the code expects data in a `data/` directory that is a **sibling of the repository**, not inside it. for example:

```
parent/
  lineage-inpainting/        this repository
  data/
    embryos/                 lineage trees, scVI latents, annotations
    2608_crispants/          Tbx5 crispant scRNA-seq
```

### from Zenodo

download the bundle from [10.5281/zenodo.23750040](https://doi.org/10.5281/zenodo.23750040) and check it against `md5sums.txt`. it contains:

| bundle path | contents | where it goes |
|---|---|---|
| `embryos/final_topology.h5td` | lineage trees, one per embryo (`treedata`) | `../data/embryos/` |
| `embryos/devmap_latents.h5ad` | scVI latents (`obsm["X_scvi"]`), cell metadata and gene panel of the lineage-traced cells (no counts) | `../data/embryos/qiu24_integrated_hvgs_and_scvi.h5ad` (the name the code uses; a symlink is fine) |
| `embryos/final_subtype_v2_lower.csv`, `embryos/final_type_info_v2_lower.csv`, `embryos/final_umap.csv` | cell subtype annotations, cell-type / germ-layer / lineage metadata, UMAP coordinates | `../data/embryos/` |
| `scvi_model/model.pt` | the scVI model that produced the latents | `../data/embryos/scvi_model_hvgs_and_scvi/model.pt` |
| `model/config.yaml`, `model/pool_ema_step20000.pt`, `model/split_ema_step20000.pt` | the trained main-text model (EMA weights) | `outputs/<main run>/` (see [quickstart](#quickstart)) |
| `model/assignments_step20000_adj_n8_splitter.pkl` | a sample of the main-text model, provides a node to cell mapping | `outputs/<main run>/assignments_nn_<main run>_step20000_adj_n8_splitter.pkl` |
| `tbx5_crispants.h5ad` | demultiplexed Tbx5 crispant cells | `../data/2608_crispants/tbx5.h5ad` |

for example, from the directory that contains the repository, with the bundle unpacked at `bundle/`:

```bash
mkdir -p data/embryos/scvi_model_hvgs_and_scvi data/2608_crispants
cp bundle/embryos/{final_topology.h5td,final_subtype_v2_lower.csv,final_type_info_v2_lower.csv,final_umap.csv} data/embryos/
cp bundle/embryos/devmap_latents.h5ad data/embryos/qiu24_integrated_hvgs_and_scvi.h5ad
cp bundle/scvi_model/model.pt data/embryos/scvi_model_hvgs_and_scvi/
cp bundle/tbx5_crispants.h5ad data/2608_crispants/tbx5.h5ad
```

`TreeRecord` pickles are built from the trees on first load and cached in `cache/` (gitignored).

### external inputs

- **gnomAD constraint metrics**, for the LoF-constraint panel of the perturbation figure: see [literature/README.md](literature/README.md). too large to track.
- the other external gene sets and orthology tables are committed under [literature/](literature/), with their sources in [literature/README.md](literature/README.md).

### upstream processing (not needed to reproduce the figures)

- scVI integration of the lineage-traced atlas with the Qiu et al. (2024) reference: [embed/](embed/).
- Tbx5 crispant processing: cellranger ([preprocessing/cellranger/](preprocessing/cellranger/)), per-read SNP calls ([preprocessing/SNP_call/](preprocessing/SNP_call/)), then embryo demultiplexing ([analysis/pipelines/tbx5/demultiplexing/](analysis/pipelines/tbx5/demultiplexing/)). the Zenodo bundle already holds its output.

---

## quickstart

train the main-text model (20k steps, one GPU):

```bash
python main.py experiment=mouse_embryo_vpsde
```

or use the released checkpoint instead:

```bash
RUN=mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09
mkdir -p outputs/$RUN
cp bundle/model/{config.yaml,pool_ema_step20000.pt,split_ema_step20000.pt} outputs/$RUN/
cp bundle/model/assignments_step20000_adj_n8_splitter.pkl outputs/$RUN/assignments_nn_${RUN}_step20000_adj_n8_splitter.pkl
```
---

## reproducing the figures

Each figure, including the supp figs, has its own directory, listed in [figures](#figures). Each directory has a `concept.md` that documents the exact commands necessary to generate the figure and associated numerical results. the scripts are (mostly) split into `compute_*.py` (writes `numerical_results/`) and `plot_*.py` / `assemble.py` (writes `imgs/`). Both output directories are gitignored and regenerable.

- **main model figures** (summary + QC, convergence, perturbation, Tbx5, recipe, division rate) need the main model in `outputs/` and the data in `../data/`. the perturbation screens behind the perturbation and Tbx5 figures are under [analysis/pipelines/perturb/](analysis/pipelines/perturb/).
- **model-comparison figures** (grid summary, aligner ablation, held-out clones) need the other runs of the model grid. all 28 grid configs are committed in [config/experiment/grid/bridge/](config/experiment/grid/bridge/); train them with `python main.py experiment=grid/bridge/<name> seed=42` (the seed is part of the run hash, and the grid configs leave it to the command line), or submit them all to slurm with `python scripts/experiment_grid.py submit --seed 42` (adapt its partition first). [scripts/grid_eval/](scripts/grid_eval/README.md) turns trained runs into the numbers the grid figures plot.
- **synthetic benchmarks** are self-contained: [synthetic_evals/](synthetic_evals/) simulates the data, trains, evaluates and plots (`synthetic_evals/figure/concept.md`). some of their result pickles are committed so the plots can be rebuilt without retraining.
- **schematics** are built by the scripts in [cartoons/](cartoons/) (see [cartoons/README.md](cartoons/README.md)).


---

## the main model

every main-text figure uses the same model: [config/experiment/mouse_embryo_vpsde.yaml](config/experiment/mouse_embryo_vpsde.yaml). it uses:

| slot | choice |
|---|---|
| dataset | `mouse_embryo`, scVI embeddings (`X_scvi`, 50-d), subtrees with up to 1024 leaves |
| pooler | `DeltaTreeFormer`: one-step tree transformer predicting a residual over a per-node baseline (4 layers, 128-d, 4 heads) |
| splitter | `BridgeDeltaTreeFlow` with a VP-SDE bridge: the same architecture trained as a bridge from noise to the complete tree, 8 rollout steps |
| aligner | `StratifiedFeatureAligner`, energy-distance loss on three features: time marginal against the real-cell bank (weight 1), root-to-leaf paths (0.25), common-ancestor triples (0.25) |
| pairing | alternating: identity on even steps, nearest-neighbour snapping to real cells (own time bin +/- 1) on odd steps |
| extender | none |
| record sampler | paired records 0.5 apart in leaf time (tolerance 0.1), subtree roots up to 0.5 before the earlier leaf time |
| training | 1024 subtrees per record, 2 records, 20k steps, EMA 0.999; the analyses load the EMA checkpoints |

its `experiment_name` is the one the model grid uses, so its run directory sits alongside the grid's. the grid itself (sampling model x imputing model x pairing, plus the aligner-feature ablation) is in [config/experiment/grid/bridge/](config/experiment/grid/bridge/), generated by [scripts/experiment_grid.py](scripts/experiment_grid.py) and evaluated by [scripts/grid_eval/](scripts/grid_eval/README.md).

the pooler and splitter are **the same architecture**, instantiated twice, in two directions: `pool_batch` (impute step) is a single-pass prediction from observed anchors, `split_layer_batch` (sample step) is a multi-step stochastic interpolant style rollout. the pooler classes (`TreeFormer`, `DeltaTreeFormer`) therefore live in `splitters/tree_former.py` next to the bridge splitters they share architecture with. every run trains a dedicated pooler next to its splitter (`share_pool_split: false`). the dedicated poolers and the per-parent flow-matching splitter (`fm`) have configs; the other per-parent splitters are in code only.

entry point: [main.py](main.py). core loop: [trainer/trainer.py](trainer/trainer.py).

---

## repository layout

| directory | purpose |
|---|---|
| [main.py](main.py) | hydra entry point; instantiates dataset + pooler + splitter + aligner + pairing + extender + record_sampler and runs `Trainer.fit`. contains a python-3.14 `argparse._check_help` shim so hydra still initializes on newer interpreters. |
| [trainer/](trainer/) | `Trainer` ([trainer/trainer.py](trainer/trainer.py)). |
| [datasets/](datasets/) | `TreeRecord` / `TreeBatch` ([datasets/tree_record.py](datasets/tree_record.py)); the `mouse_embryo` loader; the synthetic branching simulator ([datasets/synthetic.py](datasets/synthetic.py)) and the benchmark datasets built on it. |
| [poolers/](poolers/) | bottom-up aggregators (see [poolers](#poolers)). |
| [splitters/](splitters/) | the bi-directional samplers. flow/diffusion geometry in [splitters/bridges/](splitters/bridges/). |
| [aligners/](aligners/) | alignment losses ([aligners/stratified.py](aligners/stratified.py), [aligners/slice.py](aligners/slice.py)) and the pairwise observed-marginal aligner of the synthetic context benchmark ([aligners/aligner.py](aligners/aligner.py)). |
| [pairings/](pairings/) | the snap-to-real step before sampling loss computation; in [pairings/snap.py](pairings/snap.py). |
| [extenders/](extenders/) | tools to extend lineage topology forward in time. not used in manuscript. |
| [record_samplers/](record_samplers/) | choosing tree records at train time. |
| [baselines/](baselines/) | multimarginal FM and GENOT baselines (see [baselines](#baselines)). |
| [analysis/](analysis/) | figure pipelines, the shared eval library, and the pipelines that feed them (see [analysis layout](#analysis-layout)). |
| [synthetic_evals/](synthetic_evals/) | synthetic benchmarks (crossing, context, landscape) + their figure pipeline. |
| [cartoons/](cartoons/) | the schematic cartoons for fig 1 |
| [utils/](utils/) | config hashing, OT solver wrappers, id helpers, palettes, figure-label rendering and fonts, the heart-diagram renderer, experiment loading. |
| [config/](config/) | hydra yaml hierarchy. |
| [scripts/](scripts/) | assorted cli drivers and sbatch files including the grid generator and eval pipeline. |
| [preprocessing/](preprocessing/) | cache building + validation, annotation, and the cellranger + SNP-calling steps for the Tbx5 crispant embryos. |
| [literature/](literature/) | external gene sets, orthology maps, curated transitions, phenotype maps ([sources](literature/README.md)). |
| [embed/](embed/) | scVI integration scripts and slurm batch files used to produce leaf embeddings. |
| [tests/](tests/) | assorted test scripts. |
| `cache/` | `TreeRecord` pickles, populated automatically on first dataset load. gitignored. |
| `outputs/` | hashed run directories (`<experiment_name>_<config_hash>/`) with checkpoints + resolved config. gitignored. |
| `multirun/`, `logs/` | hydra multirun and slurm launcher output. gitignored. |

---

## data abstractions

### `TreeRecord`

one `TreeRecord` corresponds to a single rooted tree (e.g. one clone from one embryo). it indexes **every subtree** whose root time is close to one of the configured time bins, so subtrees rooted at any developmental time can be drawn cheaply at training time.

key fields:

- `time_bins`: sorted list of root-time bin centres.
- `time_tolerance`: half the smallest gap between bins (default), or an explicit override.
- `subtree_index`: `dict[bin_centre -> list[subtree_dict]]`. each subtree dict carries `graph` (`nx.DiGraph`), `root`, `root_time`, `leaf_time`, `leaves`, `tree_name`, `depth`, `size`.
- `leaf_time`: max leaf time of the parent tree (used by the record sampler to find paired records at the right offset).

methods:

- `get_subtrees(time, n)`: return up to `n` subtrees whose root is in the bin nearest to `time`, sampled without replacement.
- `save(path)` / `load(path)`: pickle the index. the mouse-embryo loader uses these to cache one record per source tree.

### `TreeBatch`

`TreeBatch` is the padded tensor form used inside the model. built by `collate_subtrees(subtree_dicts, leaf_states, device, max_nodes)`:

| field | shape | description |
|---|---|---|
| `adj` | `(B, N, N)` float | `adj[b, i, j] = 1` iff `i` is the parent of `j` in tree `b`. |
| `node_times` | `(B, N)` | per-node observation / branching time. |
| `mrca_times` | `(B, N, N)` | time of the most recent common ancestor, used for patristic attention bias and FGW distances. |
| `mask` | `(B, N)` bool | `True` for real nodes (vs padding). |
| `leaf_mask` | `(B, N)` bool | `True` for observed leaves. |
| `depth` | `(B, N)` long | distance from root, root = 0. |
| `child_index` | `(B, N)` long | which child of its parent each node is, used by per-rank splitter heads. |
| `parent_idx` | `(B, N)` long | parent index, `-1` for root / padding. |
| `root_idx` | `(B,)` long | root position in each padded tree. |
| `n_nodes` | `(B,)` long | real node count per tree. |
| `max_depth` | int | max depth across the batch. |
| `node_names` | list[list[str]] | per-tree node names in topological order, used by aligners that need the original `nx.DiGraph`. |
| `leaf_states` | `(B, N, dim)` | observed leaf embeddings, zero elsewhere. |
| `subtree_records` | list[dict] | the original subtree dicts, kept so aligners can reach `nx.DiGraph` topology. |

`to_pooled_tree_list(states)` unpacks the padded `(B, N, dim)` tensor back into per-tree `{graph, states, root, leaf_states}` dicts, which the path / observed-marginal aligners consume.

if a subtree exceeds `max_nodes` (default 128), `_subsample_graph` randomly drops leaves while preserving the path from each kept leaf to the root.

---

## poolers

a pooler maps a `TreeBatch` to per-node embeddings of shape `(B, N, dim)`. leaves retain their observed embeddings. all poolers expose `pool_batch(batch)`. the config default, and every figure's pooler, is the one-step tree transformer `delta_tree_former` (see below).

### dedicated poolers

| pooler | module | learnable | topology used |
|---|---|---|---|
| `MeanPool` (`mean_pool`) | [poolers/mean_pool.py](poolers/mean_pool.py) | no | bottom-up mean per parent |
| `NNSnappedMeanPool` (`nn_mean_pool`) | [poolers/nn_snapped_mean_pool.py](poolers/nn_snapped_mean_pool.py) | none | mean-pool, then **snap** every internal-node embedding to its FAISS-GPU L2 nearest neighbour in a real-cell bank |
| `OTSnappedMeanPool` (`ot_mean_pool`) | [poolers/ot_snapped_mean_pool.py](poolers/ot_snapped_mean_pool.py) | none | mean-pool, then sample each internal-node embedding from a sinkhorn OT plan against a random reference minibatch |

`PrecomputedAssignmentPool` ([poolers/precomputed_assignment_pool.py](poolers/precomputed_assignment_pool.py)) is in code but has no config.

notes:

- the two **snap poolers** are *parameter-free* but need a real-cell bank, bound with `bind_dataset(dataset)` (or `bind_cells(embeddings, times)`), before training. they guarantee every internal-node state lies in the manifold of real observed cells, and are designed to pair with a flow-matching splitter and no aligner (the snap itself plays the regulariser role).
- all poolers respect `batch.leaf_mask`: they never overwrite leaf positions, so observed leaf embeddings appear unchanged at the leaves of the output.

### tree transformers in the pooler slot

`config/pooler/` also contains `tree_former` and `delta_tree_former`, whose `_target_` classes live in [splitters/tree_former.py](splitters/tree_former.py): one-step tree transformers (no flow) that predict the whole tree from the observed anchors in a single pass. `delta_tree_former` is the figures' pooler.

---

## splitters

a splitter takes pooled `target_states` plus the `TreeBatch` and either (a) computes a training loss against the targets (`batch_loss`) or (b) generates non-root states top-down from the root state (`split_tree_batch`). none of the production models use the `split_tree_batch` path.

### per-parent splitters

`FlowMatchingSplit` and `MLPSplit` are present but are not used or evaluated in the paper. they follow the (b) path.

### tree-flow family (whole-tree simultaneously)

these are the models we use in the paper and they treat every tree node as a token, clamp the observed anchors to their real values, put a gaussian draw at every masked node, and predict the whole tree.

| splitter | module | flow | notes |
|---|---|---|---|
| `TreeFormer`, `DeltaTreeFormer`, `FullEnergyDeltaTreeFormer` (`full_energy_delta_tree_former`; `tree_former` and `delta_tree_former` exist as pooler-slot configs only) | [splitters/tree_former.py](splitters/tree_former.py) | **none** | flow-free counterpart of the bridge models: one forward pass, plain MSE (or energy score) on the masked nodes for a sampled anchor slice |
| `BridgeTreeFlow`, `BridgeDeltaTreeFlow` (`bridge_tree_flow`, `bridge_delta_tree_flow`; the `_vpsde` variant of the latter is the config default) | [splitters/bridge_tree_flow.py](splitters/bridge_tree_flow.py) | delegated to a bridge | the transformer predicts `x_1`; a bridge owns the prior, interpolant, ODE step and rollout. patristic attention bias from `mrca_times`. |

bridge geometry is swappable and lives in [splitters/bridges/](splitters/bridges/): `CFMBridge` ([flow_matching.py](splitters/bridges/flow_matching.py)) is the linear conditional-flow-matching interpolant with a deterministic probability-flow ODE sampler; `DiffusionBridge` ([vpsde.py](splitters/bridges/vpsde.py)) is a drop-in with the same interface but a variance-preserving SDE path and a stochastic reverse-SDE sampler. the `_vpsde` config variants select the latter; this is what the `vpsde` in the figure run names refers to.

implementation notes:

- **teacher forcing**: the deterministic per-parent splitters score predictions made from the *true* parent at every depth, so reconstruction errors don't compound.

---

## aligners

all aligners take `(batch_1, states_1, batch_2, states_2)` and return a scalar loss. two axes distinguish them:

- **path-based vs point-based.** path aligners build per-lineage trajectories (root-to-leaf, snapped to a common time grid) and compare distributions of those vectors. point aligners compare per-time-slice embedding distributions.
- **pairwise vs unary.** pairwise aligners compare the two batches directly. unary aligners compare each batch independently to an external reference (a real-cell bank); the pairwise forward sums the per-batch terms.

### pairwise marginal aligner ([aligners/aligner.py](aligners/aligner.py))

| aligner | compares | topology-aware | reference |
|---|---|---|---|
| `ObservedMarginalAligner` (`marginal`) | leaves of the earlier-observed batch vs internal nodes of the later batch within `time_tolerance` | no | none |

it is the aligner of the synthetic context benchmark (`experiment=synbench_fb_unshared_vpsde_idnn_leaf`); the module also defines the `Aligner` base class.

### stratified-feature aligners ([aligners/stratified.py](aligners/stratified.py), [aligners/slice.py](aligners/slice.py))

`StratifiedFeatureAligner` is what we actually use: **every paper figure uses it**. you give it a list of *features*, each of which extracts a comparable quantity from a batch, and it scores them (energy distance by default). the features are `TimeMarginalFeature` (the reference-bank marginal, in [aligners/stratified.py](aligners/stratified.py)) and, on the slice primitive of [aligners/slice.py](aligners/slice.py), `PathFeature` and `CommonAncestorFeature`.

the config group exposes these as `slice_marginal_only`, `slice_marginal_path`, `slice_marginal_ca` and `slice_marginal_ca_path`.

further notes:

- **`ObservedMarginalAligner` vs `TimeMarginalFeature`**: both anchor inferred ancestors at observation times, but observed uses *the other batch's leaves* as the thing to compare to, while the feature uses a fixed external bank covering all observation times. the feature is unary; observed is pairwise.
- **`bind_dataset` plumbing**: `main.py` calls `bind_dataset(dataset)` once at startup on every component that exposes it (aligner, pooler, splitter, pairings, extender), and each takes the slice it needs (real-cell banks via `utils.reference.reference_cells`), so a snap pooler gets its bank without special handling. the bank lives in the dataset's `.states` AnnData (`obsm[embedding_key]`, `obs["time"]`).

---

## pairings and extenders

two smaller groups that sit between pool and split.

- **pairing** ([pairings/](pairings/)) pairs stage-one synthetic states to real marginal cells before splitting. `identity` (default, no-op), `nearest_neighbor`, `nearest_neighbor_clonal`, their `_sg` (stop-gradient) variants, and `id_nn_alt` (the figures; `OTPairing` has no config). `cfg.pairing` can be a **list**, in which case the trainer alternates across steps (`pairings[t % len]`). all snapping routes through [pairings/snap.py](pairings/snap.py).
- **extender** ([extenders/](extenders/)) grafts donor topology from later records onto an early-leaf-time subtree so it can be extended forward to a later target time. `none` (default) and `conditional_graft` (ask a count sampler how many descendants a cell has at `time + delta`, then build accordingly; the count model is [extenders/count.py](extenders/count.py)); `GraftExtender` (real donor topology) and `ConditionalSkeletonExtender` have no config.

---

## record samplers

a `RecordSampler` returns `(chosen_records, sample_time)` each step. all live in [record_samplers/record_samplers.py](record_samplers/record_samplers.py).

- **`PairedOffsetSampler`** (`paired_offset_sample_offset`, default): pick an anchor at random among the records that have a partner whose `leaf_time` is within `tol` of `anchor.leaf_time + target_gap`, then pick one such partner at random. for `n_records > 2`, extras are filled uniformly at random. the sample time is `min(leaf_time)` over chosen records, minus an offset drawn uniformly from `[0, sample_time_gap]`.
- **`PairedOffsetBackfillSampler`** (no config): subclass of the above with backfilling behaviour.
- **`AnchorTimeSampler`** (`anchor_time`): samples around an anchor time.
- **`RandomTimeBinSampler`** (no config): pick records uniformly at random; pick a sample time uniformly from the union of `time_bins` strictly below `min(leaf_time)`. used when paired offsets aren't meaningful (e.g. synthetic data with many bins).

the trainer calls the sampler inside a retry loop: if any chosen record has zero subtrees at the sampled time, it resamples up to `_max_resample` times before failing.

---

## trainer

`Trainer` ([trainer/trainer.py](trainer/trainer.py)) per step:

1. **sample** records + time via `record_sampler`.
2. **collate** `n_trees` subtrees per record into a `TreeBatch`.
3. **pool** each batch to get `(B, N, dim)` states.
4. **align loss**: score every unordered pair of records with the aligner on the pooled states.
5. **split loss**: optionally pair synthetic non-observed, non-anchor states to real cells via `pairing`, then sum `splitter.batch_loss(batch, states)` over chosen records.
6. `total = split_weight * split_loss + align_weight * align_loss` (+ `split_align_weight * ...` if set). backprop, clip to `max_grad_norm`, optimizer step.
7. **EMA shadow update** (if `ema_decay` is set). uses warm-up decay `min(decay, (1+t)/(10+t))`. EMA weights are swapped in for checkpointing only.

`align_only_steps` trains pool -> aligner alone for the first N steps.

checkpoints written to `output_dir`:

- `pool_step{N}.pt`, `split_step{N}.pt`, `aligner_step{N}.pt`: live weights.
- `pool_ema_step{N}.pt`, `split_ema_step{N}.pt`: EMA weights (when EMA is on).
- `trainer_step{N}.pt`: optimizer / step state, for resuming. without it, loading restores the weights and the optimizer / EMA start fresh.
- `config.yaml`: the resolved hydra config (written once at run creation by `utils.hash.get_output_dir`).

---

## configuration

hydra-based (`version_base="1.1"`), rooted at [config/config.yaml](config/config.yaml). components are constructed via `hydra.utils.instantiate`. `hydra.job.chdir` is `false`, so paths in configs are relative to the repo root at launch time.

defaults: `dataset=mouse_embryo`, `pooler=delta_tree_former`, `splitter=bridge_delta_tree_flow_vpsde`, `aligner=slice_marginal_ca_path`, `pairing=identity`, `extender=none`, `record_sampler=paired_offset_sample_offset`, `objective=default`, `optimizer=default`, `training=default`, `hydra/launcher=local`. `seed=212121`, `device=cuda`.

**these defaults are not what the figures use.** the model behind every main-text figure is [config/experiment/mouse_embryo_vpsde.yaml](config/experiment/mouse_embryo_vpsde.yaml): every value is written out there. see [the main model](#the-main-model) and [figures](#figures).

the output directory is an md5 hash of the config minus `DEFAULT_EXCLUDE_KEYS = ["training", "hydra", "device"]` (`hash_config` in [utils/hash.py](utils/hash.py)), so runs differing only in schedule, launcher or hardware collide.

groups:

- **dataset/**: `mouse_embryo`; the synthetic benchmarks (`cell_type_crossing`, `fate_bias_multi_seed` + `fate_bias_sig{02,04,06,08}`, `high_dim_crossing/`); plus the generated per-motif sets under `crossing_landscape/` (32) and `context_landscape/` (10), both regenerated by their eval scripts.
- **pooler/**: the dedicated poolers (`mean_pool`, `nn_mean_pool`, `ot_mean_pool`) plus the one-step tree transformers `tree_former` / `delta_tree_former` (see [poolers](#poolers)).
- **splitter/**: `fm` and the tree-flow family (see [splitters](#splitters)).
- **aligner/**: the `slice_marginal_*` stratified family and `marginal` (see [aligners](#aligners)).
- **pairing/**: `identity`, `nearest_neighbor`, `nearest_neighbor_clonal`, their `_sg` (stop-gradient) variants, and `id_nn_alt` (alternates identity / nearest neighbour step by step).
- **extender/**: `none`, `conditional_graft`.
- **record_sampler/**: `paired_offset_sample_offset`, `anchor_time`.
- **objective/**: `default`.
- **training/**: `default` (`Trainer`, `n_trees=2048`, `n_records=2`, `n_steps=20000`, `log_every=100`, `save_every=5000`, `log_window=100`).
- **hydra/launcher/**: `local` (default; runs in-process), `joblib` (packs several runs onto one gpu), and `lids` / `kempner` (submitit / slurm launchers for our clusters; edit the partition and account before using them).
- **experiment/**: `mouse_embryo_vpsde` (the main-figure model), `synbench_fb_unshared_vpsde_idnn_leaf`, the `cell_type_crossing/` and `high_dim_crossing/` subgroups (the synthetic benchmarks), and `grid/bridge/` (the 28 runs of the model grid, from [scripts/experiment_grid.py](scripts/experiment_grid.py); newly generated grid configs are gitignored).

---

## running

```bash
# defaults
python main.py

# the main-figure model
python main.py experiment=mouse_embryo_vpsde

# ad-hoc override
python main.py pooler=delta_tree_former splitter=bridge_delta_tree_flow_vpsde \
               aligner=slice_marginal_ca training.n_steps=20000

# multirun, one job per seed, submitted to slurm (adapt config/hydra/launcher/lids.yaml first)
python main.py -m hydra/launcher=lids experiment=mouse_embryo_vpsde seed=1,2,3
```

use `experiment=`, not `+experiment=`: `config.yaml` already declares the group (as `optional experiment: null`), so `+` errors.

needs a GPU: `device` defaults to `cuda` and the trainer moves modules to it unconditionally.

---

## outputs

every run lives at `outputs/<experiment_name>_<config_hash>/`. two runs with the same model and dataset but different schedules or hardware share a directory and overwrite each other unless you change `experiment_name`.

post-training evaluation is driven from [scripts/run_all_eval.py](scripts/run_all_eval.py), which aggregates the metrics in [analysis/lib/metrics/](analysis/lib/metrics/) (`marginal_preservation`, `clonal_preservation`, `linkage_metrics`, `germ_layer_eval`, `curated_transition`, plus `_genot` variants for the baselines) into a dataframe and produces a per-metric lollipop summary. node->cell couplings come from [analysis/lib/coupling/](analysis/lib/coupling/). the figures' own numbers are produced by their pipelines under [analysis/figures/](analysis/figures/) and the grid's by [scripts/grid_eval/](scripts/grid_eval/README.md), both on the eval spine in [analysis/lib/eval_core/](analysis/lib/eval_core/).

---

## figures

each figure directory owns a `concept.md` that documents its panels, its caches, and its run order. **read that file first.**

| figure | directory | master output |
|---|---|---|
| schematics / cartoons (problem setup, methodology, inpainting, ...) | [cartoons/](cartoons/) | per-figure `.svg` / `.png`, see [cartoons/README.md](cartoons/README.md) |
| synthetic benchmarks | [synthetic_evals/figure/](synthetic_evals/figure/) | `imgs/synthetic_figure.{svg,pdf,png}` |
| summary + QC | [analysis/figures/summary_qc/](analysis/figures/summary_qc/) | `imgs/fig3.{svg,pdf,png}` |
| convergence | [analysis/figures/convergence/](analysis/figures/convergence/) | `imgs/fig4.{svg,pdf,png}` |
| perturbation map + validation | [analysis/figures/perturb/](analysis/figures/perturb/) | `imgs/figure.{svg,png}` |
| Tbx5 | [analysis/figures/tbx5/](analysis/figures/tbx5/) | `imgs/tbx5_complete.{svg,pdf,png}` |
| perturbation + Tbx5, combined | [analysis/figures/perturb_tbx5_combined/](analysis/figures/perturb_tbx5_combined/) | `imgs/figure.{svg,pdf,png}` |
| TF recipe / subway | [analysis/figures/recipe/](analysis/figures/recipe/) | `subway_{penetrance,specificity}.{svg,pdf,png}` |
| division rate | [analysis/figures/division_rate/](analysis/figures/division_rate/) | `figures/divrate_*.{pdf,png}` |
| theory running example | [analysis/figures/theory/](analysis/figures/theory/) | `imgs/ising_running_example.pdf` |
| grid summary (ablation strips, aligner ablation, transition-graph similarity, sampler vs imputer) | [analysis/figures/grid_summary/](analysis/figures/grid_summary/) | `figures/<step>/*.{svg,pdf,png}` |
| synthetic supplement | [synthetic_evals/figure_supp/](synthetic_evals/figure_supp/) | `imgs/*.{svg,pdf,png}` |
| supplements: attention, convergence, held-out clones, perturbation (MGI), cell-type summary, Tbx5 | [analysis/figures/supp_attn/](analysis/figures/supp_attn/), [supp_conv/](analysis/figures/supp_conv/), [supp_holdout/](analysis/figures/supp_holdout/), [supp_mgi/](analysis/figures/supp_mgi/), [supp_summary/](analysis/figures/supp_summary/), [supp_tbx5/](analysis/figures/supp_tbx5/) | `imgs/supp_<name>.{svg,pdf,png}` |

---

## analysis layout

`analysis/` has three tiers.

```
analysis/
  figures/      one directory per figure; each has a concept.md, style.py,
                compute_*.py -> numerical_results/, plot_*.py + assemble.py -> imgs/
  lib/          importable, no side effects
    eval_core/  the shared eval spine: couple a run, score the coupling, distortions
    metrics/    the eval metrics
    coupling/   node->cell coupling construction and its eval/
    celltype_map/  cell-type transition maps and sankey geometry
  pipelines/    produce the artifacts figures/ reads; run by hand or via sbatch
    perturb/    the knockout screens (heart, hvg, tf, dose response)
    context/    context-scaling evals (ancestral + forward)
    commitment/ commitment index + fate-gene trajectories
    recipe/     fate-TF tables
    division_rate/
    tbx5/       the Tbx5 crispant analysis (demultiplexing, screens)
```

`analysis/results/` (gitignored) is a shared output dir that several pipelines write into.

the `analysis/figures/*` scripts are run either as modules (`python -m analysis.figures.summary_qc.assemble`) or as plain scripts (`python analysis/figures/convergence/assemble.py`) depending on whether they use absolute or sibling imports; each `concept.md` gives the exact command.

---

## datasets

### mouse embryo (`mouse_embryo`)

`MouseEmbryoDataset` ([datasets/mouse_embryo.py](datasets/mouse_embryo.py)) reads:

| file | role |
|---|---|
| `final_topology.h5td` | `treedata` file with per-embryo lineage trees in `td.obst`. one `TreeRecord` per tree key. |
| `qiu24_integrated_hvgs_and_scvi.h5ad` | `anndata` holding `obsm["X_scvi"]` embeddings (and, in our full copy, HVG counts). filtered to `dataset == 'devmap'` when `devmap_only=True` (default). the Zenodo bundle ships the devmap rows as `devmap_latents.h5ad`. |
| `final_subtype_v2_lower.csv` | per-cell subtype annotations. |
| `final_type_info_v2_lower.csv` | cell-type / germ-layer / lineage metadata; mapped onto `states.obs.{cell_type,germ_layer,lineage}`. |
| `final_umap.csv` | UMAP coordinates for visualization. |

default time bins: `[7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]`. the default `data_dir` is `../data/embryos/` (sibling of the repo). `TreeRecord` objects are pickled to `cache/<trees_filename>_<tree_name>.pkl` on first load.

the dataset also exposes `.states` (the filtered AnnData) and `.leaf_states` (`{cell_id -> tensor}`), which `main.py` uses to bind real-cell references.

### synthetic

- **`hierarchical_ou`**: `HierarchicalOUDataset` ([datasets/hierarchical_ou.py](datasets/hierarchical_ou.py)) wraps `make_hierarchical_ou` from [datasets/synthetic.py](datasets/synthetic.py). the generator simulates a branching SDE: at each fixed-interval division a `BranchModifier` maps `(parent_params, state, generation, child_index) -> child_params`, producing correlated lineages rather than independent clones. `HierarchicalOUModifier` does brownian motion until `split_gen`, then splitting OU attractors; `ConvergentFateModifier` splits and then reconverges, for testing whether inference detects populations that merge after dividing.
- **`fate_bias`** / **`lifted_fate_bias`** ([datasets/fate_bias.py](datasets/fate_bias.py), [datasets/lifted_fate_bias.py](datasets/lifted_fate_bias.py)): the `synbench_fb_*` context benchmark.
- **`cell_type_crossing`** / **`cell_type_fate`** ([datasets/cell_type_crossing.py](datasets/cell_type_crossing.py), [datasets/cell_type_fate.py](datasets/cell_type_fate.py)): crossing and real-cell context benchmarks, built on real cells from the mouse-embryo atlas; `CellTypeFateDataset` backs the per-motif configs in `config/dataset/context_landscape/`.

`SimResult.to_tree_records()` produces one `TreeRecord` per observation time plus a unified leaf-state dict and an AnnData, so the same `Trainer` consumes synthetic and real data interchangeably.

---

## baselines

- [`baselines/genot_fgw_mm.py`](baselines/genot_fgw_mm.py) - multimarginal GENOT + fused-gromov-wasserstein with a tree-distance oracle (euler-tour + sparse-table RMQ for O(1) MRCA distance). JIT-compiled JAX training body. self-contained `#SBATCH` script.
- [`baselines/genot_sinkhorn.py`](baselines/genot_sinkhorn.py) - multimarginal GENOT + standard sinkhorn with timepoint conditioning. same multimarginal sampling structure, no tree-distance term.
- [`baselines/genot_common.py`](baselines/genot_common.py) - shared pieces.

`synthetic_evals/baselines/` holds the synthetic-side baselines (`genot_fgw`, `genot_fgw_fb`, `genot_sinkhorn`, `marginal_fm`) used by the synthetic figure.

---

## license

code: [MIT](LICENSE). the bundled fonts in [utils/fonts/](utils/fonts/) keep their own licenses (SIL OFL 1.1, GUST Font License); the external tables in [literature/](literature/) are from the sources listed in [literature/README.md](literature/README.md).
