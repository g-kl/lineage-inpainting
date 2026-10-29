#!/usr/bin/env python
#SBATCH --job-name=hvg_projection_screen
#SBATCH --output=logs/hvg_projection_screen_%A_%a.log
#SBATCH --error=logs/hvg_projection_screen_%A_%a.log
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=bates
"""highly-variable-gene knockout screen using projection, sharded over HVGs.

the strength of the perturbation is set by ``knockdown_dose`` at the top: 0.0 is
a full knockout (project every expressing cell onto its nearest gene-OFF cell);
0.5 is a 50% knockdown (move each cell to expr*0.5 and project onto the nearest
cell in that quantile bin).  no counting is done here -- the screen just saves
the propagated lineage trees so any downstream summary (cell-type counts,
off-manifold cutoffs, ...) can be computed post-hoc from the pickles.

one invocation screens ONE replicate at ONE target time (``--replicate`` /
``--target-time``); run it once per replicate to cover all three.

OUTPUTS (under ``hvg_projection_screen_v2_<dose>_E<target_time>_<replicate>/``):
  genes/<gene>.pkl     dict {tree_name: light-tree} per perturbed gene
  control/<tree>.pkl   the unperturbed baseline tree per starting tree
the gene-independent pooled states are recomputed fresh for this screen and
cached (un-tagged, shared across doses *and replicates*, keyed by tree name)
under ``hvg_projection_screen/pooled_sampler/``.

SHARDING / RESUME: genes are split across shards (gene at list-position i handled
by shard ``i % k``); a gene whose ``genes/<gene>.pkl`` already exists is skipped.
only shard 0 runs the control baseline.  with ~thousands of HVGs use a wide
array; submit one array per replicate:
    sbatch --array=0-19%20 analysis/pipelines/perturb/hvg_projection_screen.py --replicate R2
a plain ``sbatch analysis/pipelines/perturb/hvg_projection_screen.py`` is one shard (0/1)
for R1 and screens every HVG in series (slow).
"""

import os
import sys
import time
import argparse
from datetime import datetime

base_dir = os.path.abspath(os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import numpy as np
from tqdm.auto import tqdm

from datasets.mouse_embryo import MouseEmbryoDataset
from analysis.pipelines.perturb.perturb import (
    load_model, build_time_indices_for, impute_tree_states, run_perturbation)
from analysis.pipelines.perturb.projection import (
    build_zero_indices, build_bin_indices, ratio_threshold_diagnostics)


## VERBOSE LOGGING ##

_t0 = time.time()

def log(msg=""):
    """timestamped, immediately-flushed log line (one per call) for SLURM logs."""
    el = time.time() - _t0
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp} | +{el:8.1f}s] {msg}", flush=True)


## CLI (per-run tweaks; everything else stays a module constant) ##
# the screen runs ONE replicate at one target time per invocation and writes to
# its own tagged dir, so the three replicates are just three submissions.

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicate", default="R1",
                    help="replicate to push forward; selects E<target_time>-<replicate>-* trees")
    ap.add_argument("--target-time", type=float, default=9.0,
                    help="leaf/target time of the trees to perturb (also the tree leaf time)")
    ap.add_argument("--knockdown-dose", type=float, default=0.0,
                    help="projection target multiplier: 0.0 = full knockout, "
                         "(0,1) = knockdown, >1 = up-dose, inf = overexpression "
                         "(force every cell, incl. gene-off, onto the top bin)")
    ap.add_argument("--n-dose-bins", type=int, default=4,
                    help="quantile bins for the dose (knockdown) projection; "
                         "ignored for full knockout")
    ap.add_argument("--shard", type=int, default=None, help="this shard's index")
    ap.add_argument("--n-shards", type=int, default=None, help="total number of shards")
    return ap.parse_known_args()[0]

args = parse_args()


## CONFIG ##

experiment_dir = "outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09"


n_steps = 16                # bridge/flow integration steps the splitter takes per 0.5 step
n_count_samples = 1         # count-sampler draws averaged per node (1 = single draw)
split_batch_size = 2048

replicate = args.replicate          # single replicate to push forward per run
start_time = 7.5
target_time = args.target_time
tree_leaf_time = target_time
sample_time_gap = 0.5
topology = "fixed"         # only "fixed" is supported: the observed topology is
                           # replayed

# --- knockdown / overexpression dose --------------------------------------
# multiplier each cell's gene expression is moved to before re-projecting:
#   0.0  full knockout    (project onto the nearest gene-OFF cell)
#   0.5  50% knockdown     (project onto the nearest cell in the expr*0.5 bin)
#   2.0  2x up-dose        (expressing cells pushed up; gene-off cells stay off)
#   inf  overexpression    (force every cell, incl. gene-off, onto the top bin)
knockdown_dose = args.knockdown_dose
n_dose_bins = args.n_dose_bins

off_manifold_ratio = 1.1    # recorded per-node so the cutoff can be swept post-hoc
obs_times = (7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0)
seed = 42

# the output tag is just the dose value (dose0 = full knockout, dose50 = 50%
# knockdown, dose200 = 2x up-dose, doseinf = overexpression). only dose 0.0 takes
# a different projection path (snap onto the gene-OFF set rather than a bin).
if knockdown_dose < 0.0:
    raise ValueError(f"knockdown_dose must be >= 0 (inf = overexpression); "
                     f"got {knockdown_dose}")
perturb_mode = "knockout" if knockdown_dose == 0.0 else "dose"
dose_tag = f"dose{knockdown_dose * 100:.0f}"

# each (dose, target_time, replicate) run gets its own dir so the three
# replicates never share gene/control pkls.
run_tag = f"{dose_tag}_E{target_time}_{replicate}"
out_dir = os.path.join(experiment_dir, f"hvg_projection_screen_v2_{run_tag}")
# pooled states are dose-independent (pooling happens before any perturbation),
# so they live in a shared, un-tagged dir reused across this screen's own dose runs.
pooled_dir = os.path.join(experiment_dir, "hvg_projection_screen", "pooled_sampler")
# per-gene / control trees saved so summaries can be computed post-hoc
trees_dir = os.path.join(out_dir, "genes")
control_dir = os.path.join(out_dir, "control")

log("CONFIG:")
log(f"  experiment_dir   = {experiment_dir}")
log(f"  replicate={replicate} start_time={start_time} target_time={target_time}")
log(f"  topology={topology} off_manifold_ratio={off_manifold_ratio}")
log(f"  knockdown_dose={knockdown_dose} -> mode={perturb_mode!r} tag={dose_tag!r} "
    f"(n_dose_bins={n_dose_bins})")
log(f"  obs_times={obs_times} seed={seed}")
log(f"  n_steps={n_steps} n_count_samples={n_count_samples} split_batch_size={split_batch_size}")
log(f"  out_dir          = {out_dir}")
log(f"  pooled_dir       = {pooled_dir}  (dose-independent, no cross-screen reuse)")


## SHARDING ##

def resolve_shard():
    """shard index / count from CLI args, else the SLURM array env, else 0/1."""
    if args.shard is not None or args.n_shards is not None:
        return (args.shard or 0), (args.n_shards or 1)
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        tid = int(os.environ["SLURM_ARRAY_TASK_ID"])
        tmin = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
        cnt = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        return tid - tmin, cnt
    return 0, 1


shard, n_shards = resolve_shard()
log(f"resolved shard {shard}/{n_shards}")


def atomic_pickle(obj, path):
    import pickle
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def genes_with_off_reference(time_indices, t_lo, t_hi):
    """boolean mask (len = n_genes, column-aligned) of genes that have at least
    one gene-OFF reference cell at *every* timepoint in [t_lo, t_hi].

    full knockout projects each expressing cell onto its nearest gene-OFF cell,
    so a gene expressed in every reference cell at some timepoint the frontier
    passes through has an empty zero-index there and crashes project_frontier.
    """
    import scipy.sparse as sp
    has_off = None
    for t, ti in time_indices.items():
        if not (t_lo - 1e-6 <= t <= t_hi + 1e-6):
            continue
        e = ti["expression"]
        n_cells = e.shape[0]
        nnz = (np.asarray((e != 0).sum(axis=0)).ravel() if sp.issparse(e)
               else np.count_nonzero(e, axis=0))
        off_here = nnz < n_cells          # some cell has the gene off at this t
        has_off = off_here if has_off is None else (has_off & off_here)
    return has_off


## DATASET ##

log(f"[shard {shard}/{n_shards}] Loading dataset...")
dataset = MouseEmbryoDataset()
states = dataset.states
log(f"  dataset loaded: states {states.shape} ({states.n_vars} var_names / HVGs)")
starting_tree_names = [x for x in dataset.trees.keys()
                       if f"E{tree_leaf_time}-{replicate}-" in x]
log(f"  {len(starting_tree_names)} E{tree_leaf_time}-{replicate} trees to pool back.")


## GENE LIST: EVERY HVG IN states.var_names ##

# the genes ARE the dataset's var_names, so every gene is present by construction.
genes = list(states.var_names)
var_to_col = {g: i for i, g in enumerate(states.var_names)}
log(f"\n{len(genes)} HVGs total in states.var_names.")
log(f"  first few: {genes[:10]}")
# the per-shard gene list is settled after the knockout screenability filter
# below (it needs the expression indices), so every shard agrees on the list.

os.makedirs(pooled_dir, exist_ok=True)
os.makedirs(trees_dir, exist_ok=True)
os.makedirs(control_dir, exist_ok=True)
log(f"  output dirs ready: {out_dir}")


## ONE-TIME EXPENSIVE SETUP ##

log("Loading pooler/splitter model + count sampler...")
pooler, splitter, config = load_model(experiment_dir)
count_sampler = None   # unused: topology="fixed" replays the observed tree
log("  model + count sampler loaded.")

log("Building per-timepoint FAISS indices...")
time_indices = build_time_indices_for(dataset)
log(f"  built indices for times: {sorted(time_indices.keys())}")
for t in sorted(time_indices.keys()):
    log(f"    t={t}: {len(time_indices[t]['names'])} reference cells")

log(f"Off-manifold ratio diagnostics (ratio={off_manifold_ratio}, tiny subsamples):")
ratio_threshold_diagnostics(time_indices, off_manifold_ratio, n_sample=64,
                            rng=np.random.default_rng(seed))


## KNOCKOUT SCREENABILITY FILTER (before sharding, so all shards agree) ##

if perturb_mode == "knockout":
    off_ok = genes_with_off_reference(time_indices, start_time, target_time)
    dropped = [g for g, ok in zip(genes, off_ok) if not ok]
    genes = [g for g, ok in zip(genes, off_ok) if ok]
    log(f"Knockout screenability filter: kept {len(genes)}, dropped {len(dropped)} "
        f"HVGs with no gene-OFF cell at some timepoint in "
        f"[{start_time}, {target_time}].")
    if dropped:
        log(f"  dropped (first 20): {dropped[:20]}")

my_genes = [g for i, g in enumerate(genes) if i % n_shards == shard]
log(f"[shard {shard}/{n_shards}] screening {len(my_genes)} of {len(genes)} HVGs "
    f"(positions {shard}, {shard + n_shards}, {shard + 2 * n_shards}, ...).")
log(f"  this shard's first few: {my_genes[:10]}\n")


## PER-TREE POOLING (gene-independent, memoised in RAM + on disk) ##

pooled_cache = {}

def get_pooled(tree_name):
    if tree_name in pooled_cache:
        return pooled_cache[tree_name]
    import pickle
    own = os.path.join(pooled_dir, f"{tree_name}.pkl")
    # only this screen's own (freshly computed) pooled states are reused -- no
    # borrowing from the other screens' caches.
    if os.path.exists(own):
        with open(own, "rb") as fh:
            pooled_cache[tree_name] = pickle.load(fh)
        log(f"  pooled cache HIT for {tree_name} ({own})")
        return pooled_cache[tree_name]
    tree = dataset.trees[tree_name]
    log(f"  pooled cache MISS -- pooling {tree_name} ({tree.number_of_nodes()} nodes)...")
    pooled = impute_tree_states(
        splitter, tree, dataset.leaf_states, batch_size=1024, obs_times=obs_times)
    atomic_pickle(pooled, own)
    pooled_cache[tree_name] = pooled
    log(f"  pooled {tree_name} written to {own}")
    return pooled


def run_proj(tree_name, indices, gene_idx, mode):
    """pool (cached) and propagate one tree; None if it has no start-time layer."""
    pooled_states = get_pooled(tree_name)
    try:
        return run_perturbation(
            dataset.trees[tree_name], pooled_states, splitter, count_sampler,
            indices, gene_idx, target_time, start_time=start_time, mode=mode,
            topology=topology, target_fraction=knockdown_dose, ratio=off_manifold_ratio,
            sample_time_gap=sample_time_gap, n_count_samples=n_count_samples,
            n_steps=n_steps, split_batch_size=split_batch_size, obs_times=obs_times)
    except ValueError as e:
        log(f"  skipping {tree_name}: {e}")
        return None


def light(result):
    """drop the latent states; keep only meta (per leaf) + the lineage tree."""
    return {"meta": result.meta, "tree": result.tree}


def safe_name(gene):
    return gene.replace(os.sep, "_").replace(" ", "_")


## CONTROL (no perturbation) per tree -- only shard 0 writes it ##

if shard == 0:
    log("Running control (no perturbation) per tree...")
    control_indices = {"time": time_indices}
    for tree_name in tqdm(starting_tree_names, desc="Control"):
        cpath = os.path.join(control_dir, f"{tree_name}.pkl")
        if os.path.exists(cpath):
            log(f"  [control/{tree_name}] already done -- skipping.")
            continue
        result = run_proj(tree_name, control_indices, gene_idx=0, mode="control")
        if result is None:
            continue
        atomic_pickle(light(result), cpath)
        log(f"  [control/{tree_name}] saved -> {cpath}")
    log("Control pass complete.")
else:
    log(f"[shard {shard}/{n_shards}] not shard 0 -- skipping control baseline.")


## SCREEN: per (gene, tree) ##

log("\nRunning knockout screen...")
for gi, g in enumerate(tqdm(my_genes, desc=f"shard {shard}")):
    gpath = os.path.join(trees_dir, f"{safe_name(g)}.pkl")
    if os.path.exists(gpath):
        log(f"  [{g}] ({gi + 1}/{len(my_genes)}) already done -- skipping.")
        continue                         # gene fully done (trees on disk)
    log(f"  [{g}] ({gi + 1}/{len(my_genes)}) starting {perturb_mode} "
        f"(dose={knockdown_dose}) over {len(starting_tree_names)} trees...")
    gene_idx = var_to_col[g]
    indices = {"time": time_indices}
    if perturb_mode == "knockout":
        indices["zero"] = build_zero_indices(time_indices, gene_idx)
    else:
        indices["bins"] = build_bin_indices(time_indices, gene_idx, n_bins=n_dose_bins)
    gene_trees = {}
    for tree_name in tqdm(starting_tree_names, desc=g, leave=False):
        result = run_proj(tree_name, indices, gene_idx, mode=perturb_mode)
        if result is None:
            continue
        gene_trees[tree_name] = light(result)
    # all-or-nothing per gene so a resume never overwrites a full pkl with a partial
    atomic_pickle(gene_trees, gpath)
    log(f"  [{g}] done ({len(gene_trees)} trees) -> {gpath}")

log(f"\n[shard {shard}/{n_shards}] Done. Trees saved under {out_dir}")
