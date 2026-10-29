#!/usr/bin/env python
#SBATCH --job-name=tbx5_broadsnap_screen
#SBATCH --output=logs/tbx5_broadsnap_screen_%A_%a.log
#SBATCH --error=logs/tbx5_broadsnap_screen_%A_%a.log
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=bates
"""Tbx5 knockout screen by projection with the permissive (broad) snap window.

``tbx5_projection_screen.py`` with one change: each node snaps to its own
reference time bin *plus* the adjacent bin on each side, the window the coupling
protocol uses (``nearest_neighbor_coupling_w_context``, ``time_bins[i-1:i+2]``).
so on the E7.5..E10.0 grid a node at E9.0 can snap to E8.5, E9.0 or E9.5.
``--snap-window-bins 0`` reproduces the projection screen exactly.

``local_dist`` is measured in the same widened pool, so off-manifold fractions
are not comparable to the projection screen at the same ``ratio`` (the per-node
numbers are recorded, so the cutoff can still be swept post-hoc). for dose runs
the quantile bin edges likewise become window-level; dose 0.0 is unaffected.

``knockdown_dose`` sets the strength: 0.0 = full knockout (project every
expressing cell onto its nearest gene-OFF cell), 0.5 = 50% knockdown (move each
cell to expr*0.5 and project onto the nearest cell in that quantile bin). no
counting here -- the propagated lineage trees are pickled so any summary
(cell-type counts, off-manifold cutoffs, ...) can be computed post-hoc.

OUTPUTS (one dir per replicate,
``tbx5_broadsnap_screen_w<window>_<dose>_E<t>_<rep>/``):
  genes/<gene>.pkl     dict {tree_name: light-tree}
  control/<tree>.pkl   unperturbed baseline tree per starting tree
pooled states are window-, gene- and dose-independent, so they are cached
un-tagged under ``tbx5_broadsnap_screen/pooled_sampler/`` and (unless
``--no-pooled-reuse``) read, never written, from the projection screen's cache.

SHARDING / RESUME: *replicates* are split across shards (replicate at position i
handled by shard ``i % k``); each shard runs the control baseline for its own
replicates, and a gene whose ``genes/<gene>.pkl`` already exists is skipped.
    sbatch analysis/pipelines/perturb/tbx5_broadsnap_screen.py
    sbatch --array=0-2%3 analysis/pipelines/perturb/tbx5_broadsnap_screen.py
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
import scipy.sparse as sp
from tqdm.auto import tqdm

from datasets.mouse_embryo import MouseEmbryoDataset
from analysis.pipelines.perturb.perturb import (
    load_model, impute_tree_states, run_perturbation)
from analysis.pipelines.perturb.projection import (
    build_faiss_index, build_zero_indices, build_bin_indices,
    ratio_threshold_diagnostics)


## VERBOSE LOGGING ##

_t0 = time.time()

def log(msg=""):
    """timestamped, immediately-flushed log line (one per call) for SLURM logs."""
    el = time.time() - _t0
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp} | +{el:8.1f}s] {msg}", flush=True)


## CLI (per-run tweaks; everything else stays a module constant) ##

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicates", nargs="+", default=["R1", "R2", "R3"],
                    help="replicates to push forward; selects E<target_time>-<rep>-* trees")
    ap.add_argument("--target-time", type=float, default=9.5,
                    help="leaf/target time of the trees to perturb (also the tree leaf time)")
    ap.add_argument("--knockdown-dose", type=float, default=0.0,
                    help="projection target multiplier: 0.0 = full knockout, "
                         "(0,1) = knockdown, >1 = up-dose, inf = overexpression "
                         "(force every cell, incl. gene-off, onto the top bin)")
    ap.add_argument("--n-dose-bins", type=int, default=4,
                    help="quantile bins for the dose (knockdown) projection; "
                         "ignored for full knockout")
    ap.add_argument("--snap-window-bins", type=int, default=1,
                    help="adjacent reference time bins added to each side of a "
                         "node's own bin when snapping (1 = the coupling "
                         "protocol's window, 0 = the projection screen's)")
    ap.add_argument("--no-pooled-reuse", action="store_true",
                    help="do not read pooled states from the projection screen's "
                         "cache (they are snap-window independent, so reuse is safe)")
    ap.add_argument("--shard", type=int, default=None, help="this shard's index")
    ap.add_argument("--n-shards", type=int, default=None, help="total number of shards")
    return ap.parse_known_args()[0]

args = parse_args()


## CONFIG ##

experiment_dir = "outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09"
n_steps = 16                # bridge/flow integration steps the splitter takes per 0.5 step
n_count_samples = 1         # count-sampler draws averaged per node (1 = single draw)
split_batch_size = 2048

replicates = args.replicates        # every replicate is covered in one run
start_time = 8.0                    # the crispant window opens at E8.0
target_time = args.target_time
tree_leaf_time = target_time
sample_time_gap = 0.5
topology = "fixed"         # only "fixed" is supported: the observed topology is
                           # replayed

# the single crispant target screened here, as states.var_names spells it.
genes = ["Tbx5"]

# --- snap window -----------------------------------------------------------
# how many reference time bins on each side of a node's own bin join the snap
# candidate pool.  1 = the coupling / assignments protocol (own bin + one
# adjacent slice per side, clamped at the grid ends); 0 = the strict
# nearest-timepoint pool the projection screen uses.
snap_window_bins = args.snap_window_bins
if snap_window_bins < 0:
    raise ValueError(f"snap_window_bins must be >= 0, got {snap_window_bins}")

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

# the output tag is the snap window plus the dose value (dose0 = full knockout,
# dose50 = 50% knockdown, dose200 = 2x up-dose, doseinf = overexpression). only
# dose 0.0 takes a different projection path (snap onto the gene-OFF set rather
# than a bin).
if knockdown_dose < 0.0:
    raise ValueError(f"knockdown_dose must be >= 0 (inf = overexpression); "
                     f"got {knockdown_dose}")
perturb_mode = "knockout" if knockdown_dose == 0.0 else "dose"
dose_tag = f"dose{knockdown_dose * 100:.0f}"
window_tag = f"w{snap_window_bins}"

# each (window, dose, target_time, replicate) gets its own dir so runs never
# share gene/control pkls.
def run_dirs(rep):
    out = os.path.join(
        experiment_dir,
        f"tbx5_broadsnap_screen_{window_tag}_{dose_tag}_E{target_time}_{rep}")
    return out, os.path.join(out, "genes"), os.path.join(out, "control")

# pooled states are window- and dose-independent (pooling happens before any
# snapping or perturbation), so they live in a shared, un-tagged dir.  the
# projection screen's cache holds identical states for the same trees, so it is
# read from as a fallback -- never written to.
pooled_dir = os.path.join(experiment_dir, "tbx5_broadsnap_screen", "pooled_sampler")
pooled_fallback_dirs = ([] if args.no_pooled_reuse else
                        [os.path.join(experiment_dir, "tbx5_projection_screen", "pooled_sampler")])

log("CONFIG:")
log(f"  experiment_dir   = {experiment_dir}")
log(f"  replicates={replicates} start_time={start_time} target_time={target_time}")
log(f"  topology={topology} off_manifold_ratio={off_manifold_ratio}")
log(f"  snap_window_bins={snap_window_bins} -> tag={window_tag!r} "
    f"({'own bin only (= projection screen)' if snap_window_bins == 0 else 'own bin +/- ' + str(snap_window_bins)})")
log(f"  knockdown_dose={knockdown_dose} -> mode={perturb_mode!r} tag={dose_tag!r} "
    f"(n_dose_bins={n_dose_bins})")
log(f"  obs_times={obs_times} seed={seed}")
log(f"  n_steps={n_steps} n_count_samples={n_count_samples} split_batch_size={split_batch_size}")
log(f"  out_dirs         = {[run_dirs(r)[0] for r in replicates]}")
log(f"  pooled_dir       = {pooled_dir}")
log(f"  pooled_fallbacks = {pooled_fallback_dirs}  (read-only)")


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
my_replicates = [r for i, r in enumerate(replicates) if i % n_shards == shard]
log(f"resolved shard {shard}/{n_shards} -- replicates {my_replicates}")


def atomic_pickle(obj, path):
    import pickle
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


## WIDENED (BROAD) SNAP INDICES ##

class ExpressionRowView:
    """row subset of the full expression matrix that does not copy it.

    ``build_time_indices`` stores a real ``expression[mask]`` slice per timepoint;
    with a widened window every cell lands in up to ``2 * window + 1`` slices, so
    the counts matrix would be held several times over.  the core code only ever
    asks a time-index entry for ``expression[row_idxs]``
    (:func:`projection.get_expression_batch`) and ``expression[:, gene_idx]``
    (:func:`projection._gene_column`), so a shim over the global matrix serves
    both.  gene columns are pulled once and shared across windows via
    ``col_cache``.
    """

    def __init__(self, mat, rows, col_cache):
        self.mat = mat
        self.rows = np.asarray(rows)
        self.shape = (len(self.rows), mat.shape[1])
        self._col_cache = col_cache

    def _global_column(self, j):
        if j not in self._col_cache:
            col = self.mat[:, j]
            self._col_cache[j] = (np.asarray(col.todense()).ravel() if sp.issparse(col)
                                  else np.asarray(col).ravel())
        return self._col_cache[j]

    def __getitem__(self, key):
        if isinstance(key, tuple):                 # expression[:, gene_idx]
            rowkey, colkey = key
            if not (isinstance(rowkey, slice) and rowkey == slice(None)):
                raise TypeError(f"ExpressionRowView: unsupported row key {rowkey!r} "
                                f"in a 2-d index; only [:, j] is supported")
            if not np.isscalar(colkey):
                raise TypeError(f"ExpressionRowView: unsupported column key "
                                f"{colkey!r}; only a single gene column is supported")
            return self._global_column(int(colkey))[self.rows]
        return self.mat[self.rows[key]]            # expression[row_idxs]


def build_window_time_indices(dataset, window=1, expr_layer="counts"):
    """per-timepoint faiss indices whose candidate pool is the timepoint's own bin
    plus ``window`` adjacent bins on each side (clamped at the grid ends).

    ``window=0`` reproduces :func:`perturb.build_time_indices_for`.  the keys stay
    the true reference-time grid -- ``run_perturbation`` marches over them and
    places each observed node at its own nearest reference time -- so only the
    *search pool* widens, not the grid.  returns ``(time_indices, rows_by_bin,
    window_bins)``.
    """
    states = dataset.states
    expr = states.layers[expr_layer] if expr_layer in states.layers else states.X
    emb = np.asarray(dataset.reference_features, dtype=np.float32)
    names = np.asarray(states.obs_names.values)
    times = states.obs["time"].values.astype(float)

    grid = [float(t) for t in np.unique(times)]
    rows_by_bin = {t: np.flatnonzero(times == t) for t in grid}
    col_cache = {}

    time_indices, window_bins = {}, {}
    for i, t in enumerate(grid):
        bins = grid[max(0, i - window):i + window + 1]
        rows = np.concatenate([rows_by_bin[b] for b in bins])
        rows.sort()
        w_emb = np.ascontiguousarray(emb[rows], dtype=np.float32)
        time_indices[t] = {
            "index": build_faiss_index(w_emb),
            "embeddings": w_emb,
            "names": names[rows],
            "expression": ExpressionRowView(expr, rows, col_cache),
        }
        window_bins[t] = bins
    return time_indices, rows_by_bin, window_bins


def genes_with_off_reference(expr, rows_by_bin, window_bins, t_lo, t_hi):
    """boolean mask (len = n_genes, column-aligned) of genes that have at least
    one gene-OFF reference cell in *every* snap window in [t_lo, t_hi].

    full knockout projects each expressing cell onto its nearest gene-OFF cell,
    so a gene expressed in every candidate cell of some window the frontier
    passes through has an empty zero-index there and crashes project_frontier.
    counts are taken one bin at a time and summed per window, so the widened
    pools are never materialised.
    """
    in_range = lambda t: t_lo - 1e-6 <= t <= t_hi + 1e-6
    needed = sorted({b for t, bins in window_bins.items() if in_range(t) for b in bins})

    nnz_by_bin, size_by_bin = {}, {}
    for b in needed:
        rows = rows_by_bin[b]
        sub = expr[rows]
        nnz_by_bin[b] = (np.asarray((sub != 0).sum(axis=0)).ravel() if sp.issparse(sub)
                         else np.count_nonzero(sub, axis=0))
        size_by_bin[b] = len(rows)
        del sub

    has_off = None
    for t, bins in window_bins.items():
        if not in_range(t):
            continue
        nnz = sum(nnz_by_bin[b] for b in bins)
        n_cells = sum(size_by_bin[b] for b in bins)
        off_here = nnz < n_cells       # some candidate cell has the gene off here
        has_off = off_here if has_off is None else (has_off & off_here)
    return has_off


## DATASET ##

log(f"[shard {shard}/{n_shards}] Loading dataset...")
dataset = MouseEmbryoDataset()
states = dataset.states
log(f"  dataset loaded: states {states.shape} ({states.n_vars} var_names / HVGs)")
trees_by_replicate = {
    rep: [x for x in dataset.trees.keys() if f"E{tree_leaf_time}-{rep}-" in x]
    for rep in my_replicates}
for rep, names in trees_by_replicate.items():
    log(f"  {len(names)} E{tree_leaf_time}-{rep} trees to pool back.")


## GENE PRESENCE CHECK ##

# the crispant target symbol is already spelled the way states.var_names spells
# it, so a missing gene is a real absence from the HVG set, not a casing miss.
var_to_col = {g: i for i, g in enumerate(states.var_names)}
missing = [g for g in genes if g not in var_to_col]
genes = [g for g in genes if g in var_to_col]
log(f"\n{len(missing)} targets missing from states.var_names: {missing}")
log(f"  {len(genes)} present: {genes}")

os.makedirs(pooled_dir, exist_ok=True)
for rep in my_replicates:
    out_dir, trees_dir, control_dir = run_dirs(rep)
    os.makedirs(trees_dir, exist_ok=True)
    os.makedirs(control_dir, exist_ok=True)
    log(f"  output dirs ready: {out_dir}")


## ONE-TIME EXPENSIVE SETUP ##

log("Loading pooler/splitter model + count sampler...")
pooler, splitter, config = load_model(experiment_dir)
count_sampler = None   # unused: topology="fixed" replays the observed tree
log("  model + count sampler loaded.")

log(f"Building per-timepoint FAISS indices (snap window +/-{snap_window_bins} bins)...")
time_indices, rows_by_bin, window_bins = build_window_time_indices(
    dataset, window=snap_window_bins)
log(f"  built indices for times: {sorted(time_indices.keys())}")
for t in sorted(time_indices.keys()):
    log(f"    t={t}: {len(time_indices[t]['names'])} candidate cells "
        f"(window {window_bins[t]})")

log(f"Off-manifold ratio diagnostics (ratio={off_manifold_ratio}, tiny subsamples):")
ratio_threshold_diagnostics(time_indices, off_manifold_ratio, n_sample=64,
                            rng=np.random.default_rng(seed))


## KNOCKOUT SCREENABILITY FILTER ##

if perturb_mode == "knockout":
    expr_full = states.layers["counts"] if "counts" in states.layers else states.X
    off_ok = genes_with_off_reference(
        expr_full, rows_by_bin, window_bins, start_time, target_time)
    dropped = [g for g in genes if not off_ok[var_to_col[g]]]
    genes = [g for g in genes if off_ok[var_to_col[g]]]
    log(f"Knockout screenability filter: kept {len(genes)}, dropped {len(dropped)} "
        f"targets with no gene-OFF cell in some snap window in "
        f"[{start_time}, {target_time}].")
    if dropped:
        log(f"  dropped: {dropped}")
log(f"[shard {shard}/{n_shards}] screening {genes} over replicates {my_replicates}\n")


## PER-TREE POOLING (gene- and window-independent, memoised in RAM + on disk) ##

pooled_cache = {}

def get_pooled(tree_name):
    if tree_name in pooled_cache:
        return pooled_cache[tree_name]
    import pickle
    own = os.path.join(pooled_dir, f"{tree_name}.pkl")
    # pooling never sees the snap indices, so this screen's own cache and the
    # projection screen's hold identical states -- read either, write only ours.
    for src in [own] + [os.path.join(d, f"{tree_name}.pkl") for d in pooled_fallback_dirs]:
        if not os.path.exists(src):
            continue
        with open(src, "rb") as fh:
            pooled_cache[tree_name] = pickle.load(fh)
        log(f"  pooled cache HIT for {tree_name} ({src})")
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


## SCREEN: per (replicate, gene, tree), control first ##

for rep in my_replicates:
    out_dir, trees_dir, control_dir = run_dirs(rep)
    starting_tree_names = trees_by_replicate[rep]

    log(f"\n=== {rep}: control (no perturbation) per tree ===")
    control_indices = {"time": time_indices}
    for tree_name in tqdm(starting_tree_names, desc=f"Control {rep}"):
        cpath = os.path.join(control_dir, f"{tree_name}.pkl")
        if os.path.exists(cpath):
            log(f"  [control/{tree_name}] already done -- skipping.")
            continue
        result = run_proj(tree_name, control_indices, gene_idx=0, mode="control")
        if result is None:
            continue
        atomic_pickle(light(result), cpath)
        log(f"  [control/{tree_name}] saved -> {cpath}")
    log(f"{rep}: control pass complete.")

    log(f"=== {rep}: knockout screen ===")
    for gi, g in enumerate(tqdm(genes, desc=rep)):
        gpath = os.path.join(trees_dir, f"{safe_name(g)}.pkl")
        if os.path.exists(gpath):
            log(f"  [{rep}/{g}] ({gi + 1}/{len(genes)}) already done -- skipping.")
            continue                         # gene fully done (trees on disk)
        log(f"  [{rep}/{g}] ({gi + 1}/{len(genes)}) starting {perturb_mode} "
            f"(dose={knockdown_dose}) over {len(starting_tree_names)} trees...")
        gene_idx = var_to_col[g]
        indices = {"time": time_indices}
        if perturb_mode == "knockout":
            indices["zero"] = build_zero_indices(time_indices, gene_idx)
        else:
            indices["bins"] = build_bin_indices(time_indices, gene_idx, n_bins=n_dose_bins)
        gene_trees = {}
        for tree_name in tqdm(starting_tree_names, desc=f"{rep}/{g}", leave=False):
            result = run_proj(tree_name, indices, gene_idx, mode=perturb_mode)
            if result is None:
                continue
            gene_trees[tree_name] = light(result)
        # all-or-nothing per gene so a resume never overwrites a full pkl with a partial
        atomic_pickle(gene_trees, gpath)
        log(f"  [{rep}/{g}] done ({len(gene_trees)} trees) -> {gpath}")

log(f"\n[shard {shard}/{n_shards}] Done. Replicates {my_replicates} complete.")
