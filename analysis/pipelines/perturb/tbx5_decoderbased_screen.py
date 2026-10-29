#!/usr/bin/env python
#SBATCH --job-name=tbx5_decoderbased_screen
#SBATCH --output=logs/tbx5_decoderbased_screen_%A_%a.log
#SBATCH --error=logs/tbx5_decoderbased_screen_%A_%a.log
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=bates
"""Tbx5 knockout screen by projection, gene-OFF read off the scvi decoder.

``tbx5_broadsnap_screen.py`` with one change: "expressing" is the atlas scvi
model's decoded rate for the gene, not its raw count. Tbx5 is detected in only
73% of the reference cells that express it, so a count-based gene-OFF set is
~28% dropouts of the very cells the knockout is trying to leave, and both the
trigger (``changed``) and the target (``zero_index``) read it. the decoder runs
on the stored latent, so no count matrix is touched and the rate is cached.

``--off-rate`` is the cutoff, in px_scale per million; below it a cell is OFF.
the default 100 (1e-4) is the antimode of the bimodal rate over the E9.5 lateral
plate. ``--snap-window-bins`` is unchanged (1 = the coupling protocol's window,
0 = the strict projection screen pool), and so is ``knockdown_dose``: 0.0 = full
knockout, 0.5 = 50% knockdown onto the rate's quantile bins. no counting here --
the propagated lineage trees are pickled so any summary can be computed post-hoc.

OUTPUTS (one dir per replicate,
``tbx5_decoderbased_screen_v2_w<window>_r<rate>_<dose>_E<t>_<rep>/``):
  genes/<gene>.pkl     dict {tree_name: light-tree}
  control/<tree>.pkl   unperturbed baseline tree per starting tree
pooled states are window-, rate-, gene- and dose-independent, so they are cached
un-tagged under ``tbx5_decoderbased_screen/pooled_sampler/`` and (unless
``--no-pooled-reuse``) read, never written, from the other two screens' caches.

SHARDING / RESUME: *replicates* are split across shards (replicate at position i
handled by shard ``i % k``); each shard runs the control baseline for its own
replicates, and a gene whose ``genes/<gene>.pkl`` already exists is skipped.
    sbatch analysis/pipelines/perturb/tbx5_decoderbased_screen.py
    sbatch --array=0-2%3 analysis/pipelines/perturb/tbx5_decoderbased_screen.py
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
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from datasets.mouse_embryo import MouseEmbryoDataset
from analysis.pipelines.perturb.perturb import (
    load_model, impute_tree_states, run_perturbation)
from analysis.pipelines.perturb.projection import (
    build_faiss_index, build_zero_indices, build_bin_indices,
    ratio_threshold_diagnostics, _gene_column)


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
    ap.add_argument("--off-rate", type=float, default=100.0,
                    help="scvi decoded rate (px_scale per million) at or below "
                         "which a reference cell counts as gene-OFF")
    ap.add_argument("--snap-window-bins", type=int, default=1,
                    help="adjacent reference time bins added to each side of a "
                         "node's own bin when snapping (1 = the coupling "
                         "protocol's window, 0 = the projection screen's)")
    ap.add_argument("--no-pooled-reuse", action="store_true",
                    help="do not read pooled states from the other screens' "
                         "caches (they are snap-window and rate independent, so "
                         "reuse is safe)")
    ap.add_argument("--shard", type=int, default=None, help="this shard's index")
    ap.add_argument("--n-shards", type=int, default=None, help="total number of shards")
    return ap.parse_known_args()[0]

args = parse_args()


## CONFIG ##

experiment_dir = "outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09"
# the atlas's own scvi fit, the one X_scvi (and so reference_features) came out of.
scvi_model_path = "../data/embryos/scvi_model_hvgs_and_scvi/model.pt"
scvi_dataset_cov = "devmap"   # the `dataset` covariate the decoder is conditioned on
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

# --- decoder-based gene-OFF -----------------------------------------------
# a reference cell is gene-OFF when the scvi decoder's rate for it is at or below
# `off_rate`, in px_scale per million.  the column the projection reads is the
# rate with everything at or below the cutoff floored to 0, so `count != 0` in
# `project_frontier` and `count == 0` in `build_zero_indices` both become the
# rate test without either of them changing.
off_rate = args.off_rate
if off_rate <= 0.0:
    raise ValueError(f"off_rate must be > 0, got {off_rate}")
rate_scale = 1e6            # px_scale is a simplex, so per-million keeps the dose
                            # bins' `np.round` (integer counts) meaningful

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
rate_tag = f"r{off_rate:g}"

# each (window, rate, dose, target_time, replicate) gets its own dir so runs never
# share gene/control pkls.
def run_dirs(rep):
    out = os.path.join(
        experiment_dir,
        f"tbx5_decoderbased_screen_v2_{window_tag}_{rate_tag}_{dose_tag}_E{target_time}_{rep}")
    return out, os.path.join(out, "genes"), os.path.join(out, "control")

# pooled states are window-, rate- and dose-independent (pooling happens before
# any snapping or perturbation), so they live in a shared, un-tagged dir.  the
# other two screens' caches hold identical states for the same trees, so they are
# read from as fallbacks -- never written to.
pooled_dir = os.path.join(experiment_dir, "tbx5_decoderbased_screen", "pooled_sampler")
pooled_fallback_dirs = ([] if args.no_pooled_reuse else
                        [os.path.join(experiment_dir, "tbx5_broadsnap_screen", "pooled_sampler"),
                         os.path.join(experiment_dir, "tbx5_projection_screen", "pooled_sampler")])
# the decoded rate is gene-, window- and dose-independent, so it caches beside them.
rate_dir = os.path.join(experiment_dir, "tbx5_decoderbased_screen", "rates")

log("CONFIG:")
log(f"  experiment_dir   = {experiment_dir}")
log(f"  replicates={replicates} start_time={start_time} target_time={target_time}")
log(f"  topology={topology} off_manifold_ratio={off_manifold_ratio}")
log(f"  scvi_model       = {scvi_model_path} (cov {scvi_dataset_cov!r})")
log(f"  off_rate={off_rate} px_scale/1e6 -> tag={rate_tag!r} (rate_scale={rate_scale:g})")
log(f"  snap_window_bins={snap_window_bins} -> tag={window_tag!r} "
    f"({'own bin only (= projection screen)' if snap_window_bins == 0 else 'own bin +/- ' + str(snap_window_bins)})")
log(f"  knockdown_dose={knockdown_dose} -> mode={perturb_mode!r} tag={dose_tag!r} "
    f"(n_dose_bins={n_dose_bins})")
log(f"  obs_times={obs_times} seed={seed}")
log(f"  n_steps={n_steps} n_count_samples={n_count_samples} split_batch_size={split_batch_size}")
log(f"  out_dirs         = {[run_dirs(r)[0] for r in replicates]}")
log(f"  pooled_dir       = {pooled_dir}")
log(f"  rate_dir         = {rate_dir}")
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


def safe_name(gene):
    return gene.replace(os.sep, "_").replace(" ", "_")


def atomic_pickle(obj, path):
    import pickle
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


## DECODED GENE-OFF ##

def decoded_rate(z, gene_idx, var_names):
    """the scvi decoder's px_scale for one gene, over the states in `z`.

    the decoder is a function of the latent alone (plus the one-hot `dataset`
    covariate), and `z` is the same X_scvi the projection snaps in, so nothing
    reads the count matrix.  rebuilt from the checkpoint rather than through
    ``SCVI.load``, which would want the 3.9M x 3956 adata back.
    """
    ck = torch.load(scvi_model_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    assert list(ck["var_names"]) == list(var_names), \
        "scvi model var_names do not match states.var_names"
    cov_map = list(ck["attr_dict"]["registry_"]["field_registries"]
                   ["extra_categorical_covs"]["state_registry"]["mappings"]["dataset"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    layers, i = [], 0
    while f"decoder.px_decoder.fc_layers.Layer {i}.0.weight" in sd:
        q = f"decoder.px_decoder.fc_layers.Layer {i}"
        lin = nn.Linear(*sd[f"{q}.0.weight"].shape[::-1])
        lin.weight.data, lin.bias.data = sd[f"{q}.0.weight"], sd[f"{q}.0.bias"]
        bn = nn.BatchNorm1d(sd[f"{q}.1.weight"].shape[0], eps=1e-3, momentum=0.01)
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(bn, attr).data = sd[f"{q}.1.{attr}"]
        layers.append((lin.to(dev).eval(), bn.to(dev).eval()))
        i += 1
    W = sd["decoder.px_scale_decoder.0.weight"].to(dev)
    B = sd["decoder.px_scale_decoder.0.bias"].to(dev)
    cov = torch.zeros(1, len(cov_map), device=dev)
    cov[0, cov_map.index(scvi_dataset_cov)] = 1.0

    out = np.empty(len(z), dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, len(z), 200_000):
            h = torch.as_tensor(np.ascontiguousarray(z[s0:s0 + 200_000]),
                                dtype=torch.float32, device=dev)
            c = cov.expand(len(h), cov.shape[1])
            for lin, bn in layers:
                h = torch.relu(bn(lin(torch.cat([h, c], 1))))
            out[s0:s0 + 200_000] = torch.softmax(
                h @ W.T + B, dim=-1)[:, gene_idx].float().cpu().numpy()
    return out


def off_floored_rate(dataset, gene, gene_idx):
    """the decoded rate per reference cell, scaled and floored to 0 at `off_rate`.

    this is the column that stands in for the gene's counts: it is 0 exactly on
    the cells the knockout may land on, so `build_zero_indices` picks them out
    and `project_frontier`'s `changed` fires on the rest, both unmodified.
    """
    os.makedirs(rate_dir, exist_ok=True)
    z = np.asarray(dataset.reference_features, dtype=np.float32)
    path = os.path.join(rate_dir, f"rate_{safe_name(gene)}.npy")
    if os.path.exists(path):
        r = np.load(path)
        if len(r) == len(z):
            log(f"  decoded rate cache HIT for {gene} ({path})")
        else:
            log(f"  decoded rate cache STALE for {gene} ({len(r)} != {len(z)}) -- redoing")
            r = None
    else:
        r = None
    if r is None:
        log(f"  decoding {gene} rate over {len(z)} reference cells...")
        r = decoded_rate(z, gene_idx, dataset.states.var_names) * rate_scale
        np.save(path, r)
        log(f"  decoded rate written to {path}")
    off = r <= off_rate
    log(f"  {gene}: rate median={np.median(r):.4g} per 1e6, "
        f"{off.sum()} / {len(r)} cells OFF at <= {off_rate:g} ({off.mean():.1%})")
    return np.where(off, 0.0, r).astype(np.float32)


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

    ``override`` maps a global column index to a full-length column that stands
    in for the matrix's own -- here the decoded, floored rate for the screened
    gene.  it is served on both paths, so the core code reads it wherever it
    would have read the counts, and every other gene is untouched.
    """

    def __init__(self, mat, rows, col_cache, override=None):
        self.mat = mat
        self.rows = np.asarray(rows)
        self.shape = (len(self.rows), mat.shape[1])
        self._col_cache = col_cache
        self.override = override or {}

    def _global_column(self, j):
        if j in self.override:
            return self.override[j]
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
        rows = self.rows[key]                      # expression[row_idxs]
        sub = self.mat[rows]
        if not self.override:
            return sub
        sub = np.asarray(sub.todense() if sp.issparse(sub) else sub, dtype=np.float32)
        for j, col in self.override.items():
            sub[:, j] = col[rows]
        return sub


def build_window_time_indices(dataset, window=1, expr_layer="counts", override=None):
    """per-timepoint faiss indices whose candidate pool is the timepoint's own bin
    plus ``window`` adjacent bins on each side (clamped at the grid ends).

    ``override`` is handed to every :class:`ExpressionRowView`, so the screened
    gene reads as its decoded rate in every window.

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
            "expression": ExpressionRowView(expr, rows, col_cache, override),
        }
        window_bins[t] = bins
    return time_indices, rows_by_bin, window_bins


def genes_with_off_reference(time_indices, gene_idx, t_lo, t_hi):
    """whether every snap window in [t_lo, t_hi] holds at least one gene-OFF cell.

    full knockout projects each expressing cell onto its nearest gene-OFF cell,
    so a gene on in every candidate cell of some window the frontier passes
    through has an empty zero-index there and crashes project_frontier.  the
    windows already carry the decoded column, so this is the same test the
    projection will make, one gene at a time rather than all 3956 at once.
    """
    return all((_gene_column(ti, gene_idx) == 0).any()
               for t, ti in time_indices.items() if t_lo - 1e-6 <= t <= t_hi + 1e-6)


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

log(f"Decoding the gene-OFF column (off_rate={off_rate:g} per 1e6)...")
overrides = {var_to_col[g]: off_floored_rate(dataset, g, var_to_col[g]) for g in genes}

log(f"Building per-timepoint FAISS indices (snap window +/-{snap_window_bins} bins)...")
time_indices, _, window_bins = build_window_time_indices(
    dataset, window=snap_window_bins, override=overrides)
log(f"  built indices for times: {sorted(time_indices.keys())}")
for t in sorted(time_indices.keys()):
    log(f"    t={t}: {len(time_indices[t]['names'])} candidate cells "
        f"(window {window_bins[t]})")

log(f"Off-manifold ratio diagnostics (ratio={off_manifold_ratio}, tiny subsamples):")
ratio_threshold_diagnostics(time_indices, off_manifold_ratio, n_sample=64,
                            rng=np.random.default_rng(seed))


## KNOCKOUT SCREENABILITY FILTER ##

if perturb_mode == "knockout":
    off_ok = {g: genes_with_off_reference(time_indices, var_to_col[g],
                                          start_time, target_time) for g in genes}
    dropped = [g for g in genes if not off_ok[g]]
    genes = [g for g in genes if off_ok[g]]
    log(f"Knockout screenability filter: kept {len(genes)}, dropped {len(dropped)} "
        f"targets with no gene-OFF cell in some snap window in "
        f"[{start_time}, {target_time}].")
    if dropped:
        log(f"  dropped: {dropped}")
log(f"[shard {shard}/{n_shards}] screening {genes} over replicates {my_replicates}\n")


## PER-TREE POOLING (gene-, window- and rate-independent, memoised RAM + disk) ##

pooled_cache = {}

def get_pooled(tree_name):
    if tree_name in pooled_cache:
        return pooled_cache[tree_name]
    import pickle
    own = os.path.join(pooled_dir, f"{tree_name}.pkl")
    # pooling never sees the snap indices, so this screen's own cache and the
    # other two screens' hold identical states -- read either, write only ours.
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
