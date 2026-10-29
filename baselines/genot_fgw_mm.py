#!/usr/bin/env python
#SBATCH --job-name=genot_fgw_unified
#SBATCH --output=genot_fgw_unified_%j.log
#SBATCH --error=genot_fgw_unified_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=bates
"""
GENOT + FGW + timepoint conditioning for multimarginal (optimised)

Implementation notes:
  - TreeDistanceOracle: Euler-tour + sparse-table RMQ gives O(1)
    pairwise tree distance using node 'time' attributes
  - JIT-compile the JAX training body (interpolation + gradient step).
  - JIT-compile the inference transport loop.
  - keep per-transition expression data as jax arrays to avoid repeated
    numpy -> jax copies every step.
"""
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

sys.path.insert(
    0,
    os.environ.get(
        "SLURM_SUBMIT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
    ),
)
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
)

from datasets.mouse_embryo import MouseEmbryoDataset

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pickle
import flax.nnx as nnx

from ott.geometry import pointcloud, geometry
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from ott.solvers.linear import sinkhorn

from ott.neural.methods.flow_matching import (
    flow_matching_step,
    interpolate_samples,
    evaluate_velocity_field,
)
from ott.neural.networks.velocity_field.mlp import MLP

from baselines import genot_common as gc

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
FGW_BATCH_SIZE = 512
BATCH_SIZE = 512
N_ITERS = 5_000
EPSILON = 5e-2
FUSED_PENALTY = 1.0
HIDDEN_DIMS = (1024, 1024, 1024)
LR = 1e-4
N_SAMPLES = 5
SEED = 42
OUTPUT_PATH = "outputs/e_baselines/genot_fgw_unified_model.pkl"
LOG_EVERY = 5

rng_np = np.random.default_rng(SEED)


class TreeDistanceOracle:
    """O(n log n) build, O(1) per pairwise distance query.

    Uses the node 'time' attribute from the directed tree.
    dist(u, v) = time(u) + time(v) - 2 * time(LCA(u, v))

    The LCA is found via Euler tour + sparse table RMQ:
      1. Root the tree (the unique node with in-degree 0).
      2. Run an iterative DFS, recording every node visit (including
         backtracks) into an Euler tour of length 2n - 1.
      3. Record each node's time and its first position in the tour.
      4. Build a sparse table over the tour-times so that any
         range-minimum query (= LCA query) is answered in O(1).
    """

    def __init__(self, tree):
        """Build the oracle from an nx.DiGraph whose nodes have a 'time' attr.

        Parameters
        ----------
        tree : nx.DiGraph
            A rooted directed tree (arborescence).  Each node must carry
            a ``'time'`` attribute (float).
        """
        nodes = list(tree.nodes)
        n = len(nodes)
        self.node_to_int = {v: i for i, v in enumerate(nodes)}

        # node times (indexed by internal int id)
        self.node_time = np.empty(n, dtype=np.float64)
        for v in nodes:
            self.node_time[self.node_to_int[v]] = float(tree.nodes[v]["time"])

        # find root: in-degree 0 in the DiGraph
        roots = [v for v in nodes if tree.in_degree(v) == 0]
        if len(roots) != 1:
            raise ValueError(
                f"Expected exactly 1 root (in-degree 0), found {len(roots)}"
            )
        root = roots[0]
        root_int = self.node_to_int[root]

        # --- iterative DFS to build the Euler tour ---
        # For a directed tree we follow successors (children).
        euler = []                                       # node-int at each tour step
        first = np.full(n, -1, dtype=np.int32)           # first occurrence in tour

        # stack entries: (node_int, child_iterator)
        # We use the directed successors so we naturally go root -> leaves.
        # int_to_node needed to call tree.successors with the original label.
        int_to_node = nodes
        stack = [(root_int, iter(tree.successors(root)))]
        euler.append(root_int)
        first[root_int] = 0

        while stack:
            u_int, children = stack[-1]
            advanced = False
            for child_orig in children:                  # resumes the iterator
                c_int = self.node_to_int[child_orig]
                euler.append(c_int)
                if first[c_int] == -1:
                    first[c_int] = len(euler) - 1
                stack.append((c_int, iter(tree.successors(child_orig))))
                advanced = True
                break                                    # one child at a time
            if not advanced:
                stack.pop()
                if stack:
                    # backtrack: re-record the parent
                    euler.append(stack[-1][0])

        self.euler = np.array(euler, dtype=np.int32)
        self.first = first

        # tour_time[i] = time of the node at euler[i]
        tour_time = self.node_time[self.euler]
        self._tour_time = tour_time

        # --- sparse table for range-minimum query on tour_time ---
        m = len(euler)
        K = max(int(np.floor(np.log2(m))), 0) + 1
        # table[k][i] = *tour index* with min time in euler[i .. i + 2^k - 1]
        table = np.zeros((K, m), dtype=np.int32)
        table[0] = np.arange(m, dtype=np.int32)
        for k in range(1, K):
            half = 1 << (k - 1)
            limit = m - (1 << k) + 1
            left = table[k - 1, :limit]
            right = table[k - 1, half:limit + half]
            pick_right = tour_time[right] < tour_time[left]
            table[k, :limit] = np.where(pick_right, right, left)
        self._table = table

        self._log2 = np.zeros(m + 1, dtype=np.int32)
        for i in range(2, m + 1):
            self._log2[i] = self._log2[i // 2] + 1

    # ---- vectorised RMQ --------------------------------------------------

    def _rmq(self, lo, hi):
        """Return tour indices of range-min for each (lo, hi) pair.

        Parameters are 1-D int arrays of equal length.
        """
        length = hi - lo + 1
        k = self._log2[length]
        left = self._table[k, lo]
        right = self._table[k, hi - (1 << k) + 1]
        pick = self._tour_time[right] < self._tour_time[left]
        return np.where(pick, right, left)

    # ---- public API ------------------------------------------------------

    def batch_distances(self, cells):
        """Return the full pairwise distance matrix for a list of cell names.

        dist(u, v) = time(u) + time(v) - 2 * time(LCA(u, v))

        Parameters
        ----------
        cells : array-like of node labels

        Returns
        -------
        D : np.ndarray, shape (len(cells), len(cells)), float32
        """
        idx = np.array([self.node_to_int[c] for c in cells])
        times = self.node_time[idx]                        # (m,)
        pos = self.first[idx]                              # (m,)

        lo = np.minimum(pos[:, None], pos[None, :])        # (m, m)
        hi = np.maximum(pos[:, None], pos[None, :])

        flat_lo = lo.ravel()
        flat_hi = hi.ravel()
        lca_tour_idx = self._rmq(flat_lo, flat_hi)         # (m*m,)
        lca_node = self.euler[lca_tour_idx]
        lca_time = self.node_time[lca_node].reshape(lo.shape)

        D = times[:, None] + times[None, :] - 2.0 * lca_time
        return D.astype(np.float32)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
log.info("loading dataset")
dataset = MouseEmbryoDataset()
states = dataset.states
trees = dataset.trees
timepoints = sorted(states.obs["time"].unique())
log.info("found %d timepoints: %s", len(timepoints), timepoints)

t_min = float(min(timepoints))
t_max = float(max(timepoints))
t_range = t_max - t_min if t_max != t_min else 1.0


def normalise_time(t):
    return (float(t) - t_min) / t_range


log.info("building tree-distance oracles")
oracles = {}
for clone_id, tree in trees.items():
    oracles[clone_id] = TreeDistanceOracle(tree)
    log.info("  clone %s: %d nodes", clone_id, len(tree.nodes))
log.info("  built oracles for %d clones", len(oracles))


def compute_minibatch_tree_distances(cells, clone_ids):
    """Pairwise tree distances for a minibatch, using O(1) oracle lookups."""
    n = len(cells)
    D = np.zeros((n, n), dtype=np.float32)
    max_within = 0.0

    for clone in np.unique(clone_ids):
        mask = np.where(clone_ids == clone)[0]
        if len(mask) < 2:
            continue
        sub = oracles[clone].batch_distances(cells[mask])
        D[np.ix_(mask, mask)] = sub
        m = sub.max()
        if m > max_within:
            max_within = m

    cross_dist = max_within * 2 if max_within > 0 else 1.0
    cross_mask = clone_ids[:, None] != clone_ids[None, :]
    D[cross_mask] = cross_dist
    return D


# ---------------------------------------------------------------------------
# FGW coupling
# ---------------------------------------------------------------------------
def solve_fgw_coupling(expr_src, expr_tgt, tree_src, tree_tgt,
                       epsilon=EPSILON, fused_penalty=FUSED_PENALTY):
    geom_xx = geometry.Geometry(cost_matrix=jnp.array(tree_src), epsilon=epsilon)
    geom_yy = geometry.Geometry(cost_matrix=jnp.array(tree_tgt), epsilon=epsilon)
    geom_xy = pointcloud.PointCloud(
        jnp.array(expr_src), jnp.array(expr_tgt), epsilon=epsilon
    )
    prob = quadratic_problem.QuadraticProblem(
        geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty
    )
    linear_solver = sinkhorn.Sinkhorn()
    solver = gromov_wasserstein.GromovWasserstein(linear_solver, epsilon=epsilon)
    return np.asarray(solver(prob).matrix)


def sample_pairs_from_coupling(coupling, n_pairs, rng_key):
    flat = coupling.ravel()
    flat = flat / flat.sum()
    n_src, n_tgt = coupling.shape
    indices = jr.choice(
        rng_key, len(flat), shape=(n_pairs,), p=jnp.array(flat)
    )
    src_idx = np.asarray(indices // n_tgt)
    tgt_idx = np.asarray(indices % n_tgt)
    return src_idx, tgt_idx


# ---------------------------------------------------------------------------
# transition data preparation
# ---------------------------------------------------------------------------
def prepare_clone_pair(sc, tc, src_mask, tgt_mask):
    src_cells = states.obs_names[src_mask & (states.obs["clone"] == sc)]
    src_cells = np.array([x for x in src_cells if x in trees[sc].nodes])
    tgt_cells = states.obs_names[tgt_mask & (states.obs["clone"] == tc)]
    tgt_cells = np.array([x for x in tgt_cells if x in trees[tc].nodes])
    if len(src_cells) == 0 or len(tgt_cells) == 0:
        return None
    src_expr = np.asarray(states[src_cells].obsm["X_scvi"], dtype=np.float32)
    tgt_expr = np.asarray(states[tgt_cells].obsm["X_scvi"], dtype=np.float32)
    return dict(
        src_cells=src_cells, tgt_cells=tgt_cells,
        src_expr=src_expr, tgt_expr=tgt_expr,
        src_clone=sc, tgt_clone=tc,
    )


log.info("preparing transition data (forward + backward)")
transition_data = {}

for i in range(len(timepoints) - 1):
    t0, t1 = timepoints[i], timepoints[i + 1]
    src_mask = states.obs["time"] == t0
    tgt_mask = states.obs["time"] == t1
    src_clones = states.obs.loc[src_mask, "clone"].unique()
    tgt_clones = states.obs.loc[tgt_mask, "clone"].unique()

    all_pairs = []
    for tc in tgt_clones:
        sc = rng_np.choice(src_clones)
        p = prepare_clone_pair(sc, tc, src_mask, tgt_mask)
        if p is not None:
            all_pairs.append(p)

    if not all_pairs:
        log.warning("%s <-> %s: no valid clone pairs, skipping", t0, t1)
        continue

    src_exprs = np.concatenate([p["src_expr"] for p in all_pairs])
    tgt_exprs = np.concatenate([p["tgt_expr"] for p in all_pairs])
    src_cells = np.concatenate([p["src_cells"] for p in all_pairs])
    tgt_cells = np.concatenate([p["tgt_cells"] for p in all_pairs])
    src_clone_ids = np.concatenate([
        np.full(len(p["src_cells"]), p["src_clone"]) for p in all_pairs
    ])
    tgt_clone_ids = np.concatenate([
        np.full(len(p["tgt_cells"]), p["tgt_clone"]) for p in all_pairs
    ])

    src_jnp = jnp.array(src_exprs)
    tgt_jnp = jnp.array(tgt_exprs)

    base = dict(
        src_expr_np=src_exprs, tgt_expr_np=tgt_exprs,
        src_expr=src_jnp, tgt_expr=tgt_jnp,
        src_cells=src_cells, tgt_cells=tgt_cells,
        src_clone_ids=src_clone_ids, tgt_clone_ids=tgt_clone_ids,
    )

    transition_data[(t0, t1)] = {
        **base,
        "t_src_norm": normalise_time(t0),
        "t_tgt_norm": normalise_time(t1),
    }
    transition_data[(t1, t0)] = {
        "src_expr_np": tgt_exprs, "tgt_expr_np": src_exprs,
        "src_expr": tgt_jnp, "tgt_expr": src_jnp,
        "src_cells": tgt_cells, "tgt_cells": src_cells,
        "src_clone_ids": tgt_clone_ids, "tgt_clone_ids": src_clone_ids,
        "t_src_norm": normalise_time(t1), "t_tgt_norm": normalise_time(t0),
    }

    n_s, n_t = len(src_exprs), len(tgt_exprs)
    log.info("  %s -> %s: %d src, %d tgt", t0, t1, n_s, n_t)
    log.info("  %s -> %s: %d src, %d tgt (backward)", t1, t0, n_t, n_s)

transition_keys = list(transition_data.keys())
n_transitions = len(transition_keys)
log.info("total transitions (fwd + bwd): %d", n_transitions)

expr_dim = next(iter(transition_data.values()))["src_expr_np"].shape[1]
cond_dim = expr_dim + 2

# ---------------------------------------------------------------------------
# model + optimiser
# ---------------------------------------------------------------------------
log.info("training unified velocity field (%d iterations)", N_ITERS)

model = MLP(
    dim=expr_dim,
    cond_dim=cond_dim,
    hidden_dims=HIDDEN_DIMS,
    rngs=nnx.Rngs(SEED),
)
optimizer = nnx.ModelAndOptimizer(model, optax.adam(learning_rate=LR))


# ---------------------------------------------------------------------------
# JIT-compiled training step
# ---------------------------------------------------------------------------
@nnx.jit
def jit_train_step(model, optimizer, src_batch_expr, tgt_batch,
                   t_src_norm, t_tgt_norm, k_noise, k_interp):
    """one flow-matching gradient step (compiled)."""
    batch_size = src_batch_expr.shape[0]
    t_src_vec = jnp.full((batch_size, 1), t_src_norm)
    t_tgt_vec = jnp.full((batch_size, 1), t_tgt_norm)
    cond_batch = jnp.concatenate([src_batch_expr, t_src_vec, t_tgt_vec], axis=1)
    noise = jr.normal(k_noise, shape=tgt_batch.shape)
    batch = interpolate_samples(k_interp, x0=noise, x1=tgt_batch, cond=cond_batch)
    metrics = flow_matching_step(model, optimizer, batch)
    return metrics


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
master_key = jr.PRNGKey(SEED)

# loss curve, recorded at the logging cadence (avoids a host sync every step)
loss_curve = {"step": [], "loss": [], "grad_norm": []}

for step in range(N_ITERS):
    master_key, k_trans, k_src, k_tgt, k_pair, k_noise, k_interp = jr.split(
        master_key, 7
    )

    # pick a random transition
    tr_idx = int(jr.randint(k_trans, (), 0, n_transitions))
    tr_key = transition_keys[tr_idx]
    td = transition_data[tr_key]

    n_src = td["src_expr_np"].shape[0]
    n_tgt = td["tgt_expr_np"].shape[0]
    mb_src = min(FGW_BATCH_SIZE, n_src)
    mb_tgt = min(FGW_BATCH_SIZE, n_tgt)

    s_idx = np.asarray(jr.choice(k_src, n_src, shape=(mb_src,), replace=False))
    t_idx = np.asarray(jr.choice(k_tgt, n_tgt, shape=(mb_tgt,), replace=False))

    # numpy slices for the FGW solver (needs numpy cost matrices)
    mb_src_expr_np = td["src_expr_np"][s_idx]
    mb_tgt_expr_np = td["tgt_expr_np"][t_idx]

    # tree distances via O(1) oracle lookups (fully vectorised)
    mb_src_tree = compute_minibatch_tree_distances(
        td["src_cells"][s_idx], td["src_clone_ids"][s_idx]
    )
    mb_tgt_tree = compute_minibatch_tree_distances(
        td["tgt_cells"][t_idx], td["tgt_clone_ids"][t_idx]
    )

    coupling = solve_fgw_coupling(
        mb_src_expr_np, mb_tgt_expr_np, mb_src_tree, mb_tgt_tree
    )

    si, ti = sample_pairs_from_coupling(coupling, BATCH_SIZE, k_pair)

    src_batch_expr = td["src_expr"][s_idx[si]]
    tgt_batch = td["tgt_expr"][t_idx[ti]]

    metrics = jit_train_step(
        model, optimizer,
        src_batch_expr, tgt_batch,
        td["t_src_norm"], td["t_tgt_norm"],
        k_noise, k_interp,
    )

    if step % LOG_EVERY == 0 or step == N_ITERS - 1:
        loss_val = float(metrics["loss"])
        grad_val = float(metrics["grad_norm"])
        loss_curve["step"].append(step)
        loss_curve["loss"].append(loss_val)
        loss_curve["grad_norm"].append(grad_val)
        log.info(
            "step %6d/%d  transition %s  loss=%.6f  |grad|=%.4f",
            step, N_ITERS, tr_key,
            loss_val, grad_val,
        )

# ---------------------------------------------------------------------------
# serialise
# ---------------------------------------------------------------------------
log.info("serialising model parameters")
flat_numpy, model_config = gc.serialize_mlp(model, expr_dim, cond_dim, HIDDEN_DIMS)


# ---------------------------------------------------------------------------
# ancestor maps via forward transport (coupling between adjacent timepoints)
# ---------------------------------------------------------------------------
log.info("building ancestor mappings via forward transport")
forward_transitions = {}
for i in range(len(timepoints) - 1):
    t0, t1 = timepoints[i], timepoints[i + 1]
    td = transition_data.get((t0, t1))
    if td is None:
        continue
    forward_transitions[(t0, t1)] = dict(
        src_expr=td["src_expr_np"], tgt_expr=td["tgt_expr_np"],
        src_cells=td["src_cells"], tgt_cells=td["tgt_cells"],
    )

direct_ancestor = gc.build_direct_ancestor(
    model, forward_transitions, timepoints,
    expr_dim=expr_dim, normalise_time=normalise_time,
    key=master_key, n_samples=N_SAMPLES,
)

log.info("chaining direct ancestors into full ancestral mapping")
ancestral_mapping = gc.chain_ancestors(direct_ancestor, states, timepoints)

# ---------------------------------------------------------------------------
# per-node assignments by backward chain through real cells
# ---------------------------------------------------------------------------
log.info("building per-node assignments by backward chain")
parent, m = gc.backward_parent_map(
    model, states, t_min=t_min, t_max=t_max,
    key=jr.PRNGKey(SEED), n_samples=N_SAMPLES,
)
assignments = gc.build_assignments(trees, states, parent, m, seed=SEED)

# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
output = {
    "flat_params": flat_numpy,
    "model_config": model_config,
    "t_min": t_min,
    "t_max": t_max,
    "timepoints": timepoints,
    "ancestral_mapping": ancestral_mapping,
    "direct_ancestor": direct_ancestor,
    "assignments": assignments,
    "loss_curve": loss_curve,
}

log.info("saving to %s", OUTPUT_PATH)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(output, f)

log.info("done")