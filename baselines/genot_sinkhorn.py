#!/usr/bin/env python
#SBATCH --job-name=genot_w2_unified
#SBATCH --output=genot_w2_unified_%j.log
#SBATCH --error=genot_w2_unified_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=bates
"""
GENOT + Sinkhorn (standard Wasserstein) + timepoint conditioning for multimarginal

At each training step:
  1. sample a random transition (t_src, t_tgt), may be forward or backward.
  2. draw a minibatch of source/target cells for that transition.
  3. solve a small entropic OT (Sinkhorn) on gene expression.
  4. draw (source, target) pairs from the coupling.
  5. augment the source condition with normalised (t_src, t_tgt).
  6. one flow-matching gradient step on the shared velocity field.

At inference, specify any (t_src, t_tgt) pair and integrate the learned ODE
to transport source cells to the target time.
"""
import os
import sys
import logging

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from ott.neural.methods.flow_matching import (
    flow_matching_step,
    interpolate_samples,
    evaluate_velocity_field,
)
from ott.neural.networks.velocity_field.mlp import MLP

from baselines import genot_common as gc

# config
OT_BATCH_SIZE = 512       # cells per side for minibatch Sinkhorn solves
BATCH_SIZE = 512          # flow-matching pairs drawn from each minibatch OT
N_ITERS = 5_000         # total training iterations
EPSILON = 5e-2           # entropic regularisation for Sinkhorn
HIDDEN_DIMS = (1024, 1024, 1024)
LR = 1e-4
N_SAMPLES = 5            # stochastic samples per source cell at inference
SEED = 42
OUTPUT_PATH = "outputs/e_baselines/genot_w2_unified_model.pkl"

rng_np = np.random.default_rng(SEED)

# data
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


@jax.jit
def prepare_batch(src_expr_padded, tgt_expr_padded,
                  n_src, n_tgt, t_src_norm, t_tgt_norm, rng_key):
    """Sample an OT-coupled (source, target) batch -- fully on device.

    src_expr_padded / tgt_expr_padded have a common leading dimension
    (max_cells) so this function traces exactly once.  Only the first
    n_src / n_tgt rows contain real data; jr.randint respects the bound.
    """
    k_src, k_tgt, k_pair, k_noise, k_interp = jr.split(rng_key, 5)

    # sample minibatch indices (randint handles dynamic maxval without retrace)
    s_idx = jr.randint(k_src, shape=(OT_BATCH_SIZE,), minval=0, maxval=n_src)
    t_idx = jr.randint(k_tgt, shape=(OT_BATCH_SIZE,), minval=0, maxval=n_tgt)

    mb_src_expr = src_expr_padded[s_idx]   # (OT_BATCH_SIZE, expr_dim)
    mb_tgt_expr = tgt_expr_padded[t_idx]   # (OT_BATCH_SIZE, expr_dim)

    # solve Sinkhorn on the minibatch
    geom = pointcloud.PointCloud(mb_src_expr, mb_tgt_expr, epsilon=EPSILON)
    prob = linear_problem.LinearProblem(geom)
    coupling = sinkhorn.Sinkhorn()(prob).matrix

    # sample (source, target) pairs from the coupling
    flat = coupling.ravel()
    flat = flat / flat.sum()
    n_cols = coupling.shape[1]
    indices = jr.choice(k_pair, flat.shape[0], shape=(BATCH_SIZE,), p=flat)
    si = indices // n_cols
    ti = indices % n_cols

    src_batch_expr = mb_src_expr[si]   # (BATCH_SIZE, expr_dim)
    tgt_batch = mb_tgt_expr[ti]         # (BATCH_SIZE, expr_dim)

    # augment source condition with normalised timepoints
    t_src_vec = jnp.full((BATCH_SIZE, 1), t_src_norm)
    t_tgt_vec = jnp.full((BATCH_SIZE, 1), t_tgt_norm)
    cond_batch = jnp.concatenate([src_batch_expr, t_src_vec, t_tgt_vec], axis=1)

    # noise at t=0
    expr_dim = src_expr_padded.shape[1]
    noise = jr.normal(k_noise, shape=(BATCH_SIZE, expr_dim))

    # flow-matching interpolants
    batch = interpolate_samples(
        k_interp, x0=noise, x1=tgt_batch, cond=cond_batch,
    )

    return batch


# ---------------------------------------------------------------------------
# build all transition data (forward and backward)
# ---------------------------------------------------------------------------
log.info("preparing transition data (forward + backward)")

transition_data = {}

for i in range(len(timepoints) - 1):
    t0, t1 = timepoints[i], timepoints[i + 1]

    src_mask = states.obs["time"] == t0
    tgt_mask = states.obs["time"] == t1

    src_cells = states.obs_names[src_mask]
    tgt_cells = states.obs_names[tgt_mask]

    src_expr = np.asarray(states[src_cells].obsm["X_scvi"], dtype=np.float32)
    tgt_expr = np.asarray(states[tgt_cells].obsm["X_scvi"], dtype=np.float32)

    if len(src_expr) == 0 or len(tgt_expr) == 0:
        log.warning("%s <-> %s: empty, skipping", t0, t1)
        continue

    # put expression arrays on device once; keep numpy copies for inference
    src_expr_d = jnp.array(src_expr)
    tgt_expr_d = jnp.array(tgt_expr)

    # forward: t0 -> t1
    transition_data[(t0, t1)] = dict(
        src_expr=src_expr, tgt_expr=tgt_expr,       # numpy (inference)
        src_expr_d=src_expr_d, tgt_expr_d=tgt_expr_d,  # device (training)
        src_cells=np.array(src_cells), tgt_cells=np.array(tgt_cells),
        t_src_norm=normalise_time(t0), t_tgt_norm=normalise_time(t1),
    )
    # backward: t1 -> t0
    transition_data[(t1, t0)] = dict(
        src_expr=tgt_expr, tgt_expr=src_expr,
        src_expr_d=tgt_expr_d, tgt_expr_d=src_expr_d,
        src_cells=np.array(tgt_cells), tgt_cells=np.array(src_cells),
        t_src_norm=normalise_time(t1), t_tgt_norm=normalise_time(t0),
    )

    log.info("  %s -> %s: %d src, %d tgt", t0, t1, len(src_expr), len(tgt_expr))
    log.info("  %s -> %s: %d src, %d tgt (backward)", t1, t0, len(tgt_expr), len(src_expr))

transition_keys = list(transition_data.keys())
n_transitions = len(transition_keys)
log.info("total transitions (fwd + bwd): %d", n_transitions)

expr_dim = next(iter(transition_data.values()))["src_expr"].shape[1]

log.info("padding transition arrays to uniform shape for JIT")

max_cells = max(
    max(td["src_expr_d"].shape[0], td["tgt_expr_d"].shape[0])
    for td in transition_data.values()
)
log.info("  max_cells across all transitions: %d", max_cells)

for key, td in transition_data.items():
    n_src = td["src_expr_d"].shape[0]
    n_tgt = td["tgt_expr_d"].shape[0]

    td["n_src"] = jnp.int32(n_src)
    td["n_tgt"] = jnp.int32(n_tgt)

    td["t_src_norm_j"] = jnp.float32(td["t_src_norm"])
    td["t_tgt_norm_j"] = jnp.float32(td["t_tgt_norm"])

    # pad to (max_cells, expr_dim)
    if n_src < max_cells:
        pad = jnp.zeros((max_cells - n_src, expr_dim), dtype=jnp.float32)
        td["src_expr_padded"] = jnp.concatenate([td["src_expr_d"], pad], axis=0)
    else:
        td["src_expr_padded"] = td["src_expr_d"]

    if n_tgt < max_cells:
        pad = jnp.zeros((max_cells - n_tgt, expr_dim), dtype=jnp.float32)
        td["tgt_expr_padded"] = jnp.concatenate([td["tgt_expr_d"], pad], axis=0)
    else:
        td["tgt_expr_padded"] = td["tgt_expr_d"]

    # free the unpadded device copies to reclaim memory
    del td["src_expr_d"]
    del td["tgt_expr_d"]

log.info("padding done")

# ---------------------------------------------------------------------------
# train unified velocity field
# ---------------------------------------------------------------------------
log.info("training unified velocity field (%d iterations)", N_ITERS)

cond_dim = expr_dim + 2  # [src_expr, t_src_norm, t_tgt_norm]

model = MLP(
    dim=expr_dim,
    cond_dim=cond_dim,
    hidden_dims=HIDDEN_DIMS,
    rngs=nnx.Rngs(SEED),
)
optimizer = nnx.ModelAndOptimizer(model, optax.adam(learning_rate=LR))

master_key = jr.PRNGKey(SEED)

# loss curve, recorded at the logging cadence (avoids a host sync every step)
loss_curve = {"step": [], "loss": [], "grad_norm": []}

for step in range(N_ITERS):
    master_key, k_trans, k_step = jr.split(master_key, 3)

    # pick a random transition
    tr_idx = int(jr.randint(k_trans, (), 0, n_transitions))
    tr_key = transition_keys[tr_idx]
    td = transition_data[tr_key]

    # --- one JIT-compiled call per step ---
    batch = prepare_batch(
        td["src_expr_padded"], td["tgt_expr_padded"],
        td["n_src"], td["n_tgt"],
        td["t_src_norm_j"], td["t_tgt_norm_j"],
        k_step,
    )

    metrics = flow_matching_step(model, optimizer, batch)

    if step % 5 == 0 or step == N_ITERS - 1:
        loss_val = jax.block_until_ready(metrics["loss"])
        grad_val = jax.block_until_ready(metrics["grad_norm"])
        loss_curve["step"].append(step)
        loss_curve["loss"].append(float(loss_val))
        loss_curve["grad_norm"].append(float(grad_val))
        log.info(
            "step %6d/%d  transition %s  loss=%.6f  |grad|=%.4f",
            step, N_ITERS, tr_key,
            float(loss_val), float(grad_val),
        )


# serialise
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
        src_expr=td["src_expr"], tgt_expr=td["tgt_expr"],
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