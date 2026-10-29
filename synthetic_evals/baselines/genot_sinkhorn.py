"""GENOT + Sinkhorn baseline for the crossing synthetic-eval.

Adapts ``baselines/genot_sinkhorn.py`` to the same train/predict shape as
``synthetic_evals.baselines.marginal_fm`` so the crossing driver can wire
it in alongside the marginal-FM baseline.

Two coupling modes (mirroring ``genot_fgw``):

* ``coupling_mode="precomputed"`` (default): for each (t_src, t_tgt)
  transition, solve ONE global Sinkhorn coupling over the full source/target
  time slice up front (subsampled to ``max_cells`` a side to bound the dense
  solve), then presample ``n_presamples`` (src, tgt) index pairs. Train steps
  become a cheap pool lookup. This is the global OT coupling, not a per-step
  minibatch approximation, and matches the FGW baseline's precompute regime so
  the two baselines train under the same coupling assumptions.

* ``coupling_mode="minibatch"``: each step draws a fresh
  Sinkhorn minibatch of source/target cells, solves entropic OT on it, and
  samples pairs from that coupling. Cheap because the linear solve is JIT'd,
  but it is a minibatch-OT approximation refreshed every step.

Train step (either mode):
  1. sample a transition (t_src, t_tgt),
  2. obtain a batch of OT-coupled (src, tgt) pairs,
  3. flow-matching gradient step conditioned on (src, t_src, t_tgt).

Inference draws random initial noise and integrates the learned ODE.
Unlike marginal-FM this is stochastic; the caller draws N samples by varying
``seed``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.nnx as nnx
import optax

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from ott.neural.methods.flow_matching import (
    flow_matching_step,
    interpolate_samples,
    evaluate_velocity_field,
)
from ott.neural.networks.velocity_field.mlp import MLP

logger = logging.getLogger(__name__)


@dataclass
class GenotBank:
    """Per-time pools of cell expression vectors (numpy, float32)."""

    by_time: dict[float, np.ndarray]

    @classmethod
    def from_dataset(cls, dataset) -> "GenotBank":
        # ``train_states`` when the dataset rebinds ``states`` (see
        # marginal_fm.TimeBank.from_dataset), else ``states``.
        states = getattr(dataset, "train_states", dataset.states)
        X = np.asarray(states.X, dtype=np.float32)
        times = np.asarray(states.obs["time"], dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        return cls(by_time={
            float(t): X[times == t] for t in np.unique(times)
        })

    def times(self) -> list[float]:
        return sorted(self.by_time.keys())


def adjacent_pairs(times):
    times = sorted(times)
    return [(times[i], times[i + 1]) for i in range(len(times) - 1)]


def _subsample(n: int, cap: int, rng) -> np.ndarray:
    """Indices into an n-row pool, subsampled (sorted) to at most ``cap``."""
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=cap, replace=False)).astype(np.int64)


def _precompute_global_coupling(
    bank: GenotBank,
    pairs: list[tuple[float, float]],
    *,
    epsilon: float,
    max_cells: int,
    n_presamples: int,
    seed: int,
) -> dict[tuple[float, float], dict | None]:
    """Solve ONE global Sinkhorn coupling per transition, then presample pairs.

    Sinkhorn is a linear OT in expression space with no tree structure, so
    each transition has a single coupling over the full time slices
    (subsampled to ``max_cells`` a side to bound the dense solve). From each
    coupling we draw ``n_presamples`` (src, tgt) index pairs so the training
    loop is a cheap pool lookup. Mirrors ``genot_fgw._precompute_tree_pair_couplings``.
    Returns {(t_src, t_tgt): coupling-dict or None if degenerate}.
    """
    rng = np.random.default_rng(seed)
    out: dict[tuple[float, float], dict | None] = {}
    for t_src, t_tgt in pairs:
        src = bank.by_time[float(t_src)]
        tgt = bank.by_time[float(t_tgt)]
        if src.shape[0] < 2 or tgt.shape[0] < 2:
            out[(t_src, t_tgt)] = None
            continue
        s_idx = _subsample(src.shape[0], max_cells, rng)
        t_idx = _subsample(tgt.shape[0], max_cells, rng)
        geom = pointcloud.PointCloud(
            jnp.asarray(src[s_idx]), jnp.asarray(tgt[t_idx]), epsilon=epsilon,
        )
        prob = linear_problem.LinearProblem(geom)
        coupling = np.asarray(sinkhorn.Sinkhorn()(prob).matrix)
        # numpy.choice is strict: p must be non-negative and sum to 1. Clip
        # tiny negatives from the solver, then renormalise (same as genot_fgw).
        flat = coupling.ravel().astype(np.float64)
        flat = np.clip(flat, 0.0, None)
        total = flat.sum()
        if total <= 0 or not np.isfinite(total):
            logger.warning(
                "[genot-sinkhorn] degenerate coupling for (%s,%s)", t_src, t_tgt,
            )
            out[(t_src, t_tgt)] = None
            continue
        flat /= total
        n_cols = coupling.shape[1]
        pre = rng.choice(len(flat), size=n_presamples, replace=True, p=flat)
        out[(t_src, t_tgt)] = {
            "src_global_idx": s_idx,
            "tgt_global_idx": t_idx,
            "pre_si": (pre // n_cols).astype(np.int64),
            "pre_ti": (pre % n_cols).astype(np.int64),
        }
        logger.info(
            "[genot-sinkhorn] precomputed coupling for (%s,%s): %dx%d cells",
            t_src, t_tgt, len(s_idx), len(t_idx),
        )
    return out


def train(
    bank: GenotBank,
    *,
    dim: int,
    pairs: list[tuple[float, float]] | None = None,
    n_steps: int = 10_000,
    coupling_mode: str = "precomputed",
    ot_batch_size: int = 256,
    batch_size: int = 256,
    n_presamples: int = 100_000,
    max_cells: int = 4096,
    hidden_dims: tuple = (256, 256, 256),
    epsilon: float = 5e-2,
    lr: float = 1e-4,
    seed: int = 0,
    log_every: int = 500,
) -> dict:
    """Train a GENOT-Sinkhorn velocity field.

    Returns an opaque package dict (model + normalisation constants)
    consumable by :func:`predict`. See the module docstring for the two
    ``coupling_mode`` regimes.
    """
    if coupling_mode not in ("precomputed", "minibatch"):
        raise ValueError(
            f"coupling_mode must be 'precomputed' or 'minibatch', got {coupling_mode!r}"
        )
    if pairs is None:
        pairs = adjacent_pairs(bank.times())
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Need at least one (t_src, t_tgt) pair to train.")

    all_ts = [t for pair in pairs for t in pair]
    t_min = float(min(all_ts))
    t_max = float(max(all_ts))
    t_range = (t_max - t_min) if t_max != t_min else 1.0

    def _norm(t):
        return (float(t) - t_min) / t_range

    # Pad src/tgt arrays to a common (max_pad, dim) so the JIT'd minibatch
    # solver traces once across all transitions. ``src_expr``/``tgt_expr`` hold
    # the unpadded slice (jnp) for the precomputed-coupling pool lookup.
    max_pad = max(
        max(bank.by_time[float(t_s)].shape[0], bank.by_time[float(t_t)].shape[0])
        for t_s, t_t in pairs
    )

    def _pad(arr, target):
        if arr.shape[0] >= target:
            return arr
        pad = np.zeros((target - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    pair_data = []
    for t_src, t_tgt in pairs:
        src = bank.by_time[float(t_src)]
        tgt = bank.by_time[float(t_tgt)]
        pair_data.append({
            "src_padded": jnp.asarray(_pad(src, max_pad)),
            "tgt_padded": jnp.asarray(_pad(tgt, max_pad)),
            "n_src": jnp.int32(src.shape[0]),
            "n_tgt": jnp.int32(tgt.shape[0]),
            "src_expr": jnp.asarray(src),
            "tgt_expr": jnp.asarray(tgt),
            "t_src_norm": jnp.float32(_norm(t_src)),
            "t_tgt_norm": jnp.float32(_norm(t_tgt)),
        })

    precomputed: dict[tuple[float, float], dict | None] | None = None
    if coupling_mode == "precomputed":
        logger.info(
            "[genot-sinkhorn] precomputing global couplings (max_cells=%d, "
            "n_presamples=%d)", max_cells, n_presamples,
        )
        precomputed = _precompute_global_coupling(
            bank, pairs, epsilon=epsilon, max_cells=max_cells,
            n_presamples=n_presamples, seed=seed,
        )

    cond_dim = dim + 2
    model = MLP(
        dim=dim,
        cond_dim=cond_dim,
        hidden_dims=tuple(hidden_dims),
        rngs=nnx.Rngs(seed),
    )
    optimizer = nnx.ModelAndOptimizer(model, optax.adam(learning_rate=lr))

    @nnx.jit
    def _train_step(model, optimizer, src_batch, tgt_batch,
                    t_s, t_t, k_noise, k_interp):
        """Interpolate a chosen (src, tgt) batch and take one flow-matching
        step -- interp + optimizer update fused into a single jitted call, as
        in genot_fgw._train_step."""
        bs = src_batch.shape[0]
        cond = jnp.concatenate([
            src_batch,
            jnp.full((bs, 1), t_s),
            jnp.full((bs, 1), t_t),
        ], axis=1)
        noise = jr.normal(k_noise, shape=tgt_batch.shape)
        batch = interpolate_samples(k_interp, x0=noise, x1=tgt_batch, cond=cond)
        return flow_matching_step(model, optimizer, batch)

    @jax.jit
    def _minibatch_pairs(src_padded, tgt_padded, n_src, n_tgt, key):
        """Minibatch mode: draw a Sinkhorn minibatch, solve OT, and return the
        sampled (src, tgt) pairs (the heavy per-step entropic-OT solve)."""
        k_src, k_tgt, k_pair = jr.split(key, 3)
        s_idx = jr.randint(k_src, (ot_batch_size,), 0, n_src)
        t_idx = jr.randint(k_tgt, (ot_batch_size,), 0, n_tgt)
        mb_src = src_padded[s_idx]
        mb_tgt = tgt_padded[t_idx]

        geom = pointcloud.PointCloud(mb_src, mb_tgt, epsilon=epsilon)
        prob = linear_problem.LinearProblem(geom)
        coupling = sinkhorn.Sinkhorn()(prob).matrix

        flat = coupling.ravel()
        flat = flat / flat.sum()
        n_cols = coupling.shape[1]
        indices = jr.choice(k_pair, flat.shape[0], shape=(batch_size,), p=flat)
        si = indices // n_cols
        ti = indices % n_cols
        return mb_src[si], mb_tgt[ti]

    master_key = jr.PRNGKey(seed)
    n_pairs = len(pair_data)
    rng_np = np.random.default_rng(seed)  # transition + presample pool draws
    skipped = 0
    for step in range(n_steps):
        master_key, k_mb, k_noise, k_interp = jr.split(master_key, 4)
        idx = int(rng_np.integers(n_pairs))  # CPU draw -- no device sync
        pd = pair_data[idx]

        if coupling_mode == "minibatch":
            src_batch, tgt_batch = _minibatch_pairs(
                pd["src_padded"], pd["tgt_padded"], pd["n_src"], pd["n_tgt"], k_mb,
            )
        else:  # precomputed: look up presampled (src, tgt) pairs.
            cp = precomputed[pairs[idx]]
            if cp is None:
                skipped += 1
                continue
            take = rng_np.integers(0, cp["pre_si"].shape[0], size=batch_size)
            src_g = cp["src_global_idx"][cp["pre_si"][take]]
            tgt_g = cp["tgt_global_idx"][cp["pre_ti"][take]]
            src_batch = pd["src_expr"][jnp.asarray(src_g)]
            tgt_batch = pd["tgt_expr"][jnp.asarray(tgt_g)]

        metrics = _train_step(
            model, optimizer, src_batch, tgt_batch,
            pd["t_src_norm"], pd["t_tgt_norm"], k_noise, k_interp,
        )

        if step % log_every == 0 or step == n_steps - 1:
            loss = float(jax.block_until_ready(metrics["loss"]))
            logger.info(
                "[genot-sinkhorn] step %5d/%d loss=%.5f%s",
                step, n_steps, loss,
                f"  (skipped={skipped})" if skipped else "",
            )

    return {
        "model": model,
        "dim": dim,
        "cond_dim": cond_dim,
        "t_min": t_min,
        "t_range": t_range,
    }


def predict(
    pkg: dict,
    x_src,
    t_src: float,
    t_tgt: float,
    *,
    n_steps: int | None = None,
    seed: int = 0,
):
    """Transport ``x_src`` from t_src to t_tgt via the learned velocity field.

    Accepts torch tensors or numpy arrays (returns the same type).
    ``n_steps`` is accepted for API symmetry with marginal_fm.predict
    but ignored -- the underlying solver uses its own internal stepping.
    """
    del n_steps
    model = pkg["model"]
    dim = pkg["dim"]
    t_min = pkg["t_min"]
    t_range = pkg["t_range"]

    is_torch = isinstance(x_src, torch.Tensor)
    if is_torch:
        device = x_src.device
        x_np = x_src.detach().cpu().numpy()
    else:
        x_np = np.asarray(x_src)
    if x_np.ndim == 1:
        x_np = x_np[:, None]

    src = jnp.asarray(x_np, dtype=jnp.float32)
    M = src.shape[0]
    t_s = jnp.full((M, 1), (float(t_src) - t_min) / t_range)
    t_t = jnp.full((M, 1), (float(t_tgt) - t_min) / t_range)
    cond = jnp.concatenate([src, t_s, t_t], axis=1)

    key = jr.PRNGKey(seed)
    x0 = jr.normal(key, shape=(M, dim))

    def _one(x_i, cond_i):
        sol = evaluate_velocity_field(model, x_i, cond=cond_i, t0=0.0, t1=1.0)
        ys = sol.ys
        if isinstance(ys, dict):
            ys = ys["x_t"]
        return jnp.reshape(ys, (-1,))

    out = np.asarray(jax.vmap(_one)(x0, cond)).astype(np.float32)
    if is_torch:
        return torch.from_numpy(out).to(device)
    return out
