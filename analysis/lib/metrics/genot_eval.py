from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
from ott.neural.methods.flow_matching import evaluate_velocity_field

import flax.nnx as nnx
from ott.neural.networks.velocity_field.mlp import MLP


from analysis.lib.metrics.marginal_preservation import (
    _cells_at_time,
    _get_time_points,
    _subsample,
    _subsample_array,
    _wasserstein2,
)


def _load_genot_model(pkl_path, seed=0):
    """Load a saved GENOT model from a .pkl file."""

    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)

    cfg = saved["model_config"]
    flat_np = saved["flat_params"]
    t_min = saved["t_min"]
    t_max = saved["t_max"]
    t_range = t_max - t_min if t_max != t_min else 1.0

    mdl = MLP(
        dim=cfg["dim"],
        cond_dim=cfg["cond_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        rngs=nnx.Rngs(seed),
    )
    graphdef, state = nnx.split(mdl)
    _, tree_def_local = jax.tree.flatten(state)
    restored_state = jax.tree.unflatten(
        tree_def_local, [jnp.array(x) for x in flat_np]
    )
    mdl = nnx.merge(graphdef, restored_state)
    return mdl, cfg, t_min, t_range


def _genot_transport(model, source_expr, t_src, t_tgt, t_min, t_range, dim, seed=0):
    """Transport source expression points through the GENOT velocity field."""

    def _norm(t):
        return (float(t) - t_min) / t_range

    src = jnp.array(source_expr)
    n = src.shape[0]
    cond = jnp.concatenate([
        src,
        jnp.full((n, 1), _norm(t_src)),
        jnp.full((n, 1), _norm(t_tgt)),
    ], axis=1)

    x0 = jr.normal(jr.PRNGKey(seed), shape=(n, dim))

    def _integrate_one(x_i, cond_i):
        sol = evaluate_velocity_field(model, x_i, cond=cond_i, t0=0.0, t1=1.0)
        ys = sol.ys
        if isinstance(ys, dict):
            ys = ys["x_t"]
        return jnp.reshape(ys, (-1,))

    return np.asarray(jax.vmap(_integrate_one)(x0, cond))


def marginal_preservation_genot(
    pkl_path,
    trees,
    leaf_states,
    *,
    n_cells=200,
    time_tol=0.25,
    seed=212121,
):
    """W2 between GENOT-transported distribution and true marginal at each earlier time-point."""
    rng = np.random.default_rng(seed)
    model, cfg, t_min, t_range = _load_genot_model(pkl_path, seed=seed)
    dim = cfg["dim"]

    leaf_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in leaf_states.items()}

    time_points = _get_time_points(trees, tol=time_tol)
    if len(time_points) < 2:
        return pd.DataFrame(columns=["t_src", "t_tgt", "n_src", "n_tgt", "w2"])

    rows = []
    for i in tqdm(range(len(time_points) - 1), desc="marginal preservation (genot)"):
        t_early, t_late = time_points[i], time_points[i + 1]

        names_early, states_early = _cells_at_time(trees, leaf_np, t_early, tol=time_tol)
        names_late, states_late = _cells_at_time(trees, leaf_np, t_late, tol=time_tol)
        if len(names_early) < 2 or len(names_late) < 2:
            continue

        src_names, src_states = _subsample(rng, names_late, states_late, n_cells)
        tgt_states = _subsample_array(rng, states_early, n_cells)

        transported = _genot_transport(
            model, src_states,
            t_src=t_late, t_tgt=t_early,
            t_min=t_min, t_range=t_range, dim=dim, seed=seed,
        )

        rows.append({
            "t_src": t_late, "t_tgt": t_early,
            "n_src": len(src_names), "n_tgt": len(tgt_states),
            "w2": _wasserstein2(transported, tgt_states),
        })

    return pd.DataFrame(rows)


def interpolant_matching_genot(
    pkl_path,
    trees,
    leaf_states,
    interpolant_time,
    ground_truth_interpolant_states,
    *,
    n_cells=200,
    time_tol=0.25,
    seed=212121,
):
    """W2 between GENOT-transported distribution and ground-truth states at interpolant_time.

    Only time-points later than interpolant_time are evaluated.
    """
    rng = np.random.default_rng(seed)
    model, cfg, t_min, t_range = _load_genot_model(pkl_path, seed=seed)
    dim = cfg["dim"]

    leaf_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in leaf_states.items()}

    time_points = _get_time_points(trees, tol=time_tol)
    gt = np.asarray(ground_truth_interpolant_states)
    rows = []

    for t in tqdm(time_points, desc="interpolant matching (genot)"):
        if t <= interpolant_time + time_tol:
            continue

        names_t, states_t = _cells_at_time(trees, leaf_np, t, tol=time_tol)
        if len(names_t) < 2 or len(gt) < 2:
            continue

        src_names, src_states = _subsample(rng, names_t, states_t, n_cells)
        tgt_states = _subsample_array(rng, gt, n_cells)

        transported = _genot_transport(
            model, src_states,
            t_src=t, t_tgt=interpolant_time,
            t_min=t_min, t_range=t_range, dim=dim, seed=seed,
        )

        rows.append({
            "t_src": t, "t_tgt": interpolant_time,
            "n_src": len(src_names), "n_tgt": len(tgt_states),
            "w2": _wasserstein2(transported, tgt_states),
        })

    return pd.DataFrame(rows)