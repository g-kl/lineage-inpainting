"""GENOT + FGW baseline for the crossing synthetic-eval.

Adapts ``baselines/genot_fgw_mm.py`` to a train/predict shape matching
``synthetic_evals.baselines.marginal_fm``. Tree distances come from the
synthetic lineage trees rebuilt from the dataset's seeds.

For the multi-seed dataset (``n_trees > 1``), each tree i is regenerated
by calling ``make_hierarchical_ou`` with seed ``seed_base + i`` and the
same SDE/branching parameters as the dataset. Tree topology is
independent of obs_times, so we pass a placeholder obs_time during the
rebuild. The dataset's per-cell prefix ``"t{i}_"`` is reattached so cell
names match across the dataset and the rebuilt trees.

Two coupling modes:

* ``coupling_mode="precomputed"`` (default): for each (t_src, t_tgt)
  transition, randomly pair src trees with tgt trees (``n_tree_pairings``
  one-to-one shuffles) and solve one FGW problem per tree pair up
  front. Train steps just sample (src, tgt) indices from pre-drawn pools.
  Much cheaper when iters >> tree-pair count (the common case);
  also strictly *within-tree* tree distances, no cross-tree max-cost
  hack.

* ``coupling_mode="minibatch"``: the baselines/genot_fgw_mm.py scheme:
  each step samples a fresh cross-tree minibatch (cross-tree
  distance forced to ``2 * max_within``) and runs an FGW solve. Useful
  when datasets are too large to precompute, or as a sanity check.

Train step (either mode):
  1. sample a transition (t_src, t_tgt),
  2. obtain a batch of OT-coupled (src, tgt) pairs,
  3. flow-matching gradient step conditioned on (src, t_src, t_tgt).
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.nnx as nnx
import optax

from ott.geometry import geometry, pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein

from ott.neural.methods.flow_matching import (
    flow_matching_step,
    interpolate_samples,
)
from ott.neural.networks.velocity_field.mlp import MLP

from datasets.synthetic import make_hierarchical_ou
from synthetic_evals.baselines.genot_sinkhorn import predict as _gs_predict

logger = logging.getLogger(__name__)

_PREFIX_RE = re.compile(r"^(t\d+)_")


class TreeDistanceOracle:
    """O(n log n) build, O(1) per pairwise distance query via Euler-tour RMQ.

    Same algorithm as baselines/genot_fgw_mm.py; duplicated to keep this
    module self-contained.

    dist(u, v) = time(u) + time(v) - 2 * time(LCA(u, v))
    """

    def __init__(self, tree):
        nodes = list(tree.nodes)
        n = len(nodes)
        self.node_to_int = {v: i for i, v in enumerate(nodes)}
        self.node_time = np.empty(n, dtype=np.float64)
        for v in nodes:
            self.node_time[self.node_to_int[v]] = float(tree.nodes[v]["time"])

        roots = [v for v in nodes if tree.in_degree(v) == 0]
        if len(roots) != 1:
            raise ValueError(
                f"Expected exactly 1 root (in-degree 0), found {len(roots)}"
            )
        root_int = self.node_to_int[roots[0]]

        euler: list[int] = []
        first = np.full(n, -1, dtype=np.int32)
        stack = [(root_int, iter(tree.successors(roots[0])))]
        euler.append(root_int)
        first[root_int] = 0

        while stack:
            _u, children = stack[-1]
            advanced = False
            for child_orig in children:
                c_int = self.node_to_int[child_orig]
                euler.append(c_int)
                if first[c_int] == -1:
                    first[c_int] = len(euler) - 1
                stack.append((c_int, iter(tree.successors(child_orig))))
                advanced = True
                break
            if not advanced:
                stack.pop()
                if stack:
                    euler.append(stack[-1][0])

        self.euler = np.array(euler, dtype=np.int32)
        self.first = first
        self._tour_time = self.node_time[self.euler]

        m = len(euler)
        K = max(int(np.floor(np.log2(m))), 0) + 1
        table = np.zeros((K, m), dtype=np.int32)
        table[0] = np.arange(m, dtype=np.int32)
        for k in range(1, K):
            half = 1 << (k - 1)
            limit = m - (1 << k) + 1
            left = table[k - 1, :limit]
            right = table[k - 1, half:limit + half]
            pick_right = self._tour_time[right] < self._tour_time[left]
            table[k, :limit] = np.where(pick_right, right, left)
        self._table = table

        self._log2 = np.zeros(m + 1, dtype=np.int32)
        for i in range(2, m + 1):
            self._log2[i] = self._log2[i // 2] + 1

    def _rmq(self, lo, hi):
        length = hi - lo + 1
        k = self._log2[length]
        left = self._table[k, lo]
        right = self._table[k, hi - (1 << k) + 1]
        pick = self._tour_time[right] < self._tour_time[left]
        return np.where(pick, right, left)

    def batch_distances(self, cells):
        idx = np.array([self.node_to_int[c] for c in cells])
        times = self.node_time[idx]
        pos = self.first[idx]
        lo = np.minimum(pos[:, None], pos[None, :])
        hi = np.maximum(pos[:, None], pos[None, :])
        lca_tour_idx = self._rmq(lo.ravel(), hi.ravel())
        lca_node = self.euler[lca_tour_idx]
        lca_time = self.node_time[lca_node].reshape(lo.shape)
        return (times[:, None] + times[None, :] - 2.0 * lca_time).astype(np.float32)


def _parse_tree_id(cell_name: str) -> str:
    m = _PREFIX_RE.match(cell_name)
    if m is None:
        raise ValueError(
            f"cell name {cell_name!r} does not have the multi-seed prefix "
            "'t<i>_'; genot_fgw currently requires n_trees > 1"
        )
    return m.group(1)


def _sim_params_from_cfg(ds_cfg):
    a2 = ds_cfg.get("attractors_2", None)
    if a2 is not None:
        a2 = {float(k): tuple(float(x) for x in v) for k, v in dict(a2).items()}
    a1 = ds_cfg.get("attractors_1", None)
    if a1 is not None:
        a1 = tuple(float(x) for x in a1)
    extra = {}
    if a1 is not None:
        extra["attractors_1"] = a1
    if a2 is not None:
        extra["attractors_2"] = a2
    return dict(
        n_generations=int(ds_cfg.n_generations),
        split_gen=int(ds_cfg.split_gen),
        gap=int(ds_cfg.gap),
        sigma=float(ds_cfg.sigma),
        theta=float(ds_cfg.theta),
        branch_interval=float(ds_cfg.branch_interval),
        steps_per_interval=int(ds_cfg.steps_per_interval),
        **extra,
    )


def _build_oracles(ds_cfg) -> dict[str, TreeDistanceOracle]:
    n_trees = int(ds_cfg.get("n_trees", 1))
    if n_trees <= 1:
        raise ValueError(
            "genot_fgw currently requires n_trees > 1 (multi-seed dataset)"
        )
    seed_base = int(ds_cfg.seed)
    sim_kwargs = _sim_params_from_cfg(ds_cfg)
    placeholder = np.array([float(list(ds_cfg.obs_times)[0])])

    oracles: dict[str, TreeDistanceOracle] = {}
    for i in range(n_trees):
        sim_res = make_hierarchical_ou(
            seed=seed_base + i, obs_times=placeholder, **sim_kwargs,
        )
        prefix = f"t{i}_"
        tree = nx.relabel_nodes(
            sim_res.tree, lambda n, p=prefix: p + n, copy=True,
        )
        oracles[f"t{i}"] = TreeDistanceOracle(tree)
        logger.info("[genot-fgw] built oracle %s (%d nodes)", f"t{i}", len(tree.nodes))
    return oracles


@dataclass
class FGWBank:
    """Per-time cells with per-cell tree-id and per-tree distance oracles."""

    by_time: dict[float, dict]  # {expr (N,d), cells (N,), clone_ids (N,)}
    oracles: dict[str, TreeDistanceOracle]

    @classmethod
    def from_dataset(cls, dataset, ds_cfg) -> "FGWBank":
        cells_all = np.asarray(dataset.states.obs_names)
        times = np.asarray(dataset.states.obs["time"], dtype=np.float64)
        X = np.asarray(dataset.states.X, dtype=np.float32)
        if X.ndim == 1:
            X = X[:, None]

        oracles = _build_oracles(ds_cfg)
        # Filter to cells whose tree was rebuilt and that exist in the oracle.
        valid: dict[str, set] = {tid: set(o.node_to_int.keys())
                                 for tid, o in oracles.items()}

        by_time: dict[float, dict] = {}
        for t in np.unique(times):
            mask = times == t
            cells = cells_all[mask]
            clone_ids = np.array([_parse_tree_id(c) for c in cells])
            # Drop cells not present in their tree's oracle (shouldn't happen
            # for well-formed datasets, but be defensive).
            keep = np.array([
                cid in oracles and cell in valid[cid]
                for cell, cid in zip(cells, clone_ids)
            ])
            by_time[float(t)] = dict(
                expr=X[mask][keep],
                cells=cells[keep],
                clone_ids=clone_ids[keep],
            )
        return cls(by_time=by_time, oracles=oracles)

    def times(self) -> list[float]:
        return sorted(self.by_time.keys())


def adjacent_pairs(times):
    times = sorted(times)
    return [(times[i], times[i + 1]) for i in range(len(times) - 1)]


def _minibatch_tree_distances(cells, clone_ids, oracles):
    """Block-diagonal tree-distance matrix; cross-tree pairs get a large cost."""
    n = len(cells)
    D = np.zeros((n, n), dtype=np.float32)
    max_within = 0.0
    for clone in np.unique(clone_ids):
        mask = np.where(clone_ids == clone)[0]
        if len(mask) < 2:
            continue
        sub = oracles[clone].batch_distances(cells[mask])
        D[np.ix_(mask, mask)] = sub
        m = float(sub.max())
        if m > max_within:
            max_within = m
    cross_dist = max_within * 2.0 if max_within > 0 else 1.0
    cross_mask = clone_ids[:, None] != clone_ids[None, :]
    D[cross_mask] = cross_dist
    return D


@functools.partial(jax.jit, static_argnames=("epsilon", "fused_penalty"))
def _solve_fgw_jit(expr_src, expr_tgt, tree_src, tree_tgt, *, epsilon, fused_penalty):
    geom_xx = geometry.Geometry(cost_matrix=tree_src, epsilon=epsilon)
    geom_yy = geometry.Geometry(cost_matrix=tree_tgt, epsilon=epsilon)
    geom_xy = pointcloud.PointCloud(expr_src, expr_tgt, epsilon=epsilon)
    prob = quadratic_problem.QuadraticProblem(
        geom_xx, geom_yy, geom_xy, fused_penalty=fused_penalty,
    )
    solver = gromov_wasserstein.GromovWasserstein(
        sinkhorn.Sinkhorn(), epsilon=epsilon,
    )
    return solver(prob).matrix


def _solve_fgw(expr_src, expr_tgt, tree_src, tree_tgt,
               *, epsilon: float, fused_penalty: float):
    """One FGW coupling. The solve is a cached ``jax.jit``: every tree pair here
    has the same shape (2 mids x K leaves), so it compiles once and each further
    call is a fast dispatch -- fresh solver/geometry objects per call would
    otherwise re-trace (~600 ms) every time, which dominated the precompute.
    """
    matrix = _solve_fgw_jit(
        jnp.asarray(expr_src), jnp.asarray(expr_tgt),
        jnp.asarray(tree_src), jnp.asarray(tree_tgt),
        epsilon=epsilon, fused_penalty=fused_penalty,
    )
    return np.asarray(matrix)


def _precompute_tree_pair_couplings(
    bank: FGWBank,
    pairs: list[tuple[float, float]],
    *,
    n_tree_pairings: int,
    epsilon: float,
    fused_penalty: float,
    n_presamples: int,
    seed: int,
) -> dict[tuple[float, float], list[dict]]:
    """For each transition, pair src trees with tgt trees and run FGW once.

    For each transition (t_src, t_tgt) we repeat ``n_tree_pairings`` random
    one-to-one shuffles between the src clone set and the tgt clone set.
    Each shuffle yields one FGW coupling per src clone -> matched tgt
    clone, computed on strictly within-tree distances (no cross-tree
    max-cost hack). From each coupling we draw ``n_presamples`` (src, tgt)
    indices so the training loop becomes a cheap pool lookup.
    """
    rng = np.random.default_rng(seed)
    out: dict[tuple[float, float], list[dict]] = {}

    for t_src, t_tgt in pairs:
        sd = bank.by_time[float(t_src)]
        td = bank.by_time[float(t_tgt)]
        src_clones_unique = list(np.unique(sd["clone_ids"]))
        tgt_clones_unique = list(np.unique(td["clone_ids"]))
        if not src_clones_unique or not tgt_clones_unique:
            out[(t_src, t_tgt)] = []
            continue

        src_groups = {c: np.where(sd["clone_ids"] == c)[0] for c in src_clones_unique}
        tgt_groups = {c: np.where(td["clone_ids"] == c)[0] for c in tgt_clones_unique}

        couplings: list[dict] = []
        for _ in range(n_tree_pairings):
            shuffled = rng.permutation(tgt_clones_unique)
            for i, sc in enumerate(src_clones_unique):
                tc = shuffled[i % len(shuffled)]
                s_idx = src_groups[sc]
                t_idx = tgt_groups[tc]
                if len(s_idx) < 2 or len(t_idx) < 2:
                    continue
                src_expr = sd["expr"][s_idx]
                tgt_expr = td["expr"][t_idx]
                src_tree_d = bank.oracles[sc].batch_distances(sd["cells"][s_idx])
                tgt_tree_d = bank.oracles[tc].batch_distances(td["cells"][t_idx])
                coupling = _solve_fgw(
                    src_expr, tgt_expr, src_tree_d, tgt_tree_d,
                    epsilon=epsilon, fused_penalty=fused_penalty,
                )
                total = float(coupling.sum())
                if total <= 0 or not np.isfinite(total):
                    logger.warning(
                        "[genot-fgw] degenerate coupling for (%s,%s) tree pair (%s,%s)",
                        t_src, t_tgt, sc, tc,
                    )
                    continue
                n_tgt_pair = coupling.shape[1]
                # numpy.choice is strict: p must sum to 1 exactly. Cast to
                # float64 first, clip tiny negatives from the solver, then
                # renormalise so floating-point sum hits 1.0.
                flat = coupling.ravel().astype(np.float64)
                flat = np.clip(flat, 0.0, None)
                flat /= flat.sum()
                pre = rng.choice(len(flat), size=n_presamples, replace=True, p=flat)
                couplings.append({
                    "src_global_idx": s_idx.astype(np.int64),
                    "tgt_global_idx": t_idx.astype(np.int64),
                    "pre_si": (pre // n_tgt_pair).astype(np.int64),
                    "pre_ti": (pre % n_tgt_pair).astype(np.int64),
                })
        out[(t_src, t_tgt)] = couplings
        logger.info(
            "[genot-fgw] precomputed %d tree-pair couplings for transition (%s,%s)",
            len(couplings), t_src, t_tgt,
        )
    return out


def train(
    bank: FGWBank,
    *,
    dim: int,
    pairs: list[tuple[float, float]] | None = None,
    n_steps: int = 10_000,
    coupling_mode: str = "precomputed",
    n_tree_pairings: int = 1,
    n_presamples: int = 10_000,
    fgw_batch_size: int = 256,
    batch_size: int = 256,
    hidden_dims: tuple = (256, 256, 256),
    epsilon: float = 5e-2,
    fused_penalty: float = 1.0,
    lr: float = 1e-4,
    seed: int = 0,
    log_every: int = 500,
) -> dict:
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

    pair_data = []
    for t_src, t_tgt in pairs:
        sd = bank.by_time[float(t_src)]
        td = bank.by_time[float(t_tgt)]
        pair_data.append({
            "src_expr_np": sd["expr"],
            "tgt_expr_np": td["expr"],
            "src_expr": jnp.asarray(sd["expr"]),
            "tgt_expr": jnp.asarray(td["expr"]),
            "src_cells": sd["cells"],
            "tgt_cells": td["cells"],
            "src_clones": sd["clone_ids"],
            "tgt_clones": td["clone_ids"],
            "t_src_norm": jnp.float32(_norm(t_src)),
            "t_tgt_norm": jnp.float32(_norm(t_tgt)),
        })

    precomputed: dict[tuple[float, float], list[dict]] | None = None
    if coupling_mode == "precomputed":
        logger.info(
            "[genot-fgw] precomputing tree-pair couplings (n_tree_pairings=%d, "
            "n_presamples=%d)", n_tree_pairings, n_presamples,
        )
        precomputed = _precompute_tree_pair_couplings(
            bank, pairs,
            n_tree_pairings=n_tree_pairings,
            epsilon=epsilon, fused_penalty=fused_penalty,
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
        bs = src_batch.shape[0]
        cond = jnp.concatenate([
            src_batch,
            jnp.full((bs, 1), t_s),
            jnp.full((bs, 1), t_t),
        ], axis=1)
        noise = jr.normal(k_noise, shape=tgt_batch.shape)
        batch = interpolate_samples(k_interp, x0=noise, x1=tgt_batch, cond=cond)
        return flow_matching_step(model, optimizer, batch)

    master_key = jr.PRNGKey(seed)
    n_pairs = len(pair_data)
    rng_np = np.random.default_rng(seed)
    skipped = 0

    for step in range(n_steps):
        master_key, k_pair, k_noise, k_interp = jr.split(master_key, 4)
        idx = int(rng_np.integers(n_pairs))
        pd = pair_data[idx]

        if coupling_mode == "precomputed":
            cp_list = precomputed[pairs[idx]]
            if not cp_list:
                skipped += 1
                continue
            cp = cp_list[int(rng_np.integers(len(cp_list)))]
            take = rng_np.integers(0, cp["pre_si"].shape[0], size=batch_size)
            si = cp["pre_si"][take]
            ti = cp["pre_ti"][take]
            src_g = cp["src_global_idx"][si]
            tgt_g = cp["tgt_global_idx"][ti]
            src_batch = pd["src_expr"][jnp.asarray(src_g)]
            tgt_batch = pd["tgt_expr"][jnp.asarray(tgt_g)]
        else:  # minibatch mode
            n_src = pd["src_expr_np"].shape[0]
            n_tgt = pd["tgt_expr_np"].shape[0]
            mb_src = min(fgw_batch_size, n_src)
            mb_tgt = min(fgw_batch_size, n_tgt)
            s_idx = rng_np.choice(n_src, size=mb_src, replace=False)
            t_idx = rng_np.choice(n_tgt, size=mb_tgt, replace=False)

            mb_src_expr_np = pd["src_expr_np"][s_idx]
            mb_tgt_expr_np = pd["tgt_expr_np"][t_idx]
            mb_src_tree = _minibatch_tree_distances(
                pd["src_cells"][s_idx], pd["src_clones"][s_idx], bank.oracles,
            )
            mb_tgt_tree = _minibatch_tree_distances(
                pd["tgt_cells"][t_idx], pd["tgt_clones"][t_idx], bank.oracles,
            )
            coupling = _solve_fgw(
                mb_src_expr_np, mb_tgt_expr_np, mb_src_tree, mb_tgt_tree,
                epsilon=epsilon, fused_penalty=fused_penalty,
            )
            total = float(coupling.sum())
            if total <= 0 or not np.isfinite(total):
                logger.warning("[genot-fgw] degenerate coupling at step %d, skipping", step)
                skipped += 1
                continue
            flat = (coupling / total).ravel()
            indices = np.asarray(jr.choice(
                k_pair, len(flat), shape=(batch_size,), p=jnp.array(flat),
            ))
            n_tgt_mb = coupling.shape[1]
            si = indices // n_tgt_mb
            ti = indices % n_tgt_mb
            src_batch = pd["src_expr"][jnp.asarray(s_idx[si])]
            tgt_batch = pd["tgt_expr"][jnp.asarray(t_idx[ti])]

        metrics = _train_step(
            model, optimizer, src_batch, tgt_batch,
            pd["t_src_norm"], pd["t_tgt_norm"],
            k_noise, k_interp,
        )

        if step % log_every == 0 or step == n_steps - 1:
            loss = float(jax.block_until_ready(metrics["loss"]))
            logger.info(
                "[genot-fgw] step %5d/%d loss=%.5f%s",
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


def predict(pkg, x_src, t_src, t_tgt, *, n_steps=None, seed=0):
    """Same package layout as genot_sinkhorn; delegate to its predict."""
    return _gs_predict(pkg, x_src, t_src, t_tgt, n_steps=n_steps, seed=seed)
