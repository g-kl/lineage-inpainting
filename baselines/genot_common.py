"""Shared post-fit utilities for the GENOT coupling baselines.

Both baselines (Sinkhorn / fused-Gromov-Wasserstein) fit a single
timepoint-conditioned velocity field and then, *after* training, derive the
same three artifacts from it:

  * ``direct_ancestor``   -- ``{(t0, t1): {target_cell: source_cell}}`` obtained
    by transporting every source cell forward through the learned ODE and
    snapping each target cell to its nearest transported source (a coupling
    between adjacent timepoints).
  * ``ancestral_mapping`` -- ``{cell: {t: ancestor_cell}}`` obtained by chaining
    the direct-ancestor maps back through time.
  * ``assignments``       -- ``{tree_name: {internal_node: observed_cell}}`` (the
    pooling coupler's per-node format) obtained by chaining one observed leaf of
    each node back through a backward-transport parent map over real cells, then
    snapping the last hop within +-1 time bin, as the pooling coupler does.
"""
from __future__ import annotations

import logging

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.nnx as nnx
import networkx as nx

from ott.neural.methods.flow_matching import evaluate_velocity_field
from ott.neural.networks.velocity_field.mlp import MLP

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# model (de)serialisation
# ---------------------------------------------------------------------------
def serialize_mlp(model, expr_dim, cond_dim, hidden_dims):
    """Return ``(flat_numpy_params, model_config)`` ready for pickling."""
    _, param_state = nnx.split(model)
    flat_params, _ = jax.tree.flatten(param_state)
    flat_numpy = [np.asarray(x) for x in flat_params]
    model_config = {
        "dim": expr_dim,
        "cond_dim": cond_dim,
        "hidden_dims": hidden_dims,
    }
    return flat_numpy, model_config


def load_mlp(model_config, flat_params, seed=0):
    """Rebuild an :class:`MLP` from a saved config + flat parameter list."""
    cfg = model_config
    mdl = MLP(
        dim=cfg["dim"],
        cond_dim=cfg["cond_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        rngs=nnx.Rngs(seed),
    )
    graphdef, state = nnx.split(mdl)
    _, tree_def = jax.tree.flatten(state)
    restored = jax.tree.unflatten(tree_def, [jnp.array(x) for x in flat_params])
    return nnx.merge(graphdef, restored)


# ---------------------------------------------------------------------------
# inference transport
# ---------------------------------------------------------------------------
def make_transport_fn():
    """Build a JIT-compiled, vmap'd ODE integrator for one noise sample."""

    @nnx.jit
    def _batch(model, x0, cond):
        def _integrate_one(x_i, cond_i):
            sol = evaluate_velocity_field(model, x_i, cond=cond_i, t0=0.0, t1=1.0)
            ys = sol.ys
            if isinstance(ys, dict):
                ys = ys["x_t"]
            return jnp.reshape(ys, (-1,))
        return jax.vmap(_integrate_one)(x0, cond)

    return _batch


def transport_cells(model, src_expr, t_src_norm, t_tgt_norm, key, *,
                    expr_dim, n_samples=1, transport_fn=None):
    """Transport source expression points through the learned velocity field.

    ``t_src_norm`` / ``t_tgt_norm`` are already normalised to ``[0, 1]``.
    Returns an array of shape ``(n_cells, n_samples, expr_dim)``.
    """
    if transport_fn is None:
        transport_fn = make_transport_fn()
    src = jnp.array(src_expr)
    n = src.shape[0]
    cond = jnp.concatenate(
        [src, jnp.full((n, 1), t_src_norm), jnp.full((n, 1), t_tgt_norm)],
        axis=1,
    )
    all_samples = []
    for _ in range(n_samples):
        key, k_noise = jr.split(key)
        x0 = jr.normal(k_noise, shape=(n, expr_dim))
        all_samples.append(np.asarray(transport_fn(model, x0, cond)))
    return np.stack(all_samples, axis=1)


# ---------------------------------------------------------------------------
# ancestor maps (forward transport + nearest neighbour)
# ---------------------------------------------------------------------------
def build_direct_ancestor(model, forward_transitions, timepoints, *,
                          expr_dim, normalise_time, key, n_samples=5):
    """Forward-transport coupling between every adjacent timepoint pair.

    Parameters
    ----------
    forward_transitions : dict
        Maps ``(t0, t1)`` -> ``dict(src_expr, tgt_expr, src_cells, tgt_cells)``
        with numpy arrays, for the *forward* direction only.  Adjacent pairs
        absent from this mapping are emitted as an empty dict.
    timepoints : list of float
        Sorted grid timepoints.
    normalise_time : callable
        Maps a raw timepoint to ``[0, 1]``.

    Returns
    -------
    dict
        ``{(t0, t1): {target_cell: source_cell}}`` for every adjacent pair.
    """
    from sklearn.neighbors import NearestNeighbors

    transport_fn = make_transport_fn()
    direct_ancestor = {}

    for i in range(len(timepoints) - 1):
        t0, t1 = timepoints[i], timepoints[i + 1]
        fwd = forward_transitions.get((t0, t1))
        if fwd is None:
            direct_ancestor[(t0, t1)] = {}
            continue

        key, sample_key = jr.split(key)
        log.info("  transporting %d cells for %s -> %s",
                 fwd["src_expr"].shape[0], t0, t1)
        transported = transport_cells(
            model, fwd["src_expr"], normalise_time(t0), normalise_time(t1),
            sample_key, expr_dim=expr_dim, n_samples=n_samples,
            transport_fn=transport_fn,
        )
        transported_mean = transported.mean(axis=1)
        if transported_mean.ndim > 2:
            transported_mean = transported_mean.reshape(
                transported_mean.shape[0], -1
            )

        nn_model = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn_model.fit(transported_mean)
        _, indices = nn_model.kneighbors(fwd["tgt_expr"])

        src_cells = fwd["src_cells"]
        pair_ancestors = {}
        for j, tgt_cell in enumerate(fwd["tgt_cells"]):
            pair_ancestors[tgt_cell] = src_cells[indices[j, 0]]

        direct_ancestor[(t0, t1)] = pair_ancestors
        log.info("  %s -> %s: mapped %d target cells to ancestors",
                 t0, t1, len(pair_ancestors))

    return direct_ancestor


def chain_ancestors(direct_ancestor, states, timepoints, *, time_key="time"):
    """Chain adjacent direct-ancestor maps into a per-cell ancestral mapping.

    Returns ``{cell: {t: ancestor_cell}}`` keyed by every observed cell.
    """
    obs_names = states.obs_names
    times = states.obs[time_key]
    ancestral_mapping = {name: {} for name in obs_names}
    for i in range(1, len(timepoints)):
        cells = obs_names[times == timepoints[i]]
        for cell in cells:
            ancestor = cell
            for j in range(i - 1, -1, -1):
                key = (timepoints[j], timepoints[j + 1])
                dmap = direct_ancestor.get(key, {})
                if ancestor not in dmap:
                    break
                ancestor = dmap[ancestor]
                ancestral_mapping[cell][timepoints[j]] = ancestor
    return ancestral_mapping


# ---------------------------------------------------------------------------
# per-node assignments by backward chain through real cells
# ---------------------------------------------------------------------------
def backward_parent_map(model, states, *, t_min, t_max, key, n_samples=5,
                        chunk=65536):
    """one backward hop for every observed cell.

    for a cell y at bins[i], i >= 1, ``m[y]`` is the mean of ``n_samples`` draws
    from bins[i] back to bins[i - 1], and ``parent[y]`` is the row of the cell at
    exactly bins[i - 1] nearest ``m[y]``. rows at bins[0] keep parent -1.
    """
    from analysis.lib.coupling.nearest_neighbor_coupling import _nearest_neighbor

    x = np.asarray(states.obsm["X_scvi"], dtype=np.float32)
    bins, tb = np.unique(states.obs["time"].astype(float).values,
                         return_inverse=True)
    transport_fn = make_transport_fn()
    m = np.zeros_like(x)
    parent = np.full(len(x), -1, dtype=np.int64)
    for i in range(1, len(bins)):
        src, tgt = np.where(tb == i)[0], np.where(tb == i - 1)[0]
        t_src = float((bins[i] - t_min) / (t_max - t_min))
        t_tgt = float((bins[i - 1] - t_min) / (t_max - t_min))
        for s in range(0, len(src), chunk):          # E9.5 alone is 1.1M cells
            key, sample_key = jr.split(key)
            rows = src[s:s + chunk]
            m[rows] = transport_cells(
                model, x[rows], t_src, t_tgt, sample_key,
                expr_dim=x.shape[1], n_samples=n_samples,
                transport_fn=transport_fn,
            ).mean(axis=1)
        nn, _ = _nearest_neighbor(m[src], x[tgt], use_gpu=True)
        parent[src] = tgt[nn]
        log.info("  parent map %s -> %s: %d cells", bins[i], bins[i - 1], len(src))
    return parent, m


def build_assignments(trees, states, parent, m, *, seed=0, window=1,
                      min_snap_time=7.0):
    """``{tree_name: {internal_node: observed_cell}}`` from a backward parent map.

    T is a tree's leaf bin. an internal node at time >= ``min_snap_time`` takes
    its nearest bin b and is kept if b < T and it has an observed leaf below it.
    it draws one such leaf l uniformly, follows ``parent`` from l to the cell c
    one bin after b, and is assigned the cell nearest ``m[c]`` within ``window``
    bins of b. window=1 is the pooling coupler's snap rule (T included); window=0
    gives the chained ancestor of l at b.
    """
    from analysis.lib.coupling.nearest_neighbor_coupling import _nearest_neighbor

    names = states.obs_names.values
    x = np.asarray(states.obsm["X_scvi"], dtype=np.float32)
    bins, tb = np.unique(states.obs["time"].astype(float).values,
                         return_inverse=True)
    row = {n: i for i, n in enumerate(names)}

    # a dfs postorder puts each node's observed leaves in one run order[lo:hi]
    order, keys, node_bin, leaf_bin, lo, hi = [], [], [], [], [], []
    for tree_name, g in trees.items():
        leaf_time = max(g.nodes[n]["time"] for n in g if g.out_degree(n) == 0)
        T = int(np.abs(bins - leaf_time).argmin())
        n0, first = len(order), {}
        for root in [n for n in g if g.in_degree(n) == 0]:
            for n in nx.dfs_postorder_nodes(g, root):
                ch = list(g.successors(n))
                first[n] = min(first[c] for c in ch) if ch else len(order)
                if not ch:
                    if n in row:
                        order.append(row[n])
                    continue
                tau = g.nodes[n]["time"]
                b = int(np.abs(bins - tau).argmin())
                if tau >= min_snap_time and b < T and len(order) > first[n]:
                    keys.append((tree_name, n))
                    node_bin.append(b)
                    leaf_bin.append(T)
                    lo.append(first[n])
                    hi.append(len(order))
        assert (tb[order[n0:]] == T).all(), f"{tree_name}: observed leaf off the leaf bin"
    order, node_bin, leaf_bin = map(np.asarray, (order, node_bin, leaf_bin))

    leaf = order[np.random.default_rng(seed).integers(lo, hi)]
    hops = leaf_bin - node_bin
    cur = leaf.copy()
    for h in range(1, hops.max()):                   # hops - 1 steps: cur ends one bin after b
        cur[hops > h] = parent[cur[hops > h]]

    cell = np.empty(len(cur), dtype=np.int64)
    for k in np.unique(node_bin):
        q = np.where(node_bin == k)[0]
        cand = np.where(np.abs(tb - k) <= window)[0]
        nn, _ = _nearest_neighbor(m[cur[q]], x[cand], use_gpu=True)
        cell[q] = cand[nn]
        log.info("  bin %s: %d nodes", bins[k], len(q))

    assignments = {tree_name: {} for tree_name in trees}
    for (tree_name, n), name in zip(keys, names[cell]):
        assignments[tree_name][n] = name
    log.info("assignments: %d internal nodes assigned across %d trees",
             len(keys), len(trees))
    return assignments
