"""GENOT-FGW adapter for the fate-bias context-scaling benchmark.

The crossing-flavoured ``synthetic_evals.baselines.genot_fgw`` rebuilds
tree topology by calling ``make_hierarchical_ou``. Fate-bias datasets need
``make_fate_bias`` instead with a different parameter set, so we provide
a small adapter that constructs the per-tree distance oracles from a
``FateBiasDataset`` and re-uses the crossing module's ``train``/``predict``
entry points (those operate purely on the assembled ``FGWBank``).
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from datasets.synthetic import make_fate_bias
from synthetic_evals.baselines.genot_fgw import (
    FGWBank,
    TreeDistanceOracle,
    _parse_tree_id,
    train,
    predict,
)

logger = logging.getLogger(__name__)


def _sim_params_from_cfg_fb(ds_cfg) -> dict:
    return dict(
        n_generations=int(ds_cfg.n_generations),
        split_gen=int(ds_cfg.split_gen),
        n_primed_generations=int(ds_cfg.n_primed_generations),
        sigma=float(ds_cfg.sigma),
        theta=float(ds_cfg.theta),
        mu_1_primed=float(ds_cfg.mu_1_primed),
        mu_2_primed=float(ds_cfg.mu_2_primed),
        mu_1=float(ds_cfg.mu_1),
        mu_2=float(ds_cfg.mu_2),
        fate_prob=float(ds_cfg.fate_prob),
        branch_interval=float(ds_cfg.branch_interval),
        steps_per_interval=int(ds_cfg.steps_per_interval),
    )


def _build_oracles_fb(ds_cfg) -> dict[str, TreeDistanceOracle]:
    n_trees = int(ds_cfg.get("n_trees", 1))
    if n_trees <= 1:
        raise ValueError(
            "genot_fgw_fb requires n_trees > 1 (multi-seed fate-bias dataset)"
        )
    seed_base = int(ds_cfg.seed)
    sim_kw = _sim_params_from_cfg_fb(ds_cfg)
    placeholder = np.array([float(list(ds_cfg.obs_times)[0])])

    oracles: dict[str, TreeDistanceOracle] = {}
    for i in range(n_trees):
        sim_res = make_fate_bias(
            seed=seed_base + i, obs_times=placeholder, **sim_kw,
        )
        prefix = f"t{i}_"
        tree = nx.relabel_nodes(
            sim_res.tree, lambda n, p=prefix: p + n, copy=True,
        )
        oracles[f"t{i}"] = TreeDistanceOracle(tree)
    logger.info("[genot-fgw-fb] built %d tree oracles", len(oracles))
    return oracles


def bank_from_dataset(dataset, ds_cfg) -> FGWBank:
    """Build an FGWBank for a multi-seed fate-bias dataset.

    FateBiasDataset.states is the *reference-bank* AnnData when ``ref_seed``
    is set (the default), and its cell names lack the ``t<i>_`` prefix that
    FGW needs to look up tree-distance oracles. We instead pull cell states
    from ``dataset.leaf_states`` and time labels from ``dataset.records``,
    matching the layout that ``_build_leaf_anndata`` produces.
    """
    leaf_time: dict[str, float] = {}
    for rec in dataset.records.values():
        for subs in rec.subtree_index.values():
            for st in subs:
                g = st["graph"]
                for n in g.nodes:
                    if g.out_degree(n) == 0:
                        leaf_time[n] = float(rec.leaf_time)

    cells_all: list[str] = []
    X_rows: list[np.ndarray] = []
    times_list: list[float] = []
    for cell, state in dataset.leaf_states.items():
        if cell not in leaf_time:
            continue
        arr = np.asarray(state.numpy() if hasattr(state, "numpy") else state,
                         dtype=np.float32)
        if arr.ndim == 0:
            arr = arr[None]
        cells_all.append(cell)
        X_rows.append(arr)
        times_list.append(leaf_time[cell])
    cells_all_arr = np.asarray(cells_all)
    X = np.stack(X_rows, axis=0).astype(np.float32) if X_rows else np.zeros(
        (0, 1), dtype=np.float32,
    )
    if X.ndim == 1:
        X = X[:, None]
    times = np.asarray(times_list, dtype=np.float64)

    oracles = _build_oracles_fb(ds_cfg)
    valid = {tid: set(o.node_to_int.keys()) for tid, o in oracles.items()}

    by_time: dict[float, dict] = {}
    for t in np.unique(times):
        mask = times == t
        cells = cells_all_arr[mask]
        clone_ids = np.array([_parse_tree_id(c) for c in cells])
        keep = np.array([
            cid in oracles and cell in valid[cid]
            for cell, cid in zip(cells, clone_ids)
        ])
        by_time[float(t)] = dict(
            expr=X[mask][keep],
            cells=cells[keep],
            clone_ids=clone_ids[keep],
        )
    return FGWBank(by_time=by_time, oracles=oracles)


__all__ = ["bank_from_dataset", "train", "predict"]
