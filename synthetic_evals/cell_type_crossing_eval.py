"""Crossing-strength eval for the real-cell CellTypeCrossingDataset (parallel).

Self-contained analogue of ``synthetic_evals.core.crossing_eval`` for the real
mouse-embryo motif. For each imposed coupling strength ``alpha`` it loads our
trained pool/split model, trains the model-agnostic OT baselines (multimarginal
flow matching + GENOT-Sinkhorn) on the same marginals, runs forward (splitter)
and reverse (pooler) inference on a held-out draw, and reports the forward /
reverse energy-distance error of each method. Sweeping ``alpha`` draws the
"our model recovers uniformly; OT only near the OT coupling" curves.

GENOT-FGW is wired in too: with the depth, each clone has >=2 cells a side
(2 mids at t_src, K leaves at t_tgt), so the Gromov-Wasserstein solve is
non-degenerate. We build its ``FGWBank`` from the dataset's real trees directly
(``crossing_eval`` regenerates synthetic trees from seeds, which we cannot
reuse) -- random tree pairings mean it cannot see the true coupling, so like
Sinkhorn it should track the OT coupling and fail as alpha rises.

Why a parallel module: ``crossing_eval`` is hard-wired to ``make_hierarchical_ou``
(held-out sim) and regenerates trees from seeds (FGW). This driver builds the
held-out subtrees directly from a fresh ``CellTypeCrossingDataset`` instead.

Run (on a GPU node, after training):
    PYTHONPATH=. python synthetic_evals/cell_type_crossing_eval.py
    PYTHONPATH=. python synthetic_evals/cell_type_crossing_eval.py --quick   # smoke
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from datasets.cell_type_crossing import CellTypeCrossingDataset, load_cell_pools
from synthetic_evals.baselines import genot_fgw, genot_sinkhorn, marginal_fm, registry
from synthetic_evals.core import inference as inf
from synthetic_evals.core.hydra_io import DEVICE, ROOT, load_pool_split
from synthetic_evals.core.subtrees import subtrees_to_pairs
from synthetic_evals.viz import energy_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("ctc_eval")

EXPERIMENT = "cell_type_crossing/unshared_vpsde_idnn_leaf"
TIMES = (8.5, 9.0, 9.5)               # t_root, t_src, t_tgt
T_SRC, T_TGT = TIMES[-2], TIMES[-1]   # evaluated (crossing) transition 9.0->9.5
DATA_DIR = str(ROOT.parent / "data" / "embryos")
ROOT_TYPE = "cranial neural crest (anterior)"             # hidden common ancestor
PARENTS = ("frontonasal mesenchyme", "hindbrain (r1)")    # mids at t_src
CHILDREN = (("frontonasal mesenchyme", "pharyngeal arch mesenchyme (p1-p2)"),
            ("hindbrain (r5-r8)", "hindbrain (r1)"))       # children at t_tgt
EVAL_SEED = 90001


# --------------------------------------------------------------------------
#  Held-out eval subtrees (built directly from a fresh dataset)
# --------------------------------------------------------------------------


def build_eval_subtrees(ds_eval) -> list[dict]:
    """Per-parent subtrees in the format ``inference`` consumes, group-labelled.

    Mirrors ``synthetic_evals.core.subtrees.build_layer_subtrees`` output: each
    dict has ``graph``/``root``/``leaves`` (list of (name, orig))/``x_src``/
    ``leaf_states``/``parent``, plus ``group`` (0=P_A, 1=P_B).
    """
    subs = []
    for tname, g in ds_eval.records["leaf"].trees.items():
        for grp, mtag in ((0, "mA"), (1, "mB")):
            mid = f"{tname}_{mtag}"                 # hidden t_{-2} ancestor
            kids = list(g.successors(mid))
            sg = g.subgraph([mid] + kids).copy()
            subs.append({
                "graph": sg,
                "root": mid,
                "leaves": [(c, c) for c in kids],
                "x_src": ds_eval.gt_node_states[mid].numpy(),
                "leaf_states": {c: ds_eval.gt_node_states[c] for c in kids},
                "parent": mid,
                "group": grp,
            })
    return subs


def _fgw_bank_from_dataset(ds):
    """Build a GENOT-FGW ``FGWBank`` from the dataset's real deep trees.

    One clone per leaf tree; its t_src cells are the (gt) mids, its t_tgt cells
    the observed leaves, with tree distances from the real tree. FGW pairs clones
    randomly, so it never sees the true mid->leaf correspondence.
    """
    by_time = {ds.t_src: {"expr": [], "cells": [], "clone_ids": []},
               ds.t_tgt: {"expr": [], "cells": [], "clone_ids": []}}
    oracles = {}
    for tname, g in ds.records["leaf"].trees.items():
        oracles[tname] = genot_fgw.TreeDistanceOracle(g)
        for n in g.nodes:
            t = g.nodes[n]["time"]
            if t not in by_time:
                continue
            by_time[t]["expr"].append(ds.gt_node_states[n].numpy())
            by_time[t]["cells"].append(n)
            by_time[t]["clone_ids"].append(tname)
    for t in by_time:
        by_time[t]["expr"] = np.stack(by_time[t]["expr"]).astype(np.float32)
        by_time[t]["cells"] = np.array(by_time[t]["cells"])
        by_time[t]["clone_ids"] = np.array(by_time[t]["clone_ids"])
    return genot_fgw.FGWBank(by_time=by_time, oracles=oracles)


def _grouped_energy(pred_by_row, true_rows, groups, cap=2000):
    """Mean over groups of energy_distance(pooled pred, pooled true).

    ``pred_by_row`` is (S, R, d) predictions for R rows; ``true_rows`` is (R, d);
    ``groups`` is (R,) the group label of each row. Pools all samples + rows of a
    group into one cloud and compares to the true cloud of that group. Clouds are
    subsampled to ``cap`` points first -- energy distance is O(n^2), so the full
    pooled clouds (~10^4 points) would otherwise OOM the pairwise matrix.
    """
    rng = np.random.default_rng(0)

    def sub(x):
        return x if len(x) <= cap else x[rng.choice(len(x), cap, replace=False)]

    errs = []
    for grp in (0, 1):
        m = groups == grp
        if not m.any():
            continue
        pred = pred_by_row[:, m, :].reshape(-1, pred_by_row.shape[-1])
        errs.append(energy_distance(sub(pred), sub(true_rows[m])))
    return float(np.mean(errs)) if errs else float("nan")


def _marginal_energy(true_rows, groups, cap=2000):
    """Energy error of the marginal baseline: "sample from the correct marginal".

    For each group, compares that group's true cloud to a random draw from the
    POOLED target marginal (all groups mixed). This is the no-lineage-information
    ceiling: it reproduces the overall target population but is blind to which
    group a cell belongs to, so any method that actually uses the coupling must
    beat it. Same subsampling (``cap``) as ``_grouped_energy`` so the two values
    are directly comparable (and can normalise one another).
    """
    rng = np.random.default_rng(0)

    def sub(x):
        return x if len(x) <= cap else x[rng.choice(len(x), cap, replace=False)]

    errs = []
    for grp in (0, 1):
        m = groups == grp
        if not m.any():
            continue
        errs.append(energy_distance(sub(true_rows), sub(true_rows[m])))
    return float(np.mean(errs)) if errs else float("nan")


# --------------------------------------------------------------------------
#  Per-alpha evaluation
# --------------------------------------------------------------------------


def eval_one_alpha(alpha, pools, *, n_eval_trees, fm_steps, genot_steps,
                   n_samples, n_flow_steps):
    """Forward + reverse energy error for our model, FM, GENOT-Sinkhorn and GENOT-FGW."""
    # Training-distribution dataset (for baselines + reference marginals).
    train_ds = CellTypeCrossingDataset(
        data_dir=DATA_DIR, times=TIMES, root_type=ROOT_TYPE,
        parent_types=PARENTS, child_types=CHILDREN, alpha=alpha, seed=0,
        cell_pools=pools, n_trees=600, n_marginal=600,
    )
    dim = int(train_ds.x_dim)
    pairs_fwd = [(T_SRC, T_TGT)]
    pairs_rev = [(T_TGT, T_SRC)]

    # ---- baselines (trained fresh on the marginals), both directions ----
    fgw_bank = _fgw_bank_from_dataset(train_ds)
    banks = {
        "fm": lambda: marginal_fm.TimeBank.from_dataset(train_ds),
        "gs": lambda: genot_sinkhorn.GenotBank.from_dataset(train_ds),
        "gf": lambda: fgw_bank,
    }
    train_kwargs = {
        "fm": dict(n_steps=fm_steps, device=DEVICE),
        "gs": dict(n_steps=genot_steps, coupling_mode="precomputed"),
        "gf": dict(n_steps=genot_steps, coupling_mode="precomputed",
                   n_tree_pairings=1),
    }
    seeds = {"fm": (0, 1), "gs": (0, 1), "gf": (2, 3)}
    # only the baselines configured above (the registry also holds wot / moslin).
    methods = [n for n in registry.names() if n in train_kwargs]
    models = {}
    for name in methods:
        bl = registry.get(name)
        logger.info("[alpha=%.2f] training %s", alpha, bl.label)
        for direction, prs, sd in (("fwd", pairs_fwd, seeds[name][0]),
                                   ("rev", pairs_rev, seeds[name][1])):
            models[name, direction] = bl.train(
                banks[name](), dim=dim, pairs=prs, seed=sd, **train_kwargs[name])

    # ---- our trained model ----
    pool, splitter, _ = load_pool_split(
        EXPERIMENT, None, [f"dataset.alpha={alpha}"])

    # ---- held-out eval draw ----
    ds_eval = CellTypeCrossingDataset(
        data_dir=DATA_DIR, times=TIMES, root_type=ROOT_TYPE,
        parent_types=PARENTS, child_types=CHILDREN, alpha=alpha, seed=EVAL_SEED,
        cell_pools=pools, n_trees=n_eval_trees, n_marginal=32,
    )
    subs = build_eval_subtrees(ds_eval)
    pairs = subtrees_to_pairs(subs, T_SRC, T_TGT)
    x_tgt = np.stack([p[5] for p in pairs]).astype(np.float32)            # (L, d)
    leaf_group = np.array([subs[i]["group"]
                           for i, s in enumerate(subs) for _ in s["leaves"]])
    parent_group = np.array([s["group"] for s in subs])
    parent_true = np.stack([s["x_src"] for s in subs]).astype(np.float32)  # (P, d)

    out = {"alpha": alpha, "n_parents": len(subs), "n_leaves": len(pairs)}

    # ---- FORWARD: parent -> children ----
    ours_fwd = inf.splitter_predict_subtrees(
        splitter, subs, n_samples=n_samples, n_flow_steps=n_flow_steps)   # (S,L,d)
    x_src = np.stack([p[4] for p in pairs]).astype(np.float32)

    out["fwd"] = {"ours": _grouped_energy(ours_fwd, x_tgt, leaf_group)}
    for name in methods:
        pred = inf.forward_samples(
            registry.get(name), models[name, "fwd"], x_src, T_SRC, T_TGT,
            n_samples=n_samples, n_flow_steps=n_flow_steps)
        out["fwd"][name] = _grouped_energy(pred, x_tgt, leaf_group)

    # ---- REVERSE: children -> ancestral parent ----
    ours_rev = inf.pool_predict_subtrees(
        pool, subs, n_samples=n_samples, n_flow_steps=n_flow_steps)["parent_pred"]

    out["rev"] = {"ours": _grouped_energy(ours_rev, parent_true, parent_group)}
    for name in methods:
        pred = inf.reverse_per_parent(
            registry.get(name), models[name, "rev"], subs, pairs, T_SRC, T_TGT,
            n_samples=n_samples, n_flow_steps=n_flow_steps)["parent_pred"]
        out["rev"][name] = _grouped_energy(pred, parent_true, parent_group)
    logger.info("[alpha=%.2f] fwd=%s rev=%s", alpha, out["fwd"], out["rev"])
    return out


# --------------------------------------------------------------------------
#  Driver + plot
# --------------------------------------------------------------------------


def plot_curves(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = [r["alpha"] for r in results]
    methods = [("ours", "#1b7837", "our model (lineage-aware)"),
               ("fm", "#d95f02", "multimarginal FM"),
               ("gs", "#7570b3", "GENOT-Sinkhorn"),
               ("gf", "#e7298a", "GENOT-FGW")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, key, title in ((axes[0], "fwd", "Forward (split: parent -> children)"),
                           (axes[1], "rev", "Reverse (pool: children -> ancestor)")):
        for m, c, lbl in methods:
            ax.plot(alphas, [r[key][m] for r in results], "-o", color=c, label=lbl)
        ax.set_xlabel("crossing strength  alpha")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("energy-distance error")
    fig.suptitle("Real-cell crossing: our model recovers uniformly; OT error rises with alpha")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    logger.info("saved figure -> %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--n-eval-trees", type=int, default=400)
    ap.add_argument("--fm-steps", type=int, default=10000)
    ap.add_argument("--genot-steps", type=int, default=10000)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--n-flow-steps", type=int, default=20)
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke: few steps/trees (for a CPU sanity run)")
    args = ap.parse_args()
    if args.quick:
        args.fm_steps = args.genot_steps = 40
        args.n_eval_trees = 16
        args.n_samples = 2
        args.n_flow_steps = 4

    pools = load_cell_pools(
        DATA_DIR, "qiu24_integrated_hvgs_and_scvi.h5ad", "X_scvi",
        {TIMES[0]: [ROOT_TYPE], T_SRC: list(PARENTS),
         T_TGT: [c for g in CHILDREN for c in g]})

    results = [
        eval_one_alpha(
            a, pools, n_eval_trees=args.n_eval_trees,
            fm_steps=args.fm_steps, genot_steps=args.genot_steps,
            n_samples=args.n_samples, n_flow_steps=args.n_flow_steps,
        )
        for a in args.alphas
    ]

    out_dir = ROOT / "synthetic_evals" / "cell_type_crossing"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    plot_curves(results, out_dir / "alpha_curves.png")
    logger.info("done: %s", out_dir)


if __name__ == "__main__":
    main()
