"""Fate-bias context-scaling benchmark eval.

A different benchmark from crossing: per-sigma, it evaluates per-cell energy
distance and fate accuracy across clone sizes [1,2,4,8] on held-out simulation
seeds, selects a cell-of-interest for the illustrative panel, and collects
cartoon trajectories.

Structurally symmetric with crossing, though: one ``--experiment`` (the model)
is evaluated across the sigma *dataset* axis (``fate_bias_sig0{2,4,6,8}``). Each
model is loaded by config hash (no stem/glob), and sigma is read from
``cfg.dataset.sigma``. Model variant (e.g. bridge DTFM) is just the experiment
you point at; the splitter is exercised through ``split_layer_batch``, which
every splitter implements.
"""

from __future__ import annotations

import functools
import logging
import math
import pickle

import hydra
import networkx as nx
import numpy as np
import torch

from synthetic_evals import viz

from datasets.synthetic import make_fate_bias
from datasets.tree_record import collate_subtrees
from synthetic_evals.baselines import (
    genot_fgw_fb,
    genot_sinkhorn,
    marginal_fm,
    moscot_ot,
    registry,
)
from synthetic_evals.baselines.cache import train_or_load
from synthetic_evals.context.case_lineage import record_clone_lineage
from synthetic_evals.core import inference as inf
from synthetic_evals.core.hydra_io import (
    DEVICE,
    ROOT,
    load_pool_split,
    run_pool_split_multirun,
)
from synthetic_evals.core.inference import accepts_n_steps

logger = logging.getLogger("synth_eval.context")

CLONE_SIZES = [1, 2, 4, 8]
TIME_INPUT = 9.0
LEAF_TIME = 10.0
CARTOON_SIGMA = 0.4
CASE_SIGMA = 0.4


# --------------------------------------------------------------------------
#  Clone discovery & splitter inference
# --------------------------------------------------------------------------


def extend_forward(g, parent, parent_t, leaf_t):
    """Append a synthetic perfect-binary extension from parent to leaf_t."""
    if parent_t >= leaf_t - 1e-9:
        return
    child_t = parent_t + 1
    for ci in range(2):
        child = f"{parent}|c{ci}@{int(round(child_t))}"
        g.add_node(child, time=float(child_t))
        g.add_edge(parent, child)
        extend_forward(g, child, child_t, leaf_t)


def build_subtree(clone, snap_input, t_in, t_out):
    """Real topology from clone ancestor down to t_in plus synthetic
    perfect-binary extension from each anchor at t_in down to t_out."""
    g = nx.DiGraph()
    desc = nx.descendants(snap_input, clone["ancestor"]) | {clone["ancestor"]}
    src = snap_input.subgraph(desc)
    for node, data in src.nodes(data=True):
        g.add_node(node, time=data.get("time", 0.0))
    for u, v in src.edges():
        g.add_edge(u, v)
    for a in clone["anchors"]:
        extend_forward(g, a, float(t_in), float(t_out))
    return g


def find_clones(snap_input, leaves_input, snap_target, leaves_target,
                clone_size, t_in, t_out):
    """Yield clones of size k sharing an ancestor at t_in - log2(k)."""
    k = int(math.log2(clone_size))
    root_t = t_in - k
    n_expected = clone_size * (2 ** int(round(t_out - t_in)))

    for n in snap_input.nodes():
        if abs(snap_input.nodes[n].get("time", -1e9) - root_t) > 1e-3:
            continue
        anchors = [
            m for m in (nx.descendants(snap_input, n) | {n})
            if snap_input.out_degree(m) == 0
        ]
        if len(anchors) != clone_size:
            continue
        if n not in snap_target.nodes():
            continue
        leaves_t = [
            m for m in nx.descendants(snap_target, n)
            if snap_target.out_degree(m) == 0
        ]
        if len(leaves_t) != n_expected:
            continue
        yield {
            "ancestor": n,
            "anchors": anchors,
            "anchor_states": np.stack([leaves_input[m] for m in anchors], axis=0),
            "t_out_descendants": leaves_t,
            "t_out_states": np.stack([leaves_target[m] for m in leaves_t], axis=0),
        }


def predict_clone(splitter, clone, snap_input, t_in, t_out, device, *, n_steps=None):
    """One masked pass: full subtree spans (t_in - log2(k), t_out); anchors at
    t_in carry their observed states. ``split_layer_batch`` fills everything
    above and below the anchors (works for any splitter). ``n_steps`` sets the
    bridge SDE discretization (ignored by deterministic/one-step splitters)."""
    g = build_subtree(clone, snap_input, t_in, t_out)
    subtree = {"graph": g, "root": clone["ancestor"]}
    leaf_dummy = {clone["anchors"][0]: torch.zeros(1, dtype=torch.float32)}
    batch = collate_subtrees([subtree], leaf_states=leaf_dummy, max_nodes=4096).to(device)

    anchor_mask = (torch.abs(batch.node_times - float(t_in)) < 1e-3) & batch.mask
    b, n = batch.mask.shape
    dim = splitter.dim if hasattr(splitter, "dim") else 1
    anchor_states = torch.zeros(b, n, dim, device=device, dtype=torch.float32)
    name_to_state = dict(zip(clone["anchors"], clone["anchor_states"]))
    for j, name in enumerate(batch.node_names[0]):
        if anchor_mask[0, j].item():
            anchor_states[0, j, 0] = float(name_to_state[name][0])

    with torch.no_grad():
        kw = ({"n_steps": n_steps} if n_steps is not None
              and accepts_n_steps(splitter.split_layer_batch) else {})
        result = splitter.split_layer_batch(batch, anchor_states, anchor_mask, **kw).cpu()
    return result, batch


def _anchor_descendant_states(clone, result, batch, snap_target, leaves_target, t_out):
    """For every anchor, return (true_states, pred_states) at t_out."""
    names = list(batch.node_names[0])
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    result_arr = result[0].numpy()
    if result_arr.ndim == 1:
        result_arr = result_arr[:, None]
    synth_graph = batch.subtree_records[0]["graph"]

    out = []
    for anchor in clone["anchors"]:
        synth_t = [
            m for m in nx.descendants(synth_graph, anchor)
            if abs(synth_graph.nodes[m].get("time", -1) - t_out) < 1e-3
            and synth_graph.out_degree(m) == 0
        ]
        pred_states = np.stack([result_arr[name_to_idx[m]] for m in synth_t], axis=0)
        real_t = [
            m for m in nx.descendants(snap_target, anchor)
            if snap_target.out_degree(m) == 0
        ]
        real_states = np.stack([leaves_target[m] for m in real_t], axis=0)
        if real_states.ndim == 1:
            real_states = real_states[:, None]
        if pred_states.ndim == 1:
            pred_states = pred_states[:, None]
        out.append({
            "anchor": anchor,
            "true_descendants": real_t,
            "true_states": real_states.astype(np.float32),
            "pred_states": pred_states.astype(np.float32),
        })
    return out


# --------------------------------------------------------------------------
#  Baselines (per anchor, no context) -- trained once on (9,10), cached
# --------------------------------------------------------------------------


def build_baselines(train_ds, ds_cfg, bcfg) -> dict:
    """Train-or-load every registered baseline on the (9, 10) transition, forward only."""
    dim = int(train_ds.x_dim)
    pairs = [(TIME_INPUT, LEAF_TIME)]
    fm_steps = int(bcfg.fm_steps)
    genot_steps = int(bcfg.genot_steps)
    force = bool(bcfg.get("force_retrain", False))
    eps = float(bcfg.get("moscot_epsilon", 1e-2))
    alpha = float(bcfg.get("moslin_alpha", 0.5))
    max_cells = int(bcfg.get("moscot_max_cells", 2048))

    @functools.cache
    def fgw_bank():
        return genot_fgw_fb.bank_from_dataset(train_ds, ds_cfg)

    @functools.cache
    def moscot_bank():
        return moscot_ot.MoscotBank.from_fgw_bank(fgw_bank())

    # per-baseline cache-key params, train kwargs and bank builder. gf uses the
    # fate-bias bank; train/predict are shared with genot_fgw.
    specs = {
        "fm": dict(
            gate="fm",
            params={"transition": pairs, "steps": fm_steps, "seed": 0,
                    "bank": "train_leaves"},
            kwargs=dict(n_steps=fm_steps, pairs=pairs, device=DEVICE, seed=0),
            bank=lambda: marginal_fm.TimeBank.from_dataset(train_ds),
        ),
        "gs": dict(
            gate="genot",
            params={"transition": pairs, "steps": genot_steps, "seed": 0,
                    "bank": "train_leaves"},
            kwargs=dict(n_steps=genot_steps, pairs=pairs, seed=0),
            bank=lambda: genot_sinkhorn.GenotBank.from_dataset(train_ds),
        ),
        "gf": dict(
            gate="genot",
            params={"transition": pairs, "steps": genot_steps, "seed": 0,
                    "coupling_mode": "precomputed", "n_tree_pairings": 1},
            kwargs=dict(n_steps=genot_steps, pairs=pairs, seed=0,
                        coupling_mode="precomputed", n_tree_pairings=1),
            bank=fgw_bank,
        ),
        "wot": dict(
            gate="moscot",
            params={"transition": pairs, "epsilon": eps, "max_cells": max_cells},
            kwargs=dict(pairs=pairs, epsilon=eps, max_cells=max_cells),
            bank=moscot_bank,
        ),
        "moslin": dict(
            gate="moscot",
            params={"transition": pairs, "epsilon": eps, "alpha": alpha,
                    "max_cells": max_cells},
            kwargs=dict(pairs=pairs, epsilon=eps, alpha=alpha,
                        max_cells=max_cells),
            bank=moscot_bank,
        ),
    }

    out: dict = {name: None for name in specs}
    for name, spec in specs.items():
        if not bcfg.get(spec["gate"], True):
            continue
        bl = registry.get(name)
        out[name] = train_or_load(
            method=bl.cache_method, label=name, dataset_cfg=ds_cfg,
            params=spec["params"], force_retrain=force,
            train_fn=lambda bl=bl, spec=spec: bl.train(
                spec["bank"](), dim=dim, **spec["kwargs"]),
        )
    return out


def baseline_predict_anchor(method: str, model_pkg, anchor_state, n_samples: int):
    """Predict n_samples target states at t=10 from one anchor at t=9; (n_samples, dim)."""
    arr = np.asarray(anchor_state, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr[None]
    return inf.forward_samples(
        registry.get(method), model_pkg, arr.reshape(1, -1),
        TIME_INPUT, LEAF_TIME, n_samples=n_samples, n_flow_steps=None,
    )[:, 0]


# --------------------------------------------------------------------------
#  Per-sigma evaluation
# --------------------------------------------------------------------------


def _to_torch(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    return torch.as_tensor(a, dtype=torch.float32)


def _fate_acc(pred, mu, mu_other):
    """fraction of predicted states nearer the true attractor than the other one."""
    x = np.asarray(pred, dtype=np.float64).reshape(-1)   # the fate-bias state is 1-d
    return float(np.mean(np.abs(x - mu) < np.abs(x - mu_other)))


def eval_sigma(sigma, cfg, pooler, splitter, bcfg, *,
               energy_loss, n_eval_seeds, n_baseline_samples, n_flow_steps):
    """Full evaluation for one sigma, given the already-loaded model + config."""
    logger.info("[sigma=%s] evaluating", sigma)

    want_fm = bcfg.get("fm", True)
    want_genot = bcfg.get("genot", True)
    train_ds = hydra.utils.instantiate(cfg.dataset) if (want_fm or want_genot) else None
    baselines = build_baselines(train_ds, cfg.dataset, bcfg) if train_ds is not None else {}

    sim_kw = dict(
        n_generations=int(cfg.dataset.n_generations),
        split_gen=int(cfg.dataset.split_gen),
        n_primed_generations=int(cfg.dataset.n_primed_generations),
        sigma=float(cfg.dataset.sigma),
        theta=float(cfg.dataset.theta),
        mu_1_primed=float(cfg.dataset.mu_1_primed),
        mu_2_primed=float(cfg.dataset.mu_2_primed),
        mu_1=float(cfg.dataset.mu_1),
        mu_2=float(cfg.dataset.mu_2),
        fate_prob=float(cfg.dataset.fate_prob),
        branch_interval=float(cfg.dataset.branch_interval),
        steps_per_interval=int(cfg.dataset.steps_per_interval),
    )
    mu_1, mu_2 = sim_kw["mu_1"], sim_kw["mu_2"]   # the two committed attractors

    splitter_rows: list[dict] = []
    baseline_rows: list[dict] = []
    case_panel: dict[tuple[int, str, int], dict] = {}
    cartoon_data = None
    case_trajectories: dict[tuple[int, str], dict] = {}
    case_seed_backgrounds: dict[int, dict] = {}

    baseline_methods = [(m, baselines[m]) for m in registry.names()
                        if baselines.get(m) is not None]

    for seed in range(1, n_eval_seeds + 1):
        res = make_fate_bias(seed=seed, **sim_kw)
        snap_input, leaves_input = res.snapshot(TIME_INPUT)
        snap_target, leaves_target = res.snapshot(LEAF_TIME)

        if sigma == CARTOON_SIGMA and cartoon_data is None and seed == 1:
            cartoon_data = _collect_cartoon_data(res, sim_kw)
        if sigma == CASE_SIGMA and seed not in case_seed_backgrounds:
            case_seed_backgrounds[seed] = _collect_cartoon_data(res, sim_kw)

        anchors_at_tin = [
            n for n in snap_input.nodes()
            if abs(snap_input.nodes[n].get("time", -1e9) - TIME_INPUT) < 1e-3
            and snap_input.out_degree(n) == 0
        ]
        anchors_at_tin = [
            a for a in anchors_at_tin
            if any(snap_target.out_degree(m) == 0
                   for m in nx.descendants(snap_target, a))
        ]

        for anchor in (anchors_at_tin if baseline_methods else []):
            x_anchor = np.asarray(leaves_input[anchor], dtype=np.float32)
            if x_anchor.ndim == 0:
                x_anchor = x_anchor[None]
            real_t = [
                m for m in nx.descendants(snap_target, anchor)
                if snap_target.out_degree(m) == 0
            ]
            real_states = np.stack([leaves_target[m] for m in real_t], axis=0)
            if real_states.ndim == 1:
                real_states = real_states[:, None]
            truth_t = _to_torch(real_states)
            mu = res.tree.nodes[real_t[0]]["params"]["mu"]   # committed attractor
            for method, pkg in baseline_methods:
                pred = baseline_predict_anchor(method, pkg, x_anchor, n_baseline_samples)
                ed = float(energy_loss(_to_torch(pred), truth_t))
                baseline_rows.append({
                    "sigma": sigma, "seed": seed, "anchor": anchor,
                    "method": method, "ed": ed,
                    "acc": _fate_acc(pred, mu, mu_1 + mu_2 - mu),
                })

        for cs in CLONE_SIZES:
            clones = list(find_clones(
                snap_input, leaves_input, snap_target, leaves_target,
                cs, TIME_INPUT, LEAF_TIME))
            for ci, clone in enumerate(clones):
                result, batch = predict_clone(
                    splitter, clone, snap_input, TIME_INPUT, LEAF_TIME, DEVICE,
                    n_steps=n_flow_steps)
                per_anchor = _anchor_descendant_states(
                    clone, result, batch, snap_target, leaves_target, LEAF_TIME)
                clone_lineage_rec = None
                if sigma == CASE_SIGMA and cs == 8:
                    clone_lineage_rec = record_clone_lineage(
                        res, clone["ancestor"], TIME_INPUT - math.log2(cs), TIME_INPUT)
                for entry in per_anchor:
                    truth_t = _to_torch(entry["true_states"])
                    pred_t = _to_torch(entry["pred_states"])
                    ed = float(energy_loss(pred_t, truth_t))
                    mu = res.tree.nodes[entry["true_descendants"][0]]["params"]["mu"]
                    splitter_rows.append({
                        "sigma": sigma, "seed": seed, "clone_size": cs,
                        "clone_idx": ci, "anchor": entry["anchor"],
                        "anchor_state": float(np.asarray(
                            leaves_input[entry["anchor"]]).reshape(-1)[0]),
                        "ed": ed,
                        "acc": _fate_acc(entry["pred_states"], mu, mu_1 + mu_2 - mu),
                    })
                    if sigma == CASE_SIGMA and cs in (1, 8):
                        case_panel[(seed, entry["anchor"], cs)] = {
                            "true_descendants": entry["true_descendants"],
                            "true_states": entry["true_states"],
                            "pred_states": entry["pred_states"],
                            "ed": ed,
                            "clone_anchors": list(clone["anchors"]),
                            "clone_anchor_states": np.asarray(
                                clone["anchor_states"], dtype=np.float32),
                            "lineage": clone_lineage_rec,
                        }

        if sigma == CASE_SIGMA:
            for anchor in anchors_at_tin:
                if (seed, anchor) in case_trajectories:
                    continue
                real_t = [
                    m for m in nx.descendants(snap_target, anchor)
                    if snap_target.out_degree(m) == 0
                ]
                trajs = []
                for m in real_t:
                    raw = res.tree.nodes[m].get("trajectory")
                    if raw is None:
                        continue
                    arr = np.asarray(raw)
                    mask = (arr[:, 0] >= TIME_INPUT - 1e-6) & (arr[:, 0] <= LEAF_TIME + 1e-6)
                    if not mask.any():
                        continue
                    trajs.append(arr[mask].astype(np.float32))
                if not trajs:
                    continue
                case_trajectories[(seed, anchor)] = {
                    "anchor_state": float(np.asarray(leaves_input[anchor]).reshape(-1)[0]),
                    "descendant_trajs": trajs,
                    "true_t_out_states": [
                        float(np.asarray(leaves_target[m]).reshape(-1)[0]) for m in real_t
                    ],
                }

    del baselines
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "splitter_rows": splitter_rows,
        "baseline_rows": baseline_rows,
        "case_panel": case_panel,
        "case_trajectories": case_trajectories,
        "case_seed_backgrounds": case_seed_backgrounds,
        "cartoon": cartoon_data,
    }


def _collect_cartoon_data(res, sim_kw) -> dict:
    """Every cell trajectory + its terminal fate (commit mu) for cartoon plotting."""
    leaf_mu_cache: dict = {}

    def leaf_mu(node):
        if node in leaf_mu_cache:
            return leaf_mu_cache[node]
        cur = node
        while res.tree.out_degree(cur) > 0:
            cur = next(iter(res.tree.successors(cur)))
        mu = res.tree.nodes[cur]["params"].get("mu", 0.0)
        leaf_mu_cache[node] = mu
        return mu

    trajs = []
    for n in res.tree.nodes():
        t = res.tree.nodes[n].get("trajectory")
        if t is None:
            continue
        trajs.append((np.asarray(t, dtype=np.float32), float(leaf_mu(n))))
    return {
        "trajectories": trajs,
        "sigma": float(sim_kw["sigma"]),
        "split_gen": int(sim_kw["split_gen"]),
        "n_primed_generations": int(sim_kw["n_primed_generations"]),
        "branch_interval": float(sim_kw["branch_interval"]),
        "time_input": TIME_INPUT,
        "leaf_time": LEAF_TIME,
        "commit_time": float(sim_kw["split_gen"] + sim_kw["n_primed_generations"])
                        * float(sim_kw["branch_interval"]),
        "split_time": float(sim_kw["split_gen"] * sim_kw["branch_interval"]),
    }


def select_coi(case_panel: dict):
    """(seed, anchor) with the largest n=1 - n=8 ED improvement."""
    deltas = []
    seen = set()
    for (seed, anchor, cs), info in case_panel.items():
        if (seed, anchor) in seen:
            continue
        if (seed, anchor, 1) in case_panel and (seed, anchor, 8) in case_panel:
            delta = case_panel[(seed, anchor, 1)]["ed"] - case_panel[(seed, anchor, 8)]["ed"]
            deltas.append((delta, seed, anchor))
            seen.add((seed, anchor))
    if not deltas:
        return None
    deltas.sort(key=lambda x: -x[0])
    return deltas[0][1], deltas[0][2]


# --------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------


def run(ecfg, experiment, *, plot_only=False, skip_pool_split_train=True,
        retrain_pool_split=False, pool_split_steps=None, launcher="local",
        extra_overrides=None):
    """Run (or just plot) the context benchmark.

    Symmetric with the crossing driver: one ``experiment`` (the model) is
    evaluated across the ``datasets`` axis (one per-sigma dataset config).
    Each model is loaded by config hash; sigma is read from its config.
    """
    datasets = list(ecfg.datasets)
    labels = [float(s) for s in ecfg.labels]
    bcfg = ecfg.baselines
    n_flow_steps = int(ecfg.get("n_flow_steps", 20))
    eval_step = ecfg.get("eval_step", None)
    eval_step = int(eval_step) if eval_step is not None else None
    extra_overrides = list(extra_overrides or [])
    tag = experiment.split("/")[-1]

    cache_path = ROOT / "synthetic_evals" / "context" / f"results_{tag}.pkl"
    figure_dir = ROOT / "synthetic_evals" / "context" / f"figures_{tag}"

    if not plot_only:
        torch.manual_seed(0)
        energy_loss = viz.energy_distance   # the shared U-statistic

        # Train the per-sigma checkpoints (same multirun mechanism as crossing:
        # one experiment swept over the dataset axis). retrain forces it.
        run_pool_split_multirun(
            experiment, datasets,
            skip_if_done=skip_pool_split_train and not retrain_pool_split,
            n_steps=pool_split_steps, extra_overrides=extra_overrides,
            launcher=launcher,
        )

        per_sigma: dict[float, dict] = {}
        for dataset, label in zip(datasets, labels):
            logger.info("=== sigma=%s (%s / %s) ===", label, experiment, dataset)
            pooler, splitter, cfg = load_pool_split(
                experiment, dataset, extra_overrides, step=eval_step)
            sigma = float(cfg.dataset.sigma)
            per_sigma[sigma] = eval_sigma(
                sigma, cfg, pooler, splitter, bcfg, energy_loss=energy_loss,
                n_eval_seeds=int(ecfg.n_eval_seeds),
                n_baseline_samples=int(ecfg.n_baseline_samples),
                n_flow_steps=n_flow_steps,
            )

        case_block = per_sigma.get(CASE_SIGMA, {})
        case_panel = case_block.get("case_panel", {})
        case_trajs = case_block.get("case_trajectories", {})
        case_backgrounds = case_block.get("case_seed_backgrounds", {})
        coi = select_coi(case_panel)
        case_data = None
        if coi is not None:
            seed, anchor = coi
            traj_info = case_trajs.get((seed, anchor))
            n1 = case_panel.get((seed, anchor, 1))
            n8 = case_panel.get((seed, anchor, 8))
            bg = case_backgrounds.get(seed)
            if traj_info and n1 and n8:
                case_data = {
                    "sigma": CASE_SIGMA, "seed": seed, "anchor": anchor,
                    "anchor_state": traj_info["anchor_state"],
                    "descendant_trajs": traj_info["descendant_trajs"],
                    "true_t_out_states": traj_info["true_t_out_states"],
                    "n1_pred_states": n1["pred_states"],
                    "n8_pred_states": n8["pred_states"],
                    "n1_ed": n1["ed"], "n8_ed": n8["ed"],
                    "n1_clone_anchors": n1["clone_anchors"],
                    "n1_clone_anchor_states": n1["clone_anchor_states"],
                    "n8_clone_anchors": n8["clone_anchors"],
                    "n8_clone_anchor_states": n8["clone_anchor_states"],
                    "clone_lineage": n8["lineage"],
                    "background_trajectories": bg["trajectories"] if bg else [],
                    "t_in": TIME_INPUT, "t_out": LEAF_TIME,
                }
                logger.info(
                    "COI: seed=%d anchor=%s delta_ed=%.4f (n1=%.4f -> n8=%.4f)",
                    seed, anchor, n1["ed"] - n8["ed"], n1["ed"], n8["ed"])

        cartoon = per_sigma.get(CARTOON_SIGMA, {}).get("cartoon")
        if cartoon is None:
            logger.warning("CARTOON_SIGMA=%s not run; cartoon panel missing", CARTOON_SIGMA)
        results = {
            "sigmas": sorted(per_sigma.keys()),
            "clone_sizes": CLONE_SIZES,
            "splitter_rows": [r for blk in per_sigma.values() for r in blk["splitter_rows"]],
            "baseline_rows": [r for blk in per_sigma.values() for r in blk["baseline_rows"]],
            "cartoon": cartoon, "case": case_data,
            "time_input": TIME_INPUT, "leaf_time": LEAF_TIME,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        logger.info("wrote %s", cache_path)

    logger.info("=== plotting (separated rows) ===")
    from synthetic_evals.context.plot import render_rows
    with open(cache_path, "rb") as f:
        results = pickle.load(f)
    render_rows(results, figure_dir)
