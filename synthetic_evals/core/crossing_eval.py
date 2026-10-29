"""Crossing / high-dimensional-crossing benchmark eval.

One eval loop for both the native 1-D crossing benchmark and its
high-dimensional embedding -- the latter is just "a projector is configured on
the dataset". For each dataset variant (crossing / noncrossing) it loads the
trained pool/split model, trains-or-loads the model-agnostic baselines once
(cached, shared across model variants), runs forward (splitter) and reverse
(pooler) inference on a held-out simulation, and writes the results pickle the
shared ``recon_figure`` grammar renders.
"""

from __future__ import annotations

import functools
import logging
import pickle
from pathlib import Path

import numpy as np

from datasets.synthetic import (
    ProjectedSimResult,
    RandomProjector,
    make_hierarchical_ou,
)
from synthetic_evals.baselines import (
    genot_fgw,
    genot_sinkhorn,
    marginal_fm,
    moscot_ot,
    registry,
)
from synthetic_evals.baselines.cache import train_or_load
from synthetic_evals.core import inference as inf
from synthetic_evals.core.hydra_io import (
    DEVICE,
    ROOT,
    compose_cfg,
    load_pool_split,
    run_pool_split_multirun,
)
from synthetic_evals.core.subtrees import (
    build_layer_subtrees,
    collect_trajectories,
    subtrees_to_pairs,
)

logger = logging.getLogger("synth_eval.crossing")


# --------------------------------------------------------------------------
#  Held-out evaluation simulation
# --------------------------------------------------------------------------


def build_eval_sim(ds_cfg, seed: int):
    """One held-out hierarchical-OU simulation observed at every obs_time.

    When the dataset config carries ``proj_dim``, the 1-D latent is embedded
    into R^proj_dim with the SAME random projector the training data used
    (shared ``proj_seed``), so the whole eval path is D-dimensional. Absent
    ``proj_dim``, this is the native 1-D crossing eval.
    """
    a2 = ds_cfg.get("attractors_2", None)
    if a2 is not None:
        a2 = {float(k): tuple(float(x) for x in v) for k, v in dict(a2).items()}
    a1 = ds_cfg.get("attractors_1", None)
    if a1 is not None:
        a1 = tuple(float(x) for x in a1)
    sim = make_hierarchical_ou(
        n_generations=int(ds_cfg.n_generations),
        split_gen=int(ds_cfg.split_gen),
        gap=int(ds_cfg.gap),
        sigma=float(ds_cfg.sigma),
        theta=float(ds_cfg.theta),
        branch_interval=float(ds_cfg.branch_interval),
        steps_per_interval=int(ds_cfg.steps_per_interval),
        obs_times=np.array(list(ds_cfg.obs_times), dtype=float),
        seed=seed,
        attractors_1=a1,
        attractors_2=a2,
    )
    proj_dim = ds_cfg.get("proj_dim", None)
    if proj_dim is not None:
        projector = RandomProjector(
            dim_in=1, dim_out=int(proj_dim),
            seed=int(ds_cfg.get("proj_seed", 0)),
            noise=float(ds_cfg.get("proj_noise", 0.1)),
        )
        sim = ProjectedSimResult(sim, projector)
    return sim


# --------------------------------------------------------------------------
#  Model-agnostic baselines (trained once, cached, reused across variants)
# --------------------------------------------------------------------------


def build_baselines(train_ds, ds_cfg, bcfg) -> dict:
    """Train-or-load forward + reverse models for every registered baseline.

    Keys are ``<name>`` (forward) and ``<name>_rev``. Each is cached by (dataset,
    hyperparameters), so model-variant evals on the same dataset reuse the cache.
    """
    dim = int(train_ds.x_dim)
    times = [float(t) for t in ds_cfg.obs_times]
    fwd = marginal_fm.adjacent_pairs(times)
    rev = [(b, a) for (a, b) in fwd]
    force = bool(bcfg.get("force_retrain", False))
    pairs_for = {"fwd": None, "rev": rev}          # forward uses the module default

    fm_steps, fm_hidden, fm_layers = (
        int(bcfg.fm_steps), int(bcfg.fm_hidden), int(bcfg.fm_layers))
    genot_steps = int(bcfg.genot_steps)
    cmode = str(bcfg.coupling_mode)
    npair = int(bcfg.get("fgw_n_tree_pairings", 1))
    eps = float(bcfg.get("moscot_epsilon", 1e-2))
    alpha = float(bcfg.get("moslin_alpha", 0.5))
    max_cells = int(bcfg.get("moscot_max_cells", 2048))

    @functools.cache
    def fgw_bank():
        return genot_fgw.FGWBank.from_dataset(train_ds, ds_cfg)

    @functools.cache
    def moscot_bank():
        return moscot_ot.MoscotBank.from_fgw_bank(fgw_bank())

    # per-baseline: cache-key params, train kwargs, bank builder, and the (fwd, rev)
    # seeds. seeds are pinned per direction because they are part of the cache key.
    specs = {
        "fm": dict(
            params={"times": times, "steps": fm_steps,
                    "hidden": fm_hidden, "layers": fm_layers},
            kwargs=dict(hidden=fm_hidden, n_layers=fm_layers, n_steps=fm_steps,
                        device=DEVICE),
            bank=lambda: marginal_fm.TimeBank.from_dataset(train_ds),
            seeds=(0, 1),
        ),
        "gs": dict(
            params={"times": times, "steps": genot_steps, "coupling_mode": cmode},
            kwargs=dict(n_steps=genot_steps, coupling_mode=cmode),
            bank=lambda: genot_sinkhorn.GenotBank.from_dataset(train_ds),
            seeds=(0, 1), gate="genot",
        ),
        "gf": dict(
            params={"times": times, "steps": genot_steps, "coupling_mode": cmode,
                    "n_tree_pairings": npair},
            kwargs=dict(n_steps=genot_steps, coupling_mode=cmode,
                        n_tree_pairings=npair),
            bank=fgw_bank,
            seeds=(2, 3), gate="genot",
        ),
        "wot": dict(
            params={"times": times, "epsilon": eps, "max_cells": max_cells},
            kwargs=dict(epsilon=eps, max_cells=max_cells),
            bank=moscot_bank, seeds=(0, 0), gate="moscot",
        ),
        "moslin": dict(
            params={"times": times, "epsilon": eps, "alpha": alpha,
                    "max_cells": max_cells},
            kwargs=dict(epsilon=eps, alpha=alpha, max_cells=max_cells),
            bank=moscot_bank, seeds=(0, 0), gate="moscot",
        ),
    }

    out: dict = {}
    for name, spec in specs.items():
        bl = registry.get(name)
        gate = spec.get("gate")           # no gate => always built
        if gate is not None and not bcfg.get(gate, True):
            out[name] = out[f"{name}_rev"] = None
            continue
        for direction, seed in zip(("fwd", "rev"), spec["seeds"]):
            key = name if direction == "fwd" else f"{name}_rev"
            kw = dict(spec["kwargs"], seed=seed)
            if pairs_for[direction] is not None:
                kw["pairs"] = pairs_for[direction]
            out[key] = train_or_load(
                method=bl.cache_method, label=key, dataset_cfg=ds_cfg,
                params={"dir": direction, "seed": seed, **spec["params"]},
                train_fn=lambda bl=bl, spec=spec, kw=kw: bl.train(
                    spec["bank"](), dim=dim, **kw),
                force_retrain=force,
            )
    return out


# --------------------------------------------------------------------------
#  Per-variant eval
# --------------------------------------------------------------------------


def run_eval(variant, sim_res, pool, splitter, baselines, *,
             n_samples=4, n_flow_steps=20):
    """Forward (splitter + baselines) and reverse (pooler + baselines) eval for one
    variant. A baseline is skipped when its trained model is None."""
    obs_times = [float(t) for t in sim_res.obs_times]
    pair_times = marginal_fm.adjacent_pairs(obs_times)

    out: dict = {
        "obs_times": obs_times, "pair_times": pair_times,
        "gt_pairs": {}, "ps_pairs": {}, "pool_pairs": {},
    }
    for name in registry.names():
        if baselines.get(name) is not None:
            out[f"{name}_pairs"] = {}
        if baselines.get(f"{name}_rev") is not None:
            out[f"{name}_rev_pairs"] = {}

    for t_src, t_tgt in pair_times:
        subtrees = build_layer_subtrees(sim_res, t_src, t_tgt)
        if not subtrees:
            logger.warning("[%s] no subtrees for (%s, %s)", variant, t_src, t_tgt)
            continue
        pairs = subtrees_to_pairs(subtrees, t_src, t_tgt)
        x_src = np.stack([p[4] for p in pairs], axis=0).astype(np.float32)
        x_tgt = np.stack([p[5] for p in pairs], axis=0).astype(np.float32)

        out["gt_pairs"][(t_src, t_tgt)] = {"x_src": x_src, "x_tgt": x_tgt}
        out["ps_pairs"][(t_src, t_tgt)] = inf.splitter_predict_subtrees(
            splitter, subtrees, n_samples=n_samples, n_flow_steps=n_flow_steps)
        out["pool_pairs"][(t_src, t_tgt)] = inf.pool_predict_subtrees(
            pool, subtrees, n_samples=n_samples, n_flow_steps=n_flow_steps)

        for name in registry.names():
            bl = registry.get(name)
            if baselines.get(name) is not None:
                out[f"{name}_pairs"][(t_src, t_tgt)] = inf.forward_samples(
                    bl, baselines[name], x_src, t_src, t_tgt,
                    n_samples=n_samples, n_flow_steps=n_flow_steps)
            if baselines.get(f"{name}_rev") is not None:
                out[f"{name}_rev_pairs"][(t_src, t_tgt)] = inf.reverse_per_parent(
                    bl, baselines[f"{name}_rev"], subtrees, pairs, t_src, t_tgt,
                    n_samples=n_samples, n_flow_steps=n_flow_steps)

    return out


# --------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------


def results_path(experiment: str, proj_dim) -> Path:
    """Result pickle for one (experiment, dimension). Dimension keys the file so
    a dimension sweep coexists; the scaling plotter gathers them."""
    tag = experiment.split("/")[-1]
    if proj_dim is not None:
        tag += f"_d{int(proj_dim)}"
    return ROOT / "synthetic_evals" / "high_dim_crossing" / f"results_{tag}.pkl"


def run(ecfg, experiment: str, *, plot_only: bool = False,
        skip_pool_split_train: bool = True, launcher: str = "local",
        extra_overrides: list[str] | None = None):
    """Compute the crossing-family benchmark for one experiment (one dimension).

    Symmetric with context: the ``datasets`` axis (crossing / coherent) is swept;
    each model is loaded by config hash. Writes a per-dimension results pickle;
    figures are produced separately by ``synthetic_evals.figure``, which gathers
    the dimensions.
    """
    import hydra

    datasets = list(ecfg.datasets)
    labels = list(ecfg.labels)
    extra_overrides = list(extra_overrides or [])

    # proj_dim keys the output file and decides whether the eval projects.
    probe_ds = compose_cfg(experiment, datasets[0], extra_overrides).dataset
    proj_dim = probe_ds.get("proj_dim", None)
    cache_path = results_path(experiment, proj_dim)

    if not plot_only:
        logger.info("=== pool-split training (Hydra multirun) ===")
        run_pool_split_multirun(
            experiment, datasets, skip_if_done=skip_pool_split_train,
            extra_overrides=extra_overrides, launcher=launcher,
        )

        results = {}
        for label, dataset in zip(labels, datasets):
            logger.info("=== [%s] baselines (train-or-load) ===", label)
            cfg = compose_cfg(experiment, dataset, extra_overrides)
            train_ds = hydra.utils.instantiate(cfg.dataset)
            baselines = build_baselines(train_ds, cfg.dataset, ecfg.baselines)

            logger.info("=== [%s] held-out eval ===", label)
            pool, splitter, _ = load_pool_split(experiment, dataset, extra_overrides)
            eval_sim = build_eval_sim(cfg.dataset, seed=int(ecfg.eval_seed))
            eval_data = run_eval(
                label, eval_sim, pool, splitter, baselines,
                n_samples=int(ecfg.n_samples), n_flow_steps=int(ecfg.n_flow_steps),
            )
            eval_data["trajectories"] = collect_trajectories(eval_sim)
            if getattr(eval_sim, "projector", None) is not None:
                p = eval_sim.projector
                eval_data["proj"] = {
                    "w": np.asarray(p.w, dtype=np.float64),
                    "proj_dim": p.dim_out,
                    "proj_noise": p.noise,
                }
            results[label] = eval_data

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        logger.info("Saved results -> %s", cache_path)

    # the crossing panels are drawn from this pickle by the synthetic figure
    # pipeline (synthetic_evals/figure/compute_crossing.py, then plot_crossing.py).
    logger.info("crossing results: %s (figures: synthetic_evals/figure)", cache_path)
