"""All-metrics evaluation.

Runs every eval metric (clonal, marginal, germ-layer,
curated-transition, linkage, interpolant) against a dict of checkpoint
paths plus optional genot baseline pkls, writes a consolidated CSV and a
lollipop PNG. Calls the ``analysis.lib`` metric functions directly.

Example::

    PYTHONPATH=. python scripts/run_all_eval.py \\
        --cfg 'name=outputs/<experiment_name>_<hash>' \\
        --genot 'FGW=outputs/genot_fgw_unified_model.pkl' \\
        --output-csv all_eval_results.csv \\
        --output-png all_eval_lollipop.png

Each ``--cfg NAME=PATH`` pair is a trained pool+split checkpoint dir; the
dir must also contain an ``assignments_nn_*.pkl`` for germ-layer, curated
transition, and linkage metrics.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import inspect

from analysis.lib.metrics.clonal_preservation import (
    clonal_preservation, clonal_preservation_genot,
)
from analysis.lib.metrics.curated_transition import curated_transition_fraction
from analysis.lib.metrics.germ_layer_eval import (
    ancestor_map_germ_layer_transitions,
    assignment_germ_layer_transitions,
)
from analysis.lib.metrics.linkage_metrics import (
    ancestor_map_linkage_per_tree,
    assignment_linkage_per_tree,
    compare_linkage_normalized,
    real_linkage_per_tree,
)
from analysis.lib.metrics.marginal_preservation import (
    interpolant_matching, marginal_preservation,
)
try:  # genot variants only needed when --genot baselines are passed
    from analysis.lib.metrics.genot_eval import (
        interpolant_matching_genot, marginal_preservation_genot,
    )
except ImportError:  # jax / ott not installed
    interpolant_matching_genot = marginal_preservation_genot = None
from analysis.lib.coupling.nearest_neighbor_coupling import couple_all_trees
from analysis.lib.coupling.sankey import build_transition_graph, build_transition_graph_genot
from datasets.mouse_embryo import MouseEmbryoDataset
from utils.experiment_utils import get_experiment_info

# Flow-matching steps for multi-step pool inference at eval time
# (1 = the one-step head used in training). Set from --pool-steps in main().
_POOL_STEPS = 1


def _parse_kv(arg: str) -> tuple[str, str]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"expected NAME=PATH, got {arg!r}"
        )
    name, path = arg.split("=", 1)
    return name.strip(), path.strip()


def _load_pooler_and_splitter(cfg_path: str):
    d = get_experiment_info(cfg_path)
    pool_ckpts = [x for x in os.listdir(d["dir"])
                  if x.startswith("pool_step") and x.endswith(".pt")]
    split_ckpts = [x for x in os.listdir(d["dir"])
                   if x.startswith("split_step") and x.endswith(".pt")]

    def _step(fname: str) -> int:
        # pool_step{N}.pt -> int(N); tolerant of suffixes.
        base = fname.split("_step")[-1].split(".")[0]
        return int(base)

    pool_final = max(pool_ckpts, key=_step)
    split_final = max(split_ckpts, key=_step)
    print(f"  pool : {pool_final}")
    print(f"  split: {split_final}")

    pooler = hydra.utils.instantiate(d["config"]["pooler"]).cuda()
    pooler.load_state_dict(torch.load(
        os.path.join(d["dir"], pool_final), map_location="cuda",
    ))
    # When the run used share_pool_split, pool and split are the same model;
    # the saved split state dict is identical to pool's. The splitter config
    # in this case is unmodified base (different shape), so just alias.
    if d["config"].get("share_pool_split", False):
        splitter = pooler
    else:
        splitter = hydra.utils.instantiate(d["config"]["splitter"]).cuda()
        splitter.load_state_dict(torch.load(
            os.path.join(d["dir"], split_final), map_location="cuda",
        ))
    # Multi-step pool inference (flow-matching sampler) at eval. Wrap pool_batch
    # so every downstream caller -- couple_all_trees and the direct
    # pool_batch(batch) calls in clonal/marginal/interpolant -- uses _POOL_STEPS
    # steps. Poolers without an n_steps arg (e.g. TreeFormer) are left untouched.
    if _POOL_STEPS != 1 and "n_steps" in inspect.signature(pooler.pool_batch).parameters:
        _orig_pool_batch = pooler.pool_batch

        def _pool_batch(batch, _orig=_orig_pool_batch, _steps=_POOL_STEPS):
            return _orig(batch, n_steps=_steps)

        pooler.pool_batch = _pool_batch
    pooler.eval()
    splitter.eval()
    return pooler, splitter


def _find_assignments(cfg_path: str) -> str | None:
    files = [x for x in os.listdir(cfg_path)
             if "assignments_nn" in x and x.endswith(".pkl")
             and "_perparent" not in x and "_brokenrev" not in x]
    if not files:
        return None
    return os.path.join(cfg_path, files[0])


def _ensure_assignments(cfg_path: str, trees, states) -> str | None:
    """Generate ``assignments_nn_step{N}.pkl`` via couple_all_trees if missing."""
    existing = _find_assignments(cfg_path)
    if existing is not None:
        return existing
    pool, _ = _load_pooler_and_splitter(cfg_path)
    # figure out what step we're at for naming
    pool_ckpts = [x for x in os.listdir(cfg_path)
                  if x.startswith("pool_step") and x.endswith(".pt")]
    step = max(int(f[len("pool_step"):-3]) for f in pool_ckpts)
    print(f"  generating assignments_nn_step{step}.pkl (couple_all_trees)")
    coupling = couple_all_trees(pool, trees, states)
    out = os.path.join(cfg_path, f"assignments_nn_step{step}.pkl")
    with open(out, "wb") as f:
        pickle.dump(coupling, f)
    print(f"  wrote {out}")
    return out


def plot_lollipop(
    df: pd.DataFrame,
    model_order: list[str],
    colors: dict[str, str],
    metric_direction: dict[str, bool],
) -> plt.Figure:
    STEM_GREY = "#BBBBBB"
    sns.set_style("whitegrid", {"grid.linestyle": "--"})
    plt.rcParams.update({
        "text.usetex": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.linewidth": 0.5,
    })
    metrics = df["metric"].unique()
    n = len(metrics)
    fig, axes = plt.subplots(
        1, n, figsize=(3 * n, 1.5),
        gridspec_kw={"wspace": 0.45}, dpi=150,
    )
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric].copy()
        sub["model"] = pd.Categorical(sub["model"], model_order, ordered=True)
        sub = sub.sort_values("model").dropna(subset=["model"])
        higher_better = metric_direction.get(metric, False)
        best_val = sub["value"].max() if higher_better else sub["value"].min()
        best_models = set(sub.loc[sub["value"] == best_val, "model"])
        for _, row in sub.iterrows():
            is_best = row["model"] in best_models
            c = colors[row["model"]]
            stem_c = c if is_best else STEM_GREY
            lw = 2.5 if is_best else 1.2
            ms = 12 if is_best else 8
            ax.vlines(row["model"], 0, row["value"],
                      color=stem_c, linewidth=lw, zorder=2)
            ax.plot(row["model"], row["value"], "o",
                    color=c, markersize=ms, zorder=3)
        ax.set_title(f"{metric}", fontsize=8, fontfamily="monospace",
                     fontweight="black", loc="center", pad=8)
        ax.tick_params(axis="x", labelsize=7.5, length=0, rotation=70)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.grid(True, linewidth=0.25, color="#cccccc", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
        ax.set_xlim(-0.5, len(model_order) - 0.5)
    fig.canvas.draw()
    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        higher_better = metric_direction.get(metric, False)
        best_val = sub["value"].max() if higher_better else sub["value"].min()
        best_models = set(sub.loc[sub["value"] == best_val, "model"])
        for label in ax.get_xticklabels():
            if label.get_text() in best_models:
                label.set_fontweight("bold")
                label.set_color(colors[label.get_text()])
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", action="append", type=_parse_kv, default=[],
                    help="NAME=PATH of a trained checkpoint dir (repeatable)")
    ap.add_argument("--genot", action="append", type=_parse_kv, default=[],
                    help="NAME=PATH of a saved genot baseline pkl (repeatable)")
    ap.add_argument("--data-dir", default="data/lt/",
                    help="passed to MouseEmbryoDataset(data_dir=...)")
    ap.add_argument("--type-info", default="data/lt/final_type_info.csv",
                    help="CSV with cell_type -> germ_layer mapping")
    ap.add_argument("--transition-dir", default="literature/transitions",
                    help="directory of curated cell-type transition CSVs")
    ap.add_argument("--n-leaves", type=int, default=512)
    ap.add_argument("--interpolant-time", type=float, default=8.75)
    ap.add_argument("--linkage-source-time", type=float, default=10.0)
    ap.add_argument("--linkage-target-time", type=float, default=9.5)
    ap.add_argument("--output-csv", default="all_eval_results.csv")
    ap.add_argument("--output-png", default="all_eval_lollipop.png")
    ap.add_argument("--pool-steps", type=int, default=1,
                    help="flow-matching steps for multi-step pool inference "
                         "(1 = the one-step head used in training)")
    args = ap.parse_args()

    global _POOL_STEPS
    _POOL_STEPS = args.pool_steps
    if _POOL_STEPS != 1:
        print(f"[all_eval] multi-step pool inference: n_steps={_POOL_STEPS}")

    cfgs = dict(args.cfg)
    genots = dict(args.genot)
    if not cfgs and not genots:
        ap.error("need at least one --cfg or --genot")

    print(f"[all_eval] loading dataset from {args.data_dir}")
    dataset = MouseEmbryoDataset(data_dir=args.data_dir, devmap_only=False)

    # name -> (cell_subtype -> cell_type -> germ_layer / subtype_to_type)
    cell_type_info = pd.read_csv(args.type_info, index_col=0)
    name_to_subtype = dict(zip(
        dataset.cell_types_df.index, dataset.cell_types_df["cell_subtype"],
    ))
    subtype_to_type = dict(zip(cell_type_info.index, cell_type_info["cell_type"]))
    type_to_gl = dict(zip(cell_type_info["cell_type"], cell_type_info["germ_layer"]))
    name_to_gl = {
        n: type_to_gl[subtype_to_type[st]]
        for n, st in name_to_subtype.items()
        if st in subtype_to_type and subtype_to_type[st] in type_to_gl
    }
    name_to_cell_type = {
        n: subtype_to_type[st]
        for n, st in name_to_subtype.items()
        if st in subtype_to_type
    }
    types = np.unique(list(name_to_cell_type.values()))

    import gc
    import traceback as _tb

    results = {"model": [], "metric": [], "value": []}

    def _add(model: str, metric: str, value: float) -> None:
        results["model"].append(model)
        results["metric"].append(metric)
        results["value"].append(value)
        # incremental write so partial progress survives a crash
        pd.DataFrame(results).to_csv(args.output_csv, index=False)

    def _safe(name: str, fn):
        """Run ``fn``; on exception print traceback and return None."""
        try:
            return fn()
        except Exception:  # noqa: BLE001
            print(f"  ! {name} crashed:")
            _tb.print_exc()
            return None

    def _free():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- clonal preservation -----------------------------------------
    print("\n[1/6] clonal preservation")
    trees_no75 = {k: v for k, v in dataset.trees.items() if "7.5" not in k}
    for cfg_name, cfg_path in cfgs.items():
        print(f"  {cfg_name}")
        pool, split = _load_pooler_and_splitter(cfg_path)
        v = _safe(f"{cfg_name} clonal", lambda: float(clonal_preservation(
            pool, split, trees_no75, dataset.leaf_states,
            offset=0.5, n_leaves=args.n_leaves,
        )["energy_distance"].mean()))
        if v is not None:
            _add(cfg_name, "mean_clonal_preservation", v)
        del pool, split; _free()
    for g_name, g_path in genots.items():
        print(f"  {g_name}")
        v = _safe(f"{g_name} clonal", lambda: float(clonal_preservation_genot(
            g_path, trees_no75, dataset.leaf_states,
            offset=0.5, n_leaves=args.n_leaves,
        )["energy_distance"].mean()))
        if v is not None:
            _add(g_name, "mean_clonal_preservation", v)
        _free()

    # --- marginal preservation ---------------------------------------
    print("\n[2/6] marginal preservation")
    for cfg_name, cfg_path in cfgs.items():
        print(f"  {cfg_name}")
        pool, split = _load_pooler_and_splitter(cfg_path)
        v = _safe(f"{cfg_name} marginal", lambda: float(marginal_preservation(
            pool, split, dataset.trees, dataset.leaf_states,
            n_cells=args.n_leaves, time_tol=0.5,
        )["w2_normalised"].mean()))
        if v is not None:
            _add(cfg_name, "mean_marginal_preservation", v)
        del pool, split; _free()
    for g_name, g_path in genots.items():
        print(f"  {g_name}")
        v = _safe(f"{g_name} marginal", lambda: float(marginal_preservation_genot(
            g_path, dataset.trees, dataset.leaf_states,
            n_cells=args.n_leaves,
        )["w2_normalised"].mean()))
        if v is not None:
            _add(g_name, "mean_marginal_preservation", v)
        _free()

    # --- germ layer crossing -----------------------------------------
    print("\n[3/6] germ layer crossing")
    for cfg_name, cfg_path in cfgs.items():
        print(f"  {cfg_name}")
        def _germ():
            a_path = _ensure_assignments(cfg_path, dataset.trees, dataset.states)
            assignments = pickle.load(open(a_path, "rb"))
            tmat = assignment_germ_layer_transitions(
                dataset.trees, assignments, name_to_gl,
            )
            return float(np.asarray(tmat.values).sum() - np.trace(tmat.values))
        v = _safe(f"{cfg_name} germ", _germ)
        if v is not None:
            _add(cfg_name, "germ_layer_crossing", v)
        _free()
    for g_name, g_path in genots.items():
        print(f"  {g_name}")
        def _germ_g(p=g_path):
            ancestral_map = pickle.load(open(p, "rb"))["ancestral_mapping"]
            tmat = ancestor_map_germ_layer_transitions(ancestral_map, name_to_gl)
            return float(np.asarray(tmat.values).sum() - np.trace(tmat.values))
        v = _safe(f"{g_name} germ", _germ_g)
        if v is not None:
            _add(g_name, "germ_layer_crossing", v)
        _free()

    # --- curated transition fraction ---------------------------------
    print("\n[4/6] curated transition fraction")
    obs_time = {n: t for n, t in zip(
        dataset.states.obs_names, dataset.states.obs["time"],
    )}
    for cfg_name, cfg_path in cfgs.items():
        print(f"  {cfg_name}")
        def _ct(p=cfg_path):
            a_path = _ensure_assignments(p, dataset.trees, dataset.states)
            assignments = pickle.load(open(a_path, "rb"))
            G = build_transition_graph(
                dataset.trees, name_to_cell_type, assignments,
                time_key="time", dt=0.5, nn_quantile=0.9,
                min_support=20, min_fold=5, alpha=0.05,
            )
            return curated_transition_fraction(args.transition_dir, G)
        v = _safe(f"{cfg_name} curated", _ct)
        if v is not None:
            _add(cfg_name, "curated_transition_fraction", v)
        _free()
    for g_name, g_path in genots.items():
        print(f"  {g_name}")
        def _ct_g(p=g_path):
            ancestral_map = pickle.load(open(p, "rb"))["ancestral_mapping"]
            G = build_transition_graph_genot(
                ancestral_map, name_to_cell_type, obs_time,
                dt=0.5, min_support=100, min_fold=5, alpha=0.05,
            )
            return curated_transition_fraction(args.transition_dir, G)
        v = _safe(f"{g_name} curated", _ct_g)
        if v is not None:
            _add(g_name, "curated_transition_fraction", v)
        _free()

    # --- linkage correlation -----------------------------------------
    print("\n[5/6] linkage correlation")
    real_linkage = real_linkage_per_tree(
        dataset.trees, args.linkage_target_time, name_to_cell_type, types=types,
    )
    for cfg_name, cfg_path in cfgs.items():
        print(f"  {cfg_name}")
        def _lk(p=cfg_path):
            a_path = _ensure_assignments(p, dataset.trees, dataset.states)
            assignments = pickle.load(open(a_path, "rb"))
            assign_linkage = assignment_linkage_per_tree(
                dataset.trees, assignments,
                source_time=args.linkage_source_time,
                target_time=args.linkage_target_time,
                name_to_type=name_to_cell_type, types=types,
            )
            r, _ = compare_linkage_normalized(real_linkage, assign_linkage)
            return float(r)
        v = _safe(f"{cfg_name} linkage", _lk)
        if v is not None:
            _add(cfg_name, "linkage_correlation", v)
        _free()
    for g_name, g_path in genots.items():
        print(f"  {g_name}")
        def _lk_g(p=g_path):
            ancestral_map = pickle.load(open(p, "rb"))["ancestral_mapping"]
            anc_linkage = ancestor_map_linkage_per_tree(
                dataset.trees, ancestral_map,
                source_time=args.linkage_source_time,
                target_time=args.linkage_target_time,
                name_to_type=name_to_cell_type, types=types,
            )
            r, _ = compare_linkage_normalized(real_linkage, anc_linkage)
            return float(r)
        v = _safe(f"{g_name} linkage", _lk_g)
        if v is not None:
            _add(g_name, "linkage_correlation", v)
        _free()

    # --- interpolant matching ----------------------------------------
    print("\n[6/6] interpolant matching")
    gt_interp = dataset.states[
        dataset.states.obs["time"] == args.interpolant_time
    ].obsm["X_scvi"]
    print(f"  ground truth interpolant states: {gt_interp.shape}")
    for cfg_name, cfg_path in cfgs.items():
        print(f"  {cfg_name}")
        pool, split = _load_pooler_and_splitter(cfg_path)
        v = _safe(f"{cfg_name} interpolant", lambda: float(interpolant_matching(
            pool, split, dataset.trees, dataset.leaf_states,
            interpolant_time=args.interpolant_time, time_tol=0.1,
            ground_truth_interpolant_states=gt_interp,
        )["w2_normalised"].mean()))
        if v is not None:
            _add(cfg_name, "mean_interpolant_matching", v)
        del pool, split; _free()
    for g_name, g_path in genots.items():
        print(f"  {g_name}")
        v = _safe(f"{g_name} interpolant", lambda: float(interpolant_matching_genot(
            g_path, dataset.trees, dataset.leaf_states,
            interpolant_time=args.interpolant_time,
            ground_truth_interpolant_states=gt_interp,
        )["w2_normalised"].mean()))
        if v is not None:
            _add(g_name, "mean_interpolant_matching", v)
        _free()

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\n[all_eval] wrote {args.output_csv}")
    print(df.to_string(index=False))

    # --- lollipop plot ----------------------------------------------
    metric_direction = {
        "mean_clonal_preservation": False,
        "mean_marginal_preservation": False,
        "germ_layer_crossing": False,
        "curated_transition_fraction": True,
        "linkage_correlation": True,
        "mean_interpolant_matching": False,
    }
    model_order = list(cfgs.keys()) + list(genots.keys())
    palette = ["#84B5B2", "#5779A3", "#D1605E", "#E59244",
               "#8E6BA8", "#5EA877", "#D6A04F", "#6C7A89"]
    colors = {m: palette[i % len(palette)] for i, m in enumerate(model_order)}
    fig = plot_lollipop(df, model_order, colors, metric_direction)
    plt.tight_layout()
    fig.savefig(args.output_png, dpi=150, bbox_inches="tight")
    print(f"[all_eval] wrote {args.output_png}")


if __name__ == "__main__":
    main()
