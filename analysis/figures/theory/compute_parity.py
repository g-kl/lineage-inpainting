"""Exact binary Bernstein parity checks for slice features (compute half).

The standard order-``K`` parity construction draws ``K-1`` independent
Rademachers and sets the final coordinate to their product. Any strict subset
of the ``K`` variables is independent, but the full ``K``-tuple is not.

This script uses one shared order-6 lollipop tree shape for every test:

    n0 -> n1 -> ... -> n5
                    |-> l1
                    |-> ...
                    |-> l5

and embeds a lower-order parity in the location matched to each feature:

* path: place an order-``r`` parity among the first ``r`` chain nodes;
* common ancestor: place an order-``r`` parity among the terminal chain node and
  a uniformly chosen set of ``r-1`` children.

It then scores feature order ``q`` against parity order ``r``. By the parity
property, ``q < r`` should be invisible; once the full parity tuple is exposed,
the parity constraint enters the induced feature law. Every state is exactly
Rademacher-valued, with no continuous perturbation after the parity construction.

Run from the repository root with

    python -m analysis.figures.theory.compute_parity [--quick] [--force]

which writes parity_raw.csv, parity_summary.csv and parity_config.json to
numerical_results/ (numerical_results/quick/ for the smoke run) and keeps a
pickle cache of the raw scores beside them.  plot_parity.py reads the summary.
The full grid scores 4096 trees per population over 32 replicates and takes a
long time on a CPU; ``--quick`` runs 256 trees over 2 replicates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from aligners.slice import CommonAncestorFeature, PathFeature
from aligners.stratified import StratifiedFeatureAligner
from datasets.tree_record import collate_subtrees

from analysis.figures.theory import style as S

MAX_ORDER = 6
PARITY_ORDERS = [3, 4, 5, 6]
FEATURE_ORDERS = [2, 3, 4, 5, 6]
NOISE_SIGMA = 0.0
N_TREES = 4096
N_SAMPLES = 1
REPS_BY_FEATURE = {"path": 32, "ca": 32}
QUICK_N_TREES = 256
QUICK_REPS_BY_FEATURE = {"path": 2, "ca": 2}

FEATS = ("path", "ca")
PLACEMENTS = ("chain", "fan")
FEAT_SEED_OFFSET = {"path": 0, "ca": 1}
FEAT_PLACEMENT = {"path": "chain", "ca": "fan"}


def parity_values(
    rng: np.random.Generator,
    order: int,
    *,
    structured: bool,
    noise_sigma: float,
) -> np.ndarray:
    vals = rng.choice([-1.0, 1.0], size=order)
    if structured:
        vals[-1] = float(np.prod(vals[:-1]))
    if noise_sigma > 0:
        vals = vals + rng.normal(scale=noise_sigma, size=order)
    return vals.astype(np.float32)


def iid_values(
    rng: np.random.Generator,
    n: int,
    *,
    noise_sigma: float,
) -> np.ndarray:
    vals = rng.choice([-1.0, 1.0], size=n)
    if noise_sigma > 0:
        vals = vals + rng.normal(scale=noise_sigma, size=n)
    return vals.astype(np.float32)


def _build_tree(
    seed: int,
    *,
    placement: str,
    parity_order: int,
    structured: bool,
    noise_sigma: float,
) -> tuple[nx.DiGraph, dict[str, np.ndarray]]:
    if placement not in {"chain", "fan"}:
        raise ValueError(f"unknown parity placement {placement!r}")
    rng = np.random.default_rng(seed)
    chain_vals = iid_values(
        rng, MAX_ORDER, noise_sigma=noise_sigma,
    ).reshape(MAX_ORDER, 1)
    leaf_vals = iid_values(
        rng, MAX_ORDER - 1, noise_sigma=noise_sigma,
    ).reshape(MAX_ORDER - 1, 1)

    if structured and placement == "chain":
        parity_vals = parity_values(
            rng, parity_order, structured=True, noise_sigma=noise_sigma,
        )
        chain_vals[:parity_order, 0] = parity_vals
    elif structured and placement == "fan":
        parity_vals = parity_values(
            rng, parity_order, structured=True, noise_sigma=noise_sigma,
        )
        selected = rng.choice(
            MAX_ORDER - 1, size=parity_order - 1, replace=False,
        )
        chain_vals[-1, :] = parity_vals[0]
        leaf_vals[selected, 0] = parity_vals[1:]

    g = nx.DiGraph()
    states: dict[str, np.ndarray] = {}
    prev = None
    for i, value in enumerate(chain_vals):
        node = f"n{i}"
        g.add_node(node, time=float(i))
        if prev is not None:
            g.add_edge(prev, node)
        states[node] = np.asarray(value, dtype=np.float32)
        prev = node

    terminal = f"n{MAX_ORDER - 1}"
    for i, value in enumerate(leaf_vals, start=1):
        node = f"l{i}"
        g.add_node(node, time=float(MAX_ORDER))
        g.add_edge(terminal, node)
        states[node] = np.asarray(value, dtype=np.float32)
    return g, states


def _pool(
    seed: int,
    *,
    placement: str,
    parity_order: int,
    structured: bool,
    noise_sigma: float,
    n: int = N_TREES,
):
    rng = np.random.default_rng(seed)
    subtrees, leaf_states = [], {}
    for i in range(n):
        g, states = _build_tree(
            int(rng.integers(1 << 31)),
            placement=placement,
            parity_order=parity_order,
            structured=structured,
            noise_sigma=noise_sigma,
        )
        prefix = f"p{seed}_{i}_"
        rg = nx.relabel_nodes(g, lambda x, prefix=prefix: prefix + x)
        subtrees.append({"graph": rg, "root": prefix + "n0"})
        for node, value in states.items():
            leaf_states[prefix + node] = torch.tensor(value)
    return subtrees, leaf_states


def _collate(subtrees, leaf_states):
    batch = collate_subtrees(
        subtrees,
        leaf_states,
        device="cpu",
        max_nodes=64,
        randomize_child_index=True,
    )
    return batch, batch.observed_states


def _eds(A, B, *, feature: str, feature_order: int, torch_seed: int):
    torch.manual_seed(torch_seed)
    if feature == "path":
        feature_obj = PathFeature(
            time_bins=[float(i) for i in range(feature_order)],
            n_samples=N_SAMPLES,
            direction="up",
        )
    elif feature == "ca":
        feature_obj = CommonAncestorFeature(
            time_bins=[float(MAX_ORDER - 1), float(MAX_ORDER)],
            n_descendants=feature_order - 1,
            n_samples=N_SAMPLES,
            direction="up",
        )
    else:
        raise ValueError(f"unknown feature {feature!r}")
    (batch_a, states_a), (batch_b, states_b) = A, B
    aligner = StratifiedFeatureAligner([feature_obj], loss="energy_direct")
    with torch.no_grad():
        return float(aligner(batch_a, states_a, batch_b, states_b))


def compute(*, n_trees: int = N_TREES, reps_by_feature: dict = REPS_BY_FEATURE):
    raw = []
    max_reps = max(reps_by_feature.values())
    for rep in range(max_reps):
        active_features = tuple(
            feature for feature in FEATS
            if rep < reps_by_feature[feature]
        )
        ref = _collate(*_pool(
            10_000 + rep,
            n=n_trees,
            placement="chain",
            parity_order=MAX_ORDER,
            structured=False,
            noise_sigma=NOISE_SIGMA,
        ))
        null = _collate(*_pool(
            20_000 + rep,
            n=n_trees,
            placement="chain",
            parity_order=MAX_ORDER,
            structured=False,
            noise_sigma=NOISE_SIGMA,
        ))
        floors = {
            (feature, feature_order): _eds(
                ref,
                null,
                feature=feature,
                feature_order=feature_order,
                torch_seed=40_000
                + 1000 * rep
                + 10 * feature_order
                + FEAT_SEED_OFFSET[feature],
            )
            for feature in active_features
            for feature_order in FEATURE_ORDERS
        }

        for parity_order in PARITY_ORDERS:
            structured_by_placement = {
                placement: _collate(*_pool(
                    30_000
                    + 1000 * rep
                    + 10 * parity_order
                    + (0 if placement == "chain" else 1),
                    n=n_trees,
                    placement=placement,
                    parity_order=parity_order,
                    structured=True,
                    noise_sigma=NOISE_SIGMA,
                ))
                for placement in ("chain", "fan")
            }
            for placement in PLACEMENTS:
                structured = structured_by_placement[placement]
                for feature in active_features:
                    for feature_order in FEATURE_ORDERS:
                        ed = _eds(
                            ref,
                            structured,
                            feature=feature,
                            feature_order=feature_order,
                            torch_seed=50_000
                            + 1000 * rep
                            + 100 * parity_order
                            + 20 * (0 if placement == "chain" else 1)
                            + 10 * feature_order
                            + FEAT_SEED_OFFSET[feature],
                        )
                        floor = floors[(feature, feature_order)]
                        raw.append({
                            "rep": rep,
                            "placement": placement,
                            "parity_order": parity_order,
                            "feature_order": feature_order,
                            "feature": feature,
                            "ed": ed,
                            "floor": floor,
                            "excess": ed - floor,
                        })
        print(f"  rep {rep + 1}/{max_reps} done", flush=True)
    return raw, summarize(raw)


def summarize(raw):
    grouped: dict[tuple[str, str, int, int], list[float]] = {}
    for row in raw:
        key = (
            row["placement"],
            row["feature"],
            row["parity_order"],
            row["feature_order"],
        )
        grouped.setdefault(key, []).append(float(row["excess"]))

    summary = {}
    for key, vals in grouped.items():
        arr = np.array(vals, dtype=float)
        sem = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        summary[key] = (float(arr.mean()), float(sem))
    return summary


def _write_csv(raw, summary, out_dir: Path):
    raw_path = out_dir / "parity_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rep", "placement", "feature", "parity_order", "feature_order",
                "ed", "floor", "excess",
            ],
        )
        writer.writeheader()
        writer.writerows(raw)

    summary_path = out_dir / "parity_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "placement", "feature", "parity_order", "feature_order", "mean",
                "sem",
            ],
        )
        writer.writeheader()
        for (
            placement, feature, parity_order, feature_order
        ), (mean, sem) in sorted(summary.items()):
            writer.writerow({
                "placement": placement,
                "feature": feature,
                "parity_order": parity_order,
                "feature_order": feature_order,
                "mean": mean,
                "sem": sem,
            })
    return raw_path, summary_path


def _write_config(
    summary_path: Path, out_dir: Path, *, n_trees: int, reps_by_feature: dict,
) -> Path:
    path = out_dir / "parity_config.json"
    with path.open("w") as handle:
        json.dump({
            "state_law": "exact binary parity",
            "noise_sigma": NOISE_SIGMA,
            "n_trees": n_trees,
            "n_samples_per_tree": N_SAMPLES,
            "parity_orders": PARITY_ORDERS,
            "feature_orders": FEATURE_ORDERS,
            "replicates_by_feature": reps_by_feature,
            "summary_file": summary_path.name,
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        }, handle, indent=2, sort_keys=True)
    return path


def output_dir(quick: bool) -> Path:
    """Where this benchmark's numerical results live."""
    return S.NR / "quick" if quick else S.NR


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="small smoke benchmark")
    parser.add_argument("--force", action="store_true", help="ignore a matching cache")
    args = parser.parse_args()
    n_trees = QUICK_N_TREES if args.quick else N_TREES
    reps_by_feature = QUICK_REPS_BY_FEATURE if args.quick else REPS_BY_FEATURE
    out_dir = output_dir(args.quick)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"parity_cache_n{n_trees}_r{max(reps_by_feature.values())}.pkl"
    if cache.exists() and not args.force:
        with cache.open("rb") as handle:
            raw, summary = pickle.load(handle)
        print("loaded", cache)
    else:
        raw, summary = compute(n_trees=n_trees, reps_by_feature=reps_by_feature)
        with cache.open("wb") as handle:
            pickle.dump((raw, summary), handle)
        print("wrote", cache)
    raw_path, summary_path = _write_csv(raw, summary, out_dir)
    config_path = _write_config(
        summary_path, out_dir, n_trees=n_trees, reps_by_feature=reps_by_feature,
    )

    print("\ntarget-feature summary (mean+-sem):")
    for feature in FEATS:
        placement = FEAT_PLACEMENT[feature]
        print(f"  {feature}")
        for parity_order in PARITY_ORDERS:
            vals = []
            for feature_order in FEATURE_ORDERS:
                mean, sem = summary[(placement, feature, parity_order, feature_order)]
                vals.append(f"q={feature_order}:{mean:+.4f}+-{sem:.4f}")
            print(f"    r={parity_order}: " + ", ".join(vals))
    print("\nwrote", out_dir)
    print("raw", raw_path)
    print("summary", summary_path)
    print("config", config_path)


if __name__ == "__main__":
    main()
