"""Train + evaluate a ConditionalCountSampler on a run's assigned-cell table.

Each internal node -> its assigned observed cell's embedding; for every horizon
delta in DELTAS with ``time + delta <= tree_max``, one example:
``cond = {state, time, delta}``, target = ``count_at(time + delta)`` (active
descendants at the horizon).

The train/test split is **by clone (whole subtree)**, so held-out metrics measure
generalization to lineages the model never saw. We report, per horizon delta, on
held-out clones:

  * D2 (Tweedie) of the sampled mean vs observed (>0 beats the global-mean baseline);
  * per-cell-type R^2 of the mean count;
  * central-80% sample-interval coverage (~0.8 if the distribution is calibrated).

Usage (on a GPU/large-RAM node):
    python scripts/train_energy_count_sampler.py \
        --output_dir outputs/mouse_embryo_..._80f6eafa
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import d2_tweedie_score, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from extenders.count import ConditionalCountSampler   # noqa: E402


def count_at(tree, root_node, target_t):
    """# of active lineages below ``root_node`` at time ``target_t``."""
    n, stack = 0, [root_node]
    while stack:
        u = stack.pop()
        for v in tree.successors(u):
            if tree.nodes[v]["time"] >= target_t:
                n += 1
            else:
                stack.append(v)
    return n


def build_table(dataset, assignments, *, deltas, max_out_degree, exclude):
    """Return state (N, dim), time (N,), delta (N,), y (N,), ct (N,), clone (N,)."""
    emb = np.asarray(dataset.reference_features, dtype=np.float32)
    obs_names = list(map(str, dataset.states.obs_names))
    obs_idx = {n: i for i, n in enumerate(obs_names)}
    obs = dataset.states.obs
    if "cell_subtype" in obs.columns:
        ct_arr = obs["cell_subtype"].astype(str).values
    elif "cell_type" in obs.columns:
        ct_arr = obs["cell_type"].astype(str).values
    else:
        ct_arr = np.array(["na"] * len(emb))

    state, time, delta, y, ct, clone_of = [], [], [], [], [], []
    for clone, node_map in assignments.items():
        if clone == "nn_distances" or clone not in dataset.trees:
            continue
        if any(s in clone for s in exclude):
            continue
        tree = dataset.trees[clone]
        tmax = max(tree.nodes[n]["time"] for n in tree.nodes)
        for node, cell in node_map.items():
            if cell == "unknown" or cell not in obs_idx:
                continue
            if tree.out_degree(node) > max_out_degree:
                continue
            t = float(tree.nodes[node]["time"])
            i = obs_idx[cell]
            for d in deltas:
                if t + d > tmax:
                    continue
                state.append(emb[i])
                time.append(t)
                delta.append(d)
                y.append(count_at(tree, node, t + d))
                ct.append(ct_arr[i])
                clone_of.append(clone)
    return (np.asarray(state, np.float32), np.asarray(time, np.float32),
            np.asarray(delta, np.float32), np.asarray(y, np.float32),
            np.asarray(ct), np.asarray(clone_of))


def split_by_clone(clone_of, test_frac, seed):
    clones = np.array(sorted(set(clone_of.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(clones)
    n_test = max(1, int(round(test_frac * len(clones))))
    test = set(clones[:n_test].tolist())
    return np.array([c in test for c in clone_of]), sorted(test)


def evaluate(sampler, cond, y, ct, *, n_samples, seed):
    rng = np.random.default_rng(seed)
    draws = sampler.sample_batch(cond, rng, n_samples=n_samples)   # (N, S)
    mean_pred = draws.mean(axis=1)
    lo, hi = np.percentile(draws, 10, axis=1), np.percentile(draws, 90, axis=1)
    d2 = d2_tweedie_score(y, np.clip(mean_pred, 1e-6, None), power=1)
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    print(f"    rows {len(y)} | obs mean {y.mean():.3f} | pred mean {mean_pred.mean():.3f}")
    print(f"    D2 {d2:.3f} (0=baseline) | 80%-coverage {coverage:.3f} (~0.80 ideal)")
    import pandas as pd
    agg = (pd.DataFrame({"ct": ct, "y": y, "p": mean_pred})
           .groupby("ct").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size")))
    agg = agg[agg["n"] >= 5]
    if len(agg):
        print(f"    per-cell-type R2 ({len(agg)} types): {r2_score(agg['y'], agg['p']):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--max_out_degree", type=int, default=8)
    ap.add_argument("--exclude", nargs="*", default=["E10.0"])
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--noise_dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--train_samples", type=int, default=8)
    ap.add_argument("--eval_samples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--max_count", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--assignments", default=None,
                    help="explicit assignments_nn_*.pkl path (relative to repo root "
                         "or absolute); default picks the newest by mtime")
    ap.add_argument("--save_name", default="extender_count_energy.pt")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("loading dataset (this needs a lot of RAM)...")
    cfg = OmegaConf.load(out_dir / "config.yaml")
    dataset = hydra.utils.instantiate(cfg.dataset)
    if args.assignments:
        ap_path = Path(args.assignments)
        if not ap_path.is_absolute():
            ap_path = ROOT / ap_path
    else:
        ap_path = max(out_dir.glob("assignments_nn_*.pkl"), key=lambda p: p.stat().st_mtime)
    with open(ap_path, "rb") as f:
        assignments = pickle.load(f)
    print(f"assignments: {ap_path.name}")

    state, time, delta, y, ct, clone_of = build_table(
        dataset, assignments, deltas=args.deltas,
        max_out_degree=args.max_out_degree, exclude=tuple(args.exclude),
    )
    print(f"table: {len(y)} rows over deltas {args.deltas}, state dim {state.shape[1]}, "
          f"{len(set(clone_of.tolist()))} clones; y mean={y.mean():.2f} max={int(y.max())}")

    test_mask, test_clones = split_by_clone(clone_of, args.test_frac, args.seed)
    tr, te = ~test_mask, test_mask
    print(f"held out {len(test_clones)} clones ({te.sum()} rows); training on {tr.sum()} rows")

    cond = {"state": state, "time": time, "delta": delta}
    sampler = ConditionalCountSampler(
        feat_dim=state.shape[1] + 2, noise_dim=args.noise_dim, hidden=args.hidden,
        n_layers=args.n_layers, min_count=args.min_count, max_count=args.max_count,
        device=device,
    )
    print("training (energy score)...")
    cond_tr = {k: v[tr] for k, v in cond.items()}
    sampler.fit(cond_tr, y[tr], n_samples=args.train_samples, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, seed=args.seed)

    for label, mask in [("HELD-OUT clones", te), ("train clones", tr)]:
        print(f"\n=== {label} ===")
        for d in args.deltas:
            dm = mask & (delta == d)
            if dm.sum() == 0:
                continue
            print(f"  delta={d}:")
            evaluate(sampler, {k: v[dm] for k, v in cond.items()},
                     y[dm], ct[dm], n_samples=args.eval_samples, seed=args.seed)

    save_path = out_dir / args.save_name
    sampler.save(save_path)
    print(f"\nsaved -> {save_path}")


if __name__ == "__main__":
    main()
