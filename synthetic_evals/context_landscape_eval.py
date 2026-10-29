#!/usr/bin/env python
#SBATCH --job-name=context_landscape
#SBATCH --output=logs/context_landscape_%j.log
#SBATCH --error=logs/context_landscape_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=bates
"""Real-cell context landscape: a context-required fate motif, scored by
ancestral-imputation MSE.

Sibling of ``crossing_landscape_eval``: where that one sweeps how strongly two
lineages cross, this one asks whether an ancestor can be named at all when its
own descendants are uninformative and only a cousin pair resolves it.

Two motifs share an ambiguous child type B (datasets: CellTypeFateDataset)::

    A -> {A, A} | {B, B}          # motif A
    C -> {C, C} | {B, B}          # motif C

Each tree is a clone of one motif: both of its sister progenitors carry that
motif's type, and exactly one of them differentiates into B (asymmetric
division). A {B,B} pair's progenitor is therefore only nameable from its cousin
pair -- the sister lineage that kept its identity.

Samples random (root, A, B, C) draws, trains our pool/split plus baselines
(FM / GENOT-Sinkhorn / GENOT-FGW / Waddington-OT / moslin) per draw, and writes
the single metric of interest to landscape.csv / results.pkl:

  ``mse``      -- reverse per-parent squared error of the imputed ancestor at
                  t_src.
  ``type_acc`` -- fraction of imputed ancestors landing nearest the *correct*
                  progenitor-type centroid: did the method name the right
                  progenitor, ignoring where inside that type it landed.
                  Chance is 0.5.

Both are also reported split by whether the progenitor needed the cousin
(``*_ambiguous`` / ``*_unambiguous``).

Our model sees the whole tree (both cousin pairs) and imputes both hidden
progenitors at once. The baselines are cell-level maps, so they are scored one
descendant at a time: push a single leaf back t_tgt -> t_src and measure the
squared error to *its own* ancestor.

Two reference rows bracket the range, both **sample-emitting** so they are
directly comparable to a model that returns a plausible cell rather than a
conditional mean (against a specific ancestor cell, emitting a sample costs 2x
the variance of predicting the mean):
  ``marg``   -- emit a random real cell from the pooled t_src marginal (no
                lineage information at all).
  ``oracle`` -- emit a random real cell of the *correct* progenitor type. Since
                the mid cell and its children are independent draws from their
                type pools, this is the floor for any sample-emitting method.

Run (on a GPU node, or `sbatch` this file):
    PYTHONPATH=. python synthetic_evals/context_landscape_eval.py [--quick]
"""

from __future__ import annotations

import os
import sys

# Run from the repo root (resolved from SLURM_SUBMIT_DIR or this file's location).
base_dir = os.path.abspath(os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import argparse
import functools
import logging
import pickle

import numpy as np
import pandas as pd
import torch

from synthetic_evals.baselines import (
    genot_sinkhorn,
    marginal_fm,
    moscot_ot,
    registry,
)
from synthetic_evals.core import inference as inf
from synthetic_evals.core.hydra_io import (
    DEVICE, ROOT, load_pool_split, run_pool_split_multirun,
)
from synthetic_evals.cell_type_crossing_eval import EXPERIMENT, _fgw_bank_from_dataset
from datasets.cell_type_crossing import load_cell_pools
from datasets.cell_type_fate import CellTypeFateDataset
from datasets.tree_record import collate_subtrees

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("context_landscape")


## CONFIG ##

DATA_DIR = "../data/embryos"
STATES_FILENAME = "qiu24_integrated_hvgs_and_scvi.h5ad"
EMBEDDING_KEY = "X_scvi"
TYPE_INFO = "final_type_info.csv"

METHODS = ("ours", "gs", "fm", "gf", "wot", "moslin")
TIMES = (8.5, 9.0, 9.5)              # (t_root, t_src, t_tgt); fate split at t_src->t_tgt
MIN_CELLS = 50                      # a (subtype, time) slot needs at least this many real cells
N_SAMPLES = 10                      # random (root, A, B, C) cell-type draws
LAUNCHER = "joblib"                 # Hydra launcher for the pool-split multirun

DATASET_TARGET = "datasets.cell_type_fate.CellTypeFateDataset"
CFG_GROUP = "context_landscape"      # generated dataset configs -> config/dataset/<group>/
OUT_DIR = ROOT / "synthetic_evals" / "context_landscape"

# Train/load on the full root->2 mids->4 leaves tree (the experiment default
# target_gap=0.5 only pairs adjacent layers). Changes the config hash, so
# training and the eval-time load_pool_split must both pass it.
TRAIN_OVERRIDES = ["record_sampler.target_gap=1.0"]


# --------------------------------------------------------------------------
#  Random cell-type draws for (root, A, B, C)
# --------------------------------------------------------------------------


def load_present(subtypes):
    """``present[t]`` (subtypes with >= MIN_CELLS at t) and the shared cell pools."""
    pools = load_cell_pools(
        DATA_DIR, STATES_FILENAME, EMBEDDING_KEY,
        {t: list(subtypes) for t in TIMES})
    present = {t: set() for t in TIMES}
    for (name, t), X in pools.items():
        if X.shape[0] >= MIN_CELLS:
            present[t].add(name)
    return pools, present


def sample_motif(present, seed):
    """One random draw of four distinct types: A/C present at t_src & t_tgt,
    shared B at t_tgt, root at t_root."""
    t_root, t_src, t_tgt = TIMES
    rng = np.random.default_rng(seed)
    prog = sorted(present[t_src] & present[t_tgt])
    if len(prog) < 2:
        raise RuntimeError(f"need >= 2 subtypes present at both {t_src} and {t_tgt}")
    A, C = (str(x) for x in rng.choice(prog, 2, replace=False))
    B = str(rng.choice(sorted(present[t_tgt] - {A, C})))
    root = str(rng.choice(sorted(present[t_root] - {A, B, C})))
    return {"root_type": root, "A": A, "C": C, "B": B, "seed": seed}


# --------------------------------------------------------------------------
#  Per-motif dataset config + dataset
# --------------------------------------------------------------------------


def cfg_name(i):
    return f"{CFG_GROUP}/cfg{i:02d}"


def parse_variants(items):
    """``["label=experiment", ...]`` -> dict, defaulting to the single "ours" model."""
    if not items:
        return {"ours": EXPERIMENT}
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--variants entries must be label=experiment, got {it!r}")
        label, exp = it.split("=", 1)
        out[label] = exp
    return out


def write_dataset_yaml(i, motif, *, n_trees, n_marginal):
    """Emit config/dataset/<group>/cfgNN.yaml for this motif (so it gets a hash)."""
    d = ROOT / "config" / "dataset" / CFG_GROUP
    d.mkdir(parents=True, exist_ok=True)
    t_root, t_src, t_tgt = TIMES
    text = (
        f"# auto-generated by synthetic_evals/context_landscape_eval.py -- do not hand-edit.\n"
        f"_target_: {DATASET_TARGET}\n"
        f"data_dir: {DATA_DIR}\n"
        f"states_filename: {STATES_FILENAME}\n"
        f"embedding_key: {EMBEDDING_KEY}\n"
        f"times: [{t_root}, {t_src}, {t_tgt}]\n"
        f'root_type: "{motif["root_type"]}"\n'
        f'parent_types: ["{motif["A"]}", "{motif["C"]}"]\n'
        f'shared_type: "{motif["B"]}"\n'
        f"n_trees: {n_trees}\n"
        f"n_marginal: {n_marginal}\n"
        f"max_size: 128\n"
        f"seed: 0\n"
    )
    (d / f"cfg{i:02d}.yaml").write_text(text)


def build_dataset(motif, pools, *, n_trees, n_marginal):
    return CellTypeFateDataset(
        data_dir=DATA_DIR, states_filename=STATES_FILENAME, embedding_key=EMBEDDING_KEY,
        times=TIMES, root_type=motif["root_type"],
        parent_types=(motif["A"], motif["C"]), shared_type=motif["B"],
        seed=0, cell_pools=pools, n_trees=n_trees, n_marginal=n_marginal,
    )


# --------------------------------------------------------------------------
#  Full-tree (cousin-context) inference for our model
# --------------------------------------------------------------------------


def build_tree_subtrees(ds):
    """One full-tree subtree per tree (root -> 2 sister progenitors -> 4 leaves).

    Both progenitors carry the tree's motif (``group`` 0=A, 1=C); ``ambiguous``
    flags the one that differentiated into B, whose identity is recoverable only
    from its cousin pair. ``mids`` / ``leaves`` are listed mA-then-mB, and each
    mid is followed by its own two leaves.
    """
    motif_idx = {ds.parent_types[0]: 0, ds.parent_types[1]: 1}
    out = []
    for tname, g in ds.records["leaf"].trees.items():
        mids, leaves = [], []
        for k, mtag in enumerate(("mA", "mB")):
            mid = f"{tname}_{mtag}"
            mids.append((mid, k))
            for c in g.successors(mid):
                leaves.append((c, k))
        out.append({
            "graph": g, "root": f"{tname}_r", "mids": mids, "leaves": leaves,
            "mid_states": {m: ds.gt_node_states[m] for m, _ in mids},
            "leaf_states": {c: ds.gt_node_states[c] for c, _ in leaves},
            "group": motif_idx[ds.tree_motif[tname]],
            "ambiguous": [mtag == ds.tree_shared_tag[tname]
                          for mtag in ("mA", "mB")],
        })
    return out


def flatten_truth(tree_subs):
    """Per-progenitor and per-leaf ground truth in ``build_tree_subtrees`` order.

    Returns ``parent_true`` (P, dim), ``parent_group`` (P,), ``parent_ambig``
    (P,) bool, ``x_tgt`` (M, dim) the observed leaves, and ``child_to_parent``
    (M,) mapping each leaf to its progenitor row.
    """
    parent_true, parent_group, parent_ambig = [], [], []
    x_tgt, child_to_parent = [], []
    for s in tree_subs:
        for k, (m, _) in enumerate(s["mids"]):
            p = len(parent_true)
            parent_true.append(s["mid_states"][m].cpu().numpy())
            parent_group.append(s["group"])
            parent_ambig.append(s["ambiguous"][k])
            for c, grp in s["leaves"]:
                if grp == k:
                    x_tgt.append(s["leaf_states"][c].cpu().numpy())
                    child_to_parent.append(p)
    return (np.stack(parent_true).astype(np.float32), np.array(parent_group),
            np.array(parent_ambig), np.stack(x_tgt).astype(np.float32),
            np.array(child_to_parent))


def _node_index(batch):
    return [{nm: j for j, nm in enumerate(batch.node_names[i])}
            for i in range(len(batch.node_names))]


@torch.no_grad()
def pool_with_context(pool, tree_subs, *, n_samples, n_flow_steps):
    """Reverse: observe the four leaves and predict the two mids over the whole tree.

    Returns (n_samples, 2 * len(tree_subs), dim), per tree mA then mB (the
    ``build_eval_subtrees`` order, so it aligns with ``parent_true`` / ``parent_group``).
    """
    dim = int(getattr(pool, "dim"))
    subtree_dicts = [{"graph": s["graph"], "root": s["root"]} for s in tree_subs]
    leaf_obs = {name: s["leaf_states"][name].to(DEVICE)
                for s in tree_subs for name, _ in s["leaves"]}
    batch = collate_subtrees(subtree_dicts, leaf_obs, device=DEVICE,
                             randomize_child_index=False)
    idx = _node_index(batch)
    takes = inf.accepts_n_steps(pool.pool_batch)
    pos = [(i, idx[i][m]) for i, s in enumerate(tree_subs) for m, _ in s["mids"]]
    out = np.zeros((n_samples, len(pos), dim), dtype=np.float32)
    for si in range(n_samples):
        pooled = (pool.pool_batch(batch, n_steps=n_flow_steps) if takes
                  else pool.pool_batch(batch)).detach().cpu().numpy()
        for k, (i, j) in enumerate(pos):
            out[si, k] = pooled[i, j]
    return out


# --------------------------------------------------------------------------
#  Cell-level reverse inference for the baselines
# --------------------------------------------------------------------------


def baseline_reverse_cells(bl, pkg, x_tgt, t_src, t_tgt, *, n_samples, n_flow_steps):
    """Push each descendant back on its own: ``x_tgt`` (M, dim) at ``t_tgt`` ->
    (n_samples, M, dim) predicted ancestors at ``t_src``.

    The baselines are cell-level transport maps with no notion of a sibling pair, so
    every leaf is scored against its own ancestor independently.
    """
    return inf.forward_samples(
        bl, pkg, x_tgt, t_tgt, t_src,
        n_samples=n_samples, n_flow_steps=n_flow_steps,
    ).astype(np.float32)


# --------------------------------------------------------------------------
#  Metric
# --------------------------------------------------------------------------


def _grouped_mse(pred, true, groups, *, per_sample=False):
    """Mean over progenitor groups of the per-row squared error to the true
    ancestor.

    ``pred`` is (S, R, dim) samples (or (R, dim) for a point predictor);
    ``true`` is (R, dim) the ancestor each row is scored against; ``groups`` is
    (R,) the progenitor group (0=A, 1=C) of each row.

    By default the samples are averaged first, i.e. the posterior mean is
    scored -- the right reduction for a method that could in principle average
    its way to a mean. ``per_sample`` instead scores each draw and averages the
    errors, which is what a sample-emitting reference costs; the
    extra draws then only reduce Monte-Carlo noise, not the error itself.
    """
    pred = np.asarray(pred, np.float32)
    if pred.ndim == 2:
        pred = pred[None]
    se = (((pred - true) ** 2).sum(-1).mean(0) if per_sample
          else ((pred.mean(0) - true) ** 2).sum(-1))
    errs = [float(se[groups == g].mean()) for g in (0, 1) if (groups == g).any()]
    return float(np.mean(errs)) if errs else float("nan")


def _grouped_type_accuracy(pred, true_groups, type_means, *, per_sample=False):
    """Mean over groups of the fraction of rows landing nearest the *correct*
    progenitor-type centroid -- i.e. did the method name the right progenitor,
    ignoring where inside that type it put the cell.

    Same arguments and same sample reduction as :func:`_grouped_mse`, so 0.5 is
    chance (``marg``) and 1.0 is perfect (``oracle``).
    """
    pred = np.asarray(pred, np.float32)
    if pred.ndim == 2:
        pred = pred[None]
    if not per_sample:
        pred = pred.mean(0, keepdims=True)
    lab = ((pred[..., None, :] - type_means) ** 2).sum(-1).argmin(-1)    # (S, R)
    hit = (lab == true_groups).mean(0)                                   # (R,)
    accs = [float(hit[true_groups == g].mean())
            for g in (0, 1) if (true_groups == g).any()]
    return float(np.mean(accs)) if accs else float("nan")


# --------------------------------------------------------------------------
#  Per-motif evaluation
# --------------------------------------------------------------------------


def eval_one_motif(motif, dataset_name, pools, methods, *, n_trees, n_marginal,
                   fm_steps, genot_steps, n_samples, n_flow_steps,
                   n_ref_draws=64, variants=None):
    """Ancestral-imputation MSE per method on one motif, grouped by progenitor
    (0=A, 1=C). Trains baselines fresh; loads our full-tree checkpoint; scores on
    the training draw.
    """
    methods = set(methods)
    variants = dict(variants or {"ours": EXPERIMENT})
    _, t_src, t_tgt = TIMES
    train_ds = build_dataset(motif, pools, n_trees=n_trees, n_marginal=n_marginal)
    dim = int(train_ds.x_dim)
    pairs_rev = [(t_tgt, t_src)]

    # ---- baselines: trained fresh on the marginals, reverse only ----
    @functools.cache
    def fgw_bank():
        return _fgw_bank_from_dataset(train_ds)

    @functools.cache
    def moscot_bank():
        return moscot_ot.MoscotBank.from_fgw_bank(fgw_bank())

    banks = {
        "fm": lambda: marginal_fm.TimeBank.from_dataset(train_ds),
        "gs": lambda: genot_sinkhorn.GenotBank.from_dataset(train_ds),
        "gf": fgw_bank,
        "wot": moscot_bank,
        "moslin": moscot_bank,
    }
    train_kwargs = {
        "fm": dict(n_steps=fm_steps, pairs=pairs_rev, device=DEVICE, seed=1),
        "gs": dict(n_steps=genot_steps, pairs=pairs_rev, seed=1,
                   coupling_mode="precomputed"),
        "gf": dict(n_steps=genot_steps, pairs=pairs_rev, seed=3,
                   coupling_mode="precomputed", n_tree_pairings=1),
        "wot": dict(pairs=pairs_rev),
        "moslin": dict(pairs=pairs_rev),
    }
    models = {
        name: registry.get(name).train(banks[name](), dim=dim, **train_kwargs[name])
        for name in registry.names() if name in methods
    }

    # ---- our trained pool (TRAIN_OVERRIDES to match the checkpoint hash) ----
    pools_by_variant = ({label: load_pool_split(exp, dataset_name, TRAIN_OVERRIDES)[0]
                         for label, exp in variants.items()}
                        if "ours" in methods else {})

    # ---- eval draw == the training draw ----
    tree_subs = build_tree_subtrees(train_ds)    # full tree: our context inference
    (parent_true, parent_group, parent_ambig,
     x_tgt, child_to_parent) = flatten_truth(tree_subs)

    # Scoring rows: our model predicts one ancestor per progenitor; the
    # baselines predict one per descendant cell, scored against that cell's own
    # ancestor. ``ambig`` selects the {B,B} progenitors, whose identity is only
    # recoverable from the cousin pair.
    per_parent = (parent_true, parent_group, parent_ambig)
    per_cell = (parent_true[child_to_parent], parent_group[child_to_parent],
                parent_ambig[child_to_parent])

    # Progenitor-type centroids at t_src, for the "did it name the right
    # progenitor" readout.
    type_means = np.stack([pools[(pt, t_src)].mean(0)
                           for pt in train_ds.parent_types]).astype(np.float32)

    def score(pred, rows, **kw):
        true, group, ambig = rows
        sub = {tag: (pred[..., sel, :], true[sel], group[sel])
               for tag, sel in (("", slice(None)), ("_ambiguous", ambig),
                                ("_unambiguous", ~ambig))}
        out = {f"mse{tag}": _grouped_mse(p, t, g, **kw)
               for tag, (p, t, g) in sub.items()}
        out.update({f"type_acc{tag}": _grouped_type_accuracy(p, g, type_means, **kw)
                    for tag, (p, _t, g) in sub.items()})
        return out

    rev = {}
    # OURS: whole tree in, both hidden progenitors out (one prediction per parent).
    for label, pool in pools_by_variant.items():
        rev[label] = score(pool_with_context(
            pool, tree_subs, n_samples=n_samples, n_flow_steps=n_flow_steps), per_parent)
    # BASELINES: one descendant in, its own ancestor out.
    for name, pkg in models.items():
        rev[name] = score(baseline_reverse_cells(
            registry.get(name), pkg, x_tgt, t_src, t_tgt,
            n_samples=n_samples, n_flow_steps=n_flow_steps), per_cell)

    # ---- reference rows: emit a real t_src cell, so they cost the same
    # sampling penalty our model pays. `oracle` knows the progenitor type,
    # `marg` only knows the pooled marginal.
    rng = np.random.default_rng(0)
    by_group = [pools[(pt, t_src)].astype(np.float32) for pt in train_ds.parent_types]

    def draw(groups, n_draws):
        """(n_draws, R, dim) independent real t_src cells of each row's type."""
        out = np.empty((n_draws, len(groups), dim), dtype=np.float32)
        for g in (0, 1):
            m = groups == g
            if m.any():
                out[:, m] = by_group[g][rng.integers(len(by_group[g]),
                                                     size=(n_draws, int(m.sum())))]
        return out

    rev["oracle"] = score(draw(parent_group, n_ref_draws), per_parent, per_sample=True)
    # Shuffle the type labels per draw: same 50/50 marginal, no lineage information.
    rev["marg"] = score(
        np.concatenate([draw(rng.permutation(parent_group), 1) for _ in range(n_ref_draws)]),
        per_parent, per_sample=True)

    out = {**motif, "dataset": dataset_name, "n_parents": len(parent_true),
           "n_leaves": len(x_tgt), "rev": rev}
    logger.info("[%s A=%r C=%r B=%r]\n  mse       %s\n  mse ambig %s\n"
                "  acc       %s\n  acc ambig %s", dataset_name,
                motif["A"], motif["C"], motif["B"],
                {m: round(v["mse"], 2) for m, v in rev.items()},
                {m: round(v["mse_ambiguous"], 2) for m, v in rev.items()},
                {m: round(v["type_acc"], 3) for m, v in rev.items()},
                {m: round(v["type_acc_ambiguous"], 3) for m, v in rev.items()})
    return out


def write_landscape(results, out_dir):
    """Flatten results to landscape.csv (one row per motif x method)."""
    rows = [
        {"cfg": r["dataset"], "seed": r["seed"], "method": method, **vals,
         "root_type": r["root_type"], "A": r["A"], "C": r["C"], "B": r["B"]}
        for r in results for method, vals in r["rev"].items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "landscape.csv", index=False)


def merge(methods, *, n_trees, n_marginal, fm_steps, genot_steps, n_samples,
          n_flow_steps):
    """Re-run only ``methods`` on the motifs already in results.pkl and splice them in.

    Same motifs, same seed=0 draw; every other row (``oracle`` / ``marg`` included) is
    left as it was. For re-scoring a retuned baseline without retraining everything
    else. Not for "ours", whose checkpoints this does not train.
    """
    if "ours" in methods:
        raise ValueError("--merge re-runs baselines only; drop 'ours' from --methods")
    with open(OUT_DIR / "results.pkl", "rb") as f:
        results = pickle.load(f)
    t_root, t_src, t_tgt = TIMES
    need = {t_root: set(), t_src: set(), t_tgt: set()}
    for r in results:
        need[t_root].add(r["root_type"])
        need[t_src].update((r["A"], r["C"]))
        need[t_tgt].update((r["A"], r["C"], r["B"]))
    pools = load_cell_pools(DATA_DIR, STATES_FILENAME, EMBEDDING_KEY,
                            {t: sorted(s) for t, s in need.items()})
    for r in results:
        new = eval_one_motif(
            r, r["dataset"], pools, methods, n_trees=n_trees, n_marginal=n_marginal,
            fm_steps=fm_steps, genot_steps=genot_steps, n_samples=n_samples,
            n_flow_steps=n_flow_steps)
        if new["n_leaves"] != r["n_leaves"]:
            raise RuntimeError(f"[{r['dataset']}] eval draw differs from results.pkl")
        for m in methods:
            r["rev"][m] = new["rev"][m]
    with open(OUT_DIR / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    write_landscape(results, OUT_DIR)
    logger.info("merged %s into %d motifs -> %s", sorted(methods), len(results), OUT_DIR)


# --------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=["ours", "fm", "gs", "gf", "wot", "moslin"])
    ap.add_argument("--variants", nargs="+", default=None, metavar="LABEL=EXPERIMENT",
                    help="pool/split variants to train + evaluate, each contributing a "
                         f"row under LABEL (default: ours={EXPERIMENT})")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES,
                    help="number of random (root, A, B, C) cell-type draws")
    ap.add_argument("--launcher", default=LAUNCHER,
                    help="Hydra launcher for the pool-split multirun ('local' or 'joblib')")
    ap.add_argument("--n-trees", type=int, default=200)
    ap.add_argument("--n-marginal", type=int, default=200)
    ap.add_argument("--train-steps", type=int, default=10000,
                    help="pool/split training steps")
    ap.add_argument("--fm-steps", type=int, default=3000)
    ap.add_argument("--genot-steps", type=int, default=3000)
    ap.add_argument("--n-samples-infer", type=int, default=8,
                    help="prediction samples per source cell at eval")
    ap.add_argument("--n-flow-steps", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--merge", action="store_true",
                    help="re-run only --methods (baselines) on the motifs in results.pkl "
                         "and splice them into it / landscape.csv, then exit")
    args = ap.parse_args()
    if args.quick:
        args.n_trees = args.n_marginal = 64
        args.train_steps = args.fm_steps = args.genot_steps = 40
        args.n_samples_infer, args.n_flow_steps = 2, 4

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge(set(args.methods), n_trees=args.n_trees, n_marginal=args.n_marginal,
              fm_steps=args.fm_steps, genot_steps=args.genot_steps,
              n_samples=args.n_samples_infer, n_flow_steps=args.n_flow_steps)
        return

    # 1. eligible subtypes -> shared cell pools + presence, then random draws.
    subtypes = sorted(pd.read_csv(os.path.join(DATA_DIR, TYPE_INFO), index_col=0)
                      .index.astype(str))
    logger.info("%d candidate subtypes | methods=%s | %d samples",
                len(subtypes), args.methods, args.n_samples)
    pools, present = load_present(subtypes)
    motifs = [sample_motif(present, seed) for seed in range(args.n_samples)]
    for i, m in enumerate(motifs):
        m["cfg"] = cfg_name(i)
        logger.info("sample %d: root=%r A=%r C=%r B=%r",
                    i, m["root_type"], m["A"], m["C"], m["B"])
    pd.DataFrame(motifs).to_csv(OUT_DIR / "samples.csv", index=False)

    # 2. write a dataset yaml per motif and train each pool/split variant (skipped
    # without "ours").
    variants = parse_variants(args.variants)
    if "ours" in args.methods:
        for i, m in enumerate(motifs):
            write_dataset_yaml(i, m, n_trees=args.n_trees, n_marginal=args.n_marginal)
        for label, exp in variants.items():
            logger.info("=== training variant %r (%s) ===", label, exp)
            run_pool_split_multirun(
                exp, [cfg_name(i) for i in range(len(motifs))],
                n_steps=args.train_steps, launcher=args.launcher,
                extra_overrides=TRAIN_OVERRIDES)

    # 3. evaluate the requested methods on every sampled motif.
    results = [
        eval_one_motif(
            m, cfg_name(i), pools, args.methods,
            n_trees=args.n_trees, n_marginal=args.n_marginal,
            fm_steps=args.fm_steps, genot_steps=args.genot_steps,
            n_samples=args.n_samples_infer, n_flow_steps=args.n_flow_steps,
            variants=variants,
        )
        for i, m in enumerate(motifs)
    ]

    # 4. dump.
    with open(OUT_DIR / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    write_landscape(results, OUT_DIR)
    logger.info("done -> %s", OUT_DIR)


if __name__ == "__main__":
    main()
