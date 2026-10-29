#!/usr/bin/env python
#SBATCH --job-name=crossing_landscape
#SBATCH --output=logs/crossing_landscape_%j.log
#SBATCH --error=logs/crossing_landscape_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bates
"""Map *which methods fail in which crossing settings* across a 2-D landscape.

The two axes (both computed cheaply from cell-type *means*, no training) are signed
coherence scores -- the same euclidean-distance margin between the imposed parent->child
coupling and its nearest competitor, using the same norm (euclidean) and the same
aggregation (mean over matched pairs). > 0 => the imposed coupling is the tightest, so
the method recovers it; < 0 => a competitor is tighter (the method fails). See
crossing_landscape/axis_parameterization.md for the equations.

  OT coherence          -- objects are the four children individually; competitors are
                           the other balanced couplings routing two children to each
                           parent. (min competing coupling cost) - (imposed coupling
                           cost). > 0 => OT recovers the imposed coupling; < 0 => anti-OT
                           (a regrouping fits the parents better, so OT mismatches).

  Felsenstein coherence -- objects are each mid's children pooled to their centroid
                           ("pool the descendants"); the only competitor swaps the two
                           centroids onto the wrong parents. (swapped-pairing cost) -
                           (correct-pairing cost). > 0 => each centroid is nearest its
                           own ancestor, so pooling recovers it; < 0 => anti-Felsenstein
                           (a wrong ancestor is at least as close).

Pipeline (one self-contained job):
  1. read final_type_info.csv -> the subtypes in LINEAGE.
  2. load their real cell pools at (t_root, t_src, t_tgt); keep the ones with
     enough cells at the times each motif slot needs.
  3. enumerate candidate motifs (root + 2 parents/anchors + ordered inner pair),
     scoring each on the two axes from the pool means. The coupling-strength
     ``alpha`` is fixed, not swept -- an ordered inner-pair permutation already
     realises both couplings a discrete alpha would, so the cell-type choice
     alone spans the whole plane (see the ALPHA note in the config block).
  4. draw N_SELECT motifs uniformly from the candidates.
  5. write one dataset yaml per selected motif, train our pool/split on each
     (few steps, small data -- this is a landscape sketch, not a final number),
     then evaluate our model + the baselines (FM / GENOT-Sinkhorn / -FGW /
     Waddington-OT / moslin).
  6. dump results.pkl / landscape.csv for crossing_landscape_plot.py.

Run (GPU node, from the repo root or via sbatch):
    PYTHONPATH=. python synthetic_evals/crossing_landscape_eval.py
    PYTHONPATH=. python synthetic_evals/crossing_landscape_eval.py --quick   # smoke
"""

from __future__ import annotations

import os
import sys

# Resolve the repo root whether launched by sbatch (SLURM_SUBMIT_DIR) or directly,
# then run from there so the relative data / config / outputs paths line up.
base_dir = os.path.abspath(os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import argparse
import functools
import itertools
import logging
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.cell_type_crossing import CellTypeCrossingDataset, load_cell_pools
from synthetic_evals.baselines import (
    genot_sinkhorn,
    marginal_fm,
    moscot_ot,
    registry,
)
from synthetic_evals.core import inference as inf
from synthetic_evals.core import metrics
from synthetic_evals.core.hydra_io import (
    DEVICE, ROOT, load_pool_split, run_pool_split_multirun,
)
from synthetic_evals.core.subtrees import subtrees_to_pairs
# Shared helpers from the cell-type crossing eval.
from synthetic_evals.cell_type_crossing_eval import (
    EXPERIMENT, build_eval_subtrees, _fgw_bank_from_dataset, _grouped_energy,
    _marginal_energy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("crossing_landscape")


## CONFIG ##

DATA_DIR = "../data/embryos"
STATES_FILENAME = "qiu24_integrated_hvgs_and_scvi.h5ad"
EMBEDDING_KEY = "X_scvi"
TYPE_INFO = "final_type_info.csv"

# Which subtypes are eligible for the motif slots. SCOPE picks the annotation
# column to filter on; SCOPE_VALUE is the label within it. "all" ignores
# SCOPE_VALUE and lets any cell type fill any slot -- the widest exploration of
# the (OT, Felsenstein) plane (cross-lineage/germ-layer motifs included).
SCOPE = "germ_layer"                      # "lineage" | "germ_layer" | "all"
SCOPE_VALUE = "ectoderm"        # the lineage / germ-layer label (unused if SCOPE == "all")
# Methods to train + evaluate; subset of ("ours", "fm", "gs", "gf", "wot", "moslin").
# Only the requested ones are trained (baselines fresh; "ours" via the per-motif
# checkpoint).
METHODS = ("ours", "gs", "fm", "gf", "wot", "moslin")
# How each mid's two children are chosen:
#   "parent" -- each mid's ANCHOR child reuses the parent subtype (child type ==
#               ancestor type), only the inner pair varies. Parents must exist
#               at both t_src and t_tgt.
#   "free"   -- lift that: all four children are arbitrary t_tgt subtypes, none
#               tied to a parent. Parents need only exist at t_src. A strict
#               superset of "parent" that opens far more of the plane (and drops
#               the "descendant type == ancestor type" artifact).
ANCHOR_MODE = "free"
TIMES = (8.5, 9.0, 9.5)               # (t_root, t_src, t_tgt); crossing at t_src->t_tgt
# alpha (imposed coupling strength) is NOT swept: enumerating the inner pair as an
# ORDERED permutation already realises both couplings a discrete alpha would (own
# inner == alpha 0; swapped inner == alpha 1), so alpha is redundant with the
# cell-type choice, which alone spans the whole (OT, Felsenstein) plane. Fixed at 0
# => each mid keeps the inner child it is enumerated with. See
# crossing_landscape/axis_parameterization.md.
ALPHA = 0.0

# Hydra launcher for the per-motif pool-split training multirun. "local" runs
# the motifs one at a time in-process; "joblib" packs several onto the GPU
LAUNCHER = "joblib"
MIN_CELLS = 50          # a (subtype, time) slot needs at least this many real cells
N_PARENT_CANDS = 10      # most-abundant valid parents/anchors to enumerate pairs over
N_INNER_CANDS = 10       # most-abundant valid inner (swappable) child types
N_SELECT = 32           # motifs actually trained + evaluated (uniform draw over candidates)
SELECT_SEED = 0         # seed for that draw

OUT_DIR = ROOT / "synthetic_evals" / "crossing_landscape"
CFG_GROUP = "crossing_landscape"        # generated dataset configs -> config/dataset/<group>/


# --------------------------------------------------------------------------
#  1-3. Candidate motifs + cheap (OT, Felsenstein) scoring from pool means
# --------------------------------------------------------------------------


def scope_desc(scope, value):
    return "all cell types" if scope == "all" else f"{scope}={value!r}"


def scoped_subtypes(scope: str, value: str) -> list[str]:
    """Subtypes eligible for motif slots under SCOPE (lineage / germ_layer / all)."""
    df = pd.read_csv(os.path.join(DATA_DIR, TYPE_INFO), index_col=0)
    if scope == "all":
        return sorted(df.index.astype(str))
    if scope not in ("lineage", "germ_layer"):
        raise ValueError(f"scope must be lineage/germ_layer/all, got {scope!r}")
    return sorted(df.index[df[scope] == value].astype(str))


def load_means(subtypes):
    """``pools`` for every (subtype, time) and per-slot means for those that pass.

    Returns ``(pools, means, present, counts)``: ``means[(name, t)]`` is the mean
    embedding, ``present[t]`` the set of subtypes with >= MIN_CELLS at t, and
    ``counts[(name, t)]`` the real cell count (used to rank slot abundance).
    """
    pools = load_cell_pools(
        DATA_DIR, STATES_FILENAME, EMBEDDING_KEY,
        {t: list(subtypes) for t in TIMES})
    means, present, counts = {}, {t: set() for t in TIMES}, {}
    for (name, t), X in pools.items():
        counts[(name, t)] = int(X.shape[0])
        if X.shape[0] >= MIN_CELLS:
            means[(name, t)] = X.mean(0)
            present[t].add(name)
    return pools, means, present, counts


def _mean_dist(sources, targets):
    """Mean euclidean distance over paired means -- the shared metric for both axes."""
    return float(np.mean([np.linalg.norm(s - t) for s, t in zip(sources, targets)]))


def score_motif(means, *, t_src, t_tgt, pA, pB, anchorA, A2, B1, anchorB, alpha):
    """(ot_coherence, fels_coherence) for one motif at one alpha, from means alone.

    Both are signed euclidean-distance coherence scores -- (nearest competing coupling
    cost) - (imposed/true coupling cost) -- using the same norm (euclidean) and the
    same aggregation (mean over matched pairs). > 0 => the imposed coupling is the
    tightest, so the method recovers it; < 0 => a competitor is tighter (the method
    fails). See crossing_landscape/axis_parameterization.md for the equations. The two
    axes differ only in what the objects are (individual children vs pooled sibling
    centroids) and hence which competitors exist.

    mA's children are ``(anchorA, A2)`` -> pA, mB's are ``(B1, anchorB)`` -> pB.
    """
    PA, PB = means[(pA, t_src)], means[(pB, t_src)]
    a1, a2 = means[(anchorA, t_tgt)], means[(A2, t_tgt)]    # mid A: anchor, inner child
    b1, b2 = means[(B1, t_tgt)], means[(anchorB, t_tgt)]    # mid B: inner, anchor child

    # -- OT coherence: individual children vs parents. The imposed grouping routes
    # {a1,a2}->PA and {b1,b2}->PB (the inner pair a2/b1 crosses with prob alpha);
    # competitors are the OTHER balanced couplings (two children per parent).
    children = [a1, a2, b1, b2]
    imposed = float(np.mean([
        np.linalg.norm(a1 - PA), np.linalg.norm(b2 - PB),
        (1 - alpha) * np.linalg.norm(a2 - PA) + alpha * np.linalg.norm(a2 - PB),
        (1 - alpha) * np.linalg.norm(b1 - PB) + alpha * np.linalg.norm(b1 - PA),
    ]))
    imposed_grpA = (0, 1)                                   # children[0,1]={a1,a2} -> PA
    def coupling_cost(grpA):
        tgt = [PA if i in grpA else PB for i in range(4)]
        return _mean_dist(children, tgt)
    competitors = [coupling_cost(g)
                   for g in itertools.combinations(range(4), 2) if g != imposed_grpA]
    ot_coherence = min(competitors) - imposed

    # -- Felsenstein coherence: pool each mid's children to a centroid ("pool the
    # descendants"), then score the correct centroid->parent pairing against the swap.
    rA = 0.5 * (a1 + (1 - alpha) * a2 + alpha * b1)
    rB = 0.5 * (b2 + (1 - alpha) * b1 + alpha * a2)
    correct = _mean_dist([rA, rB], [PA, PB])
    swapped = _mean_dist([rA, rB], [PB, PA])
    fels_coherence = swapped - correct
    return float(ot_coherence), float(fels_coherence)


def _pack(means, root_type, pA, pB, anchorA, A2, B1, anchorB):
    """Score one motif and pack it into a candidate record."""
    t_root, t_src, t_tgt = TIMES
    ot, fels = score_motif(
        means, t_src=t_src, t_tgt=t_tgt, pA=pA, pB=pB,
        anchorA=anchorA, A2=A2, B1=B1, anchorB=anchorB, alpha=ALPHA)
    return {"root_type": root_type, "pA": pA, "pB": pB,
            "anchorA": anchorA, "A2": A2, "B1": B1, "anchorB": anchorB,
            "alpha": ALPHA, "ot_coherence": ot, "fels_coherence": fels}


def enumerate_candidates(means, present, counts, anchor_mode="parent"):
    """All scored candidate motifs over the most-abundant valid slots.

    parent-anchor mode: each mid's anchor child *is* the parent subtype, so only
    the inner pair varies (parents must exist at t_src AND t_tgt). free mode: all
    four children are arbitrary t_tgt subtypes (parents need only exist at t_src).
    """
    t_root, t_src, t_tgt = TIMES
    roots = sorted(present[t_root], key=lambda s: -counts[(s, t_root)])
    if not roots:
        raise RuntimeError(
            f"no candidate subtype with >= {MIN_CELLS} cells at t_root={t_root} "
            f"(scope too narrow, or wrong times?)")
    root_type = roots[0]                 # single fixed common ancestor

    cands = []
    if anchor_mode == "parent":
        # parents double as their own anchor child -> need presence at both times.
        anchors = sorted(present[t_src] & present[t_tgt],
                         key=lambda s: -min(counts[(s, t_src)], counts[(s, t_tgt)]))[:N_PARENT_CANDS]
        inners = sorted(present[t_tgt], key=lambda s: -counts[(s, t_tgt)])[:N_INNER_CANDS]
        logger.info("parent-anchor | root=%r | %d parents/anchors | %d inners",
                    root_type, len(anchors), len(inners))
        for pA, pB in itertools.combinations(anchors, 2):
            for A2, B1 in itertools.permutations([s for s in inners if s not in (pA, pB)], 2):
                cands.append(_pack(means, root_type, pA, pB, pA, A2, B1, pB))
    elif anchor_mode == "free":
        # parents need only exist at t_src; the 4 children are arbitrary t_tgt types.
        parents = sorted(present[t_src], key=lambda s: -counts[(s, t_src)])[:N_PARENT_CANDS]
        children = sorted(present[t_tgt], key=lambda s: -counts[(s, t_tgt)])[:N_INNER_CANDS]
        logger.info("free-anchor | root=%r | %d parents | %d children",
                    root_type, len(parents), len(children))
        for pA, pB in itertools.combinations(parents, 2):
            for cA in itertools.combinations(children, 2):        # mA's two children
                rest = [c for c in children if c not in cA]
                for cB in itertools.combinations(rest, 2):        # mB's two children
                    cands.append(_pack(means, root_type, pA, pB, cA[0], cA[1], cB[0], cB[1]))
    else:
        raise ValueError(f"anchor_mode must be parent/free, got {anchor_mode!r}")

    logger.info("enumerated %d candidate motifs (%s mode)", len(cands), anchor_mode)
    return cands


# --------------------------------------------------------------------------
#  4. Draw the motifs to evaluate
# --------------------------------------------------------------------------


def select_uniform(cands, n, seed=SELECT_SEED):
    """Uniform draw of n candidates without replacement.

    The candidate enumeration is itself uniform over the admissible cell types, so
    drawing rows uniformly makes the evaluated set an unbiased sample of the motif
    space. The coherence scores label motifs downstream; they do not select.
    """
    if len(cands) <= n:
        return list(cands)
    idx = np.random.default_rng(seed).choice(len(cands), n, replace=False)
    return [cands[i] for i in sorted(idx)]


# --------------------------------------------------------------------------
#  5. Per-motif dataset config + the parameterised eval (scores on TRAIN draw)
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
    pA, pB, A2, B1 = motif["pA"], motif["pB"], motif["A2"], motif["B1"]
    anchorA, anchorB = motif["anchorA"], motif["anchorB"]
    t_root, t_src, t_tgt = TIMES
    text = (
        f"# auto-generated by crossing_landscape_eval.py -- do not hand-edit.\n"
        f"_target_: datasets.cell_type_crossing.CellTypeCrossingDataset\n"
        f"data_dir: {DATA_DIR}\n"
        f"states_filename: {STATES_FILENAME}\n"
        f"embedding_key: {EMBEDDING_KEY}\n"
        f"times: [{t_root}, {t_src}, {t_tgt}]\n"
        f'root_type: "{motif["root_type"]}"\n'
        f'parent_types: ["{pA}", "{pB}"]\n'
        f"child_types:\n"
        f'  - ["{anchorA}", "{A2}"]\n'
        f'  - ["{B1}", "{anchorB}"]\n'
        f"alpha: {motif['alpha']}\n"
        f"n_trees: {n_trees}\n"
        f"n_marginal: {n_marginal}\n"
        f"max_size: 128\n"
        f"seed: 0\n"
    )
    (d / f"cfg{i:02d}.yaml").write_text(text)


def eval_one_motif(motif, dataset_name, pools, methods, *, n_trees, n_marginal,
                   fm_steps, genot_steps, n_samples, n_flow_steps,
                   variants=None, extra_metrics=False):
    """Forward (parent -> children) energy error for each requested method on one motif.

    Mirrors ``cell_type_crossing_eval.eval_one_alpha`` (forward only) but is
    motif-parameterised and scores on the TRAINING draw (no held-out set). ``methods``
    selects which baselines run; ``variants`` maps each pool/split variant's row label to
    its experiment, and they are evaluated when "ours" is among the methods.
    ``extra_metrics`` also scores the alternate two-sample metrics, which only the
    supplement's robustness panel needs.
    """
    methods = set(methods)
    variants = dict(variants or {"ours": EXPERIMENT})
    t_root, t_src, t_tgt = TIMES
    pA, pB, A2, B1 = motif["pA"], motif["pB"], motif["A2"], motif["B1"]
    anchorA, anchorB = motif["anchorA"], motif["anchorB"]
    train_ds = CellTypeCrossingDataset(
        data_dir=DATA_DIR, states_filename=STATES_FILENAME, embedding_key=EMBEDDING_KEY,
        times=TIMES, root_type=motif["root_type"], parent_types=(pA, pB),
        child_types=((anchorA, A2), (B1, anchorB)), alpha=motif["alpha"], seed=0,
        cell_pools=pools, n_trees=n_trees, n_marginal=n_marginal,
    )
    dim = int(train_ds.x_dim)
    pairs_fwd = [(t_src, t_tgt)]

    # ---- baselines (trained fresh on the marginals; only the requested ones) ----
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
        "fm": dict(n_steps=fm_steps, pairs=pairs_fwd, device=DEVICE, seed=0),
        "gs": dict(n_steps=genot_steps, pairs=pairs_fwd, seed=0,
                   coupling_mode="precomputed"),
        "gf": dict(n_steps=genot_steps, pairs=pairs_fwd, seed=2,
                   coupling_mode="precomputed", n_tree_pairings=1),
        "wot": dict(pairs=pairs_fwd),
        "moslin": dict(pairs=pairs_fwd),
    }
    models = {
        name: registry.get(name).train(banks[name](), dim=dim, **train_kwargs[name])
        for name in registry.names() if name in methods
    }

    # ---- our trained model(s) for this motif (forward = splitter only) ----
    splitters = ({label: load_pool_split(exp, dataset_name)[1]
                  for label, exp in variants.items()} if "ours" in methods else {})

    # ---- eval draw == the training draw ----
    subs = build_eval_subtrees(train_ds)
    pairs = subtrees_to_pairs(subs, t_src, t_tgt)
    x_src = np.stack([p[4] for p in pairs]).astype(np.float32)
    x_tgt = np.stack([p[5] for p in pairs]).astype(np.float32)
    leaf_group = np.array([subs[i]["group"]
                           for i, s in enumerate(subs) for _ in s["leaves"]])

    # ---- FORWARD: parent -> children ----
    # `fwd` keeps the headline energy distance; `fwd_metrics` adds the alternate metrics
    # (same clouds, same grouping) for the robustness panel. `preds` keeps the clouds
    # themselves, for --rescore and figure.compute_landscape.
    fwd, fwd_metrics, preds = {}, {}, {}

    def score(label, pred):
        preds[label] = pred
        fwd[label] = _grouped_energy(pred, x_tgt, leaf_group)
        if extra_metrics:
            fwd_metrics[label] = metrics.grouped_distances(pred, x_tgt, leaf_group)

    for label, splitter in splitters.items():
        score(label, inf.splitter_predict_subtrees(
            splitter, subs, n_samples=n_samples, n_flow_steps=n_flow_steps))
    for name, pkg in models.items():
        score(name, inf.forward_samples(
            registry.get(name), pkg, x_src, t_src, t_tgt,
            n_samples=n_samples, n_flow_steps=n_flow_steps))
    # marginal baseline (training-free): sample the pooled t_tgt marginal per group.
    fwd["marg"] = _marginal_energy(x_tgt, leaf_group)
    if extra_metrics:
        fwd_metrics["marg"] = metrics.marginal_distances(x_tgt, leaf_group)

    out = {**motif, "dataset": dataset_name,
           "n_parents": len(subs), "n_leaves": len(pairs), "fwd": fwd,
           "samples": {"pred": preds, "x_tgt": x_tgt, "leaf_group": leaf_group}}
    if extra_metrics:
        out["fwd_metrics"] = fwd_metrics
    logger.info("[%s ot_coh=%.2f fels_coh=%.2f] fwd=%s",
                dataset_name, motif["ot_coherence"], motif["fels_coherence"], fwd)
    return out


# --------------------------------------------------------------------------
#  Result serialisation + cheap marginal-baseline back-fill
# --------------------------------------------------------------------------


def write_landscape(results, out_dir):
    """Flatten results (one row per motif x method) to landscape.csv (forward only)."""
    rows = []
    for r in results:
        for method, err in r["fwd"].items():
            extra = r.get("fwd_metrics", {}).get(method, {})
            rows.append({
                "cfg": r["dataset"], "alpha": r["alpha"],
                "ot_coherence": r["ot_coherence"], "fels_coherence": r["fels_coherence"],
                "method": method, "energy": err,
                **{m: v for m, v in extra.items() if m != "energy"},
                "pA": r["pA"], "pB": r["pB"], "anchorA": r["anchorA"],
                "A2": r["A2"], "B1": r["B1"], "anchorB": r["anchorB"],
            })
    pd.DataFrame(rows).to_csv(out_dir / "landscape.csv", index=False)


def rescore():
    """Recompute the forward energies of an existing run from its stored samples.

    ``results.pkl`` carries ``samples = {pred: {method: (S, R, d)}, x_tgt,
    leaf_group}`` per motif, so ``fwd`` (and the ``energy`` entry of
    ``fwd_metrics``, leaving the other metrics alone) can be rebuilt without
    loading cells, training or running inference. The landscape.csv in place at
    the FIRST rescore is kept as landscape_prev.csv; later rescores leave that
    backup alone rather than overwriting it with their own input.
    """
    with open(OUT_DIR / "results.pkl", "rb") as f:
        results = pickle.load(f)
    old_csv = OUT_DIR / "landscape.csv"
    bak = OUT_DIR / "landscape_prev.csv"
    if old_csv.exists() and not bak.exists():
        shutil.copyfile(old_csv, bak)

    for r in results:
        samples = r.get("samples")
        if samples is None:
            logger.warning("[%s] no stored samples; left untouched", r["dataset"])
            continue
        x_tgt, leaf_group = samples["x_tgt"], samples["leaf_group"]
        for method, pred in samples["pred"].items():
            r["fwd"][method] = _grouped_energy(pred, x_tgt, leaf_group)
            if method in r.get("fwd_metrics", {}):
                r["fwd_metrics"][method]["energy"] = r["fwd"][method]
        r["fwd"]["marg"] = _marginal_energy(x_tgt, leaf_group)
        if "marg" in r.get("fwd_metrics", {}):
            r["fwd_metrics"]["marg"]["energy"] = r["fwd"]["marg"]
        logger.info("[%s] rescored fwd=%s", r["dataset"],
                    {k: round(v, 4) for k, v in r["fwd"].items()})

    with open(OUT_DIR / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    write_landscape(results, OUT_DIR)
    logger.info("rescored %d motifs -> %s", len(results), OUT_DIR)


def add_marg(*, n_trees, n_marginal):
    """Add the forward marginal baseline to an existing run -- no training, no model eval.

    Reloads the cells for exactly the subtypes each already-evaluated motif touches,
    rebuilds that motif's eval subtrees (the same seed=0 draw), computes the
    forward marginal-baseline energy, and merges a ``marg`` method into the existing
    results.pkl / landscape.csv. Lets an existing run gain the baseline
    without re-running the (expensive) training + inference.
    """
    with open(OUT_DIR / "results.pkl", "rb") as f:
        results = pickle.load(f)
    t_root, t_src, t_tgt = TIMES
    need = {t_root: set(), t_src: set(), t_tgt: set()}
    for r in results:
        need[t_root].add(r["root_type"])
        need[t_src].update((r["pA"], r["pB"]))
        need[t_tgt].update((r["anchorA"], r["A2"], r["B1"], r["anchorB"]))
    pools = load_cell_pools(DATA_DIR, STATES_FILENAME, EMBEDDING_KEY,
                            {t: sorted(s) for t, s in need.items()})
    for r in results:
        train_ds = CellTypeCrossingDataset(
            data_dir=DATA_DIR, states_filename=STATES_FILENAME, embedding_key=EMBEDDING_KEY,
            times=TIMES, root_type=r["root_type"], parent_types=(r["pA"], r["pB"]),
            child_types=((r["anchorA"], r["A2"]), (r["B1"], r["anchorB"])),
            alpha=r["alpha"], seed=0, cell_pools=pools,
            n_trees=n_trees, n_marginal=n_marginal,
        )
        subs = build_eval_subtrees(train_ds)
        pairs = subtrees_to_pairs(subs, t_src, t_tgt)
        x_tgt = np.stack([p[5] for p in pairs]).astype(np.float32)
        leaf_group = np.array([subs[i]["group"]
                               for i, s in enumerate(subs) for _ in s["leaves"]])
        r["fwd"]["marg"] = _marginal_energy(x_tgt, leaf_group)
        logger.info("[%s] marg fwd=%.3f", r["dataset"], r["fwd"]["marg"])
    with open(OUT_DIR / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    write_landscape(results, OUT_DIR)
    logger.info("added marginal baseline -> %s", OUT_DIR)


def merge(methods, *, n_trees, n_marginal, fm_steps, genot_steps, n_samples,
          n_flow_steps, extra_metrics):
    """Re-run only ``methods`` on the motifs already in results.pkl and splice them in.

    Same motifs, same seed=0 draw (checked against the stored ``x_tgt``); every other
    method's energies, metrics and clouds are left as they were. For re-scoring a
    retuned baseline without retraining everything else. Not for "ours", whose
    checkpoints this does not train.
    """
    if "ours" in methods:
        raise ValueError("--merge re-runs baselines only; drop 'ours' from --methods")
    with open(OUT_DIR / "results.pkl", "rb") as f:
        results = pickle.load(f)
    t_root, t_src, t_tgt = TIMES
    need = {t_root: set(), t_src: set(), t_tgt: set()}
    for r in results:
        need[t_root].add(r["root_type"])
        need[t_src].update((r["pA"], r["pB"]))
        need[t_tgt].update((r["anchorA"], r["A2"], r["B1"], r["anchorB"]))
    pools = load_cell_pools(DATA_DIR, STATES_FILENAME, EMBEDDING_KEY,
                            {t: sorted(s) for t, s in need.items()})
    for r in results:
        new = eval_one_motif(
            r, r["dataset"], pools, methods, n_trees=n_trees, n_marginal=n_marginal,
            fm_steps=fm_steps, genot_steps=genot_steps, n_samples=n_samples,
            n_flow_steps=n_flow_steps, extra_metrics=extra_metrics)
        if not np.array_equal(new["samples"]["x_tgt"], r["samples"]["x_tgt"]):
            raise RuntimeError(f"[{r['dataset']}] eval draw differs from results.pkl")
        for m in methods:
            r["fwd"][m] = new["fwd"][m]
            r["samples"]["pred"][m] = new["samples"]["pred"][m]
            if extra_metrics:
                r["fwd_metrics"][m] = new["fwd_metrics"][m]
    with open(OUT_DIR / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    write_landscape(results, OUT_DIR)
    logger.info("merged %s into %d motifs -> %s", sorted(methods), len(results), OUT_DIR)


# --------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["lineage", "germ_layer", "all"], default=SCOPE,
                    help="which subtypes may fill motif slots (default from config)")
    ap.add_argument("--scope-value", default=SCOPE_VALUE,
                    help="lineage / germ-layer label (ignored when --scope all)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=["ours", "fm", "gs", "gf", "wot", "moslin"],
                    help="methods to train + evaluate (default from config)")
    ap.add_argument("--anchor-mode", choices=["parent", "free"], default=ANCHOR_MODE,
                    help="parent: anchor child == parent subtype; free: all 4 "
                         "children arbitrary (default from config)")
    ap.add_argument("--launcher", default=LAUNCHER,
                    help="Hydra launcher for the pool-split multirun: 'local' "
                         "(serial) or 'joblib' (several motifs on the GPU at "
                         "once; default from config)")
    ap.add_argument("--n-select", type=int, default=N_SELECT)
    ap.add_argument("--select-seed", type=int, default=SELECT_SEED,
                    help="seed for the uniform motif draw")
    ap.add_argument("--variants", nargs="+", default=None, metavar="LABEL=EXPERIMENT",
                    help="pool/split variants to train + evaluate, each contributing a "
                         "row under LABEL (default: ours=%s). e.g. add "
                         "ours_meanpool=cell_type_crossing/meanpool_vpsde_idnn_leaf "
                         "for the pooler ablation" % EXPERIMENT)
    ap.add_argument("--n-trees", type=int, default=200)
    ap.add_argument("--n-marginal", type=int, default=200)
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--fm-steps", type=int, default=3000)
    ap.add_argument("--genot-steps", type=int, default=3000)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--n-flow-steps", type=int, default=20)
    ap.add_argument("--extra-metrics", action="store_true",
                    help="also score the sinkhorn / gaussian-MMD metrics (supplement)")
    ap.add_argument("--score-only", action="store_true",
                    help="enumerate + score + select, write candidates, skip train/eval")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute forward energies from the samples already in "
                         "results.pkl (no training or model eval), then exit")
    ap.add_argument("--add-marg", action="store_true",
                    help="cheaply add the marginal baseline to an existing results.pkl "
                         "/ landscape.csv (no training or model eval), then exit")
    ap.add_argument("--merge", action="store_true",
                    help="re-run only --methods (baselines) on the motifs in results.pkl "
                         "and splice them into it / landscape.csv, then exit")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke run (few configs/steps/trees)")
    args = ap.parse_args()
    if args.quick:
        args.n_select = 2
        args.n_trees = args.n_marginal = 64
        args.train_steps = args.fm_steps = args.genot_steps = 40
        args.n_samples, args.n_flow_steps = 2, 4

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.rescore:
        rescore()
        return

    if args.add_marg:
        add_marg(n_trees=args.n_trees, n_marginal=args.n_marginal)
        return

    if args.merge:
        merge(set(args.methods), n_trees=args.n_trees, n_marginal=args.n_marginal,
              fm_steps=args.fm_steps, genot_steps=args.genot_steps,
              n_samples=args.n_samples, n_flow_steps=args.n_flow_steps,
              extra_metrics=args.extra_metrics)
        return

    # 1-3: subtypes -> means -> scored candidate motifs.
    subtypes = scoped_subtypes(args.scope, args.scope_value)
    logger.info("scope=%s: %d candidate subtypes | anchor_mode=%s | methods=%s",
                scope_desc(args.scope, args.scope_value), len(subtypes),
                args.anchor_mode, args.methods)
    pools, means, present, counts = load_means(subtypes)
    cands = enumerate_candidates(means, present, counts, args.anchor_mode)
    pd.DataFrame(cands).to_csv(OUT_DIR / "candidates.csv", index=False)

    # 4: uniform draw over the candidates.
    selected = select_uniform(cands, args.n_select, args.select_seed)
    for i, m in enumerate(selected):
        m["cfg"] = cfg_name(i)
    logger.info("drew %d/%d motifs uniformly (seed=%d)",
                len(selected), len(cands), args.select_seed)
    pd.DataFrame(selected).to_csv(OUT_DIR / "selected.csv", index=False)
    if args.score_only:
        logger.info("score-only: wrote selected.csv; done")
        return

    # 5a: each pool/split variant needs a per-motif checkpoint -- write one dataset yaml
    # each and train them (skipped entirely when "ours" is not among the methods).
    variants = parse_variants(args.variants)
    if "ours" in args.methods:
        for i, m in enumerate(selected):
            write_dataset_yaml(i, m, n_trees=args.n_trees, n_marginal=args.n_marginal)
        for label, exp in variants.items():
            logger.info("=== training variant %r (%s) ===", label, exp)
            run_pool_split_multirun(
                exp, [cfg_name(i) for i in range(len(selected))],
                n_steps=args.train_steps, launcher=args.launcher)

    # 5b: evaluate the requested methods on every selected motif.
    results = [
        eval_one_motif(
            m, cfg_name(i), pools, args.methods,
            n_trees=args.n_trees, n_marginal=args.n_marginal,
            fm_steps=args.fm_steps, genot_steps=args.genot_steps,
            n_samples=args.n_samples, n_flow_steps=args.n_flow_steps,
            variants=variants, extra_metrics=args.extra_metrics,
        )
        for i, m in enumerate(selected)
    ]

    # 6: dump.
    with open(OUT_DIR / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    write_landscape(results, OUT_DIR)
    logger.info("done -> %s", OUT_DIR)


if __name__ == "__main__":
    main()
