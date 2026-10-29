"""row-3 GENOT reference: context-blind baselines for the two context-scaling panels.

The model panels show energy distance FALLING as lineage context grows (ancestral:
revealing more leaves sharpens the root reconstruction; forward/descendant: revealing
more cousins sharpens the descendant prediction). A pure optimal-transport flow
(GENOT) cannot use lineage context at all -- it only transports cells between two
times -- so its recovery error is a FLAT reference line. This script scores both
GENOT baselines (Sinkhorn / cbFGW) on the SAME stratified subtree/focal population as
the canonical scripts, via a round-trip through the bottleneck node, so it is
comparable with the splitter's cycle consistency:

  ancestral : focal leaf -> subtree-root time -> leaf time. compare the regenerated
              cloud to the subtree's true leaf set (= canonical ancestral reference Y).
  forward   : an anchor descendant leaf of v -> v's time -> leaf time. compare the
              regenerated cloud to v's true descendant set (= canonical forward Y).

Both are context-blind: one energy distance per focal, summarised (mean +/- SEM) into
a flat dotted line per method by plot_context_scaling.py. We subsample to N_REF focals
per panel so the reference is cheap; the draw is the identical stratified selection
the model panels use (so the same population is scored).

writes numerical_results/context_scaling_genot.csv
       (columns: panel, method, energy_distance).
"""
import os
import sys

# keep JAX (GENOT) from grabbing the whole GPU -- it shares with torch here.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from analysis.figures.summary_qc import style as S
from analysis.lib.metrics.clonal_preservation import (
    _energy_distance, _genot_transport, _load_genot_model,
)
from analysis.lib.celltype_map.eval import build_label_maps
from analysis.pipelines.context._context_common import load_models, pick_focals_by_celltype
from analysis.pipelines.context._context_common import pick_forward_focals
# match the canonical scripts' selection windows exactly.
from analysis.pipelines.context import context_scaling_cycle_consistency as anc
from analysis.pipelines.context import context_scaling_forward_layer as fwd

CSV_NAME = "context_scaling_genot.csv"
N_REF = int(os.environ.get("N_REF", 400))   # focals subsampled per panel (flat line)
N_SAMPLES = S.N_GENOT_SAMPLES   # stochastic forward samples -> regenerated cloud


def _node_time(tree, node):
    return float(tree.nodes[node].get("time", float("nan")))


def _ancestral_candidates(trees, name_to_ct, rng):
    """(true_leaf_states, focal_leaf_state, leaf_time, root_time) per focal leaf.

    Same stratified (subtree, focal-leaf) draw as the ancestral model panel; the
    reference Y is the subtree's FULL true leaf set, exactly as the model scores it.
    """
    selected = pick_focals_by_celltype(
        trees, name_to_ct, anc.TIME_RANGE, anc.MIN_LEAVES, anc.MAX_LEAVES,
        anc.PER_TYPE_TARGET, rng)
    cands = []
    for (tname, root), focals in selected.items():
        tree = trees[tname]
        leaves = [n for n in nx.descendants(tree, root) if tree.out_degree(n) == 0]
        if len(leaves) < 2:
            continue
        root_time = _node_time(tree, root)
        for f in focals:
            cands.append((tname, leaves, f, _node_time(tree, f), root_time))
    return cands


def _forward_candidates(trees, name_to_ct, rng):
    """(desc_leaves, anchor_leaf, anchor_time, v_time) per focal internal node v.

    Same stratified internal-node draw as the forward model panel; the reference Y
    is v's true descendant leaf set. The anchor leaf (a seeded descendant of v) gives
    GENOT an observed cell to round-trip through v's time, since v is unobserved.
    """
    selected = pick_forward_focals(
        trees, name_to_ct, fwd.TIME_RANGE, fwd.MIN_LEAVES, fwd.MAX_LEAVES,
        fwd.MIN_DESC_LEAVES, fwd.PER_TYPE_TARGET, rng)
    cands = []
    for (tname, _R), vs in selected.items():
        tree = trees[tname]
        for v in vs:
            desc = [n for n in nx.descendants(tree, v) if tree.out_degree(n) == 0]
            if len(desc) < fwd.MIN_DESC_LEAVES:
                continue
            anchor = desc[int(rng.integers(len(desc)))]
            cands.append((tname, desc, anchor, _node_time(tree, anchor),
                          _node_time(tree, v)))
    return cands


def _roundtrip(model, dim, t_min, t_range, src_state, t_anchor, t_mid, seed):
    """anchor cell -> t_mid (1 sample) -> t_anchor (N_SAMPLES) regenerated cloud."""
    inferred = _genot_transport(
        model, src_state[None, :], t_src=t_anchor, t_tgt=t_mid,
        t_min=t_min, t_range=t_range, dim=dim, n_samples=1, seed=seed)[0, 0, :]
    return _genot_transport(
        model, inferred[None, :], t_src=t_mid, t_tgt=t_anchor,
        t_min=t_min, t_range=t_range, dim=dim, n_samples=N_SAMPLES, seed=seed + 1)[0]


def _score(method, pkl, anc_cands, fwd_cands, leaf_np):
    model, cfg, t_min, t_max, t_range = _load_genot_model(pkl, seed=S.SEED)
    dim = cfg["dim"]
    rows = []
    for i, (tname, leaves, f, leaf_time, root_time) in enumerate(
            tqdm(anc_cands, desc=f"{method} ancestral")):
        true_leaves = np.stack([leaf_np[l] for l in leaves])
        recovered = _roundtrip(model, dim, t_min, t_range, leaf_np[f],
                               leaf_time, root_time, S.SEED + i)
        rows.append(dict(panel="ancestral", method=method,
                         energy_distance=_energy_distance(true_leaves, recovered)))
    for i, (tname, desc, anchor, anchor_time, v_time) in enumerate(
            tqdm(fwd_cands, desc=f"{method} forward")):
        true_desc = np.stack([leaf_np[l] for l in desc])
        recovered = _roundtrip(model, dim, t_min, t_range, leaf_np[anchor],
                               anchor_time, v_time, S.SEED + i)
        rows.append(dict(panel="forward", method=method,
                         energy_distance=_energy_distance(true_desc, recovered)))
    return rows


def main():
    out = os.path.join(S.NR, CSV_NAME)
    if os.path.exists(out):
        print(f"{out} exists; nothing to do (delete it to recompute)")
        return

    _, _model, dataset = load_models()
    del _model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trees, leaf_states = dataset.trees, dataset.leaf_states
    leaf_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in leaf_states.items()}

    name_to_labels, _ = build_label_maps(dataset)
    name_to_ct = {n: d.get("cell_type") for n, d in name_to_labels.items()}

    rng = np.random.default_rng(S.SEED)
    anc_cands = _ancestral_candidates(trees, name_to_ct, rng)
    fwd_cands = _forward_candidates(trees, name_to_ct, rng)
    # drop focals whose leaf states aren't all available, then subsample to N_REF.
    anc_cands = [c for c in anc_cands if all(l in leaf_np for l in c[1]) and c[2] in leaf_np]
    fwd_cands = [c for c in fwd_cands if all(l in leaf_np for l in c[1]) and c[2] in leaf_np]
    anc_cands = [anc_cands[i] for i in rng.choice(
        len(anc_cands), size=min(N_REF, len(anc_cands)), replace=False)]
    fwd_cands = [fwd_cands[i] for i in rng.choice(
        len(fwd_cands), size=min(N_REF, len(fwd_cands)), replace=False)]
    print(f"scoring {len(anc_cands)} ancestral + {len(fwd_cands)} forward focals "
          f"per GENOT method", flush=True)

    rows = []
    for method, pkl in S.GENOT_PKLS.items():
        rows += _score(method, pkl, anc_cands, fwd_cands, leaf_np)
        pd.DataFrame(rows).to_csv(out, index=False)   # persist after each method
        print(f"appended {method} -> {out}", flush=True)

    df = pd.read_csv(out)
    print("\nmean energy distance by panel/method:")
    print(df.groupby(["panel", "method"])["energy_distance"].agg(["mean", "sem", "count"]))


if __name__ == "__main__":
    main()
