"""couple one method on the held-out trees and score it: clonal, marginal, assignment metrics.

the fig 3 code does the work (summary_qc compute_distortion and compute_assignment_metrics,
with its constants and seeds); this script only picks the trees. with --parity every tree
counts as held out, so the paper models must reproduce the fig 3 caches (gather.py --parity).
writes RESULTS/<method>/ (RESULTS/parity/<method>/ with --parity).

    python -m analysis.figures.supp_holdout.evaluate vpsde
    python -m analysis.figures.supp_holdout.evaluate vpsde_all --parity
"""
import argparse
import glob
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch

from analysis.figures.summary_qc import compute_distortion as CD
from analysis.figures.summary_qc import style as S
from analysis.figures.summary_qc.compute_assignment_metrics import compute_for
from analysis.figures.supp_holdout import common as C
from analysis.lib.coupling.nearest_neighbor_coupling_w_context import couple_all_trees
from analysis.lib.eval_core.models import load_config, load_splitter
from analysis.lib.metrics.germ_layer_eval import assignment_germ_layer_crossing_by_bin
from analysis.lib.metrics.linkage_metrics import lineage_oscillation_fraction
from datasets.mouse_embryo import MouseEmbryoDataset

PAPER_GENOT = {"genot_sinkhorn_all": S.GENOT_SINKHORN_PKL, "genot_cbfgw_all": S.GENOT_CBFGW_PKL}
ROLLOUT = dict(n_steps=S.SPLIT_STEPS, max_nodes=S.MAX_NODES, batch_size=S.BATCH_SIZE)


def held_out_run():
    """the hold_out_vpsde run dir, checked to be trained without the held-out clones."""
    runs = glob.glob(os.path.join(C.OUTPUTS, C.RUN_NAME + "_*"))
    assert len(runs) == 1, runs
    assert sorted(load_config(runs[0]).dataset.holdout_clones) == sorted(C.HOLDOUT)
    return runs[0]


def couple(model, trees, states, n):
    """the fig 3 coupling (eval_core.models.couple) of the given trees."""
    torch.manual_seed(S.SEED)
    return couple_all_trees(model, trees, states, pool_n_steps=n,
                            sample_time_gap=0.5, min_snap_time=7.0)


def genot_assignments(pkl, dataset):
    """genot_assign.py's backward-chain coupling of every tree from a saved genot model."""
    # genot_assign.py's flags; set before jax first touches the gpu
    os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
    import jax.random as jr
    from baselines import genot_common as gc
    with open(pkl, "rb") as f:
        saved = pickle.load(f)
    model = gc.load_mlp(saved["model_config"], saved["flat_params"])
    parent, m = gc.backward_parent_map(model, dataset.states, t_min=saved["t_min"],
                                       t_max=saved["t_max"], key=jr.PRNGKey(42), n_samples=5)
    return gc.build_assignments(dataset.trees, dataset.states, parent, m, seed=42)


def agreement(a, b, tree_names):
    """nodes assigned in a, in b, in both, and the fraction of those given the same cell."""
    na = sum(len(a.get(t, {})) for t in tree_names)
    nb = sum(len(b.get(t, {})) for t in tree_names)
    same = [a[t][n] == b[t][n] for t in tree_names for n in a.get(t, {}) if n in b.get(t, {})]
    return f"{na} / {nb} nodes assigned, {len(same)} in both, {np.mean(same):.4f} same cell"


def annotations(dataset):
    """the cell -> germ layer / lineage maps of compute_assignment_metrics.main."""
    ti = dataset.type_info_df
    subtype_to_type = dict(zip(ti.index, ti["cell_type"]))
    subtype_to_lineage = dict(zip(ti.index, ti["lineage"]))
    type_to_gl = dict(zip(ti["cell_type"], ti["germ_layer"]))
    name_to_subtype = dict(zip(dataset.cell_types_df.index,
                               dataset.cell_types_df["cell_subtype"]))
    name_to_lineage = {n: subtype_to_lineage[st] for n, st in name_to_subtype.items()
                       if st in subtype_to_lineage}
    name_to_gl = {n: type_to_gl[subtype_to_type[st]] for n, st in name_to_subtype.items()
                  if st in subtype_to_type and subtype_to_type[st] in type_to_gl}
    lineages = np.unique(list(name_to_lineage.values()))
    time_bins = sorted(dataset.states.obs["time"].astype(float).unique())
    return name_to_gl, name_to_lineage, lineages, time_bins


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=C.METHODS)
    ap.add_argument("--parity", action="store_true",
                    help="every tree held out, paper models only")
    args = ap.parse_args()
    method = args.method
    assert not args.parity or method in C.PARITY_METHODS
    out_dir = os.path.join(C.RESULTS, "parity" if args.parity else "", method)
    os.makedirs(out_dir, exist_ok=True)

    dataset = MouseEmbryoDataset()
    trees, leaf_states, states = dataset.trees, dataset.leaf_states, dataset.states
    held = set(trees) if args.parity else set(C.HOLDOUT)
    held_trees = {k: v for k, v in trees.items() if k in held}    # dataset order, as fig 3
    leaf_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in leaf_states.items()}

    # fig 3's samples, drawn from the held-out trees. the marginal target stays every
    # tree's leaves at t_early (the population marginal); with --parity both are fig 3's.
    candidates = CD.collect_clonal_candidates(
        held_trees, leaf_states, offset=S.CLONAL_OFFSET, time_tol=S.CLONAL_TIME_TOL,
        n=S.N_SUBTREE_LEAVES, seed=S.SEED, min_leaves=S.MIN_CLONE_LEAVES)
    src = CD.collect_marginal_samples(held_trees, leaf_np, time_tol=S.MARGINAL_TIME_TOL,
                                      n_cells=S.N_MARGINAL_CELLS, seed=S.SEED)
    tgt = CD.collect_marginal_samples(trees, leaf_np, time_tol=S.MARGINAL_TIME_TOL,
                                      n_cells=S.N_MARGINAL_CELLS, seed=S.SEED)
    assert [s[:2] for s in src] == [t[:2] for t in tgt]
    samples = [(s[0], s[1], s[2], t[3]) for s, t in zip(src, tgt)]
    print(f"{method}: {len(held_trees)} held-out trees, {len(candidates)} clonal subtrees, "
          f"{len(samples)} marginal pairs", flush=True)

    # coupling, then the distortions on the same model
    clonal, marginal = [], []
    if method in ("vpsde", "vpsde_all"):
        run = held_out_run() if method == "vpsde" else S.EXPERIMENT_DIR
        splitter, _ = load_splitter(run, C.STEP)
        if method == "vpsde":
            assignments = couple(splitter, held_trees, states, 8)
        else:
            with open(S.MODEL_ASSIGNMENTS, "rb") as f:
                assignments = pickle.load(f)
            if not args.parity:
                redo = couple(splitter, held_trees, states, 8)
                print("hero coupling redone on the held-out trees vs saved: "
                      + agreement(redo, assignments, held_trees), flush=True)
        torch.manual_seed(S.SEED)
        clonal = CD.model_clonal(splitter, candidates, held_trees, leaf_states, **ROLLOUT)
        marginal = CD.model_marginal(splitter, samples, held_trees, leaf_states,
                                     time_tol=S.MARGINAL_TIME_TOL, **ROLLOUT)
        for row in clonal + marginal:
            row["method"] = method
    else:
        pkl = PAPER_GENOT.get(method, os.path.join(C.RESULTS, "genot", f"{method}.pkl"))
        if method in PAPER_GENOT:
            with open(pkl, "rb") as f:
                assignments = pickle.load(f)["assignments"]
            if not args.parity:
                redo = genot_assignments(pkl, dataset)
                print("genot coupling redone on every tree vs saved: "
                      + agreement(redo, assignments, trees), flush=True)
        else:
            assignments = genot_assignments(pkl, dataset)
        clonal = CD.genot_clonal(pkl, method, candidates, leaf_np,
                                 n_samples=S.N_GENOT_SAMPLES, seed=S.SEED)
        marginal = CD.genot_marginal(pkl, method, samples, leaf_np, seed=S.SEED)

    assignments = {t: assignments[t] for t in held_trees if t in assignments}
    with open(os.path.join(out_dir, "assignments.pkl"), "wb") as f:
        pickle.dump(assignments, f)

    # every tree goes in so the real linkage reference keeps all replicates; germ-layer
    # crossing and oscillation only count assigned trees, which the asserts check
    name_to_gl, name_to_lineage, lineages, time_bins = annotations(dataset)
    rows = compute_for(assignments, trees, name_to_gl, name_to_lineage, lineages, time_bins)
    value = {(m, tag): v for m, tag, v in rows}
    gl = assignment_germ_layer_crossing_by_bin(held_trees, assignments, name_to_gl,
                                               time_bins, exclude_source="epiblast")
    osc = lineage_oscillation_fraction(held_trees, assignments, name_to_lineage)
    assert np.isclose(value[("germ_layer_crossing", "")], gl["all"])
    assert np.isclose(value[("lineage_oscillation_fraction", "")], osc)

    pd.DataFrame([(method, m, tag, v) for m, tag, v in rows],
                 columns=["method", "metric", "tag", "value"]).to_csv(
        os.path.join(out_dir, "assignment_metrics.csv"), index=False)
    pd.DataFrame(marginal).to_csv(os.path.join(out_dir, "marginal_distortion.csv"), index=False)
    if clonal:
        pd.DataFrame(clonal).to_csv(os.path.join(out_dir, "clonal_distortion.csv"), index=False)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
