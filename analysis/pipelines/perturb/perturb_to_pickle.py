"""Causal TF->gene flow from a projection knockout screen, end to end.

Point it at a screen dir (control/*.pkl + genes/*.pkl) and it:

  1. builds the causal TF x gene mean-delta matrix per timepoint
        delta_t[i, gene] = mean expr(perturbed snapped @ t) - mean expr(control snapped @ t)
     with expression = log1p(counts) of the snapped observed cell; then

  2. reduces each timepoint to its TF-specific signal -- unit-norm every TF profile and
     project out the consensus (mean unit direction = the shared global axis) -- and
     squares to a gene x gene flow over the participating universe (TFs + strongest
     targets), writing a per-timepoint gene x gene flow structure.

Stage 1 needs the dataset and is slow; its matrix is cached at <screen>/perturb_delta.pkl
and reused on re-runs (pass --recompute to rebuild, e.g. after changing --ratio), so
re-tuning the flow is instant.

Usage:
    python analysis/pipelines/perturb/perturb_to_pickle.py --screen-dir tf_projection_screen
"""
import argparse
import glob
import os
import pickle

import numpy as np
import networkx as nx
import scipy.sparse as sp

from datasets.mouse_embryo import MouseEmbryoDataset

TOL = 0.25


def off_manifold(tree, ratio):
    """nodes (and their subtrees) that projected off-manifold at ``ratio``."""
    bad = set()
    for n, d in tree.nodes(data=True):
        ld = d.get("local_dist")
        if d.get("changed") and ld is not None and d.get("proj_dist", 0.0) > ratio * ld:
            bad.add(n)
            bad.update(nx.descendants(tree, n))
    return bad


def snapped_by_time(tree, ratio, times):
    """{t: [snapped_name, ...]} for on-manifold nodes nearest each timepoint."""
    bad = off_manifold(tree, ratio)
    out = {t: [] for t in times}
    for n, d in tree.nodes(data=True):
        sn, tt = d.get("snapped_name"), d.get("time")
        if n in bad or sn is None or tt is None:
            continue
        t = min(times, key=lambda x: abs(x - tt))
        if abs(t - tt) < TOL:
            out[t].append(sn)
    return out


def compute_delta(screen, ratio):
    """TF x gene causal mean-delta per timepoint -> {t: {c, row_genes, col_genes, n_cells}}."""
    states = MouseEmbryoDataset().states
    genes = list(states.var_names)
    row_of = {n: i for i, n in enumerate(states.obs_names)}
    X = sp.csr_matrix(states.X, dtype=np.float64)
    X.data = np.log1p(X.data)

    def mean_expr(names):
        rows = [row_of[s] for s in names if s in row_of]
        return (np.asarray(X[rows].sum(0)).ravel() / len(rows)) if rows else None, len(rows)

    ctrl_files = glob.glob(f"{screen}/control/*.pkl")
    times = sorted({round(d["time"], 4)
                    for _, d in pickle.load(open(ctrl_files[0], "rb"))["tree"].nodes(data=True)
                    if d.get("time") is not None})
    print(f"timepoints: {times}", flush=True)

    def screen_means(tree_dicts):
        """{t: mean log1p expr vector} over the given trees' on-manifold nodes at t."""
        acc = {t: [] for t in times}
        for d in tree_dicts:
            for t, names in snapped_by_time(d["tree"], ratio, times).items():
                acc[t] += names
        return {t: mean_expr(acc[t]) for t in times}

    ctrl = screen_means([pickle.load(open(f, "rb")) for f in ctrl_files])
    for t in times:
        print(f"  control t={t}: {ctrl[t][1]} cells", flush=True)

    tf_files = sorted(glob.glob(f"{screen}/genes/*.pkl"))
    tfs, delta = [], {t: [] for t in times}
    for i, f in enumerate(tf_files):
        pm = screen_means(list(pickle.load(open(f, "rb")).values()))
        if any(pm[t][0] is None or ctrl[t][0] is None for t in times):
            continue
        tfs.append(os.path.splitext(os.path.basename(f))[0])
        for t in times:
            delta[t].append(pm[t][0] - ctrl[t][0])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(tf_files)} TFs", flush=True)

    return {t: {"c": np.asarray(delta[t], dtype=np.float32),
                "row_genes": tfs, "col_genes": genes, "n_cells": ctrl[t][1]}
            for t in times}


def specific(C):
    U = C / np.maximum(np.linalg.norm(C, axis=1), 1e-12)[:, None]
    cons = U.mean(0)
    cons /= np.linalg.norm(cons)
    return U - np.outer(U @ cons, cons)


def build_flow(d, n_targets):
    """TF-specific signal -> gene x gene flow (per-timepoint results structure)."""
    times = sorted(d)
    tfs = d[times[0]]["row_genes"]
    genes = np.asarray(d[times[0]]["col_genes"])
    gidx = {g: i for i, g in enumerate(genes)}
    D = {t: specific(d[t]["c"].astype(np.float64)).astype(np.float32) for t in times}

    # node universe = TFs + their strongest targets; square gene x gene over it.
    strength = np.max([np.abs(D[t]).max(0) for t in times], axis=0)
    keep_targets = set(genes[np.argsort(strength)[::-1][:n_targets]])
    tf_set = set(tfs)
    universe = [g for g in genes if g in tf_set or g in keep_targets]
    uidx = {g: i for i, g in enumerate(universe)}
    ucol = [gidx[g] for g in universe]
    C_t = {}
    for t in times:
        M = np.zeros((len(universe), len(universe)), dtype=np.float32)
        sub = D[t][:, ucol]                           # only TF rows carry effects
        for r, tf in enumerate(tfs):
            M[uidx[tf]] = sub[r]
        C_t[t] = M

    # results structure: edge t1->t2 uses the effect present by t2.
    results = {}
    for i, t2 in enumerate(times):
        entry = {"c": C_t[t2], "gene_names": universe, "n_pairs": d[t2]["n_cells"]}
        results[t2] = {t2: entry}
        if i > 0:
            results[t2][times[i - 1]] = entry
    return results, universe, tfs, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-dir", default="tf_projection_screen")
    ap.add_argument("--ratio", type=float, default=1.2, help="off-manifold threshold")
    ap.add_argument("--n-targets", type=int, default=1000,
                    help="top target genes (by max |delta|) kept alongside the TFs")
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild the cached delta matrix instead of reusing it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    screen = args.screen_dir

    matrix_path = f"{screen}/perturb_delta.pkl"
    if os.path.exists(matrix_path) and not args.recompute:
        d = pickle.load(open(matrix_path, "rb"))
        print(f"loaded cached matrix {matrix_path} (--recompute to rebuild)", flush=True)
    else:
        d = compute_delta(screen, args.ratio)
        pickle.dump(d, open(matrix_path, "wb"))
        print(f"wrote {matrix_path}: {len(d[sorted(d)[0]]['row_genes'])} TFs", flush=True)

    results, universe, tfs, times = build_flow(d, args.n_targets)
    out = args.out or f"{screen}/perturb_delta_flow.pkl"
    pickle.dump(results, open(out, "wb"))
    print(f"wrote {out}: {len(universe)} genes ({len(tfs)} TFs + targets), times {times}", flush=True)


if __name__ == "__main__":
    main()
