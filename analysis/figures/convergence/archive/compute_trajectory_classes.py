"""rows 4-5 (v2): per-(gene, source) trajectory classes along the EC fate-bias axis.

reuses the validated pilot: fatebias_manual_trajectory_classes.load() builds one
raw-log trajectory per (gene, source) (gaussian-smoothed vs EC fate bias) and .classify()
rule-labels each into the 7 shapes in style.TRAJ_CLASS_LABELS. this script caches the
trajectories + labels, the source-source similarity matrix (fraction of genes in the same
non-flat class), and the row-5 example genes per behaviour group.

    python analysis/figures/convergence/compute_trajectory_classes.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import style as S
import fatebias_trajectory_distance as TD             # noqa: E402  (read_tf_list)
import fatebias_manual_trajectory_classes as MC       # noqa: E402  (load / classify)


def similarity(lab, n_src):
    """source x source: among genes in a non-flat class in BOTH sources, the fraction
    assigned the SAME class -- of the genes actively moving in both tissues, how often
    they move the same way."""
    nonflat = (lab > 0) & ~np.isin(lab, S.TRAJ_FLAT_CLASSES)
    M = np.full((n_src, n_src), np.nan)
    for i in range(n_src):
        for j in range(n_src):
            if i == j:
                continue
            both = nonflat[:, i] & nonflat[:, j]
            if both.sum():
                M[i, j] = (lab[both, i] == lab[both, j]).mean()
    return M


def pick_examples(lab, slope_mat, rng_mat, is_tf, gnames):
    """row-5 genes per behaviour group, TFs first:
      shared   -- rising strongly (class 3) in EVERY covered source (shared EC program)
      memory   -- falling strongly (class 5) in ONE source, stable (1/2) in the rest
      dynamics -- transient (class 7) in some sources but rising (3/4) in others."""
    n_cov = (lab > 0).sum(1)
    c3 = (lab == 3).sum(1)
    c5 = (lab == 5).sum(1)
    c7 = (lab == 7).sum(1)
    stable = np.isin(lab, S.TRAJ_FLAT_CLASSES).sum(1)
    rising = ((lab == 3) | (lab == 4)).sum(1)

    def take(cand, score):
        order = cand[np.lexsort((-score[cand], ~is_tf[cand]))]   # TFs first, then score
        return [gnames[g] for g in order[:S.N_TRAJ_EXAMPLE]]

    shared = np.where((n_cov >= 2) & (c3 == n_cov))[0]
    mem = np.where((c5 == 1) & (stable == n_cov - 1) & (n_cov >= 2))[0]
    dyn = np.where((c7 >= 1) & (rising >= 1) & (c7 < n_cov))[0]
    with warnings.catch_warnings():                                # all-NaN non-candidates
        warnings.simplefilter("ignore", RuntimeWarning)
        shared_sc = np.nanmean(np.where(lab == 3, slope_mat, np.nan), 1)
        mem_sc = -np.nanmin(np.where(lab == 5, slope_mat, np.nan), 1)   # steepest faller
        dyn_sc = c7 + np.nan_to_num(
            np.nanmax(np.where(lab == 7, rng_mat, np.nan), 1)) / 10.0

    return {"shared": take(shared, shared_sc), "memory": take(mem, mem_sc),
            "dynamics": take(dyn, dyn_sc)}


def main():
    grid, feat, r_gene, r_src, sources, gnames = MC.load()
    cid, slope, level, rng = MC.classify(feat, grid)
    G, n_src = len(gnames), len(sources)

    is_tf = np.array([g in TD.read_tf_list(S.TF_LIST) for g in gnames])
    lab = np.zeros((G, n_src), int)
    lab[r_gene, r_src] = cid
    slope_mat = np.full((G, n_src), np.nan); slope_mat[r_gene, r_src] = slope
    rng_mat = np.full((G, n_src), np.nan); rng_mat[r_gene, r_src] = rng

    print("class sizes:", flush=True)
    for c, name in S.TRAJ_CLASS_LABELS.items():
        m = cid == c
        print(f"  {c} {name:16s}: rows={m.sum():5d}  genes={len(np.unique(r_gene[m])):4d}",
              flush=True)

    sim = similarity(lab, n_src)
    ex = pick_examples(lab, slope_mat, rng_mat, is_tf, gnames)
    for k, genes in ex.items():
        print(f"  row5 {k:9s}: {', '.join(genes)}", flush=True)

    np.savez(os.path.join(S.NR, "traj_classes.npz"), grid=grid,
             feat=feat.astype(np.float32), r_gene=r_gene, r_src=r_src, cid=cid,
             sources=np.array(sources), gnames=gnames, is_tf=is_tf)
    pd.DataFrame(sim, index=sources, columns=sources).to_csv(
        os.path.join(S.NR, "traj_similarity.csv"))
    pd.DataFrame(lab, columns=[s.replace(" ", "_") for s in sources]).assign(
        gene=gnames, is_tf=is_tf).to_csv(
        os.path.join(S.NR, "traj_class_assignments.csv"), index=False)
    pd.DataFrame([(k, i, g) for k, gs in ex.items() for i, g in enumerate(gs)],
                 columns=["group", "order", "gene"]).to_csv(
        os.path.join(S.NR, "traj_row5_examples.csv"), index=False)
    print("wrote traj_classes.npz, traj_similarity.csv, traj_class_assignments.csv, "
          "traj_row5_examples.csv", flush=True)


if __name__ == "__main__":
    main()
