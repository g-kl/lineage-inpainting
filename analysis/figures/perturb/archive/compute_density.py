"""row-1 cache (the only h5ad step): Nkx2-5 expression + log-density-ratio UMAPs.

Nkx2-5 (the hero TF) and the Sierant et al. heart genes are all read out of the E9.5
HVG knockout screen (same screen the ring uses), which shares one control set.  for a
knockout, fit logistic regression on the scVI latent separating its landed readout
cells from control's, score every E9.5 reference cell, and subtract the prior log-odds
-- a per-cell log density ratio (red enriched / blue depleted in the KO vs control).
on-manifold leaves are selected with S.OFF_RATIO (drops off-manifold subtrees).  cache,
all over the same reference cells:
  ctrl_leaf_xy / ctrl_leaf_expr  control-tree leaves, log1p Nkx2-5 expr   (panel 1)
  nkx_ratio                      Nkx2-5 knockout density ratio            (panel 3)
  mean_ratio                     mean density ratio over the heart genes  (panel 4)
the only step that needs the scVI states (the 55GB h5ad).
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from datasets.mouse_embryo import MouseEmbryoDataset
from analysis.pipelines.perturb.perturb_umap_lib import leaf_rows, load_gene, load_control
from analysis.figures.perturb import style as S


def heart_gene_list(genes_dir):
    """cased Sierant heart-gene names that have a knockout pkl in the HVG screen."""
    cased = pd.read_csv(S.HEART_GENES_CSV)["cased_gene"].tolist()
    return [g for g in cased if os.path.exists(os.path.join(genes_dir, f"{g}.pkl"))]


def main():
    print("loading dataset (scVI embeddings) ...", flush=True)
    dataset = MouseEmbryoDataset()
    states = dataset.states
    scvi = np.asarray(states.obsm["X_scvi"], dtype=np.float32)
    name_to_row = {n: i for i, n in enumerate(states.obs_names)}
    nkx_idx = list(states.var_names).index(S.HERO_GENE)
    counts = states.layers["counts"]

    rng = np.random.default_rng(S.seed)
    ref = np.where(np.isclose(states.obs["time"].astype(float), S.RING_TARGET_TIME))[0]
    if len(ref) > S.N_REF:
        ref = rng.choice(ref, S.N_REF, replace=False)
    ref_names = states.obs_names[ref]
    ref_xy = dataset.umap_coords.reindex(ref_names)[["UMAP1", "UMAP2"]].to_numpy(float)
    ref_latent = scvi[ref]
    print(f"  {len(ref)} reference cells at t={S.RING_TARGET_TIME}", flush=True)

    def log_density_ratio(pos_rows, neg_rows):
        pos, neg = scvi[pos_rows], scvi[neg_rows]
        X = np.vstack([pos, neg])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        return clf.decision_function(ref_latent) - np.log(len(pos) / len(neg))

    def expr_of(rows):
        ge = counts[rows, nkx_idx]
        ge = ge.toarray().ravel() if sp.issparse(ge) else np.asarray(ge).ravel()
        return np.log1p(ge)

    # E9.5 HVG screen: one shared control set for both the Nkx2-5 and heart panels.
    genes_dir = os.path.join(S.RING_SCREEN, "genes")
    ctrl_rows = leaf_rows(load_control(os.path.join(S.RING_SCREEN, "control")),
                          S.RING_TARGET_TIME, S.TOL, name_to_row, ratio=S.OFF_RATIO)

    # --- panel 1: control-tree leaves hued by Nkx2-5 expression
    ctrl_leaf_xy = dataset.umap_coords.reindex(states.obs_names[ctrl_rows])
    ctrl_leaf_xy = ctrl_leaf_xy[["UMAP1", "UMAP2"]].to_numpy(float)
    ctrl_leaf_expr = expr_of(ctrl_rows)
    print(f"  {len(ctrl_rows)} control leaves (panel 1)", flush=True)

    # --- panel 3: Nkx2-5 knockout density ratio
    nkx_rows = leaf_rows(load_gene(genes_dir, S.HERO_GENE), S.RING_TARGET_TIME, S.TOL,
                         name_to_row, ratio=S.OFF_RATIO)
    nkx_ratio = log_density_ratio(nkx_rows, ctrl_rows)
    print(f"  {S.HERO_GENE} KO density ratio: mean={nkx_ratio.mean():+.2f}", flush=True)

    # --- panel 4: mean log-density ratio over the Sierant heart genes
    genes = heart_gene_list(genes_dir)
    print(f"computing density ratio for {len(genes)} heart genes ...", flush=True)
    ratios = []
    for g in genes:
        rows = leaf_rows(load_gene(genes_dir, g), S.RING_TARGET_TIME, S.TOL,
                         name_to_row, ratio=S.OFF_RATIO)
        r = log_density_ratio(rows, ctrl_rows)
        ratios.append(r)
        print(f"  {g}: mean={r.mean():+.2f}", flush=True)
    mean_ratio = np.mean(np.vstack(ratios), axis=0)

    out = os.path.join(S.NUMERICAL, "density.npz")
    np.savez(out, ref_xy=ref_xy, nkx_ratio=nkx_ratio, mean_ratio=mean_ratio,
             ctrl_leaf_xy=ctrl_leaf_xy, ctrl_leaf_expr=ctrl_leaf_expr,
             genes=np.array(genes, dtype=object))
    print(f"wrote {out}  (mean over heart genes = {mean_ratio.mean():+.2f})", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
