"""panels a and b: per cell type convergence and OT distortion, normalized (fig 4 panel
b) and raw (before the division by the leaf's local spacing). both come out of fig 4's
own aggregation, `compute_umap_scores.conv_vs_ot`, over the same dual-scored leaves.

    python -m analysis.figures.supp_conv.compute_celltypes
"""
import os

import pandas as pd
from scipy.stats import spearmanr

from analysis.figures.convergence import compute_umap_scores as CU
from analysis.figures.supp_conv import style as S


def main():
    U = pd.read_pickle(os.path.join(S.C.NR, "umap_scores.pkl"))
    norm = CU.conv_vs_ot(U)
    raw = CU.conv_vs_ot(U.assign(conv_norm=U.cov_trace, ot_norm=U.l2_distance))
    T = norm.merge(raw[["cell_type", "conv", "ot"]], on="cell_type",
                   suffixes=("", "_raw"), validate="1:1")
    T = T[~T.lineage.isin(S.C.CONV_OT_EXCLUDE_LINEAGES)].reset_index(drop=True)
    for c in ("conv", "ot", "conv_raw", "ot_raw"):
        T[f"rank_{c}"] = T[c].rank(ascending=False, method="min").astype(int)
    T["family"] = T.cell_type.map(S.family)
    T = T.sort_values("rank_conv")
    T.to_csv(os.path.join(S.NR, "celltype_scores.csv"), index=False)

    print(f"{len(T)} cell types (>= {S.C.MIN_CELLS} dual-scored leaves, blood dropped)")
    for s in ("", "_raw"):
        rho = spearmanr(T[f"conv{s}"], T[f"ot{s}"])[0]
        print(f"  {s.strip('_') or 'norm':<5} rho {rho:.2f}")
        for fam in ("endothelial", "neuron"):
            f = T[T.family == fam]
            print(f"        {fam:<12} conv ranks {sorted(f[f'rank_conv{s}'])}  "
                  f"ot ranks {sorted(f[f'rank_ot{s}'])}")
    print(f"wrote celltype_scores.csv to {S.NR}")


if __name__ == "__main__":
    main()
