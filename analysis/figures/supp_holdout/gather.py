"""merge the per-method results and score hox across them.

hox keeps only the ancestors every method assigns, so it needs all the couplings and
runs here rather than in evaluate.py. --parity diffs the merged caches against
analysis/figures/summary_qc/numerical_results/.

    python -m analysis.figures.supp_holdout.gather [--parity]
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from analysis.figures.summary_qc import compute_hox as H
from analysis.figures.summary_qc import style as S
from analysis.figures.supp_holdout import common as C
from datasets.mouse_embryo import MouseEmbryoDataset
from literature.hox_consistency import static_hox_pairs

PER_METHOD = ["clonal_distortion.csv", "marginal_distortion.csv", "assignment_metrics.csv"]
# the paper's models under their fig 3 names, for the parity diff
FIG3_NAME = {"vpsde_all": "model", "genot_sinkhorn_all": "genot_sinkhorn",
             "genot_cbfgw_all": "genot_cbfgw"}


def hox(methods, out_dir):
    """compute_hox.main's pearson r per (method, cell type, gene), on every method's common support."""
    dataset = MouseEmbryoDataset()
    genes_by_ct = defaultdict(list)
    for ct, gene in static_hox_pairs:
        genes_by_ct[ct].append(gene)
    var_names = set(dataset.states.var_names)
    present = [g for g in sorted({g for _, g in static_hox_pairs}) if g in var_names]
    hox_expr = np.asarray(dataset.states[:, present].X.todense(), dtype=np.float32)
    gene_col = {g: j for j, g in enumerate(present)}
    row = dataset.name_to_idx

    assignments = {}
    for m in methods:
        with open(os.path.join(out_dir, m, "assignments.pkl"), "rb") as f:
            assignments[m] = pickle.load(f)
    paper_methods = S.METHODS
    S.METHODS = methods     # collect_pairs reads its method list from style
    try:
        pairs = H.collect_pairs(dataset, assignments, set(genes_by_ct))[0]
    finally:
        S.METHODS = paper_methods

    rows = []
    for m in methods:
        for ct in sorted(genes_by_ct):
            for gene in genes_by_ct[ct]:
                if gene not in gene_col:
                    continue
                ok = [(row[l], row[a]) for l, a in pairs[(m, ct)] if l in row and a in row]
                desc = [hox_expr[i, gene_col[gene]] for i, _ in ok]
                anc = [hox_expr[j, gene_col[gene]] for _, j in ok]
                if len(desc) >= 3 and np.std(desc) > 0 and np.std(anc) > 0:
                    r, _ = pearsonr(desc, anc)
                    rows.append((m, ct, gene, len(desc), r, 1.0 - r))
    return pd.DataFrame(rows, columns=["method", "cell_type", "gene", "n", "r", "distortion"])


def parity_diff(out_dir):
    """our merged caches (paper models, every tree) against fig 3's."""
    for name, keys, col in [("assignment_metrics.csv", ["metric", "tag"], "value"),
                            ("hox_consistency.csv", ["cell_type", "gene", "n"], "r"),
                            ("clonal_distortion.csv", ["tree", "leaf"], "energy_distance"),
                            ("marginal_distortion.csv", ["t_src", "t_tgt"], "energy_distance")]:
        ours = pd.read_csv(os.path.join(out_dir, name)).replace({"method": FIG3_NAME})
        fig3 = pd.read_csv(os.path.join(S.NR, name))
        if "tag" in keys:
            ours["tag"], fig3["tag"] = ours["tag"].fillna(""), fig3["tag"].fillna("")
        m = ours.merge(fig3, on=["method"] + keys, how="outer", suffixes=("", "_fig3"),
                       indicator=True)
        for method, g in m.groupby("method"):
            both = g[g["_merge"] == "both"]
            diff = (both[col] - both[col + "_fig3"]).abs()
            print(f"{name:26s}{method:16s}rows ours {int((g['_merge'] != 'right_only').sum()):4d}"
                  f"  fig3 {int((g['_merge'] != 'left_only').sum()):4d}  both {len(both):4d}"
                  f"  max |diff| {diff.max():.2e}"
                  f"  mean ours {both[col].mean():.5f}  fig3 {both[col + '_fig3'].mean():.5f}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parity", action="store_true", help="the parity run's results")
    args = ap.parse_args()
    methods = C.PARITY_METHODS if args.parity else C.METHODS
    out_dir = os.path.join(C.RESULTS, "parity") if args.parity else C.RESULTS
    missing = [m for m in methods
               if not os.path.exists(os.path.join(out_dir, m, "assignment_metrics.csv"))]
    assert not missing, f"evaluate.py has not finished for {missing}"

    for name in PER_METHOD:
        parts = [os.path.join(out_dir, m, name) for m in methods]
        pd.concat([pd.read_csv(p) for p in parts if os.path.exists(p)]).to_csv(
            os.path.join(out_dir, name), index=False)
    hox(methods, out_dir).to_csv(os.path.join(out_dir, "hox_consistency.csv"), index=False)
    print(f"wrote {out_dir}", flush=True)

    if args.parity:
        parity_diff(out_dir)


if __name__ == "__main__":
    main()
