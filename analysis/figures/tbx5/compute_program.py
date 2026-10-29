#!/usr/bin/env python
"""the posterior SHF program, fit on the annotation's own anterior and posterior SHF
labels and read on the heart tube reference cells.

the axis is an LDA over the devmap cells the annotation calls second heart field
(anterior) or (posterior), taken at the stages that carry both - the posterior label
runs out before E9.5, so no cell the panel scores is in the fit and the axis is not the
E9.5 subtypes read back to themselves. the features are the cell's log1p CP10K over the
devmap panel, kept where the gene is detected in at least MIN_DETECT of the fit cells
and is not mitochondrial, and z-scored on the fit cells; the classifier is an LDA with
Ledoit-Wolf shrinkage. the heart tube reference cells - `compute_distortion.py`'s, so a
point on this panel is a point on row 3's maps - are then centred and scaled the same
way, projected on that direction, and the projection standardised over them.

    python compute_distortion.py && python compute_program.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
import data

ANTERIOR, POSTERIOR = S.PROGRAM_LABELS


def fit_cells():
    """the devmap cells the axis is fit on: the two SHF labels at the stages that carry
    both."""
    stage = data.devmap_stages()
    sub = data.devmap_subtypes().reindex(stage.index)
    at = sub.isin(S.PROGRAM_LABELS) & stage.isin(S.PROGRAM_STAGES)
    return pd.DataFrame({"subtype": sub[at], "stage": stage[at]})


def heart_tube():
    """one row per heart tube reference cell: its subtype, whether it is in the
    posterior pole and the nn distortion the model predicts there."""
    d = pd.read_csv(f"{S.NR}/distortion_cells.csv", index_col=0)
    d = d[d.cell_type.isin(S.HEART_TUBE)]
    return pd.DataFrame({"subtype": d.subtype,
                         "posterior": d.subtype.isin(S.POSTERIOR),
                         S.PRED: d[S.PRED]})


def main():
    f, d = fit_cells(), heart_tube()
    X, var = data.devmap_expression(f.index)
    keep = ((X > 0).mean(0) >= S.MIN_DETECT) & ~var.mito.values
    X, genes = X[:, keep], var.index[keep]
    mu, sd = X.mean(0), X.std(0)
    y = (f.subtype == POSTERIOR).values
    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    m.fit((X - mu) / sd, y)
    w = m.coef_.ravel()
    print(f"fit on {len(f):,} cells over {X.shape[1]:,} genes, "
          f"{y.mean():.3f} posterior SHF, stages {', '.join(S.PROGRAM_STAGES)}",
          flush=True)

    Xr, _ = data.devmap_expression(d.index)
    s = ((Xr[:, keep] - mu) / sd) @ w
    d["score"] = (s - s.mean()) / s.std()
    d.to_csv(f"{S.NR}/program_scores.csv")

    g = pd.DataFrame({"gene": genes, "loading": w / np.abs(w).max()})
    g.sort_values("loading", ascending=False).to_csv(f"{S.NR}/program_genes.csv",
                                                     index=False)

    sf = ((X - mu) / sd) @ w
    auc_fit = roc_auc_score(y, sf)
    auc = roc_auc_score(d.posterior, d.score)
    r = d.score.corr(d[S.PRED])
    pd.DataFrame([{"fit_cells": len(f), "cells": len(d), "genes": X.shape[1],
                   "auc_fit": auc_fit, "auc": auc, "pearson": r,
                   "spearman": d.score.corr(d[S.PRED], method="spearman")}]).to_csv(
        f"{S.NR}/program_fit.csv", index=False)
    print(f"transferred to E9.5 the axis calls the posterior pole at auc {auc:.3f} "
          f"and runs with the predicted distortion at r = {r:.3f} "
          f"(in-sample auc {auc_fit:.3f})", flush=True)
    print(f"\n=== up in the posterior SHF\n"
          f"{g.nlargest(10, 'loading').round(3).to_string(index=False)}", flush=True)
    print(f"\n=== up in the anterior SHF\n"
          f"{g.nsmallest(10, 'loading').round(3).to_string(index=False)}", flush=True)
    b = d.groupby("subtype").agg(n=("score", "size"), score=("score", "mean"),
                                 predicted=(S.PRED, "mean"))
    print(f"\n=== mean score per subtype\n{b.round(3).to_string()}", flush=True)


if __name__ == "__main__":
    main()
