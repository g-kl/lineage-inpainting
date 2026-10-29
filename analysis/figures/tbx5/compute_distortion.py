#!/usr/bin/env python
"""row 3: the per embryo nn distortion inside the heart, re-cut off the analysis
pipeline.

the statistic is analysis/pipelines/tbx5/support_comparison/per_embryo.py's: per E9.5
reference cell, the log ratio of its nearest neighbour distance into a perturbed draw
over its distance into a draw from another reference embryo. every draw is taken inside the heart, the
crispant embryo is the unit and its numerator is averaged over draws, and the ratio is
centred on its median over the scope. the perturbed side is either the crispants
(`measured`) or the screen's Tbx5 leaves resolved to reference cells under `freeze`
(`predicted`). this script only re-cuts the pipeline's tables; the epicardium is cut
out of every one of them, though the pipeline's draws still take it in.

    python compute_distortion.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
import data

COLS = ["subtype", "cell_type", S.PRED, S.MEAS, "spread"]


def agreement(b, cells, scope):
    """measured against predicted over the subtypes of one scope, and over its
    reference cells."""
    d = b[[S.MEAS, S.PRED]].replace([np.inf, -np.inf], np.nan).dropna()
    c = cells[[S.MEAS, S.PRED]].replace([np.inf, -np.inf], np.nan).dropna()
    return {"scope": scope, "n_subtypes": len(d), "n_cells": len(c),
            "pearson": d[S.MEAS].corr(d[S.PRED]),
            "spearman": spearmanr(d[S.MEAS], d[S.PRED])[0],
            "slope": float(np.polyfit(d[S.MEAS], d[S.PRED], 1)[0]),
            "pearson_cells": c[S.MEAS].corr(c[S.PRED])}


def main():
    cells = data.distortion(S.HEART_SCOPE)
    cells = cells[cells.cell_type.isin(S.HEART_KEPT)]
    cells[COLS].to_csv(f"{S.NR}/distortion_cells.csv")
    print(f"{len(cells):,} heart reference cells, "
          f"{cells.cell_type.isin(S.HEART_TUBE).sum():,} of them heart tube", flush=True)

    b = data.distortion_subtype(S.HEART_SCOPE)
    b = b[(b.cells >= S.MIN_CELLS) & b.cell_type.isin(S.HEART_KEPT)]
    b.to_csv(f"{S.NR}/distortion_heart.csv")

    tube = cells.cell_type.isin(S.HEART_TUBE)
    a = pd.DataFrame([agreement(b, cells, S.HEART_SCOPE),
                      agreement(b[b.cell_type.isin(S.HEART_TUBE)], cells[tube],
                                "heart tube")])
    a.to_csv(f"{S.NR}/distortion_agreement.csv", index=False)
    data.distortion_support([S.LPM, S.HEART_SCOPE]).to_csv(
        f"{S.NR}/distortion_support.csv", index=False)

    print(a.round(3).to_string(index=False), flush=True)
    print("\n=== per subtype, centred log nn distortion", flush=True)
    print(b[["cells", S.MEAS, "measured_sem", S.PRED]]
          .sort_values(S.MEAS).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
