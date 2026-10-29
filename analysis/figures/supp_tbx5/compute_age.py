"""e: lineage age, how near each lineage's crispant cells sit to E9.0 against E9.5.

a lineage is aged if it has MIN_INDEX devmap cells at each of AGE_STAGES and MIN_INDEX
crispant cells, blood and the extraembryonic lineages left out as in the support
comparison. its index is up to N_INDEX of its devmap cells from each stage, cut to the
smaller so both hold the same number, from embed_stages.py. its crispant cells, pooled
over the embryos, up to N_QUERY of them, are scored by the energy distance (the unbiased
U-statistic) to each stage's index cells. writes numerical_results/age_energy.csv.

    python -m analysis.figures.supp_tbx5.compute_age     # after embed_stages
"""
import os

import numpy as np
import pandas as pd

from analysis.figures.supp_tbx5 import style as S
from analysis.lib.metrics.energy_distance import energy_distance
from analysis.pipelines.tbx5.support_comparison import common as C


def main():
    rng = np.random.default_rng(S.SEED)
    io = pd.concat([pd.read_parquet(os.path.join(S.NR, f"stage_obs_{s}.parquet"))
                    .assign(stage=s) for s in S.AGE_STAGES])
    zi = np.vstack([np.load(os.path.join(S.NR, f"stage_latent_{s}.npy"))
                    for s in S.AGE_STAGES])
    d = C.cells()
    zq = np.vstack(C.latent())
    d["row"] = np.arange(len(d))
    cri = d[d.dataset == S.CRI]

    n_at = io[io.lineage.isin(C.lineages(d))].groupby(["lineage", "stage"]).size()
    n_at = n_at.unstack().reindex(columns=S.AGE_STAGES).fillna(0).min(1)
    n_cri = cri.lineage.value_counts()
    lins = sorted(l for l, n in n_at.items()
                  if n >= S.MIN_INDEX and n_cri.get(l, 0) >= S.MIN_INDEX)

    rows = []
    for l in lins:
        n = int(min(S.N_INDEX, n_at[l]))
        q = cri.row.values[cri.lineage.eq(l).values]
        pick = rng.choice(q, min(len(q), S.N_QUERY), replace=False)
        rows.append({"lineage": l, "cells": len(q), "n_index": n, **{
            s: energy_distance(zq[pick], zi[rng.choice(np.flatnonzero(
                io.lineage.eq(l).values & io.stage.eq(s).values), n, replace=False)])
            for s in S.AGE_STAGES}})
    a = pd.DataFrame(rows)
    a.to_csv(os.path.join(S.NR, "age_energy.csv"), index=False)
    print(a.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
