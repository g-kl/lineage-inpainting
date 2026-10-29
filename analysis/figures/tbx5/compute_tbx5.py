#!/usr/bin/env python
"""Tbx5 per devmap cell, averaged over each cell's SMOOTH_K nearest devmap neighbours
in the devmap latent. raw counts of one gene in one cell are mostly zero, so the trees'
nodes read this rather than their own snapped cell.

    python compute_tbx5.py       # gpu, under the rapids_singlecell env
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
import data
from analysis.pipelines.tbx5.support_comparison import common as TC


def main():
    v = data.devmap_counts(S.TARGET)
    lat = data.devmap_latent()
    print(f"{len(v):,} devmap cells, {(v > 0).mean():.3f} with a {S.TARGET} count",
          flush=True)

    nb = TC.knn(lat, lat, S.SMOOTH_K)
    sm = v[nb].mean(1).astype("float32")
    np.save(f"{S.NR}/tbx5_devmap.npy", sm)

    q = np.percentile(sm, [50, 90, 99, 100])
    print(f"smoothed over k = {S.SMOOTH_K}: median {q[0]:.3f}, p90 {q[1]:.3f}, "
          f"p99 {q[2]:.3f}, max {q[3]:.3f}; {(sm >= S.TBX5_ON).mean():.3f} of cells "
          f"at or above {S.TBX5_ON}", flush=True)


if __name__ == "__main__":
    main()
