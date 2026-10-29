#!/usr/bin/env python
"""the two maps row 3 is drawn on.

`umap_all` is the pipeline's integrated umap over both datasets, a draw balanced across
them. `umap_heart` is a fresh umap of the integrated latent restricted to the heart tube
- the cardiac cell types without the endocardium, which sits as its own island in this
latent - with both datasets on the same coordinates.

    python compute_umap.py       # gpu, under the rapids_singlecell env
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
import data

COLS = ["dataset", "subtype", "cell_type", "lineage", "x", "y"]


def main():
    d = data.cells()
    v = data.int_umap()
    v[COLS].to_parquet(f"{S.NR}/umap_all.parquet")

    zc, zd = data.latent()
    z = np.vstack([zc, zd])
    assert len(z) == len(d)
    at = np.flatnonzero(d.cell_type.isin(S.HEART_TUBE).values)

    from cuml.manifold import UMAP
    xy = np.asarray(UMAP(n_neighbors=S.N_UMAP_NEIGHBORS, min_dist=S.UMAP_MIN_DIST,
                         random_state=S.SEED).fit_transform(z[at]))
    u = d.iloc[at][["dataset", "subtype", "cell_type", "lineage"]].copy()
    u["x"], u["y"] = xy[:, 0], xy[:, 1]
    u.to_parquet(f"{S.NR}/umap_heart.parquet")

    n = pd.DataFrame({"cells": d.dataset.value_counts(),
                      "drawn": v.dataset.value_counts(),
                      "heart_tube": u.dataset.value_counts()})
    n.index.name = "dataset"
    n.to_csv(f"{S.NR}/cell_counts.csv")
    print(n.to_string(), flush=True)
    print(u.groupby("dataset").subtype.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
