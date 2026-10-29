#!/usr/bin/env python
"""umap of the integrated latent, N_UMAP cells per dataset, cached.

writes umap_integrated.parquet. delete it to rebuild.

    python -m analysis.pipelines.tbx5.support_comparison.integrated_umap
"""
import os

import numpy as np
import pandas as pd

from analysis.pipelines.tbx5.support_comparison import common as C

N_UMAP = 60_000     # cells per side
K_UMAP = 15
CACHE = "umap_integrated.parquet"


def umap(z, index, cache, k=K_UMAP):
    """cached umap of a latent block."""
    path = C.out(cache)
    if os.path.exists(path):
        return pd.read_parquet(path)
    import anndata as ad
    import rapids_singlecell as rsc
    a = ad.AnnData(np.zeros((len(z), 1), dtype="float32"))
    a.obsm["X_lat"] = np.ascontiguousarray(z, dtype="float32")
    rsc.pp.neighbors(a, use_rep="X_lat", n_neighbors=k)
    rsc.tl.umap(a, random_state=C.SEED)
    u = pd.DataFrame(np.asarray(a.obsm["X_umap"]), columns=["u1", "u2"], index=index)
    u.to_parquet(path)
    print(f"{cache}: {len(u):,} cells", flush=True)
    return u


def main():
    d = C.cells()
    z = np.vstack(C.latent())            # same row order as C.cells()
    rng = np.random.default_rng(C.SEED)
    keep = np.sort(np.concatenate([
        rng.choice(at, min(N_UMAP, len(at)), replace=False)
        for at in (np.flatnonzero((d.dataset == n).values)
                   for n in ("devmap", C.TARGET))]))
    umap(z[keep], d.index[keep], CACHE)
    print(d.iloc[keep].dataset.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
