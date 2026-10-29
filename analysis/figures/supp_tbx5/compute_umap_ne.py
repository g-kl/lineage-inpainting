"""f to h: a fresh umap of the integrated latent restricted to neural ectoderm, both
datasets on one set of coordinates, and the neurons per embryo.

the draw is balanced across the datasets, so a region one side is missing from shows as
a region of one colour. neurons are the annotation's `neuron` cluster. writes
numerical_results/umap_ne.parquet and ne_counts.csv.

    python -m analysis.figures.supp_tbx5.compute_umap_ne     # gpu, rapids_singlecell env
"""
import os

import numpy as np
import pandas as pd
from cuml.manifold import UMAP

from analysis.figures.supp_tbx5 import style as S
from analysis.pipelines.tbx5.support_comparison import common as C


def main():
    d = C.cells()
    z = np.vstack(C.latent())
    ti = pd.read_csv(C.TYPE_INFO)
    neuron = d.cell_type.map(dict(zip(ti.cell_type, ti.cluster))).eq("neuron")

    ne = d.lineage.eq(S.NE).values
    rng = np.random.default_rng(S.SEED)
    n = min(S.N_NE, *(d[ne].dataset.value_counts()))
    at = np.sort(np.concatenate([rng.choice(np.flatnonzero(ne & d.dataset.eq(k).values),
                                            n, replace=False) for k in (S.REF, S.CRI)]))
    xy = np.asarray(UMAP(n_neighbors=S.N_UMAP_NEIGHBORS, min_dist=S.UMAP_MIN_DIST,
                         random_state=S.SEED).fit_transform(z[at]))
    u = d.iloc[at][["dataset", "cell_type"]].copy()
    u["x"], u["y"] = xy[:, 0], xy[:, 1]
    u.to_parquet(os.path.join(S.NR, "umap_ne.parquet"))

    c = d[ne].assign(neuron=neuron.values[ne]).groupby(["dataset", "embryo"]).agg(
        cells=("neuron", "size"), neurons=("neuron", "sum"))
    c["share"] = c.neurons / c.cells
    c.to_csv(os.path.join(S.NR, "ne_counts.csv"))
    print(f"{n:,} NE cells a dataset drawn", flush=True)


if __name__ == "__main__":
    main()
