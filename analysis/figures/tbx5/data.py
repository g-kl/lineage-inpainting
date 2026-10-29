"""loaders for what the sheet does not compute: the demultiplexed crispants, the
integrated latent and its umap, the per embryo nn distortion, the devmap annotation and
counts, and the model's screens. everything but the CHD screen is read through
analysis/pipelines/tbx5/support_comparison/common.py.
"""
import numpy as np
import pandas as pd

import config as S
from analysis.pipelines.tbx5.support_comparison import common as C

screen = C.screen
cells = C.cells
latent = C.latent
HEART_TYPES = C.HEART_TYPES
LEAF_TIME, LEAF_TOL = C.LEAF_TIME, C.LEAF_TOL

assert S.SCREEN_KIND in C.SCREEN, f"pipeline screen is not {S.SCREEN_KIND}: {C.SCREEN}"
assert C.OFF_RATIO == S.OFF_RATIO, f"pipeline off-manifold ratio is {C.OFF_RATIO}"
assert C.MODE == S.MODE, f"pipeline resolved its leaves under {C.MODE}"
assert C.LEAF_TIME == S.LEAF_TIME, f"pipeline reads out at {C.LEAF_TIME}"


def type_map():
    """subtype to cell type, cell type to lineage."""
    return C.type_map()


def devmap_subtypes():
    return C.devmap_subtypes()


def lpm_subtypes():
    """every lateral plate subtype."""
    sub2type, type2lin = C.type_map()
    return sorted(s for s, t in sub2type.items() if type2lin.get(t) == S.LPM)


def int_umap():
    """the pipeline's integrated umap, with each cell's dataset and annotation. a draw
    balanced across the two datasets."""
    u = pd.read_parquet(C.out("umap_integrated.parquet")).rename(
        columns={"u1": "x", "u2": "y"})
    return u.join(cells()[["dataset", "subtype", "cell_type", "lineage"]], how="inner")


def distortion(scope):
    """per reference cell of one pipeline scope: predicted, measured and the spread
    over the crispant embryos, all centred log ratios."""
    return pd.read_csv(C.out(f"per_embryo_{C.slug(scope)}.csv"), index_col=0)


def distortion_subtype(scope):
    """per subtype of one scope, with the sem over the crispant embryos on the measured
    side and the leave-one-tree-out jackknife on the predicted one."""
    return pd.read_csv(C.out(f"per_embryo_{C.slug(scope)}_subtype.csv"), index_col=0)


def distortion_support(scopes):
    """the pipeline's support rows: cells a side, units, trees, n."""
    s = pd.read_csv(C.out("per_embryo_support.csv"))
    return s[s.scope.isin(scopes)]


def transcription_factors():
    """the mouse transcription factors in literature/tf_list.txt."""
    with open(S.TF_LIST) as f:
        return set(f.read().split())


def devmap_counts(gene):
    """log1p CP10K of one devmap gene over every devmap cell, in devmap row order."""
    import h5py
    with h5py.File(C.DEVMAP) as f:
        var = np.array([x.decode() for x in f["var"]["_index"][:]])
        j = int(np.flatnonzero(var == gene)[0])
        total = np.maximum(f["obs"]["total_counts"][:], 1)
        ip = f["X"]["indptr"][:]
        raw = np.zeros(len(total), "float32")
        for s in range(0, len(total), 200_000):
            e = min(s + 200_000, len(total))
            a, b = int(ip[s]), int(ip[e])
            m = f["X"]["indices"][a:b] == j
            rows = np.repeat(np.arange(s, e), np.diff(ip[s:e + 1]))[m]
            raw[rows] = f["X"]["data"][a:b][m]
    return np.log1p(raw / total * 1e4)


def devmap_stages():
    """the stage of every devmap cell, by cell name."""
    import h5py
    with h5py.File(C.DEVMAP) as f:
        e = f["obs"]["embryo"]
        emb = np.array([c.decode() for c in e["categories"]])[e["codes"][:]]
    return pd.Series(pd.Series(emb).str.split("-").str[0].values, index=C.devmap_index())


def devmap_latent():
    """the devmap X_scvi for every devmap cell - the latent the trees snapped in."""
    import h5py
    with h5py.File(C.DEVMAP) as f:
        return f["obsm"]["X_scvi"][:].astype("float32")


def tbx5():
    """the knn-smoothed devmap Tbx5 written by compute_tbx5.py, by cell name."""
    return pd.Series(np.load(f"{S.NR}/tbx5_devmap.npy"), index=C.devmap_index())


def devmap_expression(cells):
    """(log1p CP10K, var) over the whole devmap panel for named cells, in the order
    given. var carries the mito flag."""
    import h5py
    rows = C.devmap_index().get_indexer(pd.Index(cells))
    assert (rows >= 0).all(), "not every cell is in the devmap"
    with h5py.File(C.DEVMAP) as f:
        var = pd.DataFrame({"mito": f["var"]["mito"][:]},
                           index=[x.decode() for x in f["var"]["_index"][:]])
        total = np.maximum(f["obs"]["total_counts"][:], 1)
        ip = f["X"]["indptr"][:]
        X = np.zeros((len(rows), len(var)), "float32")
        for k, r in enumerate(rows):
            a, b = int(ip[r]), int(ip[r + 1])
            X[k, f["X"]["indices"][a:b]] = f["X"]["data"][a:b] / total[r]
    return np.log1p(X * 1e4), var
