"""paths, constants and loaders shared by the support comparison scripts."""
import glob
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "..", "data")
H5AD = os.path.join(DATA, "2608_crispants", "tbx5.h5ad")
DEVMAP = os.path.join(DATA, "embryos", "embryos_hvg_v5_scvi_devmap.h5ad")
SUBTYPES = os.path.join(DATA, "embryos", "final_subtype_v2_lower.csv")
TYPE_INFO = os.path.join(DATA, "embryos", "final_type_info_v2_lower.csv")
OUT = os.path.join(ROOT, "analysis", "results", "tbx5", "support_comparison")

TARGET = "Tbx5"
CALL = "EM_NN_embryo_call"   # the demultiplexing column the singlets are taken from
THROWAWAY = "throwaway"

REF_STAGE = "E9.5"
N_REF = 250_000              # devmap cells kept, against ~90k singlets
DOWNSAMPLE_REF = True        # thin the reference to the crispant depth before fitting
MIN_PANEL_GENES = 50
SEED = 0

BATCH = 2048
STEPS = 2000
KL_WARMUP = 400
VAL_EVERY = 25
N_LATENT, N_LAYERS, N_HIDDEN = 50, 2, 512
TRAIN_FRAC, VAL_FRAC = 0.99, 0.01

K_LABEL = 25                 # neighbours voting in the subtype transfer
N_MAX = 40_000               # cap on the cells per draw
MIN_DRAW = 200               # cells a lineage needs on every side to be scored
SKIP_LINEAGES = ("blood", "extraembryonic mesoderm", "extraembryonic ectoderm",
                 "extraembryonic endoderm")

# the heart: the cardiac cell types out of the lateral plate, ~3.4k reference cells over
# nine subtypes. juxtacardiac field and splanchnic mesoderm (cardiopulmonary) are the
# progenitor pools next door and are left out -- taking them in blurs what the set
# means and lifts the achievable draw by under a fifth.
HEART_TYPES = ("first heart field", "second heart field", "endocardial cell", "epicardium")

# the model's Tbx5 screen, E9.5 readout. a glob over the replicate runs; each names its
# trees apart, so the three merge into one set.
SCREEN = os.path.join(ROOT, "outputs", "mouse_embryo_grid_bridge_unshared_vpsde_idnn_"
                      "leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09",
                      "tbx5_broadsnap_screen_w1_dose0_E9.5_R*")
LEAF_TIME, LEAF_TOL, OFF_RATIO = 9.5, 0.1, 1.1
# off-manifold convention. `prune` drops an off-manifold leaf and its subtree, which
# is the screen's own convention; `freeze` stands the last on-manifold state in for
# the leaf that never formed and snaps it forward to E9.5.
MODE, MODES = "freeze", ("prune", "freeze")


def out(name):
    return os.path.join(OUT, name)


def slug(lineage):
    return lineage.replace(" ", "_")


# ------------------------------------------------------------------- counts

def genes():
    """the crispant panel, and where each gene sits in the devmap var."""
    with h5py.File(H5AD) as f:
        g = np.array([x.decode() for x in f["var"]["_index"][:]])
    with h5py.File(DEVMAP) as f:
        dv = np.array([x.decode() for x in f["var"]["_index"][:]])
    pos = pd.Index(dv).get_indexer(g)
    assert (pos >= 0).all(), "crispant panel is not a subset of the devmap panel"
    return g, pos


def _csr(g, r0, r1, remap, ncol):
    """rows [r0, r1) of an h5ad csr group, columns remapped by `remap` (-1 drops)."""
    ip = g["indptr"][r0:r1 + 1]
    a, b = int(ip[0]), int(ip[-1])
    ind = remap[g["indices"][a:b]]
    dat = g["data"][a:b].astype("float32")
    m = ind >= 0
    cs = np.concatenate([[0], np.cumsum(m)])
    return sp.csr_matrix((dat[m], ind[m], cs[ip - a]), shape=(r1 - r0, ncol))


def _take(g, rows, remap, ncol, chunk=50_000):
    """the named h5ad rows, read in fixed blocks so memory does not track the selection."""
    parts = []
    for s in range(rows[0], rows[-1] + 1, chunk):
        e = min(s + chunk, rows[-1] + 1)
        r = rows[(rows >= s) & (rows < e)]
        if len(r):
            parts.append(_csr(g, s, e, remap, ncol)[r - s])
    return sp.vstack(parts).tocsr()


def crispants():
    """the demultiplexed singlet Tbx5 cells: obs and raw counts on the panel."""
    g, _ = genes()
    ident = np.arange(len(g), dtype="int32")
    with h5py.File(H5AD) as f:
        o = ad.io.read_elem(f["obs"])
        rows = np.flatnonzero((o[CALL] != THROWAWAY).values)
        obs = o.iloc[rows][["total_counts"]].copy()
        obs["embryo"] = o[CALL].iloc[rows].astype(str).values
        X = _take(f["X"], rows, ident, len(g))
    return obs, X


def _devmap_obs():
    """cell name and embryo for every devmap cell."""
    with h5py.File(DEVMAP) as f:
        o = f["obs"]
        cells = pd.Index([x.decode() for x in o["_index"][:]])
        e = o["embryo"]
        emb = np.array([c.decode() for c in e["categories"]])[e["codes"][:]]
    return cells, emb


def devmap(n=N_REF, stage=REF_STAGE):
    """n devmap cells of one stage, drawn uniformly: obs and raw counts on the panel."""
    g, gpos = genes()
    cells, emb = _devmap_obs()
    at = np.flatnonzero(pd.Series(emb).str.split("-").str[0].values == stage)
    rows = np.sort(np.random.default_rng(SEED).choice(at, min(n, len(at)), replace=False))
    with h5py.File(DEVMAP) as f:
        obs = pd.DataFrame({"embryo": emb[rows],
                            "total_counts": f["obs"]["total_counts"][:][rows]},
                           index=cells[rows])
        remap = np.full(f["var"]["_index"].shape[0], -1, dtype="int32")
        remap[gpos] = np.arange(len(g), dtype="int32")
        X = _take(f["X"], rows, remap, len(g))
    return obs, X


# --------------------------------------------------------------- annotations

def type_map():
    """subtype to cell type, cell type to lineage."""
    ti = pd.read_csv(TYPE_INFO)
    return dict(zip(ti.cell_subtype, ti.cell_type)), dict(zip(ti.cell_type, ti.lineage))


def devmap_subtypes():
    return pd.read_csv(SUBTYPES, index_col=0)["cell_subtype"].str.lower()


def side(which):
    return pd.read_parquet(out(f"obs_{which}.parquet"))


def latent():
    """(crispant, devmap) integrated latents, in the order the obs parquets are in."""
    return (np.load(out("latent_crispant.npy")), np.load(out("latent_devmap.npy")))


def cells():
    """one row per cell, both sides: dataset, embryo, subtype, cell_type, lineage."""
    sub2type, type2lin = type_map()
    cri, dev = side("crispant"), side("devmap")
    cri["subtype"] = pd.read_parquet(out("subtype_crispant.parquet")).subtype.reindex(
        cri.index).values
    dev["subtype"] = devmap_subtypes().reindex(dev.index).values
    d = pd.concat([cri.assign(dataset=TARGET), dev.assign(dataset="devmap")])
    d["cell_type"] = d.subtype.map(sub2type)
    d["lineage"] = d.cell_type.map(type2lin)
    return d


def is_heart(d):
    """the heart rows of a frame `cells()` shaped."""
    return d.cell_type.isin(HEART_TYPES)


def lineages(d):
    """the lineages to score, blood and the extraembryonic ones left out."""
    return [l for l in sorted(d.lineage.dropna().unique()) if l not in SKIP_LINEAGES]


def knn(base, query, k, chunk=250_000):
    """k nearest neighbours of query in base, brute force on the gpu."""
    from cuml.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    nn.fit(np.ascontiguousarray(base, dtype="float32"))
    res = np.empty((len(query), k), dtype=np.int64)
    for s in range(0, len(query), chunk):
        i = nn.kneighbors(np.ascontiguousarray(query[s:s + chunk], dtype="float32"),
                          return_distance=False)
        res[s:s + chunk] = np.asarray(i.get() if hasattr(i, "get") else i)
    return res


# ------------------------------------------------------------------- screen

def screen(which):
    """{tree name: {"meta", "tree"}} for "control" or a knockout gene, over every
    replicate."""
    from analysis.pipelines.perturb.perturb_umap_lib import load_control, load_gene
    res = {}
    for d in sorted(glob.glob(SCREEN)):
        res.update(load_control(os.path.join(d, "control")) if which == "control"
                   else load_gene(os.path.join(d, "genes"), which))
    assert res, f"no trees under {SCREEN}"
    return res


def devmap_index():
    with h5py.File(DEVMAP) as f:
        return pd.Index([x.decode() for x in f["obs"]["_index"][:]])


def devmap_latent(cells):
    """devmap X_scvi for named cells, any stage. the devmap fit's own latent, which is
    what the trees snapped in, not the integrated one."""
    idx = devmap_index()
    with h5py.File(DEVMAP) as f:
        return f["obsm"]["X_scvi"][:][idx.get_indexer(pd.Index(cells))]
