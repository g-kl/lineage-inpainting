"""paths, cutoffs and loaders shared by the demultiplexing scripts."""
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "..", "data", "2608_crispants")
H5AD = os.path.join(DATA, "tbx5.h5ad")
SNP_DIR = os.path.join(DATA, "demultiplexing", "demux_fullsnp")
# the gene panel the h5ad is built on, and the panel the reference latent uses.
REF = os.path.join(ROOT, "..", "data", "embryos", "qiu24_integrated_hvgs_and_scvi.h5ad")
OUT = os.path.join(ROOT, "analysis", "results", "tbx5", "demultiplexing")

TARGET = "Tbx5"
K = 8                                # embryos in the pool
LANES = ("Tbx5_T1", "Tbx5_T2")       # h5ad `sample`, and the barcode suffix after "-1-"
GEMS = {"GEM1": "Tbx5_T1", "GEM2": "Tbx5_T2"}   # snp file lane -> h5ad sample
CHROMS = [f"chr{i}" for i in range(1, 20)]      # autosomes only

# allSNP positions are a cumulative coordinate over the whole reference, so they carry the
# chromosome implicitly. FINE is the resolution the read histogram is built at; N_BINS
# equal read mass bins per chromosome are cut from it. both are baked into counts.npz.
FINE = 1_000
N_BINS = 10
N_FEAT = len(CHROMS) * N_BINS

# cutoffs preprocess_counts.py loads the cellranger output at.
MIN_GENES_LOAD, MAX_PCT_MT_LOAD = 200, 15.0
# the qc flag the fit is restricted to.
MIN_UMI, MIN_GENES, MAX_PCT_MT = 750, 400, 10.0
MIN_SNP_UMI = 100                    # autosomal SNP reads a cell needs for a profile

N_PCS = 50
DOUBLET_SCORE = 0.05                 # cut on the posterior of not being a singlet
M = 0.5                              # slack in the nn doublet cutoff, p + M * (1 - p)
K_NN = 30                            # neighbours in the pca doublet pass
K_SNP = 15                           # neighbours in the snp umap
CENTRAL = 0.25                       # a component under this radius sits at the pool mean
SEED = 0

THROWAWAY = "throwaway"
EM_CALL = "EM_embryo_call"
EM_NN_CALL = "EM_NN_embryo_call"

POSTERIORS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf]


def out(name):
    return os.path.join(OUT, name)


def embryo_names():
    return [f"{TARGET}_e{i}" for i in range(K)]


def obs():
    """the h5ad per-cell table, with the qc flag but no rows dropped."""
    with h5py.File(H5AD) as f:
        df = ad.io.read_elem(f["obs"])
    df["qc"] = ((df.total_counts >= MIN_UMI) & (df.n_genes_by_counts >= MIN_GENES)
                & (df.pct_counts_mt <= MAX_PCT_MT))
    return df


def pca():
    """the h5ad X_pca, in obs() row order."""
    with h5py.File(H5AD) as f:
        return f["obsm"]["X_pca"][:]


def counts():
    """(G1, G2) binned SNP read counts per cell, both lanes pooled, in obs() order.

    both are int32 (cells, bins) with the empty bins dropped; cells under MIN_SNP_UMI go
    too. `chrom` is each surviving bin's chromosome, for summing back to chromosome
    resolution. built by aggregate.py on first use.
    """
    path = out("counts.npz")
    if not os.path.exists(path):
        from analysis.pipelines.tbx5.demultiplexing import aggregate
        aggregate.main()
    z = np.load(path)
    assert (z["fine"], z["n_bins"]) == (FINE, N_BINS), f"{path} is stale, delete it"
    ix = obs().index
    j = pd.Index(z["index"]).get_indexer(ix)
    assert (j >= 0).all(), f"{path} misses {(j < 0).sum()} h5ad cells, delete it"
    a, b = z["a"][j], z["b"][j]
    use = (a.sum(0) + b.sum(0)) > 0        # equal mass bins can leave a column empty
    keep = (a + b).sum(1) >= MIN_SNP_UMI
    return (a[np.ix_(keep, use)], b[np.ix_(keep, use)], ix[keep],
            np.arange(N_FEAT)[use] // N_BINS)


def features(a, b):
    """G1 fraction per bin, missing filled with the pool mean, then z-scored."""
    n = a + b
    X = np.where(n > 0, a / np.maximum(n, 1), np.nan).astype("float64")
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    sd = X.std(0)
    return ((X - X.mean(0)) / np.where(sd > 0, sd, 1)).astype("float32")


def calls():
    """what snp_demux.py wrote."""
    return pd.read_parquet(out("demux.parquet"))


def singlet(c, t=DOUBLET_SCORE):
    """cells the fit calls a singlet at a threshold on the score."""
    return c.score <= t


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


def write_obs(cols):
    """add columns to the h5ad obs in place, leaving X and obsm alone."""
    with h5py.File(H5AD, "a") as f:
        df = ad.io.read_elem(f["obs"])
        for name, v in cols.items():
            df[name] = pd.Categorical(pd.Series(v).reindex(df.index).fillna(THROWAWAY))
        del f["obs"]
        ad.io.write_elem(f, "obs", df)
    for name in cols:
        print(f"{name}:", flush=True)
        print(df[name].value_counts().to_string(), flush=True)
