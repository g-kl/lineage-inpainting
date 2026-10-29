#!/usr/bin/env python
"""cellranger filtered h5 -> ../data/2608_crispants/tbx5.h5ad.

both Tbx5 lanes, light qc on the full transcriptome, then subset to the reference gene
panel. X is raw counts; X_pca is 50 components of the log-normalised panel, which is the
space transcriptional_doublet_detection.py works in.

    python -m analysis.pipelines.tbx5.demultiplexing.preprocess_counts
"""
import anndata as ad
import numpy as np
import h5py
import scanpy as sc

from analysis.pipelines.tbx5.demultiplexing import common as C


def lane(name, panel):
    """one lane, qc'd on the full transcriptome and cut to the panel."""
    a = sc.read_10x_h5(f"{C.DATA}/{name}/{name}_filtered_counts.h5")
    a.var_names_make_unique()
    a.var["mt"] = a.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False,
                               inplace=True)
    keep = ((a.obs.n_genes_by_counts >= C.MIN_GENES_LOAD)
            & (a.obs.pct_counts_mt < C.MAX_PCT_MT_LOAD))
    print(f"{name}: {a.n_obs} -> {int(keep.sum())}", flush=True)
    a = a[keep.values, panel].copy()
    a.var = a.var[[]]
    return a


def embed(X):
    """50 pcs of the log-normalised panel."""
    import rapids_singlecell as rsc
    p = ad.AnnData(X.copy())
    rsc.get.anndata_to_GPU(p)
    rsc.pp.normalize_total(p, target_sum=1e4)
    rsc.pp.log1p(p)
    rsc.pp.pca(p, n_comps=C.N_PCS, random_state=C.SEED)
    rsc.get.anndata_to_CPU(p)
    return np.ascontiguousarray(p.obsm["X_pca"], dtype="float32")


def main():
    with h5py.File(C.REF) as f:
        panel = [x.decode() for x in f["var"]["_index"][:]]
    a = ad.concat([lane(n, panel) for n in C.LANES], label="sample", keys=C.LANES,
                  index_unique="-")
    a.obs["sample"] = a.obs["sample"].astype("category")
    a.obsm["X_pca"] = embed(a.X)
    a.write_h5ad(C.H5AD)
    print(f"wrote {a.shape} to {C.H5AD}", flush=True)
    print(a.obs.groupby("sample", observed=True).size().to_string(), flush=True)
    print(f"qc pass: {int(C.obs().qc.sum()):,} of {a.n_obs:,}", flush=True)


if __name__ == "__main__":
    main()
