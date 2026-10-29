"""Compute a UMAP of the scVI-integrated embryo atlas on GPU with rapids-singlecell.

Reads the `X_scvi` latent embedding from the integrated AnnData, builds an
approximate kNN graph and UMAP on the GPU, and writes the 2D coordinates to
final_umap_integrated.csv with the same structure as final_umap.csv:
an unnamed index column of cell barcodes plus UMAP1, UMAP2.

Run via embed/compute_umap.sbatch (rapids_singlecell conda env).
"""

# Use RMM managed (unified) memory so the ~3.9M-cell graph can spill to host
# RAM if it exceeds GPU memory. Must be set up before any cupy allocations.
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
import cupy as cp

rmm.reinitialize(managed_memory=True, pool_allocator=False)
cp.cuda.set_allocator(rmm_cupy_allocator)

import h5py
import numpy as np
import pandas as pd
import anndata as ad
import rapids_singlecell as rsc

IN_H5AD = "../data/embryos/qiu24_integrated_hvgs_and_scvi.h5ad"
OUT_CSV = "../data/embryos/final_umap_integrated.csv"
USE_REP = "X_scvi"

# Pull only the latent embedding + cell barcodes; the full count matrix (~55GB)
# is not needed for neighbors/UMAP.
print(f"Loading {USE_REP} and obs_names from {IN_H5AD}", flush=True)
with h5py.File(IN_H5AD, "r") as f:
    X_rep = f["obsm"][USE_REP][:]
    index_key = f["obs"].attrs.get("_index", "_index")
    raw_names = f["obs"][index_key][:]
obs_names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in raw_names]
print(f"{USE_REP} shape: {X_rep.shape}, n_obs: {len(obs_names)}", flush=True)

# Lightweight AnnData carrying only the embedding (dummy X, never used).
adata = ad.AnnData(
    X=np.zeros((X_rep.shape[0], 1), dtype=np.float32),
    obs=pd.DataFrame(index=pd.Index(obs_names, name=None)),
)
adata.obsm[USE_REP] = X_rep

# Approximate kNN (nn_descent scales to millions of cells) + UMAP, both on GPU.
print("Computing neighbors (nn_descent) on GPU", flush=True)
rsc.pp.neighbors(adata, use_rep=USE_REP, n_neighbors=50, algorithm="nn_descent")
print("Computing UMAP on GPU", flush=True)
rsc.tl.umap(adata, min_dist=0.4)

umap = adata.obsm["X_umap"]
if hasattr(umap, "get"):  # bring back from device if still a cupy array
    umap = umap.get()
df = pd.DataFrame(umap, index=adata.obs_names, columns=["UMAP1", "UMAP2"])
df.to_csv(OUT_CSV)
print(f"Wrote {df.shape[0]} UMAP coordinates to {OUT_CSV}", flush=True)
