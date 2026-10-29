"""build ../data/2610_lineage_inpainting_data_avail: the main-text checkpoint and its coupling, the
inputs it runs on (trees + devmap scvi latents), the scvi model, and the demultiplexed Tbx5 singlets.

    python scripts/build_data_avail.py
"""
import hashlib
import os
import shutil

import anndata as ad
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "..", "data")
OUT = os.path.join(DATA, "2610_lineage_inpainting_data_avail")
RUN = os.path.join(ROOT, "outputs", "mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_"
                   "nosa_std_none_57ab441ec3fcecf81d545755db5ddb09")
CKPT = ["config.yaml", "pool_ema_step20000.pt", "split_ema_step20000.pt"]
# the main-text coupling (summary_qc.style.MODEL_ASSIGNMENTS), renamed without the run name
ASSIGN = (f"assignments_nn_{os.path.basename(RUN)}_step20000_adj_n8_splitter.pkl",
          "assignments_step20000_adj_n8_splitter.pkl")
EMB = os.path.join(DATA, "embryos")
STATES = os.path.join(EMB, "qiu24_integrated_hvgs_and_scvi.h5ad")
LATENTS = "devmap_latents.h5ad"
EMB_COPY = ["final_topology.h5td", "final_subtype_v2_lower.csv",
            "final_type_info_v2_lower.csv", "final_umap.csv"]
SCVI = os.path.join(EMB, "scvi_model_hvgs_and_scvi", "model.pt")
TBX5 = os.path.join(DATA, "2608_crispants", "tbx5.h5ad")
CALL = "EM_NN_embryo_call"
KEEP = ["sample", "n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt",
        CALL]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def latents():
    """devmap rows of the integrated atlas: X_scvi, obs and the gene panel, no counts."""
    with h5py.File(STATES) as h:
        obs = ad.io.read_elem(h["obs"])
        var = ad.io.read_elem(h["var"])
        rows = np.flatnonzero((obs["dataset"] == "devmap").values)
        z = h["obsm/X_scvi"][:][rows]
    obs = obs.iloc[rows]
    obs = obs.loc[:, ~obs.columns.str.startswith("_scvi") & obs.notna().any().values].copy()
    for c in obs.columns[obs.dtypes == "category"]:
        obs[c] = obs[c].cat.remove_unused_categories()
    a = ad.AnnData(obs=obs, var=var, obsm={"X_scvi": z})
    a.write_h5ad(os.path.join(OUT, "embryos", LATENTS))
    print(f"latents: {a.shape}, obs {list(obs.columns)}", flush=True)


def tbx5():
    a = ad.read_h5ad(TBX5)
    a = a[(a.obs[CALL] != "throwaway").values].copy()
    assert (a.obs["EM_embryo_call"].astype(str) == a.obs[CALL].astype(str)).all()
    a.obs = a.obs[KEEP]
    a.obs[CALL] = a.obs[CALL].cat.remove_unused_categories()
    del a.obsm["X_pca"]
    a.write_h5ad(os.path.join(OUT, "tbx5_crispants.h5ad"))
    print(f"tbx5: {a.shape}", flush=True)
    print(a.obs[CALL].value_counts().sort_index().to_string(), flush=True)


def main():
    for f in CKPT:
        copy(os.path.join(RUN, f), os.path.join(OUT, "model", f))
    copy(os.path.join(RUN, ASSIGN[0]), os.path.join(OUT, "model", ASSIGN[1]))
    for f in EMB_COPY:
        copy(os.path.join(EMB, f), os.path.join(OUT, "embryos", f))
    copy(SCVI, os.path.join(OUT, "scvi_model", "model.pt"))
    latents()
    tbx5()

    files = sorted(os.path.relpath(os.path.join(d, f), OUT)
                   for d, _, fs in os.walk(OUT) for f in fs)
    with open(os.path.join(OUT, "md5sums.txt"), "w") as fh:
        for f in files:
            if f not in ("README.md", "md5sums.txt"):
                fh.write(f"{md5(os.path.join(OUT, f))}  {f}\n")


if __name__ == "__main__":
    main()
