"""a and b: the SNP demultiplexing umap with each cell's final call, and the depth of
every cell the analysis keeps.

the kept cells are the support comparison's crispants (`obs_crispant.parquet`): the SNP
singlets that are not transcriptional doublets and carry at least 50 panel genes. UMIs
and genes are over the whole transcriptome, as preprocess_counts.py computed them before
the panel cut. writes numerical_results/{snp_umap,qc_cells}.parquet.

    python -m analysis.figures.supp_tbx5.compute_qc
"""
import os

import anndata as ad
import h5py
import pandas as pd

from analysis.figures.supp_tbx5 import style as S
from analysis.pipelines.tbx5.support_comparison import common as C


def main():
    with h5py.File(C.H5AD) as f:
        o = ad.io.read_elem(f["obs"])
    kept = C.side("crispant")
    assert kept.index.isin(o.index).all()
    q = o.loc[kept.index, ["total_counts", "n_genes_by_counts"]]
    q.columns = ["umis", "genes"]
    q.insert(0, "embryo", kept.embryo.values)
    q.to_parquet(os.path.join(S.NR, "qc_cells.parquet"))

    u = pd.read_parquet(os.path.join(S.DEMUX, "snp_umap.parquet"))
    u = u.rename(columns={"u1": "x", "u2": "y"})
    u["embryo"] = kept.embryo.reindex(u.index).fillna("discarded").values
    u.to_parquet(os.path.join(S.NR, "snp_umap.parquet"))
    print(f"{len(u):,} cells with a SNP profile, {len(q):,} kept", flush=True)


if __name__ == "__main__":
    main()
