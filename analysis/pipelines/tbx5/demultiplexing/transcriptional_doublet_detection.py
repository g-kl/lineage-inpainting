#!/usr/bin/env python
"""second doublet pass: putative singlets whose pca neighbours are mostly SNP doublets.

the SNP cut only sees a doublet whose two cells came from different embryos, so what it
leaves is scanned in X_pca for cells sitting among the ones it flagged. `p` is the pool's
own flagged fraction, which is what a doublet neighbour fraction is worth at random, so the
cutoff is p + M * (1 - p).

writes both call columns into the h5ad obs, which is the last step of the demultiplexing.

    python -m analysis.pipelines.tbx5.demultiplexing.transcriptional_doublet_detection
"""
import numpy as np
import pandas as pd

from analysis.pipelines.tbx5.demultiplexing import common as C


def flag(o, c, z):
    """per cell with a SNP profile: doublet neighbour fraction and whether it is cut."""
    zc = z[o.index.get_indexer(c.index)]
    singlet = C.singlet(c).values
    nb = C.knn(zc, zc, C.K_NN + 1)[:, 1:]
    p = 1 - singlet.mean()
    d = pd.DataFrame({"snp_singlet": singlet, "dbl_frac": (~singlet[nb]).mean(1),
                      "thresh": p + C.M * (1 - p)}, index=c.index)
    d["nn_doublet"] = d.snp_singlet & (d.dbl_frac > d.thresh)
    d["keep"] = d.snp_singlet & ~d.nn_doublet
    d[C.EM_NN_CALL] = np.where(d.keep, c.embryo, C.THROWAWAY)
    return d


def main():
    o, c = C.obs(), C.calls()
    d = flag(o, c, C.pca())
    d.to_parquet(C.out("nn_doublets.parquet"))
    t = pd.DataFrame([{
        "cells": len(d), "snp_singlets": int(d.snp_singlet.sum()),
        "doublet_rate": 1 - d.snp_singlet.mean(), "thresh": d.thresh.iloc[0],
        "nn_doublets": int(d.nn_doublet.sum()), "kept": int(d.keep.sum()),
        "kept_frac_of_snp": d.keep.sum() / d.snp_singlet.sum()}])
    t.to_csv(C.out("nn_doublets.csv"), index=False)
    print(t.round(4).to_string(index=False), flush=True)
    C.write_obs({C.EM_CALL: c[C.EM_CALL], C.EM_NN_CALL: d[C.EM_NN_CALL]})


if __name__ == "__main__":
    main()
