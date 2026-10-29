#!/usr/bin/env python
"""transfer devmap subtypes onto the crispant cells by knn vote in the integrated latent.

the crispant cells have no labels of their own, and the per lineage distortion needs both
sides on the same vocabulary.

    python -m analysis.pipelines.tbx5.support_comparison.labels
"""
import numpy as np
import pandas as pd

from analysis.pipelines.tbx5.support_comparison import common as C


def transfer(zc, zd, labels):
    """majority label among each crispant cell's K_LABEL devmap neighbours."""
    codes, levels = pd.factorize(labels)
    nb = codes[C.knn(zd, zc, C.K_LABEL)]
    counts = np.zeros((len(nb), len(levels)), dtype=np.int32)
    np.add.at(counts, (np.repeat(np.arange(len(nb)), nb.shape[1]), nb.ravel()), 1)
    return pd.Categorical.from_codes(counts.argmax(1), levels)


def main():
    cri, dev = C.side("crispant"), C.side("devmap")
    zc, zd = C.latent()
    sub = C.devmap_subtypes().reindex(dev.index).values
    ok = pd.notna(sub)
    out = pd.DataFrame({"subtype": transfer(zc, zd[ok], sub[ok])}, index=cri.index)
    out.to_parquet(C.out("subtype_crispant.parquet"))
    print(f"transferred subtypes to {len(cri):,} crispant cells from {ok.sum():,} devmap",
          flush=True)
    print(out.subtype.value_counts().head(10).to_string(), flush=True)


if __name__ == "__main__":
    main()
