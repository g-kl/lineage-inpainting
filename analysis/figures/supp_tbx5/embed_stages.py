"""e: put devmap cells of both age stages into the support comparison's latent, for
compute_age.py's index.

that fit is the crispants against E9.5, batch key crispant against devmap. each stage
goes through its encoder as devmap cells, thinned to the crispant depth the way the fit
was, so the stages and the crispants share one space without refitting. the E9.5 cells
the fit drew are dropped, so every index cell is one the encoder never saw. writes
numerical_results/stage_{latent,obs}_{stage}.

    python -m analysis.figures.supp_tbx5.embed_stages     # gpu, rapids_singlecell env
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import scvi

from analysis.figures.supp_tbx5 import style as S
from analysis.pipelines.tbx5.support_comparison import common as C
from analysis.pipelines.tbx5.support_comparison.fit_scvi import thin


def main():
    g, _ = C.genes()
    umi = C.side("crispant").panel_umi.values
    fit = C.side("devmap").index
    sub2type, type2lin = C.type_map()
    sub = C.devmap_subtypes()
    for stage in S.AGE_STAGES:
        # the seeded draw is a prefix of the fit's, so E9.5 draws past it
        obs, X = C.devmap(S.N_STAGE + (C.N_REF if stage == C.REF_STAGE else 0), stage)
        new = ~obs.index.isin(fit)
        obs, X = obs[new], thin(X[new], umi)
        keep = np.asarray((X > 0).sum(1)).ravel() >= C.MIN_PANEL_GENES
        obs, X = obs[keep].copy(), X[keep]
        obs["panel_umi"] = np.asarray(X.sum(1)).ravel().astype(np.int64)
        obs["dataset"] = "devmap"
        a = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=g))
        model = scvi.model.SCVI.load(C.out("model"), adata=a)
        z = model.get_latent_representation(batch_size=4096).astype("float32")
        np.save(os.path.join(S.NR, f"stage_latent_{stage}.npy"), z)

        lineage = sub.reindex(obs.index).map(sub2type).map(type2lin)
        pd.DataFrame({"lineage": lineage.values}, index=obs.index).to_parquet(
            os.path.join(S.NR, f"stage_obs_{stage}.parquet"))
        print(f"{stage}: {len(obs):,} cells", flush=True)


if __name__ == "__main__":
    main()
