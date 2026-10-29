#!/usr/bin/env python
"""fit scvi on the Tbx5 singlets together with the E9.5 devmap reference.

the batch key is `crispant` against `devmap` and not the embryo: correcting the embryos
apart from each other would take the phenotype with it. the reference is deeper on this
panel, so it is thinned quantile to quantile down to the crispant depth first.

    python -m analysis.pipelines.tbx5.support_comparison.fit_scvi
"""
import os
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
from lightning.pytorch.loggers import CSVLogger

from analysis.pipelines.tbx5.support_comparison import common as C


def thin(X, to):
    """thin each row of X to the matching quantile of `to`, never upwards."""
    tot = np.asarray(X.sum(1)).ravel().astype(np.int64)
    src = np.sort(to)
    order = np.argsort(tot, kind="stable")
    q = (np.arange(len(order)) + 0.5) / len(order)
    target = np.empty(len(tot), np.int64)
    target[order] = src[np.clip((q * len(src)).astype(int), 0, len(src) - 1)]
    a = ad.AnnData(X)
    sc.pp.downsample_counts(a, counts_per_cell=np.minimum(target, tot),
                            random_state=C.SEED)
    return a.X.tocsr()


def build():
    """crispants and devmap stacked on the shared panel, with a dataset column."""
    g, _ = C.genes()
    cri_obs, cri_x = C.crispants()
    dev_obs, dev_x = C.devmap()
    panel = lambda X: np.asarray(X.sum(1)).ravel().astype(np.int64)
    cri_umi = panel(cri_x)
    print(f"crispants {cri_x.shape}, median panel umi {int(np.median(cri_umi)):,}",
          flush=True)
    print(f"devmap {dev_x.shape}, median panel umi {int(np.median(panel(dev_x))):,}",
          flush=True)
    if C.DOWNSAMPLE_REF:
        dev_x = thin(dev_x, cri_umi)
        print(f"devmap thinned, median panel umi {int(np.median(panel(dev_x))):,}",
              flush=True)

    parts = []
    for name, o, X in (("crispant", cri_obs, cri_x), ("devmap", dev_obs, dev_x)):
        keep = np.asarray((X > 0).sum(1)).ravel() >= C.MIN_PANEL_GENES
        print(f"{name} panel genes >= {C.MIN_PANEL_GENES}: {len(o):,} to "
              f"{int(keep.sum()):,}", flush=True)
        o = o[keep].copy()
        o["panel_umi"] = panel(X[keep])
        parts.append((o, X[keep]))

    obs = pd.concat([p[0] for p in parts], keys=["crispant", "devmap"], names=["dataset"])
    a = ad.AnnData(sp.vstack([p[1] for p in parts]).tocsr(),
                   obs=obs.reset_index("dataset"), var=pd.DataFrame(index=g))
    assert (a.X.data[:1000] == a.X.data[:1000].astype(int)).all(), "X is not raw counts"
    return a


def elbo(log_dir):
    """the validation curve, one row per validation step."""
    m = pd.read_csv(os.path.join(log_dir, "metrics.csv"))
    m["kl_weight"] = m["kl_weight"].ffill()
    cols = ["step", "epoch", "kl_weight", "elbo_validation",
            "reconstruction_loss_validation", "kl_local_validation"]
    return m.loc[m.elbo_validation.notna(), cols].reset_index(drop=True)


def main():
    os.makedirs(C.OUT, exist_ok=True)
    scvi.settings.seed = C.SEED
    a = build()
    per_epoch = int(len(a) * C.TRAIN_FRAC) // C.BATCH
    print(f"combined {a.shape}; {C.STEPS} steps of {C.BATCH} is "
          f"{C.STEPS / per_epoch:.2f} passes ({per_epoch} steps per pass)", flush=True)

    scvi.model.SCVI.setup_anndata(a, batch_key="dataset")
    model = scvi.model.SCVI(a, n_layers=C.N_LAYERS, n_latent=C.N_LATENT,
                            n_hidden=C.N_HIDDEN, gene_likelihood="nb",
                            dispersion="gene-batch")
    log_dir = C.out("train_log")
    shutil.rmtree(log_dir, ignore_errors=True)
    model.train(
        max_steps=C.STEPS, batch_size=C.BATCH, load_sparse_tensor=True,
        early_stopping=False, train_size=C.TRAIN_FRAC, validation_size=C.VAL_FRAC,
        plan_kwargs={"n_epochs_kl_warmup": None, "n_steps_kl_warmup": C.KL_WARMUP},
        logger=CSVLogger(log_dir, name="", version=""),
        check_val_every_n_epoch=1, val_check_interval=C.VAL_EVERY,
        log_every_n_steps=C.VAL_EVERY, enable_progress_bar=False,
    )
    model.save(C.out("model"), overwrite=True)

    z = model.get_latent_representation(batch_size=4096).astype("float32")
    is_cri = (a.obs.dataset == "crispant").values
    for which, m in (("crispant", is_cri), ("devmap", ~is_cri)):
        np.save(C.out(f"latent_{which}.npy"), z[m])
        a.obs[m].drop(columns="dataset").to_parquet(C.out(f"obs_{which}.parquet"))
    e = elbo(log_dir)
    e.to_csv(C.out("elbo.csv"), index=False)
    print(e.tail(5).to_string(index=False), flush=True)
    print(f"wrote latents ({is_cri.sum():,} crispant, {(~is_cri).sum():,} devmap)",
          flush=True)


if __name__ == "__main__":
    main()
