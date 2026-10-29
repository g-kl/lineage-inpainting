#!/usr/bin/env python
"""fit the mixture of model.md and call an embryo (or throwaway) for every cell.

also embeds the same features with umap and writes a per-component table, so "is any
component sitting at the pool mean instead of on an embryo" can be read off numbers.

    python -m analysis.pipelines.tbx5.demultiplexing.snp_demux
"""
import itertools
import os

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.special import gammaln

from analysis.pipelines.tbx5.demultiplexing import common as C

MAX_ITER, TOL = 300, 1e-9
EPS = 1e-3          # genotypes are clipped here, so no cell gets a -inf likelihood
CHUNK = 50_000
PCT = 0.5           # percentile window the umap panel is drawn over
PAIRS = np.array(list(itertools.combinations(range(C.K), 2)))


def loglik(a, b, P, s):
    """(cells, components) log-likelihood, up to the per-cell term `offset` puts back.

    binomial when `s` is None, beta-binomial with concentration `s` per bin otherwise.
    """
    if s is None:
        return a @ cp.log(P).T + b @ cp.log1p(-P).T
    al, be = P * s, (1 - P) * s
    base = (gammaln(al) + gammaln(be)).sum(1)
    out = cp.empty((len(a), len(P)))
    for i in range(0, len(a), CHUNK):
        acc = cp.zeros((len(a[i:i + CHUNK]), len(P)))
        for c in range(P.shape[1]):
            acc += gammaln(a[i:i + CHUNK, c:c + 1] + al[:, c][None, :])
            acc += gammaln(b[i:i + CHUNK, c:c + 1] + be[:, c][None, :])
        out[i:i + CHUNK] = acc - base[None, :]
    return out


def offset(n, s):
    """the per-cell term `loglik` leaves out. same for every component, so the E step does
    not need it, but it depends on the concentration. what stays dropped is log C(n, a)."""
    if s is None:
        return 0.0
    return float((gammaln(s)[None, :] - gammaln(n + s[None, :])).sum())


def genotypes(theta, pairs):
    """(components, bins): the K embryos, then one per pair at the pair's mean."""
    return cp.vstack([theta, theta[pairs].mean(1)])


def hard(f):
    """a fit's embryo assignment, to warm start the next fit from."""
    return np.argmax(cp.asnumpy(f["r"][:, :C.K]), axis=1)


def coarse(x, chrom):
    """bins summed back to one column per chromosome."""
    return np.stack([x[:, chrom == c].sum(1) for c in np.unique(chrom)], axis=1)


def init(X):
    """k-means on the z-scored G1 fractions, to start the EM from."""
    from cuml.cluster import KMeans
    lab = KMeans(n_clusters=C.K, random_state=C.SEED).fit_predict(X)
    return np.asarray(lab.get() if hasattr(lab, "get") else lab)


def em(a, b, lab, s=None):
    """EM for K embryos plus one component per pair of embryos."""
    n = a + b
    pairs = cp.asarray(PAIRS)
    onehot = cp.asarray(np.eye(C.K)[lab])
    theta = (onehot.T @ a + 1) / (onehot.T @ n + 2)
    pi = cp.full(C.K + len(PAIRS), 1 / (C.K + len(PAIRS)))
    const = offset(n, s)
    prev = -np.inf
    for it in range(MAX_ITER):
        P = cp.clip(genotypes(theta, pairs), EPS, 1 - EPS)
        ll = loglik(a, b, P, s) + cp.log(pi)
        top = ll.max(1, keepdims=True)
        w = cp.exp(ll - top)
        tot = w.sum(1, keepdims=True)
        r = w / tot
        total = float((top[:, 0] + cp.log(tot[:, 0])).sum()) + const
        rs = r[:, :C.K]
        theta = (rs.T @ a + 1) / (rs.T @ n + 2)
        pi = r.mean(0)
        if total - prev < TOL * abs(prev):
            break
        prev = total
    npar = C.K * a.shape[1] + len(pi) - 1 + (0 if s is None else a.shape[1])
    return {"theta": theta, "pi": cp.asnumpy(pi), "r": r, "loglik": total,
            "bic": -2 * total + npar * np.log(len(a)), "iters": it + 1}


def concentration(a, b, f):
    """per-bin concentration by moments on the cells the binomial pass is sure are
    singlets, and the dispersion those cells show."""
    rs = f["r"][:, :C.K]
    conf = rs.max(1) > 0.99
    th = f["theta"][cp.argmax(rs, axis=1)][conf]
    A, N = a[conf], (a + b)[conf]
    ok = (th > EPS) & (th < 1 - EPS) & (N > 0)
    v = cp.where(ok, N * th * (1 - th), cp.nan)
    d = cp.asnumpy(cp.nanmean(cp.where(ok, (A - N * th) ** 2, cp.nan) / v, axis=0))
    nmed = cp.asnumpy(cp.nanmedian(cp.where(ok, N, cp.nan), axis=0))
    d = np.where(np.isfinite(d), d, 1.0)
    s = np.where(d > 1.01, (nmed - d) / np.maximum(d - 1, 1e-9), 1e6)
    return cp.asarray(np.clip(np.nan_to_num(s, nan=1e6), 5.0, 1e6)), d, int(conf.sum())


def call(f):
    """embryo, partner and the scores, from a fit's responsibilities."""
    r = cp.asnumpy(f["r"])
    names = np.array(C.embryo_names())
    emb = r[:, :C.K].argmax(1)
    pair = PAIRS[r[:, C.K:].argmax(1)]
    partner = np.where(pair[:, 0] == emb, pair[:, 1], pair[:, 0])
    return {"embryo": names[emb], "partner": names[partner],
            "singlet_post": r[:, :C.K].max(1), "score": r[:, C.K:].sum(1)}


def cull(c, o):
    """cells left after each step and at each threshold on the score."""
    rows = [("all cells", len(o)), ("pass qc", int(o.qc.sum())),
            ("has SNP profile", len(c))]
    rows += [(f"score <= {t:.3f}", int(C.singlet(c, t).sum())) for t in C.POSTERIORS]
    t = pd.DataFrame(rows, columns=["step", "cells"])
    t["frac_of_all"] = (t.cells / len(o)).round(4)
    return t


def embed(X, ix):
    """umap of the features the fit is on, cached since it does not depend on the calls."""
    path = C.out("snp_umap.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    from cuml.manifold import UMAP
    u = UMAP(n_neighbors=C.K_SNP, min_dist=0.3, random_state=C.SEED).fit_transform(X)
    u = pd.DataFrame(np.asarray(u), columns=["u1", "u2"], index=ix)
    u.to_parquet(path)
    return u


def components(u, c, theta, pi):
    """per embryo component: where it sits, and how much of it the cut keeps.

    `geno_radius` is the component genotype's distance from the pi-weighted pool mean
    genotype over the furthest component's, so a component near 0 is fit to the pool mean
    rather than to an embryo. `umap_radius` is the same statement on the layout, over the
    cells the cut keeps -- a doublet assigned to a component sits between it and its
    partner, so including them drags every component inwards.
    """
    w = pi[:C.K] / pi[:C.K].sum()
    th = cp.asnumpy(theta)
    gr = np.linalg.norm(th - (w[:, None] * th).sum(0, keepdims=True), axis=1)
    d = u.join(c[["embryo", "score", "snp_umi"]])
    d["kept"] = C.singlet(c)
    mid = d[["u1", "u2"]].median().values
    ur = np.linalg.norm(d[d.kept].groupby("embryo")[["u1", "u2"]].median() - mid, axis=1)
    by = d.groupby("embryo")
    t = pd.DataFrame({"cells": by.size(), "kept_frac": by.kept.mean(),
                      "med_score": by.score.median(),
                      "med_snp_umi": by.snp_umi.median(),
                      "umap_radius": ur / ur.max()})
    t["geno_radius"] = pd.Series(gr / gr.max(), index=C.embryo_names())
    t["pi"] = pd.Series(pi[:C.K], index=C.embryo_names())
    return t.sort_values("geno_radius")


def plot(u, c):
    """the umap by called embryo with the thrown-away cells in grey, and the scores."""
    import matplotlib.pyplot as plt
    d = u.join(c[["embryo", "score"]])
    d["kept"] = C.singlet(c)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    lo, hi = d[["u1", "u2"]].quantile([PCT / 100, 1 - PCT / 100]).values
    off = ((d.u1 < lo[0]) | (d.u1 > hi[0]) | (d.u2 < lo[1]) | (d.u2 > hi[1])).sum()
    g = d[~d.kept]
    ax[0].scatter(g.u1, g.u2, s=1, c="#d9d9d9", lw=0, rasterized=True, label="throwaway")
    cmap = plt.get_cmap("tab10")
    for i, e in enumerate(C.embryo_names()):
        g = d[d.kept & (d.embryo == e)]
        ax[0].scatter(g.u1, g.u2, s=1, color=cmap(i % 10), lw=0, rasterized=True, label=e)
    ax[0].set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xticks=[], yticks=[],
              xlabel=f"snp umap ({off} cells off panel)")
    ax[0].legend(markerscale=8, frameon=False, fontsize=7, loc="best")
    ax[1].hist(d.score, bins=100)
    ax[1].axvline(C.DOUBLET_SCORE, color="k", lw=1)
    ax[1].set(yscale="log", xlabel="score", ylabel="cells")
    for a in ax:
        for s in a.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(C.out("snp_umap.png"), dpi=200)


def main():
    os.makedirs(C.OUT, exist_ok=True)
    o = C.obs()
    ca, cb, ix, chrom = C.counts()
    keep = ix.isin(o.index[o.qc])
    ca, cb, ix = ca[keep], cb[keep], ix[keep]
    print(f"{len(ix):,} cells with a SNP profile over {ca.shape[1]} bins", flush=True)
    a = cp.asarray(ca, dtype=cp.float64)
    b = cp.asarray(cb, dtype=cp.float64)

    kca, kcb = coarse(ca, chrom), coarse(cb, chrom)
    crs = em(cp.asarray(kca, dtype=cp.float64), cp.asarray(kcb, dtype=cp.float64),
             init(C.features(kca, kcb)))
    binom = em(a, b, hard(crs))
    s, disp, n_conf = concentration(a, b, binom)
    print(f"dispersion {disp.mean():.2f} on {n_conf:,} confident singlets, median "
          f"concentration {float(cp.median(s)):.0f}", flush=True)
    fit = em(a, b, hard(binom), s)

    log = []
    for model, f in (("coarse", crs), ("binomial", binom), ("betabinomial", fit)):
        log.append({"model": model, "bins": f["theta"].shape[1], "cells": len(ix),
                    "loglik": f["loglik"], "bic": f["bic"], "iters": f["iters"],
                    "dispersion": disp.mean(), "median_s": float(cp.median(s)),
                    "confident_singlets": n_conf, "pi_doublet": f["pi"][C.K:].sum(),
                    "min_pi": f["pi"][:C.K].min(),
                    "embryos_over_1pct": int((f["pi"][:C.K] > 0.01).sum())})
    pd.DataFrame(log).to_csv(C.out("fit.csv"), index=False)
    print(pd.DataFrame(log).round(4).to_string(index=False), flush=True)

    c = pd.DataFrame({"snp_umi": (ca + cb).sum(1), **call(fit)}, index=ix)
    c[C.EM_CALL] = np.where(C.singlet(c), c.embryo, C.THROWAWAY)
    c.to_parquet(C.out("demux.parquet"))
    t = cull(c, o)
    t.to_csv(C.out("cull.csv"), index=False)
    print(t.to_string(index=False), flush=True)

    u = embed(C.features(ca, cb), ix)
    comp = components(u, c, fit["theta"], fit["pi"])
    comp.to_csv(C.out("components.csv"))
    print(comp.round(4).to_string(), flush=True)
    central = comp.index[(comp.geno_radius < C.CENTRAL)
                         | (comp.umap_radius < C.CENTRAL)].tolist()
    print(f"components under radius {C.CENTRAL}: {central or 'none'}", flush=True)
    plot(u, c)


if __name__ == "__main__":
    main()
