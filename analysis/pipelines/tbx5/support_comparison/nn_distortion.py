"""nearest neighbour distortion of a perturbed cell set against a reference of embryos.

library only. per_embryo.py imports nn_dist from here and nothing here writes output.
"""
import numpy as np
import pandas as pd

from analysis.pipelines.tbx5.support_comparison import common as C


def nn_dist(base, query, chunk=250_000):
    """distance from each query row to its nearest base row, exact self matches dropped.

    a predicted perturbed set is made of reference cells, so a query cell can be its own
    neighbour at distance 0. base rows are distinct, so at most one ties at 0 and k=2 with
    the first strictly positive distance drops exactly it.
    """
    from cuml.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="euclidean")
    nn.fit(np.ascontiguousarray(base, dtype="float32"))
    out = np.empty(len(query), dtype="float64")
    for s in range(0, len(query), chunk):
        d = nn.kneighbors(np.ascontiguousarray(query[s:s + chunk], dtype="float32"))[0]
        d = np.asarray(d.get() if hasattr(d, "get") else d)
        out[s:s + chunk] = np.where(d[:, 0] > 0, d[:, 0], d[:, 1])
    return out


def draw_size(embryo, *perturbed, cap=C.N_MAX):
    """the n every draw is taken at: the smallest reference embryo, the thinnest perturbed
    set, and the cap. one n per call, so every source sits at the same support size and a
    thinner one does not read as distortion for being thinner."""
    return int(min([cap, pd.Series(embryo).value_counts().min()] + list(perturbed)))


def distortion(zr, embryo, zp, n, p_embryo=None, seed=C.SEED):
    """per row of zr: nn distance to the perturbed draw over nn distance to a draw from
    another reference embryo.

    `p_embryo` is the reference embryo each perturbed row belongs to, and is for a
    perturbed set that is itself made of reference cells -- the model's predictions are.
    it puts the numerator on a third embryo, so no cell can win both searches; without it
    the two draws overlap and the ratio is pinned at exactly 1. leave it None for a
    perturbed set of foreign cells, where the numerator draw is the whole set.

    both draws are at n and the denominator depends only on (zr, embryo, n, seed), so two
    perturbed sets scored at the same n share it and their distortions compare directly.
    `zp` rows must be distinct.
    """
    embs = np.array(sorted(pd.unique(embryo)))
    need = 3 if p_embryo is not None else 2
    assert len(embs) >= need, f"{len(embs)} reference embryos, need {need}"
    assert p_embryo is None or set(embs) <= set(pd.unique(p_embryo)), \
        "the perturbed set has no cells in some reference embryo"
    rden, rnum = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    pick = lambda rng, a: a[rng.choice(len(a), n, replace=False)]

    num, den = np.empty(len(zr)), np.empty(len(zr))
    if p_embryo is None:
        num[:] = nn_dist(zp[pick(rnum, np.arange(len(zp)))], zr)
    for e in embs:
        at = np.flatnonzero(embryo == e)
        rest = embs[embs != e]
        den_e = rden.choice(rest)
        den[at] = nn_dist(zr[pick(rden, np.flatnonzero(embryo == den_e))], zr[at])
        if p_embryo is not None:
            num_e = rnum.choice(rest[rest != den_e])
            num[at] = nn_dist(zp[pick(rnum, np.flatnonzero(p_embryo == num_e))], zr[at])
    return num / den
