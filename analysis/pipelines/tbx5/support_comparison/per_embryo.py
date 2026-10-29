"""per embryo nn distortion: the scope-internal, fixed-draw version of nn_distortion.py.

library only. per_embryo_nn_distortion.py is the caller and nothing here writes output.

how this differs from nn_distortion.py, in order of importance:

- every draw is taken inside the scope -- the query block, the denominator embryo, the
  crispant numerator and the predicted numerator. with a dataset-wide draw the crispant
  embryos' heart numbers track how many cells each embryo gave (rho +0.86) and what
  fraction of them were heart (rho -0.86), which is dissection and not phenotype.
  scoping into the heart drops the between-embryo sd from 0.110 to 0.065.
- the crispant embryo is the unit, not the cell. between-embryo sd runs about 4x the
  draw-to-draw sd, so a seed error bar understates the spread by a lot.
- the denominator embryo is a fixed cycle over the sorted names rather than a random
  pick, so two runs at the same n share it exactly.
- the statistic is the log ratio, centred on its median over the scope. the measured
  column carries a technical offset the predicted one does not, so only the centred
  columns compare. the level inside a scope is not identified by this design; that needs
  a second crispant target through the same pipeline. `over_draws` hands the median back
  with the column, so the uncentred log ratio -- where 0 is a 1:1 distance, the scale
  nn_distortion.py is on -- is the sum of the two.
"""
import numpy as np
import pandas as pd

from analysis.pipelines.tbx5.support_comparison import common as C
from analysis.pipelines.tbx5.support_comparison.nn_distortion import nn_dist


def rotation(embryo):
    """query embryo -> (denominator embryo, numerator embryo), one deterministic cycle.

    a predicted cell is a reference cell, so three distinct embryos are needed per query
    cell: its own, the denominator's and the numerator's. a random pick would make two
    runs incomparable, so the cycle is fixed on the sorted names.
    """
    e = sorted(pd.unique(embryo))
    assert len(e) >= 3, f"{len(e)} reference embryos, need 3"
    return {x: (e[(i + 1) % len(e)], e[(i + 2) % len(e)]) for i, x in enumerate(e)}


def by_embryo(embryo, at=None):
    """{embryo: its distinct positions}, over `at` if given else over every row.

    `embryo` is one label per row of the block the positions index into, so a predicted
    pool given with multiplicity comes back deduplicated -- a nearest neighbour distance
    is a function of the support and repeat leaves add nothing to it.
    """
    at = np.arange(len(embryo)) if at is None else np.unique(at)
    e = np.asarray(embryo)[at]
    return {k: at[e == k] for k in sorted(pd.unique(e))}


def draw_size(ref, pred, cri, cap=C.N_MAX):
    """the n every draw in one scope is taken at.

    half the thinnest reference embryo, so a denominator draw samples its embryo rather
    than being nearly all of it; the thinnest predicted embryo; the thinnest crispant
    embryo; and the cap. one n per scope, so no source reads as distorted for being
    thinner than another.
    """
    return int(min([cap, min(len(v) for v in ref.values()) // 2,
                    min(len(v) for v in pred.values()),
                    min(len(v) for v in cri.values())]))


def denominator(zq, embryo, ref, rot, n, seed):
    """log nn distance from every query cell to a draw from its denominator embryo.

    depends only on (zq, embryo, ref, n, seed), so measured and predicted scored at one n
    share it and their columns subtract.
    """
    out = np.empty(len(zq))
    rng = np.random.default_rng(seed)
    for e, (den, _) in rot.items():
        at = np.flatnonzero(embryo == e)
        out[at] = np.log(nn_dist(zq[rng.choice(ref[den], n, replace=False)], zq[at]))
    return out


def foreign(zp, zq, pool, n, seed):
    """log nn distance to a draw from a perturbed set of foreign cells -- the crispants.

    they are not reference cells, so nothing can win both searches and one draw serves
    every query cell.
    """
    rng = np.random.default_rng(seed)
    return np.log(nn_dist(zp[rng.choice(pool, n, replace=False)], zq))


def resident(zq, embryo, pred, rot, n, seed):
    """log nn distance to a draw from a perturbed set made of reference cells -- the
    model's predictions.

    the numerator is taken from the third embryo, neither the query cell's own nor the
    denominator's. without it the two draws overlap, the same cell wins both searches and
    the ratio is pinned at exactly 1.
    """
    out = np.empty(len(zq))
    rng = np.random.default_rng(seed)
    for e, (_, num) in rot.items():
        at = np.flatnonzero(embryo == e)
        out[at] = np.log(nn_dist(zq[rng.choice(pred[num], n, replace=False)], zq[at]))
    return out


def over_draws(numerator, den, draws):
    """per query cell, the centred log ratio averaged over `draws` numerator draws, and
    the median it was centred on, averaged the same way.

    `numerator(seed)` returns one draw's log distances and `den[seed]` is that seed's
    denominator. averaging here rather than after the subtype means keeps draw noise out
    of the between-unit spread, which is what the error bar is taken from.

    the centring is per draw, so it cannot be undone from the returned column alone; the
    offset comes back with it, and adding it back gives the uncentred log ratio.
    """
    x = np.array([numerator(s) - den[s] for s in range(draws)])
    m = np.median(x, axis=1)
    return np.mean(x - m[:, None], axis=0), float(m.mean())
