#!/usr/bin/env python
"""per embryo nn distortion, one scope at a time: every lineage, and the heart.

each crispant embryo is scored on its own against the same reference denominator, so the
error bar is the spread over the eight embryos rather than over draws. per_embryo.py lists
how it differs from nn_distortion.py and why.

the predicted side gets a leave-one-tree-out jackknife where the trees support it. in the
heart they do not: the screen grows its trees per reference replicate, so about 80% of a
tree's leaves land on its home replicate's cells, and after the third-embryo rule the
largest n a single tree covers for all three query embryos is 11. `n_jack` in the support
table is the n every jackknife replicate would run at; pass a smaller `--n` to buy the
interval at the cost of a noisier point.

    python -m analysis.pipelines.tbx5.support_comparison.per_embryo_nn_distortion
    python -m analysis.pipelines.tbx5.support_comparison.per_embryo_nn_distortion --scope heart --n 86
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from analysis.pipelines.tbx5.support_comparison import common as C
from analysis.pipelines.tbx5.support_comparison import per_embryo as P
from analysis.pipelines.tbx5.support_comparison import predicted as PR

DRAWS = 5        # numerator draws averaged per unit
MIN_DRAW = 100   # a scope needs this much on every side to be scored
HEART = "heart"  # the one scope that is a cell type set rather than a lineage
# seed offsets, so the three draws of one round are visibly independent of each other.
DEN, CRI, PRED = 0, 1_000, 2_000


def leaves(dev):
    """the predicted leaves that landed in the fit's reference block: rows and trees."""
    d = PR.leaves()
    d = d.assign(row=dev.index.get_indexer(pd.Index(d.cell)))
    return d.loc[d.row >= 0, ["tree", "row"]]


def scopes(d, dev, tgt):
    """{name: (reference mask, crispant mask)}: the lineages, then the heart."""
    out = {lin: ((dev.lineage == lin).values, (tgt.lineage == lin).values)
           for lin in C.lineages(d)}
    out[HEART] = (C.is_heart(dev).values, C.is_heart(tgt).values)
    return out


def run(name, dev, tgt, zd, zc, lv, keep_d, keep_t, n=None):
    """one scope: (support row, per cell table, per subtype table, centring offsets)."""
    q = np.flatnonzero(keep_d)
    zq, emb = zd[q], dev.embryo.values[q]
    into = np.full(len(dev), -1)
    into[q] = np.arange(len(q))
    inside = lv[keep_d[lv.row.values]]

    rot = P.rotation(emb)
    ref = P.by_embryo(emb)
    pred = P.by_embryo(emb, into[inside.row.values])
    cri = P.by_embryo(tgt.embryo.values, np.flatnonzero(keep_t))
    row = {"scope": name, "reference": len(q), "measured": int(keep_t.sum()),
           "predicted": sum(len(v) for v in pred.values()), "units": len(cri),
           "trees": inside.tree.nunique(), "n": 0, "n_auto": 0, "n_jack": 0, "jk": 0}
    # an embryo with no predicted cell in the scope cannot serve as a numerator, so the
    # rotation has nowhere to draw from and the scope is out.
    if set(pred) != set(ref) or not cri:
        print(f"  {name}: predicted covers {len(pred)} of {len(ref)} reference embryos, "
              f"skipped", flush=True)
        return row, None, None, None
    row["n_auto"] = auto = P.draw_size(ref, pred, cri)
    n = auto if n is None else int(n)
    assert n <= auto, f"{name}: n {n} over the {auto} this scope supports"
    row["n"] = n
    if n < MIN_DRAW:
        print(f"  {name}: n {n} under {MIN_DRAW}, skipped", flush=True)
        return row, None, None, None

    den = [P.denominator(zq, emb, ref, rot, n, DEN + s) for s in range(DRAWS)]
    meas = {c: P.over_draws(lambda s, p=p: P.foreign(zc, zq, p, n, CRI + s), den, DRAWS)
            for c, p in cri.items()}
    prd, off = P.over_draws(lambda s: P.resident(zq, emb, pred, rot, n, PRED + s), den,
                            DRAWS)
    # every column is centred on its own median, so the offsets are kept: the sum of the
    # two is the uncentred log ratio, and two scopes only compare on that.
    off = pd.Series({"predicted": off} | {f"measured_{c}": v[1] for c, v in meas.items()},
                    name="offset").rename_axis("column")
    meas = {c: v[0] for c, v in meas.items()}

    # leave one tree out. a tree whose removal leaves an embryo with no predicted cell is
    # out of the jackknife entirely, not scored at a smaller n.
    pools = {t: P.by_embryo(emb, into[inside.row.values[inside.tree.values != t]])
             for t in sorted(inside.tree.unique())} if inside.tree.nunique() > 1 else {}
    pools = {t: p for t, p in pools.items() if set(p) == set(ref)}
    row["n_jack"] = min([min(len(v) for v in p.values()) for p in pools.values()],
                        default=0)
    jk = {t: P.over_draws(lambda s, p=p: P.resident(zq, emb, p, rot, n, PRED + s),
                          den, DRAWS)[0]
          for t, p in pools.items() if min(len(v) for v in p.values()) >= n}
    row["jk"] = len(jk)

    cells = pd.DataFrame({"embryo": emb, "subtype": dev.subtype.values[q],
                          "cell_type": dev.cell_type.values[q], "predicted": prd},
                         index=dev.index[q])
    m = pd.DataFrame(meas, index=cells.index)
    cells["measured"] = m.mean(1)
    cells["spread"] = m.std(1)
    cells = pd.concat([cells, m.add_prefix("measured_")], axis=1)

    g = cells.groupby("subtype", observed=True)
    per = pd.DataFrame({c: g[f"measured_{c}"].mean() for c in cri})
    sub = pd.DataFrame({
        "cell_type": g.cell_type.first(), "cells": g.size(),
        "measured": per.mean(1), "measured_sem": per.std(1) / np.sqrt(per.shape[1]),
        "measured_min": per.min(1), "measured_max": per.max(1),
        "measured_pos": (per > 0).sum(1),
        "measured_p": ttest_1samp(per.values, 0, axis=1).pvalue,
        "predicted": g.predicted.mean()})
    if len(jk) > 1:
        j = pd.DataFrame({t: cells.assign(v=v).groupby("subtype", observed=True).v.mean()
                          for t, v in jk.items()})
        sub["predicted_jk_se"] = j.std(1) * np.sqrt(j.shape[1] - 1)
    print(f"  {name}: n {n:,} of {len(q):,} cells over {len(cri)} embryos, "
          f"{len(jk)} jackknife trees", flush=True)
    return row, cells, sub.sort_values("measured", ascending=False), off


def main(only=None, n=None):
    d = C.cells()
    zc, zd = C.latent()
    dev, tgt = d[d.dataset == "devmap"], d[d.dataset == C.TARGET]
    lv = leaves(dev)
    print(f"{lv.row.nunique():,} of {len(dev):,} reference cells are predicted Tbx5 "
          f"leaves, over {lv.tree.nunique()} trees", flush=True)

    sup = []
    for name, (keep_d, keep_t) in scopes(d, dev, tgt).items():
        if only and name != only:
            continue
        row, cells, sub, off = run(name, dev, tgt, zd, zc, lv, keep_d, keep_t, n)
        sup.append(row)
        if cells is None:
            continue
        cells.to_csv(C.out(f"per_embryo_{C.slug(name)}.csv"))
        sub.to_csv(C.out(f"per_embryo_{C.slug(name)}_subtype.csv"))
        off.to_csv(C.out(f"per_embryo_{C.slug(name)}_offset.csv"))
    s = pd.DataFrame(sup)
    if not only:
        s.to_csv(C.out("per_embryo_support.csv"), index=False)
    print(s.to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scope", help="one scope name, else every one of them")
    p.add_argument("--n", type=int, help="draw size, else the largest the scope supports")
    a = p.parse_args()
    main(a.scope, a.n)
