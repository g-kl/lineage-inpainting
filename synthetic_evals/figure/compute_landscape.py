"""distill the crossing-landscape eval outputs into the coherence caches.

reads crossing_landscape/{candidates.csv,results.pkl} (produced by
synthetic_evals.crossing_landscape_eval), scores each method's forward predictions by
child-type accuracy (0.5 = draws from the pooled marginal; clipped below at
style.ACC_VMIN), flags each motif OT-like / Felsenstein-like (top-fractions of the
candidate landscape's two likeness axes), clusters the motifs by their four-method
accuracy profile, and writes into numerical_results/:
  landscape_heat.csv      one row per motif in clustered order: flags + clipped accuracies
  landscape_bars.csv      per-category mean / sem / n of the (unclipped) accuracies, for the text
  landscape_examples.csv  one representative motif per category (cell types at the nodes)
  landscape_meta.csv      motifs evaluated, candidates in the landscape, thresholds

    python -m synthetic_evals.figure.compute_landscape
"""
import os
import pickle

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist

from synthetic_evals.figure import style as S

LEAF_KEYS = ["anchorA", "A2", "B1", "anchorB"]
NODE_KEYS = ["pA", "pB"] + LEAF_KEYS


def _thresholds(cand):
    return (float(np.quantile(cand["ot_coherence"], 1 - S.OT_TOP_FRAC)),
            float(np.quantile(cand["fels_coherence"], 1 - S.FELS_TOP_FRAC)))


def _category(ot, fels, ot_thr, fels_thr):
    return np.where(np.asarray(ot) >= ot_thr, "ot",
                    np.where(np.asarray(fels) >= fels_thr, "fels", "path"))


def _pick_examples(cand, ot_thr, fels_thr):
    """one motif per category, all three from a single six-set of cell types rewired
    three ways: candidates.csv enumerates every regrouping of a fixed six-set (at
    alpha=0 the scores depend only on the leaf->parent assignment), so grouping by the
    six-set gives triples where the progenitors stay put and only the leaves shuffle.
    picks the six-set whose weakest category sits furthest from its threshold. these are
    illustrative -- no six-set of the *evaluated* motifs is rewired into all three
    categories, so the examples are generally not columns of the clustermap."""
    cand = cand.reset_index(drop=True)
    cand["category"] = _category(cand["ot_coherence"], cand["fels_coherence"],
                                 ot_thr, fels_thr)
    cand["depth"] = np.minimum((cand["ot_coherence"] - ot_thr).abs(),
                               (cand["fels_coherence"] - fels_thr).abs())

    def six_key(r):
        return (tuple(sorted([r.pA, r.pB])), tuple(sorted(r[k] for k in LEAF_KEYS)))

    best, best_score = None, -np.inf
    for (prog, leaves), idx in cand.groupby(cand.apply(six_key, axis=1)).groups.items():
        if set(prog) & set(leaves):             # no progenitor reused as a leaf
            continue
        sub = cand.loc[idx]
        reps = {c: g.loc[g["depth"].idxmax()] for c, g in sub.groupby("category")}
        if len(reps) < len(S.CATEGORIES):
            continue
        score = min(r["depth"] for r in reps.values())
        if score > best_score:
            best_score, best = score, reps
    cols = ["category"] + NODE_KEYS + ["ot_coherence", "fels_coherence"]
    return pd.DataFrame([best[c][cols] for c in S.CATEGORIES])


def _type_acc(samples):
    """child-type accuracy per method on one motif. every true and predicted child is
    called to the nearest of the motif's four child-type centroids, and each lineage
    scores 1 - TV between its predicted and true child-type histograms. mean over the two
    lineages (the rows a/b fate accuracy, pooled per lineage like the energy was)."""
    x, grp = np.asarray(samples["x_tgt"], np.float64), samples["leaf_group"]
    # eval rows run tree by tree as (mA anchor, mA inner, mB anchor, mB inner)
    assert (grp == np.tile([0, 0, 1, 1], len(grp) // 4)).all(), "unexpected leaf order"
    lab = 2 * grp + np.arange(len(grp)) % 2
    mu = np.stack([x[lab == k].mean(0) for k in range(4)])

    def call(A):
        return ((np.asarray(A, np.float64)[..., None, :] - mu) ** 2).sum(-1).argmin(-1)

    true = call(x)
    out = {}
    for m, pred in samples["pred"].items():
        pc = call(pred)                                                  # (S, R)
        accs = []
        for g in (0, 1):
            ht = np.bincount(true[grp == g], minlength=4) / (grp == g).sum()
            hp = np.bincount(pc[:, grp == g].ravel(), minlength=4) / pc[:, grp == g].size
            accs.append(1.0 - 0.5 * np.abs(ht - hp).sum())
        out[m] = float(np.mean(accs))
    return out


def _cluster_order(X):
    """column order: ward clustering of the motifs by accuracy profile, optimally ordered."""
    return leaves_list(optimal_leaf_ordering(linkage(X, "ward"), pdist(X)))


def main():
    with open(os.path.join(S.LANDSCAPE_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)
    cand = pd.read_csv(os.path.join(S.LANDSCAPE_DIR, "candidates.csv"))
    ot_thr, fels_thr = _thresholds(cand)

    motifs = pd.DataFrame([
        dict(cfg=r["dataset"], ot_coherence=r["ot_coherence"],
             fels_coherence=r["fels_coherence"], **_type_acc(r["samples"]))
        for r in results
    ])
    motifs["category"] = _category(motifs["ot_coherence"], motifs["fels_coherence"],
                                   ot_thr, fels_thr)
    # the pooler ablations are kept when the eval ran with --variants, for the
    # supplement's widened clustermap; the main page reads S.MAIN_METHODS out of these.
    kept = S.methods_present(motifs.columns)

    bars = pd.DataFrame([
        dict(category=c, method=m, mean=g[m].mean(), sem=g[m].sem(), n=len(g))
        for c, g in motifs.groupby("category") for m in kept
    ])
    bars["category"] = pd.Categorical(bars["category"], S.CATEGORIES)
    bars.sort_values(["category", "method"]).to_csv(
        os.path.join(S.NR, "landscape_bars.csv"), index=False)

    heat = motifs.copy()
    heat[kept] = heat[kept].clip(lower=S.ACC_VMIN)
    heat["ot_like"] = heat["ot_coherence"] >= ot_thr
    heat["fels_like"] = heat["fels_coherence"] >= fels_thr
    # cluster on S.METHODS only, never on `kept`: the ablations must not be able to
    # reorder the columns, so the main page and the supplement share one column order.
    # reversed so the OT-like block sits on the left.
    heat = heat.iloc[_cluster_order(heat[S.METHODS].to_numpy())[::-1]]
    heat[["cfg", "ot_like", "fels_like"] + kept].to_csv(
        os.path.join(S.NR, "landscape_heat.csv"), index=False)

    _pick_examples(cand, ot_thr, fels_thr).to_csv(
        os.path.join(S.NR, "landscape_examples.csv"), index=False)
    pd.DataFrame([dict(n_motifs=len(motifs), n_candidates=len(cand),
                       ot_thr=ot_thr, fels_thr=fels_thr)]).to_csv(
        os.path.join(S.NR, "landscape_meta.csv"), index=False)

    counts = motifs["category"].value_counts().to_dict()
    print(f"wrote landscape caches -> {S.NR}  ({len(motifs)} motifs of {len(cand)} "
          f"candidates, {counts}, thresholds ot>={ot_thr:.4g}, fels>={fels_thr:.4g})")


if __name__ == "__main__":
    main()
