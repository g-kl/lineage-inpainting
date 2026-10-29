"""row-d robustness -> robust_abs.csv + robust_metrics.csv.

reads crossing_landscape/{landscape.csv,results.pkl}. row c in the main figure scores each
method by child-type accuracy, clipped below at chance (0.5); this sets the distributional
errors beside it:

  robust_abs.csv      per motif x method, the raw unclipped energy plus the motif's
                      marginal level, so the absolute scale and any error above the
                      marginal level are visible
  robust_metrics.csv  the same errors under every metric in style.METRICS, normalised by
                      that motif's marginal, so the ranking can be shown not to depend on
                      the metric, next to that motif x method's unclipped fate accuracy
                      (compute_landscape._type_acc on the stored predictions). only
                      populated when the eval ran with --extra-metrics.

    python -m synthetic_evals.figure_supp.compute_robustness
"""
import os
import pickle

import pandas as pd

from synthetic_evals.figure.compute_landscape import _type_acc
from synthetic_evals.figure_supp import style as S


def main():
    land = pd.read_csv(os.path.join(S.LANDSCAPE_DIR, "landscape.csv"))
    marg = land[land["method"] == "marg"].set_index("cfg")
    methods = S.methods_present(land["method"])

    rows = land[land["method"].isin(methods)].copy()
    rows["marg_energy"] = rows["cfg"].map(marg["energy"])
    with open(os.path.join(S.LANDSCAPE_DIR, "results.pkl"), "rb") as f:
        acc = pd.DataFrame([dict(cfg=r["dataset"], **_type_acc(r["samples"]))
                            for r in pickle.load(f)])
    rows = rows.merge(acc.melt("cfg", var_name="method", value_name="acc"),
                      on=["cfg", "method"], how="left")
    abs_cols = ["cfg", "method", "ot_coherence", "fels_coherence", "energy", "marg_energy"]
    rows[abs_cols].to_csv(os.path.join(S.NR, "robust_abs.csv"), index=False)

    have = [m for m in S.METRICS if m in land.columns and land[m].notna().any()]
    long = []
    for m in have:
        sub = rows[["cfg", "method", "ot_coherence", "fels_coherence", "acc", m]].copy()
        sub["metric"] = m
        sub["value"] = sub.pop(m)
        sub["norm"] = sub["value"] / sub["cfg"].map(marg[m])
        long.append(sub)
    out = pd.concat(long, ignore_index=True) if long else pd.DataFrame(
        columns=["cfg", "method", "ot_coherence", "fels_coherence", "acc", "metric",
                 "value", "norm"])
    out.to_csv(os.path.join(S.NR, "robust_metrics.csv"), index=False)

    print(f"wrote robustness caches -> {S.NR}  ({rows['cfg'].nunique()} motifs, "
          f"methods={methods}, metrics={have or 'energy only (re-run with --extra-metrics)'})")


if __name__ == "__main__":
    main()
