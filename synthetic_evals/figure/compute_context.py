"""fate-bias context-scaling eval pkl -> context_recon.pkl + context_scaling.csv.

reads context/results_<exp>.pkl. the scaling csv holds the pool-split fate-accuracy
curve vs context size n per sigma, plus the n-independent baseline levels.

    python -m synthetic_evals.figure.compute_context
"""
import os
import pickle

import pandas as pd

from synthetic_evals.context import plot as ctx
from synthetic_evals.figure import style as S

CTX_EXP = "synbench_fb_unshared_vpsde_idnn_leaf"


def main():
    with open(os.path.join(S.ROOT, "synthetic_evals", "context",
                           f"results_{CTX_EXP}.pkl"), "rb") as f:
        res = pickle.load(f)

    recon = {k: res[k] for k in ("case", "cartoon", "sigmas", "time_input", "leaf_time")}
    with open(os.path.join(S.NR, "context_recon.pkl"), "wb") as f:
        pickle.dump(recon, f)

    baselines = [m for m in S.METHODS if m != "ours"
                 and any(r["method"] == m for r in res["baseline_rows"])]
    rows = []
    for sigma in res["sigmas"]:
        ns, means, sems = ctx._model_scaling(res["splitter_rows"], sigma, key="acc")
        for n, m, s in zip(ns, means, sems):
            rows.append({"sigma": sigma, "kind": "model", "method": "ours",
                         "n": int(n), "mean": m, "sem": s})
        for method in baselines:
            m, s = ctx._baseline_level(res["baseline_rows"], sigma, method, key="acc")
            rows.append({"sigma": sigma, "kind": "baseline", "method": method,
                         "n": "", "mean": m, "sem": s})
    pd.DataFrame(rows).to_csv(os.path.join(S.NR, "context_scaling.csv"), index=False)

    print(f"wrote context caches -> {S.NR}  "
          f"({len(res['sigmas'])} sigmas, {len(rows)} scaling rows)")


if __name__ == "__main__":
    main()
