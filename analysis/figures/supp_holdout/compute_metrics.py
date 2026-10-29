"""fig 3's six metrics on the held-out clone trees, per method: mean and 95% interval.

reads the merged caches gather.py wrote into style.RESULTS and
aggregates them with fig 3's own code (summary_qc.plot_metrics._metric_values).
writes numerical_results/metrics.csv. cpu, a second.

    python -m analysis.figures.supp_holdout.compute_metrics
"""
import os

import pandas as pd

from analysis.figures.summary_qc.plot_metrics import _metric_values
from analysis.figures.supp_holdout import style as S


def main():
    rows = []
    for key, label, by_method in _metric_values(S.RESULTS):
        missing = set(S.METHODS) - set(by_method)
        assert not missing, (key, missing)
        rows += [(key, label, m, *by_method[m]) for m in S.METHODS]
    df = pd.DataFrame(rows, columns=["metric", "label", "method", "value", "lo", "hi"])
    df.to_csv(os.path.join(S.NR, "metrics.csv"), index=False)
    print(df.pivot(index="method", columns="metric", values="value")
          .reindex(S.METHODS)[list(dict.fromkeys(df.metric))]
          .to_string(float_format="%.4f"), flush=True)


if __name__ == "__main__":
    main()
