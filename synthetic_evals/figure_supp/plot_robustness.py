"""robustness row: fate accuracy against each metric, normalised by the marginal.

reads numerical_results/robust_metrics.csv.

    python -m synthetic_evals.figure_supp.plot_robustness
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from synthetic_evals.figure_supp import style as S

PANEL_H = 2.05          # tall enough for the square panels' labels
LEFT, RIGHT = 0.095, 0.995
WSPACE = 0.42


def _metric_panel(ax, df, methods, metric):
    """fate accuracy against the marginal-normalised error under `metric`, per motif."""
    d = df[df["metric"] == metric]
    for m in methods:
        g = d[d["method"] == m]
        ax.scatter(g["norm"], g["acc"], s=2.0, color=S.METHOD_COLORS[m], alpha=0.6, lw=0)
    r = d[["norm", "acc"]].corr(method="spearman").iloc[0, 1]
    ax.text(0.04, 0.04, rf"$\rho = {r:.3f}$", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=S.FS_SUB, color=S.INK)
    ax.set_xscale("log")
    ax.set_yticks([0, 0.5, 1])
    ax.set_box_aspect(1)
    ax.set_xlabel(f"{S.METRIC_LABELS[metric]} / marginal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def render(out_dir, name="supp_robustness.svg"):
    S.apply_rc()
    met = pd.read_csv(os.path.join(S.NR, "robust_metrics.csv"))
    methods = S.methods_present(met["method"])
    metrics = [m for m in S.METRICS if (met["metric"] == m).any()]

    fig = plt.figure(figsize=(S.PAGE_W_IN, PANEL_H))
    gs = fig.add_gridspec(1, len(metrics), left=LEFT, right=RIGHT, top=0.98,
                          bottom=0.34, wspace=WSPACE)
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        _metric_panel(ax, met, methods, metric)
        if i == 0:
            ax.set_ylabel("fate accuracy")
    return S.save_panel(fig, out_dir, name, (S.PAGE_W_IN, PANEL_H))


if __name__ == "__main__":
    print(render(S.IMGS))
