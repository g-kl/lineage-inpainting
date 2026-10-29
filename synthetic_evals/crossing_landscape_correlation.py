#!/usr/bin/env python
"""per-method forward energy error regressed on each coherence axis.

one row x 2 cols (vs OT coherence / vs Felsenstein coherence); each panel scatters
every method with a least-squares line and its Pearson r. reads landscape.csv.

    PYTHONPATH=. python synthetic_evals/crossing_landscape_correlation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "crossing_landscape"
METHODS = [("ours", "#1b7837", "ours (lineage-aware)"),
           ("fm", "#d95f02", "multimarginal FM"),
           ("gs", "#7570b3", "GENOT-Sinkhorn"),
           ("gf", "#e7298a", "GENOT-FGW")]
AXES = [("ot_coherence", "OT coherence"), ("fels_coherence", "Felsenstein coherence")]


def main():
    land = pd.read_csv(OUT_DIR / "landscape.csv")
    have = set(land["method"])
    methods = [(m, c, n) for m, c, n in METHODS if m in have]

    fig, axes = plt.subplots(1, len(AXES), figsize=(6.2 * len(AXES), 5.0), squeeze=False)
    for c, (metric, mlabel) in enumerate(AXES):
        ax = axes[0][c]
        for m, color, name in methods:
            g = land[land["method"] == m].dropna(subset=[metric, "energy"])
            x, y = g[metric].to_numpy(), g["energy"].to_numpy()
            rho = np.corrcoef(x, y)[0, 1] if len(x) > 2 and x.std() > 0 else np.nan
            ax.scatter(x, y, s=55, color=color, edgecolors="k", linewidths=0.5,
                       label=f"{name}  (r={rho:+.2f})", zorder=3)
            if len(x) > 2 and x.std() > 0:                    # LS trend line
                b, a = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, b * xs + a, color=color, lw=1.3, alpha=0.7, zorder=2)
        ax.set_xlabel(mlabel)
        ax.set_ylabel("energy error")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Forward error vs setting axes  (n={land.cfg.nunique()} motifs)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "crossing_landscape_correlation.png"
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
