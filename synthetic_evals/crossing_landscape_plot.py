#!/usr/bin/env python
"""Plot the crossing landscape: where each method wins in the (OT, Felsenstein) plane.

Reads what crossing_landscape_eval.py wrote (candidates.csv + landscape.csv) and
draws one row of forward (split: parent -> children) panels sharing a single
colour scale so methods are directly comparable:

    [ ours ] [ FM ] [ GENOT-Sinkhorn ] [ GENOT-FGW ] [ best method ]

Each method panel places the evaluated motifs at their (OT, Felsenstein) coherence
percentile coordinates (the space the eval selects targets in), coloured by
energy-distance error normalised by the per-motif
marginal baseline (randomly drawing from the correct pooled marginal): 0 = perfect,
1 = as bad as that random baseline, >1 clipped to the top of the scale. Capping at
the marginal keeps the interesting 0..1 band legible instead of letting a badly
failing method blow out the range. The faint grey cloud behind is the full
candidate set, for context. Once enough motifs are evaluated (>= HEATMAP_MIN) the
panels also fill a smooth interpolated field behind the points -- i.e. they become
the per-method heatmap the axes invite; below that threshold they stay scatters. (The "marg" baseline is the normaliser, not
its own panel; runs without it fall back to a shared raw-energy scale.)

The final panel is the decision-relevant one: best-baseline-minus-ours (also in
marginal units), on a diverging scale (blue = ours wins).

Pure plotting, no compute:
    PYTHONPATH=. python synthetic_evals/crossing_landscape_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "crossing_landscape"
HEATMAP_MIN = 20                 # interpolate a smooth field once >= this many motifs
# Only visualise candidates/motifs in the per-axis coherence window [p(FLOOR),
# p(100-TOP_CLIP)]: drop below OT_FLOOR on OT coherence and below FELS_FLOOR on
# Felsenstein coherence (the anti-coherent tail), and the most-coherent corner, so the
# plot focuses on the bulk. Independent of the eval's gate -- keep the two in sync by
# hand (mirror OT_FLOOR/FELS_FLOOR/TOP_CLIP). OT_FLOOR=None => plot all.
OT_FLOOR = 20
FELS_FLOOR = 50
TOP_CLIP = 0
METHODS = [("ours", "ours (lineage-aware)"),
           ("fm", "multimarginal FM"),
           ("gs", "GENOT-Sinkhorn"),
           ("gf", "GENOT-FGW")]
# For the "best method" panel: the marginal baseline is a contender here (not just
# the normaliser), so give every method a fixed label + colour for the legend.
WIN_LABELS = dict(METHODS) | {"marg": "random marginal"}
WIN_COLORS = {"ours": "#1f77b4", "fm": "#ff7f0e", "gs": "#2ca02c",
              "gf": "#d62728", "marg": "#7f7f7f"}
XLAB = "OT coherence   (percentile)"
YLAB = "Felsenstein coherence   (percentile)"


def _pctile_mapper(ref):
    """f(values) -> percentile in [0, 100] against the sorted reference array."""
    s = np.sort(np.asarray(ref, float))
    n = len(s)
    return lambda v: 100.0 * np.searchsorted(s, np.asarray(v, float), side="right") / n


def _panel(ax, x, y, z, *, cmap, vmin, vmax, cloud, title, diverging=False):
    """One (OT, Felsenstein) panel: candidate cloud + evaluated motifs by value.

    ``z`` is clipped to ``[vmin, vmax]`` so values beyond the cap saturate at the
    end of the colour scale rather than blowing it out.
    """
    zc = np.clip(z, vmin, vmax)
    ax.scatter(cloud[0], cloud[1], s=10, c="0.85", edgecolors="none", zorder=1)
    if len(x) >= HEATMAP_MIN and not diverging:
        ax.tricontourf(x, y, zc, levels=np.linspace(vmin, vmax, 13), cmap=cmap,
                       alpha=0.85, zorder=2, extend="max")
    sc = ax.scatter(x, y, c=zc, s=130, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors="k", linewidths=0.7, zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(XLAB, fontsize=8)
    return sc


def main():
    land = pd.read_csv(OUT_DIR / "landscape.csv")
    cand = pd.read_csv(OUT_DIR / "candidates.csv")

    # Plot on QUANTILE axes: map each coherence to its percentile within the FULL
    # candidate set (uniform marginals), matching the space the eval selects targets in.
    ot_pct = _pctile_mapper(cand["ot_coherence"])
    fels_pct = _pctile_mapper(cand["fels_coherence"])
    for df in (cand, land):
        df["ot_q"] = ot_pct(df["ot_coherence"])
        df["fels_q"] = fels_pct(df["fels_coherence"])

    # Gate the cloud + evaluated motifs to the eval's coherence window, computed on RAW
    # values so it matches the eval exactly, then display in percentile space. Thresholds
    # come from the full candidate set (kept in sync with the eval's PERCENTILE by hand).
    if OT_FLOOR is not None:
        ot_lo, ot_hi = np.percentile(cand["ot_coherence"], [OT_FLOOR, 100 - TOP_CLIP])
        fels_lo, fels_hi = np.percentile(cand["fels_coherence"], [FELS_FLOOR, 100 - TOP_CLIP])
        keep = lambda df: df[(df["ot_coherence"].between(ot_lo, ot_hi))
                             & (df["fels_coherence"].between(fels_lo, fels_hi))]
        n_cand, n_land = len(cand), len(land)
        cand, land = keep(cand), keep(land)
        print(f"coherence gate [ot>=p{OT_FLOOR}, fels>=p{FELS_FLOOR}, clip {TOP_CLIP}%]: "
              f"ot in [{ot_lo:.3f}, {ot_hi:.3f}] & fels in [{fels_lo:.3f}, {fels_hi:.3f}] "
              f"-> cloud {len(cand)}/{n_cand}, landscape rows {len(land)}/{n_land}")

    cloud = (cand["ot_q"].to_numpy(), cand["fels_q"].to_numpy())

    # Pad the (percentile) axes by a fraction of each range so gate-edge points aren't
    # jammed against the frame.
    def _lims(v, frac=0.05):
        lo, hi = float(v.min()), float(v.max())
        pad = (hi - lo) * frac or 0.02
        return lo - pad, hi + pad
    xlim = _lims(cand["ot_q"])
    ylim = _lims(cand["fels_q"])

    # Only plot the methods actually present; add a margin panel iff ours + >=1 baseline.
    # "marg" is the marginal baseline: not drawn as its own panel, but used to
    # normalise every other method (error / marginal) and cap the colour scale at 1.
    have = set(land["method"])
    methods = [(m, l) for m, l in METHODS if m in have]
    has_marg = "marg" in have
    # Candidate methods for the "best method" panel: every method present, marginal
    # baseline included. Show that panel whenever there are >= 2 to compare.
    win_keys = [m for m in ["ours", "fm", "gs", "gf", "marg"] if m in have]
    show_winner = len(win_keys) >= 2
    ncol = len(methods) + (1 if show_winner else 0)

    # constrained layout reserves room for the colorbars (tight_layout warns it can't
    # and lets them overlap the panels). Single row: forward (split) only.
    fig, axes = plt.subplots(1, ncol, figsize=(4.1 * ncol, 4.6), squeeze=False,
                             layout="constrained")
    row = axes[0]

    # one row per motif, wide over methods, aligned to a fixed cfg order.
    piv = land.pivot_table(index=["cfg", "ot_q", "fels_q"],
                           columns="method", values="energy").reset_index()
    x, y = piv["ot_q"].to_numpy(), piv["fels_q"].to_numpy()

    # Normalise each method's error by the per-motif marginal baseline: 0 = perfect,
    # 1 = as bad as random draws from the correct marginal, >1 (clipped) = worse.
    # Without a "marg" method fall back to a shared raw-energy scale.
    if has_marg:
        marg = piv["marg"].to_numpy()
        def norm(a): return a / marg
        vmin, vmax = 0.0, 1.0
        cb_label = "error / marginal  (1 = random)"
    else:
        def norm(a): return a
        vmin, vmax = 0.0, float(np.nanmax(land["energy"]))
        cb_label = "energy error (lower better)"

    for col, (m, label) in enumerate(methods):
        sc = _panel(row[col], x, y, norm(piv[m].to_numpy()),
                    cmap="viridis", vmin=vmin, vmax=vmax, cloud=cloud, title=f"{label}")
        row[col].set_xlim(xlim); row[col].set_ylim(ylim)
    cb = fig.colorbar(sc, ax=row[:len(methods)].tolist())
    cb.set_label(cb_label, fontsize=8)

    # best method: colour each motif by which method (marginal baseline included)
    # has the lowest energy there -- a discrete winner map, not a magnitude scale.
    if show_winner:
        E = np.stack([piv[m].to_numpy() for m in win_keys], axis=1)  # motif x method
        best = np.array([win_keys[i] for i in np.nanargmin(E, axis=1)])
        wp = row[-1]
        wp.scatter(cloud[0], cloud[1], s=10, c="0.85", edgecolors="none", zorder=1)
        for m in win_keys:                       # one scatter per winner -> legend
            mask = best == m
            if mask.any():
                wp.scatter(x[mask], y[mask], c=WIN_COLORS[m], s=130, edgecolors="k",
                           linewidths=0.7, zorder=3, label=WIN_LABELS[m])
        wp.set_title("best method (lowest energy)", fontsize=10)
        wp.set_xlabel(XLAB, fontsize=8)
        wp.set_xlim(xlim); wp.set_ylim(ylim)
        wp.legend(loc="best", fontsize=7, framealpha=0.9)

    row[0].set_ylabel(f"forward  (split: parent -> children)\n\n{YLAB}", fontsize=9)

    fig.suptitle("Crossing landscape: forward method error across the OT / Felsenstein plane",
                 fontsize=13)
    out = OUT_DIR / "crossing_landscape.png"
    fig.savefig(out, dpi=140)
    print(f"saved -> {out}  ({piv.shape[0]} motifs/direction, "
          f"{'heatmap' if piv.shape[0] >= HEATMAP_MIN else 'scatter'} mode)")


if __name__ == "__main__":
    main()
