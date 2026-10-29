"""bottom row: the most subtype-specific heart-screen knockouts drawn on the
developing-heart diagram (read from the cached depletion + specificity csvs).

panel 0     anatomical key -- each drawable subtype in its HEART_SUBTYPE_COLORS hue
panels 1..N the same heart hued by that KO's per-subtype depletion % (shared
            0-100% HEART_DEPL_CMAP scale), N = HEART_N_PICK most specific KOs

geometry comes from the reusable utils/heart_diagram.py renderer; palette + config from
style.py.  writes imgs/heart_diagram.svg (+ .png).
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from analysis.figures.perturb import style as S
from utils import heart_diagram as HD

S.apply_rc()


HIDE = {"pSHF"}                      # drop the pSHF arc + label
LABEL_OVERRIDES = {"AV canal": "AV\ncanal"}


def resolve_picks(D):
    """explicit HEART_PICKS (case-insensitive match to the screen names), else the
    HEART_N_PICK lowest-entropy qualifying knockouts."""
    if S.HEART_PICKS is None:
        rank = pd.read_csv(os.path.join(S.NUMERICAL, "heart_specificity.csv"),
                           index_col=0)
        qualified = rank[rank["max_depl"] >= S.HEART_MIN_MAX_DEPL]
        return list(qualified.sort_values("norm_entropy").index[:S.HEART_N_PICK])
    lut = {g.lower(): g for g in D.index}
    picks = []
    for g in S.HEART_PICKS:
        if g.lower() in lut:
            picks.append(lut[g.lower()])
        else:
            print(f"  warning: {g} not in the heart screen, skipping", flush=True)
    return picks


def main():
    D = pd.read_csv(os.path.join(S.NUMERICAL, "heart_subtype_depletion.csv"),
                    index_col=0)
    cols = list(D.columns)
    picks = resolve_picks(D)
    print(f"drawing {len(picks)} KOs: {picks}", flush=True)

    n = len(picks)
    fig = plt.figure(figsize=(S.HEART_W_IN, 1.55))
    gs = fig.add_gridspec(1, n + 2, wspace=0.05,
                          width_ratios=[1] * (n + 1) + [0.09],
                          left=0.005, right=0.93, top=0.86, bottom=0.02)

    # panel 0: discrete-colour anatomical key (drawable subtypes only)
    ax = fig.add_subplot(gs[0])
    key = {S.HEART_SUBTYPE_TO_REGION[s]: S.HEART_SUBTYPE_COLORS[S.HEART_SUBTYPE_TO_REGION[s]]
           for s in cols}
    HD.plot_heart(colors=key, ax=ax, show_labels=True, colorbar=False,
                  default_color="#eeeeee", linewidth=0.4, hide_regions=HIDE,
                  draw_unassigned=False, label_overrides=LABEL_OVERRIDES,
                  label_fontsize=S.HEART_LABEL_FONTSIZE)

    # panels 1..n: hue each region by the KO's per-subtype depletion %
    for i, g in enumerate(picks, start=1):
        ax = fig.add_subplot(gs[i])
        values = {S.HEART_SUBTYPE_TO_REGION[s]: float(D.loc[g, s]) for s in cols}
        HD.plot_heart(values=values, cmap=S.HEART_DEPL_CMAP, vmin=0.0, vmax=100.0,
                      ax=ax, show_labels=False, colorbar=False,
                      default_color="#eeeeee", linewidth=0.4, hide_regions=HIDE,
                      draw_unassigned=False)
        ax.text(0.5, 1.05, g, transform=ax.transAxes, ha="center", va="baseline",
                fontsize=6.5, color=S.INK, style="italic", clip_on=False)

    # shared depletion colourbar
    cax = fig.add_subplot(gs[-1])
    sm = ScalarMappable(norm=Normalize(0.0, 100.0), cmap=S.HEART_DEPL_CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("KO depletion (%)", fontsize=5.5)
    cb.set_ticks([0, 50, 100])
    cb.ax.tick_params(labelsize=5, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)

    svg = os.path.join(S.IMGS, "heart_diagram.svg")
    png = os.path.join(S.IMGS, "heart_diagram.png")
    fig.savefig(svg, transparent=True)
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {svg}\nwrote {png}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
