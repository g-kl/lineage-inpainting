"""e: lineage age. reads numerical_results/age_energy.csv.

each point is a lineage: the energy distance from its crispant cells to the E9.0 devmap
cells of that lineage against the one to the E9.5 cells, both log. on the diagonal the
two are equal, and above it a lineage sits nearer E9.0 than E9.5. every point is named,
each name at a fixed offset (NAME_AT) and a name set off far from its point gets a
hairline back to it.

    python -m analysis.figures.supp_tbx5.plot_age
"""
import os

import numpy as np
import pandas as pd
from matplotlib import patheffects

from analysis.figures.supp_tbx5 import style as S

TICKS = (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
# log units past the points (low end, high end), room for the names
PAD_X, PAD_Y = (0.42, 0.25), (0.25, 0.45)
# each lineage's name: (dx, dy) in points off its point, and its alignment. neural crest
# and intermediate sit side by side, so their names go on opposite sides of the pair
NAME_AT = {"endoderm": (5, 2, "left"), "neural ectoderm": (0, -9, "center"),
           "lateral plate mesoderm": (0, 22, "center"), "neural crest": (0, -15, "center"),
           "intermediate mesoderm": (14, 14, "right"), "paraxial mesoderm": (0, -9, "center"),
           "surface ectoderm": (0, 9, "center")}
LEAD = 10.0             # points off its point past which a name gets a hairline


def log_lim(v, pad):
    """the range of v on a log axis, `pad` in log units past (the low, the high) end."""
    v = np.log(v)
    return np.exp(v.min() - pad[0]), np.exp(v.max() + pad[1])


def plain_ticks(ax):
    """the TICKS inside each log axis, written as plain numbers, no minor ticks."""
    for lim, put in ((ax.get_xlim(), ax.set_xticks), (ax.get_ylim(), ax.set_yticks)):
        t = [v for v in TICKS if lim[0] <= v <= lim[1]]
        put(t, [f"{v:g}" for v in t])
    ax.minorticks_off()


def render():
    a = pd.read_csv(os.path.join(S.NR, "age_energy.csv"))
    x, y = a[S.EARLY].to_numpy(), a[S.ON_TIME].to_numpy()
    assert (x > 0).all() and (y > 0).all(), "a crispant energy distance is not positive"

    fig, box = S.figure(S.AGE_W, S.ROW_H[1])
    ax = box(*S.AGE_BOX)
    lo, hi = log_lim(np.r_[x, y], (max(PAD_X), max(PAD_X)))
    ax.plot([lo, hi], [lo, hi], color=S.GREY, lw=0.7, zorder=1)
    ax.scatter(x, y, s=16, c=S.MUTED, lw=0.4, ec="white", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*log_lim(x, PAD_X))
    ax.set_ylim(*log_lim(y, PAD_Y))
    plain_ticks(ax)
    ax.set_xlabel(f"energy distance to {S.EARLY}")
    ax.set_ylabel(f"energy distance to {S.ON_TIME}")
    halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]
    for l, px, py in zip(a.lineage, x, y):
        dx, dy, ha = NAME_AT.get(l, (0, 9, "center"))
        lead = dict(arrowstyle="-", lw=0.4, color=S.GREY, shrinkA=1.0, shrinkB=3.0)
        ax.annotate(S.short(l), (px, py), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=S.FS_NAME, color=S.MUTED, zorder=4,
                    annotation_clip=False, path_effects=halo,
                    arrowprops=lead if np.hypot(dx, dy) > LEAD else None)
    return S.save_svg(fig, "e_age")


if __name__ == "__main__":
    print(render())
