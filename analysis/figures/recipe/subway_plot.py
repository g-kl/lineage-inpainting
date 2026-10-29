#!/usr/bin/env python
"""radial map of fate-specifying TFs, one version per ranking.

the inner band is a per-lineage arc block holding that lineage's E7.5->E8.0
specifying TFs (computed at lineage level).  from the E8.0 ring outward each cell
type is its own spoke, with its later TFs written perpendicular across the spoke
at the radius of their divergence time.  light dotted rings mark the time bins;
spokes and blocks are coloured by lineage.  reads subway_data_<rank>.json and
writes subway_<rank>.pdf/.svg/.png.
"""

import os
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
VERSIONS = ["penetrance", "specificity"]

## STYLE ##
mpl.rcParams.update({
    "font.family": "cmss10",            # Computer Modern Sans (ships with mpl)
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "svg.fonttype": "path",
    "figure.dpi": 300,
})

INK = "#222222"
RING_COL = "#c8c8c8"
LABEL_BOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9)
# lineage palette for this panel (germ-layer families:
# ectoderm greens/teals, mesoderm purples + blood crimson, endoderm oranges).
LINEAGE_COLORS = {
    "neural ectoderm":         "#1B9E77",
    "ectoderm":                "#3CB371",
    "surface ectoderm":        "#66C2A5",
    "neural crest":            "#117A65",
    "extraembryonic ectoderm": "#A6D854",
    "mesoderm":                "#7570B3",
    "paraxial mesoderm":       "#5E4FA2",
    "lateral plate mesoderm":  "#8DA0CB",
    "intermediate mesoderm":   "#9970C2",
    "extraembryonic mesoderm": "#BCA8DD",
    "blood":                   "#C1121F",
    "endoderm":                "#D95F02",
    "extraembryonic endoderm": "#FDB462",
    "epiblast":                "#7F7F7F",
    "primordial germ cell":    "#E7298A",
}

## GEOMETRY (data units; rim radius = 1) ##
FIG_W, FIG_H = 6.5, 5.6
LEG_FRAC = 0.17              # left strip reserved for the legend
HUB_R = 0.18                # central hub radius
RIM_R = 1.00                # outermost ring (E9.0)
GROUP_GAP = 1.2             # angular gap between lineage groups (in cell-slot units)
LIM = 1.32                  # circle-axes half-extent (room for outer labels)

SPOKE_LW = 2.0
SPOKE_ALPHA = 0.65
STAGGER = 0.013             # radial offset alternating between spokes so adjacent
                           # spokes' labels interleave instead of colliding
BLOCK_ALPHA = 0.22          # lineage arc-block fill
LABEL_FS = 4.2
TF_FS = 3.7                 # gene-name font (typewriter), a touch smaller than labels
HUB_FS = 5.5
LEGEND_FS = 6.5


def angles(cells):
    """angle per cell, packed into lineage groups separated by GROUP_GAP; also the
    half-slot angular width used to give each lineage block its arc."""
    pos, out, prev = 0.0, [], None
    for c in cells:
        if prev is not None and c["lineage"] != prev:
            pos += GROUP_GAP
        out.append(pos)
        pos += 1.0
        prev = c["lineage"]
    span = pos - 1.0 + GROUP_GAP            # close the ring with one gap
    step = 2 * math.pi / span
    return [math.pi / 2 - step * p for p in out], step / 2


def upright(deg):
    return deg + 180 if 90 < deg % 360 < 270 else deg


def slot_radii(lo, hi, n, slots):
    """radii for n ticks on a fixed `slots`-position grid in (lo, hi), filled from
    the innermost slot.  fixing the grid (instead of spacing n ticks evenly) keeps
    every spoke's ticks on the same radii so the per-spoke stagger interleaves."""
    return np.linspace(lo, hi, slots + 2)[1:-1][:n]


def mathtt(s):
    """gene name as Computer Modern typewriter mathtext (\\texttt look); the unicode
    hyphen keeps Nkx2-5 etc. from being typeset as a spaced math minus."""
    return r"$\mathtt{%s}$" % s.replace("-", "‐")


def tf_label(ax, r, t, s):
    """gene name on a white box, perpendicular across the spoke at radius r."""
    x, y = r * math.cos(t), r * math.sin(t)
    ax.text(x, y, mathtt(s), rotation=upright(math.degrees(t) + 90),
            rotation_mode="anchor", ha="center", va="center", fontsize=TF_FS,
            color=INK, zorder=4, bbox=LABEL_BOX)


def plot_version(version):
    d = json.load(open(os.path.join(HERE, f"subway_data_{version}.json")))
    cells, bins, lineage_tfs = d["cells"], d["time_bins"], d["lineage_tfs"]
    slots = d["top_per_bin"]
    th, half = angles(cells)
    # rings at the inner (E7.5->E8.0 block edge) and the three outer bins
    nodes = np.linspace(HUB_R + (RIM_R - HUB_R) / len(bins), RIM_R, len(bins))
    block_r0, block_r1 = HUB_R, nodes[0]               # lineage block band
    seg_lo = nodes[:-1]                                # outer-bin segment starts

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([LEG_FRAC, 0, 1 - LEG_FRAC, 1])
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.axis("off")

    for rn in nodes:
        ax.add_patch(plt.Circle((0, 0), rn, fill=False, ec=RING_COL, lw=0.6,
                                ls=(0, (1, 2)), zorder=0))

    # lineage arc blocks (inner band) -- one wedge per contiguous lineage group
    groups = []
    for i, c in enumerate(cells):
        if not groups or c["lineage"] != cells[i - 1]["lineage"]:
            groups.append([i])
        else:
            groups[-1].append(i)
    for g in groups:
        lin = cells[g[0]]["lineage"]
        col = LINEAGE_COLORS.get(lin, INK)
        a_hi = math.degrees(th[g[0]] + half)           # clockwise: first cell is high
        a_lo = math.degrees(th[g[-1]] - half)
        ax.add_patch(Wedge((0, 0), block_r1, a_lo, a_hi, width=block_r1 - block_r0,
                           facecolor=col, edgecolor=col, lw=0.5, alpha=BLOCK_ALPHA,
                           zorder=1))
        tfs = lineage_tfs.get(lin, [])
        tc = (th[g[0]] + th[g[-1]]) / 2                 # block mid-angle
        rs = np.linspace(block_r0, block_r1, len(tfs) + 2)[1:-1]
        for tf, r in zip(tfs, rs):
            tf_label(ax, r, tc, tf)

    # cell-type spokes from the E8.0 ring outward, with their later TFs
    for i, (c, t) in enumerate(zip(cells, th)):
        col = LINEAGE_COLORS.get(c["lineage"], INK)
        u = (math.cos(t), math.sin(t))
        off = STAGGER if i % 2 else -STAGGER          # interleave neighbours' labels
        ax.plot([block_r1 * u[0], RIM_R * u[0]], [block_r1 * u[1], RIM_R * u[1]],
                color=col, lw=SPOKE_LW, alpha=SPOKE_ALPHA, solid_capstyle="round",
                zorder=2)
        for j, b in enumerate(bins[1:]):
            tfs = c["tfs"][str(b)]
            if not tfs:
                continue
            rs = slot_radii(seg_lo[j], nodes[j + 1], len(tfs), slots)
            for tf, r in zip(tfs, rs):
                tf_label(ax, r + off, t, tf)
        deg = math.degrees(t) % 360
        rot, ha = (deg + 180, "right") if 90 < deg < 270 else (deg, "left")
        end = (RIM_R + 0.04)
        ax.text(end * u[0], end * u[1], c["cell_type"], rotation=rot,
                rotation_mode="anchor", ha=ha, va="center", fontsize=LABEL_FS + 0.8,
                color=col, zorder=4)

    ax.add_patch(plt.Circle((0, 0), HUB_R, color="white", ec=INK, lw=0.6, zorder=3))
    ax.text(0, 0, "E7.5", ha="center", va="center", fontsize=HUB_FS, color=INK, zorder=4)

    # legend in its own left strip (lineages present, in plotted order)
    axl = fig.add_axes([0, 0, LEG_FRAC, 1])
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.axis("off")
    present = list(dict.fromkeys(c["lineage"] for c in cells))
    for i, lin in enumerate(present):
        y = 0.97 - i * 0.04
        axl.plot(0.10, y, "o", ms=5, color=LINEAGE_COLORS.get(lin, INK))
        axl.text(0.17, y, lin, ha="left", va="center", fontsize=LEGEND_FS, color=INK)

    out = os.path.join(HERE, f"subway_{version}")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".svg")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"wrote {out}.pdf/.svg/.png")


def main():
    for v in VERSIONS:
        plot_version(v)


if __name__ == "__main__":
    main()
