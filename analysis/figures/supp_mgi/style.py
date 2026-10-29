"""paths, protocol, palette and geometry for the MGI supplement. see concept.md.

the screen, the dataset and the TF list come from the perturb figure's style, so both read
the same screen. the off-manifold protocol is the Tbx5 figure's, freeze at 1.1.
"""
import os

import matplotlib as mpl

from analysis.figures.perturb.style import (  # noqa: F401  (re-exported)
    RING_SCREEN, RING_TARGET_TIME, RING_TOL, STATES_H5AD, TF_LIST)
from utils import palettes as P
from utils.figure_labels import FONT_FAMILY, register_fonts

register_fonts()

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.join(HERE, "numerical_results")
IMGS = os.path.join(HERE, "imgs")
PANELS = os.path.join(IMGS, "panels")
for d in (NR, PANELS):
    os.makedirs(d, exist_ok=True)

# the protocol the depletion is read under (readout.py): a node is off the manifold at a
# projection-to-local distance ratio of 1.1, and `freeze` stands in for the leaf that
# never formed
MODE, OFF_RATIO = "freeze", 1.1

## PANEL B: MGI pairs in the knockout's lowest third of its own E9.0 expression ##
# TFs only, the biggest rank gain from expression to model first, one pair per gene.
# knockouts that fully deplete more than MAX_FULL cell types are dropped (their ranks are
# ties at 1), and so are pairs the model depletes by less than MIN_DEPLETION
N_EXAMPLES = 6
MAX_FULL = 5
MIN_DEPLETION = 0.25

## PANEL C: one control subtree for each of three of b's examples ##
# the subtree is the one with the most leaves of the cell type that have the gene on above
# them, among those with at most MAX_LEAVES readout leaves (compute_subtrees.py)
SUBTREE_EXAMPLES = (("Mesp1", "first heart field"), ("Tbx1", "epicardium"),
                    ("Tbx6", "sclerotome"))
MAX_LEAVES = 40

## PALETTE ##
INK, GREY = P.INK, P.GREY
# b: the model keeps its shared colour, expression gets an ochre
SCORE_COLORS = {"model": P.MODEL, "expression": "#b8862f"}
# c: the edges are the gene's log1p CP10K in the node's snapped cell, on one scale per
# tree, and a leaf is its cell type. the three cell type hues stay off viridis and are
# chosen to stay distinguishable from each other and the grey (including under
# colour-vision deficiency)
EXPR_CMAP = P.EXPR_CMAP
LEAF_COLORS = dict(zip([ct for _, ct in SUBTREE_EXAMPLES],
                       ["#c2302b", "#d77fb3", "#e8a13a"]))
OTHER_LEAF = GREY

## TYPE (pt) ##
FS, FS_SMALL = 10.0, 9.0

## GEOMETRY (inches) ##
PAGE_W = 6.5
# row 1: a and b side by side
TOP_H = 2.45
SCATTER_W, EXAMPLES_W = 2.9, 3.6
AX_BOTTOM, AX_TOP = 0.65, 0.3   # under / over the axes, a and b
ROW_GAP = 0.1
# row 2: c's three subtrees side by side, each with its colour bar at its right
SUBTREE_H = 2.5
SUB_LEFT, SUB_W, SUB_PITCH = 0.25, 1.35, 2.0   # first tree's left edge, width, tree to tree
SUB_BOTTOM, SUB_TOP = 0.55, 0.4
SUB_CBAR = (0.08, 0.07)         # colour bar: gap after its tree, width
SUB_LW = (0.5, 1.6)             # stroke of an edge over one leaf, and over all of them
SUB_GRAD = 12                   # points an edge's expression gradient is drawn at


def apply_rc():
    mpl.rcParams.update({
        "font.family": FONT_FAMILY, "mathtext.fontset": "cm",
        "font.size": FS, "axes.labelsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS,
        "legend.fontsize": FS, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "svg.fonttype": "path", "figure.dpi": 200,
    })


def figure(w, h):
    """a figure of exactly (w, h) inches plus a `box` that places axes in inches."""
    import matplotlib.pyplot as plt
    apply_rc()
    fig = plt.figure(figsize=(w, h))
    return fig, lambda x, y, bw, bh: fig.add_axes([x / w, y / h, bw / w, bh / h])


def save_svg(fig, name):
    """write a panel at its exact size into imgs/panels/, return (path, (w_in, h_in))."""
    import matplotlib.pyplot as plt
    path = os.path.join(PANELS, f"{name}.svg")
    fig.savefig(path, format="svg", transparent=True)
    w, h = fig.get_size_inches()
    plt.close(fig)
    return path, (float(w), float(h))
