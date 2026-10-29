"""paths, constants, palette and geometry for the convergence supp figure. see concept.md.

everything this figure reads is one of fig 4's caches, so fig 4's style comes along as
`C`: its paths, programs, cutoffs and colours. the same fates keep the same colours.
"""
import os

import matplotlib as mpl
import numpy as np

from analysis.figures.convergence import style as C  # noqa: F401  (re-exported)
from utils import palettes as P
from utils.figure_labels import FONT_FAMILY, register_fonts

register_fonts()

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.join(HERE, "numerical_results")
IMGS = os.path.join(HERE, "imgs")
PANELS = os.path.join(IMGS, "panels")
for d in (NR, PANELS):
    os.makedirs(d, exist_ok=True)

## PANEL A: top and bottom of the convergence ranking ##
N_TOP = 10

## PANEL C: the genes each program is built out of ##
# per sheet, the program blocks: label, which axes.npz program and axis, and which end
BLOCKS = {
    "ec": (("lateral plate", "ec", "x", "pos"),
           ("paraxial", "ec", "x", "neg"),
           ("endothelial identity", "ec", "y", "pos")),
    "neuro": (("posterior", "sci", "x", "pos"),
              ("anterior", "sci", "x", "neg"),
              ("interneuron identity", "sci", "y", "pos"),
              ("motor neuron identity", "smn", "y", "pos")),
}
BLOCK_LABEL = {"lateral plate": "LP", "paraxial": "paraxial",
               "endothelial identity": "EC", "posterior": "post.",
               "anterior": "ant.", "interneuron identity": "SCI",
               "motor neuron identity": "SMN"}
N_GENE_CELLS = 3000   # fit cells a gene is correlated with an axis score over
N_POOL = 25           # candidate genes per block, by correlation
N_GENES = 4           # genes kept per block
N_TYPES = 4           # source cell types kept as columns
N_CELLS = 800         # cells averaged per column
MIN_EXPR = 0.25       # a kept gene reaches this mean log1p in some column
MIN_DEST = 50         # trajectories a destination needs to get a column
# unresolved regional calls that their own subdivisions cover
DROP_SRC = {"ec": set(), "neuro": {"hindbrain", "spinal cord"}}
DEST_SHORT = {"endothelial cell": "EC", "spinal cord interneuron": "SCI",
              "somatic motor neuron": "SMN"}
EXPR_CMAP = P.EXPR_CMAP

## PANEL D: the erased share against the two cutoffs ##
XCUTS = np.round(np.arange(0.5, 2.501, 0.25), 3)     # E8.5 |log2 fc|, divergent above
YCUTS = np.round(np.arange(0.25, 1.001, 0.125), 3)   # E9.5 |log2 fc|, memory above
FATES = ("ec", "smn", "sci")
FATE_LABEL = {"ec": "endothelium", "sci": "interneuron", "smn": "motor neuron"}

## PALETTE ##
INK, MUTED, GREY = P.INK, P.MUTED, P.GREY
FAMILY_COL = {"endothelial": C.ACCENT, "neuron": C.CONV_OT_NEURON, "other": P.GREY}

## TYPE (pt) ##
FS_LABEL, FS_TICK, FS_LEGEND = 10, 9, 9
FS_ROW = 8.0          # table rows and strip gene names: the one place below 9 pt

## GEOMETRY (inches) ##
PAGE_W = 6.5
PT_PER_IN = 72.0


def family(ct):
    if ct in C.EC_CELL_TYPES:
        return "endothelial"
    if isinstance(ct, str) and "neuron" in ct:
        return "neuron"
    return "other"


def apply_rc():
    mpl.rcParams.update({
        "font.family": FONT_FAMILY, "mathtext.fontset": "cm",
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.unicode_minus": False,
        "svg.fonttype": "path", "figure.dpi": 220,
    })


def figure(w, h):
    """a figure of exactly (w, h) inches plus a `box` that places axes in inches."""
    import matplotlib.pyplot as plt
    apply_rc()
    fig = plt.figure(figsize=(w, h))
    return fig, lambda x, y, bw, bh: fig.add_axes([x / w, y / h, bw / w, bh / h])


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save_svg(fig, name):
    """write a panel at its exact size into imgs/panels/, return (path, (w_in, h_in))."""
    import matplotlib.pyplot as plt
    path = os.path.join(PANELS, f"{name}.svg")
    fig.savefig(path, format="svg", transparent=True)
    w, h = fig.get_size_inches()
    plt.close(fig)
    return path, (float(w), float(h))
