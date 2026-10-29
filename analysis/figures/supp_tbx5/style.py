"""paths, constants, palette and geometry for the Tbx5 supplement. see concept.md.

the Tbx5 figure's config comes along for its names, its subtype colours and its key,
window and label helpers, so the same subtype keeps the same colour on both sheets.
"""
import os

import matplotlib as mpl

from analysis.figures.tbx5.config import (  # noqa: F401  (re-exported)
    DIST_LABEL, HEART_SCOPE, HEART_TUBE, MEAS, N_UMAP_NEIGHBORS, PRED, SEED, SUBTYPE,
    TARGET, UMAP_MIN_DIST, bare, key, short, window)
from utils import palettes as P
from utils.figure_labels import FONT_FAMILY, register_fonts

register_fonts()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
NR = os.path.join(HERE, "numerical_results")
IMGS = os.path.join(HERE, "imgs")
PANELS = os.path.join(IMGS, "panels")
for d in (NR, PANELS):
    os.makedirs(d, exist_ok=True)

DEMUX = os.path.join(ROOT, "analysis", "results", "tbx5", "demultiplexing")
NE = "neural ectoderm"
REF, CRI = "devmap", TARGET          # the two values of common.cells().dataset

## PANEL E: lineage age ##
# each lineage's crispant cells against devmap cells of that lineage at two stages, all
# put through the support comparison's scvi encoder (embed_stages.py)
EARLY, ON_TIME = "E9.0", "E9.5"
AGE_STAGES = (EARLY, ON_TIME)
N_STAGE = 100_000       # devmap cells drawn per stage
N_INDEX = 5_000         # a lineage's devmap cells per stage, cut to the smaller stage
MIN_INDEX = 500         # cells a lineage needs at both stages, and in the crispant
N_QUERY = 5_000         # crispant cells of a lineage scored, at most

## PANELS F-H: the NE map ##
N_NE = 30_000           # NE cells per dataset in the umap draw

## PALETTE ##
INK, MUTED, GREY = P.INK, P.MUTED, P.GREY
# one hue per crispant embryo, a and b
EMBRYO = {"Tbx5_e0": "#3b6ea5", "Tbx5_e1": "#d9822b", "Tbx5_e2": "#2a9d8f",
          "Tbx5_e3": "#c7a22e", "Tbx5_e4": "#b5537a", "Tbx5_e5": "#5f8f3a",
          "Tbx5_e6": "#6c4fa3", "Tbx5_e7": "#c0443b"}
# d: the model keeps its shared colour, expression the MGI supplement's ochre
PREDICTOR = {"tbx5": "#b8862f", PRED: P.MODEL}
PREDICTOR_NAME = {"tbx5": f"{TARGET} expression", PRED: "model"}
# f: the progenitors one warm grey, the five neuron types a saturated hue each. the six
# are chosen to stay distinguishable (including under colour-vision deficiency)
NE_OTHER = "#d3cfc7"
NE_NEURON = {"spinal cord interneuron": "#1d4f91", "somatic motor neuron": "#0f8f7a",
             "cranial motor neuron": "#e39b1b", "midbrain neuron": "#8c2f39",
             "hypothalamic neuron": "#b07fd6"}
NE_SHORT = {"spinal cord interneuron": "sc interneuron",
            "somatic motor neuron": "somatic MN", "cranial motor neuron": "cranial MN"}
# g, h
DATASET = {REF: "#5b7fa6", CRI: "#e07a3f"}
DATASET_NAME = {REF: "E9.5 reference", CRI: f"{TARGET} crispant"}

## TYPE (pt) ##
FS_LABEL, FS_TICK, FS_KEY = 10, 9, 9
FS_NAME = 8.0           # e's lineage names: the one place below 9 pt

## GEOMETRY (inches) ##
PAGE_W = 6.5
ROW_GAP = 0.08
ROW_H = (2.05, 2.25, 2.25)
# row 1: a | b
SNP_W, DEPTH_W = 1.90, 4.60
SNP_MAP = (0.22, 0.80, 1.62, 1.12)
SNP_KEY = (0.10, 0.02, 1.76, 0.56)
SNP_BAR = (0.0, 0.0, 0.16)     # the axis bars: corner off the map's corner, length
DEPTH_BOX = (0.58, 0.48, 0.92, 1.30)   # the first strip; the others are +DEPTH_DX
DEPTH_DX = 1.54
# row 2: c | d | e
SCATTER_W, R_W, AGE_W = 2.20, 2.20, 2.10
SCATTER_BOX = (0.62, 0.50, 1.46, 1.45)
R_BOX = (0.62, 0.50, 1.54, 1.45)
R_KEY = (0.40, 2.02, 1.80, 0.20)       # one row over the bars
AGE_BOX = (0.66, 0.50, 1.38, 1.45)
# row 3: f | g | h
NE_W, NE_DS_W, NEURON_W = 2.40, 2.30, 1.80
NE_MAP = (0.12, 0.66, 2.16, 1.30)
NE_KEY = (0.02, 0.02, 2.36, 0.58)
NE_DS_KEY = (0.12, 0.22, 2.06, 0.38)
NEURON_BOX = (0.66, 0.66, 1.10, 1.30)

S_UMAP = 0.4            # point size on the snp and NE maps


def apply_rc():
    mpl.rcParams.update({
        "font.family": FONT_FAMILY, "mathtext.fontset": "cm",
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_KEY, "axes.linewidth": 0.6,
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
