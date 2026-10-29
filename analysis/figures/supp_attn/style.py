"""paths, palette, type and geometry for the attention supplemental figure."""
import os

import matplotlib as mpl
import seaborn as sns

from utils.figure_labels import FONT_FAMILY, register_fonts

register_fonts()

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.join(HERE, "numerical_results")
IMGS = os.path.join(HERE, "imgs")
PANELS = os.path.join(IMGS, "panels")

## PALETTE ##
LAYER = ["#22335c", "#1f8a8c", "#d9903c", "#bc4a5e"]   # one per layer, panels a and b
DIVERGE = sns.color_palette("vlag", as_cmap=True)      # log2 attention / uniform, panel c
OBS, MASKED = "#8c2f4a", "#5d6a80"                     # key strip over the heatmaps
INK, MUTED = "#1c1c1c", "#6b6b6b"

## TYPE (pt) ##
FS_LABEL, FS_TICK, FS_LEGEND, FS_SMALL = 10, 9, 9, 8

## PANEL C ##
EXAMPLE_HEAD = (2, 2)      # (layer, head) drawn
STEPS = (0, 5, 10, 15)     # rollout steps drawn
LOG2_LIM = 4               # colour limit, log2 attention / uniform

## GEOMETRY (inches) ##
PAGE_W = 6.5
TOP_W, TOP_H = 3.25, 2.35                  # panels a and b
TOP_AX = (0.66, 0.46, 0.10, 0.22)          # left, bottom, right, top margins of a and b
C_LEFT, C_TOP, C_BOTTOM = 0.44, 0.44, 0.46
C_HM, C_GAP = 1.20, 0.22                   # heatmap side, gap between heatmaps
C_CB_GAP, C_CB_W = 0.12, 0.08              # colorbar offset and width


def apply_rc():
    mpl.rcParams.update({
        "font.family": FONT_FAMILY, "mathtext.fontset": "cm",
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND, "axes.linewidth": 0.7,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.unicode_minus": False,
        "svg.fonttype": "path", "figure.dpi": 220,
    })


def save_svg(fig, name):
    """write a panel at its exact size into imgs/panels/, return (path, (w_in, h_in))."""
    import matplotlib.pyplot as plt
    os.makedirs(PANELS, exist_ok=True)
    path = os.path.join(PANELS, f"{name}.svg")
    fig.savefig(path, format="svg", transparent=True)
    w, h = fig.get_size_inches()
    plt.close(fig)
    return path, (float(w), float(h))
