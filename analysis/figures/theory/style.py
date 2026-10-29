"""Shared paths, geometry, typography, and colors for the theory figures.

Every script in this folder reads its inputs from numerical_results/ (except
the committed depth2_identification_results.json) and writes its renders to
imgs/; both are gitignored and regenerable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

from utils.figure_labels import FONT_FAMILY, register_fonts


register_fonts()

HERE = Path(__file__).resolve().parent
NR = HERE / "numerical_results"
IMGS = HERE / "imgs"
PANELS = IMGS / "panels"

PT_PER_IN = 72.0
PAGE_W_IN = 6.5
PANEL_GAP_IN = 0.08
ROW_GAP_IN = 0.075
PAGE_PAD_IN = 0.020
BOX_INSET_IN = 0.025
BOX_PAD_X_IN = 0.045
BOX_PAD_TOP_IN = 0.095
BOX_PAD_BOT_IN = 0.040
BOX_GAP_IN = 0.14
BOX_TITLES = (
    r"Bernstein unresolved whole-tree energy$^2$ + sampling variance = MSE",
)

FS_PANEL = 16.0
FS_BOX = 9.3
FS_TITLE = 8.6
FS_LABEL = 7.8
FS_TICK = 6.8
FS_SUB = 6.8
FS_LEGEND = 6.8

INK = "#222222"
MUTED = "#777777"
BOX_BORDER = "#b8c1cf"
PAPER = "#ffffff"
PANEL_FILL = "#fbfcfc"
MODEL = "#0f6b6e"
ORANGE = "#c47a3d"
PURPLE = "#6b5ed8"
RATIO_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "bernstein_ratio",
    ("#365F73", "#A9BDC5", "#F0ECE5", "#D8B0A2", "#A75F50"),
)
RATIO_LOG_MIN = -10.0
RATIO_LOG_MAX = 4.0
RATIO_TICKS = (-10, -4, 0, 2, 4)
RATIO_TICKLABELS = (
    r"$\leq 2^{-10}$", r"$1/16$", "1", "4", r"$\geq 16$",
)

Q_ORDERS = (2, 3, 4)
def apply_rc() -> None:
    """Apply the manuscript's compact publication style."""
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": FS_LABEL,
        "axes.linewidth": 0.65,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.frameon": False,
        "legend.fontsize": FS_LEGEND,
        "lines.solid_capstyle": "round",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_svg(fig, name: str) -> tuple[Path, tuple[float, float]]:
    """Save a panel at its exact physical size and return path and dimensions."""
    PANELS.mkdir(parents=True, exist_ok=True)
    path = PANELS / name
    fig.savefig(path, format="svg", transparent=True)
    width, height = fig.get_size_inches()
    return path, (float(width), float(height))
