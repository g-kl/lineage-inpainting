"""Shared visual language for the synthetic-eval figures.

Single source of truth for palette, typography, the fate colormap, and the
energy-distance metric, so the crossing and context figures speak one
language. Import the semantic roles below rather than hard-coding hex; that is what keeps "our model = teal", "FM = ochre", etc.
consistent across every panel and every benchmark.

Semantic roles
--------------
METHOD       method identity colours (model + the three baselines), used
             identically for lines, markers, heatmap labels and legends.
fate_color   cell-fate / lineage identity on a single diverging cool->warm
             ramp, sampled at whatever discrete levels a panel needs.
TRUTH        ground-truth trajectories / reference geometry (dark grey).
PRESPLIT     undifferentiated / population-background cells (neutral grey).
ANCHOR       observed context cells (sibling anchors) -- warm gold.
EVENT_TINT   faint wash on "event" time bands (branch / commitment windows).
EVENT_GUIDE  dashed time-grid / event guide lines.
SEQ_CMAP     sequential colormap for magnitude heatmaps (e.g. energy distance).
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
#  Output
# --------------------------------------------------------------------------

DPI = 300

# --------------------------------------------------------------------------
#  Typography
# --------------------------------------------------------------------------

# Computer Modern Sans, the LaTeX sans-serif. Rendered via usetex with
# \sfdefault (text) + sansmath (maths), so text and maths share the one font and
# every glyph (incl. the minus in log-tick exponents) is present. "cmss10" is
# the matplotlib-bundled fallback if usetex is ever turned off.
FONT_FAMILY = ("cmss10", "DejaVu Sans")
FONT_BASE = 7.5
FONT_LABEL = 7.0
FONT_TITLE = 8.0
FONT_LEGEND = 7.0
FONT_ANNOT = 6.5


def apply_rc():
    """Install the shared rcParams. Call once at the top of each render."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(FONT_FAMILY) + ["sans-serif"],
        "text.usetex": True,
        "text.latex.preamble": (
            r"\renewcommand{\familydefault}{\sfdefault}"
            r"\usepackage{sansmath}\sansmath"
        ),
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_LABEL,
        "axes.titlesize": FONT_TITLE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.labelsize": FONT_BASE - 1.0,
        "ytick.labelsize": FONT_BASE - 1.0,
        "legend.frameon": False,
        "legend.fontsize": FONT_LEGEND,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": DPI,
        "figure.dpi": DPI,
    })


# --------------------------------------------------------------------------
#  Method palette (model + baselines)
# --------------------------------------------------------------------------

# Method hues are defined once in utils.palettes (shared across every figure);
# re-exported here so viz's own helpers keep working.
from utils.palettes import MODEL, FM, GS, GF, METHOD_COLORS as METHOD  # noqa: E402

# Human-readable labels + canonical marker per method (shared legends).
METHOD_LABEL = {
    "model": "ours", "fm": "multimarginal FM",
    "gs": "GENOT-Sinkhorn", "gf": "GENOT-FGW",
}
METHOD_MARKER = {"model": "s", "fm": "o", "gs": "^", "gf": "D"}


# --------------------------------------------------------------------------
#  Structural / identity colours
# --------------------------------------------------------------------------

TRUTH = "#3a3a3a"        # ground-truth trajectories, reference geometry
PRESPLIT = "#9a9a9a"     # undifferentiated / population background
ANCHOR = "#d4a72b"       # observed context cells (sibling anchors), warm gold
EVENT_TINT = "#f0f0f0"   # faint light-gray wash on event time bands (generation delineation)
EVENT_GUIDE = "#b9b9b9"  # dashed time-grid / event guides


# --------------------------------------------------------------------------
#  Fate colormap (lineage identity by drift mu)
# --------------------------------------------------------------------------

# One diverging cool->warm ramp. Negative drift reads cool/blue, positive reads
# warm/terracotta-red; the light midtones are only reached by intermediate
# families. Cells near mu=0 (pre-split / undifferentiated) are drawn in PRESPLIT
# grey instead, since the ramp's centre would wash out on white.
FATE_CMAP = LinearSegmentedColormap.from_list("fate", [
    (0.00, "#1f4e79"),   # deep blue   (most negative)
    (0.28, "#5f8fbd"),   # light blue
    (0.50, "#d7d2cb"),   # warm neutral (rarely sampled)
    (0.72, "#cd7350"),   # terracotta
    (1.00, "#a02c2c"),   # deep red    (most positive)
])

FATE_VMAX = 1.5          # mu magnitude that saturates the ramp
FATE_NEUTRAL_EPS = 0.1   # |mu| below this -> PRESPLIT grey


def fate_color(mu: float, vmax: float = FATE_VMAX):
    """Colour for a cell of drift ``mu`` on the shared diverging fate ramp.

    ``mu`` in ``[-vmax, vmax]`` maps to the full cool->warm range; |mu| below
    ``FATE_NEUTRAL_EPS`` returns the neutral PRESPLIT grey.
    """
    if mu is None or abs(mu) < FATE_NEUTRAL_EPS:
        return PRESPLIT
    t = 0.5 + 0.5 * float(np.clip(mu / vmax, -1.0, 1.0))
    return FATE_CMAP(t)


# --------------------------------------------------------------------------
#  Sequential colormap (magnitude heatmaps)
# --------------------------------------------------------------------------

def _seq_cmap():
    try:
        import seaborn as sns
        return sns.color_palette("mako", as_cmap=True)
    except Exception:
        # Teal-anchored fallback tying the heatmap to the model colour.
        return LinearSegmentedColormap.from_list(
            "seq_teal", ["#f3efe9", MODEL, "#10243a"],
        )


SEQ_CMAP = _seq_cmap()


# --------------------------------------------------------------------------
#  Metric
# --------------------------------------------------------------------------

def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Energy distance between two clouds: the shared unbiased U-statistic
    (analysis.lib.metrics.energy_distance), the same estimator the real-data
    metrics report. a one-point reference cloud scores as the energy score."""
    from analysis.lib.metrics.energy_distance import energy_distance as _ed  # lazy:
    return _ed(a, b)                                       # pulls torch; viz is imported by plot-only scripts


def sem(values: np.ndarray) -> float:
    """Standard error of the mean (0 for a single sample, nan for empty)."""
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    if values.size == 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))


# --------------------------------------------------------------------------
#  Shared primitives
# --------------------------------------------------------------------------

def method_handles(keys):
    """Legend handles for ``keys`` (subset of METHOD), in the given order."""
    return [
        Line2D([], [], color=METHOD[k], marker=METHOD_MARKER[k],
               ms=4.5, lw=1.4)
        for k in keys
    ], [METHOD_LABEL[k] for k in keys]
