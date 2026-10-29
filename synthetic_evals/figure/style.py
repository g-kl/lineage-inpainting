"""paths, method palette, geometry and rc shared by the synthetic figure. see concept.md."""

import os

import matplotlib as mpl

from utils import palettes as P
from utils.figure_labels import register_fonts, FONT_FAMILY

register_fonts()   # shared Latin Modern Sans (utils/fonts/)

HERE = os.path.dirname(__file__)
NR = os.path.join(HERE, "numerical_results")
IMGS = os.path.join(HERE, "imgs")
os.makedirs(NR, exist_ok=True)
os.makedirs(IMGS, exist_ok=True)

# --- upstream heavy outputs the compute scripts distill from ------------------
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CROSS_EXP = "unshared_vpsde_idnn_leaf"
HD_DIR = os.path.join(ROOT, "synthetic_evals", "high_dim_crossing")
LANDSCAPE_DIR = os.path.join(ROOT, "synthetic_evals", "crossing_landscape")
# real-cell context motif (synthetic_evals/context_landscape_eval.py writes here)
CONTEXT_LANDSCAPE_DIR = os.path.join(ROOT, "synthetic_evals", "context_landscape")
RECON_DIM = 1                    # ambient dim shown in the recon rows + per-transition strip
DIMS = (1, 4, 16, 64)            # dims gathered for the fate-vs-dimension line plot
CROSS_SPLIT_PAIR = (7.0, 8.0)    # the second split: the one transition where the variants differ

# --- coherence categories for the landscape row -------------------------------
# OT-like = OT coherence in the top OT_TOP_FRAC of the candidate landscape;
# Felsenstein-like = likewise on the Felsenstein axis. OT-like implies
# Felsenstein-like, so with "neither" the three categories partition the motifs.
OT_TOP_FRAC, FELS_TOP_FRAC = 1 / 6, 1 / 2
CATEGORIES = ["ot", "fels", "path"]
CAT_LABELS = {"ot": "OT-like", "fels": "Felsenstein-like", "path": "pathological"}

# --- methods, shared across every panel (same method == same colour) ----------
# derived from utils.palettes, which is the one definition per method for every
# figure; the local "ours" key maps to palettes' "model". MAIN_METHODS is what the
# main-text page draws; METHODS adds the baselines only the supplement shows.
MAIN_METHODS = ["ours", "gs", "gf", "fm"]
METHODS = MAIN_METHODS + ["wot", "moslin"]
ABLATIONS = P.ABLATION_ORDER
_KEY = {"ours": "model"}                        # local name -> palettes key
METHOD_COLORS = {m: P.METHOD_COLORS[_KEY.get(m, m)] for m in METHODS}
METHOD_LABELS = {m: P.METHOD_LABELS[_KEY.get(m, m)] for m in METHODS}
METHOD_MARKERS = {m: P.METHOD_MARKERS[_KEY.get(m, m)] for m in METHODS}
METHOD_COLORS.update(P.ABLATION_COLORS)
METHOD_LABELS.update(P.ABLATION_LABELS)
METHOD_LABELS["ours"] = "Lineage inpainting"
# recon cache key per method (the *_pairs entry in the trimmed ev dicts).
RECON_CACHE_KEY = {"ours": "ps_pairs", **{m: f"{m}_pairs" for m in METHODS[1:]}}

# context-recon prediction glyph; must not read as the teal "ours" series colour
PRED_COLOR = P.PRED

# --- value colormaps (their own scales, not method palettes) -------------------
ERR_CMAP = P.LANDSCAPE_CMAP
# accuracy: the same ramp flipped so tan = good in both, cut short of its white end
# since accuracy hits 1 exactly.
ACC_CMAP = P.truncate(P.LANDSCAPE_CMAP.reversed(), 0.0, 0.92, name="acc")
# the floor is chance (0.5 = draws from the pooled marginal): anything below it is just
# uninformative, so the heatmaps clip there and the colorbars read "<= 0.5".
ACC_VMIN, ACC_VMAX = 0.5, 1.0
# reference rows that bracket every motif in the real-cell context eval
REFERENCES = ["oracle", "marg"]

# --- structural colours -------------------------------------------------------
INK = P.INK
MUTED = P.MUTED

# --- type sizes (nothing below 6) ---------------------------------------------
FS_BOX = 9.0        # titled group box over rows a-c / d-f
FS_TITLE = 8.0
FS_LABEL = 7.5
FS_TICK = 6.5
FS_SUB = 6.0
FS_LEGEND = 7.5

# --- assembly geometry (inches; final page must fit A4, max width 6.5) --------
PT_PER_IN = 72.0
PAGE_W_IN = 6.5
ROW_GAP_IN = 0.06

# recon-row layout (shared so rows a and b align column-for-column)
RECON_LEFT = 0.085
DYN_WIDTH, METHOD_WIDTH = 1.5, 1.4

# titled group boxes: rows a-c (non-least-action) and rows d-f (context-dependent).
# the rows are rendered at full page width and scaled to fit inside the border.
BOX_TITLES = ("Learning non-least-action dynamics", "Learning context-dependent dynamics")
PAGE_PAD_IN = 0.025      # breathing room around the whole page so the box stroke isn't clipped
BOX_INSET_IN = 0.025     # border stroke inset from the page edge
BOX_PAD_X_IN = 0.045     # border margin left/right of the rows
BOX_PAD_TOP_IN = 0.10    # border margin above the first row (title breaks the border here)
BOX_PAD_BOT_IN = 0.05    # border margin below the last row
BOX_GAP_IN = 0.16        # gap between the two boxes


def save_panel(fig, out_dir, name, size):
    """write one panel svg and return the (path, size) pair assemble consumes."""
    import matplotlib.pyplot as plt
    out = os.path.join(out_dir, name)
    fig.savefig(out)
    plt.close(fig)
    return out, size


def thin_yaxis(ax):
    """at most 4 y ticks, formatted without trailing zeros."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def methods_present(available, order=None):
    """canonical-order subset of ``available`` (columns / a method column / any iterable).

    lets a panel draw whatever the cache actually holds, so adding a baseline upstream
    does not need a plot-side edit and a cache predating one still renders.
    """
    have = set(available)
    return [m for m in (order or METHODS + ABLATIONS) if m in have]


def apply_rc():
    """publication rcParams: Latin Modern Sans, mathtext (cm), vector-path text."""
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": FS_LABEL,
        "axes.linewidth": 0.6,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.frameon": False,
        "legend.fontsize": FS_LEGEND,
        "lines.solid_capstyle": "round",
    })
