"""shared config for the grid-summary figures: paths, axis vocabulary, palette, type
sizes, geometry. see concept.md.

the grid varies the sampling model (bridge), the imputing model (pool) and the
pool<->split pairing, plus an aligner-feature ablation on the main-text run; every figure
isolates one axis and reports the SAMPLING side: the splitter used as the coupler,
which is the main-text figure's coupling. the canonical checkpoint, the coupling
integration steps and every metric definition come from the main-text figure
(analysis/figures/summary_qc), so a grid number and a figure number are the same
quantity. compute is split from plotting: compute_grid.py distils the per-run eval
artefacts into small caches under numerical_results/<step>/, and plot_*.py +
assemble.py read only those and write the master renders into figures/<step>/ (the
panel svgs go to imgs/<step>/, gitignored).
"""
import os

import matplotlib as mpl
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

from analysis.figures.summary_qc import style as SQ
from utils import palettes as P
from utils.figure_labels import FONT_FAMILY, register_fonts

register_fonts()   # shared Latin Modern Sans (utils/fonts/)

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUTPUTS = os.path.join(ROOT, "outputs")
GRID_PREFIX = "mouse_embryo_grid_bridge_unshared_"

# --- checkpoints and coupling, from the main-text figure -------------------------
# the figure scores the main-text run at summary_qc.style.STEP with the splitter
# coupling the trees at 8 integration steps; the grid uses the same checkpoint and the
# same n for every bridge (the one-step splitters have no integration, n = 1). 5k is
# rendered alongside as a companion.
STEP = SQ.STEP
STEPS = (5000, 20000)
HERO_CFG = os.path.basename(SQ.EXPERIMENT_DIR)     # the run the figure reports


def coupling_n(bridge):
    """integration steps of the sampling-side coupling for a sampling model."""
    return 1 if bridge == "one_step" else 8


def step_label(step):
    return f"{step // 1000}k"


def nr_dir(step):
    d = os.path.join(HERE, "numerical_results", step_label(step))
    os.makedirs(d, exist_ok=True)
    return d


def imgs_dir(step):
    """scratch for the panel svgs (gitignored, like every figure's imgs/)."""
    d = os.path.join(HERE, "imgs", step_label(step))
    os.makedirs(d, exist_ok=True)
    return d


def figures_dir(step):
    """the master renders, committed: the grid summary is the one figure whose
    renders are committed (under figures/)."""
    d = os.path.join(HERE, "figures", step_label(step))
    os.makedirs(d, exist_ok=True)
    return d


# --- axis vocabulary (directory tag -> display name), in canonical order --------
BRIDGES = ["one_step", "cfm", "vpsde"]
# the "pool" axis is the residual frame of the WHOLE pool-split pair: a Tree Tx run
# drops the additive baseline from both the imputing and the sampling model. plain
# first, the main-text run's (+ baseline) last. only the additive-baseline strip
# shows both: every other figure is the main pool alone.
POOLS = ["nondelta", "dtfm"]
MAIN_POOL = "dtfm"
PAIRINGS = ["id", "id_sg", "nn", "nn_sg", "idnn_leaf"]
PAIRINGS_SHOWN = ["id", "nn", "idnn_leaf"]   # the stop-grad runs appear in the stop-gradient strip only


def in_heatmap(bridge, pool, pairing, aligner):
    """the similarity heatmap's runs: every run some other figure shows (the main
    pool with every pairing, the plain Tree Tx runs of the additive-baseline strip,
    the aligner ablation) plus the plain Tree Tx stop-gradient runs, which appear
    nowhere else."""
    return aligner != "full" or pool in POOLS
# `full` = slice_marginal_ca_path (marginal + path + common-ancestor features), the
# aligner every grid run trains with; marg / marg_path / marg_ca are the feature
# ablation on the main-text run.
ALIGNERS = ["marg", "marg_path", "marg_ca", "full"]
ORDER = {"bridge": BRIDGES, "pool": POOLS, "pairing": PAIRINGS, "aligner": ALIGNERS,
         "pair_family": ["id", "nn", "idnn"], "grad": ["sg", "flow"]}
DISPLAY = {
    "one_step": "One-step", "cfm": "CFM", "vpsde": "VP-SDE",
    "dtfm": "Tree Tx + baseline", "nondelta": "Tree Tx",
    # the pairing: what the splitter trains on -- the imputer's own nodes, their
    # nearest real cells, or the two on alternate steps
    "id": "imputer", "id_sg": "imputer (sg)", "nn": "nearest neighbor",
    "nn_sg": "nearest neighbor (sg)", "idnn_leaf": "alternating",
    "sg": "stop-grad", "flow": "gradient", "idnn": "alternating",
    "marg": "Marginal only", "marg_path": "+ Path", "marg_ca": "+ CA",
    "full": "+ Path + CA (ours)",
}


def disp(x):
    return DISPLAY.get(x, str(x))


# the main-text run: the run the main-text figure reports (compute_grid checks
# HERO_CFG has exactly these axes)
HERO = {"bridge": "vpsde", "pool": "dtfm", "pairing": "idnn_leaf", "aligner": "full"}

# --- metric columns of metrics_wide.csv ------------------------------------------
# the main-text metrics row's six (analysis/figures/summary_qc/plot_metrics, its
# names) plus three of the grid's own, in two groups that an overbar labels with the
# main figure's words: ANCESTRAL INFERENCE -- the marginal distortion (energy
# distance, the splitter's pooled-back ancestors of sampled cells against the real
# cells observed at the earlier time), its label analogue the marginal cell-type TV
# (the same pooled-back states snapped to real cells, cell-type composition against
# the cells observed then) and the four assignment metrics of the coupling;
# DESCENDANT INFERENCE -- the clonal distortion (a clone's
# leaves pooled to the root and split back, against the real leaves), the
# descendant distortion (the same subtrees split from the root's COUPLED real cell
# instead of the pooled-back root, against the real leaves) and the descendant
# cell-type TV (the generated leaves snapped to real cells, composition against
# the real leaves'). the four assignment metrics are scored on the sampling-side
# coupling, and again on the imputing-side coupling (the pool model applied once)
# as `cpl_*` for the sampler-vs-imputer figure.
ASSIGNMENT_METRICS = ["osc", "linkage", "germ", "hox"]
# the metrics with an imputing-side counterpart (`cpl_*`): the marginal pair with
# the pool model doing the pool-back, the assignment metrics on its coupling
PAIRED_METRICS = ["marginal", "label_tv", *ASSIGNMENT_METRICS]
METRIC_GROUPS = [
    ("ancestral inference", ["marginal", "label_tv", *ASSIGNMENT_METRICS]),
    ("descendant inference", ["clonal", "descendant", "descendant_tv"]),
]
METRICS = [m for _, ms in METRIC_GROUPS for m in ms]
IMPUTING_METRICS = [f"cpl_{m}" for m in PAIRED_METRICS]
METRIC_LABELS = {
    "clonal": "clonal\ndistortion", "marginal": "marginal\ndistortion",
    "osc": "lineage\noscillation", "linkage": "linkage\ndistortion",
    "germ": "germ-layer\ncrossing", "hox": "Hox-code\ndistortion",
    "label_tv": "marginal\ncell-type TV",
    # the descendant pair repeats the ancestral pair's titles: the group overbar
    # says which side (see concept.md on how far "marginal" stretches)
    "descendant": "marginal\ndistortion",
    "descendant_tv": "marginal\ncell-type TV",
}


def metric_label(col):
    return METRIC_LABELS[col[4:] if col.startswith("cpl_") else col]


def group_bars(fig, axes, metrics, *, y_in, h_in):
    """one overbar per metric group above the column titles: an ink line spanning
    the group's columns, ``y_in`` below the top of the figure, its name centred
    above. ``axes`` are the metric columns in ``metrics`` order."""
    from matplotlib.lines import Line2D
    pos = {m: ax.get_position() for m, ax in zip(metrics, axes)}
    y = 1 - y_in / h_in
    for name, members in METRIC_GROUPS:
        members = [m for m in members if m in pos]
        if not members:
            continue
        x0, x1 = min(pos[m].x0 for m in members), max(pos[m].x1 for m in members)
        fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=INK, lw=0.7))
        fig.text((x0 + x1) / 2, y + 0.025 / h_in, name, ha="center", va="bottom",
                 fontsize=FS_LABEL, color=INK)


# --- palette ---------------------------------------------------------------------
INK, MUTED, GREY = P.INK, P.MUTED, P.GREY
MODEL = P.MODEL


def tint(color, f):
    """``color`` mixed with white by fraction ``f`` (0 = the colour, 1 = white)."""
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


# levels of an ablation axis: the main-text run's setting in the model teal, the other
# learned variant a tint of it (a tint of the model teal, as in
# utils.palettes.ABLATION_COLORS). the baseline level has no colour: an open ink dot
# (``dot(None)``).
VARIANT_COLORS = {"vpsde": MODEL, "cfm": tint(MODEL, 0.45), "dtfm": MODEL, "flow": MODEL}
# the aligner-feature ablation: a ramp into the model teal as features are added
ALIGNER_COLORS = {"marg": tint(MODEL, 0.55), "marg_path": tint(MODEL, 0.4),
                  "marg_ca": tint(MODEL, 0.25), "full": MODEL}
SIM_CMAP = P.SIM_CMAP          # transition-graph cosine similarity (sequential)

# --- markers: one dot grammar for the strips and the scatter ----------------------
DOT_MS = 3.6


def dot(color=None):
    """an open ink dot for a baseline level, a filled dot in the level's colour
    otherwise (white edge, so overlapping dots stay separable)."""
    if color is None:
        return dict(marker="o", ms=DOT_MS, mfc="white", mec=INK, mew=0.8)
    return dict(marker="o", ms=DOT_MS, mfc=color, mec="white", mew=0.4)


# --- metric axes -------------------------------------------------------------------
# every metric is lower = better. the strips draw each column log-spaced from the
# largest 1-2-5 value at or below its smallest value (a round floor, close to the
# data), with gridlines at the 1-2-5 positions and minor ticks between them so the
# log spacing is visible (a column too narrow for three 1-2-5 ticks gets a line at
# every integer multiple instead). tick labels sit centred under their ticks, the floor's
# starting at the spine like an origin label; a column labels as many 1-2-5 ticks
# as fit (measured), else the floor and the highest tick whose label fits, else the
# floor alone. the scatter and the aligner lollipops are linear from 0 in the
# metric's units, as the main-text metrics row.
LEFT_CAPTION = "left = better"
LOLLI_ROW_H_IN = 0.27  # one variant row in a lollipop strip


def _fmt(t):
    return f"{t:.3g}"


def _labels_fit(ax, ticks, ha_first, overhang_pt, gap_pt=3.0):
    """whether these x tick labels fit: centred under their ticks (the first one
    left-aligned to the spine when ``ha_first`` is "left"), clear of each other by
    ``gap_pt`` and inside the panel but for ``overhang_pt`` past either spine.
    measured with the figure's renderer."""
    fig = ax.figure
    px = fig.dpi / 72.0
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(family=mpl.rcParams["font.family"], size=FS_TICK)
    w = [renderer.get_text_width_height_descent(_fmt(t), prop, ismath=False)[0] for t in ticks]
    x = ax.transData.transform([(t, 0.0) for t in ticks])[:, 0]
    ext = [(xi, xi + wi) if i == 0 and ha_first == "left" else (xi - wi / 2, xi + wi / 2)
           for i, (xi, wi) in enumerate(zip(x, w))]
    x0, x1 = ax.transAxes.transform([(0, 0), (1, 0)])[:, 0]
    if ext[0][0] < x0 - overhang_pt * px or ext[-1][1] > x1 + overhang_pt * px:
        return False
    return all(ext[i + 1][0] - ext[i][1] >= gap_pt * px for i in range(len(ext) - 1))


def scale_axis(ax, which, values, *, log=False, pad=0.08, overhang_pt=0.0):
    """set one metric axis (``which`` = "x" | "y") from the values it shows. linear:
    from 0 (or below, if a value is negative) past the largest value, ticks at 0 and
    at the largest one-digit value whose label fits, else that value alone. log:
    from the 1-2-5 floor below the smallest positive value, 1-2-5 ticks with
    gridlines and minor ticks at 3 4 6 7 8 9 (a column narrower than three 1-2-5
    ticks gets a tick and a line at every integer multiple instead), labels on the
    ticks that fit. ``overhang_pt`` is how far a label may reach past a spine (the
    gap to the next panel). returns the limits."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    hi = float(v.max()) if v.size else 1.0
    axis = ax.xaxis if which == "x" else ax.yaxis
    pos = v[v > 0]
    log = log and pos.size > 0
    if log:
        # the floor comes from the positive values; a value <= 0 (the U-statistic
        # energy distance of a tiny cloud) is drawn at the floor by the caller
        e = int(np.floor(np.log10(pos.min())))
        floor = max(m * 10.0 ** e for m in (1, 2, 5) if m * 10.0 ** e <= pos.min())
        top = 10.0 ** (np.log10(hi) + max(np.log10(hi / floor) * pad, 0.03))
        lim = (floor, top)
        ks = range(e, int(np.ceil(np.log10(top))) + 1)
        ticks = [m * 10.0 ** k for k in ks for m in (1, 2, 5) if floor <= m * 10.0 ** k <= top]
        minor = (3, 4, 6, 7, 8, 9)
        if len(ticks) < 3:      # a narrow column: a line at every integer multiple
            ticks = [m * 10.0 ** k for k in ks for m in range(1, 10) if floor <= m * 10.0 ** k <= top]
            minor = ()
        decades = [t for t in ticks if np.isclose(np.log10(t) % 1, 0)]
        # labels: every tick; three or more decades; the floor and the highest tick
        # that fits; the floor alone
        sets = ([ticks] + ([decades] if len(decades) > 2 else [])
                + [[floor, t] for t in reversed(ticks[1:])] + [[floor]])
        ha_first = "left"
    else:
        lo = min(0.0, float(v.min())) if v.size else 0.0
        span = (hi - lo) or 1.0
        lim = (lo - span * pad if lo < 0 else 0.0, hi + span * pad)
        e = int(np.floor(np.log10(hi))) if hi > 0 else 0
        cands = [m * 10.0 ** k for k in (e - 1, e) for m in range(1, 10) if 0 < m * 10.0 ** k <= hi]
        sets = [[0.0, c] for c in reversed(cands)] + [[cands[-1] if cands else hi]]
        ha_first = "center"
    getattr(ax, f"set_{which}scale")("log" if log else "linear")
    getattr(ax, f"set_{which}lim")(*lim)
    labelled = next(t for t in sets
                    if len(t) == 1 or which == "y" or _labels_fit(ax, t, ha_first, overhang_pt))
    if log:
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(
            lambda t, _: _fmt(t) if any(np.isclose(t, l) for l in labelled) else ""))
        axis.set_minor_locator(LogLocator(base=10, subs=minor, numticks=50) if minor
                               else NullLocator())
        axis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis=which, which="major", color=GREY, lw=0.4)
        ax.set_axisbelow(True)
        if which == "x":
            axis.get_majorticklabels()[0].set_ha("left")
    else:
        axis.set_major_locator(FixedLocator(labelled))
        axis.set_major_formatter(FuncFormatter(lambda t, _: _fmt(t)))
    return lim


# --- type sizes (nothing below 6) -----------------------------------------------
FS_TITLE = 8.0      # figure title
FS_LABEL = 7.0      # row labels, legend, group brackets
FS_COL = 7.0        # metric column titles
FS_TICK = 6.5
FS_SUB = 6.0        # value annotations

# --- geometry (inches; every figure is one 6.5 in page-width row) ------------------
PT_PER_IN = 72.0
PAGE_W_IN = 6.5
PAGE_PAD_IN = 0.02
ROW_H_IN = 0.17      # one config row in an ablation strip
# the head of a strip, from the top: title (0.03), legend (0.20), group overbar
# label + line (0.50), two-line column titles, axes (HEAD_IN); each band gets the
# same air
HEAD_IN = 0.92       # title + legend + group overbars + column titles above the strips
GROUP_Y_IN = 0.52    # the group overbars' line, below the top of the figure
LEGEND_Y_IN = 0.17   # the legend row, below the top of the figure
FOOT_IN = 0.36       # x ticks + the axis caption below
BRACKET_IN = 0.16    # one level of row-group brackets left of the tick labels


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
        "axes.titlesize": FS_COL,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.minor.size": 1.2,
        "xtick.minor.width": 0.5,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_LABEL,
        "legend.frameon": False,
        "legend.fontsize": FS_LABEL,
        "lines.solid_capstyle": "round",
    })


def save_panel(fig, out_dir, name, size):
    """write one panel svg and return the (path, size) pair assemble consumes."""
    import matplotlib.pyplot as plt
    out = os.path.join(out_dir, name)
    fig.savefig(out)
    plt.close(fig)
    return out, size


def sort_rows(df, axes):
    """rows in the canonical order of ``axes`` (ORDER), first axis outermost."""
    keys = [f"_{a}" for a in axes]
    tmp = df.assign(**{k: df[a].map({v: i for i, v in enumerate(ORDER[a])})
                       for a, k in zip(axes, keys)})
    return tmp.sort_values(keys).drop(columns=keys).reset_index(drop=True)


def title(fig, text, *, left_in, w_in, h_in):
    """the figure's title, top-left, in figure inches. (the checkpoint and the
    coupling are the figure directory's, not written on the figure.)"""
    fig.text(left_in / w_in, 1 - 0.03 / h_in, text, ha="left", va="top",
             fontsize=FS_TITLE, color=INK)
