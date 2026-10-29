"""paths, constants, palettes, geometry and the shared plot helpers."""
import os
import sys

import matplotlib
import numpy as np
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from utils import palettes as P  # noqa: E402
from utils.figure_labels import FONT_FAMILY, register_fonts  # noqa: E402

NR = os.path.join(HERE, "numerical_results")
IMGS = os.path.join(HERE, "imgs")
PANELS = os.path.join(IMGS, "panels")
for _d in (NR, PANELS):
    os.makedirs(_d, exist_ok=True)

# row 1: the CHD panel screens, at the E9.0 readout - the full knockout and its 50%
# knockdown counterpart - and the Tbx5 dose trees. every one of them is scored in the
# v2 lower annotation rows 2 and 3 use, so a subtype is named the same way across the
# whole sheet.
EXPERIMENT = os.path.join(ROOT, "outputs", "mouse_embryo_grid_bridge_unshared_vpsde_"
                          "idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09")
CHD_SCREEN = os.path.join(EXPERIMENT, "heart_projection_screen_v2_dose0_E9.0_R1")
CHD_SCREEN_50 = os.path.join(EXPERIMENT, "heart_projection_screen_v2_dose50_E9.0_R1")
DOSE_TREES = os.path.join(ROOT, "analysis", "results", "dose_response_broadsnap", "Tbx5_trees")
CHD_TIME, CHD_TOL, CHD_OFF_RATIO = 9.0, 0.25, 1.0
N_DOSE = 4              # most shifted subtypes the dose panel names
MIN_DOSE_LEAVES = 50    # readout trajectories a subtype needs a dose curve

TARGET = "Tbx5"
SEED = 0
LPM = "lateral plate mesoderm"
FORELIMB = "forelimb field"
HEART = "heart"          # the fate group, every cell type in data.HEART_TYPES

# the screen rows 2 and 3 are drawn under: the broad snap window, a node off-manifold
# at a projection-to-local distance ratio of 1.1, and `freeze` for the leaf that never
# formed. data.py asserts these against the pipeline's own constants.
SCREEN_KIND, OFF_RATIO, MODE = "broadsnap", 1.1, "freeze"
PRED, MEAS = "predicted", "measured"
HEART_SCOPE = "heart"

# row 2: curves over the control trees
SHIFT_STEP, SHIFT_T0, LEAF_TIME = 0.05, 7.0, 9.5
MIN_LEAVES = 100        # readout leaves a subtype needs a curve
N_SHIFT = 7             # subtypes the shift panel colours
SMOOTH_K = 1            # devmap neighbours a cell's Tbx5 is averaged over
TBX5_ON = 0.10          # smoothed Tbx5 a node counts as expressing above
# the split panel's two curves, over the same time axis as the shift panel
SPLIT_GRID = 400        # points the ecdfs are stepped over
SPLIT_X = (7.1, 9.0)    # the window they are drawn over, both events done inside it
SPLIT_AT = 0.5          # the share the gap between a fate's two curves is reported at

# row 2: the subtree illustration
MIN_BOTH = 3            # forelimb and heart leaves a subtree needs to be a candidate
MAX_LEAVES = 55         # leaves the drawn subtree may have once it is pruned
SUB_GRAD = 12           # points an edge's Tbx5 gradient is drawn at
SUB_LW = (0.4, 1.2)     # stroke of an edge under one leaf, and under all of them

# row 2: the posterior SHF program. the LDA is fit on the annotation's own anterior
# and posterior SHF labels, at the stages that carry both - no E9.5 cell is in the fit,
# since the posterior label runs out before it - and the axis is then read on the E9.5
# heart tube reference cells, which are row 3's, so a point on the panel is a point on
# its maps. POSTERIOR is only the pole the transferred axis is checked against
PROGRAM_LABELS = ("second heart field (anterior)",
                  "second heart field (posterior)")
PROGRAM_STAGES = ("E8.0", "E8.5", "E9.0")   # the stages carrying both labels
POSTERIOR = ("atria", "inflow tract", "atrioventricular canal")
MIN_DETECT = 0.05       # cells a gene has to be detected in to enter the LDA
N_PROGRAM = 3           # transcription factors named on each side of the program axis
TF_LIST = os.path.join(ROOT, "literature", "tf_list.txt")

# row 3: the maps. the heart tube is the first and second heart fields; the endocardium
# sits as its own island in this latent and is only ever a point on the scatter. the
# epicardium is out of the row, though the pipeline's draws still take it in.
HEART_TUBE = ("first heart field", "second heart field")
HEART_KEPT = HEART_TUBE + ("endocardial cell",)
N_UMAP_NEIGHBORS, UMAP_MIN_DIST = 30, 0.3
MIN_CELLS = 50          # reference cells a subtype needs to be scored
CLIM_Q = 98             # percentile the distortion limits are taken at, and its mirror
S_REF, S_CRI = 1.2, 1.2  # point size, reference and crispant, on the heart map
S_DIST = 1.8            # point size on the two distortion maps, reference only
S_SCATTER = 1.2         # point size on the scatter, one reference cell each

## PALETTE ##
INK, GREY, FAINT = "#2b2f33", "#9aa0a6", "#d7d9dc"
BACKDROP = "#d9dbde"        # the reference cloud behind the crispants
LINEAGE_COLORS = P.LINEAGE_COLORS       # shared lineage role, as in every other figure
# one colour per lateral plate subtype, shared by every panel that names a subtype.
# the nine heart subtypes take nine separated hues, since row 3 tells all of them apart
# on one map; the forelimb field and the juxtacardiac field are the two others the
# curves colour. the other lateral plate subtypes get a hue for whenever a curve or
# dose panel names one.
SUBTYPE = {
    "outflow tract": "#c9457b", "second heart field (anterior)": "#e07b39",
    "right ventricle": "#d4a72c", "left ventricle": "#5a9e5a",
    "atrioventricular canal": "#2f9c9c", "atria": "#4b5bc4",
    "inflow tract": "#9a56b8", "epicardium": "#7d7f3a",
    "endocardial cell": "#6d7a86",
    "juxtacardiac field": "#c1483a", "forelimb field": "#5c4033",
    "somatic mesoderm (anterior)": "#2e6f8e", "somatic mesoderm (posterior)": "#5b9bb8",
    "somatic mesoderm (tailbud)": "#9fc4d4", "hindlimb field": "#a894c4",
    "splanchnic mesoderm (cardiopulmonary)": "#4f9464",
    "splanchnic mesoderm (gastric)": "#7fb489",
    "splanchnic mesoderm (intestinal)": "#a8cba8",
    "splanchnic mesoderm (septum transversum)": "#35706b", "mesothelium": "#6b8e23",
}
# row 1: the gene scatters are one grey cloud with Tbx5 picked out. the dose panel's
# N_DOSE named subtypes take their SUBTYPE hue, so a colour stands for the same subtype
# there as on the row 2 curves; DOSE is the fallback for one outside the lateral plate.
GENE, GENE_HI = "#8a8f97", "#c1483a"
DOSE = ("#1f6f78", "#b8541f", "#5e4b8b", "#7d8b2f")
DOSE_WISP = "#dcdee1"
# row 2: Tbx5 along the subtree edges, and the same summed down the trajectory. two
# quantities in different units, so a map each, both off the leaf and subtype hues.
TBX5 = LinearSegmentedColormap.from_list(
    "tbx5", ["#eef1f2", "#8fc0c8", "#2e7f96", "#123049"])
TBX5_CUM = LinearSegmentedColormap.from_list(
    "tbx5_cum", ["#f0f2ec", "#a9c795", "#4a8b5f", "#16382a"])
# row 2: the subtree leaves, in the two fates left after the pruning. the forelimb
# field keeps the hue it has everywhere else on the sheet; the heart takes a hue of its
# own, since no single subtype colour stands for the whole of it.
LEAF_GROUP = {FORELIMB: SUBTYPE[FORELIMB], HEART: "#b3216d"}
ONSET = "#e8443a"       # the mark where a trajectory's Tbx5 crosses TBX5_ON
ONSET_MARK = ">"        # pointing along the trajectory, into the Tbx5+ stretch
# row 2: the two events the split panel's curves count, the second on the hue the
# subtree marks its onsets in, since it is the same crossing
# row 2: the split panel's four curves, keyed by the (fate, event) each is drawn from
# and carrying its (name, colour, dash). a colour each, so the key needs no line
# samples; the two Tbx5 curves are dashed as well, and each fate's pair shares a hue -
# the Tbx5 curve on the hue itself, the fate curve on a light tint of the same hue
SPLIT = {(FORELIMB, "t_split"): ("forelimb fate biased", "#7ebcdc", "-"),
         (HEART, "t_split"): ("heart fate biased", "#e378a7", "-"),
         (FORELIMB, "t_on"): ("Tbx5 on (forelimb)", "#1f5673", "--"),
         (HEART, "t_on"): ("Tbx5 on (heart)", "#8c1d4e", "--")}
# row 2: the two poles of the program axis, only ever a head over a column of gene
# names - the cloud under it is one colour, since no subtype is named before row 3
POLE_NAME = {True: "posterior", False: "anterior"}
# row 2: the program panel's cell density, drawn as filled contours under its points,
# with the level lines themselves drawn over the fill so the bands read as a density
# rather than as more scatter. one hue out of the page, light enough at every level that
# the points stay legible on it, so the fill carries where the cells are and the points
# only how many
PROGRAM = LinearSegmentedColormap.from_list(
    "program", ["#eef4f7", "#c3dbe6", "#8cb8ce", "#5591ae", "#2a6a8c"])
PROGRAM_LINE, PROGRAM_LW = "#1d5975", 0.35
# row 3: nn distortion, the only continuous quantity on the two lower rows. sequential
# gist_earth with its ends trimmed, since the raw map runs black to white and both ends
# read as absence on a white page.
DISTORTION = LinearSegmentedColormap.from_list(
    "distortion", matplotlib.colormaps["gist_earth"](np.linspace(0.05, 0.85, 256)))
DIST_LABEL = "$\\log\\,$nn distortion"

## TYPE ##  nothing but the row 1 gene labels is set below 6 pt
FS_TICK, FS_LABEL, FS_TAG, FS_KEY = 6.0, 6.5, 6.5, 6.0
FS_GENE = 5.0           # 21 genes named inside a 1.14 in panel
LABEL_PAD = 1.0         # points past the axes a gene label may reach

# every key is laid out in inches, so a swatch keeps its air whatever its marker and
# however many columns the key has. row 2's three keys are framed, since they sit in a
# band of their own under a panel that has other things below its axes
KEY_GAP = 0.032         # swatch edge to its label
KEY_COLGAP = 0.07       # between one column's label and the next column's swatch
KEY_PAD = 0.045         # between a frame and the block it holds, left and right
KEY_LINE = 0.10         # a line swatch's length, for a key over curves
KEY_ROUND = 0.03        # its corner radius
KEY_FRAME = "#c6cacf"   # its hairline

# the panel letters are utils.figure_labels' defaults - 16 pt bold in a 0.30 in box -
# as in every other figure on this page width. the box goes at the panel's top left
# corner, so nothing on a panel may sit inside LETTER_BOX of it. the two panels whose
# ylabel runs the full height of its axes reach into that corner, so their letter is
# lifted off it; LETTER_OFF is in inches off the corner, +y down, by panel name.
LETTER_BOX = 0.30
LETTER_OFF = {"r1_kd": (0.0, -0.045), "r3_scatter": (-0.05, -0.045)}

## GEOMETRY (inches) ##
PAGE_W = 6.5            # ceiling, asserted by the assembler
PAGE_PAD = 0.05         # white left around the whole sheet, so nothing a panel draws
                        # flush with its own edge lands on the sheet's. the panels are
                        # laid out at the sizes below and then scaled into what is left,
                        # so the sheet itself stays PAGE_W wide
ROW1_H, ROW2_H, ROW3_H = 1.56, 1.78, 1.56
R2_W = (1.58, 2.07, 1.45, 1.40)          # rows 2 and 3 share their outer columns
R3_W = (1.58, 1.45, 2.07, 1.40)

# row 1: four square panels
R1_W = (1.625, 1.625, 1.625, 1.625)
R1_BOX = (0.40, 0.30, 1.14, 1.14)

# row 2. every key on this row sits under its panel in a frame of its own, so the
# letters have the top left corners to themselves, nothing on a key can be read as part
# of a plot, and the key reads apart from the axis labels it shares that band with
CURVE_BOX = (0.34, 0.70, 1.18, 0.92)     # the shift panel
CURVE_KEY = (0.08, 0.02, 1.46, 0.42)     # its key, four rows under the axes
SPLIT_BOX = (0.52, 0.70, 0.86, 0.92)     # the split panel, on the shift panel's band
SPLIT_KEY = (0.52, 0.02, 0.86, 0.42)     # its key, four rows on the axes' own span
PROG_BOX = (0.44, 0.72, 0.90, 0.96)      # the program panel
PROG_GENE_Y = 0.35                       # the gene block's first baseline, under it
PROG_GENE_DY = 0.095                     # and its line spacing
S_PROGRAM = 1.0                          # point size on it, one reference cell each
PROG_ALPHA = 0.20                        # and its alpha, over the density under it
PROG_LEVELS = 6                          # density contours, few enough to read as bands
PROG_THRESH = 0.05                       # the density the lowest one is drawn at
SUB_KEY = (0.10, 0.02, 1.90, 0.19)       # the leaf key, one row under both subtrees
SUB_BOX = (0.07, 0.48, 0.70, 1.22)       # the left subtree; the right one is +SUB_DX
SUB_DX = 0.97
# upright, right of the left subtree and likewise offset. its two ticks are written
# at its ends rather than beside it, which is what leaves the pair of trees their
# width once each carries its own bar and that bar's label
SUB_CBAR = (0.84, 0.76, 0.035, 0.66)

# row 3. both maps carry their key underneath, so the map sits clear of the letter and
# runs the height of the panel. MAP_HEART and DIST_MAP have to keep one aspect, since
# panels 3 and 4 are drawn on the window this one cuts
MAP_ALL = (0.31, 0.60, 0.94, 0.94)       # the whole-embryo map
KEY_ALL = (0.01, 0.02, 1.56, 0.45)       # five rows of three under it
COUNT_ALL_Y = 0.53
MAP_HEART = (0.31, 0.58, 0.92, 0.92)     # the heart map
KEY_HEART = (0.08, 0.02, 1.34, 0.32)     # four rows of two under it
COUNT_Y = (0.50, 0.39)                   # the two count lines between map and key
DIST_MAP = (0.04, 0.34, 0.98, 0.98)      # the two hued maps; the right is +DIST_DX
DIST_DX = 1.03
DIST_CBAR = (0.585, 0.13, 0.90, 0.035)   # one bar, centred under both maps
DIST_TAG_Y = 1.36                        # the predicted / measured tag over each map
SCATTER_BOX = (0.38, 0.30, 0.94, 1.14)   # narrow enough that the last x tick,
# written under the right spine, stays inside the panel

WINDOW_BINS = 60        # histogram resolution a map window's density is read at
WINDOW_FRAC = 0.02      # share of the densest bin a bin has to hold to be in frame
WINDOW_PAD = 0.04       # of the larger extent, added on every side

SHORT = {"lateral plate mesoderm": "lateral plate", "extraembryonic mesoderm": "exe meso",
         "somatic mesoderm": "somatic",
         "extraembryonic ectoderm": "exe ecto", "extraembryonic endoderm": "exe endo",
         "primordial germ cell": "pgc", "intermediate mesoderm": "intermediate",
         "neural ectoderm": "neural ecto", "surface ectoderm": "surface ecto",
         "paraxial mesoderm": "paraxial",
         "second heart field (anterior)": "SHF (ant)",
         "atrioventricular canal": "AV canal", "endocardial cell": "endocardium",
         "juxtacardiac field": "juxtacardiac",
         "somatic mesoderm (anterior)": "somatic (ant)",
         "somatic mesoderm (posterior)": "somatic (post)",
         "somatic mesoderm (tailbud)": "somatic (tail)",
         "splanchnic mesoderm (cardiopulmonary)": "splanchnic (cp)",
         "splanchnic mesoderm (gastric)": "splanchnic (gast)",
         "splanchnic mesoderm (intestinal)": "splanchnic (int)",
         "splanchnic mesoderm (septum transversum)": "splanchnic (st)"}


def short(s):
    return SHORT.get(s, s)


def apply_rc():
    """latin modern sans, text saved as vector paths."""
    register_fonts()
    matplotlib.rcParams.update({
        "font.family": FONT_FAMILY, "mathtext.fontset": "cm",
        "axes.unicode_minus": False, "svg.fonttype": "path", "figure.dpi": 200,
        "font.size": FS_LABEL, "axes.linewidth": 0.5,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "legend.fontsize": FS_KEY, "legend.frameon": False})


def figure(w, h):
    """a figure of exactly (w, h) inches, plus a `box` that places axes in inches."""
    import matplotlib.pyplot as plt
    apply_rc()
    fig = plt.figure(figsize=(w, h))
    return fig, lambda x, y, bw, bh: fig.add_axes([x / w, y / h, bw / w, bh / h])


def save(fig, name, size):
    """write a panel svg and return (path, size) for the assembler."""
    import matplotlib.pyplot as plt
    p = os.path.join(PANELS, f"{name}.svg")
    fig.savefig(p, transparent=True)
    plt.close(fig)
    print(f"wrote {os.path.relpath(p, ROOT)}", flush=True)
    return p, size


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def window(x, y, box, main=False, bins=WINDOW_BINS, frac=WINDOW_FRAC):
    """(xlim, ylim) around the dense part of a cloud, padded out to `box`'s aspect.

    a bin holding at least `frac` of the densest bin is in frame; `main` keeps only the
    connected block of bins the densest one sits in.
    """
    from scipy import ndimage
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    m = h >= frac * h.max()
    if main:
        lab, _ = ndimage.label(m)
        m = lab == lab[np.unravel_index(np.argmax(h), h.shape)]
    xi, yi = np.nonzero(m)
    x0, x1 = xe[xi.min()], xe[xi.max() + 1]
    y0, y1 = ye[yi.min()], ye[yi.max() + 1]
    pad = WINDOW_PAD * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    want = box[2] / box[3]
    have = (x1 - x0) / (y1 - y0)
    if have < want:
        g = ((y1 - y0) * want - (x1 - x0)) / 2
        x0, x1 = x0 - g, x1 + g
    else:
        g = ((x1 - x0) / want - (y1 - y0)) / 2
        y0, y1 = y0 - g, y1 + g
    return (x0, x1), (y0, y1)


def key(ax, items, ncol=1, fs=FS_KEY, ms=2.4, gap=KEY_GAP, colgap=KEY_COLGAP,
        box=False, pad=KEY_PAD, line=0.0):
    """a swatch key in a bare axes, laid out in inches and filling column by column.

    an item is (label, colour) or (label, colour, marker). `line` draws every swatch as
    a sample of that length in inches instead, taking an item's third element as a
    linestyle, for a key over curves rather than over points. a column is as wide as the
    labels it holds and the block is centred in the axes, so a key spends its width on
    its labels rather than on even columns. `gap` is the swatch edge to text distance
    and `colgap` the air between columns, both in inches. the rows divide the axes'
    height, so a framed key wants a box a row taller than an unframed one; `box` draws
    that frame `pad` out from the block and over the axes' full height.
    """
    bare(ax)
    fig = ax.figure
    w_in = ax.get_position().width * fig.get_figwidth()
    h_in = ax.get_position().height * fig.get_figheight()
    ax.set_xlim(0, w_in)
    ax.set_ylim(h_in, 0)

    rows = int(np.ceil(len(items) / ncol))
    lab_w = _boxes(ax, [i[0] for i in items], fs)[:, 0] / fig.dpi
    half = line / 2 if line else ms / 144.0             # swatch half width, inches
    cols = [lab_w[c * rows:(c + 1) * rows] for c in range(ncol)]
    cw = [2 * half + gap + (c.max() if len(c) else 0.0) for c in cols]
    row_h = h_in / rows
    x0 = (w_in - sum(cw) - colgap * (ncol - 1)) / 2

    for i, item in enumerate(items):
        c, r = divmod(i, rows)
        x = x0 + sum(cw[:c]) + colgap * c
        y = (r + 0.5) * row_h
        if line:
            ax.plot([x, x + 2 * half], [y, y], color=item[1], lw=0.9,
                    ls=item[2] if len(item) > 2 else "-", clip_on=False)
        else:
            ax.plot([x + half], [y], marker=item[2] if len(item) > 2 else "o", ms=ms,
                    color=item[1], mec=item[1], mew=0.8, clip_on=False)
        ax.text(x + 2 * half + gap, y, item[0], va="center", ha="left", fontsize=fs,
                color=INK, clip_on=False)

    if box:
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch(
            (x0 - pad, 0.0), sum(cw) + colgap * (ncol - 1) + 2 * pad, h_in,
            boxstyle=f"round,pad=0,rounding_size={KEY_ROUND}", fc="none",
            ec=KEY_FRAME, lw=0.4, clip_on=False, zorder=0))


def caption(fig, w, h, text, y, x=0.03, color=INK):
    """a line of text under a panel's map, placed in inches."""
    fig.text(x / w, y / h, text, ha="left", va="baseline", fontsize=FS_KEY, color=color)


def hcbar(fig, ax, mappable, label, ticks=None, ticklabels=None):
    """a flat horizontal colourbar under a map, its label above the bar."""
    cb = fig.colorbar(mappable, cax=ax, orientation="horizontal")
    cb.ax.set_title(label, fontsize=FS_KEY, color=INK, pad=2.0)
    cb.ax.tick_params(labelsize=FS_KEY, width=0.4, length=1.8, pad=1.0)
    cb.outline.set_visible(False)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.ax.set_xticklabels(ticklabels, fontsize=FS_KEY)
    return cb


def _boxes(ax, labels, fs):
    """(width, height) of every label in pixels, measured off the renderer."""
    r = ax.figure.canvas.get_renderer()
    out = []
    for t in labels:
        a = ax.text(0, 0, t, fontsize=fs)
        e = a.get_window_extent(r)
        out.append((e.width, e.height))
        a.remove()
    return np.asarray(out, float)


def _overlap(box, others):
    """area each candidate box shares with each of `others`, in pixels squared."""
    dx = (np.minimum(box[:, None, 2], others[None, :, 2])
          - np.maximum(box[:, None, 0], others[None, :, 0]))
    dy = (np.minimum(box[:, None, 3], others[None, :, 3])
          - np.maximum(box[:, None, 1], others[None, :, 1]))
    return np.clip(dx, 0, None) * np.clip(dy, 0, None)


def _cut(a, b, boxes):
    """whether each of the segments a -> b cuts each of `boxes`, as (box, segment)."""
    d = np.where(np.abs(b - a) < 1e-9, 1e-9, b - a)
    t0 = (boxes[:, None, :2] - a[None]) / d[None]
    t1 = (boxes[:, None, 2:] - a[None]) / d[None]
    lo = np.maximum(np.minimum(t0, t1).max(2), 0.0)
    hi = np.minimum(np.maximum(t0, t1).min(2), 1.0)
    return lo < hi


def repel(ax, xs, ys, labels, fs=FS_GENE, color=INK, ms=2.4, gap=1.5, step=1.8,
          pad=0.8, rows=15, cols=7, rings=(0.0, 3.0, 7.0, 12.0, 18.0), sweeps=14):
    """place every label near its own point and sweep the collisions apart.

    a candidate sits beside the point at one of `rows` heights or over and under it at
    one of `cols` offsets, on one of the rings out. its cost is the area it covers of
    the other labels, of the markers and of the outside of the axes, plus a crossing
    for every hairline it would cut or be cut by, and a pull back towards its point.
    a sweep re-places each label against the others as they stand, so a crowded corner
    comes apart over the sweeps rather than stacking in one greedy pass. a label that
    ended up off its point gets a hairline back to it, drawn from the edge of the text.
    every length but the boxes is in points.
    """
    fig = ax.figure
    fig.canvas.draw()
    ppp = fig.dpi / 72.0
    frame = np.array(ax.get_window_extent(fig.canvas.get_renderer()).extents)
    frame += np.array([-1, -1, 1, 1]) * LABEL_PAD * ppp
    p = ax.transData.transform(np.column_stack([xs, ys]))
    wh = _boxes(ax, labels, fs) + 2 * pad * ppp
    marks = np.column_stack([p - ms * ppp, p + ms * ppp])
    ref = float(np.median(wh[:, 0] * wh[:, 1]))

    def ladder(n, r):
        """(sign, offset along the free axis, reach, ring) over `n` rungs and `r`."""
        a, b, c = (v.ravel() for v in np.meshgrid([1.0, -1.0], np.arange(n),
                                                  np.asarray(rings), indexing="ij"))
        return a, ((b + 1) // 2) * step * np.where(b % 2, 1.0, -1.0), gap + c, c
    beside, over = ladder(rows, rings), ladder(cols, rings)
    ring = np.concatenate([beside[3], over[3]])

    lab = np.zeros((len(labels), 2))                 # centre offset off the point
    for _ in range(sweeps):
        for i in range(len(labels)):
            off = np.column_stack([
                np.concatenate([beside[0] * (beside[2] * ppp + wh[i, 0] / 2),
                                over[1] * ppp]),
                np.concatenate([beside[1] * ppp,
                                over[0] * (over[2] * ppp + wh[i, 1] / 2)])])
            c = p[i] + off
            box = np.column_stack([c - wh[i] / 2, c + wh[i] / 2])
            rest = np.delete(np.column_stack([p + lab - wh / 2, p + lab + wh / 2]),
                             i, axis=0)
            out = ((np.clip(frame[0] - box[:, 0], 0, None)
                    + np.clip(box[:, 2] - frame[2], 0, None)) * wh[i, 1]
                   + (np.clip(frame[1] - box[:, 1], 0, None)
                      + np.clip(box[:, 3] - frame[3], 0, None)) * wh[i, 0])
            cost = ((12.0 * _overlap(box, rest).sum(1)
                     + 2.0 * _overlap(box, np.delete(marks, i, axis=0)).sum(1)
                     + 16.0 * out) / ref
                    + 1.5 * (_cut(np.repeat(p[[i]], len(c), 0), c, rest).sum(0)
                             + _cut(np.delete(p, i, 0), np.delete(p + lab, i, 0),
                                    box).sum(1))
                    + 0.05 * np.linalg.norm(off, axis=1) / ppp + 0.06 * ring)
            lab[i] = off[int(np.argmin(cost))]

    for i, (t, c) in enumerate(zip(labels, np.broadcast_to(color, len(labels)))):
        lead = np.hypot(*(np.abs(lab[i]) - wh[i] / 2).clip(0, None)) > gap * ppp
        ax.annotate(t, (xs[i], ys[i]), textcoords="offset points", zorder=4,
                    xytext=tuple(lab[i] / ppp), ha="center", va="center",
                    fontsize=fs, color=c, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.3, color=GREY, shrinkA=0.0,
                                    shrinkB=1.0) if lead else None)
