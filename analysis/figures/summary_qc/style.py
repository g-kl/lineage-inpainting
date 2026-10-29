"""shared config for the summary/QC figure (fig 3): paths, palette, geometry.

three conceptual rows:
  row 1  cell-type transition map (built from the model coupling)
  row 2  six coupling-quality metrics, lineage inpainting vs two GENOT baselines
  row 3  context scaling of subtree cycle consistency

compute is split from plotting: the compute_*.py scripts write small caches into
numerical_results/, the plotting scripts read only those caches so cosmetic
tweaks never touch the heavy model/dataset loads.
"""

import os

import matplotlib as mpl

from utils import palettes as P
from utils.figure_labels import register_fonts, FONT_FAMILY

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

register_fonts()   # shared Latin Modern Sans (utils/fonts/)

# --- the lineage-inpainting model (an unshared bridge-flow splitter) ----------
EXPERIMENT_DIR = os.path.join(
    base_dir, "outputs",
    "mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09")
STEP = 20000
MODEL_ASSIGNMENTS = os.path.join(
    EXPERIMENT_DIR,
    f"assignments_nn_{os.path.basename(EXPERIMENT_DIR)}_step{STEP}_adj_n8_splitter.pkl")

# --- GENOT baselines (saved flow model + its derived couplings) ---------------
BASELINE_DIR = os.path.join(base_dir, "outputs", "e_baselines")
GENOT_SINKHORN_PKL = os.path.join(BASELINE_DIR, "genot_w2_unified_model.pkl")    # entropic / W2
GENOT_CBFGW_PKL = os.path.join(BASELINE_DIR, "genot_fgw_unified_model.pkl")      # fused Gromov-W

# row 3 (context scaling) caches straight into numerical_results/context_scaling.csv
# via compute_context_scaling.py.

# --- the three methods, in display order --------------------------------------
# keys are the `method` values written into the caches.
METHODS = ["model", "genot_sinkhorn", "genot_cbfgw"]
METHOD_LABELS = {
    "model": "lineage inpainting",
    "genot_sinkhorn": "GENOT-Sinkhorn",
    "genot_cbfgw": "GENOT-cbFGW",
}
GENOT_PKLS = {
    "genot_sinkhorn": GENOT_SINKHORN_PKL,
    "genot_cbfgw": GENOT_CBFGW_PKL,
}

NR = os.path.join(os.path.dirname(__file__), "numerical_results")
OUT = os.path.join(os.path.dirname(__file__), 'imgs')
os.makedirs(NR, exist_ok=True)

# --- distortion sampling (rows 2 panels: clonal + marginal) -------------------
# clonal/marginal distortion are scored on a RANDOM SUBSET of subtrees so the
# three methods are all scored on identical leaf sets.
N_SUBTREE_LEAVES = 400      # sampled (leaf, ancestor) subtrees for clonal distortion
MIN_CLONE_LEAVES = 9        # clones must have > 8 leaves: on tiny clones the
                            # U-statistic is very high variance and often negative
CLONAL_OFFSET = 1.0         # ancestor sits leaf_time - offset back in time
CLONAL_TIME_TOL = 0.5       # tolerance matching a node to that ancestor time
N_GENOT_SAMPLES = 5         # stochastic forward samples per inferred GENOT ancestor
N_MARGINAL_CELLS = 400      # cells sampled per adjacent-timepoint pair (marginal)
MARGINAL_TIME_TOL = 0.25
SPLIT_STEPS = 8              # flow-rollout steps for the splitter cycle; the same
                             # n as the coupling (MODEL_ASSIGNMENTS, n = 8), so the
                             # distortion and the coupling roll the splitter identically
MAX_NODES = 256
BATCH_SIZE = 32
SEED = 212121

# --- row 1 transition-map params ----------------------------------------------
# a (type, time) edge is kept when its share of the target's incoming cells plus its
# share of the source's outgoing cells is >= MIN_COMPOSITION (and >= 5 cells), over
# every transition pair: no pmi cut (quantile 0) and no subsample. using every pair
# keeps the map deterministic, and the sum keeps edges moderate on both sides, like
# axial progenitor -> psm / spinal cord.
PMI_QUANTILE = 0.0
MIN_COMPOSITION = 0.2
COMPOSITION = "sum"
N_TREES = 512
# annotation level per quantized time. E7.5 is at germ layer: its internal nodes snap to
# E8.0 cells, so at lineage resolution most E7.5 nodes have no E7.5 cell behind them.
RESOLUTION_BY_TIME = {7.5: "germ_layer", 8.0: "lineage", 8.5: "cell_type",
                      9.0: "cell_type", 9.5: "cell_type", 10.0: "cell_type"}

## PALETTE ##
# method colours are shared across every figure (utils.palettes): the model
# (lineage inpainting) teal, the two GENOT baselines purple / green.
INK = P.INK
MUTED = P.MUTED   # grey for secondary text (axis subtitles, etc.)
METHOD_COLORS = {
    "model": P.MODEL,             # teal   -- lineage inpainting
    "genot_sinkhorn": P.GS,       # purple -- GENOT-Sinkhorn
    "genot_cbfgw": P.GF,          # green  -- GENOT-cbFGW
}
# germ-layer palette for the row-1 alluvial map (setting-specific; distinct from
# the shared method / lineage palettes)
TRANSITION_PALETTE = ["#d6543a", "#3c4ebf", "#e3a600", "#40ada6", "#edb1a3"]

# FONT_FAMILY imported from utils.figure_labels (shared across figures)

# row-1 transition map: transposed (time flows top->bottom over 6 levels, cell
# types spread across the WIDE axis with rotated labels) so we can keep the
# N_TOP_TYPES most abundant types. dropped types are rare terminal states; the
# kept set carries essentially all the flow mass. the dense type axis is the one
# place we drop below 10 pt -- FS_TYPE -- because ~90 columns demand it.
N_TOP_TYPES = 90
FS_TYPE = 5.5

# row-3 context scaling: all four panels share a metric (energy distance, cycle
# consistency) but the two boxes differ in inference DIRECTION, so a colour per
# direction: ancestral inference (c, d) reuses the model teal; descendant inference
# (e, f) gets a cool blue, distinct from the method trio (teal / purple / green).
# not read by the plot scripts, which draw both average panels in the model teal.
CONTEXT_DIR_COLORS = {
    "ancestral": P.MODEL,     # teal  -- leaf -> root -> leaf (= model teal)
    "forward": "#3D6FB4",     # blue  -- cousins -> v -> descendants
}

# row-3 ancestral context axis: context size = leaf_time - MRCA_time of the revealed
# clade, in 0.25-wide bins over [0, 1.5] (the upper edge is the pipeline's
# MAX_TIME_TO_MRCA), shared by both panels of the box. the scored set is the whole
# subtree, and a full reveal -- which scores far better -- can only happen once
# n_revealed reaches the pipeline's MIN_LEAVES, so a leaves-revealed axis would step at
# an arbitrary sampler floor; on the time axis full reveals ramp in smoothly with
# subtree depth. the cost: a fate's trend also picks up how fast its clades branch.
# many rows sit exactly at 0 (the focal leaf alone) and the band just above it is
# sparse (one step up moves the MRCA back by > 0.1), so narrower bins leave most fates
# without CELLTYPE_MIN_PER_POINT rows at every point.
CONTEXT_BIN_EDGES = [0.25 * i for i in range(7)]
# a per-type point is one fate's share of an average point, and both samplers give a
# fate only ~100 focals, so the far points thin out. a fate is drawn only if it has at
# least this many rows at EVERY point on the axis -- otherwise one row would draw as a
# trend.
CELLTYPE_MIN_PER_POINT = 3
# minimum leaves a fate needs (celltype_abundance.csv) to be drawn in the per-type
# panels. BOTH context samplers draw a fixed ~100 focals per fate, so row count is
# flat by construction and says nothing about abundance: a fate with 371 leaves in
# the whole tree set contributes as many rows as one with 112,036. this cut drops
# the fates whose trend rests on a handful of distinct clones, and with them the
# coarse early labels (forebrain / hindbrain / spinal cord) that only exist at the
# E8.0 leaf times and so would carry a timepoint effect into the per-type ranking.
# at 10,000 the grey background stays dense and the highlighted extremes are fates
# with tens of thousands of leaves rather than ones sitting on the cut.
CONTEXT_MIN_LEAVES = 10000
# fates left out of the per-type panels whatever their abundance: every cell type whose
# subtypes are all blood or extraembryonic lineage in final_type_info_v2_lower.csv.
# endothelial cell stays -- its yolk-sac subtype is ~2% of it. only allantois and
# extraembryonic mesoderm clear CONTEXT_MIN_LEAVES; the rest are listed for completeness.
CONTEXT_EXCLUDE_TYPES = {
    "allantois", "extraembryonic mesoderm", "allantoic endothelial progenitor",
    "amniotic ectoderm", "extraembryonic ectoderm", "visceral endoderm",
    "primitive erythrocyte", "hematopoietic progenitor", "macrophage", "megakaryocyte",
}

FS_LABEL = 10.0      # axis labels / tick labels
FS_ANNOT = 10.0      # in-panel annotations (value tags, legends)

## ASSEMBLY GEOMETRY (inches; final figure must fit an A4 page, max width 6.5) ##
PT_PER_IN = 72.0
PAGE_W_IN = 6.5
ROW_GAP_IN = 0.12       # vertical gap between assembled rows
ROW_GAP_AB_IN = 0.02    # tighter gap between the transition map (row 1) and metrics (row 2)
# row 3 = two "inference" boxes, each wrapping an (average | per type) panel pair.
BOX_PAIR_GAP_IN = 0.26   # gap between the two boxes
BOX_INNER_GAP_IN = 0.06  # gap between the two panels inside a box
BOX_PAD_X_IN = 0.05      # border margin left/right of the panels
BOX_PAD_BOT_IN = 0.05    # border margin below the panels (x-labels)
BOX_BORDER_OFF_IN = 0.16 # top border sits this far above the panel tops (title breaks it)
BOX_LEG_GAP_IN = 0.03    # gap between the boxes and the shared method legend


def apply_rc():
    """publication rcParams: Latin Modern Sans, text saved as vector paths."""
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "figure.dpi": 200,
        "font.size": 10,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": FS_LABEL,
        "ytick.labelsize": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "legend.fontsize": FS_ANNOT,
        "lines.solid_capstyle": "round",
    })
