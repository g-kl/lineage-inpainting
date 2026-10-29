"""shared config for the perturbation-map figure: paths, palette, geometry."""

import os

import matplotlib as mpl

from utils import palettes as P

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXPERIMENT_DIR = os.path.join(
    base_dir, "outputs",
    "mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09")
DATA_DIR = os.path.join(base_dir, "..", "data", "embryos")
STATES_H5AD = os.path.join(DATA_DIR, "qiu24_integrated_hvgs_and_scvi.h5ad")
UMAP_CSV = os.path.join(DATA_DIR, "final_umap.csv")

# the ring: the genome-wide HVG knockout screen, read out at E9.0 (one point per
# knockout).  NB the screen run double-nested its output dir, so the genes/control
# live one level deeper than the tag dir.
RING_SCREEN = os.path.join(EXPERIMENT_DIR, "hvg_projection_screen_v2_dose0_E9.0_R1")
RING_TARGET_TIME = 9.0         # E9.0 readout for the perturbation map
RING_TOL = 0.1                 # |leaf time - target| window for terminal leaves
RING_MIN_LEAVES = 200          # skip a knockout with fewer landed leaves than this
RING_ISOLATE_SPECIFIC = False  # project out the shared consensus response before UMAP (--specific to enable)
RING_UMAP_NEIGHBORS = 15
RING_UMAP_MIN_DIST = 0.1
RING_UMAP_METRIC = "euclidean"
TF_LIST = os.path.join(base_dir, "literature", "tf_list.txt")
RING_N_OUTER = 12              # representative TF embryo panels around the ring
RING_REPS_PER_LINEAGE = 2      # top-N most lineage-specific TFs picked per lineage
RING_MIN_PERTS = 10            # a lineage qualifies if >= this many TFs most deplete it
RING_N_REF = 40000             # reference cells drawn for each embryo panel
RING_DRAWS = 5                 # numerator draws averaged per nn-distortion panel
RING_N_MAX = 40000             # cap on the cells per nn-distortion draw
# lineages the screen's readout barely reaches -- a control leaf covers 0.3% of the
# reference blood cells and none of the extraembryonic endoderm, against ~33% everywhere
# else -- so every knockout's draw misses them for reasons that are not the knockout.
# they are out of the nn-distortion scope, and off the reference cells the panels draw.
RING_SKIP_LINEAGES = ("blood", "extraembryonic endoderm")
# lineages we show representatives for (only these get ring panels; order = key
# order in LINEAGE_ORDER).  set to None to fall back to the auto count-based rule.
RING_SHOW_LINEAGES = ["neural ectoderm", "neural crest", "paraxial mesoderm",
                      "lateral plate mesoderm", "endoderm", "surface ectoderm"]

NUMERICAL = os.path.join(os.path.dirname(__file__), "numerical_results")
IMGS = os.path.join(os.path.dirname(__file__), "imgs")
os.makedirs(NUMERICAL, exist_ok=True)
os.makedirs(IMGS, exist_ok=True)

RING_SVG = os.path.join(IMGS, "ring.svg")  # written by plot_ring.py

OFF_RATIO = 1.0                # projection-to-local distance ratio a node is off-manifold at
seed = 0

## PALETTE ##  (shared roles from utils.palettes)
INK = P.INK
GREY = "#dcdcdc"              # embryo backdrop points (setting-specific tone)
MUTED = P.MUTED               # bulk scatter points
# the ring panels' log nn distortion: blue = a reference cell sits closer to the
# knockout's leaves than to the reference cells, red = farther.
DISTORTION_CMAP = mpl.colormaps["RdBu_r"]

FONT_FAMILY = "cmss10"       # Computer Modern Sans (ships with matplotlib)

## RING UMAP (panel a) PALETTE + COSMETICS ##
# lineages grouped by germ layer (shared map in utils.palettes, so the same
# lineage keeps its colour in every figure); hues the central map by the lineage
# each knockout depletes most.
LINEAGE_COLORS = P.LINEAGE_COLORS
LINEAGE_ORDER = P.LINEAGE_ORDER
RING_CLIM_Q = 98             # percentile of |distortion| the panel colour limit sits at
RING_LINE_COL = "#9297a0"   # subway connector ink (grey; lineage hue is on the inset outlines)

## VALIDATION PANELS (b, c: MGI recall + gnomad constraint boxes) ##
# gnomad LoF-constraint box: LoF-intolerant vs LoF-tolerant gene bins.  a separate pair
# (deep maroon = intolerant/severe, warm taupe = tolerant/mild).
GNOMAD_COLORS = {"intolerant": "#8c2f39", "tolerant": "#cdb494"}
GNOMAD_LABELS = {"intolerant": "LoF\nintolerant", "tolerant": "LoF\ntolerant"}
# MGI KO-depletion box: random + all-annotated references, then coarse contexts.
# references are structural neutrals; the coarse contexts get their own qualitative set.
MGI_REF_COLORS = {"random": GREY, "all": "#4c6b8a"}   # steel-blue = annotated
MGI_COARSE_COLORS = {
    "CNS / brain":    "#3f7d6e",
    "eye":            "#6d9e4c",
    "ear":            "#b58a3c",
    "craniofacial":   "#c56b4a",
    "heart":          "#b0475e",
    "vascular":       "#9a5aa8",
    "somite":         "#5f74b3",
    "gut":            "#8a7a52",
    "liver":          "#a35c74",
    "extraembryonic": "#7f8794",
}

## ASSEMBLY GEOMETRY (inches; final figure must fit an A4 page, max width 6.5) ##
# two rows: the ring + its reference key (a), then the two validation boxes (b, c).
PAGE_W_IN = 6.5
RING_W_IN = 4.70            # ring block, left of row 1
REFKEY_W_IN = 1.70          # key column, right of row 1 (>=6pt text at this width)
TOPROW_GAP_IN = 0.05        # gap between the ring and the key
MGI_W_IN = 2.75             # panel b, left of row 2
GNOMAD_W_IN = 1.65          # panel c, right of row 2 (only two boxes)
BOX_H_IN = 1.55             # shared height of the two boxes
GAP_BC_IN = 0.15            # gap between b and c (the pair is centred on the page)
# the ring's white background is stripped, so its outer whitespace slides over the
# box row without painting over the labels.
GAP_ROW_IN = -0.10          # row 1 -> row 2


def apply_rc():
    """publication rcParams: Computer Modern Sans, text saved as vector paths."""
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "figure.dpi": 200,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
