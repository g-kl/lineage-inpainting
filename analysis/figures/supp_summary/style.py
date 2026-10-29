"""paths, constants, palette and geometry for the cell type summary supp figure. see concept.md."""
import os

from analysis.figures.summary_qc.style import (  # noqa: F401
    COMPOSITION, MIN_COMPOSITION, MODEL_ASSIGNMENTS, N_TREES, PAGE_W_IN, PMI_QUANTILE,
    PT_PER_IN, RESOLUTION_BY_TIME, apply_rc, base_dir,
)
from utils import palettes as P

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.join(HERE, "numerical_results")
IMGS = os.path.join(HERE, "imgs")
PANELS = os.path.join(IMGS, "panels")
for d in (NR, PANELS):
    os.makedirs(d, exist_ok=True)

QIU = os.path.join(base_dir, "literature", "Qiu24_mouse_tree_edges.csv")
QIU_FULL = os.path.join(base_dir, "literature", "Qiu24_mouse_tree_edges_full.csv")
MAP_CSV = os.path.join(HERE, "celltype_map.csv")
MAP_PKL = os.path.join(NR, "cell_type_map.pkl")

# lineages left out of both sides: their transitions, and the qiu edges only they carry
EXCLUDE = {"blood", "extraembryonic ectoderm", "extraembryonic endoderm",
           "extraembryonic mesoderm"}
MAX_HOPS = 2       # one of our transitions is often a two-step qiu path

# a transition's category, in panel a's stacking order. the four hues are chosen so
# adjacent pairs stay distinguishable (including under colour-vision deficiency) in
# this order; unmapped is the neutral.
CATEGORIES = ["match", "reversed", "boundary", "other", "unmapped"]
CATEGORY_LABELS = {"match": "match", "reversed": "reverse", "boundary": "boundary",
                   "other": "other mismatch", "unmapped": "unmapped"}
CATEGORY_COLORS = {"match": "#2a78d6", "reversed": "#e87ba4", "boundary": "#eda100",
                   "other": "#e34948", "unmapped": P.GREY}

N_TOP = 10         # panels c-f: the heaviest transitions
FS_ROW = 8.0       # panels c-f row labels: the one place below 10 pt

# geometry (inches). a and b run the page width
BAR_AX_H = 0.24    # the bar's axes
BAR_BOTTOM = 0.44  # ticks and x label
BAR_TOP = 0.04     # above the bar in b
LEG_H = 0.24       # the legend strip above the bar in a
ROW_GAP = 0.08
