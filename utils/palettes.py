"""Shared colour palettes for the publication figures. Import roles from here so
the same concept keeps the same colour across every figure."""

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def truncate(cmap, lo=0.0, hi=1.0, *, n=256, name="truncated"):
    """A colormap restricted to the [lo, hi] slice of ``cmap``'s ramp (e.g. to keep
    the near-white end of a ramp off a heatmap whose values reach the limit)."""
    return LinearSegmentedColormap.from_list(name, cmap(np.linspace(lo, hi, n)))

# methods: our model (signature teal) + the OT / FM baselines. adding a baseline means
# adding it here and in synthetic_evals/baselines/registry.py; the figures read this.
MODEL, FM, GS, GF = "#0f6b6e", "#c47a3d", "#7a3d8c", "#3d8c4a"
WOT, MOSLIN = "#b5476b", "#4a6fa5"
METHOD_COLORS = {"model": MODEL, "fm": FM, "gs": GS, "gf": GF,
                 "wot": WOT, "moslin": MOSLIN}
METHOD_LABELS = {"model": "ours", "fm": "multimarginal FM",
                 "gs": "GENOT-Sinkhorn", "gf": "GENOT-FGW",
                 "wot": "Waddington-OT", "moslin": "moslin"}
METHOD_MARKERS = {"model": "s", "fm": "o", "gs": "^", "gf": "D",
                  "wot": "v", "moslin": "P"}
METHOD_ORDER = ["model", "gs", "gf", "fm", "wot", "moslin"]

# pooler ablations of our model: same splitter/aligner/schedule, unlearned pooler.
# desaturated tints of MODEL so they read as variants of ours, not separate methods.
ABLATION_COLORS = {"ours_meanpool": "#5c9a9c", "ours_nnpool": "#a3c6c7"}
ABLATION_LABELS = {"ours_meanpool": "ours, mean pooler",
                   "ours_nnpool": "ours, NN-snapped pooler"}
ABLATION_ORDER = ["ours_meanpool", "ours_nnpool"]

# lineage identity, grouped by germ layer (keys = dataset lineage names)
LINEAGE_COLORS = {
    "neural ectoderm":         "#40ADA6",
    "ectoderm":                "#86C9A2",
    "surface ectoderm":        "#C2E0D0",
    "neural crest":            "#1C6F68",
    "extraembryonic ectoderm": "#C4C98A",
    "mesoderm":                "#3C4EBF", 
    "paraxial mesoderm":       "#9C74B4",
    "lateral plate mesoderm":  "#A2A8DB",
    "intermediate mesoderm":   "#6F74CC",
    "extraembryonic mesoderm": "#C4BAE2",
    "blood":                   "#D6543A",
    "endoderm":                "#E3A600",
    "extraembryonic endoderm": "#EDB1A3",
    "epiblast":                "#A39D92",
    "primordial germ cell":    "#C25D8C",
}
LINEAGE_ORDER = list(LINEAGE_COLORS)

# cell-type palette for the EC-convergence figure (fan-in / trajectory sources)
CONVERGENCE_CELLTYPES = ["#2c6e8f", "#d17a2a", "#5a9e4f", "#9e4f7a", "#c2a83e",
                         "#6f5aa0"]

# developing-heart anatomical subtypes (heart-perturbation figure diagram row),
# ordered outflow -> inflow; a fresh qualitative set, distinct from LINEAGE_COLORS
HEART_SUBTYPE_COLORS = {
    "OFT":      "#c9457b",
    "aSHF":     "#e07b39",
    "RV":       "#d4a72c",
    "LV":       "#5a9e5a",
    "AV canal": "#2f9c9c",
    "Atria":    "#4b5bc4",
    "IFT":      "#9a56b8",
}
HEART_DEPL_CMAP = LinearSegmentedColormap.from_list(
    "heart_depl", ["#fff5f0", "#a50f15"])   # 0-100% KO depletion (sequential red; depletion is bad)

# structural neutrals
INK, MUTED, GREY, BACKDROP = "#222222", "#8a8a8a", "#c9c9c9", "#e9e9ec"

# highlight hues
ACCENT, ACCENT_ALT, PRED = "#c44536", "#2A9D8F", "#c81d7a"
CONTROL, PERTURB = "#3b6fb6", "#c0392b"

# colormaps (standard named; seaborn ok)
EXPR_CMAP = mpl.colormaps["viridis"]        # gene expression
EXPR_GREEN = mpl.colormaps["Greens"]        # single-gene expression on trees
FATEBIAS_CMAP = mpl.colormaps["Reds"]       # fate bias (0..1, warm)
OT_CMAP = mpl.colormaps["Blues"]            # OT discrepancy
CONV_CMAP = mpl.colormaps["Purples"]        # convergence score
DEG_CMAP = mpl.colormaps["PuOr"]            # DEG z-score (diverging)
DENSITY_CMAP = mpl.colormaps["coolwarm"]    # log-density ratio (diverging)
DIVERGING_CMAP = sns.color_palette("vlag", as_cmap=True)   # correlation, centred
SIM_CMAP = sns.color_palette("crest", as_cmap=True)        # similarity (sequential)
LANDSCAPE_CMAP = mpl.colormaps["gist_earth_r"]             # energy-magnitude heatmap
