"""paths, geometry and rc for the supplement figures. see concept.md.

everything visual (method palette, type sizes, page width, rc) comes from the main
figure's style so the supplement matches it; only paths and supplement-specific geometry
live here.
"""

import os

from synthetic_evals.figure.style import (  # noqa: F401  (re-exported for the panels)
    ABLATIONS, ACC_CMAP, ACC_VMAX, ACC_VMIN, CATEGORIES, CAT_LABELS, CONTEXT_LANDSCAPE_DIR,
    CROSS_EXP, DIMS, ERR_CMAP, FS_LABEL, FS_LEGEND, FS_SUB, FS_TICK, FS_TITLE, HD_DIR,
    INK, LANDSCAPE_DIR, MAIN_METHODS, METHOD_COLORS, METHOD_LABELS, METHOD_MARKERS,
    METHODS, MUTED, PAGE_W_IN, PT_PER_IN, REFERENCES, ROW_GAP_IN, apply_rc,
    methods_present, save_panel, thin_yaxis,
)

HERE = os.path.dirname(__file__)
NR = os.path.join(HERE, "numerical_results")
IMGS = os.path.join(HERE, "imgs")
# the main figure's caches, read directly by supplement panels that widen a main-text
# panel rather than distilling anything new (see plot_baselines).
MAIN_NR = os.path.join(HERE, "..", "figure", "numerical_results")
os.makedirs(NR, exist_ok=True)
os.makedirs(IMGS, exist_ok=True)

# the alternate two-sample metrics the robustness figure sweeps, and how to label them
METRICS = ["energy", "sinkhorn", "mmd_gauss"]
METRIC_LABELS = {"energy": "energy distance", "sinkhorn": "Sinkhorn divergence",
                 "mmd_gauss": "gaussian MMD"}

PAGE_PAD_IN = 0.03
