"""paths, methods, palette and geometry for the held-out clone supp figure. see concept.md."""
import os

from analysis.figures.summary_qc.style import (  # noqa: F401
    INK, METHOD_COLORS, PAGE_W_IN, PT_PER_IN, apply_rc, base_dir,
)

HERE = os.path.dirname(os.path.abspath(__file__))
NR = os.path.join(HERE, "numerical_results")
IMGS = os.path.join(HERE, "imgs")
PANELS = os.path.join(IMGS, "panels")
for d in (NR, PANELS):
    os.makedirs(d, exist_ok=True)

# gather.py writes the merged per-row caches here (fig 3's schema)
RESULTS = os.path.join(base_dir, "analysis", "results", "hold_out_metrics")

# display order, top to bottom: (method, label, family). the *_all rows are the paper's
# models, trained on every tree, and draw as open dots.
ROWS = [
    ("vpsde", "lineage inpainting", "model"),
    ("vpsde_all", "lineage inpainting, all trees", "model"),
    ("genot_sinkhorn_eps0.01", "GENOT-Sinkhorn, $\\varepsilon$ 0.01", "genot_sinkhorn"),
    ("genot_sinkhorn_eps0.05", "GENOT-Sinkhorn, $\\varepsilon$ 0.05", "genot_sinkhorn"),
    ("genot_sinkhorn_eps0.2", "GENOT-Sinkhorn, $\\varepsilon$ 0.2", "genot_sinkhorn"),
    ("genot_sinkhorn_all", "GENOT-Sinkhorn, all trees", "genot_sinkhorn"),
    ("genot_cbfgw_fp0.1", "GENOT-cbFGW, fused penalty 0.1", "genot_cbfgw"),
    ("genot_cbfgw_fp1", "GENOT-cbFGW, fused penalty 1", "genot_cbfgw"),
    ("genot_cbfgw_fp10", "GENOT-cbFGW, fused penalty 10", "genot_cbfgw"),
    ("genot_cbfgw_all", "GENOT-cbFGW, all trees", "genot_cbfgw"),
]
METHODS = [m for m, _, _ in ROWS]

## TYPE (pt) ##
FS_LABEL, FS_TICK = 10, 9

## GEOMETRY (inches) ##
ROW_H = 0.17          # per method row
FAMILY_GAP = 0.5      # extra rows between method families
LEFT = 1.95           # method labels, the longest is 1.83
BOTTOM, TOP, RIGHT = 0.62, 0.05, 0.08
WSPACE = 0.32         # between metric columns, as a fraction of a column
XPAD = 1.08           # x axis runs to this times the largest interval end
