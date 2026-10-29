"""paths and geometry for the combined perturb + tbx5 sheet. see concept.md.

no compute lives here. the type and the inch-axes `figure` come from the tbx5 config, so
the re-plotted boxes match the tbx5 panels beside them; the box palettes and caches come
from the perturb style.
"""
import os

from analysis.figures.perturb.style import (  # noqa: F401  (re-exported)
    GNOMAD_COLORS, GNOMAD_LABELS, MGI_REF_COLORS, MGI_COARSE_COLORS, MUTED, seed)
from analysis.figures.perturb.style import NUMERICAL as PERTURB_NR
from analysis.figures.tbx5.config import (  # noqa: F401  (re-exported)
    FS_LABEL, FS_TICK, INK, LETTER_BOX, LETTER_OFF, PAGE_PAD, PAGE_W, ROW1_H,
    despine, figure)
from analysis.figures.tbx5.config import PANELS as TBX5_PANELS  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
IMGS = os.path.join(HERE, "imgs")
PANELS = os.path.join(IMGS, "panels")
os.makedirs(PANELS, exist_ok=True)

## GEOMETRY (inches) ##
# row 0: the ring and its key (plot_ring.py), top-aligned and centred as a pair
RING_KEY_GAP = 0.05
RING_GAP = 0.04         # ring bottom -> row 1 top
# row 1: the two boxes take what the tbx5 scatters (1.625 in each) leave of the page.
# the axes top sits under the panel letter, since the top tick is at the axes edge; the
# bottom is higher than the scatters' to fit three-line ticks
MGI_W, GNOMAD_W = 2.15, 1.10
MGI_BOX = (0.36, 0.40, 1.76, 0.92)
GNOMAD_BOX = (0.36, 0.40, 0.70, 0.92)
