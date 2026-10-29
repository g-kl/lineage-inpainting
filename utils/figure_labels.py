"""Shared figure helpers: the repo display font (Latin Modern Sans, in the
utils/fonts/ dir), bold lowercase panel letters as fixed-box vector SVGs, titled
rounded group boxes, and a text-to-SVG-path renderer for hand-assembled SVGs
(renderer-independent CM Sans)."""
import os
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.path import Path

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_FAMILY = "Latin Modern Sans"     # cmss10 successor; has real weights, so bold works
_FONT_FILES = ("lmsans10-regular.otf", "lmsans10-bold.otf")


def register_fonts():
    """Register the shared Latin Modern Sans faces with matplotlib (idempotent)."""
    have = {os.path.basename(e.fname) for e in fm.fontManager.ttflist}
    for fn in _FONT_FILES:
        p = os.path.join(FONTS_DIR, fn)
        if os.path.exists(p) and fn not in have:
            fm.fontManager.addfont(p)


register_fonts()   # on import, so FONT_FAMILY is usable immediately

PANEL_LETTER_FS = 16.0     # default panel-letter point size
LETTER_BOX_IN = 0.30       # fixed square box; equal coords => aligned letters


def panel_letter_svg(out_path, ch, *, fontsize=PANEL_LETTER_FS, box_in=LETTER_BOX_IN,
                     color="#222222"):
    """Write a bold lowercase panel letter (vector path) into a fixed square box at
    ``out_path`` and return it. Not tight-cropped, so all letters share one anchor."""
    register_fonts()
    fp = FontProperties(family=FONT_FAMILY, weight="bold", size=fontsize)
    with plt.rc_context({"svg.fonttype": "path"}):
        fig = plt.figure(figsize=(box_in, box_in))
        fig.text(0.12, 0.30, ch, ha="left", va="baseline", color=color,
                 fontproperties=fp)
        fig.savefig(out_path, transparent=True)
        plt.close(fig)
    return out_path


BOX_TITLE_FS = 9.0         # default group-box title point size
BOX_TITLE_GAP_PT = 10.0    # slack around the title in the border break
BOX_CORNER_PT = 4.0


def box_title_svg(out_path, text, *, fontsize=BOX_TITLE_FS, color="#222222"):
    """Write a tight-cropped one-line group-box title (vector path) at ``out_path``.

    Returns (path, (w_in, h_in)) so an assembler can break the box border to fit it.
    """
    register_fonts()
    with plt.rc_context({"svg.fonttype": "path", "font.family": FONT_FAMILY,
                         "mathtext.fontset": "cm"}):
        fig = plt.figure(figsize=(4.0, 0.4))
        fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize,
                 color=color)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01, transparent=True)
        plt.close(fig)
    root = ET.parse(out_path).getroot()
    w = float(root.get("width").rstrip("pt"))
    h = float(root.get("height").rstrip("pt"))
    return out_path, (w / 72.0, h / 72.0)


def box_border_d(x0, y0, x1, y1, title_w_pt, *, r=BOX_CORNER_PT,
                 gap_pt=BOX_TITLE_GAP_PT):
    """SVG path data for a rounded rectangle whose top edge carries a centred gap
    (fieldset-style break) wide enough to hold a ``title_w_pt``-wide title."""
    cx = (x0 + x1) / 2.0
    gap = title_w_pt + gap_pt
    gx0, gx1 = cx - gap / 2.0, cx + gap / 2.0
    return (f"M {gx0:.2f} {y0:.2f} L {x0 + r:.2f} {y0:.2f} "
            f"Q {x0:.2f} {y0:.2f} {x0:.2f} {y0 + r:.2f} L {x0:.2f} {y1 - r:.2f} "
            f"Q {x0:.2f} {y1:.2f} {x0 + r:.2f} {y1:.2f} L {x1 - r:.2f} {y1:.2f} "
            f"Q {x1:.2f} {y1:.2f} {x1:.2f} {y1 - r:.2f} L {x1:.2f} {y0 + r:.2f} "
            f"Q {x1:.2f} {y0:.2f} {x1 - r:.2f} {y0:.2f} L {gx1:.2f} {y0:.2f}")


_SVG_CMDS = {Path.MOVETO: "M", Path.LINETO: "L", Path.CURVE3: "Q", Path.CURVE4: "C"}
_NPTS = {Path.MOVETO: 1, Path.LINETO: 1, Path.CURVE3: 2, Path.CURVE4: 3}
_ITALIC_SHEAR = 0.2126     # tan(12 deg); synthetic slant (Latin Modern Sans has no italic face)


def text_svg_path(s, x, y, size, *, color="#222222", weight="normal", italic=False,
                  anchor="start", valign="baseline"):
    """<path> element drawing ``s`` in Latin Modern Sans at (x, y), for the given
    anchor (start|middle|end) in SVG user units (y down). y is the baseline unless
    valign="center", which centres the visible glyph bounds on y instead."""
    register_fonts()
    if not s:
        return ""
    tp = TextPath((0, 0), s, prop=FontProperties(family=FONT_FAMILY, weight=weight,
                                                 size=size))
    v = tp.vertices.copy()
    if italic:
        v[:, 0] += _ITALIC_SHEAR * v[:, 1]
    if valign == "center":
        y += (v[:, 1].min() + v[:, 1].max()) / 2
    elif valign != "baseline":
        raise ValueError(f"unknown vertical alignment {valign!r}")
    xmin, xmax = v[:, 0].min(), v[:, 0].max()
    dx = x + {"start": -xmin, "middle": -(xmin + xmax) / 2, "end": -xmax}[anchor]
    codes, d, i = tp.codes, [], 0
    while i < len(codes):
        code = codes[i]
        if code == Path.CLOSEPOLY:
            d.append("Z")
            i += 1
        else:
            n = _NPTS[code]
            d.append(_SVG_CMDS[code] + " ".join(
                f"{v[i + k, 0] + dx:.2f} {y - v[i + k, 1]:.2f}" for k in range(n)))
            i += n
    return f'<path d="{"".join(d)}" fill="{color}" fill-rule="evenodd"/>'
