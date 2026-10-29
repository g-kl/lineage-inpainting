#!/usr/bin/env python3
"""problem setup, approach 2: start from the data, then the model, then fitting.

reuses the Canvas / Node / tree helpers from 05_lineage_completion, with the type
switched to CM serif (cmunrm/cmunti in utils/fonts) so the figure sits in the same
typographic world as the paper. panel letters follow the computed figures: bold
lowercase Latin Modern Sans at 16 pt, alone at the row's top-left corner.

node styles, used the same way in every row:
  solid                 observed / given
  dashed grey, empty    unobserved or masked
  dashed colour, light  filled in by the model

two files come out, differing only in row d's attention half: the base figure has
the d_tree bias over the state-state attention, and `_quad` adds a third matrix for
their weighted sum, with the tree in the fourth corner.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import importlib.util
import math
import random
import re
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent
UTIL_PATH = ROOT.parent / "05_lineage_completion" / "make_lineage_completion_figure.py"
FIG_STEM = "problem_setup_approach2"
QUAD_STEM = FIG_STEM + "_quad"

spec = importlib.util.spec_from_file_location("lineage_completion_utils", UTIL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load drawing utilities from {UTIL_PATH}")
util = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = util
spec.loader.exec_module(util)

# the group box is the one the computed figures draw: same path generator, same
# rounded corner and title-gap constants, just fed px instead of pt.
sys.path.insert(0, str(ROOT.parents[1]))
from utils.figure_labels import (  # noqa: E402
    BOX_CORNER_PT, BOX_TITLE_GAP_PT, LETTER_BOX_IN, PANEL_LETTER_FS,
    box_border_d, text_svg_path,
)

PX_PER_PT = 240 / 72

# ---------------------------------------------------------------- type
CM_DIR = ROOT.parents[1] / "utils" / "fonts"
util.FONT_REG = CM_DIR / "cmunrm.otf"
util.FONT_BOLD = CM_DIR / "cmunrm.otf"
util.FONT_ITALIC = CM_DIR / "cmunti.otf"
util.FONT_FAMILY = "CMU Serif, Latin Modern Roman, Computer Modern Roman, serif"
SANS_BOLD = ROOT.parents[1] / "utils" / "fonts" / "lmsans10-bold.otf"

SCALE = util.SCALE
Node = util.Node
STATES = [util.RED, util.ORANGE, util.OLIVE, util.GREEN, util.TEAL, util.BLUE]

INK = "#222222"
LABEL = "#4c525c"
EDGE = "#ccd2dc"
# dash and gap of a dashed edge, as drawn. the png lines have butt ends and the svg
# ones round caps, so draw_tree shortens the svg dash by the stroke width to match.
EDGE_DASH = (7.0, 4.0)
AXIS = "#a9b1be"
RULE = "#dfe3ea"
SLICE = "#ebedf1"
PLANE = "#fafbfd"              # slab stock, shared by row b's cards and row d's planes
STEM = "#6f7682"
GOLD = util.GOLD

W, H = 1560, 1877
H_QUAD = 1861                  # the quad layout runs a touch taller than H
PAPER_WIDTH_IN = 6.5
# 240 px/in, so 34 px = 10.2 pt. TINY is only used for time-axis subscripts.
BODY, TINY = 34, 26
# the panel letter is the one the computed figures draw: PANEL_LETTER_FS bold Latin
# Modern Sans, sitting in a fixed LETTER_BOX_IN square that is anchored by its
# top-left corner, with the glyph inset inside it. those insets are the ones
# panel_letter_svg bakes in (0.12 across, 0.30 up from the bottom), so a letter here
# lands where an assemble.py would nest one. LETTER_TOP is where that corner sits
# inside a row.
LETTER = PANEL_LETTER_FS * PX_PER_PT
LETTER_BOX = LETTER_BOX_IN * 72 * PX_PER_PT
LETTER_PAD = (0.12 * LETTER_BOX, 0.70 * LETTER_BOX)
LETTER_TOP = 16
# 30 px = 9 pt, below the 10 pt floor the rest of the figure keeps:
# row d's two halves are named, and those names should not read as loudly as the
# box title above them or the labels inside the drawings.
SUB = 30

BOX_X0, BOX_X1 = 14, W - 14
LEFT = 46                      # content margin inside the box border
CW = BOX_X1 - 14 - LEFT
BAND = 216
LEVEL = {"r": 0, "a": 1, "b": 1, "a1": 2, "a2": 2, "b1": 2, "b2": 2}
KEPT = ("a1", "a2", "b1", "b2")
# even out the t1 leaves of embryo 1; the shared helper spreads the lone
# right-hand daughter much further than the pair on the left.
XFIX = {1: {"l3": 0.70, "a": 0.325, "r": 0.5125}}


class Canvas(util.Canvas):
    """adds a `letter` style: real bold Latin Modern Sans, which cmun has no face for."""

    def letter(self, x: float, y: float, ch: str, size: float = LETTER) -> None:
        """`x, y` is the top-left of the fixed letter box, as in the computed
        figures. the svg carries the glyph as a path rather than as <text>, which is
        what makes it match: live text renders in whatever sans the viewer happens
        to substitute for Latin Modern."""
        gx, gy = x + LETTER_PAD[0], y + LETTER_PAD[1]
        font = util.ImageFont.truetype(str(SANS_BOLD), round(size * SCALE))
        # both renderers align the glyph's ink, not its pen origin, so the png and
        # the path agree to the pixel.
        self.draw.text(self.xy(gx - font.getbbox(ch)[0] / SCALE, gy), ch,
                       fill=util.rgba(INK), font=font, anchor="ls")
        self.svg_body.append(text_svg_path(ch, gx, gy, size, color=INK, weight="bold"))

    def dashed_polyline_png(self, points, stroke, stroke_width, opacity,
                            dash_len=EDGE_DASH[0], gap_len=EDGE_DASH[1]) -> None:
        """the shared helper's 11/9 dash leaves these short edges mostly gap."""
        super().dashed_polyline_png(points, stroke, stroke_width, opacity, dash_len, gap_len)

    def ellipse(self, x: float, y: float, rx: float, ry: float, fill: str = "none",
                stroke: str = "none", stroke_width: float = 1.0) -> None:
        bbox = (*self.xy(x - rx, y - ry), *self.xy(x + rx, y + ry))
        self.draw.ellipse(bbox, fill=None if fill == "none" else util.rgba(fill),
                          outline=None if stroke == "none" else util.rgba(stroke),
                          width=max(1, round(stroke_width * SCALE)))
        self.svg_body.append(self.tag("ellipse", {
            "cx": x, "cy": y, "rx": rx, "ry": ry, "fill": fill, "stroke": stroke,
            "stroke-width": stroke_width if stroke != "none" else None,
        }))

    def vtext(self, x: float, y: float, value: str, size: int = 34,
              fill: str = "#4c525c", angle: float = 90) -> None:
        """reading bottom to top. pillow has no rotated text, so it is drawn flat
        into a scratch layer and composited; the svg carries a rotate transform."""
        font = self.font(size)
        layer = util.Image.new("RGBA", (round(font.getlength(value)) + 8,
                                        round(size * SCALE * 1.6)), (0, 0, 0, 0))
        util.ImageDraw.Draw(layer).text((4, layer.height / 2), value, fill=util.rgba(fill),
                                        font=font, anchor="lm")
        layer = layer.rotate(angle, expand=True)
        px, py = self.xy(x, y)
        self.image.alpha_composite(layer, (round(px - layer.width / 2),
                                           round(py - layer.height / 2)))
        self.svg_body.append(self.tag("text", {
            "x": x, "y": y, "fill": fill, "font-family": util.FONT_FAMILY,
            "font-size": size, "text-anchor": "middle", "dominant-baseline": "middle",
            "transform": f"rotate({-angle} {x} {y})",
        }, value))


def text_w(canvas: Canvas, value: str, size: int = BODY, style: str = "regular") -> float:
    return canvas.font(size, style).getlength(value) / SCALE


# ---------------------------------------------------------------- trees

def place(nodes: list[Node], x: float, w: float, y0: float, y1: float) -> list[Node]:
    """map unit x onto [x, x+w] and unit y (1 == t3) onto the shared time axis."""
    xs = [n.x for n in nodes]
    lo, hi = min(xs), max(xs)
    return [
        replace(n, x=x + (n.x - lo) / (hi - lo) * w, y=y0 + n.y * (y1 - y0))
        for n in nodes
    ]


def model_tree() -> list[Node]:
    """four generations on a clean lattice, one interior generation observed.

    shares its shape with the snapshot tree in row c, so the conditioning setup
    the model wants (row b) and the one it is trained on (row c) look identical.
    """
    observed = {"a1": util.RED, "a2": util.ORANGE, "b1": util.GREEN, "b2": util.TEAL}
    return [
        Node(n.id, n.parent, n.x, LEVEL.get(n.id, 3) / 3,
             observed.get(n.id, util.PURPLE),
             "observed" if n.id in observed else "blank")
        for n in util.observed_tree(3)
    ]


def snapshot_tree(stage: int) -> list[Node]:
    """one embryo, harvested at t{stage}: states at the bottom layer, nothing else."""
    fix = XFIX.get(stage, {})
    nodes = [replace(n, x=fix.get(n.id, n.x)) for n in util.observed_tree(stage)]
    leaf_y = max(n.y for n in nodes if n.kind == "observed")
    scale = (stage / 3) / leaf_y
    leaves = [n.id for n in nodes if n.kind == "observed"]
    ramp = {
        nid: STATES[round(i * (len(STATES) - 1) / (len(leaves) - 1))]
        for i, nid in enumerate(leaves)
    }
    return [
        replace(n, y=n.y * scale, color=ramp.get(n.id, n.color),
                kind="observed" if n.kind == "observed" else "blank")
        for n in nodes
    ]


def fill_states(nodes: list[Node], rng: random.Random) -> list[Node]:
    """one complete labelling: ancestors follow their observed descendants, children follow parents."""
    by_id = {n.id: n for n in nodes}
    kids: dict[str, list[str]] = {n.id: [] for n in nodes}
    for n in nodes:
        if n.parent:
            kids[n.parent].append(n.id)

    def observed_below(nid: str) -> list[int]:
        node = by_id[nid]
        if node.kind == "observed":
            return [STATES.index(node.color)]
        return [i for k in kids[nid] for i in observed_below(k)]

    idx: dict[str, int] = {}
    out = []
    for n in nodes:
        if n.kind == "observed":
            i = STATES.index(n.color)
        elif kids[n.id]:
            below = observed_below(n.id)
            i = round(sum(below) / len(below)) + rng.choice([-1, -1, 0, 1, 1])
        else:
            i = idx[n.parent] + rng.choice([-2, -1, 0, 0, 1, 2])
        i = max(0, min(len(STATES) - 1, i))
        idx[n.id] = i
        out.append(replace(n, color=STATES[i],
                           kind="observed" if n.kind == "observed" else "generated"))
    return out


def draw_tree(canvas: Canvas, nodes: list[Node], r: float, edge_width: float = 2.6,
              fade: float = 0.0, dashed: frozenset[str] = frozenset()) -> None:
    """fade washes everything toward the paper, for cards sitting behind the front one.
    edges out of a node in `dashed` are drawn dashed."""
    by_id = {n.id: n for n in nodes}
    for n in nodes:
        if n.parent:
            d, pts = util.edge_path(by_id[n.parent], n)
            canvas.path(d, pts, stroke=util.blend(EDGE, util.PAPER, fade),
                        stroke_width=edge_width,
                        dash=(f"{EDGE_DASH[0] - edge_width:g} {EDGE_DASH[1] + edge_width:g}"
                              if n.parent in dashed else None))
    for n in nodes:
        color = util.blend(n.color, util.PAPER, fade)
        if n.kind == "generated":
            canvas.circle(n.x, n.y, r, fill=util.blend(color, util.PAPER, 0.55),
                          stroke=color, stroke_width=2.2, dash="3 4")
        else:
            util.draw_node(canvas, replace(n, color=color), r)


def slice_band(canvas: Canvas, nodes: list[Node], pad: float = 22, height: float = 30,
               fade: float = 0.0) -> None:
    lo, hi = util.observed_span(nodes)
    y = util.frontier_y(nodes)
    canvas.rect(lo - pad, y - height / 2, (hi - lo) + 2 * pad, height,
                fill=util.blend(SLICE, util.PAPER, fade), stroke="none", rx=4)


# each plate of row b's deck has a hairline thickness, along the same down-right
# direction the stack recedes; SIDE and BASE are its two wall shades, lit from the
# upper left so the floor is the darker one. the depth comes from the stack being
# dense -- four plates close together -- not from any one plate being thick.
EXT = (6.0, 4.0)
SIDE = util.blend(RULE, util.PAPER, 0.45)
BASE = RULE


def slab(canvas: Canvas, x: float, y: float, w: float, h: float,
         fade: float = 0.0) -> None:
    """front face plus the right and bottom walls the recession exposes, with the
    corner bevel between them. flat fills and hairline edges, matching how row d
    builds its planes."""
    ex, ey = EXT
    edge = util.blend(RULE, util.PAPER, fade * 0.5)
    walls = (
        ([(x + w, y), (x + w + ex, y + ey), (x + w + ex, y + h + ey), (x + w, y + h)], SIDE),
        ([(x, y + h), (x + ex, y + h + ey), (x + w + ex, y + h + ey), (x + w, y + h)], BASE),
    )
    for face, shade in walls:
        canvas.polygon(face, fill=util.blend(shade, util.PAPER, fade), stroke=edge,
                       stroke_width=1.0)
    canvas.rect(x, y, w, h, fill=PLANE, stroke=edge, stroke_width=1.2)


def deck(canvas: Canvas, samples: list[list[Node]], x: float, y0: float, y1: float,
         tw: float, pad_x: float = 22, pad_y: float = 30, dx: float = 77,
         dy: float = 11, r: float = 10) -> tuple[float, float]:
    """the samples as a dense stack of thin plates, front one top-left, the rest
    receding down and to the right, painted back to front. cut from the same stock
    as row d's planes -- sharp corners, PLANE face, RULE edges -- so both panels
    show depth with flat polygons.

    the front plate carries a whole tree; the rest keep a sliver wide enough for
    one observed cell and its two daughters: the conditioning is the same on every
    plate, the completion is not. dx is picked so the occluding edge lands in a gap
    between node columns, and dy has to stay under pad_y, or a back plate's leaf row
    drops below the front plate's floor."""
    cw, ch = tw + 2 * pad_x, (y1 - y0) + 2 * pad_y
    for i, nodes in reversed(list(enumerate(samples))):
        cx, cy = x + i * dx, (y0 - pad_y) + i * dy
        fade = min(0.44, 0.24 * i)
        slab(canvas, cx, cy, cw, ch, fade)
        placed = place(nodes, cx + pad_x, tw, cy + pad_y, cy + ch - pad_y)
        slice_band(canvas, placed, pad=18, height=28, fade=fade)
        draw_tree(canvas, placed, r=r, edge_width=2.5, fade=fade,
                  dashed=frozenset(n.id for n in placed if n.kind == "observed"))
    n = len(samples) - 1
    return (x + n * dx + cw + EXT[0], (y0 - pad_y) + n * dy + ch + EXT[1])


# ---------------------------------------------------------------- chrome

def path_points(d: str, steps: int = 16) -> list[tuple[float, float]]:
    """sample an M/L/Q path into a polyline, so the png can trace the same `d` the
    svg carries. only the commands box_border_d emits are handled."""
    tok = re.findall(r"[MLQ]|-?\d+\.?\d*", d)
    pts: list[tuple[float, float]] = []
    cur = (0.0, 0.0)
    i = 0
    while i < len(tok):
        cmd, i = tok[i], i + 1
        if cmd in "ML":
            cur = (float(tok[i]), float(tok[i + 1]))
            i += 2
            pts.append(cur)
        else:
            c = (float(tok[i]), float(tok[i + 1]))
            e = (float(tok[i + 2]), float(tok[i + 3]))
            i += 4
            for k in range(1, steps + 1):
                t = k / steps
                u = 1 - t
                pts.append((u * u * cur[0] + 2 * u * t * c[0] + t * t * e[0],
                            u * u * cur[1] + 2 * u * t * c[1] + t * t * e[1]))
            cur = e
    return pts


def group_box(canvas: Canvas, y0: float, y1: float, title: str) -> None:
    """rounded border broken by a centred title, as in summary_qc / synthetic_evals."""
    d = box_border_d(BOX_X0, y0, BOX_X1, y1, text_w(canvas, title),
                     r=BOX_CORNER_PT * PX_PER_PT, gap_pt=BOX_TITLE_GAP_PT * PX_PER_PT)
    canvas.path(d, path_points(d), stroke=INK, stroke_width=0.8 * PX_PER_PT)
    canvas.text((BOX_X0 + BOX_X1) / 2, y0, title, size=BODY, fill=INK, anchor="middle")


def time_axis(canvas: Canvas, x: float, y0: float, y1: float, levels: int = 3) -> None:
    canvas.line(x, y0, x, y1, AXIS, 1.4)
    tw = text_w(canvas, "t", BODY, "italic")
    for i in range(levels + 1):
        y = y0 + (y1 - y0) * i / levels
        canvas.line(x - 8, y, x + 8, y, AXIS, 1.4)
        canvas.text(x - 40, y, "t", size=BODY, fill=LABEL, anchor="end", style="italic")
        canvas.text(x - 40 + tw * 0.15, y + 11, str(i), size=TINY, fill=LABEL, anchor="start")


def t_label(canvas: Canvas, x: float, y: float, index: int) -> None:
    """italic t with its generation subscript, centred -- as time_axis composes it."""
    tw = text_w(canvas, "t", BODY, "italic")
    x0 = x - (tw + text_w(canvas, str(index), TINY)) / 2
    canvas.text(x0, y, "t", size=BODY, fill=LABEL, anchor="start", style="italic")
    canvas.text(x0 + tw * 1.15, y + 11, str(index), size=TINY, fill=LABEL, anchor="start")


def brace(canvas: Canvas, x1: float, x2: float, y: float,
          h: float = 30.0, tip: float = 18.0, c: float = 56.0) -> None:
    """latex-style underbrace -- arms up at y, straight spine at y+h, cusp pointing
    down at y+h+tip. `c` is the width of the four elbows, clamped so the elbows
    cannot meet in the middle."""
    mid, s = (x1 + x2) / 2, y + h
    c = min(c, 0.45 * (mid - x1))
    segments = [
        ((x1, y), (x1, s), (x1, s), (x1 + c, s)),
        ((x1 + c, s), (x1 + c, s), (mid - c, s), (mid - c, s)),
        ((mid - c, s), (mid, s), (mid, s), (mid, s + tip)),
        ((mid, s + tip), (mid, s), (mid, s), (mid + c, s)),
        ((mid + c, s), (mid + c, s), (x2 - c, s), (x2 - c, s)),
        ((x2 - c, s), (x2, s), (x2, s), (x2, y)),
    ]
    d = [f"M {x1:.1f} {y:.1f}"]
    pts: list[tuple[float, float]] = []
    for p0, p1, p2, p3 in segments:
        d.append(f"C {p1[0]:.1f} {p1[1]:.1f}, {p2[0]:.1f} {p2[1]:.1f}, {p3[0]:.1f} {p3[1]:.1f}")
        pts.extend(util.cubic_points(p0, p1, p2, p3, steps=20))
    canvas.path(" ".join(d), pts, stroke=STEM, stroke_width=1.7)


def arrow(canvas: Canvas, x1: float, x2: float, y: float, color: str = STEM,
          width: float = 1.6, head: float = 17.0, half: float = 5.4) -> None:
    sign = 1 if x2 > x1 else -1
    canvas.line(x1, y, x2 - sign * head * 0.6, y, color, width)
    canvas.polygon([(x2, y), (x2 - sign * head, y - half), (x2 - sign * head, y + half)],
                   fill=color)


def cdots(canvas: Canvas, x: float, y: float, gap: float = 17.0, r: float = 2.6) -> None:
    for i in (-1, 0, 1):
        canvas.circle(x + i * gap, y, r, fill=LABEL)


def caption(canvas: Canvas, x: float, y: float, value: str) -> None:
    canvas.text(x, y, value, size=BODY, fill=LABEL, anchor="middle")


def subtitle(canvas: Canvas, x: float, y: float, value: str, size: int = SUB) -> None:
    """centred and underlined -- how row d names each of its two halves. the rule
    sits clear of the descenders, at the depth a text-decoration underline would."""
    half, rule_y = text_w(canvas, value, size) / 2, y + size * 0.62
    canvas.text(x, y, value, size=size, fill=LABEL, anchor="middle")
    canvas.line(x - half, rule_y, x + half, rule_y, LABEL, 1.1)


def legend(canvas: Canvas, right: float, y: float) -> None:
    items = [("observed", "observed"), ("unobserved", "blank"), ("generated", "generated")]
    gap, chip, pad = 30, 20, 12
    width = sum(chip + pad + text_w(canvas, label) for label, _ in items) + gap * (len(items) - 1)
    x = right - width
    for label, kind in items:
        draw_tree(canvas, [Node("k", None, x + chip / 2, y, util.TEAL, kind)], chip / 2)
        canvas.text(x + chip + pad, y, label, size=BODY, fill=LABEL, anchor="start")
        x += chip + pad + text_w(canvas, label) + gap


# ------------------------------------------------- interpolant / architecture
# the two panels of row d are the figures in 03_inpainting and 02_attention_module,
# redrawn here rather than inlined: same composition and geometry, but this
# figure's palette, node vocabulary (solid / dashed grey / dashed colour), axis
# chrome and 10.2 pt computer modern. 03's layout maths is imported so the plane
# projection and tree placement stay identical to the standalone figure.

BRIDGE_PATH = ROOT.parent / "03_inpainting" / "make_stacked_bridge_inpainting_figure.py"
_spec = importlib.util.spec_from_file_location("stacked_bridge_utils", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bridge
_spec.loader.exec_module(bridge)

PLANE_BOTTOM = ((150, 500), (808, 450), (1090, 664), (432, 718))
PLANE_TOP = ((198, 152), (856, 102), (1138, 316), (480, 370))
PLANE_MID = ((174, 326), (832, 276), (1114, 490), (456, 544))
# the drawn extent of the three planes in 03's coordinates, which is what gets
# mapped into this panel. 03's own axes are dropped and redrawn in this figure's
# style, so they are not part of the box.
ART = (150, 102, 1138, 718)

NOISE = "#8d80e6"
NOISE_SOFT = "#f2eeff"


def noise_node(canvas: Canvas, x: float, y: float, r: float, seed: int = 0) -> None:
    """s = 0: a pale disc with a few draws in it, for a cell that is still noise."""
    canvas.ellipse(x, y, r * 1.32, r * 0.76, fill=NOISE_SOFT,
                   stroke=util.blend(NOISE, util.PAPER, 0.28), stroke_width=1.1)
    rng = random.Random(seed)
    for _ in range(3):
        canvas.circle(x + rng.gauss(0, r * 0.44), y + rng.gauss(0, r * 0.26), r * 0.2,
                      fill=util.blend(util.PURPLE, util.PAPER, 0.3), stroke="none")


def s_label(canvas: Canvas, x: float, y: float, digit: str, anchor: str = "end") -> None:
    """italic s, upright rule and digit -- the split time_axis already makes for t.
    each piece is placed on its own, with tex's thick space (5/18 em) either side of
    the rule: as one " = 0" string, svg drops the leading space and the s touches."""
    gap = BODY * 5 / 18
    parts = [("s", "italic"), ("=", "regular"), (digit, "regular")]
    widths = [text_w(canvas, v, BODY, style) for v, style in parts]
    x0 = x - (sum(widths) + 2 * gap) if anchor == "end" else x
    for (v, style), w in zip(parts, widths):
        canvas.text(x0, y, v, size=BODY, fill=LABEL, anchor="start", style=style)
        x0 += w + gap


def numbers(value: str) -> list[tuple[float, float]]:
    n = [float(v) for v in re.findall(r"-?\d+\.?\d*", value)]
    return list(zip(n[0::2], n[1::2]))


def bridge_panel(canvas: Canvas, x: float, y: float, w: float) -> tuple[float, float]:
    """03_inpainting: one bridge lifts an entire tree from noise to a sample.

    the whole tree is one point in the space the bridge runs through, which is what
    the stacked planes show.

    the art is scaled to fill `w` (the s axis takes the left 104 px). the panel
    reports the bottom it reaches and the x of the middle of the drawn
    planes, so the row does not have to guess where its name goes.
    """
    axis_x = x + 104
    ax0, ay0, ax1, ay1 = ART
    k = (w - (axis_x - x) - 14) / (ax1 - ax0)
    ox, oy = axis_x + 14, y

    def P(px: float, py: float) -> tuple[float, float]:
        return ox + (px - ax0) * k, oy + (py - ay0) * k

    def on(quad, node) -> tuple[float, float]:
        return P(*bridge.point_on_quad(quad, node["u"], node["v"]))

    u_margin, v_margin = bridge.parameter_margins_for_clearance(PLANE_BOTTOM, clearance_px=48)
    nodes = bridge.build_tree_nodes(u_margin, v_margin)
    by_id = {n["id"]: n for n in nodes}
    hidden = [n for n in nodes if n["kind"] == "hidden"]
    r = 13.8 * k

    for quad, edged in ((PLANE_BOTTOM, True), (PLANE_MID, False), (PLANE_TOP, True)):
        canvas.polygon([P(*pt) for pt in quad], fill=PLANE,
                       stroke=RULE if edged else "none", stroke_width=1.3)

    # every hidden cell carries a few sample paths, drawn before the nodes so the
    # discs and circles sit on top of them.
    for node in hidden:
        for i in range(4):
            start = bridge.source_sample_point(bridge.point_on_quad(PLANE_BOTTOM, node["u"], node["v"]),
                                               node["seed"] + 500 + i * 101)
            end = bridge.source_sample_point(bridge.point_on_quad(PLANE_TOP, node["u"], node["v"]),
                                             node["seed"] + 1300 + i * 149, sigma_x=4.0, sigma_y=2.5)
            path = bridge.brownian_bridge_path(start, end, node["seed"] + 400 + i * 211,
                                               steps=64, sigma_x=22, sigma_y=17)
            canvas.polyline([P(*pt) for pt in numbers(path)],
                            stroke=util.blend(util.PURPLE, util.PAPER, 0.8), stroke_width=1.1)

    for quad, sampled in ((PLANE_BOTTOM, False), (PLANE_TOP, True)):
        for node in nodes:
            if not node["parent"]:
                continue
            parent = by_id[node["parent"]]
            mid_u = (parent["u"] + node["u"]) / 2
            p0 = P(*bridge.point_on_quad(quad, parent["u"], parent["v"]))
            c1 = P(*bridge.point_on_quad(quad, mid_u, parent["v"]))
            c2 = P(*bridge.point_on_quad(quad, mid_u, node["v"]))
            p1 = P(*bridge.point_on_quad(quad, node["u"], node["v"]))
            d = (f"M {p0[0]:.1f} {p0[1]:.1f} C {c1[0]:.1f} {c1[1]:.1f}, "
                 f"{c2[0]:.1f} {c2[1]:.1f}, {p1[0]:.1f} {p1[1]:.1f}")
            canvas.path(d, util.cubic_points(p0, c1, c2, p1, steps=18),
                        stroke=EDGE, stroke_width=2.4)
        for node in nodes:
            cx, cy = on(quad, node)
            if node["kind"] == "observed":
                util.draw_node(canvas, Node(node["id"], None, cx, cy, node["color"], "observed"), r)
            elif sampled:
                canvas.circle(cx, cy, r, fill=util.blend(node["color"], util.PAPER, 0.55),
                              stroke=node["color"], stroke_width=2.2, dash="3 4")
            else:
                noise_node(canvas, cx, cy, r, seed=node["seed"])

    # bridge time, as this figure draws a time axis: a rule, two ticks, no name.
    # s runs from the data (top, s = 0) to the noise (bottom, s = 1).
    top_y, bot_y = P(*PLANE_TOP[0])[1], P(*PLANE_BOTTOM[0])[1]
    canvas.line(axis_x, top_y, axis_x, bot_y, AXIS, 1.4)
    for yy, digit in ((top_y, "0"), (bot_y, "1")):
        canvas.line(axis_x - 8, yy, axis_x + 8, yy, AXIS, 1.4)
        s_label(canvas, axis_x - 22, yy, digit)

    # tree time runs across each plane; the rule under the lower plane says so
    # without repeating the slanted axis of the standalone figure.
    ty = P(ax0, ay1)[1] + 34
    tx0, tx1 = P(*PLANE_BOTTOM[0])[0], P(*PLANE_BOTTOM[2])[0]
    canvas.line(tx0, ty, tx1, ty, AXIS, 1.4)
    for xx in (tx0, tx1):
        canvas.line(xx, ty - 8, xx, ty + 8, AXIS, 1.4)
    # four generations sit on each plane, so the ends of the rule are t0 and t3 --
    # the same clock the other rows label. no interior ticks: the manuscript tree
    # is lopsided, so its generations do not land on a lattice in this projection.
    t_label(canvas, tx0 + 26, ty + 28, 0)
    t_label(canvas, tx1 - 26, ty + 28, 3)
    canvas.text((tx0 + tx1) / 2, ty + 28, "tree time", size=BODY, fill=LABEL, anchor="middle")
    return ty + 28 + BODY / 2, (tx0 + tx1) / 2


# ---------------------------------------------------------------- attention

ATTN_TREE = [
    ("r", None, 0.50, 0.00, util.PURPLE),
    ("a", "r", 0.24, 0.46, util.PURPLE),
    ("b", "r", 0.76, 0.46, util.PURPLE),
    ("a1", "a", 0.05, 1.00, util.RED),
    ("a2", "a", 0.42, 1.00, util.ORANGE),
    ("b1", "b", 0.60, 1.00, util.GREEN),
    ("b2", "b", 0.96, 1.00, util.TEAL),
]
CONDITION = ("a1", "b1")      # the observed cells, ringed in the tree and on the axes
QUERY_ROW = "a"               # the row whose attention the lower matrix reads out

DECAY = 0.52                  # what one more hop along the tree costs the bias
GAMMA = 4.0                   # weight on the bias where the two terms are added: large
                              # enough that the tree sets the shape of the sum (one hop
                              # costs GAMMA * (1 - DECAY) = 1.9, more than the state
                              # term ever contributes) while the state term still shows.
STATE_SPREAD = 0.13           # falloff of the state-state term in state units. narrow,
                              # so the term stays peaked and remains visible in the sum.
# a state coordinate per cell, not following the tree, so the state and bias terms
# show different structure.
STATE = {"r": 0.52, "a": 0.18, "b": 0.60, "a1": 0.95, "a2": 0.22, "b1": 0.56, "b2": 0.06}

RING_W = 2.4                  # the rings in the tree: the observed cells and the query
TICK_RING_W = 1.6             # the same two rings, smaller, on a matrix axis
BOX_W = 1.6                   # the row and column a matrix is there to be read along


def tree_distance(edges: list[tuple[str, str | None]]) -> dict[str, dict[str, int]]:
    """hops along the undirected tree -- the d_tree that biases attention."""
    adj: dict[str, list[str]] = {a: [] for a, _ in edges}
    for child, parent in edges:
        if parent:
            adj[child].append(parent)
            adj[parent].append(child)
    out = {}
    for source in adj:
        dist, frontier = {source: 0}, [source]
        while frontier:
            nxt = []
            for nid in frontier:
                for other in adj[nid]:
                    if other not in dist:
                        dist[other] = dist[nid] + 1
                        nxt.append(other)
            frontier = nxt
        out[source] = dist
    return out


IDS = [nid for nid, *_ in ATTN_TREE]
COLORS = {nid: c for nid, _, _, _, c in ATTN_TREE}
PARENTS = {par for _, par, *_ in ATTN_TREE if par}
DIST = tree_distance([(nid, par) for nid, par, *_ in ATTN_TREE])


def bias_scores() -> list[list[float]]:
    return [[DECAY ** DIST[i][j] for j in IDS] for i in IDS]


def state_scores() -> list[list[float]]:
    """states only, with no reference to the tree: what a plain attention layer over
    these cells would do."""
    return [[math.exp(-abs(STATE[i] - STATE[j]) / STATE_SPREAD) for j in IDS]
            for i in IDS]


def biased_scores() -> list[list[float]]:
    """gamma * bias + state-state, which is what the softmax of the layer sees."""
    return [[GAMMA * b + s for b, s in zip(brow, srow)]
            for brow, srow in zip(bias_scores(), state_scores())]


# floor and gain for `ink`. the sum matrix gets a wider range than the two terms it is
# made of, so the query row's crossings with the two observed columns (one hop and
# three hops away) are distinguishable. the gain stops short of saturating the cells
# around the diagonal.
TERM_INK = (0.12, 2.6)
BIASED_INK = (0.05, 4.0)


def ink(scores: list[list[float]], floor: float = TERM_INK[0],
        gain: float = TERM_INK[1]) -> list[list[float]]:
    """row-normalised scores, mapped onto how dark a cell gets drawn."""
    return [[min(1.0, floor + gain * v / sum(row)) for v in row] for row in scores]


def bias_ink() -> list[list[float]]:
    """floored higher than `ink`: the bias has no softmax to normalise it, and the
    far cells should stay visible rather than drop out to paper."""
    return [[0.18 + 0.82 * v for v in row] for row in bias_scores()]


def grid(canvas: Canvas, x: float, y: float, cell: float, values: list[list[float]],
         color: str) -> None:
    pad = cell * 0.16
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            canvas.rect(x + j * cell + pad / 2, y + i * cell + pad / 2,
                        cell - pad, cell - pad, rx=cell * 0.126,
                        fill=util.blend(color, util.PAPER, 1 - v), stroke="none")


def matrix(canvas: Canvas, mx: float, top: float, cell: float,
           values: list[list[float]], color: str) -> None:
    """one matrix, keyed on both axes by the cells of the tree, with the two
    readings it is there to make outlined: the observed columns every query can see
    however far away they sit, and one query's row. all of it scales off `cell`."""
    span, dot, off, ring = cell * len(IDS), cell * 0.232, cell * 0.842, cell * 0.5
    for j, nid in enumerate(IDS):
        canvas.circle(mx + (j + 0.5) * cell, top - off, dot, fill=COLORS[nid], stroke="none")
        if nid in CONDITION:
            canvas.circle(mx + (j + 0.5) * cell, top - off, ring,
                          stroke=util.PURPLE_DARK, stroke_width=TICK_RING_W)
    for i, nid in enumerate(IDS):
        canvas.circle(mx - off, top + (i + 0.5) * cell, dot, fill=COLORS[nid], stroke="none")
        if nid == QUERY_ROW:
            canvas.circle(mx - off, top + (i + 0.5) * cell, ring, stroke=GOLD,
                          stroke_width=TICK_RING_W)
    grid(canvas, mx, top, cell, values, color)
    for nid in CONDITION:
        j = IDS.index(nid)
        canvas.rect(mx + j * cell - 2, top - 4, cell + 4, span + 8, rx=4,
                    stroke=util.PURPLE_DARK, stroke_width=BOX_W)
    i = IDS.index(QUERY_ROW)
    canvas.rect(mx - 4, top + i * cell - 2, span + 8, cell + 4, rx=4, stroke=GOLD,
                stroke_width=BOX_W)


def attn_tree(canvas: Canvas, x: float, y: float, w: float, h: float, r: float = 10) -> None:
    """the tree every matrix here is indexed by. the leaves are drawn with solid
    outlines: each one is a cell the layer attends over. the internal cells keep the
    dashed outline the rest of the figure gives an unobserved cell, and the rings
    mark the two observed cells and the one query whose row is read out.

    the line weights scale with `r` so the drawing can be shrunk, but the rings do
    not: they are marks on the drawing rather than part of it, and they still have
    to carry at a smaller size.
    """
    k = r / 10
    pos = {nid: (x + ux * w, y + uy * h) for nid, _, ux, uy, _ in ATTN_TREE}
    for nid, par, *_ in ATTN_TREE:
        if par:
            d, pts = util.edge_path(Node(nid, par, *pos[par], INK, "blank"),
                                    Node(nid, par, *pos[nid], INK, "blank"))
            canvas.path(d, pts, stroke=EDGE, stroke_width=2.5 * k)
    for nid, *_ in ATTN_TREE:
        cx, cy = pos[nid]
        if nid in CONDITION:
            util.draw_node(canvas, Node(nid, None, cx, cy, COLORS[nid], "observed"), r)
        else:
            canvas.circle(cx, cy, r, fill=util.blend(COLORS[nid], util.PAPER, 0.55),
                          stroke=COLORS[nid], stroke_width=2.2 * k,
                          dash="3 4" if nid in PARENTS else None)
        if nid in CONDITION:
            canvas.circle(cx, cy, r * 1.7, stroke=util.PURPLE_DARK, stroke_width=RING_W)
        if nid == QUERY_ROW:
            canvas.circle(cx, cy, r * 1.7, stroke=GOLD, stroke_width=RING_W)


NAME_GAP = 20                  # matrix right edge to the name beside it
SUB_GAP = 6                    # subscript to the word after it
BIAS_NAME = "d_tree"           # composed by draw_bias_name rather than set as lines
ATTN_NAME = ("state–state", "attention")
BIASED_NAME = ("biased", "attention")
ATTN_CELL = 19.0               # solved so both halves of the row bottom out together
BIASED = util.blend(util.PURPLE, util.RED, 0.5)   # the sum, in the sum of its colours


def bias_name_w(canvas: Canvas) -> float:
    return (text_w(canvas, "d", BODY, "italic") * 1.05 + text_w(canvas, "tree", TINY)
            + SUB_GAP + text_w(canvas, "bias", BODY))


def draw_bias_name(canvas: Canvas, x: float, y: float) -> None:
    """d with a roman `tree` subscript, as time_axis composes t_i -- the patristic
    distance the bias is a function of, named the way the text names it."""
    dx = text_w(canvas, "d", BODY, "italic") * 1.05
    canvas.text(x, y, "d", size=BODY, fill=LABEL, anchor="start", style="italic")
    canvas.text(x + dx, y + 11, "tree", size=TINY, fill=LABEL, anchor="start")
    canvas.text(x + dx + text_w(canvas, "tree", TINY) + SUB_GAP, y, "bias",
                size=BODY, fill=LABEL, anchor="start")


def draw_name(canvas: Canvas, x: float, y: float, name: str | tuple[str, ...]) -> None:
    if name == BIAS_NAME:
        draw_bias_name(canvas, x, y)
        return
    for i, line in enumerate(name):
        canvas.text(x, y + (i - (len(name) - 1) / 2) * 40, line, size=BODY,
                    fill=LABEL, anchor="start")


def name_w(canvas: Canvas, name: str | tuple[str, ...]) -> float:
    if name == BIAS_NAME:
        return bias_name_w(canvas)
    return max(text_w(canvas, line) for line in name)


def attention_panel(canvas: Canvas, x: float, y: float, w: float) -> float:
    """02_attention_module: the module is a transformer over the cells of one tree.

    left, the tree with its observed cells ringed; right, the two terms the layer
    adds: the tree-distance bias and the state-state attention. the outlines pick
    out the observed columns and the one query whose row is read out. `y` is the top
    of the `keys` label; the bottom of the lower matrix comes back.
    """
    cell, span = ATTN_CELL, ATTN_CELL * len(IDS)
    tops = [y + 46, y + 46 + span + 52]
    # both matrices and their names end flush with the panel's right edge; the tree
    # takes what is left, minus the gutter the rotated `queries` label sits in.
    names = (BIAS_NAME, ATTN_NAME)
    mx = x + w - max(name_w(canvas, n) for n in names) - NAME_GAP - span
    # the column beside the tree is taller than its two grids: the `keys` label
    # caps it and the highlighted row's outline floors it, and the tree is centred
    # on that column.
    col_top, col_bot = tops[0] - 46 - BODY / 2, tops[1] + span + 8
    th = 240
    attn_tree(canvas, x, (col_top + col_bot) / 2 - th / 2, mx - 62 - x, th)

    values = (bias_ink(), ink(state_scores()))
    for top, vals, color, name in zip(tops, values, (util.PURPLE, util.RED), names):
        matrix(canvas, mx, top, cell, vals, color)
        draw_name(canvas, mx + span + NAME_GAP, top + span / 2, name)

    canvas.text(mx + span / 2, tops[0] - 46, "keys", size=BODY, fill=LABEL, anchor="middle")
    canvas.vtext(mx - 46, (tops[0] + tops[1] + span) / 2, "queries", size=BODY, fill=LABEL)
    return tops[1] + span


QUAD_CELL = 18.0               # solved so both halves of the row bottom out together
QUAD_VGAP = 30                 # clear paper between the two matrices of the left column
# the gap between the two column blocks is whatever the panel width leaves once the
# three matrices and their names are placed: the left block is set from the panel's
# left edge, the right one from its right edge. row_architecture picks that width so
# the gap lands near 38, which keeps the four quadrants reading as one block rather
# than as two panels that happen to share a title.
QUAD_GUTTER = 62               # the rotated `queries` label, plus the axis ticks
TREE_R = 9.0                   # a smaller drawing than the two-panel tree
TREE_ASPECT = 1.7              # wider than that tree too: the quadrant is a wide one
TREE_PAD = 6                   # paper between the outer rings and the quadrant edge


def attention_quad(canvas: Canvas, x: float, y: float, w: float) -> float:
    """the same layer in four quadrants, with the attention it ends up using.

    the d_tree bias top left, the state-state attention under it, their weighted sum
    bottom right, and top right the tree all three are indexed by. a quadrant here
    is a matrix and the name beside it, so the two column blocks are the quadrants
    and the tree is centred in the one the bias matrix bounds: same rows, same
    column. `y` is the top of the `keys` label; the bottom of the lower row of
    matrices comes back.
    """
    cell, span = QUAD_CELL, QUAD_CELL * len(IDS)
    tick = cell * 1.342            # matrix edge to the outside of its axis rings
    tops = [y + 46, y + 46 + span + QUAD_VGAP + tick]
    lmx = x + QUAD_GUTTER
    rmx = x + w - name_w(canvas, BIASED_NAME) - NAME_GAP - span

    for mx, top, vals, color, name in (
            (lmx, tops[0], bias_ink(), util.PURPLE, BIAS_NAME),
            (lmx, tops[1], ink(state_scores()), util.RED, ATTN_NAME),
            (rmx, tops[1], ink(biased_scores(), *BIASED_INK), BIASED, BIASED_NAME)):
        matrix(canvas, mx, top, cell, vals, color)
        draw_name(canvas, mx + span + NAME_GAP, top + span / 2, name)

    # the quadrant the tree gets: the column is the sum matrix and its name, not the
    # matrix alone, or the tree sits left of where the quadrant reads and leaves the
    # name with paper above it. the rows run from the cap of the `keys` label to the
    # foot of the bias matrix, which is the whole of the quadrant to fill. height sets
    # the drawing and the aspect sets the width, so resizing the tree is one number.
    row, col = (tops[0] - 46 - BODY / 2, tops[0] + span), (rmx - tick, x + w)
    th = (row[1] - row[0]) - 2 * (TREE_R * 1.7 + TREE_PAD)
    tw = th * TREE_ASPECT
    attn_tree(canvas, (col[0] + col[1] - tw) / 2, (row[0] + row[1] - th) / 2,
              tw, th, TREE_R)

    canvas.text(lmx + span / 2, tops[0] - 46, "keys", size=BODY, fill=LABEL, anchor="middle")
    canvas.vtext(lmx - 46, (tops[0] + tops[1] + span) / 2, "queries", size=BODY, fill=LABEL)
    return tops[1] + span


# ---------------------------------------------------------------- rows

def row_model(canvas: Canvas, y0: float) -> float:
    """box top at y0; returns the box bottom."""
    letter_y = y0 + 44
    canvas.letter(LEFT, y0 + LETTER_TOP, "b")
    legend(canvas, BOX_X1 - 32, letter_y)

    top, bot = y0 + 106, y0 + 106 + BAND
    mid = (top + bot) / 2
    base = model_tree()

    src = place(base, LEFT + 14, 260, top, bot)
    slice_band(canvas, src)
    draw_tree(canvas, src, r=10, edge_width=2.8,
              dashed=frozenset(n.id for n in src if n.kind == "observed"))

    ax1, ax2 = LEFT + 310, LEFT + 700
    arrow(canvas, ax1, ax2, mid)
    for i, line in enumerate(["sample from", "generative model"]):
        canvas.text((ax1 + ax2) / 2, mid + 32 + i * 40, line, size=BODY,
                    fill=LABEL, anchor="middle")

    rng = random.Random(3)
    left = LEFT + 740
    right, deck_bot = deck(canvas, [fill_states(base, rng) for _ in range(4)],
                           left, top, bot, 340)
    cdots(canvas, right + 44, top + BAND * 2 / 3 + 48)

    cap_y = deck_bot + 86        # clear of the brace cusp, which hangs lower than the spine
    # the input sits a whole deck-descent higher than the brace, so its label goes
    # under the tree rather than on the brace's baseline.
    caption(canvas, LEFT + 144, bot + 48, "observed snapshot")
    brace(canvas, left - 16, right + 68, deck_bot + 12)
    caption(canvas, (left - 16 + right + 68) / 2, cap_y,
            "possible complete trees")
    return cap_y + 32


def row_data(canvas: Canvas, y0: float) -> float:
    canvas.letter(LEFT, y0 + LETTER_TOP, "a")

    top, bot = y0 + 86, y0 + 86 + BAND
    time_axis(canvas, LEFT + 74, top, bot)
    for stage, tx, tw in [(1, LEFT + 134, 210), (2, LEFT + 499, 370), (3, LEFT + 1004, 440)]:
        nodes = place(snapshot_tree(stage), tx, tw, top, bot)
        slice_band(canvas, nodes)
        draw_tree(canvas, nodes, r=10)
        caption(canvas, tx + tw / 2, bot + 46, f"embryo {stage}")
    return bot + 78


def row_fitting(canvas: Canvas, y0: float) -> float:
    canvas.letter(LEFT, y0 + LETTER_TOP, "c")

    top, bot = y0 + 86, y0 + 86 + BAND
    mid = (top + bot) / 2
    time_axis(canvas, LEFT + 74, top, bot)

    base = snapshot_tree(3)
    filled = fill_states(base, random.Random(11))
    masked = [replace(n, kind="observed" if n.id in KEPT else "blank") for n in filled]

    tw, slots = 310, [LEFT + 120, LEFT + 620, LEFT + 1120]
    for nodes, slot, band in zip((base, filled, masked), slots, (True, False, True)):
        placed = place(nodes, slot, tw, top, bot)
        if band:
            slice_band(canvas, placed, pad=20, height=28)
        draw_tree(canvas, placed, r=9.5, edge_width=2.5)

    a1, a2 = slots[0] + tw + 32, slots[1] - 32
    arrow(canvas, a1, a2, mid)
    canvas.text((a1 + a2) / 2, mid - 34, "impute", size=BODY, fill=LABEL, anchor="middle")

    b1, b2 = slots[1] + tw + 32, slots[2] - 32
    arrow(canvas, b1, b2, mid - 24)
    arrow(canvas, b2, b1, mid + 24, color=util.PURPLE_DARK)
    canvas.text((b1 + b2) / 2, mid - 58, "mask", size=BODY, fill=LABEL, anchor="middle")
    canvas.text((b1 + b2) / 2, mid + 58, "resample", size=BODY, fill=util.PURPLE_DARK,
                anchor="middle")

    for slot, label in zip(slots, ["observed snapshot", "complete tree",
                                   "condition on one snapshot"]):
        caption(canvas, slot + tw / 2, bot + 46, label)
    return bot + 78


def row_architecture(canvas: Canvas, y0: float, quad: bool = False) -> float:
    """the two figures this model is built out of, in this figure's own hand.

    each half is named once, on the panel-letter line, underlined and a step below
    body size so the name reads as a sub-heading of the box title rather than as
    another label in the drawing. the drawings set the column widths -- the bridge
    art at its fixed aspect, the attention block at what its matrices and their
    names need -- and each name is centred over the art it belongs to.
    """
    titles = ("stochastic interpolants over complete trees", "lineage-biased attention layer")
    title_y = y0 + 46
    canvas.letter(LEFT, y0 + LETTER_TOP, "d")

    # the bridge art is narrow: its aspect is fixed, so filling half the row would
    # make this row taller than the others. the split is solved so the two halves
    # bottom out together; the quad layout fits three matrices instead of two with
    # a narrower art and a smaller cell.
    art_w, attn_w = (563, 712) if quad else (588, 722)
    ax, bx = LEFT + 75, BOX_X1 - 32 - attn_w    # clear of the letter; flush right
    top = y0 + 86
    art_bot, art_mid = bridge_panel(canvas, ax, top, art_w)
    panel = attention_quad if quad else attention_panel
    attn_bot = panel(canvas, bx, top + 4, attn_w)

    subtitle(canvas, art_mid, title_y, titles[0])
    subtitle(canvas, bx + attn_w / 2, title_y, titles[1])
    return max(art_bot, attn_bot) + 18


# ---------------------------------------------------------------- output

def save(canvas: Canvas, stem: str) -> None:
    height_in = PAPER_WIDTH_IN * canvas.height / canvas.width
    canvas.image.resize((canvas.width, canvas.height), Image.Resampling.LANCZOS).save(
        ROOT / f"{stem}.png")
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{PAPER_WIDTH_IN}in" height="{height_in:.4f}in" viewBox="0 0 {canvas.width} {canvas.height}" role="img" aria-label="Problem setup, approach 2">
<defs>
{chr(10).join(canvas.svg_defs)}
</defs>
{chr(10).join(canvas.svg_body)}
</svg>
"""
    (ROOT / f"{stem}.svg").write_text(svg, encoding="utf-8")
    import cairosvg
    cairosvg.svg2pdf(url=str(ROOT / f"{stem}.svg"), write_to=str(ROOT / f"{stem}.pdf"))
    (ROOT / f"{stem}.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Problem setup, approach 2</title>
<style>body {{ margin: 0; background: #ffffff; }} .figure {{ width: {canvas.width}px; }}</style>
</head>
<body><div class="figure">
{svg}
</div></body>
</html>
""",
        encoding="utf-8",
    )
    print(ROOT / f"{stem}.svg")
    print(ROOT / f"{stem}.png")
    print(ROOT / f"{stem}.pdf")


def build(quad: bool, height: int) -> Canvas:
    canvas = Canvas(W, height)
    canvas.rect(0, 0, W, height, fill=util.PAPER)
    y = 22
    for row, title in ((row_data, "observed data"), (row_model, "modeling objective"),
                       (row_fitting, "model fitting"),
                       (row_architecture, "model implementation")):
        bottom = row(canvas, y, quad) if row is row_architecture else row(canvas, y)
        group_box(canvas, y, bottom, title)
        y = bottom + 30
    end = y - 30 + 22
    if abs(end - height) > 2:
        print(f"note: canvas height {height}, content ends at {end:.0f}")
    return canvas


def main() -> None:
    """both layouts of row d: the two-matrix panel, and the four-quadrant one that
    also draws the attention the layer ends up using."""
    save(build(False, H), FIG_STEM)
    save(build(True, H_QUAD), QUAD_STEM)


if __name__ == "__main__":
    main()
