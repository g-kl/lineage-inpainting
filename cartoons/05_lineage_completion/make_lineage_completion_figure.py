#!/usr/bin/env python3
"""Create the Figure 05 lineage-completion schematic.

It writes an editable SVG and a high-resolution PNG from the same geometry
using only Pillow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import math
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_SVG = ROOT / "lineage_completion_figure.svg"
OUT_PNG = ROOT / "lineage_completion_figure.png"
OUT_HTML = ROOT / "lineage_completion_figure.html"

W, H = 1800, 2180
SCALE = 2
PAPER_WIDTH_IN = 7.5
PAPER_HEIGHT_IN = PAPER_WIDTH_IN * H / W

# the repo's shared faces (utils/fonts), same ones utils.figure_labels registers, so
# these cartoons match the computed figures.
FONT_DIR = ROOT.parents[1] / "utils" / "fonts"
FONT_REG = FONT_DIR / "lmsans10-regular.otf"
FONT_BOLD = FONT_DIR / "lmsans10-bold.otf"
# no oblique face ships with lmsans10; PIL measures with the regular face and the svg
# still carries font-style: italic, so the renderer synthesises the slant.
FONT_ITALIC = FONT_REG
FONT_FAMILY = "Latin Modern Sans, lmsans10, CMU Sans Serif, sans-serif"

INK = "#202124"
MUTED = "#667085"
PAPER = "#ffffff"
PANEL = "#ffffff"
GROUP = "#f6f7fa"
EDGE = "#d7dce4"
AXIS = "#a8b0bd"
PURPLE = "#6b5ed8"
PURPLE_DARK = "#5146a6"
PURPLE_SOFT = "#f1edff"
GOLD = "#d89436"
GOLD_SOFT = "#fff1df"
TEAL = "#2f8fa4"
GREEN = "#4f9a66"
RED = "#d65a3b"
ORANGE = "#e89932"
OLIVE = "#b2a637"
BLUE = "#327fa6"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def rgba(color: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    r, g, b = hex_to_rgb(color)
    return r, g, b, int(round(255 * opacity))


def blend(color: str, target: str = "#ffffff", t: float = 0.5) -> str:
    a = hex_to_rgb(color)
    b = hex_to_rgb(target)
    vals = [round(a[i] + (b[i] - a[i]) * t) for i in range(3)]
    return "#" + "".join(f"{v:02x}" for v in vals)


def cubic_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 48,
) -> list[tuple[float, float]]:
    pts = []
    for i in range(steps + 1):
        t = i / steps
        omt = 1 - t
        x = (
            omt**3 * p0[0]
            + 3 * omt**2 * t * p1[0]
            + 3 * omt * t**2 * p2[0]
            + t**3 * p3[0]
        )
        y = (
            omt**3 * p0[1]
            + 3 * omt**2 * t * p1[1]
            + 3 * omt * t**2 * p2[1]
            + t**3 * p3[1]
        )
        pts.append((x, y))
    return pts


@dataclass(frozen=True)
class Node:
    id: str
    parent: str | None
    x: float
    y: float
    color: str
    kind: str


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.image = Image.new("RGBA", (width * SCALE, height * SCALE), PAPER)
        self.draw = ImageDraw.Draw(self.image)
        self.svg_defs: list[str] = []
        self.svg_body: list[str] = []
        self.patterns: dict[str, str] = {}
        self.fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def font(self, size: int, style: str = "regular") -> ImageFont.FreeTypeFont:
        key = (style, size)
        if key not in self.fonts:
            path = FONT_REG
            if style == "bold":
                path = FONT_BOLD
            elif style == "italic":
                path = FONT_ITALIC
            if not path.exists():
                raise FileNotFoundError(
                    f"missing font {path}. the cartoons use the shared faces in "
                    f"utils/fonts/ -- see utils/figure_labels.py")
            self.fonts[key] = ImageFont.truetype(str(path), size * SCALE)
        return self.fonts[key]

    def xy(self, x: float, y: float) -> tuple[float, float]:
        return x * SCALE, y * SCALE

    def wh(self, w: float, h: float) -> tuple[float, float]:
        return w * SCALE, h * SCALE

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        rx: float = 0,
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        bbox = (*self.xy(x, y), *self.xy(x + w, y + h))
        fill_rgba = None if fill == "none" else rgba(fill, opacity)
        outline_rgba = None if stroke == "none" else rgba(stroke, opacity)
        if rx:
            self.draw.rounded_rectangle(
                bbox,
                radius=rx * SCALE,
                fill=fill_rgba,
                outline=outline_rgba,
                width=max(1, round(stroke_width * SCALE)),
            )
        else:
            self.draw.rectangle(
                bbox,
                fill=fill_rgba,
                outline=outline_rgba,
                width=max(1, round(stroke_width * SCALE)),
            )
        attrs = {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rx": rx if rx else None,
            "fill": fill,
            "stroke": stroke,
            "stroke-width": stroke_width if stroke != "none" else None,
            "opacity": opacity if opacity != 1 else None,
            "stroke-dasharray": dash,
        }
        self.svg_body.append(self.tag("rect", attrs))

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = INK,
        stroke_width: float = 1.0,
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        pts = [self.xy(x1, y1), self.xy(x2, y2)]
        self.draw.line(
            pts,
            fill=rgba(stroke, opacity),
            width=max(1, round(stroke_width * SCALE)),
        )
        attrs = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "stroke": stroke,
            "stroke-width": stroke_width,
            "stroke-linecap": "round",
            "opacity": opacity if opacity != 1 else None,
            "stroke-dasharray": dash,
        }
        self.svg_body.append(self.tag("line", attrs))

    def polyline(
        self,
        points: list[tuple[float, float]],
        stroke: str,
        stroke_width: float,
        fill: str = "none",
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        scaled = [self.xy(x, y) for x, y in points]
        if fill != "none":
            self.draw.polygon(scaled, fill=rgba(fill, opacity))
        self.draw.line(
            scaled,
            fill=rgba(stroke, opacity),
            width=max(1, round(stroke_width * SCALE)),
            joint="curve",
        )
        attrs = {
            "points": " ".join(f"{x:.1f},{y:.1f}" for x, y in points),
            "fill": fill,
            "stroke": stroke,
            "stroke-width": stroke_width,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "opacity": opacity if opacity != 1 else None,
            "stroke-dasharray": dash,
        }
        self.svg_body.append(self.tag("polyline", attrs))

    def path(
        self,
        d: str,
        points: list[tuple[float, float]],
        stroke: str,
        stroke_width: float,
        fill: str = "none",
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        scaled = [self.xy(x, y) for x, y in points]
        if len(scaled) > 1:
            if dash:
                self.dashed_polyline_png(scaled, stroke, stroke_width, opacity)
            else:
                self.draw.line(
                    scaled,
                    fill=rgba(stroke, opacity),
                    width=max(1, round(stroke_width * SCALE)),
                    joint="curve",
                )
        attrs = {
            "d": d,
            "fill": fill,
            "stroke": stroke,
            "stroke-width": stroke_width,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "opacity": opacity if opacity != 1 else None,
            "stroke-dasharray": dash,
        }
        self.svg_body.append(self.tag("path", attrs))

    def dashed_polyline_png(
        self,
        points: list[tuple[float, float]],
        stroke: str,
        stroke_width: float,
        opacity: float,
        dash_len: float = 11.0,
        gap_len: float = 9.0,
    ) -> None:
        width = max(1, round(stroke_width * SCALE))
        dash_len *= SCALE
        gap_len *= SCALE
        phase = 0.0
        drawing = True
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            dx, dy = x1 - x0, y1 - y0
            seg_len = math.hypot(dx, dy)
            if seg_len == 0:
                continue
            ux, uy = dx / seg_len, dy / seg_len
            pos = 0.0
            while pos < seg_len:
                run = (dash_len if drawing else gap_len) - phase
                end = min(seg_len, pos + run)
                if drawing:
                    self.draw.line(
                        [(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)],
                        fill=rgba(stroke, opacity),
                        width=width,
                    )
                phase += end - pos
                if (drawing and phase >= dash_len) or ((not drawing) and phase >= gap_len):
                    phase = 0.0
                    drawing = not drawing
                pos = end

    def circle(
        self,
        x: float,
        y: float,
        r: float,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        bbox = (*self.xy(x - r, y - r), *self.xy(x + r, y + r))
        fill_rgba = None if fill == "none" else rgba(fill, opacity)
        outline_rgba = None if stroke == "none" else rgba(stroke, opacity)
        if dash:
            if fill_rgba:
                self.draw.ellipse(bbox, fill=fill_rgba)
            self.dashed_circle_png(x, y, r, stroke, stroke_width, opacity)
        else:
            self.draw.ellipse(
                bbox,
                fill=fill_rgba,
                outline=outline_rgba,
                width=max(1, round(stroke_width * SCALE)),
            )
        attrs = {
            "cx": x,
            "cy": y,
            "r": r,
            "fill": fill,
            "stroke": stroke,
            "stroke-width": stroke_width if stroke != "none" else None,
            "opacity": opacity if opacity != 1 else None,
            "stroke-dasharray": dash,
        }
        self.svg_body.append(self.tag("circle", attrs))

    def dashed_circle_png(
        self,
        x: float,
        y: float,
        r: float,
        stroke: str,
        stroke_width: float,
        opacity: float,
    ) -> None:
        bbox = (*self.xy(x - r, y - r), *self.xy(x + r, y + r))
        for start in range(0, 360, 36):
            self.draw.arc(
                bbox,
                start=start,
                end=start + 20,
                fill=rgba(stroke, opacity),
                width=max(1, round(stroke_width * SCALE)),
            )

    def polygon(
        self,
        points: list[tuple[float, float]],
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        scaled = [self.xy(x, y) for x, y in points]
        self.draw.polygon(scaled, fill=rgba(fill, opacity))
        if stroke != "none":
            self.draw.line(
                scaled + [scaled[0]],
                fill=rgba(stroke, opacity),
                width=max(1, round(stroke_width * SCALE)),
            )
        attrs = {
            "points": " ".join(f"{x:.1f},{y:.1f}" for x, y in points),
            "fill": fill,
            "stroke": stroke,
            "stroke-width": stroke_width if stroke != "none" else None,
            "opacity": opacity if opacity != 1 else None,
        }
        self.svg_body.append(self.tag("polygon", attrs))

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int = 24,
        fill: str = INK,
        anchor: str = "start",
        style: str = "regular",
        opacity: float = 1.0,
    ) -> None:
        font = self.font(size, style)
        sx, sy = self.xy(x, y)
        pil_anchor = {"start": "lm", "middle": "mm", "end": "rm"}[anchor]
        self.draw.text(
            (sx, sy),
            value,
            fill=rgba(fill, opacity),
            font=font,
            anchor=pil_anchor,
        )
        attrs = {
            "x": x,
            "y": y,
            "fill": fill,
            "font-family": FONT_FAMILY,
            "font-size": size,
            "font-weight": "bold" if style == "bold" else None,
            "font-style": "italic" if style == "italic" else None,
            "text-anchor": anchor,
            "dominant-baseline": "middle",
            "opacity": opacity if opacity != 1 else None,
        }
        self.svg_body.append(
            self.tag("text", attrs, html.escape(value, quote=False))
        )

    def time_label(self, x: float, y: float, index: int, anchor: str = "end") -> None:
        offset = 0
        if anchor == "middle":
            offset = -10
        elif anchor == "end":
            offset = -18
        self.text(x + offset, y, "t", 25, MUTED, "start", "italic")
        self.text(x + offset + 14, y + 10, str(index), 17, MUTED, "start")

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = "#9aa3b2",
        stroke_width: float = 2.0,
        opacity: float = 1.0,
    ) -> None:
        self.line(x1, y1, x2, y2, stroke, stroke_width, opacity)
        angle = math.atan2(y2 - y1, x2 - x1)
        length = 13
        width = 7
        tip = (x2, y2)
        left = (
            x2 - length * math.cos(angle) + width * math.sin(angle),
            y2 - length * math.sin(angle) - width * math.cos(angle),
        )
        right = (
            x2 - length * math.cos(angle) - width * math.sin(angle),
            y2 - length * math.sin(angle) + width * math.cos(angle),
        )
        self.polygon([tip, left, right], fill=stroke, opacity=opacity)

    def hatch_pattern(self, color: str) -> str:
        if color in self.patterns:
            return self.patterns[color]
        ident = f"hatch-{color.lstrip('#')}"
        self.patterns[color] = ident
        self.svg_defs.append(
            "\n".join(
                [
                    f'<pattern id="{ident}" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">',
                    '<rect x="0" y="0" width="5" height="5" fill="#ffffff"/>',
                    f'<line x1="1.2" y1="0" x2="1.2" y2="5" stroke="{color}" stroke-width="2.2"/>',
                    "</pattern>",
                ]
            )
        )
        return ident

    def hatched_circle(
        self,
        x: float,
        y: float,
        r: float,
        color: str,
        stroke_width: float = 2.1,
        opacity: float = 1.0,
    ) -> None:
        bbox = (*self.xy(x - r, y - r), *self.xy(x + r, y + r))
        self.draw.ellipse(bbox, fill=rgba("#ffffff", opacity), outline=rgba(color, opacity), width=max(1, round(stroke_width * SCALE)))

        mask = Image.new("L", self.image.size, 0)
        md = ImageDraw.Draw(mask)
        md.ellipse(bbox, fill=255)
        hatch = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(hatch)
        step = 8 * SCALE
        span = int((r * 2 + 20) * SCALE)
        cx, cy = self.xy(x, y)
        for offset in range(-span, span + step, step):
            hd.line(
                [(cx - span, cy + offset + span), (cx + span, cy + offset - span)],
                fill=rgba(color, opacity),
                width=max(1, round(1.7 * SCALE)),
            )
        self.image.alpha_composite(Image.composite(hatch, Image.new("RGBA", self.image.size, (0, 0, 0, 0)), mask))

        ident = self.hatch_pattern(color)
        attrs = {
            "cx": x,
            "cy": y,
            "r": r,
            "fill": f"url(#{ident})",
            "stroke": color,
            "stroke-width": stroke_width,
            "opacity": opacity if opacity != 1 else None,
        }
        self.svg_body.append(self.tag("circle", attrs))

    def save(self, svg_path: Path, png_path: Path, html_path: Path) -> None:
        png = self.image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        png.save(png_path)

        defs = "\n".join(self.svg_defs)
        body = "\n".join(self.svg_body)
        svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{PAPER_WIDTH_IN}in" height="{PAPER_HEIGHT_IN:.4f}in" viewBox="0 0 {self.width} {self.height}" role="img" aria-label="Lineage completion schematic">
<defs>
{defs}
</defs>
{body}
</svg>
"""
        svg_path.write_text(svg, encoding="utf-8")
        html_path.write_text(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Lineage completion schematic</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@master/font/Serif/cmun-serif.css">
  <style>
    body {{ margin: 0; background: #ffffff; }}
    .figure {{ width: {self.width}px; height: {self.height}px; }}
  </style>
</head>
<body>
<div class="figure">
{svg}
</div>
</body>
</html>
""",
            encoding="utf-8",
        )

    @staticmethod
    def tag(name: str, attrs: dict[str, object | None], body: str | None = None) -> str:
        clean = []
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, float):
                value = f"{value:.2f}".rstrip("0").rstrip(".")
            clean.append(f'{key}="{esc(value)}"')
        head = f"<{name} {' '.join(clean)}>"
        if body is None:
            return head[:-1] + "/>"
        return f"{head}{body}</{name}>"


def tree_children(nodes: Iterable[Node]) -> dict[str, list[Node]]:
    all_nodes = list(nodes)
    children = {node.id: [] for node in all_nodes}
    by_id = {node.id: node for node in all_nodes}
    for node in all_nodes:
        if node.parent:
            children[node.parent].append(node)
    for kids in children.values():
        kids.sort(key=lambda node: node.x)
    return children


def edge_path(parent: Node, child: Node) -> tuple[str, list[tuple[float, float]]]:
    mid = (parent.y + child.y) / 2
    p0 = (parent.x, parent.y)
    p1 = (parent.x, mid)
    p2 = (child.x, mid)
    p3 = (child.x, child.y)
    d = (
        f"M {p0[0]:.1f} {p0[1]:.1f} "
        f"C {p1[0]:.1f} {p1[1]:.1f}, {p2[0]:.1f} {p2[1]:.1f}, {p3[0]:.1f} {p3[1]:.1f}"
    )
    return d, cubic_points(p0, p1, p2, p3)


def transform_nodes(
    nodes: list[Node],
    x: float,
    y: float,
    w: float,
    h: float,
) -> list[Node]:
    xs = [node.x for node in nodes]
    ys = [node.y for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    out = []
    for node in nodes:
        nx = x + (node.x - min_x) / (max_x - min_x) * w
        ny = y + (node.y - min_y) / (max_y - min_y) * h
        out.append(Node(node.id, node.parent, nx, ny, node.color, node.kind))
    return out


def draw_node(canvas: Canvas, node: Node, r: float = 11, mode: str | None = None) -> None:
    kind = mode or node.kind
    if kind == "observed":
        canvas.circle(node.x, node.y, r, fill=node.color, stroke="#ffffff", stroke_width=1.4)
    elif kind == "hidden":
        canvas.circle(node.x, node.y, r, fill=blend(PURPLE, "#ffffff", 0.74), stroke=blend(PURPLE, "#ffffff", 0.22), stroke_width=2.2)
    elif kind == "missing":
        canvas.circle(node.x, node.y, r, fill="#ffffff", stroke="#b4bbca", stroke_width=2.0, dash="4 5")
        canvas.text(node.x, node.y - 1, "?", size=max(10, int(r * 1.25)), fill="#9aa3b2", anchor="middle")
    elif kind == "blank":
        canvas.circle(node.x, node.y, r, fill="#ffffff", stroke="#b4bbca", stroke_width=1.9, dash="4 5")
    elif kind == "imputed":
        canvas.hatched_circle(node.x, node.y, r, node.color, stroke_width=2.0)
    elif kind == "sampled":
        canvas.circle(node.x, node.y, r, fill="#ffffff", stroke=node.color, stroke_width=2.2, dash="3 4")
    elif kind == "soft":
        canvas.circle(node.x, node.y, r, fill=blend(node.color, "#ffffff", 0.7), stroke=node.color, stroke_width=1.5, opacity=0.75)


def draw_tree(
    canvas: Canvas,
    nodes: list[Node],
    radius: float = 11,
    edge_color: str = EDGE,
    edge_width: float = 3.0,
    edge_opacity: float = 1.0,
    node_mode: dict[str, str] | None = None,
) -> None:
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        if not node.parent:
            continue
        d, pts = edge_path(by_id[node.parent], node)
        canvas.path(d, pts, stroke=edge_color, stroke_width=edge_width, opacity=edge_opacity)
    for node in nodes:
        mode = node_mode.get(node.id) if node_mode else None
        draw_node(canvas, node, radius, mode)


def base_tree_nodes() -> list[Node]:
    return [
        Node("r", None, 0.50, 0.02, PURPLE_DARK, "hidden"),
        Node("a", "r", 0.27, 0.19, RED, "hidden"),
        Node("b", "r", 0.73, 0.20, GREEN, "hidden"),
        Node("a1", "a", 0.16, 0.44, RED, "observed"),
        Node("a2", "a", 0.39, 0.45, ORANGE, "observed"),
        Node("b1", "b", 0.62, 0.44, OLIVE, "observed"),
        Node("b2", "b", 0.84, 0.45, TEAL, "observed"),
        Node("a1f1", "a1", 0.09, 0.76, "#c94f32", "sampled"),
        Node("a1f2", "a1", 0.21, 0.79, "#dd7440", "sampled"),
        Node("a2f1", "a2", 0.33, 0.77, "#e39b33", "sampled"),
        Node("a2f2", "a2", 0.45, 0.80, "#d9b23a", "sampled"),
        Node("b1f1", "b1", 0.57, 0.77, "#77a852", "sampled"),
        Node("b1f2", "b1", 0.68, 0.80, GREEN, "sampled"),
        Node("b2f1", "b2", 0.79, 0.77, TEAL, "sampled"),
        Node("b2f2", "b2", 0.91, 0.80, BLUE, "sampled"),
    ]


def observed_tree(stage: int) -> list[Node]:
    if stage == 1:
        return [
            Node("r", None, 0.50, 0.02, PURPLE, "hidden"),
            Node("a", "r", 0.32, 0.34, PURPLE, "hidden"),
            Node("l1", "a", 0.20, 0.72, RED, "observed"),
            Node("l2", "a", 0.45, 0.72, ORANGE, "observed"),
            Node("l3", "r", 0.77, 0.72, GREEN, "observed"),
        ]
    if stage == 2:
        return [
            Node("r", None, 0.50, 0.02, PURPLE, "hidden"),
            Node("a", "r", 0.26, 0.20, PURPLE, "hidden"),
            Node("b", "r", 0.74, 0.24, PURPLE, "hidden"),
            Node("a1", "a", 0.14, 0.48, PURPLE, "hidden"),
            Node("a2", "a", 0.38, 0.48, PURPLE, "hidden"),
            Node("l1", "a1", 0.08, 0.78, RED, "observed"),
            Node("l2", "a1", 0.20, 0.78, ORANGE, "observed"),
            Node("l3", "a2", 0.33, 0.78, "#dfad36", "observed"),
            Node("l4", "a2", 0.46, 0.78, OLIVE, "observed"),
            Node("l5", "b", 0.67, 0.78, GREEN, "observed"),
            Node("l6", "b", 0.84, 0.78, TEAL, "observed"),
        ]
    return [
        Node("r", None, 0.50, 0.02, PURPLE, "hidden"),
        Node("a", "r", 0.24, 0.18, PURPLE, "hidden"),
        Node("b", "r", 0.76, 0.18, PURPLE, "hidden"),
        Node("a1", "a", 0.13, 0.42, PURPLE, "hidden"),
        Node("a2", "a", 0.35, 0.43, PURPLE, "hidden"),
        Node("b1", "b", 0.66, 0.42, PURPLE, "hidden"),
        Node("b2", "b", 0.87, 0.43, PURPLE, "hidden"),
        Node("l1", "a1", 0.07, 0.78, RED, "observed"),
        Node("l2", "a1", 0.18, 0.78, "#dd7440", "observed"),
        Node("l3", "a2", 0.29, 0.78, ORANGE, "observed"),
        Node("l4", "a2", 0.40, 0.78, "#d9b23a", "observed"),
        Node("l5", "b1", 0.60, 0.78, "#65a354", "observed"),
        Node("l6", "b1", 0.71, 0.78, GREEN, "observed"),
        Node("l7", "b2", 0.82, 0.78, TEAL, "observed"),
        Node("l8", "b2", 0.93, 0.78, BLUE, "observed"),
    ]


def panel(
    canvas: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    letter: str,
    title: str,
    subtitle: str | None = None,
) -> None:
    canvas.rect(x, y, w, h, fill=PANEL, stroke="#ccd4df", stroke_width=1.0, rx=8)
    canvas.text(x + 22, y + 38, letter, size=30, fill=INK, anchor="start", style="bold")
    canvas.text(x + 64, y + 38, title, size=30, fill="#2b303a", anchor="start")
    if subtitle:
        canvas.text(x + 64, y + 72, subtitle, size=21, fill=MUTED, anchor="start")


def draw_time_axis(canvas: Canvas, x: float, y1: float, y2: float, labels: list[int]) -> None:
    canvas.line(x, y1, x, y2, AXIS, 1.7)
    for i, label in enumerate(labels):
        y = y1 + (y2 - y1) * i / (len(labels) - 1)
        canvas.line(x - 9, y, x + 9, y, AXIS, 1.5)
        canvas.time_label(x - 18, y, label, anchor="end")


def draw_barcode(canvas: Canvas, x: float, y: float, colors: list[str | None], scale: float = 1.0) -> None:
    bw, bh, gap = 5.8 * scale, 16 * scale, 2.8 * scale
    total = len(colors) * bw + (len(colors) - 1) * gap
    px = x - total / 2
    for color in colors:
        canvas.rect(px, y, bw, bh, fill=color or "#eef1f5", stroke="#dfe4ec", stroke_width=0.45, rx=1.6 * scale)
        px += bw + gap


def frontier_y(nodes: list[Node]) -> float:
    observed = [node.y for node in nodes if node.kind == "observed"]
    return sum(observed) / len(observed)


def observed_span(nodes: list[Node]) -> tuple[float, float]:
    observed = [node for node in nodes if node.kind == "observed"]
    return min(node.x for node in observed), max(node.x for node in observed)


def draw_capture_slice(
    canvas: Canvas,
    x1: float,
    x2: float,
    y: float,
    height: float = 28,
    opacity: float = 0.92,
) -> None:
    canvas.rect(x1, y - height / 2, x2 - x1, height, fill=GOLD_SOFT, stroke="none", rx=7, opacity=opacity)
    canvas.line(x1 + 9, y, x2 - 9, y, stroke="#dfb369", stroke_width=0.9, opacity=0.72)


def draw_cubic(
    canvas: Canvas,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    stroke: str,
    stroke_width: float,
    opacity: float = 1.0,
    dash: str | None = None,
) -> None:
    d = (
        f"M {p0[0]:.1f} {p0[1]:.1f} "
        f"C {p1[0]:.1f} {p1[1]:.1f}, {p2[0]:.1f} {p2[1]:.1f}, {p3[0]:.1f} {p3[1]:.1f}"
    )
    canvas.path(d, cubic_points(p0, p1, p2, p3), stroke=stroke, stroke_width=stroke_width, opacity=opacity, dash=dash)


def draw_snapshot_strip(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(
        canvas,
        x,
        y,
        w,
        h,
        "a",
        "lineage snapshots",
        "one observed generation per embryo: state plus barcode",
    )
    draw_time_axis(canvas, x + 55, y + 132, y + h - 86, [0, 1, 2, 3])
    boxes = [
        (x + 100, y + 124, 182, h - 220, 1, "embryo 1", 1),
        (x + 310, y + 124, 206, h - 220, 2, "embryo 2", 2),
        (x + 548, y + 124, 206, h - 220, 3, "embryo 3", 3),
    ]
    for bx, by, bw, bh, stage, label, t in boxes:
        canvas.text(bx + 18, by + 28, label, size=18, fill=MUTED)
        canvas.time_label(bx + bw - 26, by + 28, t, anchor="middle")
        nodes = transform_nodes(observed_tree(stage), bx + 18, by + 72, bw - 36, 188)
        capture_y = frontier_y(nodes)
        draw_capture_slice(canvas, bx + 4, bx + bw - 4, capture_y, height=26, opacity=0.90)
        draw_tree(canvas, nodes, radius=8.2, edge_width=2.15, edge_opacity=0.80)
        leaves = [node for node in nodes if node.kind == "observed"]
        for i, leaf in enumerate(leaves[:6]):
            code = (
                [GREEN, None, BLUE, None, PURPLE]
                if i % 3 == 0
                else [None, BLUE, GREEN, PURPLE, None]
                if i % 3 == 1
                else [ORANGE, None, None, GREEN, BLUE]
            )
            draw_barcode(canvas, leaf.x, leaf.y + 34 + (i % 2) * 19, code, scale=0.78)
    canvas.line(x + 292, y + 138, x + 292, y + h - 118, stroke="#e5e9ef", stroke_width=1.0)
    canvas.line(x + 530, y + 138, x + 530, y + h - 118, stroke="#e5e9ef", stroke_width=1.0)

    legend_x, legend_y = x + 142, y + h - 53
    canvas.circle(legend_x, legend_y, 10, fill=RED, stroke="#ffffff", stroke_width=1.0)
    canvas.circle(legend_x + 40, legend_y, 10, fill=ORANGE, stroke="#ffffff", stroke_width=1.0)
    canvas.circle(legend_x + 80, legend_y, 10, fill=GREEN, stroke="#ffffff", stroke_width=1.0)
    draw_barcode(canvas, legend_x + 40, legend_y + 18, [GREEN, None, BLUE, GREEN, PURPLE], scale=0.78)
    canvas.text(legend_x + 136, legend_y + 7, "state plus heritable barcode", size=21, fill=MUTED)


def draw_unobserved_target(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(
        canvas,
        x,
        y,
        w,
        h,
        "b",
        "complete trees are latent",
        "the full state-labeled target is never observed",
    )
    draw_time_axis(canvas, x + 70, y + 155, y + h - 70, [0, 1, 2, 3])
    tree_x, tree_y, tree_w, tree_h = x + 148, y + 118, w - 236, h - 190
    nodes = transform_nodes(base_tree_nodes(), tree_x, tree_y, tree_w, tree_h)
    band_y = frontier_y(nodes)
    obs_min, obs_max = observed_span(nodes)
    draw_capture_slice(canvas, obs_min - 46, obs_max + 46, band_y, height=34, opacity=0.94)
    modes = {node.id: "blank" for node in nodes}
    for node in nodes:
        if node.kind == "observed":
            modes[node.id] = "observed"
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        if node.parent is None:
            continue
        d, pts = edge_path(by_id[node.parent], node)
        canvas.path(d, pts, stroke=EDGE, stroke_width=2.7, opacity=0.58, dash="8 9")
    for node in nodes:
        draw_node(canvas, node, 11.2 if node.kind == "observed" else 9.4, modes[node.id])
    label_x = x + w - 118
    canvas.line(label_x - 18, tree_y + 18, label_x - 18, band_y - 34, stroke="#c7ceda", stroke_width=1.1)
    canvas.line(label_x - 24, tree_y + 18, label_x - 12, tree_y + 18, stroke="#c7ceda", stroke_width=1.1)
    canvas.line(label_x - 24, band_y - 34, label_x - 12, band_y - 34, stroke="#c7ceda", stroke_width=1.1)
    canvas.text(label_x, tree_y + 84, "unobserved", size=20, fill=MUTED, anchor="middle")
    canvas.text(label_x, tree_y + 111, "ancestors", size=20, fill=MUTED, anchor="middle")

    canvas.line(label_x - 18, band_y + 34, label_x - 18, tree_y + tree_h - 8, stroke="#c7ceda", stroke_width=1.1)
    canvas.line(label_x - 24, band_y + 34, label_x - 12, band_y + 34, stroke="#c7ceda", stroke_width=1.1)
    canvas.line(label_x - 24, tree_y + tree_h - 8, label_x - 12, tree_y + tree_h - 8, stroke="#c7ceda", stroke_width=1.1)
    canvas.text(label_x, tree_y + tree_h - 104, "unobserved", size=20, fill=MUTED, anchor="middle")
    canvas.text(label_x, tree_y + tree_h - 77, "descendants", size=20, fill=MUTED, anchor="middle")

    canvas.text(obs_min + 10, band_y - 34, "observed generation", size=21, fill="#8a5d1f", anchor="start")
    bracket_y = y + h - 59
    canvas.line(tree_x + tree_w * 0.10, bracket_y, tree_x + tree_w * 0.90, bracket_y, stroke=PURPLE_DARK, stroke_width=1.15)
    canvas.line(tree_x + tree_w * 0.10, bracket_y - 9, tree_x + tree_w * 0.10, bracket_y, stroke=PURPLE_DARK, stroke_width=1.15)
    canvas.line(tree_x + tree_w * 0.90, bracket_y - 9, tree_x + tree_w * 0.90, bracket_y, stroke=PURPLE_DARK, stroke_width=1.15)
    canvas.text(tree_x + tree_w * 0.50, y + h - 31, "full state-labeled training target is unavailable", size=22, fill=PURPLE_DARK, anchor="middle")


def draw_sample_cloud(canvas: Canvas, cx: float, cy: float, color: str, n: int = 6) -> None:
    for i in range(n):
        angle = 1.7 * i
        radius = 8 + 4 * (i % 3)
        canvas.circle(cx + math.cos(angle) * radius, cy + math.sin(angle) * radius * 0.72, 3.2, fill=color, stroke="none", opacity=0.18)


def draw_posterior_alternates(canvas: Canvas, nodes: list[Node]) -> None:
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        if node.kind != "sampled" or node.parent is None:
            continue
        parent = by_id[node.parent]
        for j, dx in enumerate((-18, 18)):
            end = (node.x + dx, node.y + 14 + 5 * j)
            mid = (parent.y + end[1]) / 2
            p0 = (parent.x, parent.y)
            p1 = (parent.x + dx * 0.10, mid)
            p2 = (end[0] - dx * 0.15, mid)
            p3 = end
            d = (
                f"M {p0[0]:.1f} {p0[1]:.1f} "
                f"C {p1[0]:.1f} {p1[1]:.1f}, {p2[0]:.1f} {p2[1]:.1f}, {p3[0]:.1f} {p3[1]:.1f}"
            )
            canvas.path(d, cubic_points(p0, p1, p2, p3), stroke=node.color, stroke_width=1.15, opacity=0.20)
            canvas.circle(end[0], end[1], 4.8, fill="#ffffff", stroke=node.color, stroke_width=1.2, opacity=0.30, dash="3 4")


def draw_model_panel(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(
        canvas,
        x,
        y,
        w,
        h,
        "c",
        "inpainting",
        None,
    )
    canvas.text(x + 64, y + 72, "condition on one observed generation; inpaint masked ancestors and descendants", size=22, fill="#566174")
    legend_y = y + 115
    legend_x = x + 935
    canvas.circle(legend_x, legend_y, 8, fill=RED, stroke="#ffffff", stroke_width=1.0)
    canvas.text(legend_x + 20, legend_y, "observed", size=19, fill=MUTED)
    canvas.hatched_circle(legend_x + 122, legend_y, 8, PURPLE_DARK, stroke_width=1.5)
    canvas.text(legend_x + 142, legend_y, "sampled state", size=19, fill=MUTED)
    canvas.circle(legend_x + 276, legend_y, 8, fill="#ffffff", stroke=TEAL, stroke_width=1.7, dash="3 4")
    canvas.text(legend_x + 296, legend_y, "posterior ensemble", size=19, fill=MUTED)

    tx, ty, tw, th = x + 352, y + 130, 805, 505
    draw_time_axis(canvas, x + 90, ty + 20, ty + th - 10, [0, 1, 2, 3])
    nodes = transform_nodes(base_tree_nodes(), tx, ty, tw, th)
    obs_y = frontier_y(nodes)
    obs_min, obs_max = observed_span(nodes)
    draw_capture_slice(canvas, obs_min - 42, obs_max + 42, obs_y, height=38, opacity=0.91)
    modes = {}
    for node in nodes:
        if node.kind == "observed":
            modes[node.id] = "observed"
        elif node.kind == "sampled":
            modes[node.id] = "sampled"
        else:
            modes[node.id] = "imputed"
    draw_posterior_alternates(canvas, nodes)
    for node in nodes:
        if node.kind != "observed":
            draw_sample_cloud(canvas, node.x, node.y, node.color, n=5 if node.kind == "hidden" else 6)
    draw_tree(canvas, nodes, radius=10.5, edge_width=3.0, node_mode=modes)

    canvas.arrow(tx + tw * 0.28, obs_y - 16, tx + tw * 0.28, ty + 112, stroke=PURPLE_DARK, stroke_width=1.55, opacity=0.72)
    canvas.arrow(tx + tw * 0.72, obs_y + 18, tx + tw * 0.72, ty + th - 100, stroke=PURPLE_DARK, stroke_width=1.55, opacity=0.72)
    bracket_y = obs_y - 52
    canvas.line(obs_min - 16, bracket_y, obs_max + 16, bracket_y, stroke="#9a6a24", stroke_width=1.25)
    canvas.line(obs_min - 16, bracket_y, obs_min - 16, bracket_y + 15, stroke="#9a6a24", stroke_width=1.25)
    canvas.line(obs_max + 16, bracket_y, obs_max + 16, bracket_y + 15, stroke="#9a6a24", stroke_width=1.25)
    canvas.text((obs_min + obs_max) / 2, bracket_y - 20, "observed generation", size=22, fill="#8a5d1f", anchor="middle")

    canvas.text(tx - 44, obs_y - 108, "masked ancestors", size=23, fill=PURPLE_DARK, anchor="start", style="bold")
    canvas.text(tx - 44, obs_y - 75, "histories are sampled", size=20, fill=MUTED, anchor="start")
    canvas.text(tx + tw + 28, obs_y + 126, "masked", size=22, fill=PURPLE_DARK, anchor="start", style="bold")
    canvas.text(tx + tw + 28, obs_y + 154, "descendants", size=22, fill=PURPLE_DARK, anchor="start", style="bold")
    canvas.text(tx + tw + 28, obs_y + 184, "future states", size=19, fill=MUTED, anchor="start")
    canvas.text(tx + tw + 28, obs_y + 208, "are sampled", size=19, fill=MUTED, anchor="start")

    lx, ly = x + 1242, y + 190
    canvas.text(lx, ly - 34, "mask/reveal training", size=20, fill="#2b303a")
    for j, (label, yy, colors) in enumerate(
        [
            ("observed", ly + 10, [RED, ORANGE, GREEN, TEAL]),
            ("masked", ly + 115, [RED, ORANGE, GREEN, TEAL]),
            ("sampled", ly + 220, [RED, ORANGE, GREEN, TEAL]),
        ]
        ):
        canvas.text(lx, yy + 5, label, size=17, fill=MUTED, anchor="start")
        for i, color in enumerate(colors):
            for k in range(4):
                px = lx + 102 + i * 34 + (k % 2) * 9
                py = yy + (k // 2) * 12
                if j == 1:
                    canvas.circle(px, py, 4.2, fill="#ffffff", stroke=color, stroke_width=0.95, opacity=0.55, dash="2 3")
                else:
                    canvas.circle(px, py, 4.2, fill=color, stroke="none", opacity=0.72 if j == 0 else 0.48)
        if j < 2:
            canvas.line(lx + 198, yy + 35, lx + 198, yy + 75, stroke="#b7bfca", stroke_width=1.0, dash="3 6")


def draw_convergence(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "d", "convergence", "sampled histories reveal distinct origins")
    ox1, ox2, oy = x + 124, x + w - 124, y + 186
    cx, cy = x + w / 2, y + 380

    for i, (ox, color, label) in enumerate(((ox1, RED, "origin A"), (ox2, TEAL, "origin B"))):
        for j in range(18):
            angle = 0.74 * j + i
            radius = 12 + (j * 7 % 31)
            canvas.circle(ox + math.cos(angle) * radius * 0.95, oy + math.sin(angle) * radius * 0.62, 3.7, fill=color, stroke="none", opacity=0.23)
        canvas.circle(ox, oy, 13, fill=color, stroke="#ffffff", stroke_width=1.0)
        canvas.text(ox, oy - 34, label, size=20, fill=color, anchor="middle")

    canvas.circle(cx, cy, 86, fill="#ffffff", stroke="#c7ceda", stroke_width=1.3, opacity=0.90)
    canvas.circle(cx, cy, 60, fill=GOLD_SOFT, stroke="none", opacity=0.62)
    for i in range(36):
        angle = 0.69 * i
        radius = 8 + (i * 13 % 45)
        color = RED if i % 2 == 0 else TEAL
        canvas.circle(cx + math.cos(angle) * radius * 0.86, cy + math.sin(angle) * radius * 0.58, 5.0, fill=color, stroke="#ffffff", stroke_width=0.55, opacity=0.80)

    for j in range(7):
        offset = (j - 3) * 5
        start = (ox1 + offset * 0.8, oy + 18 + (j % 2) * 5)
        end = (cx - 24 + offset * 0.7, cy - 45 + j * 3)
        draw_cubic(canvas, start, (ox1 - 8, y + 248 + j * 5), (cx - 132, cy - 118 + j * 4), end, RED, 1.05, opacity=0.25)
        start = (ox2 - offset * 0.8, oy + 18 + (j % 2) * 5)
        end = (cx + 24 - offset * 0.7, cy - 45 + j * 3)
        draw_cubic(canvas, start, (ox2 + 8, y + 248 + j * 5), (cx + 132, cy - 118 + j * 4), end, TEAL, 1.05, opacity=0.25)

    draw_cubic(canvas, (ox1, oy + 20), (ox1 - 3, y + 252), (cx - 132, cy - 110), (cx - 30, cy - 42), RED, 2.5, opacity=0.86)
    draw_cubic(canvas, (ox2, oy + 20), (ox2 + 3, y + 252), (cx + 132, cy - 110), (cx + 30, cy - 42), TEAL, 2.5, opacity=0.86)
    canvas.text(cx, cy + 104, "shared fate cluster", size=20, fill=MUTED, anchor="middle")
    canvas.text(cx, y + 274, "sampled histories", size=19, fill=PURPLE_DARK, anchor="middle")
    canvas.text(x + w / 2, y + h - 52, "one fate, multiple inferred routes", size=20, fill=MUTED, anchor="middle")


def draw_gene_programs(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "e", "gene programs", "sampled histories expose gene modules")
    tx, ty = x + 66, y + 168
    small = transform_nodes(base_tree_nodes()[:7], tx, ty, 126, 214)
    small_by = {node.id: node for node in small}
    modes = {node.id: "imputed" if node.kind == "hidden" else "observed" for node in small}
    draw_tree(canvas, small, radius=6.7, edge_width=1.65, edge_opacity=0.52, node_mode=modes)
    for path_ids, color in ((("r", "a", "a1"), RED), (("r", "b", "b2"), TEAL)):
        for parent_id, child_id in zip(path_ids, path_ids[1:]):
            d, pts = edge_path(small_by[parent_id], small_by[child_id])
            canvas.path(d, pts, stroke=color, stroke_width=3.0, opacity=0.72)
        for node_id in path_ids:
            node = small_by[node_id]
            canvas.circle(node.x, node.y, 7.4, fill=blend(color, "#ffffff", 0.38), stroke=color, stroke_width=1.2)
    canvas.text(tx - 8, ty - 18, "sampled", size=18, fill=MUTED, anchor="start")
    canvas.text(tx - 8, ty + 2, "histories", size=18, fill=MUTED, anchor="start")

    mx, my = x + 270, y + 178
    rows, cols, cell = 7, 7, 28
    canvas.text(mx + cols * cell / 2, my - 34, "descendant genes", size=18, fill=MUTED, anchor="middle")
    canvas.text(mx - 42, my + 78, "ancestor", size=18, fill=MUTED, anchor="middle")
    canvas.text(mx - 42, my + 101, "genes", size=18, fill=MUTED, anchor="middle")
    for r in range(rows):
        for c in range(cols):
            val = 0.10 + (((r * 5 + c * 3) % 9) / 100)
            color = "#dfe5ee"
            if r in (1, 2) and c in (3, 4):
                val = 0.66 + 0.10 * ((r + c) % 2)
                color = "#bf4057"
            elif r in (5, 6) and c in (1, 2):
                val = 0.62 + 0.12 * ((r + c) % 2)
                color = "#bf4057"
            elif r in (3, 4) and c in (5, 6):
                val = 0.48 + 0.10 * ((r + c) % 2)
                color = GREEN
            canvas.rect(mx + c * cell, my + r * cell, cell - 3, cell - 3, fill=blend(color, "#ffffff", 1 - val), stroke="#ffffff", stroke_width=0.55, rx=3)
    canvas.rect(mx - 4, my - 4, cols * cell + 3, rows * cell + 3, fill="none", stroke="#d7dde7", stroke_width=0.9, rx=4)
    canvas.rect(mx + 3 * cell - 3, my + 1 * cell - 3, 2 * cell + 1, 2 * cell + 1, fill="none", stroke=PURPLE_DARK, stroke_width=1.55, rx=4)
    canvas.rect(mx + 1 * cell - 3, my + 5 * cell - 3, 2 * cell + 1, 2 * cell + 1, fill="none", stroke=PURPLE_DARK, stroke_width=1.55, rx=4)
    canvas.line(tx + 110, ty + 90, mx + 3 * cell - 6, my + 2 * cell, stroke=RED, stroke_width=1.35, opacity=0.72, dash="4 6")
    canvas.line(tx + 116, ty + 150, mx + 1 * cell - 6, my + 6 * cell, stroke=TEAL, stroke_width=1.35, opacity=0.72, dash="4 6")
    canvas.text(mx + cols * cell / 2, my + rows * cell + 30, "conditional dependence matrix", size=18, fill=MUTED, anchor="middle")
    canvas.text(mx + cols * cell / 2, my + rows * cell + 56, "shared block recurs across histories", size=17, fill=PURPLE_DARK, anchor="middle")
    canvas.text(x + w / 2, y + h - 52, "unseen ancestors link to descendant programs", size=20, fill=MUTED, anchor="middle")


def draw_perturbations(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "f", "perturbations", "sampled histories support counterfactual edits")
    sx, sy = x + 86, y + 178
    root = (sx, sy)
    context = (sx + 38, sy + 56)
    edit = (sx + 58, sy + 118)
    draw_cubic(canvas, root, (sx + 2, sy + 26), (sx + 28, sy + 40), context, EDGE, 2.1, opacity=0.70)
    draw_cubic(canvas, context, (sx + 46, sy + 78), (sx + 56, sy + 94), edit, EDGE, 2.1, opacity=0.70)
    canvas.hatched_circle(root[0], root[1], 7.0, PURPLE_DARK, stroke_width=1.35, opacity=0.88)
    canvas.hatched_circle(context[0], context[1], 7.0, PURPLE_DARK, stroke_width=1.35, opacity=0.88)
    canvas.circle(edit[0], edit[1], 9.2, fill=blend(RED, "#ffffff", 0.55), stroke=RED, stroke_width=1.5)
    canvas.text(sx + 38, sy - 23, "shared context", size=18, fill=MUTED, anchor="middle")
    canvas.text(edit[0] - 20, edit[1] - 21, "edited", size=17, fill=RED, anchor="end")
    canvas.text(edit[0] - 20, edit[1] + 1, "state", size=17, fill=RED, anchor="end")

    lane_x0, lane_x1 = x + 206, x + 332
    lane_y = [y + 260, y + 362]
    lane_labels = [("control", MUTED), ("KO", RED)]
    fate_colors = [RED, TEAL, GREEN]
    lane_dots = [
        [(0.35, RED), (0.54, RED), (0.73, TEAL), (0.90, GREEN)],
        [(0.33, TEAL), (0.55, TEAL), (0.74, GREEN), (0.91, RED)],
    ]
    for row, ((label, label_color), yy) in enumerate(zip(lane_labels, lane_y)):
        stroke = "#a8b0bd" if row == 0 else RED
        ctrl1 = (edit[0] + 46, edit[1] - 10 + row * 28)
        ctrl2 = (lane_x0 - 44, yy)
        draw_cubic(canvas, edit, ctrl1, ctrl2, (lane_x0, yy), stroke, 2.0, opacity=0.76)
        canvas.text(lane_x0 - 10, yy - 20, label, size=18, fill=label_color, anchor="end")
        if row == 1:
            canvas.rect(lane_x0 - 18, yy + 13, 64, 26, fill=GOLD_SOFT, stroke="#e9c894", stroke_width=0.85, rx=7)
            canvas.text(lane_x0 + 14, yy + 27, "g = 0", size=16, fill="#8a5d1f", anchor="middle")
        canvas.line(lane_x0, yy, lane_x1, yy, stroke=stroke, stroke_width=1.45, opacity=0.52)
        for frac, color in lane_dots[row]:
            px = lane_x0 + (lane_x1 - lane_x0) * frac
            canvas.circle(px, yy, 5.6, fill=color, stroke="#ffffff", stroke_width=0.7, opacity=0.88)

    bx, by = x + 360, y + 190
    canvas.text(bx + 62, by - 23, "descendant", size=17, fill=MUTED, anchor="middle")
    canvas.text(bx + 62, by - 2, "composition", size=17, fill=MUTED, anchor="middle")
    bar_w, bar_h = 126, 24
    profiles = [("control", [0.48, 0.27, 0.25]), ("KO", [0.22, 0.50, 0.28])]
    for row, (label, vals) in enumerate(profiles):
        yy = lane_y[row] - bar_h / 2
        px = bx
        for val, color in zip(vals, fate_colors):
            seg_w = bar_w * val
            canvas.rect(px, yy, seg_w, bar_h, fill=color, stroke="#ffffff", stroke_width=0.55, rx=2, opacity=0.84)
            px += seg_w
    canvas.line(bx + 68, lane_y[0] + 19, bx + 68, lane_y[1] - 19, stroke=TEAL, stroke_width=1.15, opacity=0.82)
    canvas.line(bx + 60, lane_y[1] - 27, bx + 68, lane_y[1] - 19, stroke=TEAL, stroke_width=1.15, opacity=0.82)
    canvas.line(bx + 76, lane_y[1] - 27, bx + 68, lane_y[1] - 19, stroke=TEAL, stroke_width=1.15, opacity=0.82)
    canvas.text(bx + bar_w / 2, lane_y[1] + 52, "fate B +23%", size=18, fill=TEAL, anchor="middle")
    canvas.text(x + w / 2, y + h - 52, "same context; edited state changes future samples", size=20, fill=MUTED, anchor="middle")


def draw_global(canvas: Canvas) -> None:
    canvas.rect(0, 0, W, H, fill=PAPER, stroke="none")

    margin = 70
    top_y, top_h = 70, 580
    mid_y, mid_h = 720, 700
    bottom_y, bottom_h = 1500, 610
    gap = 54
    top_w = (W - 2 * margin - gap) / 2
    bottom_w = (W - 2 * margin - 2 * gap) / 3

    draw_snapshot_strip(canvas, margin, top_y, top_w, top_h)
    draw_unobserved_target(canvas, margin + top_w + gap, top_y, top_w, top_h)
    canvas.arrow(margin + top_w + 12, top_y + top_h * 0.52, margin + top_w + gap - 12, top_y + top_h * 0.52, stroke="#9aa3b2", stroke_width=1.8)

    draw_model_panel(canvas, margin, mid_y, W - 2 * margin, mid_h)

    canvas.text(W / 2, bottom_y - 48, "sampled lineage histories enable downstream biology", size=25, fill=MUTED, anchor="middle")
    canvas.line(margin + bottom_w / 2, bottom_y - 25, W - margin - bottom_w / 2, bottom_y - 25, stroke="#b7bfca", stroke_width=1.1)
    for i in range(3):
        cx = margin + i * (bottom_w + gap) + bottom_w / 2
        canvas.line(cx, bottom_y - 25, cx, bottom_y - 9, stroke="#b7bfca", stroke_width=1.1)

    draw_convergence(canvas, margin, bottom_y, bottom_w, bottom_h)
    draw_gene_programs(canvas, margin + bottom_w + gap, bottom_y, bottom_w, bottom_h)
    draw_perturbations(canvas, margin + 2 * (bottom_w + gap), bottom_y, bottom_w, bottom_h)


def main() -> None:
    canvas = Canvas(W, H)
    draw_global(canvas)
    canvas.save(OUT_SVG, OUT_PNG, OUT_HTML)
    print(OUT_SVG.resolve())
    print(OUT_PNG.resolve())
    print(OUT_HTML.resolve())


if __name__ == "__main__":
    main()
