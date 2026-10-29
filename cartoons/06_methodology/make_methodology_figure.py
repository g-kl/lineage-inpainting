#!/usr/bin/env python3
"""Create Figure 06: methodology-first lineage inpainting schematic."""

from __future__ import annotations

from pathlib import Path
import base64
import html
import importlib.util
from io import BytesIO
import math
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parent
UTIL_PATH = ROOT.parent / "05_lineage_completion" / "make_lineage_completion_figure.py"
OUT_SVG = ROOT / "methodology_figure.svg"
OUT_PNG = ROOT / "methodology_figure.png"
OUT_HTML = ROOT / "methodology_figure.html"
ASSET_DIR = ROOT / "assets"
ATTENTION_PNG = ASSET_DIR / "tree_attention_box_figure.png"
INPAINTING_PNG = ASSET_DIR / "stacked_bridge_inpainting_figure.png"

spec = importlib.util.spec_from_file_location("lineage_completion_utils", UTIL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load drawing utilities from {UTIL_PATH}")
util = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = util
spec.loader.exec_module(util)

Canvas = util.Canvas
Node = util.Node

W, H = 1800, 2070
SCALE = util.SCALE
PAPER_WIDTH_IN = 7.5
PAPER_HEIGHT_IN = PAPER_WIDTH_IN * H / W

INK = util.INK
MUTED = util.MUTED
PAPER = util.PAPER
PANEL = util.PANEL
EDGE = util.EDGE
AXIS = util.AXIS
PURPLE = util.PURPLE
PURPLE_DARK = util.PURPLE_DARK
PURPLE_SOFT = util.PURPLE_SOFT
GOLD = util.GOLD
GOLD_SOFT = util.GOLD_SOFT
TEAL = util.TEAL
GREEN = util.GREEN
RED = util.RED
ORANGE = util.ORANGE
OLIVE = util.OLIVE
BLUE = util.BLUE


def save(canvas: Canvas) -> None:
    png = canvas.image.resize((canvas.width, canvas.height), Image.Resampling.LANCZOS)
    png.save(OUT_PNG)

    defs = "\n".join(canvas.svg_defs)
    body = "\n".join(canvas.svg_body)
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{PAPER_WIDTH_IN}in" height="{PAPER_HEIGHT_IN:.4f}in" viewBox="0 0 {canvas.width} {canvas.height}" role="img" aria-label="Methodology schematic for conditional lineage inpainting">
<defs>
{defs}
</defs>
{body}
</svg>
"""
    OUT_SVG.write_text(svg, encoding="utf-8")
    OUT_HTML.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Methodology schematic</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/dreampulse/computer-modern-web-font@master/font/Serif/cmun-serif.css">
  <style>
    body {{ margin: 0; background: #ffffff; }}
    .figure {{ width: {canvas.width}px; height: {canvas.height}px; }}
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
    util.panel(canvas, x, y, w, h, letter, title, subtitle)


def draw_embedded_png(
    canvas: Canvas,
    path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    crop: tuple[int, int, int, int] | None = None,
) -> None:
    img = Image.open(path).convert("RGBA")
    if crop is not None:
        img = img.crop(crop)
    src_w, src_h = img.size
    scale = min((w * SCALE) / src_w, (h * SCALE) / src_h)
    out_w = max(1, round(src_w * scale))
    out_h = max(1, round(src_h * scale))
    resized = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", (round(w * SCALE), round(h * SCALE)), (255, 255, 255, 0))
    ox = (bg.width - out_w) // 2
    oy = (bg.height - out_h) // 2
    bg.alpha_composite(resized, (ox, oy))
    canvas.image.alpha_composite(bg, (round(x * SCALE), round(y * SCALE)))

    buf = BytesIO()
    bg.save(buf, format="PNG")
    href = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    canvas.svg_body.append(
        f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" href="{href}" preserveAspectRatio="none"/>'
    )


def lines(
    canvas: Canvas,
    x: float,
    y: float,
    values: list[str],
    size: int = 19,
    fill: str = MUTED,
    anchor: str = "start",
    line_h: float = 24,
    style: str = "regular",
) -> None:
    for i, value in enumerate(values):
        canvas.text(x, y + i * line_h, value, size=size, fill=fill, anchor=anchor, style=style)


def cross(canvas: Canvas, cx: float, cy: float, r: float = 11, color: str = RED) -> None:
    canvas.line(cx - r, cy - r, cx + r, cy + r, stroke=color, stroke_width=2.0)
    canvas.line(cx + r, cy - r, cx - r, cy + r, stroke=color, stroke_width=2.0)


def check(canvas: Canvas, x: float, y: float, color: str = GREEN) -> None:
    canvas.line(x - 9, y, x - 2, y + 8, stroke=color, stroke_width=2.2)
    canvas.line(x - 2, y + 8, x + 12, y - 10, stroke=color, stroke_width=2.2)


def rounded_label(
    canvas: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    color: str = PURPLE_DARK,
) -> None:
    canvas.rect(x, y, w, h, fill="#fbfcfe", stroke="#dde3ec", stroke_width=0.9, rx=8)
    canvas.text(x + 16, y + 23, title, size=20, fill=color, style="bold")
    canvas.text(x + 16, y + 51, subtitle, size=17, fill=MUTED)


def observed_nodes(nodes: list[Node]) -> list[Node]:
    return [node for node in nodes if node.kind == "observed"]


def observed_line(canvas: Canvas, nodes: list[Node], pad: float = 36, height: float = 30) -> None:
    y = util.frontier_y(nodes)
    x0, x1 = util.observed_span(nodes)
    util.draw_capture_slice(canvas, x0 - pad, x1 + pad, y, height=height, opacity=0.90)


def ghost_tree(canvas: Canvas, nodes: list[Node], radius: float = 8.5) -> None:
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        if node.parent is None:
            continue
        d, pts = util.edge_path(by_id[node.parent], node)
        canvas.path(d, pts, stroke=EDGE, stroke_width=2.1, opacity=0.55, dash="8 8")
    for node in nodes:
        if node.kind == "observed":
            util.draw_node(canvas, node, radius + 1.4, "observed")
        else:
            util.draw_node(canvas, node, radius, "blank")


def completed_tree(canvas: Canvas, nodes: list[Node], radius: float = 8.2) -> None:
    modes = {}
    for node in nodes:
        modes[node.id] = "observed" if node.kind == "observed" else "imputed"
    util.draw_tree(canvas, nodes, radius=radius, edge_width=2.1, edge_opacity=0.72, node_mode=modes)


def completion_variant(nodes: list[Node], shift: int) -> list[Node]:
    palette = [RED, ORANGE, GOLD, OLIVE, GREEN, TEAL, BLUE, PURPLE_DARK]
    out = []
    for i, node in enumerate(nodes):
        color = node.color if node.kind == "observed" else palette[(i + shift) % len(palette)]
        out.append(Node(node.id, node.parent, node.x, node.y, color, node.kind))
    return out


def draw_completion_card(
    canvas: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    shift: int,
    selected: bool = False,
) -> list[Node]:
    fill = "#fbfcfe" if not selected else "#fffdfa"
    stroke = "#dfe5ed" if not selected else "#d9b66b"
    canvas.rect(x, y, w, h, fill=fill, stroke=stroke, stroke_width=1.0, rx=8)
    canvas.text(x + w / 2, y + 24, label, size=16, fill=MUTED, anchor="middle")
    nodes = util.transform_nodes(completion_variant(util.base_tree_nodes(), shift), x + 15, y + 44, w - 30, h - 82)
    observed_line(canvas, nodes, pad=8, height=16)
    completed_tree(canvas, nodes, radius=4.8)
    return nodes


def draw_masked_tree(
    canvas: Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    shift: int,
    mask_ids: set[str],
    sampled: bool = False,
) -> list[Node]:
    nodes = util.transform_nodes(completion_variant(util.base_tree_nodes(), shift), x, y, w, h)
    modes: dict[str, str] = {}
    for node in nodes:
        if node.id in mask_ids:
            modes[node.id] = "sampled" if sampled else "blank"
        elif node.kind == "observed":
            modes[node.id] = "observed"
        else:
            modes[node.id] = "imputed"
    util.draw_tree(canvas, nodes, radius=5.8, edge_width=1.8, edge_opacity=0.66, node_mode=modes)
    return nodes


def sampled_tree(canvas: Canvas, nodes: list[Node], radius: float = 8.2) -> None:
    modes = {}
    for node in nodes:
        if node.kind == "observed":
            modes[node.id] = "observed"
        else:
            modes[node.id] = "sampled"
    util.draw_tree(canvas, nodes, radius=radius, edge_width=2.1, edge_opacity=0.72, node_mode=modes)


def mini_barcode(canvas: Canvas, x: float, y: float, scale: float = 1.0) -> None:
    util.draw_barcode(canvas, x, y, [GREEN, None, BLUE, GREEN, PURPLE], scale=scale)


def scatter_cloud(
    canvas: Canvas,
    cx: float,
    cy: float,
    color: str,
    n: int = 20,
    sx: float = 34,
    sy: float = 18,
    opacity: float = 0.76,
) -> None:
    for i in range(n):
        angle = 0.73 * i + (0.3 if color == TEAL else 0.0)
        radius = 0.18 + ((i * 11) % 31) / 31
        px = cx + math.cos(angle) * sx * radius
        py = cy + math.sin(angle) * sy * radius
        canvas.circle(px, py, 4.6, fill=color, stroke="#ffffff", stroke_width=0.55, opacity=opacity)


def draw_data_panel(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "a", "destructive lineage records", "one observed generation, tree topology retained")
    util.draw_time_axis(canvas, x + 54, y + 132, y + h - 80, [0, 1, 2, 3])
    specs = [
        (x + 106, y + 124, 185, h - 210, 1, "record 1"),
        (x + 333, y + 124, 203, h - 210, 2, "record 2"),
        (x + 582, y + 124, 203, h - 210, 3, "record 3"),
    ]
    for bx, by, bw, bh, stage, label in specs:
        nodes = util.transform_nodes(util.observed_tree(stage), bx + 18, by + 76, bw - 36, 190)
        observed_line(canvas, nodes, pad=12, height=26)
        ghost_tree(canvas, nodes, radius=7.9)
        canvas.text(bx + bw / 2, by + 29, label, size=18, fill=MUTED, anchor="middle")
        for i, leaf in enumerate(observed_nodes(nodes)[:5]):
            mini_barcode(canvas, leaf.x, leaf.y + 34 + (i % 2) * 18, scale=0.68)
    canvas.line(x + 306, y + 142, x + 306, y + h - 122, stroke="#e5e9ef", stroke_width=1.0)
    canvas.line(x + 560, y + 142, x + 560, y + h - 122, stroke="#e5e9ef", stroke_width=1.0)
    lx, ly = x + 150, y + h - 54
    canvas.circle(lx, ly, 10, fill=RED, stroke="#ffffff", stroke_width=1.0)
    canvas.circle(lx + 38, ly, 10, fill=ORANGE, stroke="#ffffff", stroke_width=1.0)
    canvas.circle(lx + 76, ly, 10, fill=TEAL, stroke="#ffffff", stroke_width=1.0)
    mini_barcode(canvas, lx + 38, ly + 19, scale=0.72)
    canvas.text(lx + 126, ly + 8, "cell state plus heritable barcode", size=20, fill=MUTED)


def draw_missing_target_panel(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "b", "no full-tree labels", "only indirect constraints survive destructive capture")
    tree_x, tree_y, tree_w, tree_h = x + 86, y + 112, 430, 316
    nodes = util.transform_nodes(util.base_tree_nodes(), tree_x, tree_y, tree_w, tree_h)
    observed_line(canvas, nodes, pad=34, height=34)
    ghost_tree(canvas, nodes, radius=8.7)
    canvas.text(tree_x + tree_w / 2, util.frontier_y(nodes) - 34, "observed generation", size=20, fill="#8a5d1f", anchor="middle")

    bracket_y = y + h - 82
    canvas.line(tree_x + 18, bracket_y, tree_x + tree_w - 18, bracket_y, stroke=PURPLE_DARK, stroke_width=1.15)
    canvas.line(tree_x + 18, bracket_y - 10, tree_x + 18, bracket_y, stroke=PURPLE_DARK, stroke_width=1.15)
    canvas.line(tree_x + tree_w - 18, bracket_y - 10, tree_x + tree_w - 18, bracket_y, stroke=PURPLE_DARK, stroke_width=1.15)
    canvas.text(tree_x + tree_w / 2, bracket_y + 30, "the complete state-labeled record is never observed", size=20, fill=PURPLE_DARK, anchor="middle")

    bx = x + w - 212
    canvas.rect(bx, y + 140, 158, 86, fill="#fffafa", stroke="#efd6d1", stroke_width=0.9, rx=8)
    canvas.text(bx + 18, y + 166, "missing", size=19, fill=RED, style="bold")
    canvas.text(bx + 18, y + 192, "cell targets", size=18, fill=RED)
    cross(canvas, bx + 134, y + 184, 8, RED)
    canvas.rect(bx, y + 266, 158, 124, fill="#fbfcfe", stroke="#dfe5ed", stroke_width=0.9, rx=8)
    canvas.text(bx + 18, y + 292, "available", size=19, fill=GREEN, style="bold")
    canvas.text(bx + 18, y + 318, "readout laws", size=18, fill=MUTED)
    check(canvas, bx + 132, y + 302, GREEN)
    scatter_cloud(canvas, bx + 50, y + 356, RED, n=12, sx=21, sy=11, opacity=0.55)
    scatter_cloud(canvas, bx + 108, y + 356, TEAL, n=12, sx=21, sy=11, opacity=0.55)
    canvas.text(bx + 79, y + 398, "match distributions", size=16, fill=MUTED, anchor="middle")


def draw_constraint_badge(canvas: Canvas, x: float, y: float, title: str, subtitle: str, color: str) -> None:
    canvas.rect(x, y, 218, 76, fill="#fbfcfe", stroke="#dfe5ed", stroke_width=0.9, rx=8)
    canvas.circle(x + 20, y + 24, 7, fill=color, stroke="#ffffff", stroke_width=0.7)
    canvas.text(x + 36, y + 24, title, size=19, fill="#2b303a")
    canvas.text(x + 20, y + 52, subtitle, size=16, fill=MUTED)


def draw_training_panel(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "c", "one latent complete-tree model, two training views", "observed projections constrain the model; masks turn completions into conditional samples")

    left_nodes = util.transform_nodes(util.base_tree_nodes(), x + 86, y + 226, 218, 286)
    observed_line(canvas, left_nodes, pad=16, height=26)
    ghost_tree(canvas, left_nodes, radius=6.9)
    canvas.text(x + 195, y + 158, "observed projection", size=22, fill="#2b303a", anchor="middle")
    canvas.text(x + 200, util.frontier_y(left_nodes) - 27, "destructive readout", size=17, fill="#8a5d1f", anchor="middle")
    canvas.text(x + 195, y + 532, "complete state is hidden", size=17, fill=MUTED, anchor="middle")

    law_cx = x + 640
    canvas.text(law_cx, y + 150, "latent complete-tree model", size=23, fill="#2b303a", anchor="middle")
    canvas.text(law_cx, y + 178, "p(completed record)", size=18, fill=MUTED, anchor="middle")
    canvas.rect(law_cx - 225, y + 204, 450, 342, fill="#fbfcfe", stroke="#e0e5ed", stroke_width=0.9, rx=8)
    law_nodes = util.transform_nodes(completion_variant(util.base_tree_nodes(), 2), law_cx - 168, y + 230, 336, 276)
    observed_line(canvas, law_nodes, pad=24, height=28)
    completed_tree(canvas, law_nodes, radius=7.1)
    canvas.text(law_cx, y + 526, "all states modeled jointly", size=17, fill=PURPLE_DARK, anchor="middle")

    canvas.arrow(x + 324, y + 386, law_cx - 238, y + 386, stroke="#9aa3b2", stroke_width=1.55)
    canvas.text(x + 370, y + 338, "match", size=17, fill=RED, anchor="middle")
    canvas.text(x + 370, y + 362, "readout laws", size=17, fill=RED, anchor="middle")

    mask_ids = {"r", "b", "b1f1", "b1f2", "b2f1", "b2f2"}
    mask_x = x + 1044
    sample_x = x + 1456
    canvas.text(mask_x + 70, y + 158, "mask/reveal view", size=22, fill="#2b303a", anchor="middle")
    canvas.text(mask_x + 70, y + 186, "context plus hidden targets", size=17, fill=MUTED, anchor="middle")
    draw_masked_tree(canvas, mask_x + 4, y + 246, 132, 210, 2, mask_ids, sampled=False)
    canvas.text(mask_x + 70, y + 496, "fixed context", size=16, fill=MUTED, anchor="middle")
    canvas.text(mask_x + 70, y + 522, "masked states", size=16, fill=TEAL, anchor="middle")

    canvas.text(sample_x + 70, y + 158, "conditional inpainting", size=22, fill="#2b303a", anchor="middle")
    canvas.text(sample_x + 70, y + 186, "sample masked states", size=17, fill=MUTED, anchor="middle")
    draw_masked_tree(canvas, sample_x + 4, y + 246, 132, 210, 6, mask_ids, sampled=True)
    canvas.text(sample_x + 70, y + 496, "context preserved", size=16, fill=MUTED, anchor="middle")
    canvas.text(sample_x + 70, y + 522, "sampled states", size=16, fill=TEAL, anchor="middle")

    canvas.arrow(law_cx + 238, y + 386, mask_x - 18, y + 386, stroke="#9aa3b2", stroke_width=1.55)
    canvas.text(x + 964, y + 338, "choose mask", size=18, fill=PURPLE_DARK, anchor="middle")
    canvas.text(x + 964, y + 362, "inside model", size=18, fill=PURPLE_DARK, anchor="middle")
    canvas.arrow(mask_x + 152, y + 386, sample_x - 18, y + 386, stroke="#9aa3b2", stroke_width=1.55)
    canvas.text((mask_x + sample_x) / 2 + 70, y + 336, "conditional sampling", size=20, fill=TEAL, anchor="middle", style="bold")
    canvas.text((mask_x + sample_x) / 2 + 70, y + 364, "p(masked states | context)", size=17, fill=MUTED, anchor="middle")

    canvas.line(x + 430, y + h - 82, x + 1230, y + h - 82, stroke="#c8ced9", stroke_width=1.0)
    canvas.text(x + w / 2, y + h - 48, "same model must match observable projections and sample every masked conditional", size=20, fill=PURPLE_DARK, anchor="middle")


def draw_alignment_panel(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "d", "distributional alignment", "compare comparable readouts across records")
    left = util.transform_nodes(util.base_tree_nodes(), x + 62, y + 132, 164, 236)
    right = util.transform_nodes(util.base_tree_nodes(), x + w - 226, y + 132, 164, 236)
    for nodes, label in ((left, "record r"), (right, "record r'")):
        observed_line(canvas, nodes, pad=12, height=20)
        ghost_tree(canvas, nodes, radius=5.8)
        canvas.text((min(n.x for n in nodes) + max(n.x for n in nodes)) / 2, y + 118, label, size=18, fill="#2b303a", anchor="middle")
    for nodes, color in ((left, RED), (right, TEAL)):
        focus = next(node for node in nodes if node.id == "a")
        canvas.circle(focus.x, focus.y, 10, fill=util.blend(color, "#ffffff", 0.45), stroke=color, stroke_width=1.25)
        for child_id in ("a1", "a2"):
            child = next(node for node in nodes if node.id == child_id)
            util.draw_cubic(canvas, (focus.x, focus.y + 6), (focus.x, focus.y + 43), (child.x, child.y - 43), (child.x, child.y - 6), stroke=color, stroke_width=1.35, opacity=0.65)
    canvas.text(x + w / 2, y + 180, "same time", size=17, fill=MUTED, anchor="middle")
    canvas.text(x + w / 2, y + 204, "same horizon", size=17, fill=MUTED, anchor="middle")
    canvas.arrow(x + w / 2 - 34, y + 246, x + w / 2 + 34, y + 246, stroke="#aeb6c2", stroke_width=1.2)

    def matrix_glyph(x0: float, y0: float, color: str) -> None:
        canvas.text(x0 + 46, y0 - 18, "features", size=15, fill=MUTED, anchor="middle")
        for r in range(6):
            for c in range(4):
                val = 0.24 + ((r * 5 + c * 3) % 7) / 10
                canvas.circle(x0 + c * 17, y0 + r * 12, 3.6, fill=util.blend(color, "#ffffff", 1 - val), stroke="none")
            canvas.line(x0 - 9, y0 + r * 12, x0 + 61, y0 + r * 12, stroke="#edf0f5", stroke_width=0.55)
        canvas.line(x0 - 16, y0 - 8, x0 - 16, y0 + 68, stroke="#7d8796", stroke_width=1.1)
        canvas.line(x0 + 75, y0 - 8, x0 + 75, y0 + 68, stroke="#7d8796", stroke_width=1.1)

    matrix_glyph(x + 80, y + 420, RED)
    matrix_glyph(x + w - 154, y + 420, TEAL)
    canvas.rect(x + w / 2 - 92, y + 426, 184, 58, fill=PURPLE_SOFT, stroke="#d8d1f7", stroke_width=0.9, rx=8)
    canvas.text(x + w / 2, y + 449, "energy distance", size=20, fill=PURPLE_DARK, anchor="middle", style="bold")
    canvas.text(x + w / 2, y + 473, "between laws", size=16, fill=MUTED, anchor="middle")
    canvas.text(x + w / 2, y + h - 44, "align laws, not cells", size=20, fill=PURPLE_DARK, anchor="middle")


def draw_routing_panel(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "e", "tree-aware evidence routing", "an ancestor is informed by all observed relatives")
    nodes = util.transform_nodes(util.base_tree_nodes()[:7], x + 76, y + 145, 215, 260)
    modes = {node.id: "blank" for node in nodes}
    for node in nodes:
        if node.kind == "observed":
            modes[node.id] = "observed"
    util.draw_tree(canvas, nodes, radius=7.8, edge_width=2.0, edge_opacity=0.55, node_mode=modes)
    target = next(node for node in nodes if node.id == "a")
    canvas.circle(target.x, target.y, 15, fill="none", stroke=PURPLE_DARK, stroke_width=1.5)
    for obs in observed_nodes(nodes):
        util.draw_cubic(canvas, (obs.x, obs.y - 10), (obs.x, obs.y - 60), (target.x, target.y + 62), (target.x, target.y + 13), stroke=PURPLE_DARK, stroke_width=0.9, opacity=0.22)
    canvas.text(target.x - 20, target.y - 30, "hidden", size=17, fill=PURPLE_DARK, anchor="end")
    canvas.text(target.x - 20, target.y - 8, "ancestor", size=17, fill=PURPLE_DARK, anchor="end")

    hx, hy = x + 318, y + 174
    canvas.text(hx + 78, hy - 30, "attention bias", size=19, fill=MUTED, anchor="middle")
    n = 6
    cell = 22
    for r in range(n):
        for c in range(n):
            dist = abs(r - c)
            val = max(0.14, 0.82 - 0.13 * dist)
            color = PURPLE if (r in (1, 2) or c in (1, 2)) else TEAL
            canvas.rect(hx + c * cell, hy + r * cell, cell - 2, cell - 2, fill=util.blend(color, "#ffffff", 1 - val), stroke="#ffffff", stroke_width=0.4, rx=2)
    canvas.text(hx + n * cell / 2, hy + n * cell + 31, "patristic distance", size=18, fill=MUTED, anchor="middle")
    canvas.text(x + w / 2, y + h - 52, "topology controls how evidence moves through the tree", size=20, fill=MUTED, anchor="middle")


def draw_attention_asset_panel(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "e", "tree-aware attention", "topology-biased attention routes lineage evidence")
    draw_embedded_png(canvas, ATTENTION_PNG, x + 18, y + 122, w - 36, h - 220, crop=(11, 27, 775, 545))
    canvas.text(x + w / 2, y + h - 48, "tree distance structures evidence flow", size=20, fill=PURPLE_DARK, anchor="middle")


def draw_inpainting_asset_panel(canvas: Canvas, x: float, y: float, w: float, h: float) -> None:
    panel(canvas, x, y, w, h, "f", "conditional sampling", "inpainting samples masked states from context")
    draw_embedded_png(canvas, INPAINTING_PNG, x + 8, y + 118, w - 16, h - 206, crop=(27, 17, 1233, 765))
    canvas.text(x + w / 2, y + h - 48, "sample unobserved ancestors and descendants", size=20, fill=TEAL, anchor="middle")


def draw_global(canvas: Canvas) -> None:
    canvas.rect(0, 0, W, H, fill=PAPER, stroke="none")

    margin = 70
    top_y, top_h = 70, 510
    mid_y, mid_h = 640, 720
    bottom_y, bottom_h = 1440, 560
    gap = 54
    col_w = (W - 2 * margin - gap) / 2
    bottom_w = (W - 2 * margin - 2 * gap) / 3

    util.draw_snapshot_strip(canvas, margin, top_y, col_w, top_h)
    util.draw_unobserved_target(canvas, margin + col_w + gap, top_y, col_w, top_h)
    canvas.arrow(margin + col_w + 12, top_y + top_h * 0.52, margin + col_w + gap - 12, top_y + top_h * 0.52, stroke="#9aa3b2", stroke_width=1.8)

    draw_training_panel(canvas, margin, mid_y, W - 2 * margin, mid_h)

    draw_alignment_panel(canvas, margin, bottom_y, bottom_w, bottom_h)
    draw_attention_asset_panel(canvas, margin + bottom_w + gap, bottom_y, bottom_w, bottom_h)
    draw_inpainting_asset_panel(canvas, margin + 2 * (bottom_w + gap), bottom_y, bottom_w, bottom_h)


def main() -> None:
    canvas = Canvas(W, H)
    draw_global(canvas)
    save(canvas)
    print(OUT_SVG.resolve())
    print(OUT_PNG.resolve())
    print(OUT_HTML.resolve())


if __name__ == "__main__":
    main()
