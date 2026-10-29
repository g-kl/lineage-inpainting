#!/usr/bin/env python3
"""Create a stacked tree-inpainting bridge figure.

The lopsided manuscript tree is laid sideways into two tilted planes. Hidden
nodes start from Gaussian source distributions and move along stochastic bridge
paths toward a completed tree, while observed nodes are clamped along straight
rails.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Render math/text labels as Computer Modern via matplotlib's built-in mathtext
# 'cm' fontset (no external LaTeX install required), embedded as vector paths.
matplotlib.rcParams["mathtext.fontset"] = "cm"

from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT
W, H = 1280, 840
# Physical paper width; height follows the viewBox aspect ratio. Keeps the
# figure at most the size of the attention-module figure (4.5in wide).
PAPER_WIDTH_IN = 4.5
# Labels: Computer Modern in near-black ink, matching the attention-module figure.
LABEL_FILL = "#202124"
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 28, 78, 1204, 746
# Crop the canvas a little above the (tightened) panel so the box hugs the
# content instead of leaving a tall empty band at the top.
VIEW_Y0 = 60
VIEW_H = H - VIEW_Y0


OBSERVED_IDS = {"rA1", "rA2", "rB1", "rB2", "rB3"}
BASE_NODES = [
    {"id": "r0", "parent": None, "x": 462, "y": 150, "color": "#5146a6", "seed": 3},
    {"id": "rA", "parent": "r0", "x": 349, "y": 300, "color": "#6b5ed8", "seed": 5},
    {"id": "rB", "parent": "r0", "x": 635, "y": 342, "color": "#8d80e6", "seed": 7},
    {"id": "rA1", "parent": "rA", "x": 260, "y": 575, "color": "#c95f45", "seed": 11},
    {"id": "rA2", "parent": "rA", "x": 438, "y": 615, "color": "#d89436", "seed": 13},
    {"id": "rB1", "parent": "rB", "x": 560, "y": 545, "color": "#9f9a3d", "seed": 17},
    {"id": "rB2", "parent": "rB", "x": 690, "y": 600, "color": "#4f9a66", "seed": 19},
    {"id": "rB3", "parent": "rB", "x": 770, "y": 635, "color": "#2f8fa4", "seed": 23},
]
FUTURE_OFFSETS = {
    "rA1": [(-28, 96), (30, 104)],
    "rA2": [(-26, 82), (28, 90)],
    "rB1": [(-22, 92), (25, 102)],
    "rB2": [(-24, 82), (26, 92)],
    "rB3": [(-26, 76), (25, 84)],
}
FUTURE_COLORS = {
    "rA1": ["#d66c52", "#c95f45"],
    "rA2": ["#d89436", "#e0a84c"],
    "rB1": ["#9f9a3d", "#b5ad4a"],
    "rB2": ["#4f9a66", "#72aa68"],
    "rB3": ["#2f8fa4", "#4677b8"],
}


def build_tree_nodes(u_margin: float, v_margin: float) -> list[dict]:
    nodes = [dict(node) for node in BASE_NODES]
    by_id = {node["id"]: node for node in nodes}
    for parent_id, offsets in FUTURE_OFFSETS.items():
        parent = by_id[parent_id]
        for index, (dx, _dy) in enumerate(offsets):
            nodes.append(
                {
                    "id": f"{parent_id}F{index + 1}",
                    "parent": parent_id,
                    "x": parent["x"] + dx,
                    "y": 755,
                    "color": FUTURE_COLORS[parent_id][index],
                    "seed": parent["seed"] + 100 + index * 17,
                }
            )
    out = []
    for node in nodes:
        item = dict(node)
        item["x"] = (node["x"] - 232) * 0.62
        item["y"] = (node["y"] - 150) * 0.56
        item["kind"] = "observed" if node["id"] in OBSERVED_IDS else "hidden"
        out.append(item)

    x_values = [node["x"] for node in out]
    y_values = [node["y"] for node in out]
    min_x, max_x = min(x_values), max(x_values)
    max_y = max(y_values)
    for item in out:
        depth = item["y"] / max_y
        lateral = (item["x"] - min_x) / (max_x - min_x)
        item["u"] = u_margin + (1 - 2 * u_margin) * depth
        item["v"] = v_margin + (1 - 2 * v_margin) * lateral
    return out


def point_on_quad(quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]], u: float, v: float) -> tuple[float, float]:
    p0, p1, _p2, p3 = [np.array(p) for p in quad]
    pt = p0 + u * (p1 - p0) + v * (p3 - p0)
    return float(pt[0]), float(pt[1])


def parameter_margins_for_clearance(
    quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
    clearance_px: float,
) -> tuple[float, float]:
    p0, p1, _p2, p3 = [np.array(p, dtype=float) for p in quad]
    u_vec = p1 - p0
    v_vec = p3 - p0
    area = abs(float(u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]))
    distance_between_u_edges = area / float(np.linalg.norm(v_vec))
    distance_between_v_edges = area / float(np.linalg.norm(u_vec))
    return clearance_px / distance_between_u_edges, clearance_px / distance_between_v_edges


def polygon_points(quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in quad)


def polyline_path(points: np.ndarray) -> str:
    head = f"M {points[0, 0]:.1f} {points[0, 1]:.1f}"
    tail = " ".join(f"L {x:.1f} {y:.1f}" for x, y in points[1:])
    return f"{head} {tail}"


def brownian_bridge_path(
    start: tuple[float, float],
    end: tuple[float, float],
    seed: int,
    steps: int = 30,
    sigma_x: float = 42,
    sigma_y: float = 18,
) -> str:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, steps)
    increments = rng.normal(0, 1 / np.sqrt(steps), size=(steps - 1, 2))
    bm = np.vstack([np.zeros(2), np.cumsum(increments, axis=0)])
    bridge = bm - t[:, None] * bm[-1]
    start_arr = np.array(start)
    end_arr = np.array(end)
    linear = start_arr[None, :] * (1 - t[:, None]) + end_arr[None, :] * t[:, None]
    points = linear + bridge * np.array([sigma_x, sigma_y])
    return polyline_path(points)


def source_sample_point(
    center: tuple[float, float],
    seed: int,
    sigma_x: float = 12.0,
    sigma_y: float = 6.5,
    angle_deg: float = -5.0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, size=2) * np.array([sigma_x, sigma_y])
    angle = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pt = np.array(center) + rot @ z
    return float(pt[0]), float(pt[1])


def add_gaussian_source(parts: list[str], x: float, y: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    parts.append(f'<g transform="translate({x:.1f} {y:.1f}) rotate(-5)">')
    parts.append('<ellipse cx="0" cy="0" rx="19.0" ry="10.8" fill="#f2eeff" stroke="#8d80e6" stroke-width="1.25" opacity="0.86"/>')
    parts.append('<ellipse cx="0" cy="0" rx="12.8" ry="7.1" fill="none" stroke="#8d80e6" stroke-width="1.0" opacity="0.62"/>')
    parts.append('<ellipse cx="0" cy="0" rx="6.5" ry="3.7" fill="none" stroke="#8d80e6" stroke-width="0.9" opacity="0.44"/>')
    for _ in range(12):
        px = rng.normal(0, 6.2)
        py = rng.normal(0, 3.3)
        radius = rng.uniform(1.05, 1.8)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="#6b5ed8" opacity="0.48"/>')
    parts.append("</g>")


def tree_point(
    node: dict,
    quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    return point_on_quad(quad, node["u"], node["v"])


def tree_edge_path(
    parent: dict,
    child: dict,
    quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
) -> str:
    mid_u = (parent["u"] + child["u"]) / 2
    p0 = point_on_quad(quad, parent["u"], parent["v"])
    c1 = point_on_quad(quad, mid_u, parent["v"])
    c2 = point_on_quad(quad, mid_u, child["v"])
    p1 = point_on_quad(quad, child["u"], child["v"])
    return (
        f"M {p0[0]:.1f} {p0[1]:.1f} "
        f"C {c1[0]:.1f} {c1[1]:.1f}, {c2[0]:.1f} {c2[1]:.1f}, {p1[0]:.1f} {p1[1]:.1f}"
    )


def latex_label_svg(
    x: float,
    y: float,
    expression: str,
    size: int,
    rotate: float = 0.0,
    anchor: str = "middle",
    fill: str = "#667085",
) -> str:
    text_path = TextPath((0, 0), expression, size=size, usetex=False)
    bbox = text_path.get_extents()
    if anchor == "start":
        anchor_shift = -bbox.x0
    elif anchor == "end":
        anchor_shift = -bbox.x1
    else:
        anchor_shift = -(bbox.x0 + bbox.x1) / 2

    angle = np.deg2rad(rotate)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))

    def transform_point(px: float, py: float) -> tuple[float, float]:
        local_x = px + anchor_shift
        local_y = -py
        return (
            x + cos_a * local_x - sin_a * local_y,
            y + sin_a * local_x + cos_a * local_y,
        )

    commands = []
    for verts, code in text_path.iter_segments():
        points = [transform_point(float(verts[i]), float(verts[i + 1])) for i in range(0, len(verts), 2)]
        if code == MplPath.MOVETO:
            commands.append(f"M {points[0][0]:.2f} {points[0][1]:.2f}")
        elif code == MplPath.LINETO:
            commands.append(f"L {points[0][0]:.2f} {points[0][1]:.2f}")
        elif code == MplPath.CURVE3:
            commands.append(f"Q {points[0][0]:.2f} {points[0][1]:.2f} {points[1][0]:.2f} {points[1][1]:.2f}")
        elif code == MplPath.CURVE4:
            commands.append(
                f"C {points[0][0]:.2f} {points[0][1]:.2f} {points[1][0]:.2f} {points[1][1]:.2f} "
                f"{points[2][0]:.2f} {points[2][1]:.2f}"
            )
        elif code == MplPath.CLOSEPOLY:
            commands.append("Z")
    return f'<path d="{" ".join(commands)}" fill="{fill}"/>'


def label_angle(start: tuple[float, float], end: tuple[float, float]) -> float:
    return float(np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0])))


def make_svg() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tree_bottom = ((150, 500), (808, 450), (1090, 664), (432, 718))
    tree_top = ((198, 152), (856, 102), (1138, 316), (480, 370))
    tree_mid = ((174, 326), (832, 276), (1114, 490), (456, 544))
    u_margin, v_margin = parameter_margins_for_clearance(tree_bottom, clearance_px=48)

    nodes = build_tree_nodes(u_margin, v_margin)
    by_id = {node["id"]: node for node in nodes}

    hidden_nodes = [n for n in nodes if n["kind"] == "hidden"]
    tree_axis_start = point_on_quad(tree_bottom, 0.06, 1.27)
    tree_axis_end = point_on_quad(tree_bottom, 0.91, 1.27)
    tree_axis_mid = ((tree_axis_start[0] + tree_axis_end[0]) / 2, (tree_axis_start[1] + tree_axis_end[1]) / 2)
    tree_axis_angle = label_angle(tree_axis_start, tree_axis_end)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAPER_WIDTH_IN}in" height="{PAPER_WIDTH_IN * VIEW_H / W:.4f}in" viewBox="0 {VIEW_Y0} {W} {VIEW_H}">',
        "<defs>",
        '<marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#202124"/></marker>',
        "</defs>",
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
        f'<rect x="{PANEL_X}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" rx="10" fill="#ffffff" stroke="#d9dde5" stroke-width="1.5"/>',
        '<path d="M 100 610 L 100 172" fill="none" stroke="#202124" stroke-width="2.88" marker-end="url(#arrowhead)"/>',
        latex_label_svg(74, 391, r"$\mathrm{bridge\ time}$", 41, rotate=-90, fill=LABEL_FILL),
        latex_label_svg(118, 178, r"$s=1$", 38, anchor="start", fill=LABEL_FILL),
        latex_label_svg(118, 613, r"$s=0$", 38, anchor="start", fill=LABEL_FILL),
    ]

    # Tree stack.
    parts.extend(
        [
            f'<polygon points="{polygon_points(tree_bottom)}" fill="#fbfcfe" stroke="#d9dde5" stroke-width="1.8"/>',
            f'<polygon points="{polygon_points(tree_mid)}" fill="#fbfcfe" stroke="none" opacity="0.36"/>',
            f'<polygon points="{polygon_points(tree_top)}" fill="#fbfcfe" stroke="#d9dde5" stroke-width="1.8"/>',
            f'<path d="M {tree_axis_start[0]:.1f} {tree_axis_start[1]:.1f} L {tree_axis_end[0]:.1f} {tree_axis_end[1]:.1f}" fill="none" stroke="#202124" stroke-width="2.64" marker-end="url(#arrowhead)" opacity="1"/>',
            latex_label_svg(tree_axis_start[0] - 16, tree_axis_start[1] + 30, r"$t=t_0$", 38, rotate=tree_axis_angle, anchor="end", fill=LABEL_FILL),
            latex_label_svg(tree_axis_end[0] + 18, tree_axis_end[1] + 30, r"$t=t_1$", 38, rotate=tree_axis_angle, anchor="start", fill=LABEL_FILL),
            latex_label_svg(tree_axis_mid[0] + 2, tree_axis_mid[1] + 58, r"$\mathrm{tree\ time}$", 41, rotate=tree_axis_angle, fill=LABEL_FILL),
        ]
    )

    for node in hidden_nodes:
        bottom_center = tree_point(node, tree_bottom)
        top_center = tree_point(node, tree_top)
        for sample_index in range(4):
            start = source_sample_point(bottom_center, node["seed"] + 500 + sample_index * 101)
            end = source_sample_point(top_center, node["seed"] + 1300 + sample_index * 149, sigma_x=4.0, sigma_y=2.5)
            parts.append(f'<path d="{brownian_bridge_path(start, end, node["seed"] + 400 + sample_index * 211, steps=64, sigma_x=22, sigma_y=17)}" fill="none" stroke="#6b5ed8" stroke-width="1.26" stroke-linecap="round" stroke-linejoin="round" opacity="0.18"/>')

    def add_tree_layer(
        name: str,
        quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
        opacity: float,
        mode: str,
    ) -> None:
        edge_stroke = "#d7dce4" if mode == "noise" else "#bfc7d3"
        edge_width = 3.36 if mode == "noise" else 3.84

        parts.append(f'<g id="{name}" opacity="{opacity}">')
        for node in nodes:
            if node["parent"] is None:
                continue
            parent = by_id[node["parent"]]
            parts.append(f'<path d="{tree_edge_path(parent, node, quad)}" fill="none" stroke="{edge_stroke}" stroke-width="{edge_width}" stroke-linecap="round" stroke-linejoin="round"/>')
        for node in nodes:
            x, y = tree_point(node, quad)
            if node["kind"] == "observed":
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15.0" fill="{node["color"]}" stroke="#ffffff" stroke-width="3.84"/>')
            elif mode == "noise":
                add_gaussian_source(parts, x, y, node["seed"] + 50)
            elif mode == "sample":
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="21.6" fill="{node["color"]}" opacity="0.14"/>')
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13.8" fill="{node["color"]}" stroke="#ffffff" stroke-width="3.84"/>')
        parts.append("</g>")

    add_tree_layer("tree-bottom-noise", tree_bottom, 0.76, "noise")
    add_tree_layer("tree-top-sample", tree_top, 1.0, "sample")

    parts.append("</svg>")
    out_path = OUT_DIR / "stacked_bridge_inpainting_figure.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def make_html(svg_path: Path) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tree bridge inpainting figure</title>
  <style>
    html, body {{ margin: 0; background: #ffffff; }}
    img {{ display: block; width: 100vw; height: auto; }}
  </style>
</head>
<body>
  <img src="{svg_path.name}" alt="Tree bridge inpainting figure">
</body>
</html>
"""
    (OUT_DIR / "stacked_bridge_inpainting_figure.html").write_text(html, encoding="utf-8")


def render_png(svg_path: Path) -> Path:
    import cairosvg

    out_path = OUT_DIR / "stacked_bridge_inpainting_figure.png"
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(out_path),
        output_width=W,
        output_height=VIEW_H,
    )
    return out_path


def main() -> None:
    svg_path = make_svg()
    make_html(svg_path)
    png_path = render_png(svg_path)
    print(svg_path.resolve())
    print(png_path.resolve())


if __name__ == "__main__":
    main()
