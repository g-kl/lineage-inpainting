"""Render the Bernstein product and its two balanced-tree placements."""

from __future__ import annotations

from dataclasses import dataclass
import html
import math
from pathlib import Path

from analysis.figures.theory import style as S
from utils.figure_labels import text_svg_path


WIDTH, HEIGHT = 6.30, 1.80
CANVAS_W, CANVAS_H = 1512, 432
PT_PER_IN = 72.0

PANELS = S.PANELS

INK = "#2b303a"
MUTED = "#667085"
LEADER = "#aeb6c2"
EDGE = "#d7dce4"
LAVENDER = "#e7e2f7"
LAVENDER_STROKE = "#b4aae0"
PURPLE = "#6b5ed8"
ORANGE = "#c47a3d"
POSITIVE = "#d65a3b"
NEGATIVE = "#327fa6"
WHITE = "#ffffff"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _tag(name: str, attrs: dict[str, object | None],
         body: str | None = None) -> str:
    clean = []
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, float):
            value = f"{value:.2f}".rstrip("0").rstrip(".")
        clean.append(f'{key}="{_esc(value)}"')
    head = f"<{name} {' '.join(clean)}>"
    if body is None:
        return head[:-1] + "/>"
    return f"{head}{body}</{name}>"


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.body: list[str] = []

    def text(self, x: float, y: float, value: str, *, size: float = 24,
             fill: str = INK, anchor: str = "start",
             weight: str = "normal") -> None:
        self.body.append(text_svg_path(
            value, x, y, size, color=fill, weight=weight,
            anchor=anchor, valign="center",
        ))

    def line(self, x1: float, y1: float, x2: float, y2: float, *,
             stroke: str = LEADER, width: float = 2.0,
             dash: str | None = None) -> None:
        self.body.append(_tag("line", {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "stroke": stroke, "stroke-width": width,
            "stroke-linecap": "round", "stroke-dasharray": dash,
        }))

    def path(self, d: str, *, stroke: str, width: float,
             fill: str = "none", dash: str | None = None) -> None:
        self.body.append(_tag("path", {
            "d": d, "fill": fill, "stroke": stroke,
            "stroke-width": width, "stroke-linecap": "round",
            "stroke-linejoin": "round", "stroke-dasharray": dash,
        }))

    def circle(self, x: float, y: float, r: float, *, fill: str,
               stroke: str = "none", width: float = 1.0) -> None:
        self.body.append(_tag("circle", {
            "cx": x, "cy": y, "r": r, "fill": fill,
            "stroke": stroke, "stroke-width": width,
        }))

    def brace_h(self, x_left: float, x_right: float, y: float,
                depth: float = 9) -> None:
        mid = (x_left + x_right) / 2
        q = min(14, (x_right - x_left) * 0.16)
        y_top = y + depth
        y_cusp = y_top + 6
        d = (
            f"M {x_left:.1f} {y:.1f} Q {x_left:.1f} {y_top:.1f} "
            f"{x_left + q:.1f} {y_top:.1f} L {mid - 6:.1f} {y_top:.1f} "
            f"Q {mid:.1f} {y_top:.1f} {mid:.1f} {y_cusp:.1f} "
            f"Q {mid:.1f} {y_top:.1f} {mid + 6:.1f} {y_top:.1f} "
            f"L {x_right - q:.1f} {y_top:.1f} Q {x_right:.1f} {y_top:.1f} "
            f"{x_right:.1f} {y:.1f}"
        )
        self.path(d, stroke="#8a93a3", width=1.8)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        physical_width = WIDTH * PT_PER_IN
        physical_height = HEIGHT * PT_PER_IN
        scale_x = physical_width / self.width
        scale_y = physical_height / self.height
        if not math.isclose(scale_x, scale_y, rel_tol=2e-3):
            raise ValueError("SVG canvas and physical size have different aspects")
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{physical_width:.2f}pt" height="{physical_height:.2f}pt"
     viewBox="0 0 {physical_width:.2f} {physical_height:.2f}" role="img"
     aria-label="Bernstein product embedded as path and common-ancestor interactions">
<g transform="scale({scale_x:.6f})">
{chr(10).join(self.body)}
</g>
</svg>
'''
        path.write_text(svg, encoding="utf-8")


@dataclass(frozen=True)
class Node:
    index: int
    parent: int | None
    x: float
    y: float


def _balanced_tree(center_x: float) -> list[Node]:
    """Return the same balanced depth-three tree used by the calculations."""
    nodes = []
    spacings = (0.0, 144.0, 76.0, 38.0)
    for index in range(15):
        level = int(math.floor(math.log2(index + 1)))
        within = index - (2**level - 1)
        offset = (within - (2**level - 1) / 2) * spacings[level]
        nodes.append(Node(
            index=index,
            parent=None if index == 0 else (index - 1) // 2,
            x=center_x + offset,
            y=90 + 76 * level,
        ))
    return nodes


def _edge_path(parent: Node, child: Node) -> str:
    """Draw a curved edge with vertical tangents at both ends."""
    mid = (parent.y + child.y) / 2
    return (
        f"M {parent.x:.1f} {parent.y:.1f} "
        f"C {parent.x:.1f} {mid:.1f}, {child.x:.1f} {mid:.1f}, "
        f"{child.x:.1f} {child.y:.1f}"
    )


def _state_mark(canvas: Canvas, x: float, y: float, state: int, *,
                color: str, arm: float, width: float) -> None:
    canvas.line(x - arm, y, x + arm, y, stroke=color, width=width)
    if state > 0:
        canvas.line(x, y - arm, x, y + arm, stroke=color, width=width)


def _state_disk(canvas: Canvas, x: float, y: float, state: int, *,
                radius: float = 18.0, determined: bool = False) -> None:
    canvas.circle(
        x, y, radius,
        fill=POSITIVE if state > 0 else NEGATIVE,
        stroke=WHITE, width=1.6,
    )
    _state_mark(
        canvas, x, y, state,
        color=WHITE, arm=0.36 * radius, width=max(1.8, 0.13 * radius),
    )
    if determined:
        canvas.circle(x, y, radius + 7, fill="none", stroke=PURPLE, width=3.5)


def _draw_product(canvas: Canvas) -> None:
    """Draw the four-variable Bernstein constraint at full dependence."""
    xs = (78, 164, 250, 358)
    signs = (1, -1, -1, 1)
    for index, (xx, sign) in enumerate(zip(xs, signs), start=1):
        _state_disk(canvas, xx, 147, sign, determined=index == 4)
        canvas.text(xx, 194, rf"$x_{index}$", size=25, anchor="middle")
    canvas.text(121, 147, "×", size=30, fill=MUTED, anchor="middle")
    canvas.text(207, 147, "×", size=30, fill=MUTED, anchor="middle")
    canvas.text(304, 147, "=", size=30, fill=MUTED, anchor="middle")
    canvas.brace_h(53, 275, 213, depth=9)
    canvas.text(164, 251, "independent", size=23, fill=MUTED, anchor="middle")
    canvas.text(358, 251, "determined", size=23, fill=PURPLE, anchor="middle")
    canvas.text(211, 326, r"$x_4=x_1x_2x_3$", size=32, anchor="middle")


def _draw_tree(canvas: Canvas, center_x: float, *, support: tuple[int, ...],
               signs: tuple[int, ...], color: str, geometry: str) -> None:
    """Draw one order-four support without implying a full-tree realization."""
    nodes = _balanced_tree(center_x)
    by_index = {node.index: node for node in nodes}
    support_set = set(support)

    for node in nodes:
        if node.parent is None:
            continue
        parent = by_index[node.parent]
        on_path = (
            geometry == "depth"
            and node.index in support_set
            and node.parent in support_set
        )
        canvas.path(
            _edge_path(parent, node),
            stroke=color if on_path else EDGE,
            width=5.0 if on_path else 3.7,
        )

    support_signs = dict(zip(support, signs))
    for node in nodes:
        if node.index in support_set:
            canvas.circle(node.x, node.y, 18.0, fill="none", stroke=color, width=3.6)
            _state_disk(
                canvas, node.x, node.y, support_signs[node.index],
                radius=12.0,
            )
        else:
            canvas.circle(
                node.x, node.y, 11.5,
                fill=LAVENDER, stroke=LAVENDER_STROKE, width=2.5,
            )


def render(out_dir=None):
    canvas = Canvas(CANVAS_W, CANVAS_H)

    canvas.text(211, 42, "Bernstein product", size=31, anchor="middle")
    _draw_product(canvas)

    canvas.path(
        "M 430 213 L 474 213 M 463 202 L 474 213 L 463 224",
        stroke=LEADER, width=2.3,
    )

    canvas.text(747, 42, "path", size=30, anchor="middle")
    _draw_tree(
        canvas, 747,
        support=(0, 2, 5, 11), signs=(1, -1, -1, 1),
        color=PURPLE, geometry="depth",
    )

    canvas.text(1214, 42, "common ancestor", size=30, anchor="middle")
    _draw_tree(
        canvas, 1214,
        support=(1, 7, 8, 9), signs=(1, -1, -1, 1),
        color=ORANGE, geometry="breadth",
    )

    output = Path(out_dir) if out_dir is not None else PANELS
    path = output / "a_concept.svg"
    canvas.write(path)
    return path, (WIDTH, HEIGHT)


if __name__ == "__main__":
    path, _ = render()
    print("wrote", path)
