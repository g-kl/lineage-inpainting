"""Render the Ising running example of the theory supplement.

Two pages: the binary-table story (fork laws, feature tables, the compatible
set) and the dependence--risk phase maps on the seven-node tree.  The
calculations come from depth2.py and its certified solver results.

Run from the repository root with

    python -m analysis.figures.theory.plot_ising

which writes imgs/ising_running_example.pdf.
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon,
    Rectangle, Wedge,
)
from matplotlib.path import Path as MplPath
import numpy as np

from analysis.figures.theory import depth2
from analysis.figures.theory import style as S
from utils.figure_labels import FONT_FAMILY, register_fonts


OUT = S.IMGS / "ising_running_example.pdf"

# Color roles shared with the Bernstein figure.
INK = "#222222"
MUTED = "#777777"
LIGHT = "#aeb6c2"
EDGE = "#d7dce4"
BOX_BORDER = "#b8c1cf"
PANEL_FILL = "#fbfcfc"
SELECT = "#6b5ed8"
TEAL = "#0f6b6e"
TEAL_LIGHT = "#cfe5e3"
PURPLE = "#7a3d8c"
ORANGE = "#c47a3d"
MASS = "#5f6673"
POSITIVE = "#d65a3b"
NEGATIVE = "#327fa6"
ZERO = "#f2f2f4"
WHITE = "#ffffff"
GROUP_COLORS = ("#c76f8a", "#d2a13a", "#4a978a", "#557fb3")

PAGE_W = 6.5
UNITS_PER_IN = 100.0
FS_BOX = 9.3
FS_TITLE = 8.7
FS_LABEL = 8.0
FS_SMALL = 7.6
FS_TINY = 7.6
FS_PANEL = 16.0



def _register_fonts() -> None:
    """Register the repository's Latin Modern Sans faces, shared by every figure."""
    register_fonts()


def _apply_style() -> None:
    _register_fonts()
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.solid_capstyle": "round",
    })


def page(height: float, title: str):
    fig = plt.figure(figsize=(PAGE_W, height), facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    h = height * UNITS_PER_IN
    ax.set_xlim(0, PAGE_W * UNITS_PER_IN)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    border = FancyBboxPatch(
        (2.5, 2.5), PAGE_W * UNITS_PER_IN - 5, h - 5,
        boxstyle="round,pad=0,rounding_size=5",
        facecolor=WHITE, edgecolor=BOX_BORDER, linewidth=0.7, zorder=0,
    )
    ax.add_patch(border)
    ax.text(
        PAGE_W * UNITS_PER_IN / 2, h - 10.5, title,
        ha="center", va="center", fontsize=FS_BOX, color=INK,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 2.2},
        zorder=20,
    )
    return fig, ax, h


def panel(ax, x: float, y: float, letter: str, title: str) -> None:
    ax.text(x, y, letter, ha="left", va="top", fontsize=FS_PANEL,
            fontweight="bold", color=INK)
    ax.text(x + 28, y - 4, title, ha="left", va="top",
            fontsize=FS_TITLE, color=INK)


def arrow(ax, start, end, *, color=LIGHT, width=1.2, mutation=8,
          connectionstyle="arc3,rad=0", linestyle="-") -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=width, color=color, linestyle=linestyle,
        connectionstyle=connectionstyle, shrinkA=0, shrinkB=0,
    ))


def curved_edge(ax, start, end, *, color=EDGE, width=2.1) -> None:
    x0, y0 = start
    x1, y1 = end
    mid = (y0 + y1) / 2
    path = MplPath(
        [(x0, y0), (x0, mid), (x1, mid), (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color,
                           linewidth=width, capstyle="round"))


def state_disk(ax, x: float, y: float, value: int, *, radius=9.0,
               selected=False, selected_color=SELECT, faded=False) -> None:
    if selected:
        ax.add_patch(Circle((x, y), radius + 4.0, facecolor="none",
                            edgecolor=selected_color, linewidth=1.8, zorder=4))
    fill = POSITIVE if value > 0 else NEGATIVE
    alpha = 0.28 if faded else 1.0
    ax.add_patch(Circle((x, y), radius, facecolor=fill, edgecolor=WHITE,
                        linewidth=0.8, alpha=alpha, zorder=5))
    arm = 0.38 * radius
    ax.plot((x - arm, x + arm), (y, y), color=WHITE, linewidth=1.15,
            alpha=alpha, zorder=6)
    if value > 0:
        ax.plot((x, x), (y - arm, y + arm), color=WHITE, linewidth=1.15,
                alpha=alpha, zorder=6)


def support_fork_glyph(
    ax, x: float, y: float, support: tuple[int, ...], *, scale: float = 1.0,
) -> None:
    """Draw one selected node subset on a tiny three-node fork."""
    positions = ((x, y + 5.5 * scale),
                 (x - 5.2 * scale, y - 3.0 * scale),
                 (x + 5.2 * scale, y - 3.0 * scale))
    for child in (1, 2):
        ax.plot(
            (positions[0][0], positions[child][0]),
            (positions[0][1], positions[child][1]),
            color=EDGE, linewidth=0.8 * scale, zorder=2,
        )
    for node, (node_x, node_y) in enumerate(positions):
        selected = node in support
        ax.add_patch(Circle(
            (node_x, node_y), 1.9 * scale,
            facecolor=SELECT if selected else WHITE,
            edgecolor=SELECT if selected else BOX_BORDER,
            linewidth=0.55 * scale, zorder=3,
        ))


def assignment_quadruplet(ax, x: float, y: float, signs, *, scale=1.0) -> None:
    """Draw one four-sign assignment class without an additional label."""
    for offset, sign in zip((-6.3, -2.1, 2.1, 6.3), signs):
        state_disk(ax, x + scale * offset, y, sign, radius=2.5 * scale)


def fair_sign_disk(ax, x: float, y: float, *, radius: float = 6.0) -> None:
    """Draw one fair sign as a single split positive/negative token."""
    ax.add_patch(Wedge(
        (x, y), radius, 90, 270,
        facecolor=POSITIVE, edgecolor="none", zorder=5,
    ))
    ax.add_patch(Wedge(
        (x, y), radius, -90, 90,
        facecolor=NEGATIVE, edgecolor="none", zorder=5,
    ))
    ax.add_patch(Circle(
        (x, y), radius, facecolor="none", edgecolor=WHITE,
        linewidth=0.7, zorder=6,
    ))
    ax.text(x - 0.42 * radius, y - 0.15, "+", ha="center", va="center",
            fontsize=FS_TINY - 1.1, fontweight="bold", color=WHITE, zorder=7)
    ax.text(x + 0.42 * radius, y - 0.15, "−", ha="center", va="center",
            fontsize=FS_TINY - 1.1, fontweight="bold", color=WHITE, zorder=7)


def _draw_iid_bracket(ax, xs: tuple[float, ...], y: float) -> None:
    """Mark a row of split sign tokens as independent with minimal notation."""
    left = xs[0] - 6.0
    right = xs[-1] + 6.0
    ax.plot((left, right), (y, y), color=MUTED, linewidth=0.55, zorder=2)
    ax.plot((left, left), (y, y + 3.0), color=MUTED, linewidth=0.55, zorder=2)
    ax.plot((right, right), (y, y + 3.0), color=MUTED, linewidth=0.55, zorder=2)
    ax.text((left + right) / 2, y - 4.5, r"$\mathrm{iid}$",
            ha="center", va="top", fontsize=FS_TINY, color=MUTED)


PROBABILITY_TETRAHEDRON = np.asarray((
    (16.0, 0.0),
    (0.0, 43.0),
    (-47.0, -31.0),
    (47.0, -31.0),
))


def _probability_tetrahedron_vertices(
    x: float, y: float, *, scale: float = 1.0,
) -> np.ndarray:
    return np.asarray((x, y)) + scale * PROBABILITY_TETRAHEDRON


NEGATIVE_SUBSETS = ((), (2,), (1,), (1, 2), (0,), (0, 2), (0, 1), (0, 1, 2))
MOMENT_SUPPORTS = ((0, 1), (0, 2), (1, 2), (0, 1, 2))
ASSIGNMENTS = tuple(
    tuple(-1 if node in negative_nodes else 1 for node in range(3))
    for negative_nodes in NEGATIVE_SUBSETS
)
P_PROB = (0.125,) * 8
Q_PROB = (0.25, 0, 0, 0.25, 0, 0.25, 0.25, 0)
DP = tuple(p - q for p, q in zip(P_PROB, Q_PROB))
PROBABILITY_ROWS = np.asarray((P_PROB, Q_PROB), dtype=float)
if not np.allclose(PROBABILITY_ROWS.sum(axis=1), 1):
    raise RuntimeError("fork probability rows must sum to one")
if not np.allclose(PROBABILITY_ROWS @ np.asarray(ASSIGNMENTS), 0):
    raise RuntimeError("both fork laws must have fair singleton marginals")
MOMENT_SIGNATURES = np.asarray([
    [math.prod(signs[node] for node in support)
     for support in MOMENT_SUPPORTS]
    for signs in ASSIGNMENTS
], dtype=int)
MOMENT_DM = np.asarray(DP) @ MOMENT_SIGNATURES
if not np.allclose(MOMENT_DM, (0, 0, 0, -1)):
    raise RuntimeError("fork moment difference does not match the derivation")


def render_story(pdf: PdfPages) -> None:
    """Show the Bernstein fork and its whole-versus-path comparison."""
    fig, ax, h = page(
        5.30, "Partial identification on a fork",
    )

    def brackets(left: float, right: float, bottom: float, top: float,
                 *, color=MUTED, width=0.6) -> None:
        arm = 4.0
        for xx, direction in ((left, 1), (right, -1)):
            ax.plot((xx, xx), (bottom, top), color=color, linewidth=width)
            ax.plot((xx, xx + direction * arm), (top, top),
                    color=color, linewidth=width)
            ax.plot((xx, xx + direction * arm), (bottom, bottom),
                    color=color, linewidth=width)

    def draw_triplet(center_x: float, center_y: float, signs,
                     *, scale=1.0) -> None:
        for offset, sign in zip((-6.0, 0.0, 6.0), signs):
            state_disk(ax, center_x + scale * offset, center_y, sign,
                       radius=2.8 * scale)

    def draw_fork(
        center_x: float, center_y: float, signs,
        *, derived_node: int | None = None,
    ) -> None:
        root = (center_x, center_y + 24)
        left = (center_x - 20, center_y - 7)
        right = (center_x + 20, center_y - 7)
        curved_edge(ax, root, left, color=EDGE, width=1.7)
        curved_edge(ax, root, right, color=EDGE, width=1.7)
        for node, (position, sign) in enumerate(
            zip((root, left, right), signs)
        ):
            if node == derived_node:
                ax.add_patch(Circle(
                    position, 9.0, facecolor="none", edgecolor=ORANGE,
                    linewidth=0.85, zorder=4,
                ))
            if sign is None:
                fair_sign_disk(ax, *position, radius=6.8)
            else:
                state_disk(ax, *position, sign, radius=6.8)

    def draw_cell(center_x: float, center_y: float, value: float,
                  *, width=20.0, height=20.0, face=None, edge=WHITE,
                  show_zero=True, text_color=None, fraction=False) -> None:
        if face is None:
            if value > 0:
                face = POSITIVE
            elif value < 0:
                face = NEGATIVE
            else:
                face = ZERO
        if text_color is None:
            text_color = WHITE if not np.isclose(value, 0.0) else MUTED
        ax.add_patch(Rectangle(
            (center_x - width / 2, center_y - height / 2), width, height,
            facecolor=face, edgecolor=edge, linewidth=0.65,
        ))
        if np.isclose(value, 0.0) and not show_zero:
            return
        if fraction and np.isclose(abs(value), 0.5):
            label = r"$1/2$" if value > 0 else r"$-1/2$"
        elif fraction and np.isclose(abs(value), 0.25):
            label = r"$1/4$" if value > 0 else r"$-1/4$"
        elif fraction and np.isclose(abs(value), 0.125):
            label = r"$1/8$" if value > 0 else r"$-1/8$"
        else:
            label = f"{int(value):d}" if np.isclose(value, round(value)) else "½"
        ax.text(center_x, center_y - 0.2, label, ha="center", va="center",
                fontsize=FS_TINY, color=text_color)

    def draw_horizontal_vector(center_x: float, center_y: float, values,
                               *, cell=18.0, gap=1.5,
                               bracket_color=MUTED) -> None:
        offset = (len(values) - 1) / 2
        xs = tuple(center_x + (index - offset) * (cell + gap)
                   for index in range(len(values)))
        for xx, value in zip(xs, values):
            draw_cell(xx, center_y, float(value), width=cell, height=cell)
        brackets(xs[0] - cell / 2 - 4, xs[-1] + cell / 2 + 4,
                 center_y - cell / 2 - 3, center_y + cell / 2 + 3,
                 color=bracket_color)

    def draw_vertical_vector(center_x: float, center_y: float, values,
                             *, cell=18.0, gap=1.5,
                             bracket_color=MUTED) -> None:
        offset = (len(values) - 1) / 2
        ys = tuple(center_y + (offset - index) * (cell + gap)
                   for index in range(len(values)))
        for yy, value in zip(ys, values):
            draw_cell(center_x, yy, float(value), width=cell, height=cell)
        brackets(center_x - cell / 2 - 4, center_x + cell / 2 + 4,
                 ys[-1] - cell / 2 - 3, ys[0] + cell / 2 + 3,
                 color=bracket_color)

    def draw_matrix(center_x: float, center_y: float, matrix: np.ndarray,
                    *, color: str, cell=18.0) -> None:
        rows, columns = matrix.shape
        xs = tuple(
            center_x + (index - (columns - 1) / 2) * cell
            for index in range(columns)
        )
        ys = tuple(
            center_y + ((rows - 1) / 2 - index) * cell
            for index in range(rows)
        )
        for row, yy in enumerate(ys):
            for column, xx in enumerate(xs):
                value = float(matrix[row, column])
                active = not np.isclose(value, 0.0)
                face = mpl.colors.to_rgba(color, 0.78) if active else ZERO
                label_color = WHITE if active else MUTED
                draw_cell(xx, yy, value, width=cell - 1.0,
                          height=cell - 1.0, face=face, edge=WHITE,
                          text_color=label_color)
        brackets(xs[0] - cell / 2 - 4, xs[-1] + cell / 2 + 4,
                 ys[-1] - cell / 2 - 3, ys[0] + cell / 2 + 3,
                 color=INK)

    moment_map = MOMENT_SIGNATURES.T.astype(float)
    probability_matrix = PROBABILITY_ROWS.T
    moment_matrix = moment_map @ probability_matrix
    expected_moments = np.asarray(
        ((0, 0), (0, 0), (0, 0), (0, 1)), dtype=float,
    )
    if not np.allclose(moment_matrix, expected_moments):
        raise RuntimeError("fork product-moment matrix is inconsistent")

    whole_metric = np.eye(4)
    u = np.asarray((1.0, 1.0, 0.0, 0.0)) / math.sqrt(2.0)
    parent_child_metric = np.outer(u, u)
    if not np.allclose(parent_child_metric, parent_child_metric.T):
        raise RuntimeError("parent-child quadratic form must be symmetric")
    if not np.allclose(parent_child_metric @ parent_child_metric,
                       parent_child_metric):
        raise RuntimeError("parent-child quadratic form must be a projector")
    if not np.isclose(MOMENT_DM @ whole_metric @ MOMENT_DM, 1.0):
        raise RuntimeError("normalized whole-fork comparison must equal 1")
    if not np.isclose(MOMENT_DM @ parent_child_metric @ MOMENT_DM, 0.0):
        raise RuntimeError("parent-child comparison must equal 0")
    # a: contrast an independent fork with the symmetric parity constraint.
    panel(ax, 15, h - 27, "a", "fork laws")
    p_y = h - 111
    q_y = h - 183
    ax.text(25, p_y, r"$P$", ha="center", va="center",
            fontsize=FS_TITLE, color=INK)
    draw_fork(117, p_y, (None, None, None))
    ax.text(117, p_y - 27, r"$\mathrm{iid}$", ha="center", va="center",
            fontsize=FS_SMALL, color=MUTED)
    ax.text(25, q_y, r"$Q$", ha="center", va="center",
            fontsize=FS_TITLE, color=INK)
    draw_fork(117, q_y, (None, None, None), derived_node=2)
    ax.text(117, q_y - 28, r"$x_2=x_0x_1$", ha="center", va="center",
            fontsize=FS_SMALL, color=ORANGE)

    # b: retain the complete eight-state probability table.
    panel(ax, 205, h - 27, "b", "assignment masses")
    probability_width = 47.0
    probability_xs = tuple(270 + probability_width * index
                           for index in range(8))
    assignment_y = h - 70
    table_ys = (h - 111, h - 183)
    for xx, signs in zip(probability_xs, ASSIGNMENTS):
        draw_triplet(xx, assignment_y, signs, scale=1.48)
    ax.text(240, assignment_y, r"$x$", ha="right", va="center",
            fontsize=FS_SMALL, color=MUTED)
    for yy, name, row, row_color in zip(
        table_ys, (r"$P$", r"$Q$"), (P_PROB, Q_PROB), (MASS, ORANGE),
    ):
        ax.text(240, yy, name, ha="right", va="center",
                fontsize=FS_TITLE, color=row_color)
        for xx, value in zip(probability_xs, row):
            active = not np.isclose(value, 0.0)
            draw_cell(
                xx, yy, float(value), width=probability_width - 4,
                height=28, face=row_color if active else ZERO,
                edge=EDGE, text_color=WHITE if active else MUTED,
                fraction=True,
            )
    # c: carry one uninterrupted matrix product from the probability laws to
    # the product-moment contrast.  The final 2-by-1 factor makes P minus Q
    # visible instead of leaving the difference vector to float on its own.
    separator_1 = h - 220
    ax.plot((15, 635), (separator_1, separator_1), color=EDGE, linewidth=0.55)
    panel(ax, 15, separator_1 - 12, "c", "moment map")
    moment_xs = tuple(108 + 22 * index for index in range(8))
    moment_ys = tuple(separator_1 - 66 - 23 * index for index in range(4))
    for yy, support in zip(moment_ys, MOMENT_SUPPORTS):
        support_fork_glyph(ax, 72, yy, support, scale=0.95)
    for row, yy in enumerate(moment_ys):
        for column, xx in enumerate(moment_xs):
            draw_cell(xx, yy, moment_map[row, column],
                      width=20, height=20)
    brackets(moment_xs[0] - 13, moment_xs[-1] + 13,
             moment_ys[-1] - 13, moment_ys[0] + 13)
    ax.text(np.mean(moment_xs), separator_1 - 20,
            "sign products",
            ha="center", va="center", fontsize=FS_TITLE, color=INK)

    probability_column_xs = (310.0, 330.0)
    probability_column_ys = tuple(
        separator_1 - 55 - 13.0 * index for index in range(8)
    )
    ax.text(np.mean(probability_column_xs), separator_1 - 20,
            "masses",
            ha="center", va="center", fontsize=FS_SMALL, color=INK)
    for row, yy in enumerate(probability_column_ys):
        for column, xx in enumerate(probability_column_xs):
            value = probability_matrix[row, column]
            active = not np.isclose(value, 0.0)
            color = MASS if column == 0 else ORANGE
            draw_cell(xx, yy, value, width=18, height=12.0,
                      face=color if active else ZERO, edge=WHITE,
                      text_color=WHITE if active else MUTED,
                      show_zero=False, fraction=True)
    brackets(probability_column_xs[0] - 12, probability_column_xs[-1] + 12,
             probability_column_ys[-1] - 9, probability_column_ys[0] + 9)
    ax.text(290, np.mean(moment_ys), r"$\times$",
            ha="center", va="center", fontsize=FS_TITLE, color=MUTED)

    moment_column_xs = (384.0, 406.0)
    ax.text(np.mean(moment_column_xs), separator_1 - 20,
            "product moments",
            ha="center", va="center",
            fontsize=FS_SMALL, color=INK)
    for row, yy in enumerate(moment_ys):
        for column, xx in enumerate(moment_column_xs):
            value = moment_matrix[row, column]
            color = MASS if column == 0 else ORANGE
            draw_cell(xx, yy, value, width=20, height=22,
                      edge=WHITE,
                      face=(ZERO if np.isclose(value, 0.0) else
                            (POSITIVE if value > 0 else NEGATIVE)))
    brackets(moment_column_xs[0] - 13, moment_column_xs[-1] + 13,
             moment_ys[-1] - 13, moment_ys[0] + 13)
    ax.text(360, np.mean(moment_ys), r"$=$", ha="center", va="center",
            fontsize=FS_TITLE, color=MUTED)

    ax.text(439, np.mean(moment_ys), r"$\times$", ha="center", va="center",
            fontsize=FS_TITLE, color=MUTED)
    ax.text(462, separator_1 - 20, r"$P-Q$", ha="center", va="center",
            fontsize=FS_TINY, color=MUTED)
    draw_vertical_vector(462, np.mean(moment_ys), (1, -1),
                         cell=16, bracket_color=INK)
    ax.text(491, np.mean(moment_ys), r"$=$", ha="center", va="center",
            fontsize=FS_TITLE, color=MUTED)
    ax.text(520, separator_1 - 20, "difference",
            ha="center", va="center", fontsize=FS_SMALL, color=INK)
    draw_vertical_vector(520, np.mean(moment_ys), MOMENT_DM,
                         cell=20, gap=3, bracket_color=INK)
    for yy, label in zip(
        moment_ys, (r"$01$", r"$02$", r"$12$", r"$012$"),
    ):
        ax.text(542, yy, label, ha="left", va="center",
                fontsize=FS_TINY, color=MUTED)

    # d: show the two symmetric moment projections.  The displayed difference
    # is unchanged by normalizing with its sole nonzero order-three weight.
    separator_2 = h - 382
    ax.plot((15, 635), (separator_2, separator_2), color=EDGE, linewidth=0.55)
    panel(ax, 15, separator_2 - 12, "d", "moment projection")
    for xx, label in (
        (180, "difference"),
        (315, "projection"),
        (425, "difference"),
        (530, "comparison"),
    ):
        ax.text(xx, separator_2 - 24, label, ha="center", va="center",
                fontsize=FS_TINY, color=MUTED)

    def quadratic_row(y: float, label: str, matrix: np.ndarray,
                      *, color: str, result: int) -> None:
        ax.text(104, y, label, ha="right", va="center",
                fontsize=FS_SMALL, color=color)
        draw_horizontal_vector(180, y, MOMENT_DM, cell=16, gap=2,
                               bracket_color=INK)
        ax.text(242, y, r"$\times$", ha="center", va="center",
                fontsize=FS_TITLE, color=MUTED)
        draw_matrix(315, y, matrix, color=color, cell=10.0)
        ax.text(378, y, r"$\times$", ha="center", va="center",
                fontsize=FS_TITLE, color=MUTED)
        draw_vertical_vector(425, y, MOMENT_DM, cell=9.0, gap=1.0,
                             bracket_color=INK)
        ax.text(475, y, r"$=$", ha="center", va="center",
                fontsize=FS_TITLE, color=MUTED)
        ax.text(530, y, rf"${result:d}$", ha="center", va="center",
                fontsize=FS_PANEL, color=color)

    quadratic_row(
        separator_2 - 58, "whole fork", whole_metric,
        color=TEAL, result=1,
    )
    quadratic_row(
        separator_2 - 108, "parent-child", parent_child_metric,
        color=SELECT, result=0,
    )

    pdf.savefig(fig, bbox_inches=None)
    plt.close(fig)
def _draw_depth_two_tree(
    ax, x: float, y: float, *, scale=1.0, mode="both", show_levels=True,
    show_states=False,
) -> None:
    """Draw the seven-node tree with curved edges, as in the Bernstein figure."""
    positions = {
        0: (x, y + 76 * scale),
        1: (x - 48 * scale, y + 40 * scale),
        2: (x + 48 * scale, y + 40 * scale),
        3: (x - 72 * scale, y + 3 * scale),
        4: (x - 24 * scale, y + 3 * scale),
        5: (x + 24 * scale, y + 3 * scale),
        6: (x + 72 * scale, y + 3 * scale),
    }
    for parent, child in depth2.EDGES:
        selected_edge = mode in ("path", "both") and child in (1, 3)
        color = SELECT if selected_edge else EDGE
        width = (2.2 if selected_edge else 1.1) * scale
        curved_edge(
            ax, positions[parent], positions[child],
            color=color, width=width,
        )
    signs = (1, -1, 1, 1, -1, 1, -1)
    for node, (px, py) in positions.items():
        selected_path = mode in ("path", "both") and node in (0, 1, 3)
        fill = (
            POSITIVE if signs[node] > 0 else NEGATIVE
        ) if show_states else (SELECT if selected_path else PANEL_FILL)
        edge = WHITE if show_states else (SELECT if selected_path else BOX_BORDER)
        ax.add_patch(Circle(
            (px, py), 4.3 * scale,
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.8, zorder=5,
        ))
        if show_states:
            arm = 1.7 * scale
            ax.plot((px - arm, px + arm), (py, py), color=WHITE,
                    linewidth=0.85 * scale, zorder=6)
            if signs[node] > 0:
                ax.plot((px, px), (py - arm, py + arm), color=WHITE,
                        linewidth=0.85 * scale, zorder=6)
            if selected_path:
                ax.add_patch(Circle(
                    (px, py), 7.0 * scale, facecolor="none",
                    edgecolor=SELECT, linewidth=1.2 * scale, zorder=7,
                ))
    if mode in ("ancestor", "both"):
        # A soft halo remains visible even where the common-ancestor table
        # shares nodes with the selected path.
        for node in (0, 3, 5):
            px, py = positions[node]
            ax.add_patch(Circle(
                (px, py), 10.2 * scale,
                facecolor=TEAL_LIGHT, edgecolor="none", zorder=4,
            ))
    if show_levels:
        for level, yy in enumerate((
            y + 76 * scale, y + 40 * scale, y + 3 * scale,
        )):
            ax.text(x - 98 * scale, yy, str(level), ha="center", va="center",
                    fontsize=FS_TINY, color=MUTED)
        ax.text(x - 98 * scale, y + 98 * scale, "level",
                ha="center", va="center",
                fontsize=FS_TINY, color=MUTED)


def _rho0_grandchild_mass_table(*, parity: bool) -> np.ndarray:
    """Return the four-by-four grandchild marginal at independence."""
    pair_states = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    if not parity:
        return np.full((4, 4), 1.0 / 16.0)
    return np.asarray([
        [
            (1.0 + math.prod(left) * math.prod(right)) / 16.0
            for right in pair_states
        ]
        for left in pair_states
    ])


def _draw_grandchild_mass_table(
    ax, x: float, y: float, *, label: str, masses: np.ndarray,
    color: str, group_totals: np.ndarray, cell: float = 14.0,
    sign_radius: float = 2.8,
) -> None:
    """Draw a marginal table and its four sign-count group totals."""
    pair_states = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    width = 4 * cell
    ax.text(x + width / 2, y + width + 17, label,
            ha="center", va="center", fontsize=FS_LABEL, color=color)

    for index, signs in enumerate(pair_states):
        column_x = x + (index + 0.5) * cell
        row_y = y + (3.5 - index) * cell
        offset = 1.12 * sign_radius
        for shift, sign in zip((-offset, offset), signs):
            state_disk(
                ax, column_x + shift, y + width + 7,
                sign, radius=sign_radius,
            )
            state_disk(
                ax, x - 7.7 + shift, row_y,
                sign, radius=sign_radius,
            )

    max_mass = 1.0 / 8.0
    for row in range(4):
        for column in range(4):
            left = x + column * cell
            bottom = y + (3 - row) * cell
            plus_count = sum(
                sign > 0 for sign in pair_states[row] + pair_states[column]
            )
            if plus_count in (0, 4):
                group_index = 0
            elif plus_count == 2:
                group_index = 1
            elif plus_count == 3:
                group_index = 2
            else:
                group_index = 3
            ax.add_patch(Rectangle(
                (left, bottom), cell, cell,
                facecolor=mpl.colors.to_rgba(
                    GROUP_COLORS[group_index], 0.19,
                ), edgecolor=WHITE,
                linewidth=0.45, zorder=2,
            ))
            mass = float(masses[row, column])
            if mass <= 1e-12:
                continue
            inner = (cell - 2.0) * math.sqrt(mass / max_mass)
            ax.add_patch(Rectangle(
                (left + (cell - inner) / 2, bottom + (cell - inner) / 2),
                inner, inner, facecolor=color, edgecolor="none",
                alpha=0.82, zorder=3,
            ))
    ax.add_patch(Rectangle(
        (x, y), width, width, facecolor="none", edgecolor=BOX_BORDER,
        linewidth=0.65, zorder=4,
    ))

    bar_y = y - 9
    cursor = x
    for total, group_color in zip(group_totals, GROUP_COLORS):
        segment_width = width * float(total)
        if segment_width > 1e-9:
            ax.add_patch(Rectangle(
                (cursor, bar_y), segment_width, 5.0,
                facecolor=group_color, edgecolor=WHITE,
                linewidth=0.35, zorder=3,
            ))
        cursor += segment_width
    ax.add_patch(Rectangle(
        (x, bar_y), width, 5.0, facecolor="none", edgecolor=BOX_BORDER,
        linewidth=0.6, zorder=4,
    ))


def _draw_probability_tetrahedron(
    ax, vertices: np.ndarray, *, face_alpha: float = 0.16,
    edge_width: float = 0.75, edge_alpha: float = 1.0,
    front_edges: bool = True,
) -> None:
    """Draw the common four-group probability simplex."""
    # One uniform silhouette avoids the false depth created by stacking four
    # translucent projected faces.  Vertex zero is recessed; its three edges
    # are drawn first and remain visible through the constraint sheet.
    ax.add_patch(Polygon(
        vertices[[1, 2, 3]], closed=True,
        facecolor=ZERO, edgecolor="none", alpha=face_alpha, zorder=1,
    ))
    for second in (1, 2, 3):
        ax.plot(
            vertices[[0, second], 0], vertices[[0, second], 1],
            color=LIGHT, linewidth=0.82 * edge_width,
            alpha=0.52 * edge_alpha, zorder=2,
        )
    if front_edges:
        _draw_probability_tetrahedron_silhouette(
            ax, vertices, linewidth=edge_width, alpha=edge_alpha,
        )


def _draw_probability_tetrahedron_silhouette(
    ax, vertices: np.ndarray, *, linewidth: float, alpha: float,
    zorder: int = 6,
) -> None:
    """Draw the projected outer triangle in front of a compatible section."""
    for first, second in ((1, 2), (1, 3), (2, 3)):
        ax.plot(
            vertices[[first, second], 0], vertices[[first, second], 1],
            color=LIGHT, linewidth=linewidth, alpha=alpha, zorder=zorder,
        )


def _draw_sign_count_key(
    ax, x: float, y: float, signs: tuple[int, ...], multiplicity: int,
    *, color: str, multiplicity_side: str = "right",
) -> None:
    """Label a simplex vertex with one sign pattern and its multiplicity."""
    bubble_width = 23.0
    bubble_height = 9.6
    ax.add_patch(FancyBboxPatch(
        (x - bubble_width / 2, y - bubble_height / 2),
        bubble_width, bubble_height,
        boxstyle="round,pad=0,rounding_size=3.5",
        facecolor=mpl.colors.to_rgba(color, 0.18),
        edgecolor=color, linewidth=0.55, zorder=4,
    ))
    assignment_quadruplet(ax, x, y, signs, scale=1.08)
    if multiplicity_side == "right":
        label_x, ha = x + bubble_width / 2 + 3.0, "left"
    elif multiplicity_side == "left":
        label_x, ha = x - bubble_width / 2 - 3.0, "right"
    else:
        raise ValueError(f"unknown multiplicity side: {multiplicity_side}")
    ax.text(
        label_x, y, rf"$\times {multiplicity}$",
        ha=ha, va="center", fontsize=FS_TINY - 0.4, color=MUTED,
    )


def _draw_grandchild_probability_slice(
    ax, x: float, y: float, polygon: np.ndarray,
    p0: np.ndarray, q0: np.ndarray, *, scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the exact rho=0 compatible marginal set in its tetrahedron."""
    # Vertices are the total probabilities of four disjoint sign-count groups:
    # all equal, two of each, one negative, and one positive.
    vertices = _probability_tetrahedron_vertices(x, y, scale=scale)
    _draw_probability_tetrahedron(ax, vertices)

    slice_probabilities = (
        depth2.rho0_grandchild_sign_count_probabilities(polygon)
    )
    slice_points = slice_probabilities @ vertices
    # The grouped low-order constraint alone cuts a triangle from the
    # tetrahedron.  Nonnegativity of each of the 16 assignment masses clips
    # two of its corners and leaves the exact compatible quadrilateral.
    plane_probabilities = np.asarray((
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.25, 0.75, 0.0, 0.0),
    ))
    plane_points = plane_probabilities @ vertices
    # Extend the same affine plane beyond the simplex so the equality reads
    # as a linear section rather than as another bounded probability set.
    plane_left, plane_right, plane_apex = plane_probabilities
    plane_midpoint = 0.5 * (plane_left + plane_right)
    plane_base = plane_right - plane_left
    plane_rise = plane_apex - plane_midpoint
    sheet_probabilities = np.asarray((
        plane_left - 0.075 * plane_base - 0.06 * plane_rise,
        plane_right + 0.075 * plane_base - 0.06 * plane_rise,
        plane_apex + 0.17 * plane_base + 0.14 * plane_rise,
        plane_apex - 0.17 * plane_base + 0.14 * plane_rise,
    ))
    if not np.allclose(sheet_probabilities.sum(axis=1), 1.0):
        raise RuntimeError("constraint sheet left the probability affine hull")
    if not np.allclose(
        sheet_probabilities[:, 1], 3.0 * sheet_probabilities[:, 0],
    ):
        raise RuntimeError("constraint sheet does not satisfy the equality")
    sheet_points = sheet_probabilities @ vertices
    ax.add_patch(Polygon(
        sheet_points, closed=True,
        facecolor=mpl.colors.to_rgba("#e6e2f5", 0.11),
        edgecolor=mpl.colors.to_rgba(SELECT, 0.25),
        linewidth=0.45, joinstyle="round", zorder=2.3,
    ))
    plane_patch = Polygon(
        plane_points, closed=True,
        facecolor=mpl.colors.to_rgba("#e6e2f5", 0.14),
        edgecolor=mpl.colors.to_rgba(SELECT, 0.34),
        linewidth=0.55, joinstyle="round", zorder=3,
    )
    ax.add_patch(plane_patch)
    ax.add_patch(Polygon(
        slice_points, closed=True,
        facecolor=mpl.colors.to_rgba("#d8d1f2", 0.34),
        edgecolor="none", zorder=4,
    ))
    slice_outline = Polygon(
        slice_points, closed=True, facecolor="none",
        edgecolor=SELECT, linewidth=1.15, joinstyle="round", zorder=5,
    )
    slice_outline.set_clip_path(plane_patch)
    ax.add_patch(slice_outline)
    for index, (vertex, group_color) in enumerate(
        zip(vertices, GROUP_COLORS)
    ):
        ax.scatter(
            *vertex, marker="s", s=10, facecolor=group_color,
            edgecolor=WHITE, linewidth=0.25,
            alpha=0.66 if index == 0 else 1.0,
            zorder=2.6 if index == 0 else 7,
        )
    p0_point = (
        depth2.rho0_grandchild_sign_count_probabilities(p0)[0] @ vertices
    )
    q0_point = (
        depth2.rho0_grandchild_sign_count_probabilities(q0)[0] @ vertices
    )
    ax.scatter(*p0_point, color=INK, s=13, zorder=8)
    ax.scatter(*q0_point, facecolor="none", edgecolor=ORANGE, s=15,
               linewidth=0.80, zorder=9)

    _draw_sign_count_key(
        ax, x + 62 * scale, y + 13 * scale,
        (1, 1, 1, 1), 2, color=GROUP_COLORS[0],
    )
    _draw_sign_count_key(
        ax, x, y + 60 * scale,
        (1, 1, -1, -1), 6, color=GROUP_COLORS[1],
    )
    ax.plot(
        (vertices[0, 0] + 3 * scale, x + 49 * scale),
        (vertices[0, 1] + 1 * scale, y + 13 * scale),
        color=LIGHT, linewidth=0.55, alpha=0.70, zorder=2,
    )
    _draw_sign_count_key(
        ax, x - 57 * scale, y - 43 * scale,
        (1, 1, 1, -1), 4, color=GROUP_COLORS[2],
        multiplicity_side="left",
    )
    _draw_sign_count_key(
        ax, x + 57 * scale, y - 43 * scale,
        (1, -1, -1, -1), 4, color=GROUP_COLORS[3],
    )
    return p0_point, q0_point


def _phase_axes(
    ax,
    *,
    show_x: bool,
    show_xlabel: bool,
    ylabel: str | None,
    minimum_power: int,
    maximum_power: int,
) -> None:
    """Apply the axes shared by all three MSE component maps."""
    ax.set_xlim(0, 1)
    ax.set_xticks((0, 0.5, 1.0))
    if not show_x:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    if show_xlabel:
        ax.set_xlabel(r"edge correlation  $\rho$")
    ax.set_yscale("log", base=2)
    ax.set_ylim(2**minimum_power, 2**maximum_power)
    tick_powers = tuple(range(minimum_power, maximum_power, 4))
    ax.set_yticks(
        tuple(2**power for power in tick_powers),
        tuple(rf"$2^{{{power}}}$" for power in tick_powers),
    )
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.grid(color=WHITE, linewidth=0.45, alpha=0.45)
    ax.set_axisbelow(False)



def _boundary_at(
    ratio: np.ndarray,
    rho_grid: np.ndarray,
    n_grid: np.ndarray,
    rho: float,
) -> tuple[float, float]:
    """Return a stable point near the ratio-one contour for direct labeling."""
    rho_index = int(np.argmin(np.abs(rho_grid - rho)))
    n_index = int(np.argmin(np.abs(np.log2(ratio[rho_index]))))
    return float(rho_grid[rho_index]), float(n_grid[n_index])



def render_dependence_risk(pdf: PdfPages) -> None:
    """Render the seven-node identification and sampling comparison."""
    results = depth2.load_results()
    rho_grid = np.asarray([result["rho"] for result in results])
    sample_grid = np.geomspace(2**3, 2**33, 401)
    risks = depth2.risk_arrays(results, sample_grid)
    if np.max(risks["feature_mean_gap"]) > 2e-7:
        raise RuntimeError("compatible laws do not match the four feature tables")

    fig, ax, h = page(
        6.30,
        "Identification and sampling on the seven-node tree",
    )

    projection = depth2.GRANDCHILD_MOMENT_PROJECTION
    compatible = depth2.rho0_grandchild_polygon()
    compatible_from_full_tree = depth2.compatible_grandchild_polygon(
        results[0]["reference"],
    )
    # The explicit two-dimensional marginal calculation must agree with the
    # projection of the full 42-orbit compatible polytope.
    marginal_to_full = np.min(
        np.linalg.norm(
            compatible[:, None, :] - compatible_from_full_tree[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    full_to_marginal = np.min(
        np.linalg.norm(
            compatible_from_full_tree[:, None, :] - compatible[None, :, :],
            axis=2,
        ),
        axis=1,
    )
    if max(np.max(marginal_to_full), np.max(full_to_marginal)) > 1e-8:
        raise RuntimeError("marginal and full-tree compatible polygons disagree")
    p0 = projection @ results[0]["reference"]
    q0 = projection @ results[0]["law"]
    q0[0] = abs(q0[0])
    p0_probabilities = depth2.rho0_grandchild_sign_count_probabilities(p0)[0]
    q0_probabilities = depth2.rho0_grandchild_sign_count_probabilities(q0)[0]
    if not np.allclose(p0_probabilities, (1 / 8, 3 / 8, 1 / 4, 1 / 4)):
        raise RuntimeError("P0 grandchild marginal is not independent")
    if not np.allclose(q0_probabilities, (1 / 4, 3 / 4, 0, 0), atol=1e-9):
        raise RuntimeError("Q0 grandchild marginal does not satisfy parity")

    # a: restore the full signed tree and its selected path/common-ancestor
    # placements before marginalizing to the four grandchildren.
    panel(ax, 12, h - 25, "a", "selected tables")
    _draw_depth_two_tree(
        ax, 78, h - 150, scale=0.82,
        mode="both", show_levels=False, show_states=True,
    )
    ax.text(78, h - 161, r"$x_3,\ldots,x_6$", ha="center", va="top",
            fontsize=FS_TINY, color=MUTED)

    # b: show the exact generative laws of the four-grandchild marginals.
    # A split disk denotes a fair sign; only the first three Q0 signs are free.
    panel(ax, 145, h - 25, "b", "grandchild laws")
    p_xs = (184.0, 207.0, 230.0, 253.0)
    p_y = h - 90
    ax.text(158, p_y, r"$P_0$", ha="center", va="center",
            fontsize=FS_LABEL, color=MASS)
    for token_x in p_xs:
        fair_sign_disk(ax, token_x, p_y)
    _draw_iid_bracket(ax, p_xs, p_y - 10)

    q_xs = p_xs[:3]
    q_y = h - 140
    ax.text(158, q_y, r"$Q_0$", ha="center", va="center",
            fontsize=FS_LABEL, color=ORANGE)
    for token_x in q_xs:
        fair_sign_disk(ax, token_x, q_y)
    _draw_iid_bracket(ax, q_xs, q_y - 10)
    output_x = p_xs[-1]
    ax.add_patch(Circle(
        (output_x, q_y), 8.0, facecolor="none", edgecolor=ORANGE,
        linewidth=0.75, zorder=4,
    ))
    fair_sign_disk(ax, output_x, q_y)
    ax.text(np.mean(p_xs), h - 174, r"$x_6=x_3x_4x_5$",
            ha="center", va="center", fontsize=FS_TINY, color=INK)

    p0_masses = _rho0_grandchild_mass_table(parity=False)
    q0_masses = _rho0_grandchild_mass_table(parity=True)
    if not np.isclose(np.sum(p0_masses), 1.0):
        raise RuntimeError("P0 marginal table does not sum to one")
    if not np.isclose(np.sum(q0_masses), 1.0):
        raise RuntimeError("Q0 marginal table does not sum to one")
    # c: expand those two generative laws into exact joint-mass tables.
    panel(ax, 280, h - 25, "c", "joint mass")
    table_y = h - 154
    table_cell = 13.0
    p_table_x = 291.0
    q_table_x = 361.0
    _draw_grandchild_mass_table(
        ax, p_table_x, table_y,
        label=r"$P_0$", masses=p0_masses, color=MASS,
        group_totals=p0_probabilities, cell=table_cell, sign_radius=2.3,
    )
    _draw_grandchild_mass_table(
        ax, q_table_x, table_y,
        label=r"$Q_0$", masses=q0_masses, color=ORANGE,
        group_totals=q0_probabilities, cell=table_cell, sign_radius=2.3,
    )

    # d: grouped low-order constraints give a plane, while the individual
    # assignment probabilities cut the compatible marginal set inside it.
    panel(ax, 430, h - 25, "d", "compatible slice")
    ax.text(
        545, h - 68,
        r"$\Pr(2{+}2)=3\,\Pr(\mathrm{all\ equal})$",
        ha="center", va="center", fontsize=FS_TINY, color=SELECT,
    )
    _draw_grandchild_probability_slice(
        ax, 545, h - 139, compatible, p0, q0, scale=0.87,
    )
    _grandchild_polytope_tube(fig, ax, results, h=h, y_offset=h - 630)

    lefts = (0.098, 0.389, 0.679)
    width = 0.228
    map_height = 125.0 / h
    top_bottom = 40.0 / h
    risk_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "mse_component_ratio",
        (
            (0.00, "#365F73"),
            (0.34, "#7195A4"),
            (0.48, "#B5CAD0"),
            (0.50, "#F0ECE5"),
            (0.52, "#E7CFC6"),
            (0.66, "#C98573"),
            (1.00, "#A75F50"),
        ),
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    image = None
    row_specs = (
        (
            (
                ("f", r"worst-case discrepancy$^2$", risks["unresolved_ratio"]),
                ("g", "feature sampling variance", risks["sampling_ratio"]),
                ("h", "feature MSE", risks["total_ratio"]),
            ),
            top_bottom,
            risks,
            sample_grid,
            3,
            33,
            "exact child-order average\n" + r"$n$ trees per population",
        ),
    )
    for row_index, (
        maps,
        bottom,
        row_risks,
        vertical_grid,
        minimum_power,
        maximum_power,
        ylabel,
    ) in enumerate(row_specs):
        rho_mesh, vertical_mesh = np.meshgrid(
            rho_grid, vertical_grid, indexing="ij",
        )
        phase_axes = []
        for column, ((letter, title, component), left) in enumerate(
            zip(maps, lefts)
        ):
            fig.text(left - 0.058, bottom + map_height + 35.0 / h, letter,
                     ha="left", va="top", fontsize=FS_PANEL,
                     fontweight="bold", color=INK)
            fig.text(left - 0.015, bottom + map_height + 30.0 / h, title,
                     ha="left", va="top", fontsize=FS_TITLE, color=INK)
            phase_ax = fig.add_axes((left, bottom, width, map_height))
            log_ratio = np.log2(np.maximum(component, 2**-20))
            image = phase_ax.pcolormesh(
                rho_mesh, vertical_mesh, np.clip(log_ratio, -3, 3),
                shading="auto", cmap=risk_cmap, norm=norm, rasterized=True,
            )
            _phase_axes(
                phase_ax,
                show_x=row_index == len(row_specs) - 1,
                show_xlabel=row_index == len(row_specs) - 1 and column == 1,
                ylabel=ylabel if column == 0 else None,
                minimum_power=minimum_power,
                maximum_power=maximum_power,
            )
            phase_axes.append(phase_ax)
        total_ax = phase_axes[-1]
        contour = total_ax.contour(
            rho_mesh, vertical_mesh, np.log2(row_risks["total_ratio"]),
            levels=(0,), colors=(INK,), linewidths=(0.9,),
        )
        total_ax.clabel(
            contour, fmt={0.0: "equal MSE"},
            manual=(
                _boundary_at(
                    row_risks["total_ratio"], rho_grid, vertical_grid, 0.50,
                ),
            ),
            inline=True, inline_spacing=1, fontsize=FS_TINY,
        )

    colorbar_ax = fig.add_axes((
        0.916,
        top_bottom,
        0.012,
        map_height,
    ))
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks((-3, -1, 0, 1, 3))
    colorbar.set_ticklabels(
        (r"$\leq 1/8$", "1/2", "1", "2", r"$\geq 8$"),
    )
    colorbar.ax.tick_params(labelsize=FS_TINY, width=0.45, length=1.8)
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.set_title(
        "relative to\ncorresponding\nwhole-tree MSE",
        fontsize=FS_TINY, color=MUTED, pad=3,
    )

    pdf.savefig(fig, bbox_inches=None)
    plt.close(fig)





def _grandchild_polytope_tube(
    fig,
    ax,
    results: list[dict],
    *,
    h: float,
    y_offset: float,
) -> None:
    """Draw the continuum of projected compatible marginals across rho."""
    panel(ax, 15, 405 + y_offset, "e", r"compatible marginal projections")
    handles = (
        Rectangle((0, 0), 1, 1, facecolor="#e6e2f5", edgecolor=SELECT,
                  linewidth=0.9),
        Line2D((0,), (0,), color=INK, marker="o", markersize=3,
               linewidth=0),
        Line2D((0,), (0,), color=ORANGE, marker="o", markersize=3,
               markerfacecolor="none", linewidth=0),
    )
    fig.legend(
        handles, ("compatible laws", r"$P_\rho$", r"$Q_\rho$"),
        loc="center",
        bbox_to_anchor=(
            0.57, (393.75 + y_offset) / h, 0.28, 22.05 / h,
        ),
        ncol=3, frameon=False, fontsize=FS_TINY,
        handlelength=1.15, handletextpad=0.4, columnspacing=1.35,
    )

    projection = depth2.GRANDCHILD_MOMENT_PROJECTION
    rhos = np.asarray([result["rho"] for result in results])
    polygons = [
        depth2.compatible_grandchild_polygon(result["reference"])
        for result in results
    ]
    references = np.asarray([
        projection @ result["reference"] for result in results
    ])
    candidates = np.asarray([
        projection @ result["law"] for result in results
    ])
    # A global sign flip reflects the odd three-grandchild moment without
    # changing the objective, so display the selected representatives at t>=0.
    candidates[:, 0] = np.abs(candidates[:, 0])
    def project(points: np.ndarray, rho: float) -> np.ndarray:
        """Project (mean triple moment, fourfold moment, rho) to the page."""
        points = np.asarray(points)
        return np.column_stack((
            92 + 455 * rho + 52 * points[:, 0],
            316 + y_offset + 20 * points[:, 0] + 42 * points[:, 1],
        ))

    # Layer all computed sections to retain the continuous purple trajectory.
    for rho, polygon in reversed(list(zip(rhos, polygons))):
        section = project(polygon, rho)
        if len(section) >= 3:
            ax.add_patch(Polygon(
                section, closed=True, facecolor=SELECT, edgecolor=SELECT,
                linewidth=0.25, alpha=0.035, zorder=3,
            ))
        elif len(section) == 2:
            ax.plot(section[:, 0], section[:, 1], color=SELECT,
                    linewidth=0.5, alpha=0.25, zorder=3)
        elif len(section) == 1:
            ax.scatter(*section[0], color=SELECT, s=7, alpha=0.35, zorder=3)

    selected_rhos = (0.0, 0.25, 0.5, 0.75, 1.0)
    selected_indices = [
        int(np.argmin(np.abs(rhos - rho))) for rho in selected_rhos
    ]
    for index in selected_indices:
        actual_rho = float(rhos[index])
        section = project(polygons[index], actual_rho)
        if len(section) >= 3:
            ax.add_patch(Polygon(
                section, closed=True, facecolor="none", edgecolor=SELECT,
                linewidth=0.9, alpha=0.95, zorder=5,
            ))
        elif len(section) == 2:
            ax.plot(section[:, 0], section[:, 1], color=SELECT,
                    linewidth=0.9, alpha=0.95, zorder=5)
        elif len(section) == 1:
            ax.scatter(
                *section[0], color=SELECT, marker="D", s=26,
                alpha=0.58, zorder=5,
            )
        reference_point = project(
            np.asarray([references[index]]), actual_rho,
        )[0]
        candidate_point = project(
            np.asarray([candidates[index]]), actual_rho,
        )[0]
        ax.scatter(*reference_point, color=INK, s=11, zorder=7)
        ax.scatter(
            *candidate_point, facecolor="none", edgecolor=ORANGE,
            s=18, linewidth=0.85, zorder=8,
        )

    axis_y = 244.0 + y_offset
    arrow(ax, (92, axis_y), (558, axis_y),
          color=INK, width=0.65, mutation=6)
    for rho, label in zip(selected_rhos, ("0", "0.25", "0.5", "0.75", "1")):
        x = 92 + 455 * rho
        ax.plot((x, x), (axis_y - 2.5, axis_y + 2.5),
                color=INK, linewidth=0.55, zorder=9)
        ax.text(x + (4 if rho == 0 else 0), axis_y - 10, label,
                ha="center", va="top", fontsize=FS_TINY, color=INK)
    ax.text(566, axis_y, r"$\rho$", ha="left", va="center",
            fontsize=FS_LABEL, color=INK)

    arrow(ax, (92, axis_y), (92, 374 + y_offset),
          color=INK, width=0.65, mutation=6)
    for value, y in (
        ("-1", 274 + y_offset),
        ("0", 316 + y_offset),
        ("1", 358 + y_offset),
    ):
        ax.plot((89.5, 94.5), (y, y), color=INK, linewidth=0.55)
        ax.text(83, y, value, ha="right", va="center",
                fontsize=FS_TINY, color=INK)
    ax.text(36, 316 + y_offset, r"$m_{3456}$", ha="center", va="center",
            rotation=90, fontsize=FS_LABEL, color=INK)
    arrow(ax, (92, 316 + y_offset), (126, 329 + y_offset),
          color=INK, width=0.60, mutation=5)
    ax.text(132, 333 + y_offset, r"$m_{345}$", ha="left", va="center",
            fontsize=FS_LABEL, color=INK)



def render() -> Path:
    _apply_style()
    S.IMGS.mkdir(parents=True, exist_ok=True)
    raw = OUT.with_name("ising_running_example_raw.pdf")
    with PdfPages(raw) as pdf:
        render_story(pdf)
        render_dependence_risk(pdf)
    subprocess.run([
        "gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5", "-dPDFSETTINGS=/prepress",
        "-dEmbedAllFonts=true", "-dSubsetFonts=true",
        f"-sOutputFile={OUT}", str(raw),
    ], check=True)
    raw.unlink()
    print(f"wrote {OUT}")
    return OUT


if __name__ == "__main__":
    render()
