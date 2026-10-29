"""motif-cartoon primitives and canvas helpers shared by the two one-canvas rows.

rows c and f are drawn on a single equal-aspect y-down axes whose unit is 0.01 in, so the
problem-setup path math (cartoons/04_problem_setup) ports over verbatim and heatmap axes
can be placed in the same coordinates. both rows draw the same kind of little tree, so the
edge/brace/node primitives live here rather than being imported privately across modules.
"""

from __future__ import annotations

from matplotlib.patches import PathPatch
from matplotlib.path import Path

from synthetic_evals.figure import style as S

# --- canvas: 1 unit == 0.01 in ----------------------------------------------------
U = 100.0
W = S.PAGE_W_IN * U

# --- shared cartoon palette -------------------------------------------------------
EDGE = "#d7dce4"           # tree edges (problem_setup .edge)
BRACE = "#5a626f"          # layer labels (problem_setup label ink)
GRID, GRID_LW = "#dfe3ea", 0.4       # heatmap cell separators
NODE_ROOT = "#c9ccd3"      # hidden common ancestor
# generic motif letters, one hue each (same letter == same colour in rows c and f)
LETTER_COLORS = {"A": "#2b4570", "B": "#4f8a3c", "C": "#a63d40",
                 "D": "#d0962f", "E": "#3f97b5", "F": "#8a5a9e"}


def rect(fig, x, y, w, h, *, height):
    """axes at a canvas-units rectangle on a canvas of ``height`` units."""
    return fig.add_axes([x / W, 1 - (y + h) / height, w / W, h / height])


def tint(hexc, w):
    """blend a hex colour toward white by fraction w (1 = white)."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * w) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def curve(ax, a, b, lw=1.3):
    """light curved edge from parent a=(x,y) to child b=(x,y)."""
    (ax0, ay0), (bx, by) = a, b
    mid = 0.5 * (ay0 + by)
    p = Path([(ax0, ay0), (ax0, mid), (bx, mid), (bx, by)],
             [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(PathPatch(p, fill=False, ec=EDGE, lw=lw,
                           capstyle="round", joinstyle="round", zorder=1))


def brace_h(ax, xL, xR, yBody, depth, direction, lw=1.15):
    """horizontal curly brace on y=yBody; direction -1 = over (cusp up), +1 = under."""
    xm = 0.5 * (xL + xR)
    q = min(12.0, (xR - xL) * 0.16)
    yt = yBody + depth * direction
    yc = yt + 6.0 * direction
    verts = [(xL, yBody), (xL, yt), (xL + q, yt), (xm - 6, yt), (xm, yt), (xm, yc),
             (xm, yt), (xm + 6, yt), (xR - q, yt), (xR, yt), (xR, yBody)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fill=False, ec=BRACE, lw=lw,
                           capstyle="round", joinstyle="round", zorder=4))


def parents(cx, py, sep_p):
    return [(cx - sep_p, py), (cx + sep_p, py)]


def leaf_xy(cx, ys, sep_p, sep_l):
    px = [cx - sep_p, cx + sep_p]
    xs = [px[0] - sep_l, px[0] + sep_l, px[1] - sep_l, px[1] + sep_l]
    return [(xs[i], ys[i]) for i in range(4)]


def edges(ax, par, leaves):
    curve(ax, par[0], leaves[0]); curve(ax, par[0], leaves[1])
    curve(ax, par[1], leaves[2]); curve(ax, par[1], leaves[3])


def node_text(ax, x, y, label, fc, ec, lw=0.9):
    ax.text(x, y, label, ha="center", va="center", fontsize=S.FS_SUB, color=S.INK,
            zorder=5, linespacing=0.95, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.28", fc=fc, ec=ec, lw=lw))


def assert_within_page(fig, ax, panel_h):
    """raise if any text/patch overflows the page width."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    arts = list(ax.texts) + list(ax.patches)
    ext = [a.get_window_extent(renderer=r) for a in arts]
    x0 = min(e.x0 for e in ext) / fig.dpi
    x1 = max(e.x1 for e in ext) / fig.dpi
    if x0 < -1e-3 or x1 > S.PAGE_W_IN + 1e-3:
        raise AssertionError(
            f"content x-extent [{x0:.3f}, {x1:.3f}] in overflows the "
            f"{S.PAGE_W_IN} in page")
