#!/usr/bin/env python3
"""a sheet of plain 8-leaf binary trees to copy into illustrator.

same drawing as the 04/05 cartoons: s-curve edges (vertical out of the parent,
vertical into the child) and round nodes, in the same 240 px/in units. every tree
has the footprint of row a's input tree in problem_setup_approach2: 8 leaves
across TREE_W, root to leaves over BAND, r = 10 nodes, 2.8 px edges. no colours or
node styles; every tree is its own <g>, with its edges and nodes in separate
subgroups.

the sheet opens with row a's tree itself, read from observed_tree(3) in 05 the way
model_tree and place build it. then all 23 shapes an 8-leaf binary tree can take
(mirror images counted once), each drawn N_TIMINGS times: first on an even
lattice, dividing once per generation, then with random division times. the leaves
always sit on the bottom row, with row a's spacing: 0.11 between neighbours and
0.20 across the root's split.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
import importlib.util
import random
import sys

ROOT = Path(__file__).resolve().parent
OUT_SVG = ROOT / "assorted_trees.svg"
UTIL_PATH = ROOT.parent / "05_lineage_completion" / "make_lineage_completion_figure.py"

PX_PER_IN = 240
N_LEAVES = 8
N_TIMINGS = 3                # per shape: the even lattice, then random division times
SEED = 0

# row a of problem_setup_approach2: place(base, LEFT + 14, 260, top, top + BAND)
TREE_W = 260
BAND = 216
ROOT_GAP = 0.20 / 0.11       # leaf gap across the root's split, in leaf spacings
R = 10
MIN_DT = 26                  # closest two successive divisions may sit, in px
COLS = 6                     # trees per row of the sheet
GAP_X, GAP_Y = 70, 70        # between trees / rows
MARGIN = 40

EDGE = "#ccd2dc"
EDGE_W = 2.8
NODE_FILL = "#ffffff"
NODE_STROKE = "#6f7682"
NODE_W = 2.0

# a tree is a nested tuple: () is a leaf, (left, right) a division.


@cache
def shapes(n: int) -> tuple[tuple, ...]:
    """every binary tree with n leaves, one per mirror class."""
    if n == 1:
        return ((),)
    out = []
    for a in range(1, n // 2 + 1):
        left, right = shapes(a), shapes(n - a)
        for i, l in enumerate(left):
            for r in right[i if a == n - a else 0:]:
                out.append((l, r))
    return tuple(out)


def height(t: tuple) -> int:
    return 0 if not t else 1 + max(height(c) for c in t)


def row_a_tree() -> tuple[list, list]:
    """row a's tree, exactly: observed_tree(3)'s x, one generation per third of BAND."""
    spec = importlib.util.spec_from_file_location("lineage_completion_utils", UTIL_PATH)
    util = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = util
    spec.loader.exec_module(util)
    base = util.observed_tree(3)
    by_id = {n.id: n for n in base}
    gen = lambda n: 0 if n.parent is None else 1 + gen(by_id[n.parent])
    lo, hi = min(n.x for n in base), max(n.x for n in base)
    xy = {n.id: ((n.x - lo) / (hi - lo) * TREE_W, gen(n) / 3 * BAND) for n in base}
    return list(xy.values()), [(xy[n.parent], xy[n.id]) for n in base if n.parent]


def layout(t: tuple, rng: random.Random | None) -> tuple[list, list]:
    """leaves along the bottom row at row a's spacing, parents centred over their
    children. with no rng each division sits on a generation of an even lattice;
    with one, each division lands somewhere between its parent's and the latest
    time that still leaves MIN_DT for every division below it."""
    depth = height(t)
    dx = TREE_W / (N_LEAVES - 2 + ROOT_GAP)
    nodes, edges = [], []
    next_leaf = [0.0]

    def walk(sub: tuple, level: int, y_parent: float) -> tuple[float, float]:
        if not sub:
            y = BAND
        elif level == 0:
            y = 0.0
        elif rng is None:
            y = level / depth * BAND
        else:
            h = height(sub)
            lo, hi = y_parent + MIN_DT, BAND - h * MIN_DT
            gap = (BAND - y_parent) / (h + 1) * rng.uniform(0.5, 1.5)
            y = min(hi, max(lo, y_parent + gap))
        if sub:
            kids = []
            for i, c in enumerate(sub):
                if level == 0 and i > 0:
                    next_leaf[0] += ROOT_GAP - 1
                kids.append(walk(c, level + 1, y))
            x = (kids[0][0] + kids[-1][0]) / 2
        else:
            x = next_leaf[0] * dx
            next_leaf[0] += 1
            kids = []
        nodes.append((x, y))
        edges.extend(((x, y), k) for k in kids)
        return x, y

    walk(t, 0, 0.0)
    return nodes, edges


def edge_d(p: tuple[float, float], c: tuple[float, float]) -> str:
    mid = (p[1] + c[1]) / 2
    return (f"M {p[0]:.1f} {p[1]:.1f} "
            f"C {p[0]:.1f} {mid:.1f}, {c[0]:.1f} {mid:.1f}, {c[0]:.1f} {c[1]:.1f}")


def tree_svg(nodes: list, edges: list, x0: float, y0: float, name: str) -> str:
    tr = lambda p: (p[0] + x0, p[1] + y0)
    paths = "\n".join(
        f'    <path d="{edge_d(tr(p), tr(c))}" fill="none" stroke="{EDGE}" '
        f'stroke-width="{EDGE_W}" stroke-linecap="round"/>'
        for p, c in edges)
    circles = "\n".join(
        f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="{R}" fill="{NODE_FILL}" '
        f'stroke="{NODE_STROKE}" stroke-width="{NODE_W}"/>'
        for x, y in map(tr, nodes))
    return (f'<g id="{name}">\n  <g id="{name}_edges">\n{paths}\n  </g>\n'
            f'  <g id="{name}_nodes">\n{circles}\n  </g>\n</g>')


def main() -> None:
    rng = random.Random(SEED)
    trees = [("problem_setup_row_a", row_a_tree())]
    for k, t in enumerate(sorted(shapes(N_LEAVES), key=height)):
        trees.append((f"shape{k:02d}_even", layout(t, None)))
        trees += [(f"shape{k:02d}_rand{j}", layout(t, rng)) for j in range(1, N_TIMINGS)]

    body = []
    for i, (name, (nodes, edges)) in enumerate(trees):
        x = MARGIN + R + (i % COLS) * (TREE_W + 2 * R + GAP_X)
        y = MARGIN + R + (i // COLS) * (BAND + 2 * R + GAP_Y)
        body.append(tree_svg(nodes, edges, x, y, name))
    rows = -(-len(trees) // COLS)
    sheet_w = 2 * MARGIN + COLS * (TREE_W + 2 * R) + (COLS - 1) * GAP_X
    sheet_h = 2 * MARGIN + rows * (BAND + 2 * R) + (rows - 1) * GAP_Y

    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{sheet_w / PX_PER_IN:.4f}in" height="{sheet_h / PX_PER_IN:.4f}in" '
           f'viewBox="0 0 {sheet_w:.1f} {sheet_h:.1f}">\n'
           + "\n".join(body) + "\n</svg>\n")
    OUT_SVG.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT_SVG}: {len(trees)} trees, {len(shapes(N_LEAVES))} shapes")


if __name__ == "__main__":
    main()
