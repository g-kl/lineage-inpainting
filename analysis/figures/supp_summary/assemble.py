"""render every panel and nest them into imgs/supp_summary.{svg,pdf,png}: a, then b, then
c and d, then e and f. see concept.md.

    python -m analysis.figures.supp_summary.assemble
"""
import os

from analysis.figures.supp_summary import plot_composition, plot_lists
from analysis.figures.supp_summary import style as S
from utils import svg_assembly as sa
from utils.figure_labels import LETTER_BOX_IN, panel_letter_svg

TOP = 0.18    # room above each row for its letters


def main():
    PT = S.PT_PER_IN
    a, b = plot_composition.render(), plot_composition.render_cells()
    c, d, e, f = [plot_lists.render(v, f"{ch}_{v}.svg")
                  for ch, v in zip("cdef", S.CATEGORIES[1:])]
    x_r = S.PAGE_W_IN - max(d[1][0], f[1][0])
    assert max(c[1][0], e[1][0]) <= x_r, "the tables no longer fit two to a row"

    y2 = a[1][1] + TOP + S.ROW_GAP
    y3 = y2 + b[1][1] + TOP + S.ROW_GAP
    y4 = y3 + TOP + max(c[1][1], d[1][1]) + S.ROW_GAP
    corners = [(0.0, 0.0), (0.0, y2), (0.0, y3), (x_r, y3), (0.0, y4), (x_r, y4)]
    placed = [(p, x, y + TOP) for p, (x, y) in zip((a, b, c, d, e, f), corners)]
    H = y4 + TOP + max(e[1][1], f[1][1])

    master = sa.new_master()
    for i, ((path, (w, h)), x, y) in enumerate(placed):
        master.append(sa.nest_panel(path, w, h, x * PT, y * PT, i))
    for j, (ch, (x, y)) in enumerate(zip("abcdef", corners)):
        lp = panel_letter_svg(os.path.join(S.PANELS, f"L_{ch}.svg"), ch, color="black")
        master.append(sa.nest_panel(lp, LETTER_BOX_IN, LETTER_BOX_IN, x * PT, y * PT,
                                    len(placed) + j))
    sa.write_master(master, os.path.join(S.IMGS, "supp_summary.svg"),
                    S.PAGE_W_IN * PT, H * PT)


if __name__ == "__main__":
    main()
