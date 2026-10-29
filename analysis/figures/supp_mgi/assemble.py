"""render every panel and nest them into imgs/supp_mgi.{svg,pdf,png}: a and b side by
side, over c. see concept.md.

    python -m analysis.figures.supp_mgi.assemble
"""
import os

from analysis.figures.supp_mgi import plot_examples, plot_scatter, plot_subtrees
from analysis.figures.supp_mgi import style as S
from utils import svg_assembly as SA
from utils.figure_labels import LETTER_BOX_IN, panel_letter_svg

PT = SA.PT_PER_IN


def main():
    a, b, c = plot_scatter.render(), plot_examples.render(), plot_subtrees.render()
    assert a[1][0] + b[1][0] <= S.PAGE_W, "a and b are wider than the page"
    y_c = max(a[1][1], b[1][1]) + S.ROW_GAP
    master = SA.new_master()
    for i, ((path, (w, h)), (x, y)) in enumerate(zip((a, b, c),
                                                     ((0.0, 0.0), (a[1][0], 0.0), (0.0, y_c)))):
        master.append(SA.nest_panel(path, w, h, x * PT, y * PT, i))
        ch = "abc"[i]
        lp = panel_letter_svg(os.path.join(S.PANELS, f"L_{ch}.svg"), ch, color=S.INK)
        master.append(SA.nest_panel(lp, LETTER_BOX_IN, LETTER_BOX_IN, x * PT, y * PT,
                                    100 + i))
    SA.write_master(master, os.path.join(S.IMGS, "supp_mgi.svg"), S.PAGE_W * PT,
                    (y_c + c[1][1]) * PT)


if __name__ == "__main__":
    main()
