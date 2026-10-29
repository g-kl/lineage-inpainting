"""render every panel and nest them into imgs/supp_tbx5.{svg,pdf,png}: a and b, then c
to e, then f to h, each row filling the page width. see concept.md.

    python -m analysis.figures.supp_tbx5.assemble
"""
import os

from analysis.figures.supp_tbx5 import plot_age, plot_expression, plot_ne, plot_qc
from analysis.figures.supp_tbx5 import style as S
from utils import svg_assembly as SA
from utils.figure_labels import LETTER_BOX_IN, panel_letter_svg

PT = SA.PT_PER_IN


def main():
    rows = (plot_qc.render(), (*plot_expression.render(), plot_age.render()),
            plot_ne.render())
    master, y, i = SA.new_master(), 0.0, 0
    for row, h in zip(rows, S.ROW_H):
        assert abs(sum(w for _, (w, _) in row) - S.PAGE_W) < 1e-6, f"row {i} is off the page"
        x = 0.0
        for path, (w, _) in row:
            master.append(SA.nest_panel(path, w, h, x * PT, y * PT, i))
            ch = "abcdefgh"[i]
            lp = panel_letter_svg(os.path.join(S.PANELS, f"L_{ch}.svg"), ch, color=S.INK)
            master.append(SA.nest_panel(lp, LETTER_BOX_IN, LETTER_BOX_IN, x * PT, y * PT,
                                        100 + i))
            x += w
            i += 1
        y += h + S.ROW_GAP
    SA.write_master(master, os.path.join(S.IMGS, "supp_tbx5.svg"), S.PAGE_W * PT,
                    (y - S.ROW_GAP) * PT)


if __name__ == "__main__":
    main()
