"""render every panel and nest them into imgs/supp_conv.{svg,pdf,png}: a and b, then c
and d, each row's second panel flush right. see concept.md.

    python -m analysis.figures.supp_conv.assemble
"""
import os

from analysis.figures.supp_conv import plot_cutoffs, plot_genes, plot_raw, plot_table
from analysis.figures.supp_conv import style as S
from utils import svg_assembly as SA
from utils.figure_labels import LETTER_BOX_IN, panel_letter_svg

TOP = 0.20        # room above each row for its letters
ROW_GAP = 0.12


def main():
    PT = S.PT_PER_IN
    rows = ((plot_table.render(), plot_raw.render()),
            (plot_genes.render(), plot_cutoffs.render()))
    master, y, i = SA.new_master(), 0.0, 0
    for row in rows:
        w_right = row[1][1][0]
        assert row[0][1][0] + w_right <= S.PAGE_W, f"row {i // 2 + 1} is too wide"
        for (path, (w, h)), x in zip(row, (0.0, S.PAGE_W - w_right)):
            master.append(SA.nest_panel(path, w, h, x * PT, (y + TOP) * PT, i))
            ch = "abcd"[i]
            lp = panel_letter_svg(os.path.join(S.PANELS, f"L_{ch}.svg"), ch, color=S.INK)
            master.append(SA.nest_panel(lp, LETTER_BOX_IN, LETTER_BOX_IN, x * PT, y * PT,
                                        100 + i))
            i += 1
        y += TOP + max(h for _, (_, h) in row) + ROW_GAP
    SA.write_master(master, os.path.join(S.IMGS, "supp_conv.svg"), S.PAGE_W * PT,
                    (y - ROW_GAP) * PT)


if __name__ == "__main__":
    main()
