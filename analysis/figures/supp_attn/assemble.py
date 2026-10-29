#!/usr/bin/env python
"""nest the three panel svgs into imgs/supp_attn.{svg,pdf,png}: a and b side by side
over c, with the panel letters at each panel's top left corner.

    python -m analysis.figures.supp_attn.assemble
"""
import os

from analysis.figures.supp_attn import plot_panels
from analysis.figures.supp_attn import style as S
from utils import svg_assembly as SA
from utils.figure_labels import LETTER_BOX_IN, panel_letter_svg

PT = SA.PT_PER_IN


def main():
    a, b, c = plot_panels.render()
    master = SA.new_master()
    idx, y, page_w = 0, 0.0, 0.0
    for row in ([a, b], [c]):
        x = 0.0
        for path, (w, h) in row:
            master.append(SA.nest_panel(path, w, h, x * PT, y * PT, idx))
            ch = "abc"[idx]
            lp = panel_letter_svg(os.path.join(S.PANELS, f"L_{ch}.svg"), ch, color=S.INK)
            master.append(SA.nest_panel(lp, LETTER_BOX_IN, LETTER_BOX_IN, x * PT, y * PT,
                                        100 + idx))
            x += w
            idx += 1
        page_w = max(page_w, x)
        y += max(h for _, (_, h) in row)
    assert page_w <= S.PAGE_W, f"{page_w:.2f} in is wider than the {S.PAGE_W} in page"
    SA.write_master(master, os.path.join(S.IMGS, "supp_attn.svg"), page_w * PT, y * PT)


if __name__ == "__main__":
    main()
