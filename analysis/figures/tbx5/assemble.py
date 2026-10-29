#!/usr/bin/env python
"""nest the twelve panel svgs into the sheet, three rows of four.

row 1  the CHD panel screen: what the model does to the heart across the panel
row 2  the Tbx5 knockout inside the model: when it shifts what, Tbx5 on the trees,
       and where in the heart tube the shift lands
row 3  the crispants against the prediction, at the E9.5 heart tube

a to l are drawn at each panel's top left corner, along the rows, at utils/'s default
size so they match every other figure; the names below are what the panels are called
here and in concept.md. the rows are laid out at the panel sizes in config and then
scaled into a PAGE_PAD border, so the sheet stays PAGE_W wide and nothing a panel puts
flush with its own edge - a corner letter, a right-aligned label - lands on the sheet's.
svg nesting and the letters are utils/'s. writes imgs/tbx5_complete.{svg,png,pdf}.

    python assemble.py [--no-regen]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
from utils import svg_assembly as SA
from utils.figure_labels import panel_letter_svg

PT = SA.PT_PER_IN
LETTERS = "abcdefghijkl"
NAMES = (("r1_shifted", "r1_specificity", "r1_kd", "r1_dose"),
         ("r2_shift", "r2_subtree", "r2_split", "r2_program"),
         ("r3_umap_all", "r3_umap_heart", "r3_distortion", "r3_scatter"))
WIDTHS = (S.R1_W, S.R2_W, S.R3_W)
HEIGHTS = (S.ROW1_H, S.ROW2_H, S.ROW3_H)


def regenerate():
    import plot_curves
    import plot_distortion
    import plot_metrics
    import plot_program
    import plot_scatter
    import plot_subtree
    import plot_umaps
    a, b, c, d = plot_metrics.render()
    e, g = plot_curves.render()
    f = plot_subtree.render()
    i, j = plot_umaps.render()
    return [[a, b, c, d], [e, f, g, plot_program.render()],
            [i, j, plot_distortion.render(), plot_scatter.render()]]


def cached():
    return [[(os.path.join(S.PANELS, f"{n}.svg"), (w, h))
             for n, w in zip(row, ws)]
            for row, ws, h in zip(NAMES, WIDTHS, HEIGHTS)]


def assemble(regen=True):
    rows = regenerate() if regen else cached()
    w_in = max(sum(ws) for ws in WIDTHS)
    assert w_in <= S.PAGE_W, f"{w_in:.2f} in is wider than the {S.PAGE_W} in page"

    pad = S.PAGE_PAD
    k = (w_in - 2 * pad) / w_in           # the panels, shrunk to sit inside the border
    at = lambda u, v: ((pad + u * k) * PT, (pad + v * k) * PT)

    master = SA.new_master()
    idx, y = 0, 0.0
    for row, names, h in zip(rows, NAMES, HEIGHTS):
        x = 0.0
        for (path, (w, _)), name in zip(row, names):
            master.append(SA.nest_panel(path, w * k, h * k, *at(x, y), idx))
            ch = LETTERS[idx]
            lp = panel_letter_svg(os.path.join(S.PANELS, f"L_{ch}.svg"), ch,
                                  color=S.INK)
            dx, dy = S.LETTER_OFF.get(name, (0.0, 0.0))
            master.append(SA.nest_panel(lp, S.LETTER_BOX * k, S.LETTER_BOX * k,
                                        *at(x + dx, y + dy), 100 + idx))
            x += w
            idx += 1
        y += h

    out = os.path.join(S.IMGS, "tbx5_complete.svg")
    return SA.write_master(master, out, w_in * PT, (2 * pad + y * k) * PT)


if __name__ == "__main__":
    assemble(regen="--no-regen" not in sys.argv[1:])
