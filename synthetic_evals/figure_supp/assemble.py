"""renders the four supplement rows and nests them in one master svg. see concept.md.

writes imgs/synthetic_supp.{svg,pdf,png}.

    python -m synthetic_evals.figure_supp.assemble
"""
import os

from synthetic_evals.figure_supp import style as S
from synthetic_evals.figure_supp import plot_baselines as pb
from synthetic_evals.figure_supp import plot_context_realcells as pcr
from synthetic_evals.figure_supp import plot_crossing_transitions as pct
from synthetic_evals.figure_supp import plot_robustness as pr
from utils import svg_assembly as sa
from utils.figure_labels import panel_letter_svg, LETTER_BOX_IN

# rows: a crossing, b baselines clustermap, c baselines context, d robustness.
# gap after each row but the last (inches). the rows carry uneven blank margins of
# their own, so these even out the visible space between them.
GAPS_IN = (0.02, 0.0, 0.12)
# pt to raise each panel letter: a-c up to the row's title / label line, d to the
# top of its axes
LETTER_LIFT = (6.0, 7.0, 7.0, 2.5)
# the robustness row ends in blank space below its x labels
BOTTOM_TRIM_IN = 0.12


def assemble(out_dir=None):
    out_dir = out_dir or S.IMGS
    os.makedirs(out_dir, exist_ok=True)
    rows = [pct.render(out_dir), pb.render(out_dir), pcr.render(out_dir), pr.render(out_dir)]

    PT = S.PT_PER_IN
    W = S.PAGE_W_IN * PT
    master = sa.new_master()
    y = S.PAGE_PAD_IN * PT
    for i, (path, (w_in, h_in)) in enumerate(rows):
        master.append(sa.nest_panel(path, w_in, h_in, 0.0, y, i))
        ch = "abcd"[i]
        lp = panel_letter_svg(os.path.join(out_dir, f"L_{ch}.svg"), ch, color=S.INK)
        master.append(sa.nest_panel(lp, LETTER_BOX_IN, LETTER_BOX_IN,
                                    0.0, y - LETTER_LIFT[i], 100 + i))
        y += h_in * PT
        if i < len(rows) - 1:
            y += GAPS_IN[i] * PT
    H = y + (S.PAGE_PAD_IN - BOTTOM_TRIM_IN) * PT
    return sa.write_master(master, os.path.join(out_dir, "synthetic_supp.svg"), W, H)


if __name__ == "__main__":
    assemble()
