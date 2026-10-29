"""renders the six panel rows and nests them in one master svg. see concept.md.

writes imgs/synthetic_figure.{svg,pdf,png}.

    python -m synthetic_evals.figure.assemble
"""
import os

from synthetic_evals.figure import style as S
from synthetic_evals.figure import plot_crossing as pc
from synthetic_evals.figure import plot_context as pctx
from synthetic_evals.figure import plot_context_realcells as pcr
from synthetic_evals.figure import plot_landscape as pl
from utils import svg_assembly as sa
from utils.figure_labels import (
    box_border_d, box_title_svg, panel_letter_svg, LETTER_BOX_IN,
)

LETTER_LIFT = 6.0    # pt to raise panel letters a-d up to the title line

# per-boundary gap overrides (inches), keyed by the index of the row the gap follows.
# rows: 0 recon-coherent, 1 recon-crossing, 2 coherence categories, 3 context-recon,
# 4 context-metrics, 5 context-realcells. the 2 -> 3 boundary is the box boundary and
# is spaced by the box padding instead.
GAP_AFTER_IN = {
    1: 0.12,   # recon rows -> categories: different experiment, give it air
    3: 0.02,   # context recon -> context metrics: same experiment, keep tight
    4: 0.14,   # context metrics -> real-cell context: synthetic -> real cells
}
BOXES = ((0, 2), (3, 5))   # (first row, last row) per titled box


def assemble(out_dir=None):
    out_dir = out_dir or S.IMGS
    os.makedirs(out_dir, exist_ok=True)

    rows = [
        pc.render_recon(out_dir, "noncrossing", "a_recon_coherent.svg", titles=True, xlabel=False),
        pc.render_recon(out_dir, "crossing", "b_recon_crossing.svg", titles=False, xlabel=True),
        pl.render(out_dir),
        pctx.render_recon(out_dir),
        pctx.render_metrics(out_dir),
        pcr.render(out_dir),
    ]
    titles = [box_title_svg(os.path.join(out_dir, f"T_box{i}.svg"), t,
                            fontsize=S.FS_BOX, color=S.INK)
              for i, t in enumerate(S.BOX_TITLES)]

    PT = S.PT_PER_IN
    W = S.PAGE_W_IN * PT
    inset, padx = S.BOX_INSET_IN * PT, S.BOX_PAD_X_IN * PT
    pad_top, pad_bot = S.BOX_PAD_TOP_IN * PT, S.BOX_PAD_BOT_IN * PT
    # rows are rendered at full page width, then scaled to fit inside the border
    x_row = inset + padx
    scale = (W - 2 * (inset + padx)) / W

    master = sa.new_master()

    box_tops, box_bots = [], []
    # headroom for the first box title, which straddles the top border
    y = S.PAGE_PAD_IN * PT + max(t[1][1] for t in titles) * PT / 2.0
    row_tops = []
    for b, (first, last) in enumerate(BOXES):
        box_tops.append(y)
        y += pad_top
        for idx in range(first, last + 1):
            path, (w_in, h_in) = rows[idx]
            row_tops.append(y)
            master.append(sa.nest_panel(path, w_in * scale, h_in * scale, x_row, y, idx))
            y += h_in * scale * PT
            if idx < last:
                y += GAP_AFTER_IN.get(idx, S.ROW_GAP_IN) * PT
        y += pad_bot
        box_bots.append(y)
        if b < len(BOXES) - 1:
            y += S.BOX_GAP_IN * PT
    H = y + S.PAGE_PAD_IN * PT

    # the two group boxes: rounded border broken by a centred title
    for i, ((y0, y1), (tpath, tsz)) in enumerate(zip(zip(box_tops, box_bots), titles)):
        sa.sub(master, "path", {
            "d": box_border_d(inset, y0, W - inset, y1, tsz[0] * PT),
            "fill": "none", "stroke": S.INK, "stroke-width": "0.8",
            "stroke-linejoin": "round", "stroke-linecap": "round"})
        master.append(sa.nest_panel(tpath, *tsz, (W - tsz[0] * PT) / 2.0,
                                    y0 - tsz[1] * PT / 2.0, 300 + i))

    # bold lowercase panel letters (a..f), one per row at its top-left. most sit level
    # with the panel titles; rows c and f are left-flush (their headings reach x=0), so
    # their letters are lifted into the gap above.
    lift = {0: LETTER_LIFT, 1: LETTER_LIFT, 2: 8.0, 3: LETTER_LIFT, 4: LETTER_LIFT,
            5: 8.0}
    for i, ytop in enumerate(row_tops):
        lp = panel_letter_svg(os.path.join(out_dir, f"L_{'abcdef'[i]}.svg"),
                              "abcdef"[i], color=S.INK)
        master.append(sa.nest_panel(lp, LETTER_BOX_IN, LETTER_BOX_IN,
                                    x_row, ytop - lift[i], 500 + i))

    return sa.write_master(master, os.path.join(out_dir, "synthetic_figure.svg"), W, H)


if __name__ == "__main__":
    assemble()
