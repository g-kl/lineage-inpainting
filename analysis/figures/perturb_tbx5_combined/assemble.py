"""nest the combined sheet: the ring over three rows of four.

row 0  a: the ring and its key, re-laid out here (plot_ring.py)
row 1  b, c: the MGI and gnomad boxes, re-plotted here (plot_boxes.py)
       d, e: tbx5 a, b
row 2  f-i: tbx5 e-h
row 3  j-m: tbx5 i-l

every panel is nested at the size its svg declares. the rows are laid out at
those sizes and then scaled into a PAGE_PAD border, as in tbx5/assemble.py. writes
imgs/figure.{svg,png,pdf}.

    python -m analysis.figures.perturb_tbx5_combined.assemble
"""
import os
import re

from analysis.figures.perturb_tbx5_combined import plot_boxes, plot_ring
from analysis.figures.perturb_tbx5_combined import style as S
from utils import svg_assembly as SA
from utils.figure_labels import panel_letter_svg

PT = SA.PT_PER_IN
LETTERS = "abcdefghijklm"
TBX5_ROWS = (("r1_shifted", "r1_specificity"),
             ("r2_shift", "r2_subtree", "r2_split", "r2_program"),
             ("r3_umap_all", "r3_umap_heart", "r3_distortion", "r3_scatter"))


def size_in(path):
    """(w, h) in inches off an svg's root width / height, given in in or pt."""
    with open(path) as f:
        head = f.read(4096)
    head = head[head.index("<svg"):]
    head = head[:head.index(">")]
    out = []
    for a in ("width", "height"):
        v, unit = re.search(rf'\s{a}="([\d.]+)(in|pt)"', head).groups()
        out.append(float(v) / (1.0 if unit == "in" else PT))
    return tuple(out)


def borrowed():
    """(name, path) of every panel taken as rendered, after checking it exists."""
    want = [(n, os.path.join(S.TBX5_PANELS, f"{n}.svg"), "analysis/figures/tbx5/assemble.py")
            for row in TBX5_ROWS for n in row]
    missing = [f"{p}  (written by {who})" for _, p, who in want if not os.path.exists(p)]
    assert not missing, "missing panels:\n  " + "\n  ".join(missing)
    return {n: p for n, p, _ in want}


def assemble():
    paths = borrowed()
    (ring, _), (key, _) = plot_ring.render()
    (mgi, _), (gnomad, _) = plot_boxes.render()
    rows = [[("mgi_box", mgi), ("gnomad_box", gnomad)]
            + [(n, paths[n]) for n in TBX5_ROWS[0]]]
    rows += [[(n, paths[n]) for n in row] for row in TBX5_ROWS[1:]]

    ring_w, ring_h = size_in(ring)
    key_w, _ = size_in(key)
    widths = [ring_w + S.RING_KEY_GAP + key_w]
    widths += [sum(size_in(p)[0] for _, p in row) for row in rows]
    assert max(widths) <= S.PAGE_W + 1e-6, f"a row is {max(widths):.2f} in wide"

    pad = S.PAGE_PAD
    k = (S.PAGE_W - 2 * pad) / S.PAGE_W
    at = lambda u, v: ((pad + u * k) * PT, (pad + v * k) * PT)

    # (name, path, w, h, x, y), in drawing order
    x0 = (S.PAGE_W - widths[0]) / 2
    placed = [("ring", ring, ring_w, ring_h, x0, 0.0),
              ("refkey", key, *size_in(key), x0 + ring_w + S.RING_KEY_GAP, 0.0)]
    y = ring_h + S.RING_GAP
    for row, row_w in zip(rows, widths[1:]):
        x = (S.PAGE_W - row_w) / 2
        h_row = 0.0
        for name, p in row:
            w, h = size_in(p)
            placed.append((name, p, w, h, x, y))
            x += w
            h_row = max(h_row, h)
        y += h_row

    master = SA.new_master()
    letters = iter(LETTERS)
    for idx, (name, p, w, h, x, y0) in enumerate(placed):
        master.append(SA.nest_panel(p, w * k, h * k, *at(x, y0), idx))
        if name == "refkey":
            continue
        ch = next(letters)
        lp = panel_letter_svg(os.path.join(S.PANELS, f"L_{ch}.svg"), ch, color=S.INK)
        dx, dy = S.LETTER_OFF.get(name, (0.0, 0.0))
        master.append(SA.nest_panel(lp, S.LETTER_BOX * k, S.LETTER_BOX * k,
                                    *at(x + dx, y0 + dy), 100 + idx))

    out = os.path.join(S.IMGS, "figure.svg")
    return SA.write_master(master, out, S.PAGE_W * PT, (2 * pad + y * k) * PT)


if __name__ == "__main__":
    assemble()
