"""assemble the final figure: the ring UMAP + its reference key (a) over a row of
the two validation boxes (b, c).

each piece is an SVG (ring.svg / refkey.svg from plot_ring.py, mgi_box.svg from
plot_mgi_box.py, gnomad_box.svg from plot_gnomad_box.py); each is nested into one
parent SVG at the right offset and scale, so the whole figure stays vector and fits
a 6.5in-wide A4 column.
"""

import os
import re

from analysis.figures.perturb import style as S
from utils.figure_labels import panel_letter_svg, LETTER_BOX_IN

DPI = 96.0                     # user-unit resolution for the parent viewBox


def svg_size(text):
    """(viewBox width, viewBox height) of an SVG's root element, in user units."""
    s = text.index("<svg")
    head = text[s:text.index(">", s) + 1]
    vb = re.search(r'viewBox="([\d.\- ]+)"', head)
    if vb:
        _, _, w, h = (float(v) for v in vb.group(1).split())
        return w, h
    w = float(re.search(r'width="([\d.]+)', head).group(1))
    h = float(re.search(r'height="([\d.]+)', head).group(1))
    return w, h


def nest(text, x, y, w, h):
    """rewrite an SVG's root <svg ...> so it renders as a sub-element at (x,y,w,h)."""
    text = text[text.index("<svg"):]                   # drop xml decl / doctype
    vw, vh = svg_size(text)
    open_tag = text[:text.index(">") + 1]
    attrs = (f'<svg x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
             f'viewBox="0 0 {vw:.3f} {vh:.3f}" preserveAspectRatio="xMidYMid meet" '
             f'xmlns="http://www.w3.org/2000/svg" '
             f'xmlns:xlink="http://www.w3.org/1999/xlink">')
    return attrs + text[len(open_tag):]


def strip_white_bg(text):
    """drop the ring's full-canvas white background rect (the parent already paints
    one).  without it, sliding the ring down over the box row's top margin paints
    over the boxes instead of letting the ring's transparent whitespace pass."""
    vw, vh = svg_size(text)
    pat = re.compile(r'<rect\s+width="%s"\s+height="%s"\s+fill="white"\s*/>'
                     % (re.escape(f"{vw:g}"), re.escape(f"{vh:g}")))
    return pat.sub("", text, count=1)


def block(text, width_in):
    """(text, px_w, px_h) for an SVG scaled to width_in inches, keeping aspect."""
    w, h = svg_size(text)
    px_w = width_in * DPI
    return text, px_w, px_w * h / w


def main():
    ring = strip_white_bg(open(S.RING_SVG).read())
    refkey = open(os.path.join(S.IMGS, "refkey.svg")).read()
    mgi = open(os.path.join(S.IMGS, "mgi_box.svg")).read()
    gnomad = open(os.path.join(S.IMGS, "gnomad_box.svg")).read()

    W = S.PAGE_W_IN * DPI
    ring_b = block(ring, S.RING_W_IN)
    refkey_b = block(refkey, S.REFKEY_W_IN)
    mgi_b = block(mgi, S.MGI_W_IN)
    gnomad_b = block(gnomad, S.GNOMAD_W_IN)

    # row 1: the ring (left) beside the reference key (right, top-aligned); both are
    # panel a.  row 2: the two boxes side by side, centred under it.
    row1_w = ring_b[1] + S.TOPROW_GAP_IN * DPI + refkey_b[1]
    x_ring = (W - row1_w) / 2
    x_refkey = x_ring + ring_b[1] + S.TOPROW_GAP_IN * DPI
    row2_w = mgi_b[1] + S.GAP_BC_IN * DPI + gnomad_b[1]
    x_mgi = (W - row2_w) / 2
    x_gnomad = x_mgi + mgi_b[1] + S.GAP_BC_IN * DPI
    y_row2 = ring_b[2] + S.GAP_ROW_IN * DPI
    H = y_row2 + max(mgi_b[2], gnomad_b[2])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{S.PAGE_W_IN}in" height="{H / DPI:.3f}in" '
        f'viewBox="0 0 {W:.2f} {H:.2f}">',
        f'<rect width="{W:.2f}" height="{H:.2f}" fill="white"/>',
        nest(refkey_b[0], x_refkey, 0.0, refkey_b[1], refkey_b[2]),
        nest(ring_b[0], x_ring, 0.0, ring_b[1], ring_b[2]),
        nest(mgi_b[0], x_mgi, y_row2, mgi_b[1], mgi_b[2]),
        nest(gnomad_b[0], x_gnomad, y_row2, gnomad_b[1], gnomad_b[2]),
    ]

    # bold lowercase panel letters at each block's top-left corner, shared renderer
    lb = LETTER_BOX_IN * DPI
    for ch, lx, ly in [("a", x_ring, 0.0), ("b", x_mgi, y_row2),
                       ("c", x_gnomad, y_row2)]:
        lp = panel_letter_svg(os.path.join(S.IMGS, f"L_{ch}.svg"), ch, color=S.INK)
        parts.append(nest(open(lp).read(), lx, ly, lb, lb))
    parts.append('</svg>')
    svg = "\n".join(parts)

    out = os.path.join(S.IMGS, "figure.svg")
    with open(out, "w") as fh:
        fh.write(svg)
    print(f"wrote {out}  ({S.PAGE_W_IN}in x {H / DPI:.2f}in)", flush=True)

    import cairosvg
    png = os.path.join(S.IMGS, "figure.png")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=png,
                     output_width=int(S.PAGE_W_IN * 300))
    print(f"wrote {png}", flush=True)
    pdf = os.path.join(S.IMGS, "figure.pdf")
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=pdf)
    print(f"wrote {pdf}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
