"""assemble fig 3 from the six panel SVGs (no matplotlib subplot juggling).

renders each panel to imgs/panels/*.svg (each at a known physical size, Latin Modern Sans
text as vector paths), then nests them as <svg> viewports in one master SVG laid
out in points -- row 1 (transition map, a) / row 2 (metric lollipops, b) / row 3
(context scaling, c-f). writes imgs/fig3.svg, imgs/fig3.pdf and a preview imgs/fig3.png.

    python -m analysis.figures.summary_qc.assemble
"""
import os
import xml.etree.ElementTree as ET

import cairosvg

from analysis.figures.summary_qc import style as S
from analysis.figures.summary_qc import (
    plot_transition_map, plot_metrics, plot_context_scaling,
)
from utils.figure_labels import box_border_d, panel_letter_svg, LETTER_BOX_IN

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

GAP = S.ROW_GAP_IN * S.PT_PER_IN
GAP_AB = S.ROW_GAP_AB_IN * S.PT_PER_IN   # tighter row 1 -> row 2

def _letter_svg(panels_dir, ch):
    """bold lowercase panel letter SVG (shared renderer)."""
    return panel_letter_svg(os.path.join(panels_dir, f"L_{ch}.svg"), ch, color=S.INK)


def _panel(path, w_in, h_in, x_pt, y_pt, idx):
    """parse a panel svg, turn its root into a positioned nested <svg> viewport.

    ids are prefixed per panel so glyph/clip defs can't collide once merged.
    """
    root = ET.parse(path).getroot()
    _prefix_ids(root, f"p{idx}_")
    root.set("x", f"{x_pt:.2f}")
    root.set("y", f"{y_pt:.2f}")
    root.set("width", f"{w_in * S.PT_PER_IN:.2f}")
    root.set("height", f"{h_in * S.PT_PER_IN:.2f}")
    root.attrib.pop("xmlns", None)
    return root


def _box_border(master, x0, y0, x1, y1, title_w_pt):
    """rounded rectangle border with a centred gap in the top edge (a fieldset-style
    break) sized to hold the box title. drawn as one stroked path, no font needed."""
    ET.SubElement(master, f"{{{SVG_NS}}}path", {
        "d": box_border_d(x0, y0, x1, y1, title_w_pt), "fill": "none", "stroke": S.INK,
        "stroke-width": "0.8", "stroke-linejoin": "round", "stroke-linecap": "round"})


def _prefix_ids(root, pre):
    """namespace every id and every #ref (id=, xlink:href, url(#..), clip-path)."""
    href = f"{{{XLINK_NS}}}href"
    for el in root.iter():
        if "id" in el.attrib:
            el.set("id", pre + el.attrib["id"])
        h = el.get(href) or el.get("href")
        if h and h.startswith("#"):
            new = "#" + pre + h[1:]
            (el.set(href, new) if el.get(href) else el.set("href", new))
        for attr in ("clip-path", "fill", "stroke", "mask"):
            v = el.get(attr)
            if v and v.startswith("url(#"):
                el.set(attr, f"url(#{pre}{v[5:]}")


def assemble(out_dir=None):
    out_dir = out_dir or S.OUT
    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    (_, A), (_, B) = (
        plot_transition_map.render(panels_dir),
        plot_metrics.render(panels_dir),
    )
    # row 3: two "inference" boxes, each wrapping an (average | per type) panel pair.
    #   box 1 = ancestral (leaf -> root -> leaf), box 2 = descendant (split-down).
    _, anc_avg = plot_context_scaling.render_box(panels_dir)
    _, anc_pt = plot_context_scaling.render_box_celltype(panels_dir)
    _, desc_avg = plot_context_scaling.render_forward(panels_dir)
    _, desc_pt = plot_context_scaling.render_forward_celltype(panels_dir)
    _, t_anc = plot_context_scaling.render_box_title(panels_dir, "ancestral inference", "T_anc.svg")
    _, t_desc = plot_context_scaling.render_box_title(panels_dir, "descendant inference", "T_desc.svg")
    _, legend = plot_context_scaling.render_legend(panels_dir)

    PT = S.PT_PER_IN
    W = S.PAGE_W_IN * PT
    pw, ph = anc_avg[0] * PT, anc_avg[1] * PT
    inner = S.BOX_INNER_GAP_IN * PT
    padx = S.BOX_PAD_X_IN * PT
    boxw = 2 * pw + inner + 2 * padx
    pairgap = S.BOX_PAIR_GAP_IN * PT
    border_off = S.BOX_BORDER_OFF_IN * PT
    th = max(t_anc[1], t_desc[1]) * PT          # box-title height (breaks the border)

    yA = 0.0
    yB = yA + A[1] * PT + GAP_AB
    yPanels = yB + B[1] * PT + GAP + border_off + th / 2.0
    y_box_top = yPanels - border_off
    y_box_bot = yPanels + ph + S.BOX_PAD_BOT_IN * PT
    yLeg = y_box_bot + S.BOX_LEG_GAP_IN * PT
    H = yLeg + legend[1] * PT
    x_start = (W - (2 * boxw + pairgap)) / 2.0

    master = ET.Element(f"{{{SVG_NS}}}svg", {
        "width": f"{W:.2f}pt", "height": f"{H:.2f}pt",
        "viewBox": f"0 0 {W:.2f} {H:.2f}",
    })
    ET.SubElement(master, f"{{{SVG_NS}}}rect", {
        "x": "0", "y": "0", "width": f"{W:.2f}", "height": f"{H:.2f}",
        "fill": "white"})
    master.append(_panel(os.path.join(panels_dir, "A_transition_map.svg"),
                         *A, 0, yA, 0))
    master.append(_panel(os.path.join(panels_dir, "B_metrics.svg"),
                         *B, 0, yB, 1))

    # (box x0, avg panel, per-type panel, title svg, title size)
    boxes = [
        (x_start, "C_context_box.svg", "E_context_box_celltype.svg",
         (anc_avg, anc_pt), "T_anc.svg", t_anc),
        (x_start + boxw + pairgap, "D_context_forward.svg", "F_context_forward_celltype.svg",
         (desc_avg, desc_pt), "T_desc.svg", t_desc),
    ]
    idx = 2
    for bx, avg_name, pt_name, (avg_sz, pt_sz), tname, tsz in boxes:
        _box_border(master, bx, y_box_top, bx + boxw, y_box_bot, tsz[0] * PT)
        cx = bx + boxw / 2.0
        master.append(_panel(os.path.join(panels_dir, tname), *tsz,
                             cx - tsz[0] * PT / 2.0, y_box_top - tsz[1] * PT / 2.0, idx))
        master.append(_panel(os.path.join(panels_dir, avg_name), *avg_sz,
                             bx + padx, yPanels, idx + 1))
        master.append(_panel(os.path.join(panels_dir, pt_name), *pt_sz,
                             bx + padx + pw + inner, yPanels, idx + 2))
        idx += 3
    master.append(_panel(os.path.join(panels_dir, "G_context_legend.svg"),
                         *legend, (W - legend[0] * PT) / 2.0, yLeg, idx))

    # panel letters (a..f), in reading order. each is placed at the top-left corner
    # of its panel's viewport: a/b share x (page left) so they stack in a column;
    # c..f share y (the row-3 panel tops) so they sit on one baseline.
    bx2 = x_start + boxw + pairgap                       # left edge of the 2nd box
    r3 = -1.0                                            # row-3 letters nudged left
    yr3 = yPanels - 5.0                                  # and up, off the panel tops
    xL = x_start + padx + r3                             # avg (left) sub-panel x
    xR = x_start + padx + pw + inner + r3                # per-type (right) sub-panel x
    lb = LETTER_BOX_IN                                   # (w, h) passed to _panel
    letters = [
        ("a", 0.0,             yA),                      # transition map
        ("b", 0.0,             yB),                      # metrics
        ("c", xL,              yr3),                     # ancestral / average
        ("d", xR,              yr3),                     # ancestral / per type
        ("e", bx2 + padx + r3, yr3),                     # descendant / average
        ("f", bx2 + padx + pw + inner + r3, yr3),        # descendant / per type
    ]
    for i, (ch, lx, ly) in enumerate(letters):
        master.append(_panel(_letter_svg(panels_dir, ch), lb, lb, lx, ly, 200 + i))

    svg_path = os.path.join(out_dir, "fig3.svg")
    ET.ElementTree(master).write(svg_path, xml_declaration=True, encoding="utf-8")
    cairosvg.svg2png(url=svg_path, write_to=os.path.join(out_dir, "fig3.png"),
                     output_width=int(W / 72 * 300))
    cairosvg.svg2pdf(url=svg_path, write_to=os.path.join(out_dir, "fig3.pdf"))
    print(f"wrote {svg_path}  ({W/72:.2f} x {H/72:.2f} in)  + fig3.png, fig3.pdf")
    return svg_path


if __name__ == "__main__":
    assemble()
