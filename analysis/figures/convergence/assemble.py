"""assemble fig 4 (convergence) from the row / panel SVGs (no subplot juggling).

regenerates every panel (each plot_*.render writes a known-size SVG with CM Sans as
vector paths), then nests them as positioned <svg> viewports in one master laid out
in points:

  a  row 1: three UMAPs (lineage | OT discrepancy | convergence)
  b  row 2: per-cell-type convergence-vs-OT scatter
  c  row 2: EC fan-in (compact)
  d  row 3: sources on the umap, hued by the positional program
  e  row 3: identity and position against time, one column per fate
  f  row 4: per gene, E8.5 source specificity against E9.5 leaf divergence
  g  row 4: the erased / memory split swept over six fates

full-width rows pass through at scale 1; the panel rows are scaled uniformly to the
page width. bold lowercase panel letters (a..g) sit at each panel's top-left. writes
imgs/fig4.{svg,png,pdf} and prints the per-row scale (a scale far below 1 would shrink
fonts under the 6 pt floor).

    python analysis/figures/convergence/assemble.py [--no-regen]
"""
import os
import sys
import xml.etree.ElementTree as ET

import cairosvg

sys.path.insert(0, os.path.dirname(__file__))
import style as S

sys.path.insert(0, S.base_dir)                  # repo root, for the shared label renderer
from utils.figure_labels import panel_letter_svg, LETTER_BOX_IN

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

W = S.PAGE_W_IN * S.PT_PER_IN
PANEL_GAP = 0.08 * S.PT_PER_IN          # horizontal gap between panels in a row
LB = LETTER_BOX_IN * S.PT_PER_IN        # panel-letter box side (points)
LETTER_LIFT = 6.0                       # pt to raise panel letters above the row top
LETTER_DX = -2.2                        # pt to nudge them left of the panel edge

LETTERS = "abcdefg"

# per-boundary row gaps (inches), keyed by the index of the row the gap follows.
GAP_AFTER_IN = {
    0: 0.05,   # umaps -> conv-vs-OT scatter
    1: 0.04,   # scatter/fan-in -> sources/phase
    2: 0.05,   # sources/phase -> divergence/sweep
}


def _rows():
    """each row is a list of panel svg paths, tiled left->right."""
    imgs, panels = S.OUT, S.PANELS
    return [
        [os.path.join(imgs, "row1_umaps.svg")],
        [os.path.join(panels, "celltype_conv_vs_ot.svg"),   # conv-vs-OT scatter far left
         os.path.join(imgs, "fanin_ec_compact.svg")],       # narrowed EC fan-in
        [os.path.join(panels, "sources.svg"),
         os.path.join(panels, "phase.svg")],
        [os.path.join(panels, "divergence.svg"),
         os.path.join(panels, "sweep.svg")],
    ]


def _regenerate():
    import plot_umap_scores
    import plot_celltype_conv_vs_ot
    import plot_fanin
    import plot_sources
    import plot_phase
    import plot_divergence
    import plot_sweep
    plot_umap_scores.render()
    plot_celltype_conv_vs_ot.render()
    plot_fanin.render("ec", compact=True)
    plot_sources.render()
    plot_phase.render()
    plot_divergence.render()
    plot_sweep.render()


def _intrinsic(path):
    """panel's physical size in points, from its viewBox (falls back to width/height)."""
    root = ET.parse(path).getroot()
    vb = root.get("viewBox")
    if vb:
        _, _, w, h = (float(v) for v in vb.replace(",", " ").split())
        return w, h
    return (float(root.get("width").rstrip("pt")),
            float(root.get("height").rstrip("pt")))


def _prefix_ids(root, pre):
    """namespace every id and every #ref so glyph/clip defs can't collide once merged."""
    href = f"{{{XLINK_NS}}}href"
    for el in root.iter():
        if "id" in el.attrib:
            el.set("id", pre + el.attrib["id"])
        h = el.get(href) or el.get("href")
        if h and h.startswith("#"):
            new = "#" + pre + h[1:]
            el.set(href, new) if el.get(href) else el.set("href", new)
        for attr in ("clip-path", "fill", "stroke", "mask"):
            v = el.get(attr)
            if v and v.startswith("url(#"):
                el.set(attr, f"url(#{pre}{v[5:]}")


def _panel(path, w_pt, h_pt, x_pt, y_pt, idx):
    """parse a panel svg, return it as a positioned, scaled nested <svg> viewport."""
    root = ET.parse(path).getroot()
    _prefix_ids(root, f"p{idx}_")
    root.set("x", f"{x_pt:.2f}")
    root.set("y", f"{y_pt:.2f}")
    root.set("width", f"{w_pt:.2f}")
    root.set("height", f"{h_pt:.2f}")
    root.attrib.pop("xmlns", None)
    return root


def assemble(out_dir=None, regen=True):
    out_dir = out_dir or S.OUT
    panels_dir = S.PANELS
    if regen:
        _regenerate()

    rows = _rows()
    layout, letters, y, idx = [], [], 0.0, 0
    for ri, row in enumerate(rows):
        sizes = [_intrinsic(p) for p in row]
        raw = sum(w for w, _ in sizes) + PANEL_GAP * (len(row) - 1)
        scale = W / raw
        row_h = max(h * scale for _, h in sizes)
        x = 0.0
        for path, (w, h) in zip(row, sizes):
            pw, ph = w * scale, h * scale
            layout.append((path, pw, ph, x, y + (row_h - ph) / 2, idx))
            letters.append((x + LETTER_DX, y - LETTER_LIFT))
            x += pw + PANEL_GAP * scale
            idx += 1
        print(f"  row {ri + 1}: {len(row)} panel(s), scale {scale:.3f}"
              + ("  <-- fonts shrink below 6 pt" if scale < 0.86 else ""))
        y += row_h + GAP_AFTER_IN.get(ri, S.ROW_GAP_IN) * S.PT_PER_IN
    H = y - GAP_AFTER_IN.get(len(rows) - 1, S.ROW_GAP_IN) * S.PT_PER_IN + LETTER_LIFT

    master = ET.Element(f"{{{SVG_NS}}}svg", {
        "width": f"{W:.2f}pt", "height": f"{H:.2f}pt",
        "viewBox": f"0 {-LETTER_LIFT:.2f} {W:.2f} {H:.2f}",
    })
    ET.SubElement(master, f"{{{SVG_NS}}}rect", {
        "x": "0", "y": f"{-LETTER_LIFT:.2f}", "width": f"{W:.2f}", "height": f"{H:.2f}",
        "fill": "white"})
    for path, pw, ph, x, yy, i in layout:
        master.append(_panel(path, pw, ph, x, yy, i))

    for i, (ch, (lx, ly)) in enumerate(zip(LETTERS, letters)):
        lp = panel_letter_svg(os.path.join(panels_dir, f"L_{ch}.svg"), ch, color=S.INK)
        master.append(_panel(lp, LB, LB, lx, ly, 200 + i))

    svg_path = os.path.join(out_dir, "fig4.svg")
    ET.ElementTree(master).write(svg_path, xml_declaration=True, encoding="utf-8")
    cairosvg.svg2png(url=svg_path, write_to=os.path.join(out_dir, "fig4.png"),
                     output_width=int(W / 72 * 300))
    cairosvg.svg2pdf(url=svg_path, write_to=os.path.join(out_dir, "fig4.pdf"))
    print(f"wrote {svg_path}  ({W/72:.2f} x {H/72:.2f} in)  + fig4.png, fig4.pdf")
    return svg_path


if __name__ == "__main__":
    assemble(regen="--no-regen" not in sys.argv[1:])
