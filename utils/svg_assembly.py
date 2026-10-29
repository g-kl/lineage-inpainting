"""nest per-panel svgs as positioned viewports in one master svg, laid out in points.

each panel renderer returns ``(path, (w_in, h_in))``; ``nest_panel`` places one at a
point offset and namespaces its ids so glyph/clip defs from different panels cannot
collide. ``write_master`` writes the svg plus a pdf and a 300-dpi png preview.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
PT_PER_IN = 72.0

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def prefix_ids(root, pre: str) -> None:
    """Namespace every id and internal reference in ``root`` with ``pre``."""
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


def nest_panel(path, w_in: float, h_in: float, x_pt: float, y_pt: float, idx: int):
    """One panel svg as a positioned <svg> viewport, ids prefixed by ``idx``."""
    root = ET.parse(path).getroot()
    prefix_ids(root, f"p{idx}_")
    root.set("x", f"{x_pt:.2f}")
    root.set("y", f"{y_pt:.2f}")
    root.set("width", f"{w_in * PT_PER_IN:.2f}")
    root.set("height", f"{h_in * PT_PER_IN:.2f}")
    root.attrib.pop("xmlns", None)
    return root


def new_master():
    """Empty master svg with a white background rect (sized later by write_master)."""
    master = ET.Element(f"{{{SVG_NS}}}svg")
    ET.SubElement(master, f"{{{SVG_NS}}}rect", {"x": "0", "y": "0", "fill": "white"})
    return master


def sub(master, tag: str, attrib: dict):
    """Append a child element in the svg namespace (for borders, rules, ...)."""
    return ET.SubElement(master, f"{{{SVG_NS}}}{tag}", attrib)


def write_master(master, out_path, w_pt: float, h_pt: float, *, dpi: int = 300,
                 png: bool = True, pdf: bool = True) -> str:
    """Size the master, write the svg, and render the pdf / png siblings."""
    import cairosvg

    master.set("width", f"{w_pt:.2f}pt")
    master.set("height", f"{h_pt:.2f}pt")
    master.set("viewBox", f"0 0 {w_pt:.2f} {h_pt:.2f}")
    master[0].set("width", f"{w_pt:.2f}")     # the background rect
    master[0].set("height", f"{h_pt:.2f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ET.ElementTree(master).write(out_path, xml_declaration=True, encoding="utf-8")
    stem = os.path.splitext(out_path)[0]
    if png:
        cairosvg.svg2png(url=out_path, write_to=f"{stem}.png",
                         output_width=int(w_pt / PT_PER_IN * dpi))
    if pdf:
        cairosvg.svg2pdf(url=out_path, write_to=f"{stem}.pdf")
    print(f"wrote {out_path}  ({w_pt / PT_PER_IN:.2f} x {h_pt / PT_PER_IN:.2f} in)"
          f"{'  + .png' if png else ''}{', .pdf' if pdf else ''}")
    return out_path
