"""Assemble the Bernstein panels into one manuscript-width SVG/PDF/PNG.

The individual panels are rendered at their intended physical sizes, then
nested as independent SVG viewports.  This avoids subplot-wide compromises and
matches the assembly pattern used by the main-text figures.

Run from the repository root with

    python -m analysis.figures.theory.assemble
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from analysis.figures.theory import plot_attenuation
from analysis.figures.theory import plot_concept
from analysis.figures.theory import plot_phase
from analysis.figures.theory import style as S
from utils.figure_labels import (
    LETTER_BOX_IN,
    box_border_d,
    box_title_svg,
    panel_letter_svg,
)


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

PANEL_GAP = S.PANEL_GAP_IN * S.PT_PER_IN
LETTER_LIFT = 4.0
BOXES = ((0, 2),)


def _prefix_ids(root, prefix: str) -> None:
    """Namespace ids and references before nesting one SVG inside another."""
    href = f"{{{XLINK_NS}}}href"
    for element in root.iter():
        if "id" in element.attrib:
            element.set("id", prefix + element.attrib["id"])
        target = element.get(href) or element.get("href")
        if target and target.startswith("#"):
            value = "#" + prefix + target[1:]
            element.set(href, value) if element.get(href) else element.set("href", value)
        for attribute in ("clip-path", "fill", "stroke", "mask"):
            value = element.get(attribute)
            if value and value.startswith("url(#"):
                element.set(attribute, f"url(#{prefix}{value[5:]}")


def _intrinsic(path: Path) -> tuple[float, float]:
    root = ET.parse(path).getroot()
    viewbox = root.get("viewBox")
    if viewbox:
        _, _, width, height = (float(value) for value in viewbox.replace(",", " ").split())
        return width, height
    return (
        float(root.get("width", "0pt").rstrip("pt")),
        float(root.get("height", "0pt").rstrip("pt")),
    )


def _panel(path: Path, width: float, height: float, x: float, y: float, index: int):
    root = ET.parse(path).getroot()
    _prefix_ids(root, f"p{index}_")
    root.set("x", f"{x:.2f}")
    root.set("y", f"{y:.2f}")
    root.set("width", f"{width:.2f}")
    root.set("height", f"{height:.2f}")
    root.attrib.pop("xmlns", None)
    return root


def _regenerate() -> list[list[Path]]:
    concept, _ = plot_concept.render()
    phase, _ = plot_phase.render()
    attenuation, _ = plot_attenuation.render()
    rows = [
        [concept],
        [phase],
        [attenuation],
    ]
    return rows


def _export(svg_path: Path) -> None:
    cairo_pdf = svg_path.with_name(svg_path.stem + "_cairo.pdf")
    final_pdf = svg_path.with_suffix(".pdf")
    subprocess.run(
        ["rsvg-convert", "--format", "pdf", "--output",
         str(cairo_pdf), str(svg_path)],
        check=True,
    )
    # Cairo writes PDF 1.7 and pdfTeX accepts at most PDF 1.5.  Preserve the
    # vector output while lowering only the container version used for
    # inclusion.
    subprocess.run(
        [
            "gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5", "-dPDFSETTINGS=/prepress",
            "-dEmbedAllFonts=true", "-dSubsetFonts=true",
            f"-sOutputFile={final_pdf}", str(cairo_pdf),
        ],
        check=True,
    )
    cairo_pdf.unlink()
    subprocess.run(
        ["rsvg-convert", "--format", "png", "--dpi-x", "300", "--dpi-y", "300",
         "--output", str(svg_path.with_suffix(".png")), str(svg_path)],
        check=True,
    )


def assemble() -> Path:
    S.IMGS.mkdir(parents=True, exist_ok=True)
    S.PANELS.mkdir(parents=True, exist_ok=True)
    rows = _regenerate()
    titles = [
        box_title_svg(S.PANELS / f"T_box{i}.svg", title,
                      fontsize=S.FS_BOX, color=S.INK)
        for i, title in enumerate(S.BOX_TITLES)
    ]

    pt = S.PT_PER_IN
    width = S.PAGE_W_IN * pt
    inset = S.BOX_INSET_IN * pt
    inner_width = width - 2 * (inset + S.BOX_PAD_X_IN * pt)
    pad_top = S.BOX_PAD_TOP_IN * pt
    pad_bottom = S.BOX_PAD_BOT_IN * pt
    row_gap = S.ROW_GAP_IN * pt

    master = ET.Element(f"{{{SVG_NS}}}svg")
    background = ET.SubElement(master, f"{{{SVG_NS}}}rect", {
        "x": "0", "y": "0", "fill": "white",
    })

    layouts: list[tuple[Path, float, float, float, float, int]] = []
    letters: list[tuple[str, float, float]] = []
    box_tops, box_bottoms = [], []
    title_half_height = max(size[1] for _, size in titles) * pt / 2
    y = S.PAGE_PAD_IN * pt + title_half_height
    panel_index = 0

    for box_index, (first_row, last_row) in enumerate(BOXES):
        box_tops.append(y)
        y += pad_top
        for row_index in range(first_row, last_row + 1):
            row = rows[row_index]
            sizes = [_intrinsic(path) for path in row]
            raw_width = sum(w for w, _ in sizes) + PANEL_GAP * (len(row) - 1)
            scale = min(1.0, inner_width / raw_width)
            used_width = raw_width * scale
            content_height = max(h * scale for _, h in sizes)
            row_height = content_height
            x = (width - used_width) / 2
            print(f"  row {row_index + 1}: {len(row)} panel(s), scale {scale:.3f}")
            for path, (panel_width, panel_height) in zip(row, sizes):
                placed_width = panel_width * scale
                placed_height = panel_height * scale
                yy = y + (content_height - placed_height) / 2
                layouts.append((path, placed_width, placed_height, x, yy, panel_index))
                if row_index == 0:
                    letters.append(("a", x, yy - LETTER_LIFT))
                x += placed_width + PANEL_GAP * scale
                panel_index += 1
            y += row_height
            if row_index < last_row:
                y += row_gap
        y += pad_bottom
        box_bottoms.append(y)
        if box_index < len(BOXES) - 1:
            y += S.BOX_GAP_IN * pt

    height = y + S.PAGE_PAD_IN * pt
    master.set("width", f"{width:.2f}pt")
    master.set("height", f"{height:.2f}pt")
    master.set("viewBox", f"0 0 {width:.2f} {height:.2f}")
    background.set("width", f"{width:.2f}")
    background.set("height", f"{height:.2f}")

    for path, panel_width, panel_height, x, yy, index in layouts:
        master.append(_panel(path, panel_width, panel_height, x, yy, index))

    for index, ((top, bottom), (title_path, title_size)) in enumerate(
        zip(zip(box_tops, box_bottoms), titles),
    ):
        ET.SubElement(master, f"{{{SVG_NS}}}path", {
            "d": box_border_d(inset, top, width - inset, bottom, title_size[0] * pt),
            "fill": "none", "stroke": S.BOX_BORDER, "stroke-width": "0.7",
            "stroke-linejoin": "round", "stroke-linecap": "round",
        })
        master.append(_panel(
            Path(title_path), title_size[0] * pt, title_size[1] * pt,
            (width - title_size[0] * pt) / 2,
            top - title_size[1] * pt / 2,
            300 + index,
        ))

    letter_box = LETTER_BOX_IN * pt
    for index, (letter, x, yy) in enumerate(letters):
        path = panel_letter_svg(S.PANELS / f"L_{letter}.svg", letter, color=S.INK)
        master.append(_panel(Path(path), letter_box, letter_box, x, yy, 500 + index))

    svg_path = S.IMGS / "bernstein_figure.svg"
    ET.ElementTree(master).write(svg_path, xml_declaration=True, encoding="utf-8")
    _export(svg_path)
    print(
        f"wrote {svg_path} ({width / pt:.2f} x {height / pt:.2f} in) + PDF and PNG",
    )
    return svg_path


if __name__ == "__main__":
    assemble()
