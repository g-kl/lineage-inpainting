#!/usr/bin/env python3
"""Export problem_setup_figure.html to an editable standalone SVG.

Follows the slice-feature / attention export pattern: browser SVG for the
geometry plus MathJax SVG for labels. This script inlines SVG styles and embeds
MathJax labels as vector paths for Illustrator/Inkscape.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


EXPORT_JS = r"""
() => {
  const NS = 'http://www.w3.org/2000/svg';
  const XLINK = 'http://www.w3.org/1999/xlink';
  const fig = document.querySelector('.figure');
  const source = document.querySelector('svg.canvas');
  if (!fig || !source) {
    throw new Error('Expected .figure and svg.canvas in the source HTML.');
  }

  const figRect = fig.getBoundingClientRect();
  const canvas = source.cloneNode(true);
  const width = source.viewBox.baseVal.width || source.getBoundingClientRect().width;
  const height = source.viewBox.baseVal.height || source.getBoundingClientRect().height;

  // Physical size for the paper: lock the rendered width to 7.5in and let the
  // height follow the viewBox aspect ratio. The viewBox stays in source px so
  // all geometry and MathJax label positions remain valid.
  const PAPER_WIDTH_IN = 7.5;
  const paperHeightIn = (PAPER_WIDTH_IN * height) / width;

  canvas.setAttribute('xmlns', NS);
  canvas.setAttribute('xmlns:xlink', XLINK);
  canvas.setAttribute('width', `${PAPER_WIDTH_IN}in`);
  canvas.setAttribute('height', `${paperHeightIn.toFixed(4)}in`);
  canvas.setAttribute('viewBox', `0 0 ${width} ${height}`);
  canvas.removeAttribute('style');
  canvas.querySelectorAll('style').forEach(style => style.remove());

  const styleProps = [
    'fill',
    'stroke',
    'stroke-width',
    'stroke-linecap',
    'stroke-linejoin',
    'stroke-dasharray',
    'opacity',
    'display'
  ];

  const originalItems = Array.from(source.querySelectorAll('rect,path,line,circle'));
  const cloneItems = Array.from(canvas.querySelectorAll('rect,path,line,circle'));
  originalItems.forEach((original, index) => {
    const clone = cloneItems[index];
    if (!clone) return;

    const computed = getComputedStyle(original);
    const inlineFill = original.getAttribute('fill');
    const keepInlineFill = inlineFill && inlineFill.indexOf('url(') === 0;
    styleProps.forEach(prop => {
      // Preserve pattern fills (e.g. the hatch) as their original relative
      // url(#id) reference; the computed value resolves to an absolute URL
      // that would not resolve inside the standalone SVG.
      if (prop === 'fill' && keepInlineFill) {
        clone.setAttribute('fill', inlineFill);
        return;
      }
      const value = computed.getPropertyValue(prop);
      if (value && value !== 'normal' && value !== 'auto') {
        clone.setAttribute(prop, value);
      }
    });
  });

  const background = canvas.querySelector('rect[data-background="true"]');
  if (background) {
    background.setAttribute('fill', '#ffffff');
    background.setAttribute('stroke', 'none');
    background.setAttribute('opacity', '1');
  }

  document.querySelectorAll('.math').forEach(label => {
    const mathSvg = label.querySelector('mjx-container svg');
    if (!mathSvg) return;

    const rect = mathSvg.getBoundingClientRect();
    const clone = mathSvg.cloneNode(true);
    clone.setAttribute('x', String(rect.left - figRect.left));
    clone.setAttribute('y', String(rect.top - figRect.top));
    clone.setAttribute('width', String(rect.width));
    clone.setAttribute('height', String(rect.height));
    clone.setAttribute('color', getComputedStyle(label).color);
    clone.removeAttribute('style');
    canvas.appendChild(clone);
  });

  return new XMLSerializer().serializeToString(canvas);
}
"""


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate an editable SVG from the problem-setup HTML figure."
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=script_dir / "problem_setup_figure.html",
        help="Source HTML figure.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=script_dir / "problem_setup_figure.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=script_dir / "problem_setup_figure.png",
        help="Optional PNG preview path. Use an empty string to skip.",
    )
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=1200,
        help="Extra wait after MathJax renders, in milliseconds.",
    )
    return parser.parse_args()


def html_with_editable_math(html_path: Path) -> str:
    html = html_path.read_text(encoding="utf-8")
    return html.replace("fontCache: 'global'", "fontCache: 'none'")


def export_svg(
    html_path: Path,
    out_path: Path,
    png_path: Path | None,
    width: int,
    height: int,
    wait_ms: int,
) -> None:
    html_path = html_path.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".html",
        prefix=".problem_setup_export_",
        dir=html_path.parent,
        delete=False,
    ) as tmp:
        tmp.write(html_with_editable_math(html_path))
        tmp_path = Path(tmp.name)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
            )
            page.goto(tmp_path.as_uri(), wait_until="domcontentloaded")
            # MathJax labels are optional; wait for them only if the page has any.
            try:
                page.wait_for_selector("mjx-container svg", state="attached", timeout=4000)
            except PlaywrightTimeoutError:
                pass
            # Ensure the Computer Modern web font is loaded before screenshotting.
            page.evaluate("document.fonts ? document.fonts.ready : true")
            page.wait_for_timeout(wait_ms)
            if png_path is not None:
                png_path = png_path.resolve()
                png_path.parent.mkdir(parents=True, exist_ok=True)
                page.locator("#figure").screenshot(path=str(png_path))
            svg_text = page.evaluate(EXPORT_JS)
            browser.close()
    finally:
        tmp_path.unlink(missing_ok=True)

    out_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_text + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    png_path = None if str(args.png) == "" else args.png
    export_svg(args.html, args.out, png_path, args.width, args.height, args.wait_ms)
    print(args.out.resolve())
    if png_path is not None:
        print(png_path.resolve())


if __name__ == "__main__":
    main()
