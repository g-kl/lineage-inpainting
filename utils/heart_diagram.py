"""
heart_diagram.py
================
Programmatic rendering + recolouring of the developing-mouse-heart diagram.

The geometry (path `d` data, draw order, stroke styling) is taken from the
source Adobe Illustrator SVG; only the *fill* of each anatomical region changes.

Two entry points
----------------
plot_heart(...)   -> draw with matplotlib (pass a {region: value} dict and a
                     colormap; optional colourbar).
recolor_svg(...)  -> write a new *SVG* whose regions are recoloured: same paths,
                     strokes and draw order, region fills overridden and
                     data-region tags added (vector output).

Regions
-------
    OFT, aSHF, RV, LV, AV canal, Atria, IFT      (fillable tissue)
    pSHF                                          (boundary arc only, no filled area)

Each SVG <path> is assigned to a region below in REGION_OF. Ambiguous
assignments are marked <judgement> in REGION_OF.
"""

import xml.etree.ElementTree as ET
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

# --------------------------------------------------------------------------- #
#  Region assignment (path index in original draw order -> (region, role))    #
#                                                                             #
#  role:  'fill'    normal tissue, recoloured by value                        #
#         'cavity'  a lumen (the grey ellipse inside the LV); neutral default #
#         'outline' stroke-only guide line (fill:none in the source)          #
# --------------------------------------------------------------------------- #
REGION_OF = {
    0:  ("LV",       "cavity"),   # grey gradient ellipse = LV lumen
    1:  ("OFT",      "fill"),
    2:  ("aSHF",     "fill"),
    3:  ("RV",       "fill"),
    4:  ("LV",       "fill"),     # main LV myocardium
    5:  ("LV",       "fill"),     # <judgement> thin outer rim of the LV
    6:  ("LV",       "fill"),     # <judgement> teal wedge above the LV cavity
    7:  ("IFT",      "fill"),     # <judgement> far bottom-right lobe (inflow tract tip)
    8:  ("IFT",      "fill"),     # <judgement> highlight sliver beside path 7
    9:  ("Atria",    "fill"),
    10: ("Atria",    "fill"),     # thin highlight band
    11: ("Atria",    "outline"),  # small internal guide arc
    12: ("Atria",    "fill"),     # dark inner atrial cavity wall
    13: ("IFT",      "fill"),
    14: ("IFT",      "fill"),     # highlight sliver
    15: ("IFT",      "fill"),
    16: ("AV canal", "fill"),     # <judgement> superior connecting band/cushion
    17: ("AV canal", "fill"),     # main AV canal body
    18: ("AV canal", "fill"),     # <judgement> inferior connecting band/cushion
    19: ("AV canal", "fill"),     # lower AV canal body
    20: ("pSHF",     "outline"),  # pSHF boundary arc (there is no filled pSHF area)
}

# Canonical region order (also the colourbar / legend order)
REGIONS = ["OFT", "aSHF", "RV", "LV", "AV canal", "Atria", "IFT", "pSHF"]

# Anatomical text-label anchor points (from the original SVG), for show_labels=True
LABEL_POS = {
    "OFT": (45.6, 18.3), "aSHF": (23.3, 36.1), "RV": (7.0, 39.4),
    "LV": (19.3, 70.7), "AV canal": (46.1, 70.4), "Atria": (69.6, 59.5),
    "IFT": (74.7, 102.6), "pSHF": (87.2, 72.7),
}

# Original fill colour of each style class, for reference / default rendering.
CLASS_COLOR = {
    "st0": "#a9c0bd", "st1": "#fed1a3", "st2": "#808080", "st3": "#95b6df",
    "st4": "#efcdb8", "st5": "#0a8a7c", "st6": "#a6a47b", "st7": "#505f25",
    "st8": "#3d7fc2", "st9": "none",    "st10": "#a2d39a", "st11": "#9ecae1",
    "st12": "#d75526", "st13": "#6b7a34",
}

ORIGINAL_SVG = r"""<?xml version="1.0" encoding="UTF-8"?>
<svg id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" viewBox="0 0 102.49 116.34">
  <!-- Generator: Adobe Illustrator 30.6.0, SVG Export Plug-In . SVG Version: 2.1.4 Build 109)  -->
  <defs>
    <style>
      .st0 {
        fill: #a9c0bd;
      }

      .st0, .st1, .st2, .st3, .st4, .st5, .st6, .st7, .st8, .st9, .st10, .st11, .st12, .st13 {
        stroke: #000;
      }

      .st0, .st2, .st3, .st4, .st5, .st6, .st7, .st8, .st9, .st10, .st11, .st12, .st13 {
        stroke-width: .5px;
      }

      .st0, .st2, .st5, .st6, .st7, .st9, .st13 {
        stroke-miterlimit: 10;
      }

      .st1 {
        fill: #fed1a3;
        stroke-width: .54px;
      }

      .st1, .st3, .st4, .st8, .st10, .st11, .st12 {
        stroke-linejoin: round;
      }

      .st2 {
        fill: url(#linear-gradient);
      }

      .st3 {
        fill: #95b6df;
      }

      .st4 {
        fill: #efcdb8;
      }

      .st5 {
        fill: #0a8a7c;
      }

      .st14, .st15, .st16 {
        isolation: isolate;
      }

      .st6 {
        fill: #a6a47b;
      }

      .st15, .st16 {
        font-family: ArialMT, Arial;
        font-size: 6px;
      }

      .st15, .st17 {
        fill: #fff;
      }

      .st16 {
        fill: #010101;
      }

      .st7 {
        fill: #505f25;
      }

      .st8 {
        fill: #3d7fc2;
      }

      .st9 {
        fill: none;
      }

      .st10 {
        fill: #a2d39a;
      }

      .st11 {
        fill: #9ecae1;
      }

      .st12 {
        fill: #d75526;
      }

      .st13 {
        fill: #6b7a34;
      }
    </style>
    <linearGradient id="linear-gradient" x1="18.15" y1="88.87" x2="43.9" y2="74" gradientTransform="translate(0 135.4) scale(1 -1)" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#333"/>
      <stop offset=".15" stop-color="#383838"/>
      <stop offset=".34" stop-color="#474747"/>
      <stop offset=".54" stop-color="#5f5f5f"/>
      <stop offset=".75" stop-color="#818181"/>
      <stop offset=".97" stop-color="#acacac"/>
      <stop offset="1" stop-color="#b3b3b3"/>
    </linearGradient>
  </defs>
  <g>
    <g>
      <path class="st2" d="M16.28,49.9c.17-.42.38-.83.63-1.22.75-1.2,1.89-2.09,3.11-2.8,2.63-1.54,5.74-2.25,8.77-2.25,2.15,0,4.29.36,6.34.99,1.92.58,3.82,1.34,5.41,2.58,1.47,1.14,3.02,2.42,4.01,4.03.59.97.96,2.05,1.29,3.13.19.63.41,1.33.37,1.99-.04.7-.27,1.39-.58,2.02-.38.74-.88,1.41-1.41,2.05-.7.84-1.45,1.64-2.33,2.28-1.85,1.36-4.03,1.39-6.23,1.23-1.91-.14-3.82-.34-5.71-.62-1.79-.27-3.63-.53-5.36-1.09-1.19-.38-2.31-.97-3.41-1.56-1.08-.59-2.11-1.27-2.97-2.15-.89-.91-1.57-2.03-2-3.23-.67-1.83-.62-3.68.07-5.38Z"/>
      <path class="st11" d="M61.92.34c-.01,1.5.91,2.81,1.81,3.93.93,1.15,2.03,2.16,3.24,3,1.01.71,2.1,1.3,3.23,1.79.57.25,1.14.47,1.73.67.55.19,1.1.47,1.69.47-.64.67-1.18,1.44-1.57,2.29-.36.8-.41,1.65-.69,2.48-.59,1.76-1.74,3.29-3,4.64-.89.95-1.84,1.84-2.91,2.59-2,1.39-4.34,2.21-6.72,2.74-1.76.39-3.49.64-5.29.79-1.2.1-2.44.31-3.43,1-1.2.82-1.85,2.17-2.48,3.43-.31.61-1.04.36-1.61.32-.71-.05-1.41-.12-2.11-.2-1.4-.16-2.8-.39-4.19-.66-2.78-.55-5.52-1.31-8.21-2.19-1.92-.63-3.81-1.32-5.69-2.05-.54-.21-1.07-.42-1.59-.67-.25-.12-.82-.29-1-.5-.22-.27.05-.59.27-.8,1.83-1.72,3.62-3.49,5.38-5.29,1.86-1.92,3.73-3.84,5.65-5.71,1.78-1.74,3.53-3.55,5.63-4.91,1.14-.73,2.36-1.33,3.61-1.83,2.35-.95,4.81-1.62,7.31-1.98,1.28-.19,2.57-.27,3.87-.32,1.15-.05,2.44-.09,3.52-.53.71-.29,1.32-.78,1.88-1.3.51-.49.97-1.08,1.68-1.29,0,.02,0,.05,0,.09Z"/>
      <path class="st10" d="M45.01,36.47c-.13.36-.24.72-.31,1.1,0,.01,0,.02-.01.03-.17,1.04.02,2.13.46,3.08.34.74,1.17,1.25,1.27,2.1v.02c-3.06-1.26-6.02-2.35-9.32-2.82-4.14-.59-8.35-.76-12.53-.58-2.09.09-4.18.27-6.26.53-.49.06-.98.15-1.47.2-1.54.18-.61-3.03-.3-3.85.46-1.22,1.12-2.35,1.62-3.56.73-1.76,1.09-3.67,1.06-5.57-.02-1.54,1.74-2.41,3.02-2.87.78-.28,1.65.37,2.39.67,1,.4,2,.78,3,1.15,2.05.75,4.13,1.44,6.24,2.05,4.19,1.21,8.49,2.09,12.84,2.47.09.01.18.02.27.06.57.31-.98,3.3-1.16,3.81-.25.67-.57,1.31-.81,1.98Z"/>
      <path class="st1" d="M18.91,26.42c-.4-.07-1,.21-1.31.32-1.33.46-2.54,1.22-3.66,2.06-1.2.9-2.32,1.89-3.53,2.77-1.19.87-2.35,1.79-3.33,2.91-3.2,3.67-3.77,9.18-2.87,13.81.01.06.03.13.07.17.06.05.14.04.21.03.54-.11.9-.6,1.24-1.03,2.32-2.98,5.92-4.49,9.46-5.44.13-.04.27-.07.37-.17.09-.09.13-.23.15-.35.11-.58-.06-1.14-.01-1.72.08-.93.35-1.84.69-2.71.5-1.28,1.16-2.49,1.66-3.77.43-1.09.75-2.23.94-3.38.1-.59.17-1.19.21-1.79.03-.43.22-1.04.03-1.46-.05-.15-.17-.23-.32-.25Z"/>
      <path class="st5" d="M7.14,45.77c-1.79,1.92-2.45,4.62-2.78,7.22-.7,5.56-.39,12.34,3.36,16.84,2.33,2.79,5.09,5.26,8.26,7.06,3.47,1.97,7.57,2.93,11.5,3.54,2.92.45,5.92.54,8.8-.12,1.67-.38,3.29-1,4.95-1.4,1.69-.41,3.41-.54,5.1-.04-.48,1.71-.97,3.41-1.46,5.12-.24.83-.48,1.66-.72,2.49-.18.62-.15,1.7-.61,2.19-.56.59-2.35.07-3.11,0-2.53-.23-5.07-.46-7.6-.7-3.2-.3-6.41-.59-9.55-1.23-4.84-.97-9.92-2.85-14.1-5.51-1.85-1.18-3.53-2.71-4.57-4.65-1.97-3.65-4.05-7.8-4.31-12.01-.4-6.35,1.37-13.7,5.96-18.33.24-.24.56-.34.88-.47Z"/>
      <path class="st0" d="M46.1,78.93c-.01-.18-.14-.4-.21-.57-.23-.55-.47-1.5-1.11-1.7-.23-.07-.47-.04-.71-.01-3.55.47-7.09,1.1-10.67,1.21-6.51.2-13.49-1.11-18.7-5.24-1.94-1.54-3.59-3.43-4.86-5.55-1.59-2.67-2.55-5.71-2.81-8.81-.26-3.03-.1-6.72,1.16-9.53,1.38-3.09,4.96-5.08,8.02-6.17,3.5-1.24,7.27-1.5,10.98-1.45,5,.07,9.99.71,14.79,2.12.45.13,1.01.54,1.48.51.71-.04.72-.86.73-1.51,0-.16-.01-.33-.12-.45-.08-.08-.19-.12-.29-.16-4.45-1.57-9.17-2.13-13.87-2.21-4.62-.08-9.52-.04-14.01,1.14-.42.11-.84.23-1.26.37-5.56,1.8-10.05,5.96-10.52,12.01-.23,2.9-.49,5.64.01,8.56.87,5.1,3.92,9.7,7.97,12.91,5.33,4.22,12.7,6.46,19.47,6.33,2.1-.04,4.18-.41,6.22-.9,1.92-.46,3.74-1.15,5.72-1.22.52-.02,1.05-.04,1.56.01.37.04.68.19,1.03.31Z"/>
      <path class="st5" d="M43.77,43.91c.22.09-.14,1.68-.18,1.89-.2.95-.5,2.45-1.35,3.06-1.07-1.52-2.75-2.5-4.45-3.25-2.88-1.27-6.01-2.03-9.17-2s-6.32.86-8.97,2.56c-1.34.86-2.56,1.98-3.25,3.41-.73,1.5-.81,3.27-.39,4.89.68,2.64,2.41,4.62,4.73,5.94,1.14.65,2.35,1.19,3.58,1.64.72.26,1.44.53,2.17.73.71.19,1.43.24,2.17.33,1.58.21,3.15.48,4.74.64,1.57.16,3.15.2,4.72.14.34-.01.68-.03,1.02-.07.18-.02.36-.04.53-.06.71-.09,2.04,3.59,2.33,4.22.78,1.72,1.49,3.47,2.12,5.25.22.63.44,1.26.6,1.91.12.5.43,1.03-.03,1.36-.41.29-1.76.27-2.28.36-1.56.26-3.12.48-4.7.64-3.19.33-6.41.43-9.61.09-8.13-.87-15.54-4.64-19.16-12.27-1.29-2.71-1.9-5.68-1.96-8.68-.06-2.87.1-6.07,1.61-8.59,1.93-3.22,5.76-5.04,9.27-5.97,4.28-1.14,8.76-1.2,13.15-.79,4.28.41,8.75.98,12.76,2.62Z"/>
      <path class="st8" d="M87.3,89.27c.65,1.24,1.6,2.3,2.64,3.24.51.47,1.05.91,1.59,1.34.28.23.44.29.38.66-.04.24-.12.46-.17.69-.11.51-.14,1.04-.12,1.56.04,1.07.28,2.12.56,3.14.2.74.43,1.48.73,2.18.32.73.78,1.33,1.19,2.01.14.24.24.52-.14.44-.13-.02-.24-.1-.35-.18-1.33-.97-2.47-2.16-3.59-3.36-1.24-1.34-2.4-2.74-3.5-4.2-.38-.5-.76-1.02-1.02-1.6-.27-.61-.27-1.23-.22-1.89.1-1.46.79-3.13,2.02-4.03Z"/>
      <path class="st3" d="M83.86,96.74c.06.13.2.28.26.36,1.38,1.97,2.83,3.88,4.53,5.58.61.61,1.25,1.2,1.87,1.8.68.67,1.33,1.38,2.05,2,.14.12.4.43.58.47.37.09,1.12-1,1.31-1.26.07-.09.14-.19.16-.3.09-.38-.42-.57-.66-.75-2.81-2.07-5.11-4.72-7.22-7.48-.32-.42-.65-.84-.98-1.27-.07-.09-.13-.17-.19-.27-.06-.1-.12-.21-.21-.3-.01-.01-.03-.02-.05-.02s-.03.01-.05.02c-.33.28-.65.56-.98.83-.1.08-.45.29-.48.41.03.08.04.13.06.18Z"/>
      <path class="st13" d="M87.14,89.27s0,.04-.01.06c-.02.06-.07.11-.12.15-.78.69-1.18,1.67-1.44,2.66-.27,1.02-.29,1.84-.21,2.89.04.45-.84,1.14-1.14,1.39-.09.08-.19.15-.31.16-.09.01-.19-.03-.27-.07-3.73-1.69-7.15-4.07-10.02-6.99-.18-.18-.54-.6-.37-.88.12-.19.4-.18.62-.16.59.04,1.2-.12,1.7-.45.11-.07.23-.17.24-.31,0-.01-1.56-.7-1.7-.78-2.23-1.3-3.93-3.57-5.13-5.8-1.22-2.28-1.98-4.81-2.19-7.39-.1-1.25-.04-2.5-.01-3.74.02-.89.09-1.79.21-2.67.06-.46.15-.92.24-1.38.08-.37.28-.76.25-1.13-.02-.23-.09-.45-.17-.67-.32-.87-.64-1.74-.94-2.62,2.89-.1,5.74.07,8.56.86.99.28,1.96.62,2.89,1.05,3.46,1.6,6.32,4.41,7.57,8.04.95,2.75,1.31,5.67,1.33,8.57v.52c-.01,1.44-.14,2.87-.14,4.31-.01,1.48.22,2.8.56,4.23,0,.04.01.1,0,.15Z"/>
      <path class="st6" d="M66.39,61.61c-.01.12.14.3.17.39.1.21.14.44.21.66.13.47.32.92.51,1.36.08.18,2,.07,2.24.11.91.13,1.8.42,2.62.83,1.36.67,2.54,1.66,3.57,2.77,1.89,2.04,3.28,4.53,4.05,7.2.81,2.82.84,5.87,1.1,8.78.18,2.03.17,4.1.6,6.1.31,1.46.85,2.86,1.39,4.25.34.89.69,1.79,1.27,2.54.39-.28,1.36-.93,1.38-1.46.01-.3-.1-.59-.2-.87-.84-2.23-1.44-4.54-1.79-6.9-.72-4.83-.41-9.91-2.26-14.52-1.02-2.55-2.72-5.26-4.6-7.27-2.59-2.78-6.27-3.86-9.96-4.09-.09-.01-.18-.01-.25.05-.04,0-.05.04-.05.07Z"/>
      <path class="st9" d="M80.44,78.56c0,5.06-2.07,9.16-4.63,9.16"/>
      <path class="st7" d="M67.32,64.16c.3.49.22,1.12.05,1.67-.31,1.01-.48,2-.58,3.06-.15,1.53-.14,3.08.02,4.61.49,4.66,2.39,9.75,6.09,12.81.97.8,2.18,1.47,3.42,1.27,1.47-.24,2.46-1.62,3.06-2.99,1.76-4.05,1.1-8.73-.86-12.58-1.94-3.8-5.34-7.57-9.85-7.88-.45-.03-.91-.03-1.36.01,0,.01,0,.02,0,.02Z"/>
      <path class="st8" d="M73.15,89.28c-.07-.02-.12-.02-.16,0-1.5.86-2.98,1.74-4.53,2.5-.04.02-.08.04-.1.08-.02.04-.01.08-.01.13.37,2.07,2.01,4.12,3.08,5.91,1.23,2.06,2.52,4.09,3.89,6.06,1.35,1.94,2.78,3.82,4.32,5.62.8.94,1.63,1.85,2.5,2.74.85.88,1.36,1.82,2.56,2.24-1.25-.44,2.33-5.68,2.9-6.19.97-.87,2.17-1.24,3.38-1.63.93-.3,1.36-.62.45-1.42-1.65-1.42-3.19-2.98-4.6-4.64-.72-.85-1.41-1.73-2.06-2.64-.69-.96-1.12-1.5-2.21-2-1.16-.53-2.27-1.2-3.35-1.89-1.16-.75-2.27-1.57-3.33-2.45-.39-.33-2.1-2.23-2.73-2.42Z"/>
      <path class="st3" d="M68.1,92s-.04-.02-.07-.02c-.06-.01-.12.02-.18.04-.37.15-.77.31-1.1.54-.48.33-.43.33-.16.84.45.85.93,1.68,1.39,2.52,1.89,3.43,3.74,6.81,6.04,9.99,2.28,3.14,4.83,6.09,7.61,8.79.25.24.5.48.76.72.79.74,1.51.76,2.08-.24.29-.51.08-.5-.39-.94-.71-.66-1.4-1.34-2.08-2.04-1.35-1.38-2.63-2.83-3.85-4.33-2.44-2.99-4.64-6.19-6.62-9.5-.32-.53-.63-1.07-.94-1.61-.14-.25-.29-.51-.43-.76-.4-.71-.78-1.42-1.16-2.15-.18-.36-.56-1.68-.9-1.85Z"/>
      <path class="st8" d="M60.63,96.94c.97-.6,1.98-1.2,2.93-1.85.49-.33.97-.68,1.43-1.05.24-.19.47-.39.69-.59.12-.11.4-.49.6-.41.08.03.13.12.18.2,1.22,2.21,2.39,4.47,3.65,6.66.42.74.86,1.46,1.31,2.18.9,1.44,1.85,2.84,2.85,4.21.43.59.96,1.17,1.32,1.81.55.71,1.12,1.4,1.7,2.08.83.97,1.7,1.91,2.59,2.83.44.46.9.91,1.35,1.37.4.41.77.85,1.24,1.19.19.14.41.28.48.51-.32.07-.63-.13-.89-.32-.81-.58-1.58-1.21-2.39-1.79-.83-.59-1.68-1.15-2.5-1.74-2.07-1.55-4.21-3.02-6.32-4.52-1.79-1.27-3.49-2.58-5.03-4.15-1.04-1.06-2.08-2.12-3.05-3.24-.49-.57-.96-1.15-1.4-1.76-.22-.3-.43-.61-.63-.92-.2-.3-.46-.48-.11-.7Z"/>
      <path class="st4" d="M44.12,41.8s-.03.1-.02.16c.1.6-.02,1.22-.32,1.74-.03.05-.06.1-.05.16.01.03.03.06.05.08.42.44,1.08.42,1.59.64.65.28,1.27.74,1.85,1.13,1.2.81,2.32,1.74,3.34,2.77.73.73,1.4,1.51,1.93,2.4.99,1.68,1.4,3.64,2.19,5.41.46,1.04,1.04,2.03,1.7,2.96,1.97,2.73,5.46,4.09,8.71,4.37.32.03.65.02.97.05.23.02,1,.19,1.09,0,.08-.17-.31-.95-.37-1.15,0-.24-.11-.5-.18-.73-.11-.33-.25-.4-.6-.49-1.42-.37-2.89-.6-4.26-1.15-1.53-.6-3.11-1.59-3.91-3.07-.79-1.45-1.44-2.97-1.95-4.54-1.2-3.7-3.93-6.78-7.23-8.76-1.07-.64-2.2-1.13-3.35-1.6-.35-.14-.68-.31-1.04-.41-.04-.01-.07-.02-.11,0,0,.01-.02.02-.03.03Z"/>
      <path class="st12" d="M46.59,45.29c-.57-.38-1.16-.73-1.8-.97-.17-.07-.35-.13-.53-.18-.14-.04-.28-.07-.41-.15-.04.52-.07,1.01-.21,1.53-.14.54-.33,1.07-.45,1.61-.05.23-.08.47-.18.68-.19.42-.61.71-.81,1.12.07-.14,1.81,1.47,1.94,1.63.58.71.93,1.54,1.25,2.4.34.91.7,1.93.75,2.91.05,1.01-.27,2.01-.73,2.9-.92,1.77-2.55,3.6-4.38,4.44-.27.12-.81.2-.98.46-.09.14-.01.32.07.47.47.87.93,1.73,1.36,2.62.55,1.14,1.06,2.29,1.49,3.47.39,1.08.71,2.18,1.16,3.23.21.5.45.99.59,1.51.12.44.17.9.2,1.36.04.62,1.49.33,1.92.3.9-.06,1.8-.17,2.7-.17,1.34-.01,2.68.2,3.98.54,1.1.29,2.19.68,3.19,1.21,1.04.56,1.98,1.31,2.92,2.02,1.55,1.16,2.87,2.65,4.02,4.19.36.48.7.96,1.03,1.46,1.2,1.81,2.22,3.72,3.26,5.62.07.14.16.28.3.35.24.11.78-.35.99-.46.45-.24.91-.49,1.36-.73.9-.49,1.8-1,2.65-1.59.33-.23.67-.48,1.05-.62.4-.15.79-.2,1.17-.42.09-.05.19-.15.14-.24-.02-.05-.09-.08-.14-.1-.93-.31-1.74-.63-2.51-1.25-2.06-1.64-3.4-3.75-4.44-6.13-.74-1.7-1.28-3.5-1.58-5.34-.38-2.25-.47-4.66-.08-6.91.13-.75.32-1.48.45-2.23.09-.51.37-1.56-.11-1.96-.1-.08-.23-.11-.36-.12-1.22-.16-2.46-.19-3.67-.43-1.2-.24-2.38-.69-3.46-1.26-1.56-.82-2.79-1.92-3.77-3.39-1.24-1.86-1.78-3.95-2.61-5.98-.73-1.77-1.9-3.35-3.27-4.68-.79-.76-1.64-1.45-2.53-2.09-.32-.21-.62-.43-.93-.63Z"/>
      <path class="st4" d="M44.74,76.55c.54.49.99,1.02,1.18,1.73.06.21.1.45.28.57.21.13.48.04.72-.01.43-.09.88-.04,1.33-.01.94.05,1.88-.04,2.82.03,1.07.09,2.1.39,3.1.76.96.36,1.89.78,2.73,1.4.32.24.62.5.92.76.95.83,1.88,1.68,2.75,2.58.88.91,1.59,1.93,2.32,2.96s1.42,2.09,2.06,3.18c.26.45.52.9.77,1.35.13.23.25.47.37.71.09.17.1.47.34.29.51-.38,1.08-.67,1.69-.84.02-.01.05-.01.07-.03.07-.09-.25-.56-.3-.64-.15-.27-.3-.55-.45-.82-.31-.55-.62-1.1-.94-1.64-.63-1.07-1.28-2.13-1.98-3.17-.67-1-1.39-1.96-2.18-2.87-.65-.75-1.36-1.44-2.11-2.09-.91-.78-1.89-1.49-2.91-2.13-2.05-1.27-4.31-1.96-6.72-2.12-.95-.06-1.92-.04-2.87.08-.51.06-1.02.14-1.54.15-.26,0-.51-.02-.76-.08-.14-.03-.54-.26-.65-.18-.03.02-.04.05-.06.08,0-.02.01,0,.02,0Z"/>
      <path class="st12" d="M65.26,91.05c.14.25.27.49.4.74.14.25.32.51.42.78.15.39-.03.44-.34.72-.95.85-1.97,1.62-3.03,2.33-.61.41-1.23.8-1.86,1.17-.38.23-.48.33-.83.02-.25-.23-.45-.5-.67-.76-2.07-2.53-4.82-4.83-7.94-5.91-1.31-.45-2.68-.69-4.05-.82-.7-.07-1.41-.11-2.12-.15-.21-.01-1.82-.28-1.91-.17.36-.49.44-1.03.6-1.6.25-.89.5-1.77.75-2.66.3-1.08.6-2.16.9-3.24.14-.53.29-1.05.43-1.58.08-.3.16-.97.55-1.04.75-.13,1.49-.09,2.24-.13.83-.04,1.67,0,2.5.12,1.65.24,3.26.79,4.7,1.63,1.81,1.05,3.35,2.53,4.73,4.11,1.72,1.99,3.26,4.15,4.53,6.44Z"/>
    </g>
    <g id="text_6">
      <text class="st16" transform="translate(23.28 36.08)"><tspan x="0" y="0">aSHF</tspan></text>
    </g>
    <g id="text_61" data-name="text_6">
      <text class="st16" transform="translate(45.59 18.34)"><tspan x="0" y="0">OFT</tspan></text>
    </g>
    <g id="text_62" data-name="text_6">
      <text class="st16" transform="translate(6.97 39.39)"><tspan x="0" y="0">RV</tspan></text>
    </g>
    <g id="text_63" data-name="text_6">
      <text class="st15" transform="translate(19.33 70.67)"><tspan x="0" y="0">LV</tspan></text>
      <g id="text_64" data-name="text_6">
        <g class="st14">
          <text class="st15" transform="translate(49.28 67.36)"><tspan x="0" y="0">AV</tspan></text>
          <text class="st15" transform="translate(46.11 73.36)"><tspan x="0" y="0">canal</tspan></text>
        </g>
      </g>
    </g>
    <g id="text_65" data-name="text_6">
      <text class="st16" transform="translate(69.57 59.45)"><tspan x="0" y="0">Atria</tspan></text>
    </g>
    <g id="text_66" data-name="text_6">
      <text class="st15" transform="translate(74.68 102.63)"><tspan x="0" y="0">IFT</tspan></text>
    </g>
  </g>
  <path class="st9" d="M82.66,54.87c6.63,4.35,10.74,11.81,12.56,19.53s1.58,15.76,1.09,23.68"/>
  <rect class="st17" x="91.49" y="67.36" width="5.65" height="6.57"/>
  <g id="text_67" data-name="text_6">
    <text class="st16" transform="translate(87.15 72.73)"><tspan x="0" y="0">pSHF</tspan></text>
  </g>
</svg>"""

# --------------------------------------------------------------------------- #
#  Minimal SVG path parser (handles M m C c S s L l H h V v Z z)               #
# --------------------------------------------------------------------------- #
_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
_TOK = re.compile(r"[MmCcSsLlHhVvZz]|" + _NUM)


def _parse_d(d):
    """Convert an SVG path 'd' string into (vertices, codes) for matplotlib."""
    toks = _TOK.findall(d)
    i = 0
    verts, codes = [], []
    cur = np.array([0.0, 0.0]); start = cur.copy(); prev_ctrl2 = None; prev_cmd = None
    cmd = None

    def _n():
        nonlocal i
        v = float(toks[i]); i += 1; return v

    while i < len(toks):
        if re.fullmatch(r"[A-Za-z]", toks[i]):
            cmd = toks[i]; i += 1
        if cmd in ("M", "m"):
            x, y = _n(), _n()
            cur = cur + [x, y] if cmd == "m" else np.array([x, y])
            verts.append(tuple(cur)); codes.append(MplPath.MOVETO)
            start = cur.copy(); prev_ctrl2 = None
            cmd = "l" if cmd == "m" else "L"           # extra pairs become line-to
        elif cmd in ("C", "c"):
            c1, c2, e = (np.array([_n(), _n()]) for _ in range(3))
            if cmd == "c":
                c1, c2, e = cur + c1, cur + c2, cur + e
            verts += [tuple(c1), tuple(c2), tuple(e)]; codes += [MplPath.CURVE4] * 3
            prev_ctrl2 = c2; cur = e
        elif cmd in ("S", "s"):
            c2r, e = np.array([_n(), _n()]), np.array([_n(), _n()])
            c2, e = (cur + c2r, cur + e) if cmd == "s" else (c2r, e)
            c1 = 2 * cur - prev_ctrl2 if prev_cmd in ("C", "c", "S", "s") else cur.copy()
            verts += [tuple(c1), tuple(c2), tuple(e)]; codes += [MplPath.CURVE4] * 3
            prev_ctrl2 = c2; cur = e
        elif cmd in ("V", "v"):
            y = _n(); cur = np.array([cur[0], cur[1] + y]) if cmd == "v" else np.array([cur[0], y])
            verts.append(tuple(cur)); codes.append(MplPath.LINETO); prev_ctrl2 = None
        elif cmd in ("H", "h"):
            x = _n(); cur = np.array([cur[0] + x, cur[1]]) if cmd == "h" else np.array([x, cur[1]])
            verts.append(tuple(cur)); codes.append(MplPath.LINETO); prev_ctrl2 = None
        elif cmd in ("L", "l"):
            x, y = _n(), _n(); cur = cur + [x, y] if cmd == "l" else np.array([x, y])
            verts.append(tuple(cur)); codes.append(MplPath.LINETO); prev_ctrl2 = None
        elif cmd in ("Z", "z"):
            verts.append(tuple(start)); codes.append(MplPath.CLOSEPOLY)
            cur = start.copy(); prev_ctrl2 = None
        else:
            i += 1; continue
        prev_cmd = cmd
    return verts, codes


_SVG_NS = "{http://www.w3.org/2000/svg}"


def _load_segments():
    """Parse the embedded SVG once into an ordered list of segment dicts."""
    root = ET.fromstring(ORIGINAL_SVG)
    vb = [float(v) for v in root.get("viewBox").split()]
    segs = []
    for idx, el in enumerate(root.iter(_SVG_NS + "path")):
        cls = el.get("class")
        region, role = REGION_OF.get(idx, (None, "fill"))
        segs.append(dict(id=idx, region=region, role=role, svg_class=cls,
                         d=el.get("d", ""), orig_color=CLASS_COLOR.get(cls, "#cccccc")))
    return vb, segs


VIEWBOX, SEGMENTS = _load_segments()


# --------------------------------------------------------------------------- #
#  Colour helpers                                                             #
# --------------------------------------------------------------------------- #
def colors_from_values(values, cmap="viridis", vmin=None, vmax=None):
    """Map a {region: number} dict to a {region: hex-color} dict + return the
    ScalarMappable (useful for a colourbar)."""
    vals = np.array(list(values.values()), dtype=float)
    vmin = np.nanmin(vals) if vmin is None else vmin
    vmax = np.nanmax(vals) if vmax is None else vmax
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = {r: mpl.colors.to_hex(sm.to_rgba(v)) for r, v in values.items()}
    return colors, sm


# --------------------------------------------------------------------------- #
#  1) matplotlib renderer                                                     #
# --------------------------------------------------------------------------- #
def plot_heart(values=None, colors=None, *, cmap="viridis", vmin=None, vmax=None,
               ax=None, show_labels=True, colorbar=True, colorbar_label="",
               default_color="#eeeeee", cavity_color="#d9d9d9",
               edgecolor="black", linewidth=0.5, outline_style="-",
               hide_regions=None, draw_unassigned=True, label_overrides=None,
               label_fontsize=6):
    """
    Draw the heart, colouring each region.

    Provide exactly one of:
        values : {region: number}   -> mapped through `cmap` (vmin..vmax), colourbar shown
        colors : {region: color}    -> explicit per-region colours

    Omitted regions are drawn in `default_color`. Returns (ax, scalar_mappable).
    `scalar_mappable` is None when explicit `colors` are used.

    hide_regions   : iterable of region names to skip entirely (paths + label)
    draw_unassigned: draw paths with no region when True
    label_overrides: {region: text} to replace the default label text (e.g. multi-line)
    label_fontsize : point size of the anatomical labels
    """
    sm = None
    if values is not None and colors is not None:
        raise ValueError("Pass either `values` or `colors`, not both.")
    if values is not None:
        colors, sm = colors_from_values(values, cmap=cmap, vmin=vmin, vmax=vmax)
    colors = dict(colors or {})
    hide = set(hide_regions or ())
    label_overrides = dict(label_overrides or {})

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 7))

    for seg in SEGMENTS:                       # original draw order preserved
        if seg["region"] in hide or (seg["region"] is None and not draw_unassigned):
            continue
        mpath = MplPath(*_parse_d(seg["d"]))
        if seg["role"] == "outline":
            ax.add_patch(PathPatch(mpath, facecolor="none", edgecolor=edgecolor,
                                   lw=linewidth, ls=outline_style))
        elif seg["role"] == "cavity":
            fc = colors.get(seg["region"] + " cavity", cavity_color)
            ax.add_patch(PathPatch(mpath, facecolor=fc, edgecolor=edgecolor, lw=linewidth))
        else:
            fc = colors.get(seg["region"], default_color)
            ax.add_patch(PathPatch(mpath, facecolor=fc, edgecolor=edgecolor,
                                   lw=linewidth, joinstyle="round"))

    if show_labels:
        for region, (x, y) in LABEL_POS.items():
            if region in hide:
                continue
            ax.text(x, y, label_overrides.get(region, region), fontsize=label_fontsize,
                    ha="left", va="baseline", zorder=10)

    x0, y0, w, h = VIEWBOX
    ax.set_xlim(x0, x0 + w); ax.set_ylim(y0, y0 + h)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")

    if colorbar and sm is not None:
        cb = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        if colorbar_label:
            cb.set_label(colorbar_label)
    return ax, sm


# --------------------------------------------------------------------------- #
#  2) SVG recolourer (keeps paths, strokes and draw order)                    #
# --------------------------------------------------------------------------- #
def recolor_svg(colors, out_path=None, *, recolor_cavity=False):
    """
    Return (and optionally write) a recoloured copy of the original SVG.

    `colors` : {region: color}. Only the `fill` of matching paths is overridden
    (via an inline style, which beats the CSS class); every path, stroke,
    gradient, transform and the draw order is left exactly as in the source.
    A `data-region` attribute is added to each path so the output is traceable.
    Outline (fill:none) paths are never touched. The LV cavity is only recoloured
    if recolor_cavity=True and `colors` has a "LV cavity" key.
    """
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(ORIGINAL_SVG)

    for idx, el in enumerate(root.iter(_SVG_NS + "path")):
        region, role = REGION_OF.get(idx, (None, "fill"))
        if region:
            el.set("data-region", region)
        if role == "outline":
            continue
        key = (region + " cavity") if role == "cavity" else region
        if role == "cavity" and not recolor_cavity:
            continue
        if key in colors:
            style = el.get("style", "")
            style = re.sub(r"fill:[^;]*;?", "", style).strip()
            el.set("style", (style + ";" if style else "") + "fill:%s" % colors[key])

    out = ET.tostring(root, encoding="unicode")
    out = '<?xml version="1.0" encoding="UTF-8"?>\n' + out
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(out)
    return out


# --------------------------------------------------------------------------- #
#  Demo when run directly                                                     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Example per-region scores.
    preds = {"OFT": 0.12, "aSHF": 0.44, "RV": 0.71, "LV": 0.95,
             "AV canal": 0.33, "Atria": 0.58, "IFT": 0.20}
    ax, sm = plot_heart(values=preds, cmap="magma", colorbar_label="model score")
    ax.figure.savefig("heart_demo.png", dpi=200, bbox_inches="tight", facecolor="white")

    recolor_svg({"OFT": "#4575b4", "aSHF": "#91bfdb", "RV": "#e0f3f8",
                 "LV": "#fee090", "AV canal": "#fc8d59", "Atria": "#d73027",
                 "IFT": "#999999"}, out_path="heart_recolored.svg")
    print("wrote heart_demo.png and heart_recolored.svg")