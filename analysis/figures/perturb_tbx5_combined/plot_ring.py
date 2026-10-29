"""panel a for this sheet: perturb's ring, drawn smaller.

the insets, the ring radius and the central map are shrunk, and the key's reference
umap is smaller. the radius is the smallest that keeps neighbouring insets apart.
drawn from perturb/numerical_results/ with perturb/plot_ring.py's own helpers and pixel
scale, so the text keeps its size. writes imgs/panels/{ring,refkey}.svg.
"""
import itertools
import os
import pickle

import numpy as np
import pandas as pd

from analysis.figures.perturb import plot_ring as PR
from analysis.figures.perturb import style as PS
from analysis.figures.perturb.compute_ring import select_representatives
from analysis.figures.perturb_tbx5_combined import style as S
from utils.figure_labels import text_svg_path as T

PX = PR.W / PR.PAGE_W   # px per inch, perturb's
INK = PR.INK

## RING CANVAS (px) ##
# the slots sit at 15 + 30k degrees, so two neighbours are at least 30 degrees off the
# diagonal and clear each other once 2 R sin 15 cos 30 >= PANEL + GAP
N_SLOT = 12
PANEL = 112.0           # inset side
GAP = 6.0               # between neighbouring insets, at their closest
R = (PANEL + GAP) / (2 * np.sin(np.pi / 12) * np.cos(np.pi / 6))
C = 340.0               # central map square side; the map clears the 45 degree insets
EDGE = 4.0              # room for the inset outlines at the canvas edge
W = H = 2 * (R * np.cos(np.pi / 12) + PANEL / 2 + EDGE)
LABEL_FS = 17.5         # gene names, ~6 pt

## KEY CANVAS (px) ##
KW = 270.0              # the widest line of text
REF = 250.0             # reference umap side
REF_Y = 24.0
LEG_STEP = 22.0
CBAR_H = 17.0


def ring_slots():
    """inset centres on the ring, in angular order."""
    a = np.deg2rad(15.0 + 30.0 * np.arange(N_SLOT))
    return [(W / 2 + R * np.cos(t), H / 2 + R * np.sin(t)) for t in a]


def route(q, slot):
    """perturb's connector: from the inset edge towards the point q, a straight leg
    then a 45 degree run."""
    qx, qy = q
    ax_, ay_ = PR.panel_anchor(slot[0], slot[1], qx, qy, PANEL / 2)
    dx, dy = qx - ax_, qy - ay_
    corner = ((ax_ + np.sign(dx) * (abs(dx) - abs(dy)), ay_) if abs(dx) >= abs(dy)
              else (ax_, ay_ + np.sign(dy) * (abs(dy) - abs(dx))))
    return [np.array([ax_, ay_]), np.array(corner), np.array([qx, qy], float)]


def tables(qs, slots):
    """per (rep, slot): how many other insets its connector runs through (the insets
    are transparent) and its length; per two of those: their crossings."""
    n = len(qs)
    poly = [[route(q, s) for s in slots] for q in qs]
    t = np.linspace(0.0, 1.0, 24)[:, None]
    half = PANEL / 2 + 2
    hits = np.zeros((n, n), int)
    for r, b in itertools.product(range(n), range(n)):
        a, e, q = poly[r][b]
        on = np.vstack([a + t * (e - a), e + t * (q - e)])
        hits[r, b] = sum(np.any((np.abs(on[:, 0] - s[0]) < half)
                                & (np.abs(on[:, 1] - s[1]) < half))
                         for j, s in enumerate(slots) if j != b)
    length = np.array([[np.hypot(*(p[2] - p[0])) for p in row] for row in poly])
    cross = np.zeros((n, n, n, n), int)
    for r1, b1, r2, b2 in itertools.product(range(n), repeat=4):
        if r1 < r2 and b1 != b2:
            p, q = poly[r1][b1], poly[r2][b2]
            cross[r1, b1, r2, b2] = cross[r2, b2, r1, b1] = sum(
                PR.seg_cross(p[i], p[i + 1], q[j], q[j + 1])
                for i in range(2) for j in range(2))
    return hits, length, cross


def assign(rep_idx, pts, lineages, slots, cx, cy):
    """perturb's like-next-to-like assignment: the reps depleting one
    coarse type stay in adjacent slots, and the group order, each group's direction
    and the ring offset are searched for the fewest crossings and inset hits, then
    the shortest connectors."""
    reps = list(rep_idx)
    hits, length, cross = tables([pts[r] for r in reps], slots)
    pos = {r: i for i, r in enumerate(reps)}
    n = len(slots)

    def cost(order, off):
        r = np.array([pos[o] for o in order])
        b = (off + np.arange(len(r))) % n
        bad = hits[r, b].sum() + cross[r[:, None], b[:, None], r, b].sum() // 2
        return int(bad), float(length[r, b].sum())

    bearing = {i: np.arctan2(pts[i][1] - cy, pts[i][0] - cx) for i in reps}
    circmean = lambda v: np.arctan2(np.mean(np.sin(v)), np.mean(np.cos(v)))
    groups = {}
    for i in reps:
        groups.setdefault(lineages[i], []).append(i)
    blocks = [sorted(m, key=bearing.get) for m in groups.values()]
    blocks.sort(key=lambda b: circmean([bearing[i] for i in b]))
    flat = lambda seq: [i for b in seq for i in b]

    best = None
    for off in range(n):
        seq, c = blocks, cost(flat(blocks), off)
        improved = True
        while improved:
            improved = False
            m = len(seq)
            for cand in ([seq[a:] + seq[:a] for a in range(1, m)]
                         + [seq[:i] + [seq[i][::-1]] + seq[i + 1:]
                            for i in range(m) if len(seq[i]) > 1]
                         + [[seq[j] if k == i else seq[i] if k == j else seq[k]
                             for k in range(m)]
                            for i in range(m) for j in range(i + 1, m)]):
                cc = cost(flat(cand), off)
                if cc < c:
                    seq, c, improved = cand, cc, True
        if best is None or c < best[0]:
            best = (c, flat(seq), off)
    (bad, _), order, off = best
    print(f"connector crossings + inset hits: {bad}", flush=True)
    return {r: slots[(off + i) % n] for i, r in enumerate(order)}


def write(name, text, w, h):
    p = os.path.join(S.PANELS, f"{name}.svg")
    with open(p, "w") as fh:
        fh.write(text)
    print(f"wrote {p}", flush=True)
    return p, (w / PX, h / PX)


def ring_svg(df, panel_png):
    xy = df[["umap1", "umap2"]].to_numpy(float)
    lin = df["depleted_lineage"].to_numpy()
    pts = PR.make_mapper(xy, W / 2, H / 2, C)(xy)
    rep_idx, _ = select_representatives(df)
    at = assign(rep_idx, pts, lin, ring_slots(), W / 2, H / 2)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{W / PX:.3f}in" height="{H / PX:.3f}in" viewBox="0 0 {W:.0f} {H:.0f}">',
           '<defs>']
    for i, idx in enumerate(rep_idx):
        x0, y0 = at[idx][0] - PANEL / 2, at[idx][1] - PANEL / 2
        svg.append(f'<clipPath id="clip{i}"><rect x="{x0:.2f}" y="{y0:.2f}" '
                   f'width="{PANEL:.2f}" height="{PANEL:.2f}" rx="9" ry="9"/></clipPath>')
    svg.append('</defs>')
    svg.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>')

    # central map, then the representative markers
    svg.append('<g>')
    for (x, y), ln in zip(pts, lin):
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{PR.PT_R}" '
                   f'fill="{PR.LINEAGE_COLORS.get(ln, "#b0b0b0")}" fill-opacity="0.85"/>')
    svg.append('</g>')
    for idx in rep_idx:
        qx, qy = pts[idx]
        for col, sw in (("white", 3.2), (INK, 1.5)):
            svg.append(f'<circle cx="{qx:.2f}" cy="{qy:.2f}" r="{PR.MARK_R}" fill="none" '
                       f'stroke="{col}" stroke-width="{sw}"/>')

    # connectors, white casing under the grey line
    stop = PR.MARK_R + 2        # the line ends at the marker's ring
    for idx in rep_idx:
        a, e, q = route(pts[idx], at[idx])
        u = q - e
        if np.hypot(*u) > stop:
            path = PR.rounded_path([a, e, q - u / np.hypot(*u) * stop], PR.CORNER)
        else:
            v = (q - a) / np.hypot(*(q - a))
            path = PR.rounded_path([a, q - v * stop], PR.CORNER)
        svg.append(f'<path d="{path}" fill="none" stroke="white" stroke-width="5.0" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')
        svg.append(f'<path d="{path}" fill="none" stroke="{PR.LINE_COL}" stroke-width="2.7" '
                   f'stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>')

    # insets, outlined in the depleted lineage's colour, gene name top right
    fs = LABEL_FS
    for i, idx in enumerate(rep_idx):
        x0, y0 = at[idx][0] - PANEL / 2, at[idx][1] - PANEL / 2
        gene = df["gene"].iloc[idx]
        svg.append(f'<image x="{x0:.2f}" y="{y0:.2f}" width="{PANEL:.2f}" '
                   f'height="{PANEL:.2f}" clip-path="url(#clip{i})" '
                   f'preserveAspectRatio="xMidYMid slice" '
                   f'xlink:href="data:image/png;base64,{panel_png[gene]}"/>')
        svg.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{PANEL:.2f}" '
                   f'height="{PANEL:.2f}" rx="9" ry="9" fill="none" '
                   f'stroke="{PR.LINEAGE_COLORS.get(lin[idx], "#d3d6da")}" stroke-width="3.8"/>')
        tx, w = x0 + PANEL - 8, 0.56 * fs * len(gene) + 12
        svg.append(f'<rect x="{tx - w:.2f}" y="{y0 + 5:.2f}" width="{w:.2f}" '
                   f'height="{fs + 8:.2f}" rx="5" ry="5" fill="white" fill-opacity="0.72"/>')
        svg.append(T(gene, tx, y0 + 7 + fs, fs, color=INK, italic=True, anchor="end"))
    svg.append('</svg>')
    return write("ring", "\n".join(svg), W, H)


def key_svg(cache, present, vmax):
    """perturb's key column at this sheet's size: the reference umap, the
    coarse-type legend, the distortion colorbar."""
    ref_png = PR.render_ref_png(cache["ref_xy"], cache["ref_lineage"])
    stops = PR.cmap_hex(PS.DISTORTION_CMAP, 11)
    leg_y = REF_Y + REF + 34
    cb_y0 = leg_y + LEG_STEP * (len(present) + 1) + 20
    KH = cb_y0 + 72
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{KW / PX:.3f}in" height="{KH / PX:.3f}in" viewBox="0 0 {KW:.0f} {KH:.0f}">',
           '<defs><linearGradient id="cbar" x1="0" y1="0" x2="1" y2="0">'
           + "".join(f'<stop offset="{j / (len(stops) - 1):.3f}" stop-color="{hx}"/>'
                     for j, hx in enumerate(stops))
           + '</linearGradient>'
           f'<clipPath id="refclip"><rect x="0" y="{REF_Y:.1f}" width="{REF:.1f}" '
           f'height="{REF:.1f}" rx="9" ry="9"/></clipPath></defs>']

    svg.append(T(f"E{PS.RING_TARGET_TIME:g} reference", 0, REF_Y - 8, 19, color=INK))
    svg.append(f'<image x="0" y="{REF_Y:.1f}" width="{REF:.1f}" height="{REF:.1f}" '
               f'clip-path="url(#refclip)" preserveAspectRatio="xMidYMid slice" '
               f'xlink:href="data:image/png;base64,{ref_png}"/>')
    svg.append(f'<rect x="0" y="{REF_Y:.1f}" width="{REF:.1f}" height="{REF:.1f}" '
               f'rx="9" ry="9" fill="none" stroke="#d3d6da" stroke-width="2.4"/>')

    svg.append(T("most perturbed coarse type", 0, leg_y, 19, color=INK))
    ly = leg_y + LEG_STEP + 4
    for ln in present:
        svg.append(f'<circle cx="7" cy="{ly - 5:.1f}" r="6.2" '
                   f'fill="{PR.LINEAGE_COLORS[ln]}"/>')
        svg.append(T(ln, 22, ly, 18, color=INK))
        ly += LEG_STEP

    svg.append(T("panel hue: log nn distortion", 0, cb_y0, 19, color=INK))
    svg.append(T("distance to knockout / to reference", 0, cb_y0 + 19, 18, color="#777777"))
    cb_w, cb_y, cap = KW - 2, cb_y0 + 30, 14.0
    bar_x, bar_w, mid = cap, cb_w - 2 * cap, cb_y + CBAR_H / 2
    svg.append(f'<rect x="{bar_x:.1f}" y="{cb_y:.1f}" width="{bar_w:.1f}" '
               f'height="{CBAR_H:.1f}" fill="url(#cbar)" stroke="#cccccc" stroke-width="0.8"/>')
    for x_base, x_tip, col in ((bar_x, 0.0, stops[0]), (bar_x + bar_w, cb_w, stops[-1])):
        svg.append(f'<path d="M {x_base:.1f} {cb_y:.1f} L {x_base:.1f} {cb_y + CBAR_H:.1f} '
                   f'L {x_tip:.1f} {mid:.1f} Z" fill="{col}" stroke="#cccccc" '
                   f'stroke-width="0.8" stroke-linejoin="round"/>')
    for frac, lab, anchor in ((0.0, f"< {-vmax:.2f}", "start"), (0.5, "0", "middle"),
                              (1.0, f"> +{vmax:.2f}", "end")):
        svg.append(T(lab, frac * cb_w, cb_y + CBAR_H + 19, 18, color=INK, anchor=anchor))
    svg.append('</svg>')
    return write("refkey", "\n".join(svg), KW, KH)


def render():
    df = pd.read_csv(os.path.join(S.PERTURB_NR, "perturb_umap.csv"))
    with open(os.path.join(S.PERTURB_NR, "perturb_umap_panels.pkl"), "rb") as fh:
        cache = pickle.load(fh)
    # one symmetric colour limit over every inset, as in perturb
    vmax = float(np.percentile(np.abs(np.concatenate(list(cache["panel_val"].values()))),
                               PS.RING_CLIM_Q))
    panel_png = {g: PR.render_panel_png(cache["ref_xy"], v, vmax)
                 for g, v in cache["panel_val"].items()}
    present = [ln for ln in PR.LINEAGE_ORDER if ln in set(df["depleted_lineage"])]
    return ring_svg(df, panel_png), key_svg(cache, present, vmax)


if __name__ == "__main__":
    render()
