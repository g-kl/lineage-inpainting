"""panel a: the perturbation-delta ring UMAP, drawn from the cached results.

centre: the E9.0 HVG perturbation map (one point per knockout, hued by the lineage
it depletes most).  ringing it: representative TF knockouts (from
select_representatives), each a small embryo UMAP hued by the knockout's log nn
distortion (red = the reference cell sits farther from the knockout's leaves than
from the reference cells), tied to its point by a rounded "subway" connector.  the
key (refkey.svg): a cell-level reference UMAP, the lineage legend, and the distortion
colorbar.

reads only numerical_results/{perturb_umap.csv, .npz, perturb_umap_panels.pkl}
(written by compute_ring.py) -- no dataset load.  the embryo panels and the reference
UMAP are both rasterised here from the cached per-cell values (panel_val, ref_xy /
ref_lineage), so the palette and the colour limit are set at plot time.  writes
ring.svg / ring.png + refkey.svg / refkey.png into imgs/ for assemble.py.
"""

import os
import io
import sys
import base64
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from analysis.figures.perturb import style as S
from analysis.figures.perturb.compute_ring import select_representatives
from utils.figure_labels import text_svg_path as T

LINEAGE_COLORS = S.LINEAGE_COLORS
LINEAGE_ORDER = S.LINEAGE_ORDER


## CANVAS GEOMETRY (viewBox px; the SVG is emitted at RING_W_IN inches wide) ##
# ring.svg holds the central map + surrounding TF panels (a ~square block);
# the reference UMAP + lineage legend + distortion colorbar are emitted as a separate
# compact key (refkey.svg) that assemble.py places to the right of the ring.

PAGE_W = S.RING_W_IN
W, H = 985.0, 1060.0
ring_cx, ring_cy = 500.0, 530.0
R = 405.0                  # ring radius (panel centres)
C = 450.0                  # central map square side
PANEL = 150.0             # outer panel side
MARK_R = 7.5              # hollow marker radius around a representative point
PT_R = 2.0               # central-map point radius
CORNER = 15.0            # subway-connector corner radius

INK = S.INK
LINE_COL = S.RING_LINE_COL


## REFERENCE-KEY GEOMETRY (refkey.svg viewBox px; >=6pt text at the emitted width) ##
# one column: reference UMAP, then the legend, then the colorbar; the number of
# legend entries sets the column height.
KW = 360.0               # key column width
REF = 360.0              # reference-UMAP panel side
ref_kx, ref_ky = 0.0, 24.0
legend_kx = 0.0
LEG_STEP = 26.0                        # legend line spacing
CBAR_W, CBAR_H = 340.0, 17.0           # colorbar bar size


## SVG HELPERS ##

def cmap_hex(name, k):
    cmap = plt.get_cmap(name)
    return [mcolors.to_hex(cmap(i / (k - 1))) for i in range(k)]


def render_panel_png(ref_xy, val, vmax, px=620):
    """rasterise one knockout's embryo panel, hued by its log nn distortion, strongest
    on top.  same square bbox as render_ref_png, so panels and key share a frame."""
    lo, hi = np.nanmin(ref_xy, axis=0), np.nanmax(ref_xy, axis=0)
    o = np.argsort(np.abs(val))
    fig = plt.figure(figsize=(px / 200, px / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.axis("off")
    ax.scatter(ref_xy[o, 0], ref_xy[o, 1], c=val[o], cmap=S.DISTORTION_CMAP,
               vmin=-vmax, vmax=vmax, s=3.0, linewidths=0, rasterized=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=200)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_ref_png(ref_xy, ref_lineage, px=620):
    """rasterise the cell-level reference UMAP, hued by coarse type, live from the
    current LINEAGE_COLORS -- so a palette change tracks without recomputing the cache.
    a square figure spanning the data bbox, matching render_panel_png's geometry, so it
    drops cleanly into the square REF panel (preserveAspectRatio slice)."""
    lo, hi = np.nanmin(ref_xy, axis=0), np.nanmax(ref_xy, axis=0)
    cols_hex = np.array([LINEAGE_COLORS.get(l, "#d0d0d0") for l in ref_lineage])
    fig = plt.figure(figsize=(px / 200, px / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.axis("off")
    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], c=cols_hex, s=2.6, linewidths=0,
               rasterized=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=200)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_mapper(data_xy, cx, cy, side, pad=0.06):
    """linear map from a data bbox into a centred square (SVG pixels, y down)."""
    lo, hi = data_xy.min(axis=0), data_xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scale = side * (1 - 2 * pad) / span.max()
    off = np.array([cx, cy]) - 0.5 * scale * (lo + hi) * np.array([1, -1])

    def to_px(p):
        p = np.atleast_2d(np.asarray(p, float))
        out = np.empty_like(p)
        out[:, 0] = off[0] + scale * p[:, 0]
        out[:, 1] = off[1] - scale * p[:, 1]          # flip y for screen coords
        return out
    return to_px


def panel_anchor(px, py, qx, qy, half):
    """point on a panel's square edge (centre px,py; half-side half) toward q."""
    dx, dy = qx - px, qy - py
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return px, py + half
    t = half / max(abs(dx), abs(dy))
    return px + dx * t, py + dy * t


def rounded_path(points, r):
    """SVG path through ``points`` with corners rounded to radius r."""
    pts_ = [np.asarray(p, float) for p in points]
    if len(pts_) < 3:
        return "M {:.2f} {:.2f} L {:.2f} {:.2f}".format(*pts_[0], *pts_[-1])
    d = "M {:.2f} {:.2f}".format(*pts_[0])
    for i in range(1, len(pts_) - 1):
        p0, p1, p2 = pts_[i - 1], pts_[i], pts_[i + 1]
        v1, v2 = p0 - p1, p2 - p1
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        rr = min(r, l1 / 2, l2 / 2)
        a = p1 + v1 / max(l1, 1e-9) * rr
        b = p1 + v2 / max(l2, 1e-9) * rr
        d += " L {:.2f} {:.2f} Q {:.2f} {:.2f} {:.2f} {:.2f}".format(*a, *p1, *b)
    d += " L {:.2f} {:.2f}".format(*pts_[-1])
    return d


def subway(start, end, r=CORNER):
    """rounded 'subway' route: axis-aligned leg, then a 45 degree run to end."""
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    adx, ady = abs(dx), abs(dy)
    corner = (sx + np.sign(dx) * (adx - ady), sy) if adx >= ady \
        else (sx, sy + np.sign(dy) * (ady - adx))
    return rounded_path([start, corner, end], r)


## RING SLOT ASSIGNMENT (even ring, then 2-opt to minimise line crossings) ##

def waypoints(rep, slot, pts):
    """3-point connector polyline (panel anchor -> elbow -> near point)."""
    sx, sy = slot
    qx, qy = pts[rep]
    ax_, ay_ = panel_anchor(sx, sy, qx, qy, PANEL / 2)
    dx, dy = qx - ax_, qy - ay_
    adx, ady = abs(dx), abs(dy)
    corner = ((ax_ + np.sign(dx) * (adx - ady), ay_) if adx >= ady
              else (ax_, ay_ + np.sign(dy) * (ady - adx)))
    return [(ax_, ay_), corner, (qx, qy)]


def seg_cross(a, b, c, d):
    """do segments a-b and c-d properly cross?"""
    cr = lambda o, p, q: (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    return (cr(c, d, a) * cr(c, d, b) < 0) and (cr(a, b, c) * cr(a, b, d) < 0)


def n_crossings(assign, slot_pos, pts):
    polys = [waypoints(assign[k], slot_pos[k], pts) for k in range(len(assign))]
    total = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            for s in range(len(polys[i]) - 1):
                for t in range(len(polys[j]) - 1):
                    total += seg_cross(polys[i][s], polys[i][s + 1],
                                       polys[j][t], polys[j][t + 1])
    return total


def assign_slots(rep_idx, pts, lineages):
    """place reps on an even ring with the reps depleting the same coarse type in
    adjacent slots (like-next-to-like), then a block 2-opt to cut line crossings.

    reps are grouped by depleted coarse type; each group is kept contiguous on the
    ring, so its members always sit next to each other.  the group order (and each
    group's internal order + the ring's rotation) is then optimised to minimise
    connector crossings, never breaking a group apart.
    """
    rep_idx = list(rep_idx)
    n = len(rep_idx)
    bearing = {idx: np.arctan2(pts[idx][1] - ring_cy, pts[idx][0] - ring_cx)
               for idx in rep_idx}
    circmean = lambda vals: np.arctan2(np.mean(np.sin(vals)), np.mean(np.cos(vals)))

    # group reps by depleted coarse type; order groups by mean bearing, members within
    groups = {}
    for idx in rep_idx:
        groups.setdefault(lineages[idx], []).append(idx)
    blocks = [sorted(members, key=lambda i: bearing[i]) for members in groups.values()]
    blocks.sort(key=lambda blk: circmean([bearing[i] for i in blk]))

    # even ring, rotated to best match the reps' bearings
    desired = np.array([bearing[i] for i in rep_idx])
    base = -np.pi + (np.arange(n) + 0.5) * (2 * np.pi / n)
    phi = np.arctan2(np.mean(np.sin(desired - base)), np.mean(np.cos(desired - base)))
    slot_pos = [(ring_cx + R * np.cos(a + phi), ring_cy + R * np.sin(a + phi)) for a in base]

    flatten = lambda seq: [i for blk in seq for i in blk]
    cost = lambda seq: n_crossings(flatten(seq), slot_pos, pts)

    best_seq, best = blocks, cost(blocks)
    improved = True
    while improved:
        improved = False
        for cand in (                                       # moves that keep groups intact
            [best_seq[a:] + best_seq[:a] for a in range(1, len(best_seq))]  # rotate ring
            + [best_seq[:i] + [best_seq[i][::-1]] + best_seq[i + 1:]        # flip a group
               for i in range(len(best_seq)) if len(best_seq[i]) > 1]
            + [[best_seq[j] if k == i else best_seq[i] if k == j else best_seq[k]
                for k in range(len(best_seq))]                              # swap two groups
               for i in range(len(best_seq)) for j in range(i + 1, len(best_seq))]
        ):
            c = cost(cand)
            if c < best:
                best_seq, best, improved = cand, c, True
    seq = flatten(best_seq)
    print(f"connector crossings (like-next-to-like layout): {best}", flush=True)
    return {seq[k]: slot_pos[k] for k in range(n)}


## REFERENCE KEY (reference UMAP + lineage legend + distortion colorbar) ##

def build_refkey(cache, present, vmax):
    """key column (refkey.svg): the E9.0 reference UMAP hued by coarse type, the
    depleted-coarse-type legend under it, and the distortion colorbar below that.  the
    reference UMAP is rasterised live from LINEAGE_COLORS (like render_ref_png)."""
    ref_png = render_ref_png(cache["ref_xy"], cache["ref_lineage"])
    stops = cmap_hex(S.DISTORTION_CMAP, 11)
    legend_ky = ref_ky + REF + 40                       # legend title baseline
    cbar_ky = legend_ky + LEG_STEP * (len(present) + 1) + 24
    KH = cbar_ky + 100
    kw = KW * PAGE_W / W                                # emit at the ring's px scale
    kh = KH * PAGE_W / W
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{kw:.3f}in" height="{kh:.3f}in" viewBox="0 0 {KW:.0f} {KH:.0f}">',
        '<defs><linearGradient id="cbar" x1="0" y1="0" x2="1" y2="0">'
        + "".join(f'<stop offset="{j / (len(stops) - 1):.3f}" stop-color="{hx}"/>'
                  for j, hx in enumerate(stops))
        + '</linearGradient>'
        f'<clipPath id="refclip"><rect x="{ref_kx:.1f}" y="{ref_ky:.1f}" '
        f'width="{REF:.1f}" height="{REF:.1f}" rx="9" ry="9"/></clipPath></defs>',
    ]

    # reference UMAP (top) + title
    svg.append(T(f"E{S.RING_TARGET_TIME:g} reference", ref_kx, ref_ky - 8, 19, color=INK))
    svg.append(
        f'<image x="{ref_kx:.1f}" y="{ref_ky:.1f}" width="{REF:.1f}" height="{REF:.1f}" '
        f'clip-path="url(#refclip)" preserveAspectRatio="xMidYMid slice" '
        f'xlink:href="data:image/png;base64,{ref_png}"/>')
    svg.append(
        f'<rect x="{ref_kx:.1f}" y="{ref_ky:.1f}" width="{REF:.1f}" height="{REF:.1f}" '
        f'rx="9" ry="9" fill="none" stroke="#d3d6da" stroke-width="2.4"/>')

    # coarse-type legend (middle)
    svg.append(T("most perturbed coarse type", legend_kx, legend_ky, 19, color=INK))
    ly = legend_ky + LEG_STEP + 4
    for ln in present:
        svg.append(f'<circle cx="{legend_kx + 7:.1f}" cy="{ly - 5:.1f}" r="6.2" '
                   f'fill="{LINEAGE_COLORS[ln]}"/>')
        svg.append(T(ln, legend_kx + 22, ly, 18, color=INK))
        ly += LEG_STEP

    # distortion colorbar (bottom, spanning the column)
    svg.append(T("panel hue: log nn distortion", ref_kx, cbar_ky, 19, color=INK))
    svg.append(T("distance to knockout / to reference", ref_kx, cbar_ky + 19, 18,
                 color="#777777"))
    cb_w, cb_h = CBAR_W, CBAR_H
    cb_x, cb_y = ref_kx, cbar_ky + 30
    cap = 14.0
    bar_x, bar_w = cb_x + cap, cb_w - 2 * cap
    cb_mid = cb_y + cb_h / 2
    svg.append(f'<rect x="{bar_x:.1f}" y="{cb_y:.1f}" width="{bar_w:.1f}" height="{cb_h:.1f}" '
               f'fill="url(#cbar)" stroke="#cccccc" stroke-width="0.8"/>')
    svg.append(f'<path d="M {bar_x:.1f} {cb_y:.1f} L {bar_x:.1f} {cb_y + cb_h:.1f} '
               f'L {cb_x:.1f} {cb_mid:.1f} Z" fill="{stops[0]}" '
               f'stroke="#cccccc" stroke-width="0.8" stroke-linejoin="round"/>')
    svg.append(f'<path d="M {bar_x + bar_w:.1f} {cb_y:.1f} L {bar_x + bar_w:.1f} {cb_y + cb_h:.1f} '
               f'L {cb_x + cb_w:.1f} {cb_mid:.1f} Z" fill="{stops[-1]}" '
               f'stroke="#cccccc" stroke-width="0.8" stroke-linejoin="round"/>')
    for frac, lab in [(0.0, f"< {-vmax:.2f}"), (0.5, "0"), (1.0, f"> +{vmax:.2f}")]:
        anchor = "start" if frac == 0.0 else ("end" if frac == 1.0 else "middle")
        svg.append(T(lab, cb_x + frac * cb_w, cb_y + cb_h + 19, 18, color=INK, anchor=anchor))

    svg.append('</svg>')
    text = "\n".join(svg)
    path = os.path.join(S.IMGS, "refkey.svg")
    with open(path, "w") as fh:
        fh.write(text)
    print(f"saved {path}", flush=True)
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=text.encode(),
                         write_to=os.path.join(S.IMGS, "refkey.png"),
                         output_width=int(kw * 300))
    except Exception as e:
        print(f"(skipped refkey PNG: {e})", flush=True)


## BUILD ##

def main():
    df = pd.read_csv(os.path.join(S.NUMERICAL, "perturb_umap.csv"))
    with open(os.path.join(S.NUMERICAL, "perturb_umap_panels.pkl"), "rb") as fh:
        cache = pickle.load(fh)
    panel_val = cache["panel_val"]
    xy = df[["umap1", "umap2"]].to_numpy(float)
    print(f"{len(df)} perturbations", flush=True)

    # one symmetric colour limit over every panel, so a hue reads the same on all of them
    vmax = float(np.percentile(np.abs(np.concatenate(list(panel_val.values()))),
                               S.RING_CLIM_Q))
    panel_png = {g: render_panel_png(cache["ref_xy"], v, vmax)
                 for g, v in panel_val.items()}
    print(f"nn distortion colour limit +-{vmax:.3f} (q{S.RING_CLIM_Q} of |value|), "
          f"draw n={cache['draw_n']:,}", flush=True)

    rep_idx, _ = select_representatives(df)
    to_px = make_mapper(xy, ring_cx, ring_cy, C)
    pts = to_px(xy)
    slots = assign_slots(rep_idx, pts, df["depleted_lineage"].to_numpy())

    ph = H / W * PAGE_W
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{PAGE_W}in" height="{ph:.3f}in" '
        f'viewBox="0 0 {W:.0f} {H:.0f}">',
        '<defs>',
    ]
    for i, idx in enumerate(rep_idx):
        px, py = slots[idx]
        x0, y0 = px - PANEL / 2, py - PANEL / 2
        svg.append(
            f'<clipPath id="clip{i}"><rect x="{x0:.2f}" y="{y0:.2f}" '
            f'width="{PANEL:.2f}" height="{PANEL:.2f}" rx="9" ry="9"/></clipPath>')
    svg.append('</defs>')
    svg.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>')

    # --- central map: all points, hued by depleted lineage
    lin = df["depleted_lineage"].to_numpy()
    svg.append('<g>')
    for (x, y), ln in zip(pts, lin):
        col = LINEAGE_COLORS.get(ln, "#b0b0b0")
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{PT_R}" fill="{col}" '
                   f'fill-opacity="0.85"/>')
    svg.append('</g>')

    # --- representative markers (hollow ring with white halo)
    for idx in rep_idx:
        qx, qy = pts[idx]
        svg.append(f'<circle cx="{qx:.2f}" cy="{qy:.2f}" r="{MARK_R}" fill="none" '
                   f'stroke="white" stroke-width="3.2"/>')
        svg.append(f'<circle cx="{qx:.2f}" cy="{qy:.2f}" r="{MARK_R}" fill="none" '
                   f'stroke="{INK}" stroke-width="1.5"/>')

    # --- connectors (white casing under a grey subway line; over the map and markers).
    # the depleted lineage is instead carried by each inset panel's outline colour.
    for idx in rep_idx:
        px, py = slots[idx]
        qx, qy = pts[idx]
        ax_, ay_ = panel_anchor(px, py, qx, qy, PANEL / 2)
        d = np.array([qx - ax_, qy - ay_], float)
        dn = d / max(np.linalg.norm(d), 1e-9)
        end = (qx - dn[0] * (MARK_R + 2), qy - dn[1] * (MARK_R + 2))
        path = subway((ax_, ay_), end)
        svg.append(f'<path d="{path}" fill="none" stroke="white" stroke-width="5.0" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')
        svg.append(f'<path d="{path}" fill="none" stroke="{LINE_COL}" '
                   f'stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round" '
                   f'opacity="0.95"/>')

    # --- ring panels (rasterised embryo UMAPs) + TF labels inside, top-right
    fs = 17.5                                    # ~6pt at the emitted ring width
    for i, idx in enumerate(rep_idx):
        px, py = slots[idx]
        x0, y0 = px - PANEL / 2, py - PANEL / 2
        gene = df["gene"].iloc[idx]
        svg.append(
            f'<image x="{x0:.2f}" y="{y0:.2f}" width="{PANEL:.2f}" height="{PANEL:.2f}" '
            f'clip-path="url(#clip{i})" preserveAspectRatio="xMidYMid slice" '
            f'xlink:href="data:image/png;base64,{panel_png[gene]}"/>')
        border_col = LINEAGE_COLORS.get(lin[idx], "#d3d6da")
        svg.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{PANEL:.2f}" height="{PANEL:.2f}" '
            f'rx="9" ry="9" fill="none" stroke="{border_col}" stroke-width="3.8"/>')
        tx, ty = x0 + PANEL - 8, y0 + 7 + fs
        w = 0.56 * fs * len(gene) + 12
        svg.append(
            f'<rect x="{tx - w:.2f}" y="{y0 + 5:.2f}" width="{w:.2f}" height="{fs + 8:.2f}" '
            f'rx="5" ry="5" fill="white" fill-opacity="0.72"/>')
        svg.append(T(gene, tx, ty, fs, color=INK, italic=True, anchor="end"))

    svg.append('</svg>')
    svg_text = "\n".join(svg)

    svg_path = os.path.join(S.IMGS, "ring.svg")
    with open(svg_path, "w") as fh:
        fh.write(svg_text)
    print(f"saved {svg_path}", flush=True)

    present = [ln for ln in LINEAGE_ORDER if ln in set(lin)]
    build_refkey(cache, present, vmax)

    try:
        import cairosvg
        png_path = os.path.join(S.IMGS, "ring.png")
        cairosvg.svg2png(bytestring=svg_text.encode(), write_to=png_path,
                         output_width=int(PAGE_W * 300))
        print(f"saved {png_path}", flush=True)
    except Exception as e:
        print(f"(skipped PNG preview: {e})", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
