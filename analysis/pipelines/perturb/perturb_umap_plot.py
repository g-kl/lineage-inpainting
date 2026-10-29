#!/usr/bin/env python
#SBATCH --job-name=perturb_umap_plot
#SBATCH --output=logs/perturb_umap_plot_%j.log
#SBATCH --error=logs/perturb_umap_plot_%j.log
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
"""publication-style figure panel for the perturbation-delta UMAP.

centre: the perturbation map from ``perturb_umap_compute.py`` -- one point per
knockout, embedded on its scVI mean-delta and hued by the *lineage* it depletes
most.  ringing it: representative *transcription-factor* knockouts -- for every
lineage depleted-most by at least ``min_perts`` TFs (~6 lineages), the top
``reps_per_lineage`` TFs that deplete it most *specifically* (largest margin
between their top and runner-up lineage depletion), trimmed/padded to
``n_outer`` panels.  every outer panel is a small embryo UMAP hued by the log
density ratio of the knockout's landed cells vs control (red = enriched, blue =
depleted).  a bold rounded "subway" line ties each panel to its point.  right
column: a reference embryo UMAP hued by lineage, the lineage legend, and the
density colorbar.

the whole figure is composed as an SVG document (HTML-like declarative layout,
native .svg output, Computer Modern Sans embedded so it is self-contained and
fits in a one-column figure).  the central map, markers, connectors, legend and
colorbar are vector; the dense embryo panels are rasterised once and embedded.
all real computation -- embedding, selection, differential abundance -- happens
in Python; the SVG is just the layout surface.

reads analysis/results/perturb_umap/{perturb_umap.csv,perturb_umap.npz}, the
screen pickles those point to, and literature/tf_list.txt; writes
analysis/results/perturb_umap/perturb_umap.svg (+ a .png preview).
"""

import os
import sys

base_dir = os.path.abspath(os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import base64
import io
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.linear_model import LogisticRegression

from datasets.mouse_embryo import MouseEmbryoDataset
from analysis.pipelines.perturb.perturb_umap_lib import (
    LINEAGE_COLORS, LINEAGE_ORDER, leaf_rows, load_gene, load_control,
    load_cell_categories)


## CONFIG ##

out_dir = os.path.join(base_dir, "analysis", "results", "perturb_umap")
tf_list_path = os.path.join(base_dir, "literature", "tf_list.txt")
n_outer = 12               # number of representative TF embryo panels
reps_per_lineage = 2       # top-N most lineage-specific TFs picked per lineage
min_perts = 20             # a lineage qualifies if >= this many TFs most deplete it
n_ref = 40000              # reference cells drawn for each embryo panel
vmax = 5.0                 # fixed density-ratio colour scale (panels comparable)
accent_cmap = "coolwarm"
seed = 0

# --- canvas geometry (viewBox px; the SVG is emitted at PAGE_W inches wide).
PAGE_W = 5.8               # output width (inches) -- fits one figure column
W, H = 1230.0, 1060.0
ring_cx, ring_cy = 500.0, 530.0
R = 405.0                  # ring radius (panel centres)
C = 450.0                  # central map square side
PANEL = 150.0             # outer panel side
MARK_R = 5.0              # hollow marker radius around a representative point
PT_R = 2.0               # central-map point radius
CORNER = 15.0            # subway-connector corner radius

# --- right-hand column: reference UMAP (top), legend (middle), colorbar (bottom)
col_x = 1000.0           # left edge of the right column
REF = 186.0              # reference-UMAP panel side
ref_x0, ref_y0 = col_x, 270.0
legend_y0 = 490.0        # baseline of the lineage-legend title
cbar_y0 = 734.0          # baseline of the colorbar title

FONT = "CMSS, 'CMU Sans Serif', 'Latin Modern Sans', sans-serif"
INK = "#222222"
LINE_COL = "#2b2f33"


## LOAD RESULTS ##

df = pd.read_csv(os.path.join(out_dir, "perturb_umap.csv"))
npz = np.load(os.path.join(out_dir, "perturb_umap.npz"), allow_pickle=True)
screen_dir = str(npz["screen_dir"])
data_dir = str(npz["data_dir"])
target_time = float(npz["target_time"])
tol = float(npz["tol"])
genes_dir = os.path.join(screen_dir, "genes")
control_dir = os.path.join(screen_dir, "control")
xy = df[["umap1", "umap2"]].to_numpy(float)
print(f"{len(df)} perturbations; screen {screen_dir}", flush=True)


## REPRESENTATIVES = TOP-2 MOST LINEAGE-SPECIFIC TFs PER POPULATED LINEAGE ##

with open(tf_list_path) as fh:
    tf_set = {ln.strip() for ln in fh if ln.strip()}

# specificity of a knockout = how much more it depletes its top lineage than the
# runner-up (the margin between the two largest lineage depletions): a large
# margin means the knockout hits that lineage selectively, not broadly.
lineage_cols = [f"d_{ln}" for ln in LINEAGE_ORDER]
depl = -df[lineage_cols].to_numpy()                      # positive = depleted
top2 = np.sort(depl, axis=1)[:, -2:]
df["specificity"] = top2[:, 1] - top2[:, 0]

tf = df[df["gene"].isin(tf_set)]
counts = tf["depleted_lineage"].value_counts()
eligible = [ln for ln in counts.index if counts[ln] >= min_perts]   # ~6 lineages

# top reps_per_lineage most-specific TFs for each eligible lineage, then trim or
# extend (by global specificity) to exactly n_outer panels.
picks = pd.concat([tf[tf["depleted_lineage"] == ln].nlargest(reps_per_lineage, "specificity")
                   for ln in eligible])
if len(picks) > n_outer:
    picks = picks.nlargest(n_outer, "specificity")
elif len(picks) < n_outer:
    extra = tf[~tf.index.isin(picks.index)].nlargest(n_outer - len(picks), "specificity")
    picks = pd.concat([picks, extra])

rep_idx = picks.index.to_numpy()
print(f"{len(tf)} TFs; {len(eligible)} eligible lineages ({', '.join(eligible)})",
      flush=True)
print("selected representatives: "
      + ", ".join(f"{df['gene'].iloc[i]} [{df['depleted_lineage'].iloc[i]}]"
                  for i in rep_idx), flush=True)


## RASTERISED EMBRYO PANELS (cached -- the only step that needs the 55GB h5ad) ##

# the differential-abundance panels + lineage reference are expensive (load the
# scVI states, fit a classifier per TF, rasterise).  cache the rendered PNGs
# keyed by gene so cosmetic re-runs of the layout skip the dataset entirely;
# delete perturb_umap_panels.pkl (or set reuse_panels=False) to recompute.
import pickle
reuse_panels = True
cache_path = os.path.join(out_dir, "perturb_umap_panels.pkl")
rep_genes = list(df["gene"].iloc[rep_idx])

cache = {}
if reuse_panels and os.path.exists(cache_path):
    with open(cache_path, "rb") as fh:
        cache = pickle.load(fh)

if cache.get("ref_png") and set(rep_genes) <= set(cache.get("panel_png", {})):
    print("reusing cached panels (skipping dataset load)", flush=True)
    panel_png = {g: cache["panel_png"][g] for g in rep_genes}
    ref_png = cache["ref_png"]
else:
    print("loading dataset ...", flush=True)
    dataset = MouseEmbryoDataset()
    states = dataset.states
    scvi = np.asarray(states.obsm["X_scvi"], dtype=np.float32)
    name_to_row = {n: i for i, n in enumerate(states.obs_names)}

    rng = np.random.default_rng(seed)
    ref = np.where(np.isclose(states.obs["time"].astype(float), target_time))[0]
    if len(ref) > n_ref:
        ref = rng.choice(ref, n_ref, replace=False)
    ref_names = states.obs_names[ref]
    ref_xy = dataset.umap_coords.reindex(ref_names)[["UMAP1", "UMAP2"]].to_numpy(float)
    ref_latent = scvi[ref]

    _, name_to_lineage, _ = load_cell_categories(data_dir)   # reference-cell lineage
    ref_lineage = name_to_lineage.reindex(ref_names).to_numpy()

    control = load_control(control_dir)
    ctrl_latent = scvi[leaf_rows(control, target_time, tol, name_to_row)]

    def log_density_ratio(gene):
        """logit(perturb vs control) - log(n_pos/n_neg) over the reference cells."""
        pos = scvi[leaf_rows(load_gene(genes_dir, gene), target_time, tol, name_to_row)]
        X = np.vstack([pos, ctrl_latent])
        y = np.r_[np.ones(len(pos)), np.zeros(len(ctrl_latent))]
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        return clf.decision_function(ref_latent) - np.log(len(pos) / len(ctrl_latent))

    xlim = (np.nanmin(ref_xy[:, 0]), np.nanmax(ref_xy[:, 0]))
    ylim = (np.nanmin(ref_xy[:, 1]), np.nanmax(ref_xy[:, 1]))

    def blank_axes(px):
        fig = plt.figure(figsize=(px / 200, px / 200), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
        return fig, ax

    def png_b64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=200)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    print("computing differential abundance + rendering panels ...", flush=True)
    panel_png = {}
    for gene in rep_genes:
        ratio = log_density_ratio(gene)
        o = np.argsort(np.abs(ratio))             # strongest signal drawn on top
        fig, ax = blank_axes(620)
        ax.scatter(ref_xy[o, 0], ref_xy[o, 1], c=ratio[o], cmap=accent_cmap,
                   vmin=-vmax, vmax=vmax, s=3.0, linewidths=0, rasterized=True)
        panel_png[gene] = png_b64(fig)

    cols = np.array([LINEAGE_COLORS.get(l, "#d0d0d0") for l in ref_lineage])
    fig, ax = blank_axes(620)
    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], c=cols, s=2.6, linewidths=0,
               rasterized=True)
    ref_png = png_b64(fig)

    with open(cache_path, "wb") as fh:
        pickle.dump({"panel_png": panel_png, "ref_png": ref_png}, fh)
    print(f"cached panels to {cache_path}", flush=True)


## SCREEN -> PIXEL COORDINATE MAP (central map) ##

def make_mapper(data_xy, cx, cy, side, pad=0.06):
    """linear map from a data bbox into a centred square (SVG pixels, y down)."""
    lo = data_xy.min(axis=0)
    hi = data_xy.max(axis=0)
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


to_px = make_mapper(xy, ring_cx, ring_cy, C)
pts = to_px(xy)


## RING SLOT ASSIGNMENT (even ring, then 2-opt to minimise line crossings) ##

n = len(rep_idx)
rep_pts = pts[rep_idx]

# even ring, rotated so the grid's phase best matches the points' bearings.
desired = np.arctan2(rep_pts[:, 1] - ring_cy, rep_pts[:, 0] - ring_cx)
base = -np.pi + (np.arange(n) + 0.5) * (2 * np.pi / n)
phi = np.arctan2(np.mean(np.sin(desired - base)), np.mean(np.cos(desired - base)))
slot_pos = [(ring_cx + R * np.cos(a + phi), ring_cy + R * np.sin(a + phi))
            for a in base]


def _anchor(sx, sy, qx, qy, half=PANEL / 2):
    dx, dy = qx - sx, qy - sy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return sx, sy + half
    t = half / max(abs(dx), abs(dy))
    return sx + dx * t, sy + dy * t


def _waypoints(rep, slot):
    """3-point connector polyline (panel anchor -> elbow -> near point)."""
    sx, sy = slot
    qx, qy = pts[rep]
    ax_, ay_ = _anchor(sx, sy, qx, qy)
    dx, dy = qx - ax_, qy - ay_
    adx, ady = abs(dx), abs(dy)
    corner = ((ax_ + np.sign(dx) * (adx - ady), ay_) if adx >= ady
              else (ax_, ay_ + np.sign(dy) * (ady - adx)))
    return [(ax_, ay_), corner, (qx, qy)]


def _seg_x(a, b, c, d):
    """do segments a-b and c-d properly cross?"""
    def cr(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1, d2 = cr(c, d, a), cr(c, d, b)
    d3, d4 = cr(a, b, c), cr(a, b, d)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _crossings(assign):
    polys = [_waypoints(assign[k], slot_pos[k]) for k in range(n)]
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            for s in range(len(polys[i]) - 1):
                for t in range(len(polys[j]) - 1):
                    total += _seg_x(polys[i][s], polys[i][s + 1],
                                    polys[j][t], polys[j][t + 1])
    return total


# init by angular order, then greedy 2-opt swaps until no crossing is removed
assign = list(rep_idx[np.argsort(desired)])
best = _crossings(assign)
improved = True
while improved:
    improved = False
    for i in range(n):
        for j in range(i + 1, n):
            assign[i], assign[j] = assign[j], assign[i]
            c = _crossings(assign)
            if c < best:
                best, improved = c, True
            else:
                assign[i], assign[j] = assign[j], assign[i]
print(f"connector crossings after 2-opt: {best}", flush=True)
slots = {assign[k]: slot_pos[k] for k in range(n)}


## SVG HELPERS ##

def esc(s):
    return _xml_escape(str(s))


def cmap_hex(name, k):
    cmap = plt.get_cmap(name)
    return [mcolors.to_hex(cmap(i / (k - 1))) for i in range(k)]


def panel_anchor(px, py, qx, qy, half):
    """point on a panel's square edge (centre px,py; half-side half) toward q."""
    dx, dy = qx - px, qy - py
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return px, py + half
    t = half / max(abs(dx), abs(dy))
    return px + dx * t, py + dy * t


def rounded_path(points, r):
    """SVG path string through ``points`` with corners rounded to radius r."""
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
    if adx >= ady:
        corner = (sx + np.sign(dx) * (adx - ady), sy)
    else:
        corner = (sx, sy + np.sign(dy) * (ady - adx))
    return rounded_path([start, corner, end], r)


## BUILD SVG ##

with open(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "cmss10.ttf"), "rb") as fh:
    _font_b64 = base64.b64encode(fh.read()).decode("ascii")

ph = H / W * PAGE_W
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'xmlns:xlink="http://www.w3.org/1999/xlink" '
    f'width="{PAGE_W}in" height="{ph:.3f}in" '
    f'viewBox="0 0 {W:.0f} {H:.0f}" font-family="{FONT}">',
    '<defs>',
    f"<style>@font-face{{font-family:'CMSS';font-style:normal;"
    f"src:url(data:font/ttf;base64,{_font_b64}) format('truetype');}}</style>",
]
for i, idx in enumerate(rep_idx):
    px, py = slots[idx]
    x0, y0 = px - PANEL / 2, py - PANEL / 2
    svg.append(
        f'<clipPath id="clip{i}"><rect x="{x0:.2f}" y="{y0:.2f}" '
        f'width="{PANEL:.2f}" height="{PANEL:.2f}" rx="9" ry="9"/></clipPath>')
svg.append(
    f'<clipPath id="refclip"><rect x="{ref_x0:.2f}" y="{ref_y0:.2f}" '
    f'width="{REF:.2f}" height="{REF:.2f}" rx="9" ry="9"/></clipPath>')
svg.append('</defs>')
svg.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>')

# --- connectors (white casing under a bold ink line; sit beneath markers)
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

# --- central map: all points, hued by depleted lineage
svg.append('<g>')
lin = df["depleted_lineage"].to_numpy()
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

# --- ring panels (rasterised embryo UMAPs) + TF labels inside, top-right
fs = 15.0
for i, idx in enumerate(rep_idx):
    px, py = slots[idx]
    x0, y0 = px - PANEL / 2, py - PANEL / 2
    gene = df["gene"].iloc[idx]
    svg.append(
        f'<image x="{x0:.2f}" y="{y0:.2f}" width="{PANEL:.2f}" height="{PANEL:.2f}" '
        f'clip-path="url(#clip{i})" preserveAspectRatio="xMidYMid slice" '
        f'xlink:href="data:image/png;base64,{panel_png[gene]}"/>')
    svg.append(
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{PANEL:.2f}" height="{PANEL:.2f}" '
        f'rx="9" ry="9" fill="none" stroke="#d3d6da" stroke-width="1.1"/>')
    # label inside top-right, on a soft white pill so it reads over any points
    tx, ty = x0 + PANEL - 8, y0 + 7 + fs
    w = 0.56 * fs * len(gene) + 12
    svg.append(
        f'<rect x="{tx - w:.2f}" y="{y0 + 5:.2f}" width="{w:.2f}" height="{fs + 8:.2f}" '
        f'rx="5" ry="5" fill="white" fill-opacity="0.72"/>')
    svg.append(
        f'<text x="{tx:.2f}" y="{ty:.2f}" text-anchor="end" font-size="{fs:g}" '
        f'font-style="italic" fill="{INK}">{esc(gene)}</text>')

# =============================================================================
# RIGHT-HAND COLUMN: reference UMAP (top), lineage legend (middle), colorbar
# =============================================================================

# --- reference embryo UMAP, hued by lineage
svg.append(
    f'<text x="{ref_x0:.2f}" y="{ref_y0 - 8:.2f}" font-size="14" fill="{INK}">'
    f'E9.0 UMAP (hued by lineage)</text>')
svg.append(
    f'<image x="{ref_x0:.2f}" y="{ref_y0:.2f}" width="{REF:.2f}" height="{REF:.2f}" '
    f'clip-path="url(#refclip)" preserveAspectRatio="xMidYMid slice" '
    f'xlink:href="data:image/png;base64,{ref_png}"/>')
svg.append(
    f'<rect x="{ref_x0:.2f}" y="{ref_y0:.2f}" width="{REF:.2f}" height="{REF:.2f}" '
    f'rx="8" ry="8" fill="none" stroke="#d3d6da" stroke-width="1.1"/>')

# --- lineage legend; only lineages present in the map
present = [ln for ln in LINEAGE_ORDER if ln in set(lin)]
svg.append(
    f'<text x="{col_x:.1f}" y="{legend_y0:.1f}" font-size="14" fill="{INK}">'
    f'depleted lineage</text>')
ly = legend_y0 + 23
for ln in present:
    svg.append(f'<circle cx="{col_x + 6:.1f}" cy="{ly - 4.5:.1f}" r="5.3" '
               f'fill="{LINEAGE_COLORS[ln]}"/>')
    svg.append(f'<text x="{col_x + 19:.1f}" y="{ly:.1f}" font-size="12.5" '
               f'fill="{INK}">{esc(ln)}</text>')
    ly += 20.5

# --- density-ratio colorbar
stops = cmap_hex(accent_cmap, 11)
svg.insert(1, '<defs><linearGradient id="cbar" x1="0" y1="0" x2="1" y2="0">'
           + "".join(f'<stop offset="{j / (len(stops) - 1):.3f}" stop-color="{hx}"/>'
                     for j, hx in enumerate(stops))
           + '</linearGradient></defs>')
cb_w, cb_h = 186.0, 12.0
cb_x, cb_y = col_x, cbar_y0 + 22
svg.append(f'<text x="{col_x:.1f}" y="{cbar_y0:.1f}" font-size="14" fill="{INK}">'
           f'panel hue: log density ratio</text>')
svg.append(f'<text x="{col_x:.1f}" y="{cbar_y0 + 16:.1f}" font-size="12" fill="#777">'
           f'knockout / control</text>')
svg.append(f'<rect x="{cb_x:.1f}" y="{cb_y:.1f}" width="{cb_w:.1f}" height="{cb_h:.1f}" '
           f'fill="url(#cbar)" stroke="#cccccc" stroke-width="0.8"/>')
for frac, lab in [(0.0, f"-{vmax:g} (depleted)"), (0.5, "0"), (1.0, f"+{vmax:g} (enriched)")]:
    anchor = "start" if frac == 0.0 else ("end" if frac == 1.0 else "middle")
    svg.append(f'<text x="{cb_x + frac * cb_w:.1f}" y="{cb_y + cb_h + 14:.1f}" '
               f'text-anchor="{anchor}" font-size="11.5" fill="{INK}">{lab}</text>')

svg.append('</svg>')
svg_text = "\n".join(svg)

svg_path = os.path.join(out_dir, "perturb_umap.svg")
with open(svg_path, "w") as fh:
    fh.write(svg_text)
print(f"saved {svg_path}", flush=True)

# high-res PNG preview
try:
    import cairosvg
    png_path = os.path.join(out_dir, "perturb_umap.png")
    cairosvg.svg2png(bytestring=svg_text.encode(), write_to=png_path,
                     output_width=int(PAGE_W * 300))
    print(f"saved {png_path}", flush=True)
except Exception as e:
    print(f"(skipped PNG preview: {e})", flush=True)
