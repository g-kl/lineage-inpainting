"""Fate-bias context-scaling figure, in the shared crossing-family grammar.

Mirrors the reconstructed-dynamics layout of ``synthetic_evals.recon_figure`` so
this benchmark reads as the same figure family:

  reconstruction row    [dynamics (truth) | pool-split n=1 | pool-split n=8],
                        state-vs-time with time running DOWNWARD (t_in at top,
                        t_out at bottom). Each method panel draws the predicted
                        leaf states as interpolants from the cell-of-interest
                        anchor; the true descendant lineages are ghosted. More
                        context (n) sharpens the prediction toward the truth.
  metrics strip         one panel per sigma: energy distance (log-y) vs context
                        size n. The pool-split model is a line over n=1..8;
                        the n-independent baselines (FM / GENOT) are horizontal
                        reference lines. Shared method legend along the bottom.

Style (palette, type, metric) is imported from ``synthetic_evals.viz``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from synthetic_evals import viz

# --------------------------------------------------------------------------
#  CONFIG (layout knobs -- shared palette/type/metric live in synthetic_evals.viz)
# --------------------------------------------------------------------------

RECON_H = 1.1          # height (in) of the reconstruction row (trajectory panels)
METRIC_H = 1.1         # height (in) of the metrics strip
RECON_WIDTH = 2.1      # width (in) per reconstruction panel (matched to crossing total width)
METRIC_WIDTH = 1.7     # width (in) per sigma panel (matched to crossing total width)
WSPACE = 0.16
METRIC_WSPACE = 0.18
HSPACE = 0.42

TRUTH_LW = 0.9
GHOST_ALPHA = 0.22     # ghosted truth in the method panels
BG_ALPHA = 0.10        # wispy population background (kept faint so foreground reads)
INTERP_LW = 1.3
PT_SIZE = 32.0         # prediction dots
ANCHOR_SIZE = 42.0     # pooled context cells at t_in (countable)
TARGET_SIZE = 52.0     # true leaf targets at t_out (open rings)
STAR_SIZE = 110.0      # starting cell
ANCESTOR_SIZE = 46.0   # clone common ancestor (diamond)
MARKER_SIZE = 4.5
LINE_LW = 1.4


POP_ALPHA_HI = 0.46    # population alpha at the earliest shown time (few cells)
POP_ALPHA_LO = 0.16    # ... fading to here at t_out (dense committed cloud)
T_LO = 0.0             # all reconstruction panels start at the root time
LATE_FRAC = 0.42       # axis fraction given to the focused t_in -> t_out window


def _fate_color(state):
    """Fate colour from a leaf state's sign (+ up, - down)."""
    return viz.fate_color(1.0 if float(state) >= 0 else -1.0)


def _blend(c1, c2, f):
    """Linear RGB blend; f=0 -> c1, f=1 -> c2."""
    a = np.asarray(mcolors.to_rgb(c1))
    b = np.asarray(mcolors.to_rgb(c2))
    return tuple((1.0 - f) * a + f * b)


def _division_color(mu, t_mid, *, split_t, split_gen, max_gen, bi):
    """Population segment colour. Neutral grey while fate is undecided
    (t < split); at the split the fate is fixed to its red/blue, and each
    subsequent division lightens the shade one step so generation depth -- how
    many descendants the cell now has -- is legible."""
    if t_mid < split_t:
        return viz.PRESPLIT
    g = int(np.floor(t_mid / bi))
    depth = max(0, g - split_gen)
    frac = min(1.0, depth / max(1, max_gen - split_gen))
    return _blend(viz.fate_color(mu), "#ffffff", 0.52 * frac)


def _warp(t_lo, t_in, t_out, late):
    """Piecewise-linear time warp: [t_lo, t_in] -> [0, 1-late], [t_in, t_out] ->
    [1-late, 1], so the t_in->t_out window fills ``late`` of the axis."""
    s1 = (1.0 - late) / (t_in - t_lo)
    s2 = late / (t_out - t_in)
    ub = 1.0 - late

    def fwd(t):
        t = np.asarray(t, dtype=float)
        return np.where(t <= t_in, (t - t_lo) * s1, ub + (t - t_in) * s2)

    def inv(u):
        u = np.asarray(u, dtype=float)
        return np.where(u <= ub, t_lo + u / s1, t_in + (u - ub) / s2)

    return fwd, inv

# Method order for the metric strip / legend.
BASELINE_METHODS = (("fm", "fm"), ("gs", "gs"), ("gf", "gf"))


# --------------------------------------------------------------------------
#  Metric aggregation
# --------------------------------------------------------------------------


def _model_scaling(splitter_rows, sigma, key="ed"):
    """(ns, means, sems) of pool-split energy distance (or ``key``) vs context size n."""
    ns = sorted({int(r["clone_size"]) for r in splitter_rows})
    means, sems = [], []
    for n in ns:
        eds = np.array([r[key] for r in splitter_rows
                        if r["sigma"] == sigma and int(r["clone_size"]) == n],
                       dtype=np.float64)
        means.append(float(eds.mean()) if eds.size else np.nan)
        sems.append(viz.sem(eds))
    return ns, np.asarray(means), np.asarray(sems)


def _baseline_level(baseline_rows, sigma, method, key="ed"):
    """(mean, sem) of an n-independent baseline's energy distance (or ``key``)."""
    eds = np.array([r[key] for r in baseline_rows
                    if r["sigma"] == sigma and r["method"] == method],
                   dtype=np.float64)
    if not eds.size:
        return np.nan, np.nan
    return float(eds.mean()), viz.sem(eds)


# --------------------------------------------------------------------------
#  Reconstruction row (downward time)
# --------------------------------------------------------------------------


def _clip_traj(traj, t_lo, t_hi):
    t = traj[:, 0]
    m = (t >= t_lo - 1e-9) & (t <= t_hi + 1e-9)
    return traj[m]


def _warped_axis(ax, t_in, t_out, ticks, xlim):
    """state on x, warped time on y running downward. Every panel shares the
    same warp (t_lo=T_LO, focused on the t_in->t_out window) so they are
    consistent -- no per-panel time clipping."""
    fwd, inv = _warp(T_LO, t_in, t_out, LATE_FRAC)
    ax.set_yscale("function", functions=(fwd, inv))
    # headroom so the t_out leaf markers clear the bottom spine (and a touch at
    # the top) -- markers sitting exactly on an axis get clipped oddly.
    ax.set_ylim(t_out + 0.16 * (t_out - t_in), T_LO - 0.04 * (t_in - T_LO))
    ax.set_yticks(ticks)
    ax.set_yticklabels([fr"$t={int(t)}$" for t in ticks])
    ax.set_xlim(*xlim)


def _generation_bands(ax, t_out, bi):
    """Alternating light-orange bands, one per generation (cf. the crossing
    figures' branch-band language), so each division reads as a distinct stripe.
    Drawn behind everything."""
    g = 0
    while g * bi < t_out - 1e-9:
        if g % 2 == 0:
            ax.axhspan(g * bi, min((g + 1) * bi, t_out),
                       color=viz.EVENT_TINT, zorder=0)
        g += 1


def _draw_lineage(ax, lineage, *, color, alpha, lw, zorder):
    """Draw the clone family tree: every subtree segment from the common
    ancestor down to the t_in cousins (1 -> 2 -> 4 -> 8)."""
    for seg in lineage["segments"]:
        seg = np.asarray(seg)
        ax.plot(seg[:, 1], seg[:, 0], color=color, alpha=alpha, lw=lw,
                solid_capstyle="round", rasterized=True, zorder=zorder)


def _ancestor_dot(ax, lineage, zorder=10):
    """Mark the clone's common ancestor (a single cell) with a diamond."""
    ax.scatter([lineage["ancestor_state"]], [lineage["root_t"]], marker="D",
               s=ANCESTOR_SIZE, color=viz.TRUTH, edgecolors="white",
               linewidths=0.7, zorder=zorder)


def _fate_targets(ax, tgt, t_out, *, zorder):
    """True leaf targets as open rings, coloured by committed fate. Each ring is
    backed by a white halo (drawn under ALL rings) so it lifts cleanly off the
    trajectory cloud; the rings themselves are unfilled so that close/overlapping
    leaves cross cleanly instead of one ring's fill clipping its neighbour."""
    tgt = np.asarray(tgt, dtype=float).reshape(-1)
    ax.scatter(tgt, np.full(tgt.shape, t_out), s=TARGET_SIZE * 1.9,
               color="white", edgecolors="none", zorder=zorder - 0.1)
    for s in tgt:
        ax.scatter([s], [t_out], s=TARGET_SIZE, facecolors="none",
                   edgecolors=_fate_color(s), linewidths=1.6, zorder=zorder)


def _truth_panel(ax, cartoon, case, xlim):
    """Full fate dynamics: the whole population coloured by committed fate, the
    split + commitment marked, and the starting cell's own fate-coloured
    descendant lineages overlaid. This is where the two fate outcomes -- and how
    early they commit -- are legible."""
    t_out = float(cartoon["leaf_time"])
    t_in = float(cartoon["time_input"])
    split_t = float(cartoon["split_time"])
    commit_t = float(cartoon["commit_time"])
    split_gen = int(cartoon["split_gen"])
    bi = float(cartoon["branch_interval"])
    max_gen = int(round(t_out / bi))

    _generation_bands(ax, t_out, bi)

    # Colour each one-generation segment: neutral grey while fate is undecided
    # (t < split); fixed red/blue at the split; one shade lighter per division
    # afterwards so generation depth (number of descendants) is legible. Alpha is
    # high early (few cells -- divisions read) and fades later.
    span = max(t_out - T_LO, 1e-6)
    for traj, mu in cartoon["trajectories"]:
        seg = _clip_traj(np.asarray(traj), T_LO, t_out)
        if seg.shape[0] < 2:
            continue
        t_mid = float(seg[:, 0].mean())
        color = _division_color(mu, t_mid, split_t=split_t, split_gen=split_gen,
                                max_gen=max_gen, bi=bi)
        a = POP_ALPHA_HI - (POP_ALPHA_HI - POP_ALPHA_LO) * \
            float(np.clip((t_mid - T_LO) / span, 0.0, 1.0))
        ax.plot(seg[:, 1], seg[:, 0], color=color, alpha=a, lw=0.3,
                solid_capstyle="round", rasterized=True, zorder=1)

    ytrans = ax.get_yaxis_transform()  # x in axes-fraction, y in data
    for tg, lab in ((split_t, "commit"), (commit_t, "diverge"), (t_out, None)):
        ax.axhline(tg, color=viz.EVENT_GUIDE, lw=0.6, ls=(0, (2.5, 2.0)),
                   alpha=0.8, zorder=2)
        if lab is not None:
            ax.text(0.985, tg, lab, transform=ytrans, ha="right", va="bottom",
                    fontsize=viz.FONT_ANNOT, color="#777777")

    if case is not None:
        anchor = float(np.asarray(case["anchor_state"]).reshape(-1)[0])
        for tr in case["descendant_trajs"]:
            tr = np.asarray(tr)
            ax.plot(tr[:, 1], tr[:, 0], color=_fate_color(tr[-1, 1]),
                    lw=TRUTH_LW + 0.3, alpha=0.95, solid_capstyle="round", zorder=5)
        _fate_targets(ax, np.asarray(case["true_t_out_states"], dtype=float).reshape(-1),
                      t_out, zorder=7)
        if "clone_lineage" in case:
            _ancestor_dot(ax, case["clone_lineage"])
        ax.scatter([anchor], [t_in], marker="*", s=STAR_SIZE, color=viz.TRUTH,
                   edgecolors="white", linewidths=0.7, zorder=8)

    root_t = case["clone_lineage"]["root_t"] if case and "clone_lineage" in case \
        else split_t
    _warped_axis(ax, t_in, t_out, [split_t, root_t, t_in, t_out], xlim)


def _pred_panel(ax, case, n_key, split_t, t_in, t_out, xlim):
    """Reconstruction under context size n. The SAME clone lineage (root -> the
    common-ancestor diamond -> 8 cousins) is shown in every panel; gold marks
    the cells actually pooled -- just the starting cell for n=1, all 8 cousins
    for n=8. True leaves are fate-coloured rings: with n=1 the prediction commits
    to the WRONG fate, with n=8 it lands on the right one."""
    lin = case["clone_lineage"]
    anchor = float(np.asarray(case["anchor_state"]).reshape(-1)[0])
    preds = np.asarray(case[f"{n_key}_pred_states"], dtype=float).reshape(-1)
    sibs = np.asarray(case["n8_clone_anchor_states"], dtype=float).reshape(-1)
    pooled = np.asarray(case[f"{n_key}_clone_anchor_states"], dtype=float).reshape(-1)
    tgt = np.asarray(case["true_t_out_states"], dtype=float).reshape(-1)

    _generation_bands(ax, t_out, 1.0)

    # single ancestral lineage (root -> common ancestor), then the shared family
    # tree (ancestor -> 8 cousins), both faint; plus a fate-coloured ghost of
    # the starting cell's true descendants.
    for seg in lin.get("ancestry", []):
        seg = np.asarray(seg)
        ax.plot(seg[:, 1], seg[:, 0], color=viz.PRESPLIT, alpha=0.32, lw=0.6,
                solid_capstyle="round", rasterized=True, zorder=2)
    _draw_lineage(ax, lin, color=viz.PRESPLIT, alpha=0.28, lw=0.5, zorder=2)
    for tr in case["descendant_trajs"]:
        tr = np.asarray(tr)
        ax.plot(tr[:, 1], tr[:, 0], color=_fate_color(tr[-1, 1]), lw=0.7,
                alpha=GHOST_ALPHA + 0.12, solid_capstyle="round", zorder=3)

    # every cousin endpoint at t_in; pooled ones gold, the rest faint grey.
    is_pooled = np.array([np.any(np.abs(pooled - s) < 1e-6) for s in sibs])
    if (~is_pooled).any():
        ax.scatter(sibs[~is_pooled], np.full((~is_pooled).sum(), t_in),
                   s=ANCHOR_SIZE * 0.7, color=viz.PRESPLIT, alpha=0.55,
                   edgecolors="white", linewidths=0.4, zorder=4)
    ax.scatter(sibs[is_pooled], np.full(is_pooled.sum(), t_in), s=ANCHOR_SIZE,
               color=viz.ANCHOR, alpha=0.95, edgecolors="white", linewidths=0.5,
               zorder=5)
    _fate_targets(ax, tgt, t_out, zorder=6)
    # predicted leaves as interpolants from the starting cell.
    segs = [[(anchor, t_in), (p, t_out)] for p in preds]
    ax.add_collection(LineCollection(segs, colors=viz.MODEL, linewidths=INTERP_LW,
                                     alpha=0.9, zorder=7, capstyle="round"))
    ax.scatter(preds, np.full(preds.shape, t_out), s=PT_SIZE, color=viz.MODEL,
               edgecolors="white", linewidths=0.5, zorder=8)
    _ancestor_dot(ax, lin)
    ax.scatter([anchor], [t_in], marker="*", s=STAR_SIZE, color=viz.TRUTH,
               edgecolors="white", linewidths=0.7, zorder=11)
    _warped_axis(ax, t_in, t_out, [split_t, lin["root_t"], t_in, t_out], xlim)
    for tg in (t_in, t_out):
        ax.axhline(tg, color=viz.EVENT_GUIDE, lw=0.5, ls=(0, (2.5, 2.0)),
                   alpha=0.6, zorder=1)


# --------------------------------------------------------------------------
#  Metrics strip (energy distance vs context size n, per sigma)
# --------------------------------------------------------------------------


def _metric_panel(ax, splitter_rows, baseline_rows, sigma, *, title):
    ns, means, sems = _model_scaling(splitter_rows, sigma)
    x = np.log2(np.asarray(ns, dtype=float))

    # n-independent baselines: horizontal reference lines spanning the panel.
    for key, method in BASELINE_METHODS:
        m, _s = _baseline_level(baseline_rows, sigma, method)
        if np.isfinite(m):
            ax.axhline(m, color=viz.METHOD[key], lw=1.1, ls=(0, (3, 2)),
                       alpha=0.9, zorder=1)

    # pool-split scaling curve.
    ax.errorbar(x, means, yerr=sems, color=viz.MODEL,
                marker=viz.METHOD_MARKER["model"], ms=MARKER_SIZE, lw=LINE_LW,
                capsize=2.0, elinewidth=0.8, capthick=0.8, zorder=3)

    ax.set_yscale("log")
    # keep only major decade labels; the narrow per-sigma ranges otherwise
    # clutter the strip with minor-tick labels (cf. the crossing metric strip).
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
    ax.set_xlabel("context size $n$")
    ax.set_title(title, fontsize=viz.FONT_TITLE)
    ax.grid(axis="y", which="both", color="0.92", lw=0.6, zorder=0)


# --------------------------------------------------------------------------
#  Layout (shared by the composite and the separated-row renderers)
# --------------------------------------------------------------------------


def _layout(results: dict):
    """Resolve the shared drawing context: clone-view + truth-view x-limits and
    the key times. Used by both ``render_figure`` and ``render_rows``."""
    case = results.get("case")
    t_in = float(results["time_input"])
    t_out = float(results["leaf_time"])

    xs = []
    if case is not None:
        xs.append(float(np.asarray(case["anchor_state"]).reshape(-1)[0]))
        for k in ("true_t_out_states", "n1_pred_states", "n8_pred_states",
                  "n8_clone_anchor_states"):
            xs.extend(np.asarray(case[k], dtype=float).reshape(-1).tolist())
        for tr in case["descendant_trajs"]:
            xs.extend(np.asarray(tr)[:, 1].tolist())
        for seg in (case["clone_lineage"]["segments"]
                    + case["clone_lineage"].get("ancestry", [])):
            xs.extend(np.asarray(seg)[:, 1].tolist())
    xlo, xhi = (min(xs), max(xs)) if xs else (-1.5, 1.5)
    pad = max(0.15, 0.12 * (xhi - xlo))
    xlim = (xlo - pad, xhi + pad)

    cartoon = results.get("cartoon", {})
    split_t = float(cartoon.get("split_time", t_in))
    pop = [np.asarray(t)[:, 1] for t, _ in cartoon.get("trajectories", [])]
    if pop:
        allx = np.concatenate(pop)
        tlo, thi = float(allx.min()), float(allx.max())
        tpad = 0.06 * (thi - tlo)
        xlim_truth = (tlo - tpad, thi + tpad)
    else:
        xlim_truth = xlim

    return {"case": case, "cartoon": cartoon, "t_in": t_in, "t_out": t_out,
            "split_t": split_t, "xlim": xlim, "xlim_truth": xlim_truth}


def _glyph_handles():
    """Recon-row glyph legend handles (ancestor / start / pooled / leaf / pred)."""
    return [
        Line2D([], [], marker="D", linestyle="none", color=viz.TRUTH,
               markersize=5.5, markeredgecolor="white", markeredgewidth=0.5,
               label="common ancestor"),
        Line2D([], [], marker="*", linestyle="none", color=viz.TRUTH,
               markersize=9, markeredgecolor="white", markeredgewidth=0.5,
               label="starting cell"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=viz.ANCHOR,
               markeredgecolor="white", markersize=5.5, markeredgewidth=0.5,
               label="pooled cousins"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=viz.TRUTH, markersize=6.5, markeredgewidth=1.0,
               label="true leaf (fate-coloured)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=viz.MODEL,
               markeredgecolor="white", markersize=5.5, markeredgewidth=0.5,
               label="prediction"),
    ]


def _save(fig, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PDF + SVG (editable vectors for hand assembly) + PNG (quick preview).
    # bbox_inches="tight" so multi-line axis labels (e.g. the "sigma=..\ntime"
    # y-label) aren't clipped at the figure edge.
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(out_path.with_suffix(suffix), dpi=viz.DPI,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------
#  Separated rows (one chrome-free file per row, for hand assembly)
# --------------------------------------------------------------------------


def render_rows(results: dict, out_dir: str | Path):
    """Emit the context figure as separated, chrome-free rows -- mirroring the
    crossing-family panels so they can be assembled and labelled by hand. Writes into *out_dir*:

      recon_row.pdf      [dynamics (truth) | pool-split n=1 | pool-split n=8]
      metrics_strip.pdf  energy distance vs context size n, one panel per sigma
      legend_glyph.pdf   recon-row glyph legend
      legend_method.pdf  method legend for the metrics strip
    """
    viz.apply_rc()
    out_dir = Path(out_dir)
    L = _layout(results)
    case, cartoon = L["case"], L["cartoon"]
    sigmas = list(results["sigmas"])

    # --- reconstruction row: truth | n=1 | n=8 (no titles/legend) ---
    fig, axes = plt.subplots(1, 3, figsize=(3 * RECON_WIDTH, RECON_H), dpi=viz.DPI,
                             gridspec_kw={"wspace": WSPACE})
    _truth_panel(axes[0], cartoon, case, L["xlim_truth"])
    axes[0].set_xlabel("state")
    cs = case["sigma"] if case is not None else None
    axes[0].set_ylabel(f"$\\sigma = {cs:g}$\ntime" if cs is not None else "time")
    for c, n_key in enumerate(("n1", "n8")):
        ax = axes[1 + c]
        if case is not None:
            _pred_panel(ax, case, n_key, L["split_t"], L["t_in"], L["t_out"], L["xlim"])
        ax.set_xlabel("state")
        ax.tick_params(labelleft=False)
    _save(fig, out_dir / "recon_row.pdf")

    # --- metrics strip: one panel per sigma (no legend) ---
    # Trim 0.15in like the crossing scaling: this strip carries an x-axis label
    # ("context size n") on top of the title, so it would otherwise render one
    # line taller than the recon row / crossing strips.
    fig, axes = plt.subplots(1, len(sigmas),
                             figsize=(len(sigmas) * METRIC_WIDTH, METRIC_H - 0.15),
                             dpi=viz.DPI, squeeze=False,
                             gridspec_kw={"wspace": METRIC_WSPACE})
    for s, sigma in enumerate(sigmas):
        _metric_panel(axes[0, s], results["splitter_rows"], results["baseline_rows"],
                      sigma, title=fr"$\sigma = {sigma:g}$")
        if s == 0:
            axes[0, s].set_ylabel("energy distance")
        else:
            axes[0, s].tick_params(labelleft=False)
    _save(fig, out_dir / "metrics_strip.pdf")

    # --- standalone legends ---
    gh = _glyph_handles()
    fig = plt.figure(figsize=(0.5 + 1.7 * len(gh), 0.4), dpi=viz.DPI)
    fig.legend(gh, [h.get_label() for h in gh], loc="center", ncol=len(gh),
               frameon=False, handlelength=1.2, columnspacing=1.4, handletextpad=0.3,
               fontsize=viz.FONT_ANNOT)
    _save(fig, out_dir / "legend_glyph.pdf")

    handles, labels = viz.method_handles(["model", "fm", "gs", "gf"])
    fig = plt.figure(figsize=(0.5 + 1.1 * len(handles), 0.4), dpi=viz.DPI)
    fig.legend(handles, labels, loc="center", ncol=len(handles), frameon=False,
               handlelength=1.6, columnspacing=1.4, handletextpad=0.4)
    _save(fig, out_dir / "legend_method.pdf")

    print(f"wrote context rows -> {out_dir}")
    return out_dir


# --------------------------------------------------------------------------
#  Top-level composite
# --------------------------------------------------------------------------


def render_figure(results: dict, out_path: str | Path):
    """Render the context-scaling figure in the shared crossing-family grammar."""
    viz.apply_rc()

    sigmas = list(results["sigmas"])
    splitter_rows = results["splitter_rows"]
    baseline_rows = results["baseline_rows"]
    L = _layout(results)
    case, cartoon = L["case"], L["cartoon"]
    t_in, t_out, split_t = L["t_in"], L["t_out"], L["split_t"]
    xlim, xlim_truth = L["xlim"], L["xlim_truth"]

    recon_cols = 3        # truth | n=1 | n=8
    nS = len(sigmas)
    fig_w = max(nS * METRIC_WIDTH, recon_cols * RECON_WIDTH) + 1.1
    fig_h = RECON_H + METRIC_H + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=viz.DPI)
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[RECON_H, METRIC_H],
        hspace=HSPACE, left=0.10, right=0.975, top=0.845, bottom=0.16,
    )

    # ---- reconstruction row ----
    recon = outer[0, 0].subgridspec(1, recon_cols, wspace=WSPACE)
    case_sigma = case["sigma"] if case is not None else None
    ax_t = fig.add_subplot(recon[0, 0])
    _truth_panel(ax_t, cartoon, case, xlim_truth)
    ax_t.set_title("dynamics (truth)", color="0.25")
    ax_t.set_ylabel(f"$\\sigma = {case_sigma:g}$\ntime" if case_sigma is not None else "time")
    ax_t.set_xlabel("state")

    for c, (n_key, n_lab) in enumerate(
        [("n1", "pool-split  $n=1$"), ("n8", "pool-split  $n=8$")]
    ):
        ax = fig.add_subplot(recon[0, 1 + c])
        if case is not None:
            _pred_panel(ax, case, n_key, split_t, t_in, t_out, xlim)
        ax.set_title(n_lab, color=viz.MODEL)
        ax.set_xlabel("state")
        ax.tick_params(labelleft=False)

    # ---- metrics strip: one panel per sigma ----
    strip = outer[1, 0].subgridspec(1, nS, wspace=METRIC_WSPACE)
    for s, sigma in enumerate(sigmas):
        axm = fig.add_subplot(strip[0, s])
        _metric_panel(axm, splitter_rows, baseline_rows, sigma,
                      title=fr"$\sigma = {sigma:g}$")
        if s == 0:
            axm.set_ylabel("energy distance")
        else:
            axm.tick_params(labelleft=False)

    # glyph legend for the reconstruction row, placed under the title (with
    # room) rather than crammed between the two tiers.
    glyph_handles = _glyph_handles()
    fig.legend(glyph_handles, [h.get_label() for h in glyph_handles],
               loc="center", ncol=len(glyph_handles), bbox_to_anchor=(0.5, 0.915),
               handlelength=1.2, columnspacing=1.4, handletextpad=0.3,
               fontsize=viz.FONT_ANNOT)

    # shared method legend along the bottom (for the metric strip).
    keys = ["model", "fm", "gs", "gf"]
    handles, labels = viz.method_handles(keys)
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.012), handlelength=1.6,
               columnspacing=1.4, handletextpad=0.4)

    fig.suptitle("context scaling  (n cousins → leaf prediction)",
                 fontsize=viz.FONT_TITLE + 2.0, y=0.965)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=viz.DPI)
    fig.savefig(out_path.with_suffix(".png"), dpi=viz.DPI)
    plt.close(fig)
    return [out_path]
