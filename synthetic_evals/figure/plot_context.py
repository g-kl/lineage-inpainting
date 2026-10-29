"""render the context (fate-bias) panels from the numerical_results caches.

  panel d  reconstruction row  [truth | pool-split n=1 | pool-split n=8], state on x,
           time downward. more context (n) sharpens the leaf prediction toward truth.
           a glyph key sits to the right (common ancestor / starting cell / cousin
           context / true leaf / prediction); the n=1 panel drops the cousin context
           and common ancestor (there is none yet).
  panel e  fate accuracy vs context size n, one panel per sigma, on one 0.4-1
           accuracy range shared across the sigmas. the pool-split curve rises with n;
           the n-independent OT baselines are flat dotted lines. a matching method key
           sits to the right.

the recon panels reuse the small drawers/helpers in synthetic_evals.context.plot; the
metric drawer reads context_scaling.csv. writes panel svgs into imgs/.
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from synthetic_evals.context import plot as ctx
from synthetic_evals import viz
from synthetic_evals.figure import style as S

RECON_H, STRIP_H = 1.32, 1.06
PANELS_RIGHT = 0.82           # panels stop here; the (compact) legend sits snug to the right
LEGEND_X = 0.835
BASELINE_LS = (0, (3, 2))     # dotted/dashed style of the context baseline lines






def _glyph_handles():
    """recon-row glyph key; prediction in the fresh context colour (not a method hue)."""
    return [
        Line2D([], [], marker="D", ls="none", color=viz.TRUTH, ms=5.5,
               mec="white", mew=0.5, label="common ancestor"),
        Line2D([], [], marker="*", ls="none", color=viz.TRUTH, ms=9,
               mec="white", mew=0.5, label="starting cell"),
        Line2D([], [], marker="o", ls="none", mfc=viz.ANCHOR, mec="white", ms=5.5,
               mew=0.5, label="cousin context"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec=viz.TRUTH, ms=6.5,
               mew=1.0, label="true leaf"),
        Line2D([], [], marker="o", ls="none", mfc=S.PRED_COLOR, mec="white", ms=5.5,
               mew=0.5, label="prediction"),
    ]


def _pred_panel(ax, case, n_key, split_t, t_in, t_out, xlim, *, show_context):
    """reconstruction under context size n (reuses context.plot's helpers). gold marks
    the pooled cousins. with show_context=False (n=1) the non-pooled cousins and the
    common-ancestor diamond are dropped -- there is no cousin context yet."""
    lin = case["clone_lineage"]
    anchor = float(np.asarray(case["anchor_state"]).reshape(-1)[0])
    preds = np.asarray(case[f"{n_key}_pred_states"], float).reshape(-1)
    sibs = np.asarray(case["n8_clone_anchor_states"], float).reshape(-1)
    pooled = np.asarray(case[f"{n_key}_clone_anchor_states"], float).reshape(-1)
    tgt = np.asarray(case["true_t_out_states"], float).reshape(-1)

    ctx._generation_bands(ax, t_out, 1.0)
    for seg in lin.get("ancestry", []):
        seg = np.asarray(seg)
        ax.plot(seg[:, 1], seg[:, 0], color=viz.PRESPLIT, alpha=0.32, lw=0.6,
                solid_capstyle="round", rasterized=True, zorder=2)
    ctx._draw_lineage(ax, lin, color=viz.PRESPLIT, alpha=0.28, lw=0.5, zorder=2)
    for tr in case["descendant_trajs"]:
        tr = np.asarray(tr)
        ax.plot(tr[:, 1], tr[:, 0], color=ctx._fate_color(tr[-1, 1]), lw=0.7,
                alpha=ctx.GHOST_ALPHA + 0.12, solid_capstyle="round", zorder=3)

    is_pooled = np.array([np.any(np.abs(pooled - s) < 1e-6) for s in sibs])
    if show_context and (~is_pooled).any():
        ax.scatter(sibs[~is_pooled], np.full((~is_pooled).sum(), t_in),
                   s=ctx.ANCHOR_SIZE * 0.7, color=viz.PRESPLIT, alpha=0.55,
                   edgecolors="white", linewidths=0.4, zorder=4)
    ax.scatter(sibs[is_pooled], np.full(is_pooled.sum(), t_in), s=ctx.ANCHOR_SIZE,
               color=viz.ANCHOR, alpha=0.95, edgecolors="white", linewidths=0.5, zorder=5)
    ctx._fate_targets(ax, tgt, t_out, zorder=6)
    segs = [[(anchor, t_in), (p, t_out)] for p in preds]
    ax.add_collection(LineCollection(segs, colors=S.PRED_COLOR, linewidths=ctx.INTERP_LW,
                                     alpha=0.9, zorder=7, capstyle="round"))
    ax.scatter(preds, np.full(preds.shape, t_out), s=ctx.PT_SIZE, color=S.PRED_COLOR,
               edgecolors="white", linewidths=0.5, zorder=8)
    if show_context:
        ctx._ancestor_dot(ax, lin)
    ax.scatter([anchor], [t_in], marker="*", s=ctx.STAR_SIZE, color=viz.TRUTH,
               edgecolors="white", linewidths=0.7, zorder=11)
    ctx._warped_axis(ax, t_in, t_out, [split_t, lin["root_t"], t_in, t_out], xlim)
    for tg in (t_in, t_out):
        ax.axhline(tg, color=viz.EVENT_GUIDE, lw=0.5, ls=(0, (2.5, 2.0)), alpha=0.6, zorder=1)


def render_recon(out_dir, name="e_context_recon.svg"):
    S.apply_rc()
    with open(os.path.join(S.NR, "context_recon.pkl"), "rb") as f:
        recon = pickle.load(f)
    L = ctx._layout(recon)
    case, cartoon = L["case"], L["cartoon"]
    cs = case["sigma"] if case is not None else None

    fig = plt.figure(figsize=(S.PAGE_W_IN, RECON_H))
    gs = fig.add_gridspec(1, 3, left=S.RECON_LEFT, right=PANELS_RIGHT,
                          top=0.80, bottom=0.24, wspace=0.16)
    ax_t = fig.add_subplot(gs[0, 0])
    ctx._truth_panel(ax_t, cartoon, case, L["xlim_truth"])
    ax_t.set_ylabel("time", fontsize=S.FS_LABEL)
    ax_t.set_xlabel("state", labelpad=1.0)
    ax_t.set_title("ground truth", color=viz.TRUTH, fontsize=S.FS_TITLE, pad=11)
    if cs is not None:                       # sigma sits in grey under the truth title
        ax_t.text(0.5, 1.015, fr"$\sigma = {cs:g}$", transform=ax_t.transAxes,
                  ha="center", va="bottom", color=S.MUTED, fontsize=S.FS_SUB)
    for c, (n_key, lab) in enumerate((("n1", r"context size $n=1$"),
                                      ("n8", r"context size $n=8$"))):
        ax = fig.add_subplot(gs[0, 1 + c])
        if case is not None:
            _pred_panel(ax, case, n_key, L["split_t"], L["t_in"], L["t_out"], L["xlim"],
                        show_context=(n_key != "n1"))
        ax.set_xlabel("state", labelpad=1.0)
        ax.tick_params(labelleft=False)
        ax.set_title(lab, color=S.INK, fontsize=S.FS_TITLE)
    gh = _glyph_handles()
    fig.legend(gh, [h.get_label() for h in gh], loc="center left",
               bbox_to_anchor=(LEGEND_X, 0.5), frameon=False, handlelength=1.0,
               handletextpad=0.4, labelspacing=1.0, fontsize=S.FS_SUB)
    return S.save_panel(fig, out_dir, name, (S.PAGE_W_IN, RECON_H))


def _metric_panel(ax, dfs, title):
    for m in ("gs", "gf", "fm"):
        b = dfs[(dfs["kind"] == "baseline") & (dfs["method"] == m)]
        if len(b) and np.isfinite(b["mean"].iloc[0]):
            ax.axhline(float(b["mean"].iloc[0]), color=S.METHOD_COLORS[m], lw=1.1,
                       ls=BASELINE_LS, alpha=0.9, zorder=1)
    model = dfs[dfs["kind"] == "model"].sort_values("n")
    x = np.log2(model["n"].to_numpy(float))
    ax.errorbar(x, model["mean"], yerr=model["sem"], color=S.METHOD_COLORS["ours"],
                marker=S.METHOD_MARKERS["ours"], ms=3.6, lw=1.3, capsize=1.8,
                elinewidth=0.7, capthick=0.7, zorder=3)
    S.thin_yaxis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(n)) for n in model["n"]])
    ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
    ax.set_xlabel(r"context size $n$")
    ax.set_title(title, fontsize=S.FS_TITLE)
    ax.grid(axis="y", which="both", color="0.92", lw=0.6, zorder=0)


def _metric_legend_handles():
    """ours as a solid marked line, the baselines as dotted lines -- matching panel e."""
    h = []
    for m in S.MAIN_METHODS:
        if m == "ours":
            h.append(Line2D([], [], color=S.METHOD_COLORS[m], marker=S.METHOD_MARKERS[m],
                            ms=4.2, lw=1.3, label=S.METHOD_LABELS[m]))
        else:
            h.append(Line2D([], [], color=S.METHOD_COLORS[m], lw=1.1, ls=BASELINE_LS,
                            label=S.METHOD_LABELS[m]))
    return h


def render_metrics(out_dir, name="f_context_metrics.svg"):
    S.apply_rc()
    df = pd.read_csv(os.path.join(S.NR, "context_scaling.csv"))
    sigmas = sorted(df["sigma"].unique())
    fig = plt.figure(figsize=(S.PAGE_W_IN, STRIP_H))
    gs = fig.add_gridspec(1, len(sigmas), left=S.RECON_LEFT, right=PANELS_RIGHT,
                          top=0.84, bottom=0.30, wspace=0.18)
    axes = []
    for s, sigma in enumerate(sigmas):
        ax = fig.add_subplot(gs[0, s], sharey=axes[0] if axes else None)
        _metric_panel(ax, df[df["sigma"] == sigma], fr"$\sigma = {sigma:g}$")
        if s == 0:
            ax.set_ylabel("fate accuracy")
        else:
            ax.tick_params(which="both", labelleft=False)
        axes.append(ax)
    axes[0].set_ylim(0.4, S.ACC_VMAX + 0.03)   # headroom: markers at 1 clear the spine
    h = _metric_legend_handles()
    fig.legend(h, [x.get_label() for x in h], loc="center left",
               bbox_to_anchor=(LEGEND_X, 0.5), frameon=False, handlelength=1.4,
               handletextpad=0.5, labelspacing=0.9, fontsize=S.FS_SUB)
    return S.save_panel(fig, out_dir, name, (S.PAGE_W_IN, STRIP_H))


if __name__ == "__main__":
    print(render_recon(S.IMGS))
    print(render_metrics(S.IMGS))
