"""row 3 (panels c-f): context scaling of subtree cycle consistency.

A four-column row, all sharing one metric (energy distance, lower = better
recovery) and split by inference DIRECTION:

  panel c  "ancestral inference"  (leaf -> root -> leaf). energy distance binned by
           context size = leaf_time - MRCA_time of the revealed clade (how much added
           leaf context sharpens the root reconstruction). reads
           numerical_results/context_scaling.csv. FALLS with context.
  panel e  "descendant inference"  (predict a focal node v's descendants by splitting
           down from a cross-section frontier above it). x-axis is how many same-level
           cousins the frontier reveals (0 = f_v alone). reads
           context_scaling_forward.csv. FALLS as more cousin context is revealed.
  panels d, f  the same two directions resolved BY CELL TYPE: every terminal fate
           above the rare-fate cut (S.CONTEXT_MIN_LEAVES, celltype_abundance.csv), minus
           blood and extraembryonic fates (S.CONTEXT_EXCLUDE_TYPES), is
           drawn as a grey wispy per-type trend, and the two fates that benefit most
           and least from added context are highlighted.

c and e are mean +/- SEM line plots (markers + error bars) and carry dotted
horizontal reference lines for the two context-blind GENOT baselines
(numerical_results/context_scaling_genot.csv, written by
compute_context_scaling_genot.py) -- a pure OT flow cannot use lineage context, so its
recovery error is flat.

writes panels/{C_context_box,D_context_forward,E_context_box_celltype,
F_context_forward_celltype}.svg (drawn as panels c / e / d / f).
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from analysis.figures.summary_qc import style as S

PANEL_H_IN = 1.7
LEGEND_H_IN = 0.3            # the shared horizontal method legend strip below row 3
PANEL_W_IN = 1.45          # per-column width; two boxed pairs + gaps fit the page
XLAB_ANC = "context size\n" + r"$t_{\mathrm{leaf}}\!-\!t_{\mathrm{MRCA}}$"
XLAB_FWD = "cousins revealed"
YLAB = "energy distance"
YLAB_SUB = "(cycle consistency)"
FWD_MAX_COUSINS = 8        # shared by panels e and f; keeps 98% of the descendant rows

AX_RECT = (0.38, 0.33, 0.58, 0.55)       # axes box within each narrow column
MODEL_COLOR = S.METHOD_COLORS["model"]   # model teal, shared by both overall panels
GREY = "#c9c9c9"             # wispy per-cell-type trends
HL_COLORS = ["#D1495B", "#457B9D"]       # benefits-most / benefits-least fate
# the highlighted-fate legend sits inside a 1.45 in column, so fate names are
# abbreviated to fit beside the curves rather than over them. phrases first (they
# would otherwise be abbreviated word by word into something longer), then words.
LEGEND_PHRASES = {"dorsal root ganglion": "DRG", "spinal cord": "sp. cord"}
LEGEND_WORDS = {
    "progenitor": "prog.", "mesoderm": "meso.", "mesenchyme": "mesen.",
    "ectoderm": "ecto.", "endoderm": "endo.", "endothelial": "endoth.",
    "epidermal": "epiderm.", "pharyngeal": "phar.", "presomitic": "presom.",
    "intermediate": "interm.", "extraembryonic": "extraemb.", "paraxial": "parax.",
    "splanchnic": "splanch.", "frontonasal": "frontonas.", "trigeminal": "trigem.",
    "interneuron": "interneur.", "neuron": "neur.", "sensory": "sens.",
    "diencephalon": "dienceph.", "telencephalon": "telenceph.",
    "hypothalamus": "hypothal.", "dermomyotome": "dermomyot.", "cranial": "cran.",
    "somatic": "som.", "motor": "mot.", "cervical": "cerv.", "thoracic": "thor.",
    "lumbar": "lumb.", "sacral": "sacr.", "hindlimb": "hindl.", "forelimb": "forel.",
    "myogenic": "myog.", "hypaxial": "hypax.", "allantoic": "allant.",
    "epibranchial": "epibr.", "olfactory": "olf.",
}


def _short(name):
    """abbreviate a fate name for the in-panel legend. word-boundary substitution, not
    a split on spaces, so parenthesised region names abbreviate too."""
    for long, short in LEGEND_PHRASES.items():
        name = name.replace(long, short)
    for long, short in LEGEND_WORDS.items():
        name = re.sub(rf"\b{long}\b", short, name)
    return name


def _mean_sem(values):
    """mean and standard error of a 1-d array (nan if empty)."""
    v = np.asarray(values, float)
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def _trend(bin_key, energy, bin_vals):
    """mean +/- SEM energy distance per bin value (aligned to ``bin_vals``)."""
    bin_key = np.asarray(bin_key)
    energy = np.asarray(energy, float)
    means, sems = [], []
    for v in bin_vals:
        m, s = _mean_sem(energy[bin_key == v])
        means.append(m)
        sems.append(s)
    return np.asarray(means), np.asarray(sems)


def _new_ax(xlab, title, xlim, xticks, integer_y=False):
    """a fresh narrow-column axis with the shared y-label, title and x ticks.

    integer_y forces whole-number y ticks (used on the ancestral column so its tick
    labels stay as narrow as the descendant side and clear the rotated y-label)."""
    S.apply_rc()
    fig = plt.figure(figsize=(PANEL_W_IN, PANEL_H_IN))
    ax = fig.add_axes(list(AX_RECT))
    if integer_y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_xlabel(xlab)
    # short subplot title ("average" / "per type"), centred over the plot.
    ax.set_title(title, fontsize=S.FS_LABEL, color=S.INK, pad=5)
    # two-line y-axis label: metric in ink, small parenthetical clarifier in grey.
    # Both sit clear of the tick numbers in the widened left margin (no clipping).
    ax.text(-0.53, 0.5, YLAB, transform=ax.transAxes, rotation=90, va="center",
            ha="center", color=S.INK, fontsize=S.FS_LABEL)
    ax.text(-0.36, 0.5, YLAB_SUB, transform=ax.transAxes, rotation=90, va="center",
            ha="center", color=S.INK, fontsize=S.FS_LABEL - 2.5)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:g}" for t in xticks])
    return fig, ax


def _save(fig, out_dir, out_name):
    _despine(fig.axes[0])
    out = os.path.join(out_dir, out_name)
    fig.savefig(out)
    plt.close(fig)
    return out, (PANEL_W_IN, PANEL_H_IN)


def _genot_lines(ax, panel, x0, x1):
    """draw a dotted horizontal reference line per GENOT method for ``panel``.

    Returns proxy handles (one per drawn method) for a shared legend.
    """
    path = os.path.join(S.NR, "context_scaling_genot.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df = df[df["panel"] == panel]
    handles = []
    for method in ("genot_sinkhorn", "genot_cbfgw"):
        vals = df.loc[df["method"] == method, "energy_distance"].to_numpy()
        if vals.size == 0:
            continue
        m = float(vals.mean())
        color = S.METHOD_COLORS[method]
        ax.hlines(m, x0, x1, color=color, lw=1.3, ls=(0, (1.5, 1.5)), zorder=3)
        handles.append(Line2D([0], [0], color=color, lw=1.3, ls=(0, (1.5, 1.5)),
                              label=S.METHOD_LABELS[method]))
    return handles


def _overall_panel(out_dir, out_name, xs, means, sems, xlab, title, panel, xlim,
                   xticks, integer_y=False):
    """shared renderer for the two overall panels: a mean +/- SEM model trend (teal)
    plus dotted GENOT reference lines. The method trio is identical across both, so
    it is not labelled per-panel; the shared legend lives below the row (render_legend)."""
    fig, ax = _new_ax(xlab, title, xlim, xticks, integer_y=integer_y)
    _genot_lines(ax, panel, xlim[0], xlim[1])
    ax.errorbar(xs, means, yerr=sems, color=MODEL_COLOR, lw=1.6, marker="o", ms=3.6,
                mec="white", mew=0.6, capsize=2.0, elinewidth=0.9, zorder=5)
    return _save(fig, out_dir, out_name)


def render_legend(out_dir):
    """the shared horizontal method legend, drawn once below the four row-3 columns
    (the teal model trend + the two dotted GENOT reference lines are identical across
    panels c and e, so the trio is labelled once for the whole row)."""
    S.apply_rc()
    fig = plt.figure(figsize=(S.PAGE_W_IN, LEGEND_H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    handles = [
        Line2D([0], [0], color=MODEL_COLOR, lw=1.6, marker="o", ms=3.6, mec="white",
               mew=0.6, label=S.METHOD_LABELS["model"]),
        Line2D([0], [0], color=S.METHOD_COLORS["genot_sinkhorn"], lw=1.3,
               ls=(0, (1.5, 1.5)), label=S.METHOD_LABELS["genot_sinkhorn"]),
        Line2D([0], [0], color=S.METHOD_COLORS["genot_cbfgw"], lw=1.3,
               ls=(0, (1.5, 1.5)), label=S.METHOD_LABELS["genot_cbfgw"]),
    ]
    ax.legend(handles=handles, loc="center", ncol=3, frameon=False,
              fontsize=S.FS_ANNOT, handlelength=1.8, columnspacing=2.0,
              handletextpad=0.5)
    out = os.path.join(out_dir, "G_context_legend.svg")
    fig.savefig(out)
    plt.close(fig)
    return out, (S.PAGE_W_IN, LEGEND_H_IN)


def _abundance():
    """cell type -> leaves in the tree set (compute_celltype_abundance.py)."""
    path = os.path.join(S.NR, "celltype_abundance.csv")
    return pd.read_csv(path, index_col=0)["n_leaves"]


def _celltype_panel(out_dir, out_name, df, bin_key, bin_vals, xs, group_col, fit_x,
                    xlab, title, xlim, xticks, integer_y=False):
    """render the per-cell-type panels: a grey wispy mean trend per fate, with the
    fates that benefit MOST and LEAST from added context highlighted. Benefit is the
    fitted slope of energy distance vs context (most negative = benefits most).

    a fate is drawn only if it is not blood / extraembryonic (S.CONTEXT_EXCLUDE_TYPES),
    has >= S.CONTEXT_MIN_LEAVES leaves in the tree set AND
    >= S.CELLTYPE_MIN_PER_POINT rows at every point on the axis."""
    fig, ax = _new_ax(xlab, title, xlim, xticks, integer_y=integer_y)
    xs = np.asarray(xs, float)
    abundance = _abundance()
    ranked = {}   # cell type -> (slope, means array)
    for ct, idx in df.groupby(group_col).groups.items():
        if ct in S.CONTEXT_EXCLUDE_TYPES or abundance.get(ct, 0) < S.CONTEXT_MIN_LEAVES:
            continue
        counts = bin_key.loc[idx].value_counts().reindex(bin_vals, fill_value=0)
        if counts.min() < S.CELLTYPE_MIN_PER_POINT:
            continue
        m, e = _trend(bin_key.loc[idx], df.loc[idx, "energy_distance"], bin_vals)
        ax.plot(xs, m, color=GREY, lw=0.5, alpha=0.4, zorder=1)
        slope = np.polyfit(fit_x.loc[idx].to_numpy(float),
                           df.loc[idx, "energy_distance"].to_numpy(float), 1)[0]
        ranked[ct] = (slope, m, e)

    order = sorted(ranked, key=lambda c: ranked[c][0])   # ascending slope
    handles = []
    for ct, color in zip((order[0], order[-1]), HL_COLORS):   # most, least benefit
        # the two highlighted fates carry SEM bars; the grey background does not
        # (30-odd overlaid error bars would be unreadable).
        ax.errorbar(xs, ranked[ct][1], yerr=ranked[ct][2], color=color, lw=1.6,
                    marker="o", ms=3.0, mec="white", mew=0.5, capsize=1.6,
                    elinewidth=0.8, zorder=5)
        handles.append(Line2D([0], [0], color=color, lw=1.6, label=_short(ct)))
    # headroom so the fate legend sits in an empty band above the (descending) curves
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.22)
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=S.FS_ANNOT - 3.0, handlelength=0.9, labelspacing=0.15,
              borderpad=0.1, borderaxespad=0.15, handletextpad=0.35)
    return _save(fig, out_dir, out_name)


def _anc_bins():
    """ancestral binning: energy distance vs context size (time_to_mrca)."""
    df = pd.read_csv(os.path.join(S.NR, "context_scaling.csv"))
    edges = np.asarray(S.CONTEXT_BIN_EDGES, float)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_key = pd.cut(df["time_to_mrca"], edges, include_lowest=True, labels=False)
    # sparse ticks: labels don't collide in a narrow column (and none at the right edge,
    # where a label would be clipped by the panel border)
    xticks = np.array([0.0, 0.5, 1.0])
    return df, bin_key, np.arange(len(centers)), centers, (edges[0], edges[-1]), xticks


def _fwd_bins():
    """descendant binning: energy distance vs cousins revealed, truncated at the cap."""
    df = pd.read_csv(os.path.join(S.NR, "context_scaling_forward.csv"))
    df = df[df["n_cousins"] <= FWD_MAX_COUSINS].reset_index(drop=True)
    xs = np.arange(FWD_MAX_COUSINS + 1)
    xticks = np.arange(0, FWD_MAX_COUSINS + 1, 2)
    return df, df["n_cousins"], xs, xs, (-0.3, FWD_MAX_COUSINS + 0.3), xticks


def render_box(out_dir):
    """panel c: ancestral inference (leaf -> root -> leaf), binned by context size."""
    df, bin_key, bin_vals, xs, xlim, xticks = _anc_bins()
    means, sems = _trend(bin_key, df["energy_distance"], bin_vals)
    return _overall_panel(out_dir, "C_context_box.svg", xs, means, sems, XLAB_ANC,
                          "average", "ancestral", xlim, xticks, integer_y=True)


def render_forward(out_dir):
    """panel e: descendant inference -- energy distance vs cousins revealed."""
    df, bin_key, bin_vals, xs, xlim, xticks = _fwd_bins()
    means, sems = _trend(bin_key, df["energy_distance"], bin_vals)
    return _overall_panel(out_dir, "D_context_forward.svg", xs, means, sems, XLAB_FWD,
                          "average", "forward", xlim, xticks)


def render_box_celltype(out_dir):
    """panel d: ancestral inference resolved by terminal fate (leaf_cell_type)."""
    df, bin_key, bin_vals, xs, xlim, xticks = _anc_bins()
    return _celltype_panel(out_dir, "E_context_box_celltype.svg", df, bin_key, bin_vals,
                           xs, "leaf_cell_type", df["time_to_mrca"],
                           XLAB_ANC, "per type", xlim, xticks, integer_y=True)


def render_forward_celltype(out_dir):
    """panel f: descendant inference resolved by descendant fate (desc_cell_type)."""
    df, bin_key, bin_vals, xs, xlim, xticks = _fwd_bins()
    return _celltype_panel(out_dir, "F_context_forward_celltype.svg", df, bin_key,
                           bin_vals, xs, "desc_cell_type", bin_key,
                           XLAB_FWD, "per type", xlim, xticks, integer_y=True)


def render_box_title(out_dir, text, out_name):
    """render one box title (Latin Modern Sans, ink, text as paths) tightly cropped; returns
    its size in inches so the assembler can break the box border to fit it."""
    import xml.etree.ElementTree as ET
    S.apply_rc()
    fig = plt.figure(figsize=(3.0, 0.35))
    fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=S.FS_LABEL,
             color=S.INK)
    out = os.path.join(out_dir, out_name)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    root = ET.parse(out).getroot()
    w = float(root.get("width").rstrip("pt"))
    h = float(root.get("height").rstrip("pt"))
    return out, (w / 72.0, h / 72.0)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(S.INK)
    ax.tick_params(colors=S.INK)


if __name__ == "__main__":
    panels = os.path.join(S.OUT, "panels")
    os.makedirs(panels, exist_ok=True)
    print(render_box(panels))
    print(render_forward(panels))
    print(render_box_celltype(panels))
    print(render_forward_celltype(panels))
    print(render_legend(panels))
