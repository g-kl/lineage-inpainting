"""Shared "reconstructed-dynamics" panel grammar for the crossing benchmark.

The single source of truth for the panel drawers used by the crossing-family
eval across all ambient dimensions (native 1-D is just ``proj_dim=1``). The
synthetic figure (``synthetic_evals.figure``) reuses these drawers. Style
(palette, type, metric) comes from ``synthetic_evals.viz``.

One figure per mode, presenting a single model (the "pool-split" bridge) against
every baseline as *reconstructed dynamics*. Layout:

  reconstruction rows   one row per variant (crossing, coherent). Each row is
                        [true dynamics | one panel per method]. Every panel is
                        state-vs-time with time running downward; a method panel
                        draws each predicted transition as a linear interpolant
                        (in the method's colour) from the true source state to
                        the predicted target, so the panel reconstructs the
                        process. A faithful model rebuilds the branching tree;
                        a collapsed one (e.g. deterministic FM) does not.
                        Branch-transition time bands are tinted.
  metrics strip         full-width bottom row: two side-by-side panels
                        (crossing, coherent) of the per-transition error
                        (energy distance for the splitter, ancestral MSE for the
                        pooler), one log-y series per method.

mode="splitter" -> forward (root -> leaves), branching reconstruction.
mode="pooler"   -> reverse (leaves -> ancestor), converging reconstruction.

High-dim vs 1-D is handled transparently: when a variant's results carry a
``proj`` entry, state arrays are read out as plotting coordinates by projecting
onto the recorded latent direction ``w`` (so the panels stay 1-D); native 1-D
runs have no ``proj`` and the coordinate is just the scalar state. Error metrics
are always computed in the full ambient space. Layout knobs are in the CONFIG
block below; the palette/type/metric live in ``synthetic_evals.viz``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from synthetic_evals import viz

# --------------------------------------------------------------------------
#  CONFIG (layout knobs -- shared palette/type/metric live in synthetic_evals.viz)
# --------------------------------------------------------------------------

DPI = viz.DPI

RECON_H = 1.1          # height (in) of each reconstruction row (trajectory panels)
METRIC_H = 1.1         # height (in) of the bottom metrics strip
DYN_WIDTH = 1.5
METHOD_WIDTH = 1.4
WSPACE = 0.16
RECON_HSPACE = 0.20    # gap between the (tight) reconstruction rows
HSPACE = 0.40          # larger gap before the metrics strip
METRIC_WSPACE = 0.16

FONT_TITLE = viz.FONT_TITLE

TRAJ_LW = 0.45
TRAJ_ALPHA = 0.35

# Reconstruction interpolants (coloured by method).
RECON_LW = 0.5
RECON_ALPHA_C = 7.0    # segment alpha ~ C / sqrt(n_segments), clamped
RECON_ALPHA_CLAMP = (0.04, 0.55)
RECON_PT_SIZE = 3.0
RECON_PT_C = 11.0
RECON_PT_CLAMP = (0.15, 0.7)

MARKER_SIZE = 4.5
LINE_LW = 1.4

AXES_LIM = (-2.0, 2.0)
AXES_TICKS = (-1, 0, 1)

BRANCH_TINT = viz.EVENT_TINT  # faint wash on branch-transition time bands
BRANCH_SPREAD_THRESH = 0.3   # transition is a "branch" if siblings spread > this

ROW_LABELS = {"crossing": "crossing", "noncrossing": "coherent"}
# Display order across the two dataset variants: coherent (top/left) then crossing.
VARIANT_ORDER = ("noncrossing", "crossing")


# Back-projection direction, set per variant by ``render_figure`` to the latent
# direction ``w`` recorded in results["proj"]; None for native 1-D runs. State
# arrays are read out as plotting coordinates by projecting onto w (the signal
# lives on span(w); off-direction noise averages away), so the reconstruction
# panels stay 1-D. Error metrics below are untouched and remain full-D.
_PROJ_W = None


def _flat(a):
    """Reduce a state array to its scalar plotting coordinate: project onto the
    latent direction in high-dim, else drop a trailing singleton axis."""
    a = np.asarray(a)
    if _PROJ_W is not None and a.shape[-1] == _PROJ_W.shape[0] and _PROJ_W.shape[0] > 1:
        return a @ _PROJ_W
    return a.squeeze(-1) if a.shape[-1] == 1 else a


# --------------------------------------------------------------------------
#  Metrics  (energy_distance / sem live in synthetic_evals.viz)
# --------------------------------------------------------------------------


# (cache_key, color, label, marker). ``ps``/``pool`` is our shared bridge model.
_FORWARD_KEYS = {
    "fm": ("fm_pairs", viz.METHOD["fm"], viz.METHOD_LABEL["fm"], viz.METHOD_MARKER["fm"]),
    "gs": ("gs_pairs", viz.METHOD["gs"], viz.METHOD_LABEL["gs"], viz.METHOD_MARKER["gs"]),
    "gf": ("gf_pairs", viz.METHOD["gf"], viz.METHOD_LABEL["gf"], viz.METHOD_MARKER["gf"]),
    "ps": ("ps_pairs", viz.METHOD["model"], viz.METHOD_LABEL["model"], viz.METHOD_MARKER["model"]),
}
_REVERSE_KEYS = {
    "fm_rev": ("fm_rev_pairs", viz.METHOD["fm"], viz.METHOD_LABEL["fm"], viz.METHOD_MARKER["fm"]),
    "gs_rev": ("gs_rev_pairs", viz.METHOD["gs"], viz.METHOD_LABEL["gs"], viz.METHOD_MARKER["gs"]),
    "gf_rev": ("gf_rev_pairs", viz.METHOD["gf"], viz.METHOD_LABEL["gf"], viz.METHOD_MARKER["gf"]),
    "pool":   ("pool_pairs",   viz.METHOD["model"], viz.METHOD_LABEL["model"], viz.METHOD_MARKER["model"]),
}


def _per_ancestor_energy_stats(eval_data, key):
    """Per-transition mean and SEM of per-ancestor energy distance (splitter)."""
    return forward_energy_stats(eval_data, _FORWARD_KEYS[key][0])


def forward_energy_stats(eval_data, cache_key):
    """Per-transition mean and SEM of per-ancestor forward energy score: each parent's
    predicted children against each of its true children (the shared estimator's
    one-point form, >= 0), averaged over the true children. Not the two-sample
    U-statistic against the children: at a branch the siblings sit one per attractor,
    not two iid draws, so that would score a perfect model below 0.

    Takes the ``*_pairs`` cache key directly, so it works for any method present in
    ``eval_data`` rather than only the four in ``_FORWARD_KEYS``. Full-D throughout.
    """
    means, sems = [], []
    for t_src, t_tgt in eval_data["pair_times"]:
        if (t_src, t_tgt) not in eval_data["gt_pairs"]:
            means.append(np.nan); sems.append(np.nan); continue
        x_tgt = eval_data["gt_pairs"][(t_src, t_tgt)]["x_tgt"]
        if x_tgt.ndim == 1:
            x_tgt = x_tgt[:, None]
        samples = eval_data[cache_key][(t_src, t_tgt)]
        if samples.ndim == 2:
            samples = samples[..., None]
        c2p = eval_data["pool_pairs"][(t_src, t_tgt)]["child_to_parent"]
        per_parent = []
        for p in range(int(c2p.max() + 1) if c2p.size else 0):
            mask = c2p == p
            if not mask.any():
                continue
            pred = samples[:, mask, :].reshape(-1, samples.shape[-1])
            per_parent.append(np.mean([viz.energy_distance(pred, y[None])
                                       for y in x_tgt[mask]]))
        per_parent = np.asarray(per_parent, dtype=np.float64)
        means.append(float(per_parent.mean()) if per_parent.size else np.nan)
        sems.append(viz.sem(per_parent))
    return np.asarray(means), np.asarray(sems)


def _per_ancestor_reverse_energy_stats(eval_data, key):
    """Per-transition mean and SEM of per-ancestor energy distance (pooler):
    each parent's sampled ancestral predictions vs its true ancestor. Symmetric
    with the splitter. Handles parent_pred as (S, P, dim) or (P, dim)."""
    cache_key, *_ = _REVERSE_KEYS[key]
    means, sems = [], []
    for t in eval_data["pair_times"]:
        if t not in eval_data[cache_key]:
            means.append(np.nan); sems.append(np.nan); continue
        rev = eval_data[cache_key][t]
        pp = rev["parent_pred"]
        if pp.ndim == 2:
            pp = pp[None]                      # (P, dim) -> (1, P, dim)
        pt = rev["parent_true"]
        if pt.ndim == 1:
            pt = pt[:, None]
        if pp.shape[1] == 0:
            means.append(np.nan); sems.append(np.nan); continue
        per = np.asarray([viz.energy_distance(pp[:, p, :], pt[p:p + 1])
                          for p in range(pt.shape[0])], dtype=np.float64)
        means.append(float(per.mean()) if per.size else np.nan)
        sems.append(viz.sem(per))
    return np.asarray(means), np.asarray(sems)


def _branch_rows(ev):
    """A transition is a 'branch' when a parent's children fan across attractor
    modes -- detected (GT-only) as a large mean within-parent sibling spread of
    x_tgt (~0.4-0.9 at crossings vs ~0.03 otherwise)."""
    flags = []
    for t in ev["pair_times"]:
        xt = _flat(ev["gt_pairs"][t]["x_tgt"])
        c2p = ev["pool_pairs"][t]["child_to_parent"]
        stds = [xt[c2p == p].std()
                for p in range(int(c2p.max() + 1) if c2p.size else 0)
                if (c2p == p).sum() >= 2]
        flags.append((float(np.mean(stds)) if stds else 0.0) >= BRANCH_SPREAD_THRESH)
    return flags


# --------------------------------------------------------------------------
#  Panel drawers
# --------------------------------------------------------------------------


def _time_axis(ax, obs):
    """Shared y = time, running downward (earliest at top)."""
    t0, t1 = float(min(obs)), float(max(obs))
    ax.set_ylim(t1, t0)
    ax.set_yticks(list(obs))
    ax.set_xlim(*AXES_LIM)
    ax.set_xticks(AXES_TICKS)


def _tint_branches(ax, pair_times, branch):
    """Shade the branch-transition time bands (light orange) behind everything."""
    for (ts, tt), b in zip(pair_times, branch):
        if b:
            ax.axhspan(min(ts, tt), max(ts, tt), color=BRANCH_TINT, zorder=0)


def _plot_dynamics(ax, trajectories, obs_times, pair_times, branch):
    """True SDE trajectories, state on x and time on y (downward)."""
    _tint_branches(ax, pair_times, branch)
    for traj, leaf_mu in trajectories:
        ax.plot(
            traj[:, 1], traj[:, 0],
            color=viz.fate_color(leaf_mu), alpha=TRAJ_ALPHA, lw=TRAJ_LW,
            solid_capstyle="round", rasterized=True,
        )
    for t in obs_times:
        ax.axhline(t, color=viz.EVENT_GUIDE, lw=0.5, ls=(0, (2, 2)), alpha=0.8)
    _time_axis(ax, obs_times)
    ax.set_ylabel("time")


def _recon_panel(ax, ev, cache_key, color, *, mode, obs, branch):
    """Reconstruct the dynamics from a method: each predicted transition is a
    linear interpolant in the method's colour, from the true source state to the
    predicted target. Splitter goes forward (down in time, branching); pooler
    goes backward (up in time, converging to the predicted ancestor)."""
    _tint_branches(ax, ev["pair_times"], branch)
    for t in obs:
        ax.axhline(t, color=viz.EVENT_GUIDE, lw=0.5, ls=(0, (2, 2)), alpha=0.8, zorder=0)

    segs, ex, ey = [], [], []
    for (ts, tt) in ev["pair_times"]:
        if mode == "splitter":
            xs = _flat(ev["gt_pairs"][(ts, tt)]["x_src"])
            S = ev[cache_key][(ts, tt)]
            ny = _flat(S)                                  # (n_samples, M)
            for s in range(ny.shape[0]):
                for m in range(ny.shape[1]):
                    segs.append([(xs[m], ts), (ny[s, m], tt)])
                    ex.append(ny[s, m]); ey.append(tt)
        else:
            xt = _flat(ev["gt_pairs"][(ts, tt)]["x_tgt"])
            rev = ev[cache_key][(ts, tt)]
            pp = rev["parent_pred"]
            if pp.ndim == 2:
                pp = pp[None]                      # (P, dim) -> (1, P, dim)
            c2p = rev["child_to_parent"]
            for m in range(len(xt)):
                par = c2p[m]
                for s in range(pp.shape[0]):       # one interpolant per sample
                    yv = float(_flat(pp[s, par]))
                    segs.append([(xt[m], tt), (yv, ts)])
                    ex.append(yv); ey.append(ts)

    n = max(1, len(segs))
    a_seg = float(np.clip(RECON_ALPHA_C / np.sqrt(n), *RECON_ALPHA_CLAMP))
    a_pt = float(np.clip(RECON_PT_C / np.sqrt(n), *RECON_PT_CLAMP))
    ax.add_collection(LineCollection(
        segs, colors=color, linewidths=RECON_LW, alpha=a_seg, zorder=1,
        capstyle="round", rasterized=True))
    ax.scatter(ex, ey, s=RECON_PT_SIZE, c=color, alpha=a_pt,
               linewidths=0, zorder=2, rasterized=True)
    _time_axis(ax, obs)


def _metric_panel(ax, ev, keys_dict, method_keys, stats_fn, branch, *, arrow, title):
    """Per-transition error, one log-y series per method. x = transitions."""
    pts = ev["pair_times"]
    x = np.arange(len(pts), dtype=float)
    for i, b in enumerate(branch):
        if b:
            ax.axvspan(i - 0.5, i + 0.5, color=BRANCH_TINT, zorder=0)
    offs = np.linspace(-0.1, 0.1, len(method_keys))
    for i, key in enumerate(method_keys):
        _, color, label, marker = keys_dict[key]
        m, s = stats_fn(ev, key)
        ax.errorbar(x + offs[i], m, yerr=s, color=color, marker=marker,
                    ms=MARKER_SIZE, lw=LINE_LW, label=label,
                    capsize=2.0, elinewidth=0.8, capthick=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    if arrow == "forward":
        ax.set_xticklabels([fr"${int(a)}\!\to\!{int(b)}$" for a, b in pts])
    else:
        ax.set_xticklabels([fr"${int(b)}\!\to\!{int(a)}$" for a, b in pts])
    ax.set_xlim(-0.5, len(pts) - 0.5)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)


# --------------------------------------------------------------------------
#  Top-level figure (one per mode; both variants inside)
# --------------------------------------------------------------------------


def render_figure(results: dict, out_path: str | Path, *, mode: str = "splitter"):
    """Render one figure for `mode`: reconstruction rows (one per variant) over
    a full-width metrics strip (one panel per variant). Returns [out_path].

    mode="splitter" -> forward reconstruction + energy distance.
    mode="pooler"   -> reverse reconstruction + ancestral MSE.
    """
    if mode not in ("splitter", "pooler"):
        raise ValueError(f"mode must be 'splitter' or 'pooler', got {mode!r}")
    viz.apply_rc()

    if mode == "splitter":
        keys_dict, order = _FORWARD_KEYS, ("fm", "gs", "gf", "ps")
        stats_fn, err_lab, arrow = _per_ancestor_energy_stats, "energy distance", "forward"
        subtitle = "splitter  (root → leaves)"
    else:
        keys_dict, order = _REVERSE_KEYS, ("fm_rev", "gs_rev", "gf_rev", "pool")
        stats_fn, err_lab, arrow = _per_ancestor_reverse_energy_stats, "energy distance", "reverse"
        subtitle = "pooler  (leaves → ancestor)"

    variants = [v for v in VARIANT_ORDER if v in results]
    method_keys = [k for k in order if keys_dict[k][0] in results[variants[0]]]
    nV, nM = len(variants), len(method_keys)
    ncol = 1 + nM

    fig_w = DYN_WIDTH + nM * METHOD_WIDTH + 1.0
    fig_h = nV * RECON_H + METRIC_H + 1.15
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=viz.DPI)
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[nV * RECON_H, METRIC_H],
        hspace=HSPACE, left=0.07, right=0.975, top=0.91, bottom=0.13,
    )
    recon_block = outer[0, 0].subgridspec(nV, 1, hspace=RECON_HSPACE)

    # reconstruction rows (one per variant)
    global _PROJ_W
    for r, variant in enumerate(variants):
        ev = results[variant]
        _PROJ_W = np.asarray(ev["proj"]["w"]) if "proj" in ev else None
        branch = _branch_rows(ev); obs = list(ev["obs_times"])
        row = recon_block[r, 0].subgridspec(1, ncol, width_ratios=[DYN_WIDTH] + [METHOD_WIDTH] * nM,
                                            wspace=WSPACE)
        axd = fig.add_subplot(row[0, 0])
        _plot_dynamics(axd, ev["trajectories"], obs, ev["pair_times"], branch)
        axd.set_ylabel(f"{ROW_LABELS.get(variant, variant)}\ntime")
        if r == 0:
            axd.set_title("dynamics (truth)", color="0.25")
        if r == nV - 1:
            axd.set_xlabel("state")
        for c, key in enumerate(method_keys):
            ax = fig.add_subplot(row[0, 1 + c])
            _recon_panel(ax, ev, keys_dict[key][0], keys_dict[key][1],
                         mode=mode, obs=obs, branch=branch)
            ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(keys_dict[key][2], color=keys_dict[key][1])
            if r == nV - 1:
                ax.set_xlabel("state")

    # metrics strip: one panel per variant, side by side; shared label + legend
    # below (never overlaying the data).
    strip = outer[1, 0].subgridspec(1, nV, wspace=METRIC_WSPACE)
    strip_axes = []
    for r, variant in enumerate(variants):
        ev = results[variant]
        _PROJ_W = np.asarray(ev["proj"]["w"]) if "proj" in ev else None
        axm = fig.add_subplot(strip[0, r])
        _metric_panel(axm, ev, keys_dict, method_keys, stats_fn, _branch_rows(ev),
                      arrow=arrow, title=ROW_LABELS.get(variant, variant))
        if r == 0:
            axm.set_ylabel(err_lab)
        strip_axes.append(axm)

    # single shared method legend at the very bottom (no x-label).
    handles = [Line2D([], [], color=keys_dict[k][1], marker=keys_dict[k][3],
                      ms=MARKER_SIZE, lw=LINE_LW) for k in method_keys]
    labels = [keys_dict[k][2] for k in method_keys]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.012), handlelength=1.4,
               columnspacing=1.4, handletextpad=0.4)

    fig.suptitle(subtitle, fontsize=FONT_TITLE + 2.0, y=0.965)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=viz.DPI)
    fig.savefig(out_path.with_suffix(".png"), dpi=viz.DPI)
    plt.close(fig)
    return [out_path]
