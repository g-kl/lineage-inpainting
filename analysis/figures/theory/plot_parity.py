"""Render the Bernstein parity benchmark (plot half).

Three renders from the summary written by compute_parity.py: the target-feature
heatmaps, the four-panel placement-by-feature grid, and the lollipop trajectory
clouds that illustrate the null, path and sibling dependence.

Run from the repository root with

    python -m analysis.figures.theory.plot_parity [--quick]

which reads numerical_results/parity_summary.csv (numerical_results/quick/ with
``--quick``) and writes bernstein_parity.*, bernstein_parity_four_panel.* and
bernstein_parity_trajectories.* to imgs/ (imgs/quick/ with ``--quick``).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from analysis.figures.theory import style as S
from analysis.figures.theory.compute_parity import (
    FEATS,
    FEATURE_ORDERS,
    FEAT_PLACEMENT,
    NOISE_SIGMA,
    PARITY_ORDERS,
    PLACEMENTS,
    iid_values,
    parity_values,
    output_dir,
)
from synthetic_evals import viz

FEAT_LABEL = {
    "path": "path",
    "ca": "common ancestor",
}
PLACEMENT_LABEL = {
    "chain": "chain parity",
    "fan": "fan parity",
}
TRAJECTORY_ORDER = 3
TRAJECTORY_SAMPLES = 160
TRAJECTORY_LW = 0.45
TRAJECTORY_ALPHA = 0.12
TRAJECTORY_FG_ALPHA = 0.20
TRAJECTORY_BG_ALPHA = 0.035
TRAJECTORY_BRANCH_BOW = 0.18
TRAJECTORY_BRANCH_SLOT = 0.055
TRAJECTORY_BRANCH_Y_SLOT = 0.075
TRAJECTORY_CHAIN_SCALE = np.array([0.38, 0.72, 1.04], dtype=float)
TRAJECTORY_LEAF_SCALE = 1.28


def _mat(summary, *, placement: str, feature: str, vmax: float) -> np.ndarray:
    return np.array([
        [
            max(summary[(placement, feature, parity_order, feature_order)][0], 0.0)
            / vmax
            for feature_order in FEATURE_ORDERS
        ]
        for parity_order in PARITY_ORDERS
    ])


def _draw_heatmap(ax, mat, *, title: str, x_lo: float, x_hi: float, y_lo: float, y_hi: float):
    image = ax.imshow(
        mat,
        origin="lower",
        aspect="equal",
        cmap=viz.SEQ_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=[x_lo, x_hi, y_lo, y_hi],
    )
    ax.plot(
        [y_lo, y_hi],
        [y_lo, y_hi],
        color="white",
        lw=0.8,
        alpha=0.75,
    )
    ax.set_title(title, fontsize=viz.FONT_TITLE)
    ax.set_xticks(FEATURE_ORDERS)
    ax.set_yticks(PARITY_ORDERS)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    return image


def render(summary, out_dir: Path = S.IMGS):
    viz.apply_rc()
    x_lo, x_hi = FEATURE_ORDERS[0] - 0.5, FEATURE_ORDERS[-1] + 0.5
    y_lo, y_hi = PARITY_ORDERS[0] - 0.5, PARITY_ORDERS[-1] + 0.5
    fig, axes = plt.subplots(
        1, len(FEATS),
        figsize=(5.0, 2.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for ax, feature in zip(axes, FEATS):
        placement = FEAT_PLACEMENT[feature]
        vmax = max(
            max(summary[(placement, feature, parity_order, feature_order)][0], 0.0)
            for parity_order in PARITY_ORDERS
            for feature_order in FEATURE_ORDERS
        )
        vmax = max(vmax, 1e-12)
        image = _draw_heatmap(
            ax,
            _mat(summary, placement=placement, feature=feature, vmax=vmax),
            title=FEAT_LABEL[feature],
            x_lo=x_lo,
            x_hi=x_hi,
            y_lo=y_lo,
            y_hi=y_hi,
        )
        ax.set_xlabel("feature order q", fontsize=viz.FONT_LABEL)
    axes[0].set_ylabel("parity order r", fontsize=viz.FONT_LABEL)
    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes,
            shrink=0.82,
            ticks=[0.0, 0.5, 1.0],
            label="within-feature normalized excess energy",
        )
        cbar.outline.set_linewidth(0.8)
    _save(fig, "bernstein_parity", out_dir)


def render_four_panel(summary, out_dir: Path = S.IMGS):
    viz.apply_rc()
    x_lo, x_hi = FEATURE_ORDERS[0] - 0.5, FEATURE_ORDERS[-1] + 0.5
    y_lo, y_hi = PARITY_ORDERS[0] - 0.5, PARITY_ORDERS[-1] + 0.5
    fig, axes = plt.subplots(
        len(FEATS), len(PLACEMENTS),
        figsize=(5.3, 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for row, feature in enumerate(FEATS):
        vmax = max(
            max(summary[(placement, feature, parity_order, feature_order)][0], 0.0)
            for placement in PLACEMENTS
            for parity_order in PARITY_ORDERS
            for feature_order in FEATURE_ORDERS
        )
        vmax = max(vmax, 1e-12)
        for col, placement in enumerate(PLACEMENTS):
            ax = axes[row, col]
            image = _draw_heatmap(
                ax,
                _mat(summary, placement=placement, feature=feature, vmax=vmax),
                title=PLACEMENT_LABEL[placement] if row == 0 else "",
                x_lo=x_lo,
                x_hi=x_hi,
                y_lo=y_lo,
                y_hi=y_hi,
            )
            if col == 0:
                ax.set_ylabel(
                    f"{FEAT_LABEL[feature]}\nparity order r",
                    fontsize=viz.FONT_LABEL,
                )
            if row == len(FEATS) - 1:
                ax.set_xlabel("feature order q", fontsize=viz.FONT_LABEL)
    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes,
            shrink=0.84,
            ticks=[0.0, 0.5, 1.0],
            label="within-feature normalized excess energy",
        )
        cbar.outline.set_linewidth(0.8)
    _save(fig, "bernstein_parity_four_panel", out_dir)


def _sample_lollipop(
    rng: np.random.Generator,
    *,
    placement: str,
    structured: bool,
) -> tuple[np.ndarray, np.ndarray]:
    chain = iid_values(
        rng, TRAJECTORY_ORDER, noise_sigma=NOISE_SIGMA,
    ).astype(float)
    leaves = iid_values(
        rng, TRAJECTORY_ORDER - 1, noise_sigma=NOISE_SIGMA,
    ).astype(float)

    if structured:
        parity_vals = parity_values(
            rng,
            TRAJECTORY_ORDER,
            structured=True,
            noise_sigma=NOISE_SIGMA,
        ).astype(float)
        if placement == "chain":
            chain[:] = parity_vals
        elif placement == "fan":
            chain[-1] = parity_vals[0]
            leaves[:] = parity_vals[1:]
        else:
            raise ValueError(f"unknown trajectory placement {placement!r}")
    return chain, leaves


def _lollipop_cloud(
    *,
    seed: int,
    placement: str,
    structured: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chains, leaves = [], []
    for _ in range(TRAJECTORY_SAMPLES):
        chain, leaf = _sample_lollipop(
            rng,
            placement=placement,
            structured=structured,
        )
        chains.append(chain)
        leaves.append(leaf)
    return np.stack(chains), np.stack(leaves)


def _trajectory_color(chain: np.ndarray, leaves: np.ndarray):
    # Use the shared split ancestor as the trajectory label. Its marginal is
    # balanced in the null and both Bernstein placements, unlike tuple means.
    return viz.fate_color(float(chain[-1]), vmax=1.0)


def _branch_curve(
    x0: float,
    x1: float,
    *,
    side: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    y0 = float(TRAJECTORY_ORDER - 1)
    y1 = float(TRAJECTORY_ORDER) + side * TRAJECTORY_BRANCH_Y_SLOT
    u = np.linspace(0.0, 1.0, 18)
    x2 = x1 + side * TRAJECTORY_BRANCH_SLOT
    xc = 0.5 * (x0 + x2) + side * TRAJECTORY_BRANCH_BOW
    yc = 0.5 * (y0 + y1)
    x = (1 - u) ** 2 * x0 + 2 * (1 - u) * u * xc + u ** 2 * x2
    y = (1 - u) ** 2 * y0 + 2 * (1 - u) * u * yc + u ** 2 * y1
    return x, y, x2, y1


def _plot_lollipop_cloud(
    ax,
    chains: np.ndarray,
    leaves: np.ndarray,
    *,
    focus: str,
):
    chain_times = np.arange(TRAJECTORY_ORDER, dtype=float)
    leaf_time = float(TRAJECTORY_ORDER)
    branch_time = float(TRAJECTORY_ORDER - 1)
    if focus == "path":
        chain_alpha, branch_alpha = TRAJECTORY_FG_ALPHA, TRAJECTORY_BG_ALPHA
        chain_lw, branch_lw = TRAJECTORY_LW + 0.05, TRAJECTORY_LW - 0.08
    elif focus == "sibling":
        chain_alpha, branch_alpha = TRAJECTORY_BG_ALPHA, TRAJECTORY_FG_ALPHA
        chain_lw, branch_lw = TRAJECTORY_LW - 0.08, TRAJECTORY_LW + 0.05
    elif focus == "all":
        chain_alpha = branch_alpha = TRAJECTORY_ALPHA
        chain_lw = branch_lw = TRAJECTORY_LW
    else:
        raise ValueError(f"unknown trajectory focus {focus!r}")

    for chain, leaf_vals in zip(chains, leaves):
        color = _trajectory_color(chain, leaf_vals)
        chain_x = chain * TRAJECTORY_CHAIN_SCALE
        leaf_x = leaf_vals * TRAJECTORY_LEAF_SCALE
        ghost = viz.PRESPLIT
        chain_color = color if focus in {"all", "path"} else ghost
        branch_color = color if focus in {"all", "sibling"} else ghost
        ax.plot(
            chain_x,
            chain_times,
            color=chain_color,
            lw=chain_lw,
            alpha=chain_alpha,
            solid_capstyle="round",
            rasterized=True,
        )
        if len(leaf_x) == 1:
            sides = np.array([0.0])
        else:
            sides = np.linspace(-1.0, 1.0, len(leaf_x))
        leaf_x_slots = []
        leaf_y_slots = []
        for leaf, side in zip(leaf_x, sides):
            bx, by, x2, y2 = _branch_curve(
                float(chain_x[-1]), float(leaf), side=float(side),
            )
            leaf_x_slots.append(x2)
            leaf_y_slots.append(y2)
            ax.plot(
                bx,
                by,
                color=branch_color,
                lw=branch_lw,
                alpha=branch_alpha,
                solid_capstyle="round",
                rasterized=True,
            )
        ax.scatter(
            leaf_x_slots,
            leaf_y_slots,
            s=5.0,
            color=branch_color,
            alpha=min(0.65, 2.0 * branch_alpha),
            linewidths=0.0,
            rasterized=True,
        )


def _style_lollipop_axis(ax):
    branch_time = TRAJECTORY_ORDER - 1
    leaf_time = TRAJECTORY_ORDER
    ax.add_patch(Rectangle(
        (-1.50, branch_time - 0.16),
        3.00,
        leaf_time - branch_time + 0.32,
        facecolor=viz.EVENT_TINT,
        edgecolor=viz.EVENT_GUIDE,
        lw=0.55,
        zorder=-3,
    ))
    ax.axvline(0.0, color=viz.EVENT_GUIDE, lw=0.55, alpha=0.65, zorder=-1)
    ax.axhline(branch_time, color=viz.EVENT_GUIDE, lw=0.55, alpha=0.65, zorder=-1)
    ax.set_xlim(-1.55, 1.55)
    ax.set_xticks([-1, 0, 1])
    ax.set_ylim(leaf_time + 0.25, -0.25)
    ax.set_yticks(range(TRAJECTORY_ORDER + 1))
    ax.tick_params(axis="y", length=0)


def render_trajectories(out_dir: Path = S.IMGS):
    clouds = {
        ("chain", True): _lollipop_cloud(seed=9101, placement="chain", structured=True),
        ("null", False): _lollipop_cloud(seed=9100, placement="chain", structured=False),
        ("fan", True): _lollipop_cloud(seed=9201, placement="fan", structured=True),
    }

    viz.apply_rc()
    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.5, 2.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = [
        (axes[0], "independent null", "null", False, "all"),
        (axes[1], "path dependence", "chain", True, "path"),
        (axes[2], "sibling dependence", "fan", True, "sibling"),
    ]
    for ax, title, placement, structured, focus in panels:
        chain, leaves = clouds[(placement, structured)]
        _style_lollipop_axis(ax)
        _plot_lollipop_cloud(ax, chain, leaves, focus=focus)
        ax.set_title(title, fontsize=viz.FONT_TITLE)
        ax.set_xlabel("state coordinate", fontsize=viz.FONT_LABEL)
    axes[0].set_ylabel("time", fontsize=viz.FONT_LABEL)
    _save(fig, "bernstein_parity_trajectories", out_dir)


def _save(fig, stem, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            out_dir / f"{stem}.{ext}",
            bbox_inches="tight",
            dpi=300 if ext == "png" else None,
        )
    plt.close(fig)


def load_summary(path: Path) -> dict[tuple[str, str, int, int], tuple[float, float]]:
    """Read parity_summary.csv back into the keyed (mean, sem) table."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (
            row["placement"], row["feature"],
            int(row["parity_order"]), int(row["feature_order"]),
        ): (float(row["mean"]), float(row["sem"]))
        for row in rows
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="plot the smoke benchmark")
    args = parser.parse_args()
    summary = load_summary(output_dir(args.quick) / "parity_summary.csv")
    out_dir = S.IMGS / "quick" if args.quick else S.IMGS
    render(summary, out_dir)
    render_four_panel(summary, out_dir)
    render_trajectories(out_dir)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
