"""Diagnostic renders of the seven-node compatible-law search.

These two diagnostic panels are not paper figures.  They show the risk
decomposition and one compatible law found by the search, so a fresh solve
from compute_depth2.py can be checked by eye before the certified results next
to depth2.py are replaced.

Run from the repository root with

    python -m analysis.figures.theory.plot_depth2

which reads the certified depth2_identification_results.json and writes
imgs/depth2_identification_pilot.png and imgs/depth2_compatible_law_pilot.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from analysis.figures.theory import depth2
from analysis.figures.theory import style as S
from analysis.figures.theory.depth2 import (
    EDGES,
    ORBITS,
    risk_arrays,
)


INK = "#24232b"
MUTED = "#77717f"
EDGE = "#cfd5df"
PAPER = "#ffffff"
SELECT = "#6959e8"
ORANGE = "#d17635"
TEAL = "#117a7d"
TEAL_LIGHT = "#dcebea"


def _tree_positions(x: float, y: float, scale: float) -> dict[int, tuple[float, float]]:
    return {
        0: (x, y + 0.28 * scale),
        1: (x - 0.18 * scale, y + 0.12 * scale),
        2: (x + 0.18 * scale, y + 0.12 * scale),
        3: (x - 0.27 * scale, y - 0.08 * scale),
        4: (x - 0.09 * scale, y - 0.08 * scale),
        5: (x + 0.09 * scale, y - 0.08 * scale),
        6: (x + 0.27 * scale, y - 0.08 * scale),
    }


def _curved_edge(ax, left, right, *, color, width, bend=0.07) -> None:
    ax.add_patch(FancyArrowPatch(
        left, right, arrowstyle="-", connectionstyle=f"arc3,rad={bend}",
        linewidth=width, color=color, shrinkA=2.5, shrinkB=2.5,
        capstyle="round", zorder=1,
    ))


def _draw_tree(ax, *, x=0.50, y=0.58, scale=1.0, mode="both") -> None:
    positions = _tree_positions(x, y, scale)
    for parent, child in EDGES:
        left = positions[parent]
        right = positions[child]
        selected = mode in ("path", "both") and child in (1, 3)
        _curved_edge(
            ax, left, right,
            color=SELECT if selected else EDGE,
            width=3.0 if selected else 1.5,
            bend=0.07 if child % 2 else -0.07,
        )
    signs = (1, -1, 1, 1, -1, 1, -1)
    for node, position in positions.items():
        selected = mode in ("path", "both") and node in (0, 1, 3)
        ax.add_patch(plt.Circle(
            position, 0.026 * scale,
            facecolor=ORANGE if signs[node] > 0 else "#347fa1",
            edgecolor=SELECT if selected else PAPER,
            linewidth=2.0 if selected else 0.8,
            zorder=3,
        ))
        ax.text(position[0], position[1], "+" if signs[node] > 0 else "−",
                ha="center", va="center", fontsize=8, color=PAPER,
                fontweight="bold", zorder=4)
    if mode in ("ancestor", "both"):
        for node in (0, 5, 6):
            ax.add_patch(plt.Circle(
                positions[node], 0.036 * scale, facecolor="none",
                edgecolor=ORANGE, linewidth=1.3, linestyle=(0, (2, 1)), zorder=5,
            ))


def _draw_state_tree(ax, state, *, x, y, scale) -> None:
    positions = _tree_positions(x, y, scale)
    for parent, child in EDGES:
        _curved_edge(
            ax, positions[parent], positions[child], color=EDGE,
            width=0.9, bend=0.07 if child % 2 else -0.07,
        )
    for node, position in positions.items():
        ax.add_patch(plt.Circle(
            position, 0.026 * scale,
            facecolor=ORANGE if state[node] > 0 else "#347fa1",
            edgecolor=PAPER, linewidth=0.55, zorder=3,
        ))
        ax.text(position[0], position[1], "+" if state[node] > 0 else "−",
                ha="center", va="center", fontsize=max(5.0, 6.5 * scale),
                color=PAPER, fontweight="bold", zorder=4)


def render_pilot(results: list[dict], out_dir: Path) -> Path:
    """Render the feature tables, the largest compatible gap and the risk maps."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
    })
    rhos = np.asarray([result["rho"] for result in results])
    ns = np.geomspace(2.0, 2.0**24, 321)
    risks = risk_arrays(results, ns)
    if np.max(risks["feature_mean_gap"]) > 2e-7:
        raise RuntimeError("compatible laws do not match the feature embedding")

    fig = plt.figure(figsize=(7.1, 5.05), facecolor=PAPER)

    tree_ax = fig.add_axes((0.065, 0.625, 0.400, 0.285))
    _draw_tree(tree_ax, x=0.28, y=0.53, scale=0.67, mode="path")
    _draw_tree(tree_ax, x=0.74, y=0.53, scale=0.67, mode="ancestor")
    tree_ax.set_xlim(0, 1)
    tree_ax.set_ylim(0, 1)
    tree_ax.axis("off")
    tree_ax.text(0.28, 0.92, "path", ha="center", va="center",
                 fontsize=9, color=INK)
    tree_ax.text(0.74, 0.92, "common ancestor", ha="center", va="center",
                 fontsize=9, color=INK)
    tree_ax.text(0.28, 0.12, "levels 0, 1, 2", ha="center", va="center",
                 fontsize=7, color=MUTED)
    tree_ax.text(0.74, 0.12, "pairs 0–1, 0–2, 1–2", ha="center", va="center",
                 fontsize=7, color=MUTED)
    tree_ax.text(-0.10, 1.06, "a", transform=tree_ax.transAxes,
                 ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    tree_ax.set_title("four separate size-three feature tables", pad=2, color=INK)

    gap_ax = fig.add_axes((0.575, 0.625, 0.350, 0.285))
    gap_ax.plot(rhos, risks["gap"], color=TEAL, linewidth=1.8)
    gap_ax.fill_between(rhos, 0, risks["gap"], color=TEAL_LIGHT, alpha=0.65)
    gap_ax.set_xlim(0, 1)
    gap_ax.set_ylim(bottom=0)
    gap_ax.set_xlabel(r"edge correlation  $\rho$")
    gap_ax.set_ylabel("largest found whole-tree difference\nwith the same four tables", labelpad=3)
    gap_ax.spines[["top", "right"]].set_visible(False)
    gap_ax.text(-0.12, 1.06, "b", transform=gap_ax.transAxes,
                ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    gap_ax.set_title("42 configurations  ·  13 feature directions  ·  28 left open",
                     pad=5, fontsize=8, color=INK)

    components = (
        ("c", "largest found discrepancy$^2$", risks["unresolved_ratio"]),
        ("d", "feature sampling variance", risks["sampling_ratio"]),
        ("e", "MSE", risks["total_ratio"]),
    )
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "depth2-risk", ("#365f73", "#a9bdc5", "#f0ece5", "#d8b0a2", "#a75f50"),
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=4)
    rho_mesh, n_mesh = np.meshgrid(rhos, ns, indexing="ij")
    image = None
    map_lefts = (0.080, 0.380, 0.680)
    map_bottom, map_width, map_height = 0.105, 0.250, 0.375
    for index, ((letter, title, values), left) in enumerate(zip(components, map_lefts)):
        ax = fig.add_axes((left, map_bottom, map_width, map_height))
        image = ax.pcolormesh(
            rho_mesh, n_mesh,
            np.clip(np.log2(np.maximum(values, 2.0**-16)), -8, 4),
            cmap=cmap, norm=norm, shading="nearest", rasterized=True,
        )
        ax.set_xlim(0, 1)
        ax.set_yscale("log", base=2)
        ax.set_ylim(2, 2**24)
        ax.set_xticks((0, 0.5, 1.0))
        ax.set_yticks((2, 2**6, 2**12, 2**18, 2**24))
        if index == 0:
            ax.set_yticklabels((r"$2^1$", r"$2^6$", r"$2^{12}$", r"$2^{18}$", r"$2^{24}$"))
            ax.set_ylabel(r"trees per population  $n$")
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
        if index == 1:
            ax.set_xlabel(r"edge correlation  $\rho$")
        ax.set_title(title, pad=6, color=INK)
        ax.text(-0.22, 1.10, letter, transform=ax.transAxes,
                ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
        ax.grid(color=PAPER, linewidth=0.45, alpha=0.5)
        if letter == "e":
            contour = ax.contour(
                rho_mesh, n_mesh, np.log2(np.maximum(values, 2.0**-16)),
                levels=(0.0,), colors=(INK,), linewidths=(0.9,),
            )
            if contour.allsegs[0]:
                ax.clabel(contour, fmt={0.0: "equal MSE"}, fontsize=6.5, inline=True)

    maps_y = map_bottom + map_height / 2
    fig.text(0.355, maps_y, "+", ha="center", va="center", fontsize=18, color=MUTED)
    fig.text(0.655, maps_y, "=", ha="center", va="center", fontsize=18, color=MUTED)

    colorbar_ax = fig.add_axes((0.947, map_bottom, 0.012, map_height))
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks((-8, -4, 0, 4))
    colorbar.set_ticklabels((r"$2^{-8}$", r"$2^{-4}$", "1", r"$\geq16$"))
    colorbar.ax.tick_params(labelsize=7, width=0.45, length=2)
    colorbar.ax.set_title("÷ whole-tree\nMSE", fontsize=7, color=MUTED, pad=4)

    fig.suptitle(
        "Compatible discrepancy$^2$  +  feature sampling variance  =  feature MSE",
        x=0.51, y=0.982, fontsize=10.5, color=INK,
    )
    path = out_dir / "depth2_identification_pilot.png"
    fig.savefig(path, dpi=220, facecolor=PAPER)
    plt.close(fig)
    return path


def render_compatible_law(results: list[dict], out_dir: Path, rho: float = 0.5) -> Path:
    """Show the sparse orbit law for one representative optimization."""
    result = min(results, key=lambda item: abs(item["rho"] - rho))
    law = result["law"]
    support = np.flatnonzero(law > 1e-8)
    support = support[np.argsort(law[support])[::-1]]

    fig, ax = plt.subplots(figsize=(7.1, 2.55), facecolor=PAPER)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "one compatible law found by the search", ha="left", va="top",
            fontsize=13, color=INK)
    ax.text(0.98, 0.94,
            rf"$\rho={result['rho']:.2f}$",
            ha="right", va="top", fontsize=9, color=MUTED)

    columns = 5
    for position, orbit_index in enumerate(support):
        row = position // columns
        column = position % columns
        x = 0.11 + 0.195 * column
        y = 0.61 - 0.36 * row
        _draw_state_tree(
            ax, ORBITS[int(orbit_index)][0], x=x, y=y,
            scale=0.30,
        )
        mass = law[int(orbit_index)]
        denominator = 32
        numerator = int(round(mass * denominator))
        exact = abs(mass - numerator / denominator) < 2e-6
        label = rf"${numerator}/{denominator}$" if exact else rf"${mass:.3f}$"
        ax.text(x, y - 0.105, label, ha="center", va="top",
                fontsize=7.5, color=INK)
    path = out_dir / "depth2_compatible_law_pilot.png"
    fig.savefig(path, dpi=220, facecolor=PAPER, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def main() -> None:
    results = depth2.load_results()
    S.IMGS.mkdir(parents=True, exist_ok=True)
    print("wrote", render_pilot(results, S.IMGS))
    print("wrote", render_compatible_law(results, S.IMGS))


if __name__ == "__main__":
    main()
