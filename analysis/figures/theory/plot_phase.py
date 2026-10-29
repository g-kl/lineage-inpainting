"""Render the exact all-times-feature MSE decomposition."""

from __future__ import annotations

import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from analysis.figures.theory import style as S
from analysis.figures.theory.exact import (
    PHASE_FEATURE_ORDER,
    PHASE_PARITY_ORDER,
)


WIDTH, HEIGHT = 6.30, 2.25
N_GRID = np.geomspace(2, 2**30, 401)


def _load() -> dict[str, np.ndarray]:
    path = S.NR / "cell_phase_exact.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row for row in rows
        if row["feature"] == "all_times"
        and int(row["parity_order"]) == PHASE_PARITY_ORDER
        and int(row["feature_order"]) == PHASE_FEATURE_ORDER
    ]
    if not rows:
        raise RuntimeError("the exact phase table does not contain its focal case")
    return {
        field: np.asarray([float(row[field]) for row in rows])
        for field in rows[0]
        if field not in ("feature", "interaction")
    }


def _variance(
    ns: np.ndarray,
    leading: np.ndarray,
    within: np.ndarray,
    cross: np.ndarray,
) -> np.ndarray:
    return (
        4.0 * leading[:, None] / ns[None, :]
        + 2.0 * within[:, None] / (ns[None, :] * (ns[None, :] - 1.0))
        + 4.0 * cross[:, None] / ns[None, :] ** 2
    )


def _configure_heatmap(
    ax,
    *,
    show_y: bool,
    show_x: bool,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_xticks((0, 0.5, 1.0))
    ax.set_yscale("log", base=2)
    ax.set_ylim(2, 2**30)
    ticks = (2, 2**5, 2**9, 2**13, 2**17, 2**21, 2**25, 2**29)
    ax.set_yticks(
        ticks,
        (r"$2^1$", r"$2^5$", r"$2^9$", r"$2^{13}$",
         r"$2^{17}$", r"$2^{21}$", r"$2^{25}$", r"$2^{29}$"),
    )
    if show_x:
        ax.set_xlabel(r"interaction strength  $\theta$")
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False, length=0)
    if show_y:
        ax.set_ylabel(r"trees per population  $n$")
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)
    ax.grid(color="#ffffff", linewidth=0.45, alpha=0.45)
    ax.set_axisbelow(False)


def _boundary_at(
    ratio: np.ndarray,
    theta_grid: np.ndarray,
    n_grid: np.ndarray,
    theta: float,
) -> tuple[float, float]:
    theta_index = int(np.argmin(np.abs(theta_grid - theta)))
    n_index = int(np.argmin(np.abs(np.log2(ratio[theta_index]))))
    return float(theta_grid[theta_index]), float(n_grid[n_index])


def render(out_dir=None):
    S.apply_rc()
    data = _load()
    theta_grid = data["theta"]
    full_variance = _variance(
        N_GRID,
        data["full_leading"],
        data["full_within"],
        data["full_cross"],
    )
    feature_variance = _variance(
        N_GRID,
        data["feature_leading"],
        data["feature_within"],
        data["feature_cross"],
    )
    unresolved_ratio = data["squared_unresolved"][:, None] / full_variance
    variance_ratio = feature_variance / full_variance
    total_ratio = unresolved_ratio + variance_ratio

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    heatmap_lefts = (0.085, 0.385, 0.685)
    # Preserve the maps' physical height while removing the unused lower
    # margin that otherwise opens a gap before panels e--f.
    bottom, heatmap_width, height = 0.180, 0.235, 0.616
    theta_mesh, n_mesh = np.meshgrid(theta_grid, N_GRID, indexing="ij")
    cmap = S.RATIO_CMAP
    norm = mpl.colors.TwoSlopeNorm(
        vmin=S.RATIO_LOG_MIN,
        vcenter=0,
        vmax=S.RATIO_LOG_MAX,
    )
    heatmaps = (
        ("b", r"unresolved energy$^2$", unresolved_ratio),
        ("c", "sampling variance", variance_ratio),
        ("d", "MSE", total_ratio),
    )
    image = None
    axes = []
    for index, ((letter, title, ratio), left) in enumerate(
        zip(heatmaps, heatmap_lefts),
    ):
        fig.text(
            left - 0.060,
            0.984,
            letter,
            ha="left",
            va="top",
            fontsize=S.FS_PANEL,
            fontweight="bold",
            color=S.INK,
        )
        fig.text(
            left - 0.015,
            0.951,
            title,
            ha="left",
            va="top",
            fontsize=S.FS_TITLE,
            color=S.INK,
        )
        ax = fig.add_axes((left, bottom, heatmap_width, height))
        log_ratio = np.log2(np.maximum(ratio, 2**-20))
        image = ax.pcolormesh(
            theta_mesh,
            n_mesh,
            np.clip(log_ratio, S.RATIO_LOG_MIN, S.RATIO_LOG_MAX),
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        _configure_heatmap(
            ax,
            show_y=index == 0,
            show_x=index == 1,
        )
        axes.append(ax)

    # Keep the pointwise decomposition visible at the map level, as in the
    # Ising companion figure.
    fig.text(
        0.347,
        bottom + height / 2,
        "+",
        ha="center",
        va="center",
        fontsize=18,
        color=S.MUTED,
    )
    fig.text(
        0.657,
        bottom + height / 2,
        "=",
        ha="center",
        va="center",
        fontsize=18,
        color=S.MUTED,
    )

    contour = axes[-1].contour(
        theta_mesh,
        n_mesh,
        np.log2(total_ratio),
        levels=(0,),
        colors=(S.INK,),
        linewidths=(0.9,),
    )
    axes[-1].clabel(
        contour,
        fmt={0.0: "equal MSE"},
        manual=(_boundary_at(total_ratio, theta_grid, N_GRID, 0.65),),
        inline=True,
        inline_spacing=1,
        fontsize=S.FS_SUB,
    )

    _, crossover = _boundary_at(total_ratio, theta_grid, N_GRID, 1.0)
    axes[2].plot(
        1.0,
        crossover,
        marker="^",
        markersize=3.5,
        markerfacecolor=S.PAPER,
        markeredgecolor=S.INK,
        markeredgewidth=0.8,
        clip_on=False,
        zorder=5,
    )

    colorbar_ax = fig.add_axes((0.929, bottom, 0.012, height))
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks(S.RATIO_TICKS)
    colorbar.set_ticklabels(S.RATIO_TICKLABELS)
    colorbar.ax.tick_params(labelsize=S.FS_SUB, width=0.45, length=1.8)
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.set_title(
        "÷ whole-tree\nMSE",
        fontsize=S.FS_SUB,
        color=S.MUTED,
        pad=3,
    )

    result = S.save_svg(fig, "bcd_phase.svg")
    plt.close(fig)
    return result


if __name__ == "__main__":
    path, _ = render()
    print("wrote", path)
