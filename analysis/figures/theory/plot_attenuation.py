"""Render exact Bernstein omission and attenuation across feature order."""

from __future__ import annotations

import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from analysis.figures.theory import style as S


WIDTH, HEIGHT = 6.30, 1.82
R_ORDERS = (2, 3, 4)
Q_ORDERS = (2, 3, 4)
ZERO_FILL = "#f1f0f4"
BOUNDARY = "#6b5ed8"
COEFFICIENT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "bernstein_raw_coefficient",
    ("#dcebea", "#72aaa7", "#0f6b6e"),
)


def _load() -> list[dict]:
    path = S.NR / "attenuation_exact.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["parity_order"] = int(row["parity_order"])
        row["feature_order"] = int(row["feature_order"])
        row["raw_discrepancy_ratio"] = float(row["raw_discrepancy_ratio"])
    return rows


def _matrix(
    rows: list[dict],
    *,
    feature: str,
    interaction: str,
) -> np.ndarray:
    values = np.full((len(R_ORDERS), len(Q_ORDERS)), np.nan)
    lookup = {
        (row["parity_order"], row["feature_order"]): row["raw_discrepancy_ratio"]
        for row in rows
        if row["feature"] == feature and row["interaction"] == interaction
    }
    for row_index, r in enumerate(R_ORDERS):
        for column_index, q in enumerate(Q_ORDERS):
            values[row_index, column_index] = lookup[(r, q)]
    return values


def _fraction(value: float) -> str:
    fractions = {
        0.0: "0",
        1 / 12: r"$1/12$",
        1 / 8: r"$1/8$",
        1 / 4: r"$1/4$",
        3 / 8: r"$3/8$",
    }
    for target, label in fractions.items():
        if np.isclose(value, target, atol=2e-14):
            return label
    raise RuntimeError(f"unformatted Bernstein coefficient {value}")


def _draw_matrix(
    ax,
    values: np.ndarray,
    *,
    show_y: bool,
) -> None:
    """Show the exact coefficient for every interaction/feature order pair."""
    norm = mpl.colors.Normalize(vmin=0.0, vmax=3 / 8)
    masked = np.ma.masked_where(values == 0, values)
    ax.set_facecolor(ZERO_FILL)
    ax.imshow(masked, cmap=COEFFICIENT_CMAP, norm=norm, aspect="equal")
    for row_index, r in enumerate(R_ORDERS):
        for column_index, q in enumerate(Q_ORDERS):
            value = float(values[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                _fraction(value),
                ha="center",
                va="center",
                fontsize=S.FS_LABEL,
                color=S.PAPER if norm(value) > 0.48 else S.INK,
                fontweight="bold" if value else "normal",
            )
    ax.plot(
        (-0.5, 0.5, 0.5, 1.5, 1.5, 2.5),
        (0.5, 0.5, 1.5, 1.5, 2.5, 2.5),
        color=BOUNDARY,
        linewidth=0.9,
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.set_xticks(range(len(Q_ORDERS)), [rf"${q}$" for q in Q_ORDERS])
    ax.set_yticks(range(len(R_ORDERS)), [rf"${r}$" for r in R_ORDERS])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlabel(r"feature order  $q$", labelpad=3)
    if show_y:
        ax.set_ylabel(r"interaction order  $r$", labelpad=4)
    else:
        ax.set_yticklabels(())
    for spine in ax.spines.values():
        spine.set_visible(False)


def render(out_dir=None):
    S.apply_rc()
    rows = _load()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))

    panels = (
        ("e", "path · all selected times", "path", "depth", 0.095),
        ("f", "common ancestor · all time pairs", "ca", "breadth", 0.555),
    )
    for letter, title, feature, interaction, left in panels:
        fig.text(
            left - 0.060,
            0.955,
            letter,
            ha="left",
            va="top",
            fontsize=S.FS_PANEL,
            fontweight="bold",
            color=S.INK,
        )
        fig.text(
            left - 0.012,
            0.918,
            title,
            ha="left",
            va="top",
            fontsize=S.FS_TITLE,
            color=S.INK,
        )
        values = _matrix(rows, feature=feature, interaction=interaction)
        ax = fig.add_axes((left, 0.155, 0.315, 0.66))
        _draw_matrix(ax, values, show_y=feature == "path")

    colorbar_ax = fig.add_axes((0.910, 0.155, 0.014, 0.66))
    scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0.0, vmax=3 / 8),
        cmap=COEFFICIENT_CMAP,
    )
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_ticks((0, 1 / 8, 1 / 4, 3 / 8))
    colorbar.set_ticklabels(("0", "1/8", "1/4", "3/8"))
    colorbar.ax.tick_params(labelsize=S.FS_SUB, width=0.45, length=1.8)
    colorbar.outline.set_linewidth(0.45)
    colorbar.ax.set_title(
        "feature /\nwhole tree",
        fontsize=S.FS_SUB,
        color=S.MUTED,
        pad=3,
    )

    result = S.save_svg(fig, "ef_feature_laws.svg")
    plt.close(fig)
    return result


if __name__ == "__main__":
    path, _ = render()
    print("wrote", path)
