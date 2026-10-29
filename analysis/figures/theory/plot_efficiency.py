"""Render the Bernstein efficiency benchmark (plot half).

Three panels from the summaries written by compute_efficiency.py: detection
probability against tree count, energy against exact Wasserstein excess on the
same feature samples, and wall-clock scaling of the two discrepancies.

Run from the repository root with

    python -m analysis.figures.theory.plot_efficiency [--quick]

which reads power_summary.csv, energy_wasserstein_scatter.csv and
timing_summary.csv from numerical_results/ (numerical_results/quick/ with
``--quick``) and writes bernstein_efficiency.{pdf,svg,png} to imgs/ (imgs/quick/
with ``--quick``).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from analysis.figures.theory import style as S
from analysis.figures.theory.compute_efficiency import output_dir
from synthetic_evals import viz


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _read_csv(path: Path) -> list[dict]:
    """Read a summary written by compute_efficiency.py with numeric columns restored."""
    with path.open(newline="") as handle:
        return [
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _rows_for(rows: list[dict], **conditions) -> list[dict]:
    return [
        row for row in rows
        if all(row[key] == value for key, value in conditions.items())
    ]


def render(
    power_summary: list[dict], scatter: list[dict], timing_summary: list[dict],
    out_dir: Path = S.IMGS,
) -> dict:
    viz.apply_rc()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    all_orders = sorted(
        {row["feature_order"] for row in power_summary}
        | {row["feature_order"] for row in scatter}
    )
    order_colors = {
        q: color for q, color in zip(
            all_orders,
            [viz.SEQ_CMAP(v) for v in np.linspace(0.35, 0.95, len(all_orders))],
        )
    }

    # (a) Independent-tree count and statistical detectability.
    ax = axes[0]
    orders = sorted({row["feature_order"] for row in power_summary})
    for q in orders:
        color = order_colors[q]
        rows = sorted(_rows_for(power_summary, feature_order=q), key=lambda z: z["tree_count"])
        n = np.asarray([row["tree_count"] for row in rows])
        detection = np.asarray([row["detection_probability"] for row in rows])
        lo = np.asarray([row["detection_lo"] for row in rows])
        hi = np.asarray([row["detection_hi"] for row in rows])
        ax.fill_between(n, lo, hi, color=color, alpha=0.10, lw=0)
        ax.plot(
            n, detection,
            color=color, marker="o", ms=3.4, lw=1.35, label=rf"$q={q}$",
        )
    ax.axhline(0.05, color="0.70", lw=0.7, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("independent trees $n$")
    ax.set_ylabel("detection probability\n(null-calibrated at 5\%)")
    ax.set_title(r"order-3 chain parity")
    ax.legend(title="feature order", ncol=2, handlelength=1.4, columnspacing=0.8)

    # (b) The two discrepancies on exactly the same empirical laws.
    ax = axes[1]
    visible = [
        row for row in scatter
        if row["contains_full_tuple"] and row["structured_fraction"] > 0
    ]
    scatter_orders = sorted({row["feature_order"] for row in scatter})
    for q in scatter_orders:
        rows = _rows_for(scatter, feature_order=q)
        color = order_colors[q]
        ax.scatter(
            [row["energy_excess"] for row in rows],
            [row["wasserstein_excess"] for row in rows],
            s=8, color=color, alpha=0.22, linewidths=0, label=rf"$q={q}$",
        )
        full = [row for row in rows if row["structured_fraction"] == 1.0]
        ax.scatter(
            np.mean([row["energy_excess"] for row in full]),
            np.mean([row["wasserstein_excess"] for row in full]),
            s=25, color=color, edgecolor="white", linewidth=0.6, zorder=3,
        )
    vx = np.asarray([row["energy_excess"] for row in visible], dtype=float)
    vy = np.asarray([row["wasserstein_excess"] for row in visible], dtype=float)
    rho = float(spearmanr(vx, vy).statistic) if vx.size > 1 else float("nan")
    ax.axhline(0.0, color="0.80", lw=0.6)
    ax.axvline(0.0, color="0.80", lw=0.6)
    ax.text(
        0.04, 0.96, rf"Spearman $\rho={rho:.2f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=viz.FONT_ANNOT,
    )
    ax.set_xlabel("energy distance minus\nempirical null mean")
    ax.set_ylabel("Wasserstein distance minus\nempirical null mean")
    ax.set_title("same feature samples")
    ax.legend(title="feature order", ncol=2, loc="lower right", handletextpad=0.3)

    # (c) Discrepancy-only runtime.
    ax = axes[2]
    method_style = {
        "energy": (viz.MODEL, "o", "Energy update"),
        "wasserstein": (viz.GS, "s", r"exact $W_1$"),
    }
    for method in ("energy", "wasserstein"):
        rows = sorted(_rows_for(timing_summary, method=method), key=lambda z: z["sample_count"])
        n = np.asarray([row["sample_count"] for row in rows])
        med = np.asarray([row["median_seconds"] for row in rows])
        lo = np.asarray([row["q25_seconds"] for row in rows])
        hi = np.asarray([row["q75_seconds"] for row in rows])
        color, marker, label = method_style[method]
        ax.fill_between(n, lo, hi, color=color, alpha=0.14, lw=0)
        ax.plot(n, med, color=color, marker=marker, ms=3.4, lw=1.35, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("feature samples $m$")
    ax.set_ylabel("wall time (seconds)")
    ax.set_title("discrepancy computation")
    ax.legend()

    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.18, 1.06, label, transform=ax.transAxes,
            fontsize=viz.FONT_TITLE, fontweight="bold", va="top",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            out_dir / f"bernstein_efficiency.{ext}",
            bbox_inches="tight", dpi=300 if ext == "png" else None,
        )
    plt.close(fig)

    max_n = max(row["sample_count"] for row in timing_summary)
    med = {
        row["method"]: row["median_seconds"]
        for row in timing_summary if row["sample_count"] == max_n
    }
    return {
        "spearman_rho": rho,
        "max_timing_n": max_n,
        "max_timing_ratio": med["wasserstein"] / med["energy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="plot the smoke benchmark")
    args = parser.parse_args()
    source = output_dir(args.quick)
    out_dir = S.IMGS / "quick" if args.quick else S.IMGS
    diagnostics = render(
        _read_csv(source / "power_summary.csv"),
        _read_csv(source / "energy_wasserstein_scatter.csv"),
        _read_csv(source / "timing_summary.csv"),
        out_dir,
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print("wrote", out_dir / "bernstein_efficiency.pdf")


if __name__ == "__main__":
    main()
