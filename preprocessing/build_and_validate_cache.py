"""Build the mouse-embryo TreeRecord cache and emit validation plots.

For each tree in the treedata file the script either loads a cached
TreeRecord pickle (if present and ``--force`` is not set) or builds and
saves one with the current indexer in ``datasets/tree_record.py``. It
then aggregates per-(tree, bin) statistics and writes a CSV plus a small
set of validation figures to ``--output_dir``.

Use ``--force`` after changing the indexer logic so stale pickles get
rebuilt rather than silently reloaded.

Run from the repo root:
    python preprocessing/build_and_validate_cache.py --force
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import treedata  # noqa: E402

from datasets.tree_record import TreeRecord  # noqa: E402

logger = logging.getLogger(__name__)


def _cache_prefix(trees_filename: str) -> str:
    return f"{trees_filename}_"


def _list_cached(cache_dir: Path, trees_filename: str) -> dict[str, Path]:
    prefix = _cache_prefix(trees_filename)
    out: dict[str, Path] = {}
    if not cache_dir.is_dir():
        return out
    for child in cache_dir.iterdir():
        if child.name.startswith(prefix) and child.name.endswith(".pkl"):
            name = child.name[len(prefix):-len(".pkl")]
            out[name] = child
    return out


def build_records(
    data_dir: str,
    trees_filename: str,
    time_bins: list[float],
    max_size: int,
    cache_dir: Path,
    force: bool,
    tree_names: list[str] | None = None,
) -> dict[str, TreeRecord]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = _cache_prefix(trees_filename)

    if force:
        for name, path in _list_cached(cache_dir, trees_filename).items():
            if tree_names is not None and name not in set(tree_names):
                continue
            path.unlink()
            logger.info("removed stale cache %s", path.name)

    cached = _list_cached(cache_dir, trees_filename)
    if tree_names is not None:
        wanted = set(tree_names)
        cached = {k: v for k, v in cached.items() if k in wanted}

    records: dict[str, TreeRecord] = {}
    for name, path in cached.items():
        records[name] = TreeRecord.load(str(path))

    td = treedata.read_h5td(os.path.join(data_dir, trees_filename))
    available = list(td.obst.keys())
    if tree_names is not None:
        available = [n for n in available if n in set(tree_names)]
    missing = [n for n in available if n not in records]
    if missing:
        logger.info("building %d records (cache miss)", len(missing))
    for name in missing:
        rec = TreeRecord(
            td, time_bins=time_bins, max_size=max_size,
            tree_names=[name],
        )
        cache_path = cache_dir / f"{prefix}{name}.pkl"
        rec.save(str(cache_path))
        records[name] = rec

    return records


def compute_stats(
    records: dict[str, TreeRecord], time_bins: list[float],
) -> pd.DataFrame:
    rows: list[dict] = []
    for name, rec in records.items():
        if not rec.trees:
            logger.warning("record %s has no associated tree; skipping", name)
            continue
        tree = next(iter(rec.trees.values()))
        total_leaves = sum(1 for n in tree.nodes if tree.out_degree(n) == 0)
        n_nodes = tree.number_of_nodes()
        leaf_time = rec.leaf_time

        for b in time_bins:
            pool = rec.subtree_index.get(b, [])
            n_roots = len(pool)
            if n_roots == 0:
                rows.append({
                    "tree": name, "bin": b, "leaf_time": leaf_time,
                    "total_leaves": total_leaves, "n_nodes": n_nodes,
                    "n_roots": 0, "mean_size": np.nan, "median_size": np.nan,
                    "max_size": 0, "mean_depth": np.nan, "max_depth": 0,
                    "leaf_coverage": 0.0, "singleton_frac": np.nan,
                    "root_time_min": np.nan, "root_time_max": np.nan,
                    "root_time_offset_mean": np.nan,
                })
                continue
            sizes = np.array([s["size"] for s in pool], dtype=np.int64)
            depths = np.array([s["depth"] for s in pool], dtype=np.int64)
            rtimes = np.array([s["root_time"] for s in pool], dtype=np.float64)
            covered = int(sizes.sum())  # snapshot lineages are disjoint per bin
            rows.append({
                "tree": name, "bin": b, "leaf_time": leaf_time,
                "total_leaves": total_leaves, "n_nodes": n_nodes,
                "n_roots": n_roots,
                "mean_size": float(sizes.mean()),
                "median_size": float(np.median(sizes)),
                "max_size": int(sizes.max()),
                "mean_depth": float(depths.mean()),
                "max_depth": int(depths.max()),
                "leaf_coverage": float(covered / total_leaves),
                "singleton_frac": float((sizes == 1).mean()),
                "root_time_min": float(rtimes.min()),
                "root_time_max": float(rtimes.max()),
                "root_time_offset_mean": float((b - rtimes).mean()),
            })
    return pd.DataFrame(rows)


def _tree_order(stats: pd.DataFrame) -> list[str]:
    first = stats.groupby("tree", as_index=False).first()
    return list(first.sort_values(["leaf_time", "tree"])["tree"])


def _leaf_time_norm(stats: pd.DataFrame) -> mcolors.Normalize:
    leaf_times = stats["leaf_time"].dropna()
    if leaf_times.empty:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    return mcolors.Normalize(vmin=float(leaf_times.min()), vmax=float(leaf_times.max()))


def plot_heatmap_n_roots(
    stats: pd.DataFrame, time_bins: list[float], out_path: Path,
) -> None:
    order = _tree_order(stats)
    pivot = (
        stats.pivot(index="tree", columns="bin", values="n_roots")
        .reindex(index=order, columns=time_bins)
        .fillna(0)
    )
    fig_h = max(4.0, 0.25 * len(order) + 2.0)
    fig, ax = plt.subplots(figsize=(1.1 + 1.0 * len(time_bins), fig_h))
    vmax = max(1, int(pivot.values.max()))
    norm = mcolors.LogNorm(vmin=1, vmax=vmax)
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="viridis", norm=norm,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(time_bins)))
    ax.set_xticklabels([f"{b:.2f}" for b in time_bins])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("time bin")
    ax.set_ylabel("tree (ordered by leaf_time)")
    ax.set_title("n_roots per (tree, bin)")
    cb = plt.colorbar(im, ax=ax, label="n_roots (log scale)")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _line_plot_per_tree(
    stats: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    *,
    log_y: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    order = _tree_order(stats)
    norm = _leaf_time_norm(stats)
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for name in order:
        sub = stats[stats["tree"] == name].sort_values("bin")
        if sub.empty:
            continue
        color = cmap(norm(sub["leaf_time"].iloc[0]))
        ax.plot(
            sub["bin"], sub[metric],
            marker="o", linewidth=1.2, alpha=0.85, color=color,
        )
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("time bin")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, label="tree leaf_time")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_root_time_offsets(
    stats: pd.DataFrame, time_bins: list[float], out_path: Path,
) -> None:
    """For each (tree, bin) row, plot the gap between the bin and the
    mean actual root time of the indexed subtrees. With the snapshot
    indexer this is always >= 0 (roots sit at-or-before the bin)."""
    norm = _leaf_time_norm(stats)
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for name in _tree_order(stats):
        sub = stats[(stats["tree"] == name) & (stats["n_roots"] > 0)]
        if sub.empty:
            continue
        color = cmap(norm(sub["leaf_time"].iloc[0]))
        ax.plot(
            sub["bin"], sub["root_time_offset_mean"],
            marker="o", linewidth=1.2, alpha=0.85, color=color,
        )
    ax.axhline(0.0, color="k", linewidth=0.5, linestyle=":")
    ax.set_xlabel("time bin")
    ax.set_ylabel("mean (bin - root_time) over indexed roots")
    ax.set_title("How far below the bin do indexed roots actually sit?")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, label="tree leaf_time")
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def report_sanity(stats: pd.DataFrame) -> None:
    """Log warnings for empty pools and low leaf coverage, then a per-bin summary."""
    # 1) bins below or equal to leaf_time should always have content.
    below = stats[stats["bin"] <= stats["leaf_time"]]
    empty = below[below["n_roots"] == 0]
    if not empty.empty:
        logger.warning(
            "EMPTY pools at bins <= leaf_time -- max_size likely too small:"
            "\n%s", empty[["tree", "bin", "leaf_time", "total_leaves"]],
        )

    # 2) leaf coverage = 1 only when max_size doesn't bite.
    under = stats[
        (stats["bin"] <= stats["leaf_time"])
        & (stats["leaf_coverage"] < 0.99)
    ]
    if not under.empty:
        worst = under.nsmallest(10, "leaf_coverage")[
            ["tree", "bin", "leaf_time", "leaf_coverage", "n_roots", "mean_size"]
        ]
        logger.warning(
            "leaf coverage <99%% in %d (tree, bin) cells. Worst 10:\n%s",
            len(under), worst,
        )

    # 3) Per-bin summary across all trees.
    per_bin = stats.groupby("bin").agg(
        trees=("tree", "nunique"),
        nonempty_trees=("n_roots", lambda v: int((v > 0).sum())),
        n_roots_sum=("n_roots", "sum"),
        mean_size_avg=("mean_size", "mean"),
        leaf_coverage_avg=("leaf_coverage", "mean"),
        singleton_frac_avg=("singleton_frac", "mean"),
    ).round(3)
    logger.info("per-bin summary across trees:\n%s", per_bin.to_string())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../data/embryos")
    p.add_argument("--trees_filename", default="final_topology.h5td")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument(
        "--output_dir", default="preprocessing/cache_validation",
        help="where stats CSV and plots are written",
    )
    p.add_argument(
        "--time_bins", type=float, nargs="+",
        default=[7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    )
    p.add_argument("--max_size", type=int, default=128)
    p.add_argument(
        "--tree_names", type=str, nargs="+", default=None,
        help="restrict to these tree names",
    )
    p.add_argument(
        "--force", action="store_true",
        help="delete matching cache pickles and rebuild",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    records = build_records(
        args.data_dir, args.trees_filename, args.time_bins,
        args.max_size, cache_dir, args.force, tree_names=args.tree_names,
    )
    logger.info("loaded/built %d records", len(records))

    stats = compute_stats(records, args.time_bins)
    csv_path = out_dir / "stats.csv"
    stats.to_csv(csv_path, index=False)
    logger.info("wrote stats to %s", csv_path)

    plot_heatmap_n_roots(
        stats, args.time_bins, out_dir / "heatmap_n_roots.png",
    )
    _line_plot_per_tree(
        stats, "n_roots",
        ylabel="n_roots (log)",
        title="n_roots across bins, one line per tree",
        out_path=out_dir / "lines_n_roots.png",
        log_y=True,
    )
    _line_plot_per_tree(
        stats, "mean_size",
        ylabel="mean subtree size (leaves, log)",
        title="mean subtree size across bins, one line per tree",
        out_path=out_dir / "lines_mean_size.png",
        log_y=True,
    )
    _line_plot_per_tree(
        stats, "leaf_coverage",
        ylabel="fraction of tree leaves covered",
        title="leaf coverage per (tree, bin) -- snapshot ancestor under max_size",
        out_path=out_dir / "lines_leaf_coverage.png",
        ylim=(-0.02, 1.05),
    )
    _line_plot_per_tree(
        stats, "mean_depth",
        ylabel="mean subtree depth (nodes)",
        title="mean subtree depth across bins, one line per tree",
        out_path=out_dir / "lines_mean_depth.png",
    )
    _line_plot_per_tree(
        stats, "singleton_frac",
        ylabel="fraction of size-1 subtrees",
        title="singleton fraction per (tree, bin)",
        out_path=out_dir / "lines_singleton_frac.png",
        ylim=(-0.02, 1.05),
    )
    plot_root_time_offsets(
        stats, args.time_bins, out_dir / "lines_root_time_offset.png",
    )

    report_sanity(stats)


if __name__ == "__main__":
    main()
