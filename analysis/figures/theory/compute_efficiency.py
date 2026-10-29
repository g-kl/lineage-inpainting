"""Statistical and computational trade-offs for Bernstein feature laws (compute half).

This benchmark complements :mod:`analysis.figures.theory.compute_parity`.  That
benchmark shows *which* feature orders and tree geometries reveal a parity
constraint.  Here we isolate two consequences of using low-order
feature laws:

1. Statistical: hold an order-3 chain parity fixed, append independent nuisance
   states to raise the feature order ``q``, and measure how many independent
   trees are needed for a null-calibrated energy-distance test to detect it.
2. Computational: compare energy distance with exact empirical 1-Wasserstein
   distance on the same equally weighted feature samples.  The latter is the
   minimum-cost assignment under Euclidean cost.

The figure rendered by plot_efficiency.py has three panels: detection probability versus tree
count, energy versus Wasserstein excess across parity mixtures, and
wall-clock scaling.  Inputs are generated before either discrepancy is timed.

Run from the repository root with

    python -m analysis.figures.theory.compute_efficiency [--quick] [--force]

which writes the CSV summaries and config.json to numerical_results/
(numerical_results/quick/ for the smoke run) and keeps a pickle cache of the
raw results beside them; plot_efficiency.py renders the figure from the CSVs.
Use ``--quick`` for a smoke run and ``--force`` to ignore a matching cache.
Every cache filename is keyed by the full numerical configuration, so changing
the constants below cannot silently reuse results from another design.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from analysis.figures.theory import style as S

FULL_CONFIG = {
    "seed": 73_091,
    "noise_sigma": 0.0,
    "power_parity_order": 3,
    "power_feature_orders": [2, 3, 4, 5, 6],
    "power_tree_counts": [8, 16, 32, 64, 128, 256, 512],
    "power_reps": 256,
    "scatter_parity_order": 3,
    "scatter_feature_orders": [2, 3, 4, 5, 6],
    "scatter_tree_count": 256,
    "scatter_calibration_reps": 96,
    "scatter_reps": 64,
    "scatter_fractions": [0.0, 0.25, 0.5, 0.75, 1.0],
    "timing_feature_order": 6,
    "timing_parity_order": 3,
    "timing_sample_counts": [32, 64, 128, 256, 512, 1024, 2048, 4096],
    "timing_reps": 7,
}

QUICK_OVERRIDES = {
    "power_tree_counts": [16, 64],
    "power_reps": 6,
    "scatter_feature_orders": [2, 3, 4, 5, 6],
    "scatter_tree_count": 32,
    "scatter_calibration_reps": 4,
    "scatter_reps": 3,
    "scatter_fractions": [0.0, 0.5, 1.0],
    "timing_sample_counts": [16, 32, 64],
    "timing_reps": 3,
}


def output_dir(quick: bool) -> Path:
    """Where this benchmark's numerical results live."""
    return S.NR / "quick" if quick else S.NR


def _config(*, quick: bool) -> dict:
    cfg = dict(FULL_CONFIG)
    if quick:
        cfg.update(QUICK_OVERRIDES)
    return cfg


def _cache_path(cfg: dict) -> Path:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return S.NR / f"efficiency_cache_{digest}.pkl"


def _parity_features(
    rng: np.random.Generator,
    n: int,
    *,
    parity_order: int,
    feature_order: int,
    structured_fraction: float,
    noise_sigma: float,
) -> np.ndarray:
    """Draw a chain-feature law with optional Bernstein parity.

    The first ``parity_order`` states carry the parity constraint in structured
    rows; any states beyond them are independent nuisance variables.  When
    ``feature_order < parity_order``, the returned strict subset is exactly iid,
    as required by the Bernstein construction.
    """
    if parity_order < 2:
        raise ValueError("parity_order must be at least 2")
    if feature_order < 1:
        raise ValueError("feature_order must be positive")
    if not 0.0 <= structured_fraction <= 1.0:
        raise ValueError("structured_fraction must lie in [0, 1]")

    width = max(parity_order, feature_order)
    values = rng.choice([-1.0, 1.0], size=(n, width)).astype(np.float64)
    n_structured = int(round(structured_fraction * n))
    if n_structured:
        rows = rng.permutation(n)[:n_structured]
        values[rows, parity_order - 1] = np.prod(
            values[rows, : parity_order - 1], axis=1,
        )
    values = values[:, :feature_order]
    if noise_sigma > 0:
        values = values + rng.normal(scale=noise_sigma, size=values.shape)
    return values


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Biased empirical energy distance under Euclidean ground distance."""
    cross = cdist(x, y, metric="euclidean").mean()
    within_x = cdist(x, x, metric="euclidean").mean()
    within_y = cdist(y, y, metric="euclidean").mean()
    return float(max(0.0, 2.0 * cross - within_x - within_y))


def energy_update_terms(reference: np.ndarray, model: np.ndarray) -> float:
    """Terms of empirical energy distance that vary with a fixed reference.

    The reference--reference average is constant during model optimization, so
    timing an update only requires the cross average and the model--model
    average.  The returned value differs from :func:`energy_distance` by that
    fixed constant and is used only by the timing benchmark.
    """
    cross = cdist(reference, model, metric="euclidean").mean()
    within_model = cdist(model, model, metric="euclidean").mean()
    return float(2.0 * cross - within_model)


def wasserstein_assignment(x: np.ndarray, y: np.ndarray) -> float:
    """Exact empirical ``W_1`` for equal-size, equally weighted samples."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("assignment W1 requires equal sample counts")
    cost = cdist(x, y, metric="euclidean")
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


def _energy_prefix_grid(
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_orders: list[int],
    sample_counts: list[int],
) -> dict[tuple[int, int], float]:
    """Energy distances for nested state and sample prefixes.

    Building the three squared-distance matrices one state at a time avoids
    recomputing the same lower-order Euclidean distances for every ``q``.
    """
    n_max = max(sample_counts)
    q_max = max(feature_orders)
    x, y = x[:n_max, :q_max], y[:n_max, :q_max]
    cross_sq = np.zeros((n_max, n_max), dtype=np.float64)
    within_x_sq = np.zeros_like(cross_sq)
    within_y_sq = np.zeros_like(cross_sq)
    wanted = set(feature_orders)
    out = {}

    for j in range(q_max):
        xj, yj = x[:, j], y[:, j]
        delta = xj[:, None] - yj[None, :]
        cross_sq += delta * delta
        delta = xj[:, None] - xj[None, :]
        within_x_sq += delta * delta
        delta = yj[:, None] - yj[None, :]
        within_y_sq += delta * delta
        q = j + 1
        if q not in wanted:
            continue
        cross = np.sqrt(cross_sq)
        within_x = np.sqrt(within_x_sq)
        within_y = np.sqrt(within_y_sq)
        for n in sample_counts:
            value = (
                2.0 * cross[:n, :n].mean()
                - within_x[:n, :n].mean()
                - within_y[:n, :n].mean()
            )
            out[(n, q)] = float(max(0.0, value))
    return out


def compute_power(cfg: dict) -> list[dict]:
    """Independent calibration, size-check, and alternative energy scores."""
    seed = int(cfg["seed"])
    sigma = float(cfg["noise_sigma"])
    parity_order = int(cfg["power_parity_order"])
    orders = [int(q) for q in cfg["power_feature_orders"]]
    counts = [int(n) for n in cfg["power_tree_counts"]]
    n_max, q_max = max(counts), max(orders)
    raw = []

    banks = (
        ("calibration_null", 0.0),
        ("size_null", 0.0),
        ("alternative", 1.0),
    )
    for rep in range(int(cfg["power_reps"])):
        for bank_idx, (bank, structured_fraction) in enumerate(banks):
            rng = np.random.default_rng(seed + 100_000 * rep + 10_000 * bank_idx)
            reference = _parity_features(
                rng, n_max, parity_order=parity_order, feature_order=q_max,
                structured_fraction=0.0, noise_sigma=sigma,
            )
            comparison = _parity_features(
                rng, n_max, parity_order=parity_order, feature_order=q_max,
                structured_fraction=structured_fraction, noise_sigma=sigma,
            )
            scores = _energy_prefix_grid(
                reference, comparison,
                feature_orders=orders, sample_counts=counts,
            )
            for (n, q), score in sorted(scores.items()):
                raw.append({
                    "rep": rep,
                    "bank": bank,
                    "tree_count": n,
                    "feature_order": q,
                    "energy": score,
                })
        print(f"  power rep {rep + 1}/{cfg['power_reps']} done", flush=True)
    return raw


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return float(center - half), float(center + half)


def summarize_power(raw: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["tree_count"], row["feature_order"], row["bank"])].append(
            float(row["energy"]),
        )
    summary = []
    keys = sorted({(row["tree_count"], row["feature_order"]) for row in raw})
    for n, q in keys:
        calibration = np.asarray(grouped[(n, q, "calibration_null")], dtype=float)
        size_null = np.asarray(grouped[(n, q, "size_null")], dtype=float)
        alternative = np.asarray(grouped[(n, q, "alternative")], dtype=float)
        threshold = float(np.quantile(calibration, 0.95, method="higher"))
        alt_hits = int(np.sum(alternative > threshold))
        size_hits = int(np.sum(size_null > threshold))
        power_lo, power_hi = _wilson_interval(alt_hits, alternative.size)
        size_lo, size_hi = _wilson_interval(size_hits, size_null.size)
        summary.append({
            "tree_count": n,
            "feature_order": q,
            "null_95_threshold": threshold,
            "detection_probability": alt_hits / alternative.size,
            "detection_lo": power_lo,
            "detection_hi": power_hi,
            "verified_size": size_hits / size_null.size,
            "verified_size_lo": size_lo,
            "verified_size_hi": size_hi,
        })
    return summary


def compute_scatter(cfg: dict) -> list[dict]:
    """Energy and exact Wasserstein on identical calibration/alternative laws."""
    seed = int(cfg["seed"]) + 3_000_000
    n = int(cfg["scatter_tree_count"])
    parity_order = int(cfg["scatter_parity_order"])
    feature_orders = [int(q) for q in cfg["scatter_feature_orders"]]
    fractions = [float(f) for f in cfg["scatter_fractions"]]
    sigma = float(cfg["noise_sigma"])
    q_max = max(feature_orders)
    raw = []

    # An independent null bank estimates one fixed finite-sample floor for each
    # discrepancy and q.  Using a fixed floor avoids inducing correlation by
    # subtracting the same noisy reference score from both axes point by point.
    for rep in range(int(cfg["scatter_calibration_reps"])):
        rng = np.random.default_rng(seed + 100_000 * rep)
        reference = _parity_features(
            rng, n, parity_order=parity_order, feature_order=q_max,
            structured_fraction=0.0, noise_sigma=sigma,
        )
        null = _parity_features(
            rng, n, parity_order=parity_order, feature_order=q_max,
            structured_fraction=0.0, noise_sigma=sigma,
        )
        for q in feature_orders:
            x, y = reference[:, :q], null[:, :q]
            raw.append({
                "bank": "calibration_null",
                "rep": rep,
                "parity_order": parity_order,
                "feature_order": q,
                "structured_fraction": 0.0,
                "energy": energy_distance(x, y),
                "wasserstein": wasserstein_assignment(x, y),
            })

    for rep in range(int(cfg["scatter_reps"])):
        rng = np.random.default_rng(seed + 20_000_000 + 100_000 * rep)
        reference = _parity_features(
            rng, n, parity_order=parity_order, feature_order=q_max,
            structured_fraction=0.0, noise_sigma=sigma,
        )
        null = _parity_features(
            rng, n, parity_order=parity_order, feature_order=q_max,
            structured_fraction=0.0, noise_sigma=sigma,
        )
        structured = _parity_features(
            rng, n, parity_order=parity_order, feature_order=q_max,
            structured_fraction=1.0, noise_sigma=sigma,
        )
        row_order = rng.permutation(n)
        for fraction in fractions:
            n_structured = int(round(fraction * n))
            mixture = null.copy()
            chosen = row_order[:n_structured]
            mixture[chosen] = structured[chosen]
            for q in feature_orders:
                x, y = reference[:, :q], mixture[:, :q]
                raw.append({
                    "bank": "alternative",
                    "rep": rep,
                    "parity_order": parity_order,
                    "feature_order": q,
                    "structured_fraction": fraction,
                    "energy": energy_distance(x, y),
                    "wasserstein": wasserstein_assignment(x, y),
                })
        print(f"  scatter rep {rep + 1}/{cfg['scatter_reps']} done", flush=True)
    return raw


def center_scatter(raw: list[dict]) -> list[dict]:
    floors = {}
    feature_orders = sorted({row["feature_order"] for row in raw})
    for q in feature_orders:
        rows = [
            row for row in raw
            if row["bank"] == "calibration_null" and row["feature_order"] == q
        ]
        floors[q] = (
            float(np.mean([row["energy"] for row in rows])),
            float(np.mean([row["wasserstein"] for row in rows])),
        )

    centered = []
    for row in raw:
        if row["bank"] != "alternative":
            continue
        floor_ed, floor_w1 = floors[row["feature_order"]]
        centered.append({
            **row,
            "energy_floor": floor_ed,
            "wasserstein_floor": floor_w1,
            "energy_excess": row["energy"] - floor_ed,
            "wasserstein_excess": row["wasserstein"] - floor_w1,
            "contains_full_tuple": int(row["feature_order"] >= row["parity_order"]),
        })
    return centered


def compute_timing(cfg: dict) -> list[dict]:
    """Time fixed-reference Energy updates and exact empirical Wasserstein."""
    seed = int(cfg["seed"]) + 6_000_000
    sigma = float(cfg["noise_sigma"])
    parity_order = int(cfg["timing_parity_order"])
    feature_order = int(cfg["timing_feature_order"])
    counts = [int(n) for n in cfg["timing_sample_counts"]]
    reps = int(cfg["timing_reps"])

    warm_rng = np.random.default_rng(seed - 1)
    warm_x = _parity_features(
        warm_rng, 16, parity_order=parity_order, feature_order=feature_order,
        structured_fraction=0.0, noise_sigma=sigma,
    )
    warm_y = _parity_features(
        warm_rng, 16, parity_order=parity_order, feature_order=feature_order,
        structured_fraction=1.0, noise_sigma=sigma,
    )
    energy_update_terms(warm_x, warm_y)
    wasserstein_assignment(warm_x, warm_y)

    raw = []
    for n in counts:
        for rep in range(reps):
            rng = np.random.default_rng(seed + 100_000 * n + rep)
            x = _parity_features(
                rng, n, parity_order=parity_order, feature_order=feature_order,
                structured_fraction=0.0, noise_sigma=sigma,
            )
            y = _parity_features(
                rng, n, parity_order=parity_order, feature_order=feature_order,
                structured_fraction=1.0, noise_sigma=sigma,
            )
            methods = [
                ("energy", energy_update_terms),
                ("wasserstein", wasserstein_assignment),
            ]
            if rep % 2:
                methods.reverse()
            for name, fn in methods:
                start = time.perf_counter()
                value = fn(x, y)
                elapsed = time.perf_counter() - start
                raw.append({
                    "sample_count": n,
                    "rep": rep,
                    "method": name,
                    "seconds": elapsed,
                    "value": value,
                })
        print(f"  timing n={n} done", flush=True)
    return raw


def summarize_timing(raw: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["sample_count"], row["method"])].append(row["seconds"])
    summary = []
    for (n, method), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        summary.append({
            "sample_count": n,
            "method": method,
            "median_seconds": float(np.median(arr)),
            "q25_seconds": float(np.quantile(arr, 0.25)),
            "q75_seconds": float(np.quantile(arr, 0.75)),
        })
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    power: list[dict], power_summary: list[dict], scatter_raw: list[dict],
    scatter_centered: list[dict],
    timing: list[dict], timing_summary: list[dict], cfg: dict, out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    power_path = out_dir / "power_raw.csv"
    _write_csv(power_path, power)
    _write_csv(out_dir / "power_summary.csv", power_summary)
    _write_csv(out_dir / "energy_wasserstein_raw.csv", scatter_raw)
    _write_csv(out_dir / "energy_wasserstein_scatter.csv", scatter_centered)
    _write_csv(out_dir / "timing_raw.csv", timing)
    _write_csv(out_dir / "timing_summary.csv", timing_summary)
    output_config = dict(cfg)
    output_config["power_raw_sha256"] = hashlib.sha256(
        power_path.read_bytes(),
    ).hexdigest()
    with open(out_dir / "config.json", "w") as handle:
        json.dump(output_config, handle, indent=2, sort_keys=True)


def _self_check() -> None:
    """Cheap invariants that catch a broken parity or discrepancy routine."""
    rng = np.random.default_rng(91)
    x = _parity_features(
        rng, 128, parity_order=3, feature_order=2,
        structured_fraction=1.0, noise_sigma=0.0,
    )
    assert set(np.unique(x)).issubset({-1.0, 1.0})
    full = _parity_features(
        rng, 128, parity_order=3, feature_order=3,
        structured_fraction=1.0, noise_sigma=0.0,
    )
    assert np.all(np.prod(full, axis=1) == 1.0)
    z = np.zeros((8, 3), dtype=float)
    assert energy_distance(z, z) == 0.0
    assert wasserstein_assignment(z, z) == 0.0
    check_x = rng.normal(size=(8, 3))
    check_y = rng.normal(size=(8, 3))
    grid = _energy_prefix_grid(
        check_x, check_y, feature_orders=[1, 3], sample_counts=[4, 8],
    )
    for (n, q), value in grid.items():
        assert np.isclose(value, energy_distance(check_x[:n, :q], check_y[:n, :q]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke benchmark")
    parser.add_argument("--force", action="store_true", help="ignore matching cache")
    args = parser.parse_args()

    _self_check()
    cfg = _config(quick=args.quick)
    out_dir = output_dir(args.quick)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(cfg)
    S.NR.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.force:
        with open(cache_path, "rb") as handle:
            results = pickle.load(handle)
        print("loaded", cache_path)
    else:
        results = {
            "power": compute_power(cfg),
            "scatter": compute_scatter(cfg),
            "timing": compute_timing(cfg),
        }
        with open(cache_path, "wb") as handle:
            pickle.dump(results, handle)
        print("wrote", cache_path)

    power_summary = summarize_power(results["power"])
    scatter_centered = center_scatter(results["scatter"])
    timing_summary = summarize_timing(results["timing"])
    write_outputs(
        results["power"], power_summary, results["scatter"], scatter_centered,
        results["timing"], timing_summary, cfg, out_dir,
    )
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
