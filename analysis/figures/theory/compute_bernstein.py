"""Prepare the exact inputs for the Bernstein feature figure.

Run from the repository root with

    python -m analysis.figures.theory.compute_bernstein
"""

from __future__ import annotations

import csv
import json

from analysis.figures.theory import style as S
from analysis.figures.theory.exact import (
    PARITY_ORDERS,
    breadth_orbit,
    cell_phase_rows,
    child_symmetric_identification,
    path_orbit,
    population_rows,
    verify_child_order_symmetrization,
)


def _write(path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty numerical result {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    S.NR.mkdir(parents=True, exist_ok=True)
    symmetry_checks = verify_child_order_symmetrization()
    identification = child_symmetric_identification()
    population = population_rows()
    phase = cell_phase_rows()
    _write(S.NR / "attenuation_exact.csv", population)
    _write(S.NR / "cell_phase_exact.csv", phase)

    null = phase[0]
    null_ratio = null["feature_within"] / null["full_within"]
    if not abs(null_ratio - null["feature_cross"] / null["full_cross"]) < 2e-13:
        raise RuntimeError("the exact null variance ratio depends on n")
    diagnostics = {
        "state_law": (
            "exact binary parity on the complete depth-three tree, averaged "
            "over aligned supports"
        ),
        "noise_sigma": 0.0,
        "tree_nodes": 15,
        "path_strata": "last r nodes of every root-to-leaf path",
        "common_ancestor_strata": (
            "level-one ancestor plus r-1 of its four level-three descendants"
        ),
        "candidate_descendants_per_ancestor": 4,
        "path_support_counts": {str(r): len(path_orbit(r)) for r in PARITY_ORDERS},
        "breadth_support_counts": {
            str(r): len(breadth_orbit(r)) for r in PARITY_ORDERS
        },
        "aligned_moment_gap": "theta / M_r on every support moment",
        "child_order_automorphisms_checked_per_case": (
            symmetry_checks[0]["automorphisms_checked"]
        ),
        "binary_assignments_checked_per_case": (
            symmetry_checks[0]["assignments_checked"]
        ),
        "population_panels": (
            "raw loss coefficients across interaction and feature order, with "
            "exact omission below the arity boundary"
        ),
        "path_feature_orders": "1 through 4",
        "common_ancestor_feature_orders": "1 through 4",
        "canonical_visible_ratio": (
            "one for every matched feature with q >= r; zero for q < r"
        ),
        "phase_law": ("uniform mixture over the 8 aligned order-four breadth supports"),
        "phase_case": "all-times q=3 loss versus breadth r=4",
        "phase_target": "complete child-symmetric tree energy",
        "phase_features": (
            "joint row space of the path law and the separate common-ancestor "
            "laws at time pairs (1,2), (1,3), and (2,3)"
        ),
        "phase_null_variance_ratio": null_ratio,
        "child_symmetric_identification": identification,
        "finite_sample_statistic": "unbiased tree-level two-sample U-statistic",
        "finite_calculation": (
            "32,768 binary assignments for exact law checks; all 128 tree "
            "automorphisms; sparse moment algebra and a full Walsh-Hadamard "
            "autocorrelation for the full-tree variance"
        ),
    }
    with (S.NR / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print("wrote", S.NR)


if __name__ == "__main__":
    main()
