"""Search for the compatible law farthest from each reference law.

For every edge correlation rho this looks, over the child-symmetric laws on the
seven-node tree that share all four size-three feature tables with the
reference law P_rho, for the law Q_rho at maximal whole-tree energy distance.
The default solver is SCIP through pyscipopt; ``--search`` falls back to a
multi-start local search on scipy alone, which is cheaper but not certified.

Run from the repository root with

    python -m analysis.figures.theory.compute_depth2 [--search] [--render]

Results go to numerical_results/depth2_identification_results.json.  The file
next to depth2.py is the certified copy that plot_ising.py reads; replace it by
hand once a fresh solve has been checked (``--render`` writes the two
diagnostic panels of plot_depth2.py to imgs/ for that purpose).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.figures.theory import depth2
from analysis.figures.theory import plot_depth2
from analysis.figures.theory import style as S


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rho", type=float, nargs="*", default=(0.0, 0.5, 0.9, 1.0))
    parser.add_argument(
        "--out", type=Path, default=S.NR / "depth2_identification_results.json",
    )
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--show-scip", action="store_true")
    parser.add_argument(
        "--search", action="store_true",
        help="multi-start local search instead of SCIP",
    )
    parser.add_argument("--starts", type=int, default=400)
    parser.add_argument(
        "--render", action="store_true", help="also write the diagnostic panels to imgs/",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(depth2.structural_diagnostics(), indent=2))
    results = []
    for rho in args.rho:
        if args.search:
            result = depth2.farthest_compatible_search(rho, starts=args.starts)
        else:
            result = depth2.farthest_compatible_scip(
                rho, time_limit=args.time_limit, hide_output=not args.show_scip,
            )
        results.append(result)
        search_gap = result.get(
            "relative_gap", result.get("replicate_difference", float("nan")),
        )
        print(
            f"rho={rho:.6f} status={result['status']} "
            f"energy={result['energy']:.9g} gap={search_gap:.3g} "
            f"support={result['support_size']} "
            f"violation={result['constraint_violation']:.3g}"
        )
    depth2.save_results(results, args.out)
    print("wrote", args.out)
    if args.render:
        S.IMGS.mkdir(parents=True, exist_ok=True)
        print("wrote", plot_depth2.render_pilot(results, S.IMGS))
        print("wrote", plot_depth2.render_compatible_law(results, S.IMGS))


if __name__ == "__main__":
    main()
