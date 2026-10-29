"""Exact identification--sampling tradeoff on a seven-node binary tree.

The calculations behind the Ising running example: the child-symmetric
assignment orbits of the depth-two binary tree, the Walsh geometry of the
energy distance on binary states, the path and common-ancestor feature tables,
the compatible set of whole-tree laws that share those tables, the exact
U-statistic variances, and the search for the compatible law farthest from a
reference law.

``depth2_identification_results.json`` next to this module holds the certified
solutions, one per edge correlation rho, that plot_ising.py renders.  It was
produced by compute_depth2.py with SCIP; regenerate it only through that script
and copy the fresh file over this one once the solve has been checked.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE / "depth2_identification_results.json"

NODES = 7
EDGES = ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6))
MASKS = np.arange(1, 1 << NODES, dtype=np.int16)
ORDERS = np.asarray([bin(int(mask)).count("1") for mask in MASKS])


def gamma_weight(nodes: int, order: int) -> float:
    """Walsh weight for Euclidean energy distance on binary vectors."""
    total = 0.0
    for inside in range(order + 1):
        for outside in range(nodes - order + 1):
            count = inside + outside
            total += (
                math.comb(order, inside)
                * math.comb(nodes - order, outside)
                * ((-1) ** inside)
                * 2
                * math.sqrt(count)
            )
    return -total / (2**nodes)


GAMMA = np.asarray([gamma_weight(NODES, int(order)) for order in ORDERS])


def _tree_automorphisms() -> tuple[tuple[int, ...], ...]:
    """Return the eight local child-swap automorphisms."""
    permutations = []
    for swap_root, swap_left, swap_right in itertools.product((0, 1), repeat=3):
        permutation = list(range(NODES))
        if swap_left:
            permutation[3], permutation[4] = permutation[4], permutation[3]
        if swap_right:
            permutation[5], permutation[6] = permutation[6], permutation[5]
        if swap_root:
            old = permutation.copy()
            permutation[1], permutation[2] = old[2], old[1]
            permutation[3], permutation[5] = old[5], old[3]
            permutation[4], permutation[6] = old[6], old[4]
        permutations.append(tuple(permutation))
    return tuple(permutations)


def _apply_permutation(state: tuple[int, ...], permutation) -> tuple[int, ...]:
    result = [0] * NODES
    for old, new in enumerate(permutation):
        result[new] = state[old]
    return tuple(result)


def assignment_orbits() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Partition the 128 sign assignments into child-symmetry orbits."""
    remaining = set(itertools.product((-1, 1), repeat=NODES))
    automorphisms = _tree_automorphisms()
    orbits = []
    while remaining:
        state = min(remaining)
        orbit = tuple(sorted({_apply_permutation(state, p) for p in automorphisms}))
        orbits.append(orbit)
        remaining.difference_update(orbit)
    return tuple(orbits)


ORBITS = assignment_orbits()
ORDERED_STATES = tuple(state for orbit in ORBITS for state in orbit)


def _walsh(state: tuple[int, ...], mask: int) -> float:
    value = 1
    for node in range(NODES):
        if mask & (1 << node):
            value *= state[node]
    return float(value)


ORBIT_MOMENTS = np.asarray([
    [np.mean([_walsh(state, int(mask)) for state in orbit]) for orbit in ORBITS]
    for mask in MASKS
])
ORDERED_MOMENTS = np.asarray([
    [_walsh(state, int(mask)) for state in ORDERED_STATES]
    for mask in MASKS
])


def feature_blocks() -> tuple[np.ndarray, ...]:
    """Return the path and three separate common-ancestor strata."""
    path = ((0, 1, 3), (0, 1, 4), (0, 2, 5), (0, 2, 6))
    ca01 = ((0, 1, 2), (0, 2, 1))
    ca02 = tuple(
        (0, left, right)
        for left in (3, 4, 5, 6)
        for right in (3, 4, 5, 6)
        if left != right
    )
    ca12 = ((1, 3, 4), (1, 4, 3), (2, 5, 6), (2, 6, 5))
    return tuple(np.asarray(block, dtype=np.int16) for block in (path, ca01, ca02, ca12))


def feature_rows(blocks: tuple[np.ndarray, ...]) -> np.ndarray:
    """Map complete-tree moments to the nonconstant rows of feature tables."""
    rows = []
    for placements in blocks:
        order = int(placements.shape[1])
        for selected in range(1, 1 << order):
            row = np.zeros(len(MASKS))
            for placement in placements:
                mask = 0
                for local_index, node in enumerate(placement):
                    if selected & (1 << local_index):
                        mask ^= 1 << int(node)
                row[mask - 1] += 1.0 / len(placements)
            rows.append(row)
    return np.asarray(rows)


def pooled_feature_rows() -> np.ndarray:
    """Return the rows of the four size-three feature tables."""
    return feature_rows(feature_blocks())


FEATURE_ROWS = pooled_feature_rows()
FEATURE_ORBIT_MOMENTS = FEATURE_ROWS @ ORBIT_MOMENTS


def constraint_matrix() -> np.ndarray:
    """Return normalization and an independent basis of feature constraints."""
    candidates = np.vstack((np.ones(len(ORBITS)), FEATURE_ORBIT_MOMENTS))
    _, _, pivots = qr(candidates.T, pivoting=True, mode="economic")
    rank = np.linalg.matrix_rank(candidates, tol=1e-10)
    return candidates[np.sort(pivots[:rank])]


CONSTRAINTS = constraint_matrix()


def ising_orbit_masses(rho: float) -> np.ndarray:
    """Return orbit masses for the depth-two broadcast Ising law."""
    masses = []
    for orbit in ORBITS:
        mass = 0.0
        for state in orbit:
            probability = 2.0 ** (-NODES)
            for parent, child in EDGES:
                probability *= 1.0 + rho * state[parent] * state[child]
            mass += probability
        masses.append(mass)
    masses = np.asarray(masses)
    masses /= masses.sum()
    return masses


def whole_tree_energy(left: np.ndarray, right: np.ndarray) -> float:
    difference = ORBIT_MOMENTS @ (left - right)
    return float(np.dot(GAMMA, difference * difference))


def _embedding() -> np.ndarray:
    return np.sqrt(GAMMA)[:, None] * ORBIT_MOMENTS


FULL_EMBEDDING = _embedding()
FULL_QUADRATIC = FULL_EMBEDDING.T @ FULL_EMBEDDING


def feature_projection(blocks: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return an orthonormal feature-row map in weighted moment space."""
    rows = feature_rows(blocks)
    inverse_gamma = 1.0 / GAMMA
    gram = (rows * inverse_gamma[None, :]) @ rows.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > 1e-10 * max(float(eigenvalues[-1]), 1.0)
    whitener = (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])).T
    return whitener @ rows


FEATURE_PROJECTION = feature_projection(feature_blocks())
FEATURE_EMBEDDING = FEATURE_PROJECTION @ ORBIT_MOMENTS
ORDERED_FEATURE_EMBEDDING = FEATURE_PROJECTION @ ORDERED_MOMENTS


def _check_feature_tables_child_order_invariant() -> None:
    """Verify that the projected feature tables do not vary with child order."""
    offset = 0
    for orbit in ORBITS:
        block = ORDERED_FEATURE_EMBEDDING[:, offset:offset + len(orbit)]
        if not np.allclose(block, block[:, :1], atol=2e-12, rtol=2e-10):
            raise RuntimeError("the selected feature tables vary with child order")
        offset += len(orbit)


_check_feature_tables_child_order_invariant()


def grandchild_moment_projection() -> np.ndarray:
    """Project an orbit law to mean triple and four-way grandchild moments."""
    triple_masks = (
        (3, 4, 5),
        (3, 4, 6),
        (3, 5, 6),
        (4, 5, 6),
    )

    def moment_row(nodes: tuple[int, ...]) -> np.ndarray:
        mask = sum(1 << node for node in nodes)
        index = int(np.flatnonzero(MASKS == mask)[0])
        return ORBIT_MOMENTS[index]

    return np.vstack((
        np.mean([moment_row(nodes) for nodes in triple_masks], axis=0),
        moment_row((3, 4, 5, 6)),
    ))


GRANDCHILD_MOMENT_PROJECTION = grandchild_moment_projection()


def grandchild_probability_halfplanes(rho: float) -> np.ndarray:
    """Return all probability half-planes after grandchild marginalization.

    Each row is ``(offset, triple, four_way)`` and represents the exact
    inequality ``offset + triple * m3 + four_way * m4 >= 0`` for one orbit of
    four-grandchild sign assignments.  The offset contains the one- and
    two-node moments fixed by the rho-indexed feature tables.
    """
    sibling_pair = float(rho) ** 2
    cross_pair = float(rho) ** 4
    rows = []
    for state in itertools.product((-1, 1), repeat=4):
        sibling_sum = state[0] * state[1] + state[2] * state[3]
        cross_sum = sum(
            state[left] * state[right]
            for left in (0, 1)
            for right in (2, 3)
        )
        triple_sum = sum(
            math.prod(state[index] for index in range(4) if index != omitted)
            for omitted in range(4)
        )
        rows.append((
            1.0 + sibling_pair * sibling_sum + cross_pair * cross_sum,
            float(triple_sum),
            float(math.prod(state)),
        ))
    return np.unique(np.round(np.asarray(rows), decimals=14), axis=0)


def marginal_grandchild_polygon(rho: float) -> np.ndarray:
    """Return the exact two-dimensional slice of the marginal probability simplex."""
    rows = grandchild_probability_halfplanes(rho)
    candidates = []
    for first, second in itertools.combinations(rows, 2):
        matrix = np.vstack((first[1:], second[1:]))
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        point = np.linalg.solve(matrix, -np.asarray((first[0], second[0])))
        if np.min(rows[:, 0] + rows[:, 1:] @ point) >= -1e-10:
            candidates.append(point)
    points = np.unique(np.round(np.asarray(candidates), decimals=11), axis=0)
    if len(points) < 3:
        return points
    hull = ConvexHull(points)
    return points[hull.vertices]


def grandchild_sign_count_probabilities(
    points: np.ndarray, rho: float,
) -> np.ndarray:
    """Map two grandchild moments to four sign-count probabilities.

    The columns give the total marginal probability that the four
    grandchildren are all equal, split two-and-two, have one negative sign,
    or have one positive sign.  They are nonnegative and sum to one, so they
    place every compatible marginal directly in a tetrahedron.  The
    compatible set itself is still computed from all full-tree constraints.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    sibling_pair = float(rho) ** 2
    cross_pair = float(rho) ** 4
    pair_sum = 2.0 * sibling_pair + 4.0 * cross_pair
    triple_moment = points[:, 0]
    four_way_moment = points[:, 1]
    probabilities = np.column_stack((
        (1.0 + pair_sum + four_way_moment) / 8.0,
        (3.0 - pair_sum + 3.0 * four_way_moment) / 8.0,
        (1.0 - 2.0 * triple_moment - four_way_moment) / 4.0,
        (1.0 + 2.0 * triple_moment - four_way_moment) / 4.0,
    ))
    if np.max(np.abs(np.sum(probabilities, axis=1) - 1.0)) > 1e-10:
        raise RuntimeError("grandchild sign-count probabilities do not sum to one")
    return probabilities


def rho0_grandchild_sign_count_probabilities(points: np.ndarray) -> np.ndarray:
    """Return sign-count probabilities for the rho=0 marginal slice."""
    return grandchild_sign_count_probabilities(points, 0.0)


def rho0_grandchild_polygon() -> np.ndarray:
    """Intersect the marginalized rho=0 half-planes in the two free moments."""
    return marginal_grandchild_polygon(0.0)


def _grandchild_polygon(constraints: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Project a linearly constrained orbit simplex into two moments."""

    def support(projected_direction: np.ndarray) -> np.ndarray:
        objective = projected_direction @ GRANDCHILD_MOMENT_PROJECTION
        result = linprog(
            -objective,
            A_eq=constraints,
            b_eq=target,
            bounds=(0.0, None),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"compatible-set projection failed: {result.message}")
        return GRANDCHILD_MOMENT_PROJECTION @ result.x

    points = np.asarray([
        support(np.asarray(direction, dtype=float))
        for direction in ((1, 0), (0, 1), (-1, 0), (0, -1))
    ])
    for _ in range(64):
        points = np.unique(np.round(points, decimals=11), axis=0)
        if len(points) < 3:
            return points
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
        additions = []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = end - start
            outward = np.asarray((edge[1], -edge[0]))
            outward /= np.linalg.norm(outward)
            candidate = support(outward)
            if outward @ candidate > outward @ start + 1e-9:
                additions.append(candidate)
        if not additions:
            return polygon
        points = np.vstack((points, additions))
    raise RuntimeError("compatible-set projection did not converge")


def compatible_grandchild_polygon(reference: np.ndarray) -> np.ndarray:
    """Return the projected marginal polytope with the same feature tables."""
    return _grandchild_polygon(CONSTRAINTS, CONSTRAINTS @ reference)


def _covariance(embedding: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = embedding @ masses
    centered = embedding - mean[:, None]
    covariance = (centered * masses[None, :]) @ centered.T
    return mean, covariance


def variance_coefficients(
    left: np.ndarray,
    right: np.ndarray,
    embedding: np.ndarray,
) -> tuple[float, float, float]:
    """Coefficients of the exact two-sample energy U-statistic variance."""
    mean_left, covariance_left = _covariance(embedding, left)
    mean_right, covariance_right = _covariance(embedding, right)
    difference = mean_left - mean_right
    leading = float(difference @ (covariance_left + covariance_right) @ difference)
    within = float(
        np.trace(covariance_left @ covariance_left)
        + np.trace(covariance_right @ covariance_right)
    )
    cross = float(np.trace(covariance_left @ covariance_right))
    return leading, within, cross


def u_variance(coefficients, ns: np.ndarray) -> np.ndarray:
    leading, within, cross = coefficients
    return (
        4.0 * leading / ns
        + 2.0 * within / (ns * (ns - 1.0))
        + 4.0 * cross / ns**2
    )


def farthest_compatible_scip(
    rho: float,
    *,
    time_limit: float = 120.0,
    relative_gap: float = 1e-8,
    hide_output: bool = True,
) -> dict:
    """Globally maximize whole-tree energy over the exact compatible polytope."""
    from pyscipopt import Model, quicksum

    reference = ising_orbit_masses(rho)
    target = CONSTRAINTS @ reference
    reference_moments = ORBIT_MOMENTS @ reference

    model = Model(f"depth2-rho-{rho:.6f}")
    if hide_output:
        model.hideOutput(True)
    model.setRealParam("limits/time", float(time_limit))
    model.setRealParam("limits/gap", float(relative_gap))
    model.setRealParam("numerics/feastol", 1e-9)
    model.setRealParam("numerics/epsilon", 1e-10)
    model.setIntParam("display/verblevel", 0 if hide_output else 4)

    masses = [model.addVar(lb=0.0, ub=1.0, name=f"q_{index}") for index in range(len(ORBITS))]
    for row_index, row in enumerate(CONSTRAINTS):
        model.addCons(
            quicksum(float(value) * masses[index] for index, value in enumerate(row))
            == float(target[row_index]),
            name=f"feature_{row_index}",
        )

    differences = []
    for moment_index, row in enumerate(ORBIT_MOMENTS):
        difference = model.addVar(lb=-2.0, ub=2.0, name=f"d_{moment_index}")
        model.addCons(
            difference
            == float(reference_moments[moment_index])
            - quicksum(float(value) * masses[index] for index, value in enumerate(row)),
            name=f"moment_{moment_index}",
        )
        differences.append(difference)

    upper_bound = float(4.0 * np.sum(GAMMA))
    objective = model.addVar(lb=0.0, ub=upper_bound, name="energy")
    model.addCons(
        objective
        <= quicksum(float(weight) * difference * difference
                    for weight, difference in zip(GAMMA, differences)),
        name="energy_definition",
    )
    model.setObjective(objective, "maximize")
    model.optimize()

    status = str(model.getStatus())
    solution = model.getBestSol()
    if solution is None:
        raise RuntimeError(f"SCIP found no compatible law at rho={rho}: {status}")
    law = np.asarray([model.getSolVal(solution, variable) for variable in masses])
    law[np.abs(law) < 5e-10] = 0.0
    law /= law.sum()
    energy = whole_tree_energy(reference, law)
    violation = float(np.max(np.abs(CONSTRAINTS @ law - target)))
    primal_bound = float(model.getPrimalbound())
    dual_bound = float(model.getDualbound())
    gap = float(model.getGap())
    return {
        "rho": float(rho),
        "status": status,
        "energy": energy,
        "primal_bound": primal_bound,
        "dual_bound": dual_bound,
        "relative_gap": gap,
        "constraint_violation": violation,
        "support_size": int(np.count_nonzero(law > 1e-8)),
        "reference": reference,
        "law": law,
    }


def farthest_compatible_search(
    rho: float,
    *,
    starts: int = 400,
    seed: int = 12345,
    extra_directions: tuple[np.ndarray, ...] = (),
) -> dict:
    """Search vertices of the compatible polytope by random LP objectives.

    This search is not certified; SCIP provides the certificate.  Each
    sampled vertex is improved by repeatedly maximizing the quadratic's linear
    gradient over the same polytope.
    """
    reference = ising_orbit_masses(rho)
    target = CONSTRAINTS @ reference
    if rho >= 1.0 - 1e-12:
        return {
            "rho": float(rho),
            "status": "endpoint",
            "energy": 0.0,
            "replicate_energy": 0.0,
            "replicate_difference": 0.0,
            "constraint_violation": 0.0,
            "support_size": int(np.count_nonzero(reference > 1e-8)),
            "reference": reference,
            "law": reference.copy(),
        }

    def energy(law):
        difference = law - reference
        return float(difference @ FULL_QUADRATIC @ difference)

    def vertex(direction):
        result = linprog(
            -np.asarray(direction),
            A_eq=CONSTRAINTS,
            b_eq=target,
            bounds=(0.0, None),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"LP vertex search failed at rho={rho}: {result.message}")
        law = result.x
        law[np.abs(law) < 5e-10] = 0.0
        law /= law.sum()
        return law

    def ascend(law):
        current = energy(law)
        for _ in range(24):
            gradient = 2.0 * FULL_QUADRATIC @ (law - reference)
            candidate = vertex(gradient)
            candidate_value = energy(candidate)
            if candidate_value <= current + 1e-11:
                break
            law, current = candidate, candidate_value
        return law, current

    def one_search(current_seed):
        rng = np.random.default_rng(current_seed)
        directions = [
            FULL_QUADRATIC[:, index]
            for index in range(FULL_QUADRATIC.shape[1])
        ]
        directions.extend(-direction for direction in tuple(directions))
        directions.extend(np.asarray(direction) for direction in extra_directions)
        directions.extend(-np.asarray(direction) for direction in extra_directions)
        directions.extend(
            rng.standard_normal(len(ORBITS))
            for _ in range(max(starts - len(directions), 0))
        )
        best_law = None
        best_value = -np.inf
        hits = 0
        for direction in directions:
            law, value = ascend(vertex(direction))
            if value > best_value + 1e-10:
                best_law, best_value, hits = law, value, 1
            elif abs(value - best_value) <= 1e-10:
                hits += 1
        return best_law, best_value, hits

    law, best_value, hits = one_search(seed)
    replicate_law, replicate_value, replicate_hits = one_search(seed + 104729)
    replicate_difference = abs(best_value - replicate_value)
    replicate_law_distance = float(np.linalg.norm(law - replicate_law, ord=1))
    variance_difference = max(
        np.max(np.abs(
            np.asarray(variance_coefficients(reference, law, embedding))
            - np.asarray(variance_coefficients(reference, replicate_law, embedding))
        ))
        for embedding in (FULL_EMBEDDING, FEATURE_EMBEDDING)
    )
    if replicate_value > best_value:
        law, best_value, hits = replicate_law, replicate_value, replicate_hits
    violation = float(np.max(np.abs(CONSTRAINTS @ law - target)))
    return {
        "rho": float(rho),
        "status": "search-stable" if replicate_difference < 1e-9 else "search-disagrees",
        "energy": best_value,
        "replicate_energy": replicate_value,
        "replicate_difference": replicate_difference,
        "best_hits": int(hits),
        "replicate_hits": int(replicate_hits),
        "replicate_law_distance": replicate_law_distance,
        "replicate_variance_difference": float(variance_difference),
        "constraint_violation": violation,
        "support_size": int(np.count_nonzero(law > 1e-8)),
        "reference": reference,
        "law": law,
    }


def risk_arrays(results: list[dict], ns: np.ndarray) -> dict[str, np.ndarray]:
    """Risk when every sampled tree is averaged over its child orderings."""
    gaps = np.asarray([result["energy"] for result in results])
    full_variance = []
    feature_variance = []
    feature_mean_gaps = []
    for result in results:
        left = result["reference"]
        right = result["law"]
        full_variance.append(u_variance(
            variance_coefficients(left, right, FULL_EMBEDDING), ns,
        ))
        feature_variance.append(u_variance(
            variance_coefficients(left, right, FEATURE_EMBEDDING), ns,
        ))
        feature_mean_gaps.append(
            np.linalg.norm(FEATURE_EMBEDDING @ (left - right))
        )
    full_variance = np.asarray(full_variance)
    feature_variance = np.asarray(feature_variance)
    unresolved = gaps[:, None] ** 2 / full_variance
    sampling = feature_variance / full_variance
    return {
        "gap": gaps,
        "full_variance": full_variance,
        "feature_variance": feature_variance,
        "unresolved_ratio": unresolved,
        "sampling_ratio": sampling,
        "total_ratio": unresolved + sampling,
        "feature_mean_gap": np.asarray(feature_mean_gaps),
    }


def structural_diagnostics() -> dict:
    return {
        "assignment_orbits": len(ORBITS),
        "probability_directions": len(ORBITS) - 1,
        "feature_table_rank_with_normalization": int(np.linalg.matrix_rank(CONSTRAINTS)),
        "feature_directions": int(np.linalg.matrix_rank(CONSTRAINTS) - 1),
        "compatible_directions": int(len(ORBITS) - np.linalg.matrix_rank(CONSTRAINTS)),
        "feature_embedding_rank": int(FEATURE_EMBEDDING.shape[0]),
    }


def save_results(results: list[dict], path: Path = RESULTS_FILE) -> None:
    serializable = {
        "structure": structural_diagnostics(),
        "solutions": [
            {key: value for key, value in result.items() if key not in ("reference", "law")}
            | {
                "reference": result["reference"].tolist(),
                "law": result["law"].tolist(),
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(serializable, indent=2) + "\n")


def load_results(path: Path = RESULTS_FILE) -> list[dict]:
    """Load the cached search-stable compatible laws."""
    raw = json.loads(path.read_text())["solutions"]
    results = []
    for row in raw:
        row["reference"] = np.asarray(row["reference"], dtype=float)
        row["law"] = np.asarray(row["law"], dtype=float)
        results.append(row)
    return results
