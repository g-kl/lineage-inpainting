"""Exact depth-three binary-tree calculations for the Bernstein figure.

The benchmark uses the same 15-node balanced binary tree and time strata as
the running Ising example. An order-``r`` depth interaction occupies the last
``r`` nodes of a root-to-leaf path. An order-``r`` breadth interaction
contains a level-one ancestor and ``r - 1`` of its four level-three
descendants. Both laws are averaged over every aligned support, so they are
invariant under every local exchange of two children.

Population discrepancies use sparse Walsh coefficients. The full-tree
U-statistic variance is evaluated without materializing its 32,767-dimensional
covariance matrix: an exact Walsh--Hadamard autocorrelation supplies the only
dense sum required by the calculation.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import itertools
import math

import numpy as np


TREE_DEPTH = 3
FULL_ORDER = 2 ** (TREE_DEPTH + 1) - 1
PARITY_ORDERS = (2, 3, 4)
PATH_FEATURE_ORDERS = (1, 2, 3, 4)
CA_FEATURE_ORDERS = (1, 2, 3, 4)
LEVEL_ONE_INDICES = (1, 2)
LEAF_INDICES = tuple(range(2 ** TREE_DEPTH - 1, FULL_ORDER))
DESCENDANTS = 4
PHASE_PARITY_ORDER = 4
PHASE_FEATURE_ORDER = 3


@lru_cache(maxsize=None)
def gamma(n_states: int, order: int) -> float:
    """Walsh weight of an order-``order`` moment in ``n_states`` states."""
    if not 1 <= order <= n_states:
        raise ValueError("order must lie between one and n_states")
    coefficient = 0.0
    for selected_negative in range(order + 1):
        sign = -1.0 if selected_negative % 2 else 1.0
        selected_count = math.comb(order, selected_negative)
        for other_negative in range(n_states - order + 1):
            total_negative = selected_negative + other_negative
            coefficient += (
                2.0
                * math.sqrt(total_negative)
                * sign
                * selected_count
                * math.comb(n_states - order, other_negative)
            )
    value = -coefficient / (2 ** n_states)
    if value <= 0.0:
        raise RuntimeError("a nonconstant energy-distance Walsh weight is not positive")
    return value


def _node_path(node: int) -> tuple[int, ...]:
    """Return the root-to-``node`` path in heap-indexed tree coordinates."""
    if not 0 <= node < FULL_ORDER:
        raise ValueError(f"node {node} does not belong to the depth-three tree")
    path = [node]
    while node:
        node = (node - 1) // 2
        path.append(node)
    return tuple(reversed(path))


def _leaf_descendants(ancestor: int) -> tuple[int, ...]:
    descendants = tuple(
        leaf for leaf in LEAF_INDICES if ancestor in _node_path(leaf)[:-1]
    )
    if len(descendants) != DESCENDANTS:
        raise RuntimeError(
            "a level-one ancestor does not have four level-three descendants"
        )
    return descendants


@lru_cache(maxsize=None)
def path_orbit(r: int) -> tuple[tuple[int, ...], ...]:
    """All last-``r`` path supports in the fixed Ising time strata."""
    if r not in PARITY_ORDERS:
        raise ValueError(f"unsupported parity order {r}")
    return tuple(_node_path(leaf)[-r:] for leaf in LEAF_INDICES)


@lru_cache(maxsize=None)
def breadth_orbit(r: int) -> tuple[tuple[int, ...], ...]:
    """All aligned level-one-ancestor breadth supports of order ``r``."""
    if r not in PARITY_ORDERS:
        raise ValueError(f"unsupported parity order {r}")
    return tuple(
        (ancestor,) + descendants
        for ancestor in LEVEL_ONE_INDICES
        for descendants in itertools.combinations(
            _leaf_descendants(ancestor),
            r - 1,
        )
    )


def _support_nodes(interaction: str, r: int) -> tuple[tuple[int, ...], ...]:
    if interaction == "depth":
        return path_orbit(r)
    if interaction == "breadth":
        return breadth_orbit(r)
    raise ValueError(f"unknown interaction {interaction!r}")


def _mask(nodes: tuple[int, ...]) -> int:
    return sum(1 << node for node in nodes)


def _support_masks(interaction: str, r: int) -> tuple[int, ...]:
    masks = tuple(_mask(nodes) for nodes in _support_nodes(interaction, r))
    if len(set(masks)) != len(masks):
        raise RuntimeError("an aligned support family contains duplicates")
    return masks


@lru_cache(maxsize=None)
def _states() -> np.ndarray:
    """All assignments, indexed by the mask of negative tree states."""
    assignment_masks = np.arange(2 ** FULL_ORDER, dtype=np.uint16)
    bits = (
        assignment_masks[:, None] >> np.arange(FULL_ORDER, dtype=np.uint16)[None, :]
    ) & 1
    return np.where(bits, -1.0, 1.0)


def _fwht(values: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh--Hadamard transform in binary-mask order."""
    transformed = np.asarray(values, dtype=np.float64).copy()
    if transformed.ndim != 1 or transformed.size & (transformed.size - 1):
        raise ValueError("the Walsh--Hadamard transform requires a power-of-two vector")
    width = 1
    while width < transformed.size:
        blocks = transformed.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return transformed


def _law_weights(
    interaction: str,
    r: int,
    theta: float,
) -> np.ndarray:
    _validate_theta(theta)
    states = _states()
    supports = _support_nodes(interaction, r)
    interaction_average = np.mean(
        [np.prod(states[:, support], axis=1) for support in supports], axis=0
    )
    weights = (1.0 + theta * interaction_average) / len(states)
    _assert_probability_law(weights)
    return weights


def _validate_theta(theta: float) -> None:
    if abs(theta) > 1.0:
        raise ValueError("theta must lie in [-1, 1]")


def _assert_probability_law(weights: np.ndarray) -> None:
    if np.any(weights < -1e-14) or not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("the symmetrized parity law is not a probability law")


@lru_cache(maxsize=None)
def _tree_automorphisms() -> tuple[tuple[int, ...], ...]:
    """All 128 automorphisms generated by local child exchanges."""
    internal_nodes = tuple(range(2 ** TREE_DEPTH - 1))
    automorphisms = []
    for flips in itertools.product((0, 1), repeat=len(internal_nodes)):
        permutation = []
        for node in range(FULL_ORDER):
            path = _node_path(node)
            original = 0
            image = 0
            for parent, child in zip(path, path[1:]):
                branch = 0 if child == 2 * parent + 1 else 1
                image_branch = branch ^ flips[original]
                image = 2 * image + 1 + image_branch
                original = 2 * original + 1 + branch
            permutation.append(image)
        automorphisms.append(tuple(permutation))
    unique = tuple(dict.fromkeys(automorphisms))
    if len(unique) != 2 ** len(internal_nodes):
        raise RuntimeError("the child exchanges did not generate 128 automorphisms")
    return unique


def _permuted_mask(mask: int, permutation: tuple[int, ...]) -> int:
    result = 0
    for source, target in enumerate(permutation):
        if mask & (1 << source):
            result |= 1 << target
    return result


def verify_child_order_symmetrization(theta: float = 0.73) -> list[dict]:
    """Check exact moments and all child-order automorphisms for both laws."""
    _validate_theta(theta)
    automorphisms = _tree_automorphisms()
    checks = []
    for interaction in ("depth", "breadth"):
        for r in PARITY_ORDERS:
            supports = _support_masks(interaction, r)
            support_set = set(supports)
            for permutation in automorphisms:
                permuted = {_permuted_mask(mask, permutation) for mask in supports}
                if permuted != support_set:
                    raise RuntimeError(
                        f"the {interaction} support family depends on child order",
                    )

            weights = _law_weights(interaction, r, theta)
            moments = _fwht(weights)
            expected = np.zeros_like(moments)
            expected[0] = 1.0
            expected[list(supports)] = theta / len(supports)
            if not np.allclose(moments, expected, rtol=2e-12, atol=3e-14):
                raise RuntimeError(
                    f"the {interaction} law failed its exact Walsh-moment check",
                )

            checks.append(
                {
                    "interaction": interaction,
                    "parity_order": r,
                    "support_count": len(supports),
                    "moment_gap": theta / len(supports),
                    "automorphisms_checked": len(automorphisms),
                    "assignments_checked": len(weights),
                }
            )
    return checks


@lru_cache(maxsize=None)
def _all_times_feature_blocks(
    feature: str,
    q: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Placement blocks for the separate strata used by the loss."""
    if q not in PATH_FEATURE_ORDERS:
        raise ValueError(f"unsupported feature order {q}")
    if feature == "path":
        return (tuple(_node_path(leaf)[-q:] for leaf in LEAF_INDICES),)
    if feature != "ca":
        raise ValueError(f"unknown feature {feature!r}")

    levels = {
        1: LEVEL_ONE_INDICES,
        2: tuple(range(3, 7)),
        3: LEAF_INDICES,
    }
    blocks = []
    for start, end in ((1, 2), (1, 3), (2, 3)):
        placements = []
        for ancestor in levels[start]:
            descendants = tuple(
                node for node in levels[end]
                if ancestor in _node_path(node)[:-1]
            )
            draws = q - 1
            selections = (
                itertools.permutations(descendants, draws)
                if len(descendants) >= draws
                else itertools.product(descendants, repeat=draws)
            )
            placements.extend((ancestor, *selection) for selection in selections)
        blocks.append(tuple(placements))
    return tuple(blocks)


def _pooled_rows_from_blocks(blocks) -> tuple[dict[int, float], ...]:
    """Full-tree Walsh polynomials for separately pooled feature strata."""
    rows = []
    for placements in blocks:
        order = len(placements[0])
        for selected in range(1, 1 << order):
            coefficients = Counter()
            for placement in placements:
                mask = 0
                for index, node in enumerate(placement):
                    if selected & (1 << index):
                        mask ^= 1 << node
                coefficients[mask] += 1.0 / len(placements)
            rows.append(dict(coefficients))
    return tuple(rows)


def _all_times_rows(q: int) -> tuple[dict[int, float], ...]:
    return _pooled_rows_from_blocks(
        _all_times_feature_blocks("path", q)
        + _all_times_feature_blocks("ca", q)
    )


def _row_space_whitener(rows) -> np.ndarray:
    """Whiten pooled rows in the complete-tree energy geometry."""
    gram = np.zeros((len(rows), len(rows)))
    for left_index, left in enumerate(rows):
        for right_index, right in enumerate(rows):
            gram[left_index, right_index] = sum(
                coefficient * right.get(mask, 0.0)
                / gamma(FULL_ORDER, bin(mask).count("1"))
                for mask, coefficient in left.items()
                if mask
            )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = 1e-10 * max(float(eigenvalues[-1]), 1.0)
    retained = eigenvalues > tolerance
    whitener = (
        eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
    ).T
    if not np.allclose(
        whitener @ gram @ whitener.T,
        np.eye(int(np.count_nonzero(retained))),
        rtol=2e-10,
        atol=2e-10,
    ):
        raise RuntimeError("the all-times feature rows did not whiten")
    return whitener


def _all_times_feature_moments(
    q: int,
    interaction: str,
    r: int,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean difference and covariances for the complete stratified operator."""
    rows = _all_times_rows(q)
    whitener = _row_space_whitener(rows)
    interaction_supports = set(_support_masks(interaction, r))
    amplitude = theta / len(interaction_supports)

    def moment(mask, current_amplitude):
        if mask == 0:
            return 1.0
        return current_amplitude if mask in interaction_supports else 0.0

    def raw_moments(current_amplitude):
        mean = np.asarray([
            sum(coefficient * moment(mask, current_amplitude)
                for mask, coefficient in row.items())
            for row in rows
        ])
        second = np.empty((len(rows), len(rows)))
        for left_index, left in enumerate(rows):
            for right_index, right in enumerate(rows):
                second[left_index, right_index] = sum(
                    left_coefficient * right_coefficient
                    * moment(left_mask ^ right_mask, current_amplitude)
                    for left_mask, left_coefficient in left.items()
                    for right_mask, right_coefficient in right.items()
                )
        return mean, second - np.outer(mean, mean)

    mean_p, covariance_p = raw_moments(amplitude)
    mean_q, covariance_q = raw_moments(0.0)
    return (
        whitener @ (mean_p - mean_q),
        whitener @ covariance_p @ whitener.T,
        whitener @ covariance_q @ whitener.T,
    )


def child_symmetric_identification() -> dict[str, int]:
    """Dimensions of the full child-symmetric law and q=3 feature image."""
    fixed_assignments = 0
    for permutation in _tree_automorphisms():
        visited = set()
        cycles = 0
        for node in range(FULL_ORDER):
            if node in visited:
                continue
            cycles += 1
            current = node
            while current not in visited:
                visited.add(current)
                current = permutation[current]
        fixed_assignments += 2 ** cycles
    assignment_orbits = fixed_assignments // len(_tree_automorphisms())
    feature_rank = _row_space_whitener(_all_times_rows(PHASE_FEATURE_ORDER)).shape[0]
    return {
        "assignment_orbits": assignment_orbits,
        "probability_directions": assignment_orbits - 1,
        "feature_directions": feature_rank,
        "unidentified_directions": assignment_orbits - 1 - feature_rank,
    }


def _raw_feature_gaps(
    feature: str,
    q: int,
    interaction: str,
    r: int,
) -> np.ndarray:
    support_set = set(_support_masks(interaction, r))
    coefficient = 1.0 / len(support_set)
    return np.asarray(
        [
            coefficient
            * sum(
                probability for mask, probability in row.items()
                if mask in support_set
            )
            for row in _pooled_rows_from_blocks(
                _all_times_feature_blocks(feature, q)
            )
        ]
    )


def _canonical_visible_ratio(
    feature: str,
    q: int,
    interaction: str,
    r: int,
) -> float:
    rows = _pooled_rows_from_blocks(_all_times_feature_blocks(feature, q))
    difference = _raw_feature_gaps(feature, q, interaction, r)
    visible = np.linalg.norm(_row_space_whitener(rows) @ difference) ** 2
    whole = gamma(FULL_ORDER, r) / len(_support_masks(interaction, r))
    return float(visible / whole)


def population_rows() -> list[dict]:
    """Return exact population matrices for both aligned geometries."""
    rows = []
    for feature, interaction, feature_orders in (
        ("path", "depth", PATH_FEATURE_ORDERS),
        ("ca", "breadth", CA_FEATURE_ORDERS),
    ):
        for r in PARITY_ORDERS:
            orbit_size = len(_support_masks(interaction, r))
            normalized_whole = 1.0 / orbit_size
            for q in feature_orders:
                gaps = _raw_feature_gaps(feature, q, interaction, r)
                nonzero = np.abs(gaps) > 2e-14
                count = int(np.count_nonzero(nonzero))
                expected_count = (
                    math.comb(q - 1, r - 1)
                    if feature == "ca" and q >= r
                    else int(q >= r)
                )
                if count != expected_count:
                    raise RuntimeError("feature parity-moment count failed")
                if count and not np.allclose(
                    gaps[nonzero],
                    1.0 / orbit_size,
                    rtol=2e-13,
                    atol=2e-14,
                ):
                    raise RuntimeError("feature parity-moment gap failed")

                normalized_feature = float(gaps @ gaps)
                raw_ratio = normalized_feature / normalized_whole
                expected_ratio = count / orbit_size
                if not np.isclose(raw_ratio, expected_ratio, atol=2e-14):
                    raise RuntimeError("feature discrepancy ratio failed")

                visible_ratio = _canonical_visible_ratio(
                    feature,
                    q,
                    interaction,
                    r,
                )
                expected_visible = float(q >= r)
                if not np.isclose(visible_ratio, expected_visible, atol=2e-14):
                    raise RuntimeError(
                        "the canonical feature did not retain an aligned law",
                    )

                rows.append(
                    {
                        "feature": feature,
                        "interaction": interaction,
                        "parity_order": r,
                        "feature_order": q,
                        "nonzero_parity_moments": count,
                        "orbit_size": orbit_size,
                        "moment_gap_coefficient": (
                            0.0 if count == 0 else 1.0 / orbit_size
                        ),
                        "normalized_raw_feature_discrepancy": normalized_feature,
                        "normalized_whole_tree_discrepancy": normalized_whole,
                        "raw_discrepancy_ratio": raw_ratio,
                        "normalized_canonical_visible_discrepancy": (
                            visible_ratio * normalized_whole
                        ),
                        "canonical_visible_ratio": visible_ratio,
                        "status": "detected" if count else "omitted",
                    }
                )
    return rows


def _u_variance_terms(
    difference: np.ndarray,
    covariance_p: np.ndarray,
    covariance_q: np.ndarray,
) -> tuple[float, float, float]:
    """Return the three exact coefficients in the U-statistic variance."""
    leading = float(difference @ (covariance_p + covariance_q) @ difference)
    within = float(np.sum(covariance_p ** 2) + np.sum(covariance_q ** 2))
    cross = float(np.trace(covariance_p @ covariance_q))
    return leading, within, cross


@lru_cache(maxsize=None)
def _weighted_xor_autocorrelation() -> np.ndarray:
    """Autocorrelation of all nonconstant full-tree Walsh weights."""
    weights = np.zeros(2 ** FULL_ORDER)
    for mask in range(1, len(weights)):
        weights[mask] = gamma(FULL_ORDER, bin(mask).count("1"))
    spectrum = _fwht(weights)
    correlation = _fwht(spectrum ** 2) / len(weights)

    # Check the transform normalization and radial symmetry directly.
    if not np.isclose(correlation[0], weights @ weights, rtol=3e-13):
        raise RuntimeError("the Walsh autocorrelation has the wrong normalization")
    indices = np.arange(len(weights), dtype=np.uint16)
    for order in PARITY_ORDERS:
        representative = (1 << order) - 1
        direct = weights @ weights[indices ^ representative]
        if not np.isclose(
            correlation[representative],
            direct,
            rtol=3e-13,
            atol=2e-14,
        ):
            raise RuntimeError("the Walsh autocorrelation failed a direct check")
    return correlation


def _full_u_variance_terms(
    interaction: str,
    r: int,
    theta: float,
) -> tuple[float, float, float]:
    """Full-tree coefficients from sparse moments and one FWHT sum."""
    supports = _support_masks(interaction, r)
    support_set = set(supports)
    orbit_size = len(supports)
    amplitude = theta / orbit_size
    scale = gamma(FULL_ORDER, r)
    autocorrelation = _weighted_xor_autocorrelation()
    weight_square_sum = float(autocorrelation[0])
    shifted_weight_sum = float(sum(autocorrelation[mask] for mask in supports))
    support_xor_count = sum(
        (left ^ right) in support_set for left in supports for right in supports
    )

    leading = scale ** 2 * (
        2.0 * orbit_size * amplitude ** 2
        + amplitude ** 3 * support_xor_count
        - orbit_size ** 2 * amplitude ** 4
    )
    covariance_p_square = (
        weight_square_sum
        + amplitude ** 2 * shifted_weight_sum
        + (orbit_size * scale * amplitude ** 2) ** 2
        - 2.0 * orbit_size * scale ** 2 * amplitude ** 2
        - 2.0 * amplitude ** 3 * scale ** 2 * support_xor_count
    )
    covariance_q_square = weight_square_sum
    cross = weight_square_sum - orbit_size * scale ** 2 * amplitude ** 2
    return leading, covariance_p_square + covariance_q_square, cross


def cell_phase_rows(thetas: np.ndarray | None = None) -> list[dict]:
    """Exact full-tree MSE for the q=3 all-times loss and breadth order four."""
    if thetas is None:
        thetas = np.linspace(0.0, 1.0, 241)
    r = PHASE_PARITY_ORDER
    q = PHASE_FEATURE_ORDER
    orbit_size = len(breadth_orbit(r))
    full_scale = gamma(FULL_ORDER, r)

    rows = []
    for theta_value in np.asarray(thetas, dtype=float):
        full_terms = tuple(
            value / full_scale ** 2
            for value in _full_u_variance_terms(
                "breadth",
                r,
                float(theta_value),
            )
        )
        feature_difference, feature_covariance_p, feature_covariance_q = (
            _all_times_feature_moments(
                q, "breadth", r, float(theta_value),
            )
        )
        feature_terms = tuple(
            value / full_scale ** 2
            for value in _u_variance_terms(
                feature_difference,
                feature_covariance_p,
                feature_covariance_q,
            )
        )
        if not np.isclose(feature_terms[0], 0.0, atol=2e-14):
            raise RuntimeError("order-three feature did not omit order four")
        feature_terms = (0.0, feature_terms[1], feature_terms[2])

        normalized_full = float(theta_value ** 2 / orbit_size)
        normalized_feature = 0.0
        unresolved = normalized_full - normalized_feature
        rows.append(
            {
                "feature": "all_times",
                "interaction": "breadth",
                "parity_order": r,
                "feature_order": q,
                "theta": float(theta_value),
                "orbit_size": orbit_size,
                "normalized_whole_tree_discrepancy": normalized_full,
                "normalized_feature_discrepancy": normalized_feature,
                "unresolved_discrepancy": unresolved,
                "squared_unresolved": unresolved ** 2,
                "full_leading": full_terms[0],
                "full_within": full_terms[1],
                "full_cross": full_terms[2],
                "feature_leading": feature_terms[0],
                "feature_within": feature_terms[1],
                "feature_cross": feature_terms[2],
            }
        )
    return rows
