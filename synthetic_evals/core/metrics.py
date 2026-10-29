"""two-sample distances between predicted and true clouds, for the robustness panel.

``energy`` is the canonical metric every benchmark reports (viz.energy_distance). the two
extra ones come from geomloss so a conclusion can be shown not to depend on the metric:

  sinkhorn  entropic-OT (Sinkhorn) divergence, an unbiased W2-like distance
  mmd_gauss gaussian-kernel MMD

both bandwidths are set by the median pairwise distance of the two clouds, not fixed.
"""

from __future__ import annotations

import numpy as np
import torch

from synthetic_evals import viz

# geomloss SamplesLoss kwargs per metric, as (loss, blur-as-a-fraction-of-the-median)
_GEOMLOSS = {
    "sinkhorn": (dict(loss="sinkhorn", p=2), 0.5),
    "mmd_gauss": (dict(loss="gaussian"), 1.0),
}
NAMES = ["energy", *_GEOMLOSS]
_MEDIAN_CAP = 512          # rows used to estimate the median pairwise distance


def _as2d(x):
    a = np.asarray(x, dtype=np.float32)
    return a[:, None] if a.ndim == 1 else a


def median_scale(a, b, seed=0) -> float:
    """Median pairwise distance over the pooled clouds (the median heuristic)."""
    rng = np.random.default_rng(seed)
    x = np.vstack([_as2d(a), _as2d(b)])
    if len(x) > _MEDIAN_CAP:
        x = x[rng.choice(len(x), _MEDIAN_CAP, replace=False)]
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    med = float(np.median(d[np.triu_indices(len(x), k=1)])) if len(x) > 1 else 0.0
    return med if med > 0 else 1.0


def geomloss_distance(a, b, metric: str, scale: float | None = None) -> float:
    """One geomloss two-sample distance between clouds a (N, d) and b (M, d)."""
    from geomloss import SamplesLoss

    a, b = _as2d(a), _as2d(b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    kw, frac = _GEOMLOSS[metric]
    blur = (scale if scale is not None else median_scale(a, b)) * frac
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    return float(SamplesLoss(**kw, blur=blur)(ta, tb).item())


def distances(a, b) -> dict[str, float]:
    """All of NAMES between clouds a and b, sharing one median-heuristic bandwidth."""
    scale = median_scale(a, b)
    out = {"energy": viz.energy_distance(a, b)}
    for m in _GEOMLOSS:
        out[m] = geomloss_distance(a, b, m, scale=scale)
    return out


def marginal_distances(true_rows, groups, cap=2000, seed=0) -> dict[str, float]:
    """Every metric for the marginal reference: each group's true cloud vs a draw from the
    POOLED target marginal. Mirrors cell_type_crossing_eval._marginal_energy, so the
    supplement can normalise the alternate metrics the same way as the energy."""
    rng = np.random.default_rng(seed)

    def sub(x):
        return x if len(x) <= cap else x[rng.choice(len(x), cap, replace=False)]

    per_group = {m: [] for m in NAMES}
    for grp in np.unique(groups):
        mask = groups == grp
        if not mask.any():
            continue
        for m, v in distances(sub(true_rows), sub(true_rows[mask])).items():
            per_group[m].append(v)
    return {m: (float(np.mean(v)) if v else float("nan")) for m, v in per_group.items()}


def grouped_distances(pred_by_row, true_rows, groups, cap=2000, seed=0) -> dict[str, float]:
    """Mean over groups of every metric, matching cell_type_crossing_eval._grouped_energy.

    ``pred_by_row`` is (S, R, d); ``true_rows`` is (R, d); ``groups`` is (R,). Each group's
    samples are pooled into one cloud and compared to that group's true cloud, subsampled
    to ``cap`` points first (the distances are O(n^2)).
    """
    rng = np.random.default_rng(seed)

    def sub(x):
        return x if len(x) <= cap else x[rng.choice(len(x), cap, replace=False)]

    per_group = {m: [] for m in NAMES}
    for grp in np.unique(groups):
        mask = groups == grp
        if not mask.any():
            continue
        pred = pred_by_row[:, mask, :].reshape(-1, pred_by_row.shape[-1])
        for m, v in distances(sub(pred), sub(true_rows[mask])).items():
            per_group[m].append(v)
    return {m: (float(np.mean(v)) if v else float("nan")) for m, v in per_group.items()}
