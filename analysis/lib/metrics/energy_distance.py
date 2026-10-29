"""the energy distance between two point clouds: the one estimator every reported
number uses (the real-data distortion, context-scaling and splitter metrics, and the
synthetic benchmarks through synthetic_evals.viz.energy_distance).

    ED(x, y) = 2 E|x - y| - E|x - x'| - E|y - y'|

as the unbiased U-statistic: the self terms average over the n(n-1) / m(m-1)
off-diagonal pairs, full scale. a one-point cloud has no self term (it is 0), which
makes the value the energy score of the other cloud against that point.

not geomloss's "energy" loss: that is the V-statistic at half scale, whose zero
diagonal inflates every value by ~E|x-x'|/n + E|y-y'|/m and so penalises small clouds.
"""
import numpy as np
import torch


def energy_distance(x, y) -> float:
    x = torch.as_tensor(np.asarray(x, dtype=np.float64))
    y = torch.as_tensor(np.asarray(y, dtype=np.float64))
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = x.to(device), y.to(device)
    mode = "donot_use_mm_for_euclid_dist"
    cross = torch.cdist(x, y, compute_mode=mode).mean()
    return float(2 * cross - _self_term(x, mode) - _self_term(y, mode))


def _self_term(z, mode):
    n = len(z)
    if n < 2:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    return torch.cdist(z, z, compute_mode=mode).sum() / (n * (n - 1))
