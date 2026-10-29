#!/usr/bin/env python
"""viz.energy_distance is the shared U-statistic energy distance, and the extra metrics behave.

checks the alternate metrics used by the robustness panel.

no GPU or data needed:
    PYTHONPATH=. python tests/test_metrics_agreement.py
"""

from __future__ import annotations

import sys

import numpy as np

from synthetic_evals import viz
from synthetic_evals.core import metrics as M

_failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


rng = np.random.default_rng(0)
a = rng.normal(0, 1, (400, 5)).astype(np.float32)
b = rng.normal(1.5, 1, (350, 5)).astype(np.float32)
a2 = rng.normal(0, 1, (380, 5)).astype(np.float32)   # same law as a


## viz.energy_distance vs the shared U-statistic ##

print("energy distance: viz is the shared unbiased U-statistic")
from analysis.lib.metrics.energy_distance import energy_distance as shared


def _pd(u, v):
    return np.linalg.norm(u[:, None, :].astype(np.float64) - v[None, :, :].astype(np.float64), axis=-1)


def _ustat(u, v):
    n, m = len(u), len(v)
    return (2 * _pd(u, v).mean() - _pd(u, u).sum() / (n * (n - 1))
            - _pd(v, v).sum() / (m * (m - 1)))


ours = viz.energy_distance(a, b)
check("viz.energy_distance IS the shared implementation", viz.energy_distance(a, b) == shared(a, b))
check("the shared implementation is the U-statistic", abs(ours - _ustat(a, b)) < 1e-9,
      f"viz={ours:.10f} numpy={_ustat(a, b):.10f}")
check("symmetric in its two clouds",
      abs(ours - viz.energy_distance(b, a)) < 1e-9,
      f"ab={ours:.10f} ba={viz.energy_distance(b, a):.10f}")
# unbiased: on one law the value is 0 up to sampling noise, of either sign.
same_law = viz.energy_distance(a, a2)
check("~0 on one law", abs(same_law) < 0.01 * ours,
      f"same-law {same_law:.6f} vs different-law {ours:.4f}")

# a single reference point (recon_figure's per-ancestor reverse energy) is a
# degenerate but legal cloud: no self term, so the value is the energy score of
# the sample cloud against that point, which is >= 0.
one = viz.energy_distance(a, b[:1])
check("single reference point (n_b == 1) is the energy score, finite and >= 0",
      np.isfinite(one) and one >= 0.0 and abs(one - (2 * _pd(a, b[:1]).mean() - _pd(a, a).sum() / (len(a) * (len(a) - 1)))) < 1e-9,
      f"{one:.4f}")


## every metric separates different laws and vanishes on the same law ##

print("\nseparation")
diff = M.distances(a, b)
same = M.distances(a, a2)
for m in M.NAMES:
    check(f"{m}: different laws > same law", diff[m] > same[m],
          f"{diff[m]:.4f} vs {same[m]:.4f}")
    ratio = diff[m] / max(abs(same[m]), 1e-12)
    check(f"{m}: separation ratio > 5", ratio > 5.0,
          f"diff={diff[m]:.4f} same={same[m]:.4f} ratio={ratio:.1f}")


## the median heuristic keeps the metrics alive at the 50-d scVI scale ##

print("\nseparation at d=50 (scVI latent scale)")
hi_a = rng.normal(0, 1, (400, 50)).astype(np.float32)
hi_b = rng.normal(1.5, 1, (350, 50)).astype(np.float32)
hi_a2 = rng.normal(0, 1, (380, 50)).astype(np.float32)
hd, hs = M.distances(hi_a, hi_b), M.distances(hi_a, hi_a2)
for m in M.NAMES:
    ratio = hd[m] / max(abs(hs[m]), 1e-12)
    check(f"{m}: separates at d=50 (ratio > 5)", ratio > 5.0,
          f"diff={hd[m]:.4f} same={hs[m]:.4f} ratio={ratio:.1f}")


## grouped aggregation matches _grouped_energy ##

print("\ngrouped aggregation")
from synthetic_evals.cell_type_crossing_eval import _grouped_energy

R, S_, d = 60, 4, 3
true_rows = rng.normal(0, 1, (R, d)).astype(np.float32)
groups = np.array([0] * (R // 2) + [1] * (R - R // 2))
pred = rng.normal(0.3, 1, (S_, R, d)).astype(np.float32)

want = _grouped_energy(pred, true_rows, groups)
got = M.grouped_distances(pred, true_rows, groups)
check("grouped energy matches _grouped_energy", abs(want - got["energy"]) < 1e-9,
      f"{want:.8f} vs {got['energy']:.8f}")
check("all metric names present", set(got) == set(M.NAMES), str(sorted(got)))
check("grouped values finite", all(np.isfinite(v) for v in got.values()),
      str({k: round(v, 4) for k, v in got.items()}))


## SUMMARY ##

print("\n" + "=" * 60)
if _failures:
    print(f"FAILED {len(_failures)} check(s): {_failures}")
    sys.exit(1)
print("all checks passed.")
