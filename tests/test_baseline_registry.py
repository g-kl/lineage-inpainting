#!/usr/bin/env python
"""registry invariants: every baseline is displayable, and forward/reverse prediction
tiles deterministic maps and varies the seed for stochastic ones.

no GPU or data needed:
    PYTHONPATH=. python tests/test_baseline_registry.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from synthetic_evals.baselines import registry as R
from synthetic_evals.baselines.registry import Baseline
from synthetic_evals.core import inference as inf
from utils import palettes as P

_failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


def stub(*, stochastic):
    if stochastic:
        return Baseline(name="s", label="s", cache_method="s", stochastic=True,
                        dim_of=lambda p: 3, train=None,
                        predict=lambda p, x, a, b, seed=0: x + seed)
    return Baseline(name="d", label="d", cache_method="d", stochastic=False,
                    dim_of=lambda p: 3, train=None,
                    predict=lambda p, x, a, b, n_steps=50: x * n_steps)


## every registered baseline is displayable ##

print("palette entries")
for name in R.names():
    bl = R.get(name)
    check(f"{name}: colour/label/marker/order present",
          all(name in d for d in (P.METHOD_COLORS, P.METHOD_LABELS,
                                  P.METHOD_MARKERS)) and name in P.METHOD_ORDER)
    check(f"{name}: label matches palettes", bl.label == P.METHOD_LABELS[name],
          f"{bl.label!r}")


## predict kwargs follow stochasticity ##

print("\npredict kwargs")
for name in R.names():
    bl = R.get(name)
    want = {"seed": 0} if bl.stochastic else {"n_steps": 20}
    check(f"{name}: kwargs at n_flow_steps=20",
          R.predict_kwargs(bl, n_flow_steps=20) == want, str(want))
    check(f"{name}: None omits n_steps",
          "n_steps" not in R.predict_kwargs(bl, n_flow_steps=None))


## forward sampling ##

print("\nforward_samples")
x = np.arange(12, dtype=np.float32).reshape(4, 3)

det = inf.forward_samples(stub(stochastic=False), None, x, 0.0, 1.0,
                          n_samples=3, n_flow_steps=20)
check("deterministic: one pass tiled n_samples",
      det.shape == (3, 4, 3) and np.array_equal(
          det, np.broadcast_to((x * 20)[None], (3, 4, 3))))

sto = inf.forward_samples(stub(stochastic=True), None, x, 0.0, 1.0,
                          n_samples=3, n_flow_steps=20)
check("stochastic: seeds 0..n-1",
      np.array_equal(sto, np.stack([x + s for s in range(3)], axis=0)))

one = inf.forward_samples(stub(stochastic=False), None, np.ones((2, 3), np.float32),
                          0.0, 1.0, n_samples=1, n_flow_steps=None)
check("n_flow_steps=None uses the baseline's own default (50)",
      np.array_equal(one[0], np.full((2, 3), 50.0, np.float32)))


## reverse (per-parent) sampling ##

print("\nreverse_per_parent")
subtrees = [
    {"x_src": np.array([1.0, 2.0, 3.0], np.float32),
     "leaves": [("a", 0), ("b", 0)],
     "leaf_states": {"a": torch.zeros(3), "b": torch.zeros(3)}},
    {"x_src": np.array([4.0, 5.0, 6.0], np.float32),
     "leaves": [("c", 0), ("d", 0)],
     "leaf_states": {"c": torch.zeros(3), "d": torch.zeros(3)}},
]
pairs = [(None,) * 5 + (x[i],) for i in range(4)]

for stochastic in (False, True):
    tag = "stochastic" if stochastic else "deterministic"
    bl = stub(stochastic=stochastic)
    got = inf.reverse_per_parent(bl, None, subtrees, pairs, 0.0, 1.0,
                                 n_samples=2, n_flow_steps=20)
    check(f"{tag}: per-parent shapes",
          got["parent_pred"].shape == (2, 2, 3)
          and got["parent_true"].shape == (2, 3)
          and got["parent_children_mean"].shape == (2, 3)
          and got["child_to_parent"].tolist() == [0, 0, 1, 1])
    # children of parent 0 are rows 0,1 -> mean of x*20 over those rows
    if not stochastic:
        check(f"{tag}: averages children per parent",
              np.allclose(got["parent_pred"][0, 0], (x[0] + x[1]) / 2 * 20))
    empty = inf.reverse_per_parent(bl, None, [], [], 0.0, 1.0,
                                   n_samples=2, n_flow_steps=20)
    check(f"{tag}: empty-subtree guard",
          empty["parent_pred"].shape == (2, 0, 3)
          and empty["parent_true"].shape == (0, 3))


## pooler-ablation variants ##

print("\nmodel variants")
from synthetic_evals.crossing_landscape_eval import EXPERIMENT, parse_variants

check("default is a single 'ours' variant",
      parse_variants(None) == {"ours": EXPERIMENT})
check("label=experiment parses",
      parse_variants(["ours=a/b", "ours_meanpool=c/d"])
      == {"ours": "a/b", "ours_meanpool": "c/d"})
try:
    parse_variants(["nope"])
    check("malformed entry raises", False)
except ValueError:
    check("malformed entry raises", True)
for label in P.ABLATION_ORDER:
    check(f"{label}: colour + label present",
          label in P.ABLATION_COLORS and label in P.ABLATION_LABELS)


## SUMMARY ##

print("\n" + "=" * 60)
if _failures:
    print(f"FAILED {len(_failures)} check(s): {_failures}")
    sys.exit(1)
print("all checks passed.")
