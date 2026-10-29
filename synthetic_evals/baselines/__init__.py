"""model-agnostic baselines for the synthetic pool-split evals.

each depends only on the dataset and its transitions, never on the pool/split model under
evaluation, so one trained artifact is shared across benchmarks and model variants (see
cache.py). submodules expose train / predict / adjacent_pairs plus a per-method *Bank
container; registry.py maps short names (fm / gs / gf / ...) onto them.
"""

from __future__ import annotations

from synthetic_evals.baselines import (  # noqa: F401
    genot_fgw,
    genot_fgw_fb,
    genot_sinkhorn,
    marginal_fm,
    moscot_ot,
    registry,
)
from synthetic_evals.baselines.cache import train_or_load  # noqa: F401

__all__ = [
    "marginal_fm",
    "genot_sinkhorn",
    "genot_fgw",
    "genot_fgw_fb",
    "moscot_ot",
    "registry",
    "train_or_load",
]
