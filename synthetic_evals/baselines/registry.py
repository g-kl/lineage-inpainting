"""registry of the model-agnostic baselines, keyed by the short name used everywhere.

each entry carries what every eval needs to know about a baseline: how to train it, how
to predict with it, and whether it is stochastic. bank construction stays at the call
sites because it depends on the dataset flavour, not on the baseline. display names come
from utils.palettes, which is also what the figures read -- the plot half never imports
this module, so it stays free of torch/jax.

to add a baseline: write a module exposing train/predict, register it here, and add a
colour/label/marker/order entry in utils/palettes.py.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

from utils import palettes as P
from synthetic_evals.baselines import genot_fgw, genot_sinkhorn, marginal_fm, moscot_ot


@dataclass(frozen=True)
class Baseline:
    name: str            # key in result dicts, landscape.csv rows and style.METHODS
    label: str           # display name
    cache_method: str    # namespace for baselines.cache filenames
    train: Callable      # (bank, *, dim, pairs, n_steps, seed, **kw) -> model or pkg
    predict: Callable    # (pkg, x, t_from, t_to, **kw) -> (M, dim) tensor
    stochastic: bool     # redraw per seed, vs one deterministic pass tiled n_samples
    dim_of: Callable     # pkg -> output dim


REGISTRY: dict[str, Baseline] = {}


def register(bl: Baseline) -> Baseline:
    REGISTRY[bl.name] = bl
    return bl


def get(name: str) -> Baseline:
    return REGISTRY[name]


def names() -> list[str]:
    return list(REGISTRY)


def predict_kwargs(bl: Baseline, *, n_flow_steps: int | None, seed: int = 0) -> dict:
    """kwargs for bl.predict: stochastic baselines take a seed, deterministic ones steps.

    n_flow_steps=None omits the kwarg so the baseline's own default applies.
    """
    if bl.stochastic:
        return {"seed": seed}
    return {} if n_flow_steps is None else {"n_steps": n_flow_steps}


register(Baseline(
    name="fm", label=P.METHOD_LABELS["fm"], cache_method="marginal_fm",
    train=marginal_fm.train, predict=marginal_fm.predict, stochastic=False,
    dim_of=lambda pkg: int(pkg.net[-1].out_features),
))
register(Baseline(
    name="gs", label=P.METHOD_LABELS["gs"], cache_method="genot_sinkhorn",
    train=genot_sinkhorn.train, predict=genot_sinkhorn.predict, stochastic=True,
    dim_of=lambda pkg: int(pkg["dim"]),
))
register(Baseline(
    name="gf", label=P.METHOD_LABELS["gf"], cache_method="genot_fgw",
    train=genot_fgw.train, predict=genot_fgw.predict, stochastic=True,
    dim_of=lambda pkg: int(pkg["dim"]),
))
register(Baseline(
    name="wot", label=P.METHOD_LABELS["wot"], cache_method="moscot_wot",
    train=functools.partial(moscot_ot.train, lineage=False),
    predict=moscot_ot.predict, stochastic=True,
    dim_of=lambda pkg: int(pkg["dim"]),
))
# moslin gets its own epsilon/alpha: at wot's settings its tree term is too weak to
# move the plan. callers that pass epsilon/alpha explicitly (the SDE core evals)
# override these.
register(Baseline(
    name="moslin", label=P.METHOD_LABELS["moslin"], cache_method="moscot_moslin",
    train=functools.partial(moscot_ot.train, lineage=True, epsilon=1e-4, alpha=0.999),
    predict=moscot_ot.predict, stochastic=True,
    dim_of=lambda pkg: int(pkg["dim"]),
))
