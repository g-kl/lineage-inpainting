"""Train-once / load-many cache for the model-agnostic eval baselines.

The marginal-FM and GENOT baselines depend only on the dataset (its
transitions) and their own hyperparameters, never on the pool/split model under
evaluation. Retraining them per model variant would repeat the same ~20k-step
GENOT solve on the same data. This module trains each baseline once, keyed by

    (method, resolved dataset config, baseline hyperparameters),

and caches it under ``outputs/_synth_baselines/`` so any later eval -- of any
model variant -- reuses it.

Serialisation is framework-aware. Torch baselines (marginal-FM) are pickled
whole via ``torch.save``. GENOT baselines wrap a flax ``nnx`` model, which is
persisted via ``nnx.split`` -> (graphdef, state) and restored with
``nnx.merge`` (round-trips exactly). Everything else in the GENOT package is plain Python / arrays.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("baselines.cache")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "outputs" / "_synth_baselines"


# --------------------------------------------------------------------------
#  Cache key
# --------------------------------------------------------------------------


def _stable_hash(payload: dict[str, Any]) -> str:
    """Deterministic short hash of a JSON-able payload."""
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:16]


def cache_key(method: str, dataset_cfg, params: dict[str, Any]) -> str:
    """Hash identifying a trained baseline.

    ``dataset_cfg`` is the (resolved) dataset config -- the data the baseline
    is trained on. ``params`` are the baseline's own hyperparameters (steps,
    seed, transition pairs, architecture, coupling mode, ...). Two evals that
    agree on all of these get the same key, hence the same cached artifact.
    """
    if isinstance(dataset_cfg, DictConfig):
        ds = OmegaConf.to_container(dataset_cfg, resolve=True)
    else:
        ds = dict(dataset_cfg)
    return _stable_hash({"method": method, "dataset": ds, "params": params})


# --------------------------------------------------------------------------
#  Framework-aware (de)serialisation
# --------------------------------------------------------------------------


def _is_genot_pkg(obj) -> bool:
    return isinstance(obj, dict) and "model" in obj


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if _is_genot_pkg(obj):
        import cloudpickle
        import flax.nnx as nnx
        import jax
        import numpy as np

        graphdef, state = nnx.split(obj["model"])
        state = jax.tree.map(lambda a: np.asarray(a), state)
        blob = {k: v for k, v in obj.items() if k != "model"}
        blob["__nnx_graphdef__"] = graphdef
        blob["__nnx_state__"] = state
        # cloudpickle, not pickle: the velocity-field MLP bakes its activation
        # (jax.nn.silu) into the nnx graphdef, and plain pickle can't serialise
        # that function reference. cloudpickle pickles it by value.
        with open(tmp, "wb") as f:
            cloudpickle.dump(blob, f)
    else:  # torch module (marginal-FM velocity field, carries t_min/t_span)
        import torch

        torch.save(obj, tmp)
    tmp.replace(path)


def _load(path: Path):
    with open(path, "rb") as f:
        head = f.read(2)
    # torch.save streams start with the zip magic 'PK'; our genot blobs don't.
    if head == b"PK":
        import torch

        # Load onto the active device so a cache-hit baseline matches the
        # GPU eval inputs (GENOT predict is device-agnostic; the torch
        # marginal-FM model is not -- a CPU-loaded model would mismatch).
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.load(path, map_location=device, weights_only=False)
    import cloudpickle

    with open(path, "rb") as f:
        blob = cloudpickle.load(f)
    if isinstance(blob, dict) and "__nnx_graphdef__" in blob:
        import flax.nnx as nnx

        model = nnx.merge(blob.pop("__nnx_graphdef__"), blob.pop("__nnx_state__"))
        blob["model"] = model
        return blob
    return blob


# --------------------------------------------------------------------------
#  Train-or-load
# --------------------------------------------------------------------------


def train_or_load(
    *,
    method: str,
    label: str,
    dataset_cfg,
    params: dict[str, Any],
    train_fn: Callable[[], Any],
    cache_dir: Path | str | None = None,
    force_retrain: bool = False,
):
    """Return a trained baseline, training it only if not already cached.

    ``method`` selects the implementation (used in the key and filename);
    ``label`` is a human tag for logging (e.g. ``"fm_rev"``). ``params`` must
    capture every hyperparameter that changes the result. ``train_fn`` is a
    no-arg closure that trains and returns the model / package.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    key = cache_key(method, dataset_cfg, params)
    path = cache_dir / f"{method}_{key}.pkl"

    if path.exists() and not force_retrain:
        logger.info("[%s] cache hit -> %s", label, path.name)
        return _load(path)

    logger.info("[%s] cache miss; training (key=%s)", label, key)
    obj = train_fn()
    _save(obj, path)
    logger.info("[%s] cached -> %s", label, path.name)
    return obj
