"""Load a run's pooler / splitter and derive its coupling -- one implementation.

Share-aware loading of the pool- or split-side model from its EMA checkpoint,
plus the context-aware nearest-neighbour coupling over the dataset trees.

Kept separate from the light ``annotations`` / ``metrics`` / ``checkpoints``
modules so importing the spine for label maps or the config decode never drags in
torch. Import these explicitly: ``from analysis.lib.eval_core.models import ...``.
"""
from __future__ import annotations

import hydra
import torch

from analysis.lib.eval_core.checkpoints import resolve_checkpoint
from utils.experiment_utils import get_experiment_info


def load_config(experiment_dir):
    """The run's resolved hydra config (``get_experiment_info(...)['config']``)."""
    return get_experiment_info(experiment_dir)["config"]


def _instantiate(cfg, key, ckpt_path, device):
    model = hydra.utils.instantiate(cfg[key]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def load_pooler(experiment_dir, step=None, *, device="cuda", cfg=None):
    """``(pooler, step)`` from ``pool_ema_step{step}.pt`` (the ``pooler`` config)."""
    cfg = cfg or load_config(experiment_dir)
    path, step = resolve_checkpoint(experiment_dir, step, which="pool")
    return _instantiate(cfg, "pooler", path, device), step


def load_splitter(experiment_dir, step=None, *, device="cuda", cfg=None):
    """``(splitter, step)``, share-aware.

    Shared runs (``share_pool_split=true``) have one model instance: instantiate
    from ``pooler`` and load ``pool_ema_step``. Unshared runs (the grid)
    instantiate from ``splitter`` and load ``split_ema_step`` -- otherwise the eval
    silently scores the pool model (e.g. a DeltaTreeFormer) instead of the bridge.
    """
    cfg = cfg or load_config(experiment_dir)
    shared = bool(cfg.get("share_pool_split", False))
    which, key = ("pool", "pooler") if shared else ("split", "splitter")
    path, step = resolve_checkpoint(experiment_dir, step, which=which)
    return _instantiate(cfg, key, path, device), step


def couple(model, dataset, *, pool_n_steps=1, sample_time_gap=0.5, min_snap_time=7.0):
    """Per-node coupling for ``model`` over the dataset trees.

    Wraps the canonical context-aware nearest-neighbour coupler. ``pool_n_steps``
    is the multi-step flow integration for poolers that expose ``n_steps``
    (BridgeTreeFlow); one-step heads (TreeFormer) ignore it.
    """
    from analysis.lib.coupling.nearest_neighbor_coupling_w_context import couple_all_trees
    return couple_all_trees(
        model, dataset.trees, dataset.states,
        pool_n_steps=pool_n_steps, sample_time_gap=sample_time_gap,
        min_snap_time=min_snap_time)
