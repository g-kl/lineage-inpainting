"""Resolve EMA checkpoints in an experiment dir (grid naming).

One implementation of the ``pool_ema_step{N}.pt`` / ``split_ema_step{N}.pt``
resolution. ``step=None`` picks the latest available checkpoint.
"""
from __future__ import annotations

import os

_PREFIX = {"pool": "pool_ema_step", "split": "split_ema_step"}


def _ema_step(fname, prefix):
    return int(fname[len(prefix):-len(".pt")])


def resolve_checkpoint(experiment_dir, step=None, which="pool"):
    """``(checkpoint_path, step)`` for the pool- or split-side EMA checkpoint.

    ``which`` is ``"pool"`` or ``"split"``. ``step=None`` -> latest.
    """
    prefix = _PREFIX[which]
    ckpts = [f for f in os.listdir(experiment_dir)
             if f.startswith(prefix) and f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"no {prefix}*.pt in {experiment_dir}")
    if step is None:
        fname = max(ckpts, key=lambda f: _ema_step(f, prefix))
        step = _ema_step(fname, prefix)
    else:
        fname = f"{prefix}{step}.pt"
        if fname not in ckpts:
            raise FileNotFoundError(f"{fname} not in {experiment_dir}")
    return os.path.join(experiment_dir, fname), step
