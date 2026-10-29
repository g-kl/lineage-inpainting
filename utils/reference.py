"""Pull a real-cell reference bank off a dataset.

Snap poolers and marginal aligners want a flat bank of real-cell
``(embeddings, times)`` before training. Rather than have ``main.py`` assemble
it, every consumer receives the whole ``dataset`` (via a uniform
``bind_dataset(dataset)`` hook) and those that need cells call this helper, so
the HVG / log1p / scVI feature matrix is read the same way everywhere. (Bank
extenders read ``dataset.records`` directly -- no helper needed.)
"""

from __future__ import annotations

import numpy as np
import torch


def reference_cells(
    dataset,
    *,
    clonal_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(embeddings, times)`` for a dataset's real-cell bank.

    Reads the dataset's ``reference_features`` (the same feature matrix it built
    ``leaf_states`` from -- HVG / log1p / scVI) and the index-aligned per-cell
    time from ``states.obs['time']``.

    Parameters
    ----------
    dataset
        Exposes ``reference_features`` (n_cells, dim) and a ``states`` AnnData
        whose ``obs`` carries ``time`` (and, when ``clonal_only``, ``clone``).
    clonal_only
        If True, restrict to clonally-tracked cells -- ``states.obs['clone']``
        not NaN.
    """
    states = dataset.states
    ref = np.asarray(dataset.reference_features)
    times = states.obs["time"].values.astype(float)

    if clonal_only:
        if "clone" not in states.obs.columns:
            raise ValueError(
                "clonal_reference is on but this dataset has no 'clone' column "
                "in states.obs (it carries no clonal annotation). Set "
                "clonal_reference=False for this dataset."
            )
        keep = ~states.obs["clone"].isna().values
        ref = ref[keep]
        times = times[keep]

    return (
        torch.as_tensor(ref, dtype=torch.float32),
        torch.as_tensor(times, dtype=torch.float32),
    )
