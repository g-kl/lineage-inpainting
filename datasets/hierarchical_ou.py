"""Hydra-compatible wrapper around the hierarchical-OU synthetic generator.

Two regimes, picked by ``n_trees``:

* ``n_trees=1`` (default) -- one SDE realization observed at all
  ``obs_times``; aligner pairs records from the same simulation.
* ``n_trees>1`` -- that many independent simulations, each observed at a
  single randomly-chosen obs_time (mirrors lineage-tracing data
  structure where each embryo is a different SDE realization). Builds
  via :func:`datasets.synthetic.make_multi_seed_records`.
"""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import torch

from datasets.synthetic import (
    RandomProjector,
    make_hierarchical_ou,
    make_multi_seed_records,
)


class HierarchicalOUDataset:
    """Synthetic hierarchical Ornstein-Uhlenbeck branching process.

    Exposes the same interface as
    :class:`datasets.mouse_embryo.MouseEmbryoDataset` (``records``,
    ``leaf_states``, ``x_dim``).

    Parameters
    ----------
    n_generations : int
    split_gen : int
    gap : int
    sigma : float
    obs_times : list of float, optional
        Defaults to ``[5, 6, 7, 8]``. With ``n_trees=1`` every observation
        time produces a record from the single sim; with ``n_trees>1`` each
        tree is randomly assigned one of these obs_times.
    seed : int
        Base seed. Tree i uses ``seed + i`` when ``n_trees>1``.
    theta : float
    branch_interval : float
    steps_per_interval : int
    n_trees : int
        ``1`` = single sim observed at all obs_times. ``>1`` = that many
        independent sims, each observed at one randomly-chosen obs_time.
    """

    def __init__(
        self,
        n_generations: int = 10,
        split_gen: int = 5,
        gap: int = 3,
        sigma: float = 0.1,
        obs_times: Optional[list[float]] = None,
        seed: int = 0,
        theta: float = 2.0,
        branch_interval: float = 1.0,
        steps_per_interval: int = 200,
        n_trees: int = 1,
        attractors_1: Optional[tuple] = None,
        attractors_2: Optional[dict] = None,
        proj_dim: Optional[int] = None,
        proj_seed: int = 0,
        proj_noise: float = 0.1,
    ):
        if obs_times is None:
            obs_times = [5.0, 6.0, 7.0, 8.0]

        extra = {}
        if attractors_1 is not None:
            extra["attractors_1"] = attractors_1
        if attractors_2 is not None:
            extra["attractors_2"] = attractors_2

        if n_trees > 1:
            records, leaf_states, gt_node_states = make_multi_seed_records(
                n_trees=n_trees,
                seed_base=seed,
                obs_times=tuple(obs_times),
                n_generations=n_generations,
                split_gen=split_gen,
                gap=gap,
                sigma=sigma,
                theta=theta,
                branch_interval=branch_interval,
                steps_per_interval=steps_per_interval,
                **extra,
            )
            self.sim_res = None
            states = _build_leaf_anndata(records, leaf_states)
        else:
            sim_res = make_hierarchical_ou(
                n_generations=n_generations,
                split_gen=split_gen,
                gap=gap,
                sigma=sigma,
                obs_times=np.array(obs_times),
                seed=seed,
                theta=theta,
                branch_interval=branch_interval,
                steps_per_interval=steps_per_interval,
                **extra,
            )
            records, leaf_states, states = sim_res.to_tree_records()
            self.sim_res = sim_res
            gt_node_states = {
                node: torch.tensor(attrs["state"], dtype=torch.float32)
                for node, attrs in sim_res.tree.nodes(data=True)
                if "state" in attrs
            }

        # Optional high-dimensional embedding: project the 1-D latent states
        # up to ``proj_dim`` along a fixed random direction (+ small noise).
        # The projector is shared (by seed) with the held-out eval sim so both
        # use the same direction; see ``datasets.synthetic.RandomProjector``.
        self.projector = None
        if proj_dim is not None:
            self.projector = RandomProjector(
                dim_in=1, dim_out=int(proj_dim),
                seed=proj_seed, noise=proj_noise,
            )
            leaf_states = {
                n: torch.as_tensor(
                    self.projector.project(v.numpy() if torch.is_tensor(v) else v),
                    dtype=torch.float32,
                )
                for n, v in leaf_states.items()
            }
            gt_node_states = {
                n: torch.as_tensor(
                    self.projector.project(v.numpy() if torch.is_tensor(v) else v),
                    dtype=torch.float32,
                )
                for n, v in gt_node_states.items()
            }
            if states.n_obs > 0:
                states = ad.AnnData(
                    X=self.projector.project(states.X).astype(np.float32),
                    obs=states.obs.copy(),
                )

        self.records = records
        self.leaf_states = leaf_states
        # AnnData of leaf states with obs["time"] -- consumed by main.py to
        # bind reference cells for OT/NN snap poolers and time-aware aligners.
        self.states = states
        self.reference_features = states.X
        # Birth state of every node in the underlying SDE simulation
        # (leaves + internals). Used as ground truth in synthetic evals;
        # not the same as ``states`` which is observed-leaf only.
        self.gt_node_states = gt_node_states
        # 1-D OU process, optionally embedded into ``proj_dim`` ambient dims.
        self.x_dim = int(proj_dim) if proj_dim is not None else 1


def _build_leaf_anndata(records, leaf_states) -> ad.AnnData:
    """Assemble an AnnData of leaf states with obs["time"] from records.

    Each leaf appears in exactly one record (the one for its obs time), so
    we can read off its time from ``rec.leaf_time``.
    """
    leaf_time: dict[str, float] = {}
    for rec in records.values():
        for subs in rec.subtree_index.values():
            for st in subs:
                g = st["graph"]
                for n in g.nodes:
                    if g.out_degree(n) == 0:
                        leaf_time[n] = float(rec.leaf_time)

    names = [n for n in leaf_states if n in leaf_time]
    if not names:
        return ad.AnnData()
    X = np.stack([leaf_states[n].numpy() for n in names], axis=0).astype(np.float32)
    obs = pd.DataFrame({"time": [leaf_time[n] for n in names]}, index=names)
    return ad.AnnData(X=X, obs=obs)
