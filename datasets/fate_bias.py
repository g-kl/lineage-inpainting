"""Hydra-compatible wrapper around the fate-bias synthetic generator.

Multi-seed dataset of independent fate-bias simulations, each observed at a
single (randomly chosen) ``obs_time``. Mirrors lineage-tracing data where
every embryo is a separate SDE realization.
"""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import torch

from datasets.synthetic import make_fate_bias, make_multi_seed_records


class FateBiasDataset:
    """Two-stage fate-bias branching process, multi-seed.

    Exposes the same interface as
    :class:`datasets.hierarchical_ou.HierarchicalOUDataset`
    (``records``, ``leaf_states``, ``states``, ``reference_features``,
    ``gt_node_states``, ``x_dim``).

    Generations 0..split_gen-1 are pure Brownian motion; at ``split_gen``
    sisters diverge to primed attractors; ``n_primed_generations`` later they
    commit to strong fates.
    """

    def __init__(
        self,
        n_generations: int = 10,
        split_gen: int = 3,
        n_primed_generations: int = 6,
        sigma: float = 0.3,
        theta: float = 2.0,
        mu_1_primed: float = 0.2,
        mu_2_primed: float = -0.2,
        mu_1: float = 1.0,
        mu_2: float = -1.0,
        fate_prob: float = 0.75,
        branch_interval: float = 1.0,
        steps_per_interval: int = 200,
        obs_times: Optional[list[float]] = None,
        seed: int = 1000,
        n_trees: int = 200,
        ref_seed: Optional[int] = 9999,
    ):
        if obs_times is None:
            obs_times = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        records, leaf_states, gt_node_states = make_multi_seed_records(
            n_trees=n_trees,
            seed_base=seed,
            obs_times=tuple(obs_times),
            sim_factory=make_fate_bias,
            n_generations=n_generations,
            split_gen=split_gen,
            n_primed_generations=n_primed_generations,
            sigma=sigma,
            theta=theta,
            mu_1_primed=mu_1_primed,
            mu_2_primed=mu_2_primed,
            mu_1=mu_1,
            mu_2=mu_2,
            fate_prob=fate_prob,
            branch_interval=branch_interval,
            steps_per_interval=steps_per_interval,
        )

        self.records = records
        self.leaf_states = leaf_states
        self.states = _build_leaf_anndata(records, leaf_states)
        # The training leaves, kept under their own name because ``states`` is
        # rebound to the reference bank below when ``ref_seed`` is set.
        self.train_states = self.states
        self.gt_node_states = gt_node_states
        self.x_dim = 1

        # Bind the OT reference bank from a *separate* simulation so the pool's
        # snap targets are not the training leaves themselves.
        if ref_seed is not None:
            ref_emb, ref_times = _build_reference_bank(
                ref_seed=int(ref_seed),
                obs_times=tuple(obs_times),
                n_generations=n_generations,
                split_gen=split_gen,
                n_primed_generations=n_primed_generations,
                sigma=sigma,
                theta=theta,
                mu_1_primed=mu_1_primed,
                mu_2_primed=mu_2_primed,
                mu_1=mu_1,
                mu_2=mu_2,
                fate_prob=fate_prob,
                branch_interval=branch_interval,
                steps_per_interval=steps_per_interval,
            )
            self._ref_emb = ref_emb
            self._ref_times = ref_times
            self.reference_features = ref_emb.numpy()
            self.states = ad.AnnData(
                X=self.reference_features,
                obs=pd.DataFrame(
                    {"time": ref_times.numpy()},
                    index=[f"ref_{i}" for i in range(ref_emb.shape[0])],
                ),
            )
        else:
            self._ref_emb = None
            self._ref_times = None
            self.reference_features = self.states.X

        # Preserve simulation parameters so eval can rebuild matching scans
        # without re-deriving them from the config.
        self.sim_kwargs = dict(
            n_generations=n_generations,
            split_gen=split_gen,
            n_primed_generations=n_primed_generations,
            sigma=sigma,
            theta=theta,
            mu_1_primed=mu_1_primed,
            mu_2_primed=mu_2_primed,
            mu_1=mu_1,
            mu_2=mu_2,
            fate_prob=fate_prob,
            branch_interval=branch_interval,
            steps_per_interval=steps_per_interval,
        )
        self.train_obs_times = tuple(obs_times)
        self.train_seed_base = int(seed)
        self.ref_seed = int(ref_seed) if ref_seed is not None else None


def _build_reference_bank(ref_seed, obs_times, **sim_kw):
    """Gather (embedding, time) pairs from one sim across all obs_times."""
    res = make_fate_bias(
        seed=ref_seed,
        obs_times=np.array(obs_times),
        **sim_kw,
    )
    _, _, states = res.to_tree_records()
    emb = torch.tensor(states.X, dtype=torch.float32)
    times = torch.tensor(
        states.obs["time"].to_numpy().astype(float), dtype=torch.float32
    )
    return emb, times


def _build_leaf_anndata(records, leaf_states) -> ad.AnnData:
    """Assemble an AnnData of leaf states with obs["time"] from records."""
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
    X = np.stack(
        [leaf_states[n].numpy() for n in names], axis=0
    ).astype(np.float32)
    obs = pd.DataFrame({"time": [leaf_time[n] for n in names]}, index=names)
    return ad.AnnData(X=X, obs=obs)
