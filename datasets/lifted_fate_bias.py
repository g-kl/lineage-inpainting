"""Lift fate-bias 1D dynamics into mouse-embryo scVI space.

The branching topology, observation times, and ground-truth scalars come
from a :class:`datasets.fate_bias.FateBiasDataset` simulation. Each 1D
scalar (leaf state + internal ``gt_node_states``) is replaced with a real
mouse-embryo scVI vector via a random 1D projection slice: project every
embryo cell onto a random unit vector, rescale the synthetic scalars to
the projection range, then nearest-neighbour-match.

Exposes the same interface as :class:`datasets.fate_bias.FateBiasDataset`
(``records``, ``leaf_states``, ``states``, ``reference_features``,
``gt_node_states``, ``x_dim``). Additionally stores:

* ``projection_w`` -- the unit vector used to slice scVI space.
* ``leaf_scalars_1d`` / ``gt_scalars_1d`` -- the original 1D values, so
  eval can compare projected lifted samples against the 1D ground truth.
* ``leaf_cell_idx`` / ``gt_cell_idx`` -- index of the chosen embryo cell
  per node (into ``reference_features``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from datasets.fate_bias import FateBiasDataset
from datasets.mouse_embryo import MouseEmbryoDataset


class LiftedFateBiasDataset:
    """Fate-bias 1D dynamics lifted into mouse-embryo scVI space."""

    def __init__(
        self,
        # base 1D dynamics
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
        # real-data bank
        embryo_data_dir: str = "../data/embryos",
        embryo_states_filename: str = "qiu24_integrated_hvgs_and_scvi.h5ad",
        embryo_trees_filename: str = "final_topology.h5td",
        embryo_annotation_filename: str = "final_subtype.csv",
        embryo_umap_filename: str = "final_umap.csv",
        embedding_key: str = "X_scvi",
        embryo_cache_dir: str = "cache",
        # projection
        projection_seed: int = 0,
    ):
        # Default base-sim obs_times to mouse-embryo time bins so NN-snap
        # poolers find real cells at each leaf time.
        if obs_times is None:
            obs_times = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

        base = FateBiasDataset(
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
            obs_times=obs_times,
            seed=seed,
            n_trees=n_trees,
            ref_seed=None,  # we override the reference bank with real cells
        )

        embryo = MouseEmbryoDataset(
            data_dir=embryo_data_dir,
            states_filename=embryo_states_filename,
            trees_filename=embryo_trees_filename,
            annotation_filename=embryo_annotation_filename,
            umap_filename=embryo_umap_filename,
            embedding_key=embedding_key,
            cache_dir=embryo_cache_dir,
        )

        ref = np.asarray(embryo.reference_features, dtype=np.float32)
        D = ref.shape[1]

        rng = np.random.default_rng(projection_seed)
        w = rng.standard_normal(D).astype(np.float32)
        w /= np.linalg.norm(w) + 1e-12
        proj = ref @ w  # (n_cells,)

        leaf_names = list(base.leaf_states.keys())
        leaf_scalars = np.array(
            [float(base.leaf_states[n].item()) for n in leaf_names],
            dtype=np.float32,
        )

        gt_names = list(base.gt_node_states.keys())
        gt_scalars = np.array(
            [float(base.gt_node_states[n].item()) for n in gt_names],
            dtype=np.float32,
        )

        all_scalars = np.concatenate([leaf_scalars, gt_scalars])
        s_min, s_max = float(all_scalars.min()), float(all_scalars.max())
        p_min, p_max = float(proj.min()), float(proj.max())
        if s_max - s_min < 1e-9:
            scaled = np.full_like(all_scalars, 0.5 * (p_min + p_max))
        else:
            scaled = p_min + (all_scalars - s_min) * (p_max - p_min) / (s_max - s_min)
        scaled_leaf = scaled[: len(leaf_scalars)]
        scaled_gt = scaled[len(leaf_scalars) :]

        order = np.argsort(proj)
        proj_sorted = proj[order]

        def _nn(values: np.ndarray) -> np.ndarray:
            pos = np.searchsorted(proj_sorted, values)
            pos = np.clip(pos, 1, len(proj_sorted) - 1)
            left = proj_sorted[pos - 1]
            right = proj_sorted[pos]
            choose_left = (values - left) <= (right - values)
            sorted_idx = np.where(choose_left, pos - 1, pos)
            return order[sorted_idx]

        leaf_cell_idx_arr = _nn(scaled_leaf)
        gt_cell_idx_arr = _nn(scaled_gt)

        leaf_states = {
            name: torch.from_numpy(ref[leaf_cell_idx_arr[i]].copy())
            for i, name in enumerate(leaf_names)
        }
        gt_node_states = {
            name: torch.from_numpy(ref[gt_cell_idx_arr[i]].copy())
            for i, name in enumerate(gt_names)
        }

        self.records = base.records
        self.leaf_states = leaf_states
        self.gt_node_states = gt_node_states
        # ``states`` and ``reference_features`` must stay index-aligned --
        # utils.reference.reference_cells reads (reference_features[i], states.obs["time"][i]).
        # Expose the full mouse-embryo bank so OT/NN-snap poolers sample from
        # real cells at the right embryo timepoints.
        self.states = embryo.states
        self.reference_features = ref
        self.x_dim = D

        self.projection_w = w
        self.projection_scale = (s_min, s_max, p_min, p_max)
        self.leaf_scalars_1d = {n: float(leaf_scalars[i]) for i, n in enumerate(leaf_names)}
        self.gt_scalars_1d = {n: float(gt_scalars[i]) for i, n in enumerate(gt_names)}
        self.leaf_cell_idx = {n: int(leaf_cell_idx_arr[i]) for i, n in enumerate(leaf_names)}
        self.gt_cell_idx = {n: int(gt_cell_idx_arr[i]) for i, n in enumerate(gt_names)}

        self.sim_kwargs = dict(base.sim_kwargs)
        self.train_obs_times = base.train_obs_times
        self.train_seed_base = base.train_seed_base
        self.projection_seed = int(projection_seed)

    def project(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Project a (..., D) batch of lifted samples back to 1D via w.

        Useful for synthetic eval: compare ``project(predicted)`` against
        ``leaf_scalars_1d`` / ``gt_scalars_1d``.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32) @ self.projection_w
