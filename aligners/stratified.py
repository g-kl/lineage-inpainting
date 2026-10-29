"""Composable stratified feature aligners for completed tree signals.

The core abstraction is small: a feature object maps ``(TreeBatch, states)``
to a dictionary of empirical feature laws keyed by a deterministic stratum.
The aligner compares matching strata across records with a distributional
loss.  This keeps topology-dependent sampling separate from the discrepancy.
"""

from __future__ import annotations

import itertools
from typing import Hashable, Optional

import torch
import torch.nn as nn
from geomloss import SamplesLoss

from utils.energy_scores import energy_distance as _energy_distance


def _energy_direct_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Energy distance via direct subtraction (precision-safe gradient).

    Drop-in for ``SamplesLoss(loss="energy")``: takes ``[N, D]`` / ``[M, D]``
    sample sets, returns a scalar. Halved to match geomloss's ``ED/2``
    convention so loss magnitudes stay comparable. Unlike geomloss's Gram-based
    pairwise distances (which downcast to TF32 and leak ~1e-4 noise into
    ``||u-v||^2``), this keeps a clean gradient near coincident samples -- the
    regime that matters when sibling predictions collapse together.
    """
    return 0.5 * _energy_distance(x.unsqueeze(0), y.unsqueeze(0), beta=1.0)

from datasets.tree_record import TreeBatch
from utils.reference import reference_cells


FeatureLaws = dict[Hashable, torch.Tensor]


def _default_tolerance(time_bins: list[float]) -> float:
    if len(time_bins) > 1:
        return min(b - a for a, b in zip(time_bins, time_bins[1:])) / 2.0
    return 0.5


def _ancestor_table(batch: TreeBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return root-to-node path indices for every node.

    ``path_idx[b, n, d]`` is the node at depth ``d`` on the root-to-``n``
    path. Invalid slots beyond the node's depth repeat the endpoint and are
    marked false in ``on_path``.
    """
    B, N = batch.mask.shape
    D = batch.max_depth + 1
    device = batch.mask.device

    self_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    par_safe = torch.where(batch.parent_idx >= 0, batch.parent_idx, self_idx)

    ancestors = torch.empty(B, N, D, dtype=torch.long, device=device)
    ancestors[..., 0] = self_idx
    ptr = self_idx.clone()
    for k in range(1, D):
        ptr = par_safe.gather(1, ptr)
        ancestors[..., k] = ptr

    depth_n = batch.depth.unsqueeze(-1)
    depth_slots = torch.arange(D, device=device).view(1, 1, -1)
    ancestor_hops = (depth_n - depth_slots).clamp(min=0)
    on_path = depth_slots <= depth_n
    path_idx = ancestors.gather(2, ancestor_hops)
    return path_idx, on_path


class StratifiedFeature(nn.Module):
    """Base class for feature laws consumed by ``StratifiedFeatureAligner``.

    Subclasses implement ``collect(batch, states) -> {stratum_key: tensor}``.
    The aligner then compares the per-stratum tensors across record pairs
    with a distributional loss.
    """

    def collect(self, batch: TreeBatch, states: torch.Tensor) -> FeatureLaws:
        raise NotImplementedError


class AnchoredFeature(StratifiedFeature):
    """Feature variant whose laws compare to a feature-provided anchor.

    ``collect`` returns ``{stratum_key: (query, target)}`` instead of
    ``{stratum_key: tensor}``. ``StratifiedFeatureAligner`` detects the
    ``AnchoredFeature`` subclass and computes ``loss(query, target)`` once
    per record per stratum (no pair-wise iteration), then averages.

    Use this when the target is an external reference (e.g. real cells from
    an annotated cell bank) rather than another record's pooled states.
    """

    def collect(self, batch: TreeBatch, states: torch.Tensor) -> "FeatureLaws":
        raise NotImplementedError


class TimeMarginalFeature(AnchoredFeature):
    """Per-observation-time anchor of pooled states to real cells.

    For each time ``t`` in the bound cell bank's unique observation times,
    the feature contributes one stratum:

      - query  = pooled embeddings at nodes whose ``node_time`` is within
                 ``time_tolerance`` of ``t`` (intersected with the eligible
                 mask: by default, ``skip_observed=True`` so only predicted
                 positions are anchored, regardless of whether they are leaves
                 in the graph).
      - target = up to ``n_reference_samples`` real cells drawn from the
                 bound cell bank at the same time.

    Every step it pulls a fresh random subset of real cells per bin, so the
    reference set is large and uniform across time bins regardless of how
    many subtrees happen to have nodes there.

    Bind the reference bank via ``bind_dataset(dataset)`` before training
    (``main.py`` does this when the dataset exposes a ``states`` AnnData and
    ``reference_features``).
    """

    def __init__(
        self,
        *,
        name: str = "time",
        n_reference_samples: int = 4096,
        skip_observed: bool = True,
        time_tolerance: Optional[float] = None,
        clonal_reference: bool = True,
    ):
        super().__init__()
        # The marginal only constrains positions the model had to predict.
        # Observed real cells already match the reference distribution by
        # construction; with the extender, grafted descendants sit at leaf
        # positions but are unobserved and must still be constrained.
        self.name = name
        self.n_reference_samples = int(n_reference_samples)
        self.skip_observed = bool(skip_observed)
        self.time_tolerance = time_tolerance
        # Match predictions against the clonal reference distribution only.
        self.clonal_reference = clonal_reference
        self._ref_emb: Optional[torch.Tensor] = None
        self._ref_times: Optional[torch.Tensor] = None
        self._ref_unique_times: Optional[torch.Tensor] = None
        self._effective_time_tolerance: Optional[float] = time_tolerance

    def bind_dataset(self, dataset) -> None:
        embeddings, times = reference_cells(dataset, clonal_only=self.clonal_reference)
        self._ref_emb = embeddings.detach()
        self._ref_times = times.detach().float()
        self._ref_unique_times = torch.unique(self._ref_times)
        if self.time_tolerance is None:
            unique = torch.sort(self._ref_unique_times).values
            self._effective_time_tolerance = (
                float((unique[1:] - unique[:-1]).min().item()) / 2.0
                if unique.numel() > 1 else 0.5
            )

    def _sample_reference_at(
        self, t: float, device: torch.device, dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        tol = float(self._effective_time_tolerance or 0.0)
        mask = (self._ref_times - float(t)).abs() <= tol
        if not mask.any():
            return None
        idx = torch.nonzero(mask, as_tuple=True)[0]
        if idx.shape[0] <= self.n_reference_samples:
            choice = idx
        else:
            perm = torch.randperm(idx.shape[0], device=idx.device)
            choice = idx[perm[: self.n_reference_samples]]
        return self._ref_emb[choice].to(device=device, dtype=dtype)

    def collect(self, batch: TreeBatch, states: torch.Tensor) -> FeatureLaws:
        if self._ref_unique_times is None:
            return {}  # graceful degrade when no cell bank bound
        device, dtype = states.device, states.dtype
        tol = float(self._effective_time_tolerance or 0.0)
        eligible = batch.mask.clone()
        if self.skip_observed:
            eligible = eligible & ~batch.observed_mask

        laws: FeatureLaws = {}
        for t in self._ref_unique_times.tolist():
            hits = eligible & ((batch.node_times - float(t)).abs() <= tol)
            if not hits.any():
                continue
            target = self._sample_reference_at(t, device, dtype)
            if target is None:
                continue
            laws[(self.name, float(t))] = (states[hits], target)
        return laws


class StratifiedFeatureAligner(nn.Module):
    """Compare feature laws across records within matching strata."""

    def __init__(
        self,
        features: list[StratifiedFeature],
        *,
        weights: Optional[list[float]] = None,
        loss: str = "energy",
        p: int = 2,
        min_samples: int = 1,
    ):
        super().__init__()
        self.features = nn.ModuleList(features)
        if weights is None:
            weights = [1.0] * len(features)
        if len(weights) != len(features):
            raise ValueError("weights must match features")
        self.weights = [float(w) for w in weights]
        if loss == "energy_direct":
            self.loss_fn = _energy_direct_loss
        else:
            self.loss_fn = SamplesLoss(loss=loss, p=p, backend="tensorized")
        self.min_samples = int(min_samples)

    def bind_dataset(self, dataset) -> None:
        for feature in self.features:
            if hasattr(feature, "bind_dataset"):
                feature.bind_dataset(dataset)

    def forward(self, batch_1, states_1, batch_2, states_2) -> torch.Tensor:
        return self.forward_many([batch_1, batch_2], [states_1, states_2])

    def forward_many(
        self,
        batches: list[TreeBatch],
        states_list: list[torch.Tensor],
        pair_indices: Optional[list[tuple[int, int]]] = None,
    ) -> torch.Tensor:
        if len(batches) != len(states_list):
            raise ValueError("batches and states_list must have the same length")
        if len(batches) < 2:
            return states_list[0].new_zeros(()) if states_list else torch.zeros(())
        if pair_indices is None:
            pair_indices = list(itertools.combinations(range(len(batches)), 2))

        # Determine the time window common to all records in the batch and
        # propagate it to features that support per-batch clipping (e.g.
        # ``PathFeature``). Path features will then use bins inside this
        # window only, so cross-record comparisons are well-defined under
        # both grafted and non-grafted setups (records with shorter trees
        # would otherwise drop every anchor whose bin set extends past their
        # leaf time).
        t_min, t_max = None, None
        for batch in batches:
            valid = batch.mask & batch.leaf_mask
            if not valid.any():
                continue
            cur_max = batch.node_times[valid].max().item()
            t_max = cur_max if t_max is None else min(t_max, cur_max)
            roots = batch.mask & (batch.parent_idx < 0)
            if roots.any():
                cur_min = batch.node_times[roots].max().item()
                t_min = cur_min if t_min is None else max(t_min, cur_min)
        for feature in self.features:
            if hasattr(feature, "set_active_window"):
                feature.set_active_window(t_min=t_min, t_max=t_max)

        total = states_list[0].new_zeros(())
        # Expose per-feature losses for diagnostic logging in the trainer.
        self.last_feature_losses: dict[str, float] = {}
        for feature, weight in zip(self.features, self.weights):
            if weight == 0.0:
                continue
            if isinstance(feature, AnchoredFeature):
                feature_loss = self._anchored_loss(feature, batches, states_list)
            else:
                feature_loss = self._pairwise_loss(
                    feature, batches, states_list, pair_indices,
                )
            key = getattr(feature, "name", type(feature).__name__)
            self.last_feature_losses[key] = float(feature_loss.detach().item())
            total = total + weight * feature_loss
        return total

    def _anchored_loss(
        self,
        feature: "AnchoredFeature",
        batches: list[TreeBatch],
        states_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Loss for an anchored feature: ``loss(query, target)`` per record per
        stratum, summed across strata."""
        feature_loss = states_list[0].new_zeros(())
        for batch, states in zip(batches, states_list):
            laws = feature.collect(batch, states)
            for key, value in laws.items():
                query, target = value  # AnchoredFeature contract
                if (
                    query.shape[0] < self.min_samples
                    or target.shape[0] < self.min_samples
                ):
                    continue
                if query.shape[-1] != target.shape[-1]:
                    raise ValueError(
                        f"anchored stratum {key!r} has incompatible dims "
                        f"{query.shape[-1]} and {target.shape[-1]}"
                    )
                feature_loss = feature_loss + self.loss_fn(query, target)
        return feature_loss

    def _pairwise_loss(
        self,
        feature: StratifiedFeature,
        batches: list[TreeBatch],
        states_list: list[torch.Tensor],
        pair_indices: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Loss for a pairwise feature: ``loss(record_i, record_j)`` per pair
        per stratum, averaged within stratum and then summed across strata."""
        laws_by_record = [
            feature.collect(batch, states)
            for batch, states in zip(batches, states_list)
        ]
        keys = set().union(*(laws.keys() for laws in laws_by_record))
        feature_loss = states_list[0].new_zeros(())
        # A feature may provide its own ground discrepancy via a
        # ``pairwise_loss`` attribute; otherwise use the aligner's.
        loss_fn = getattr(feature, "pairwise_loss", None) or self.loss_fn

        for key in keys:
            stratum_loss = states_list[0].new_zeros(())
            n_pairs = 0
            for i, j in pair_indices:
                x = laws_by_record[i].get(key)
                y = laws_by_record[j].get(key)
                if x is None or y is None:
                    continue
                if x.shape[0] < self.min_samples or y.shape[0] < self.min_samples:
                    continue
                if x.shape[-1] != y.shape[-1]:
                    raise ValueError(
                        f"feature stratum {key!r} has incompatible dimensions "
                        f"{x.shape[-1]} and {y.shape[-1]}"
                    )
                stratum_loss = stratum_loss + loss_fn(x, y)
                n_pairs += 1
            if n_pairs == 0:
                continue
            feature_loss = feature_loss + stratum_loss / n_pairs

        return feature_loss
