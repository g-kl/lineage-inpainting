"""
Alignment losses between two sets of pooled trees.

Each aligner receives (batch, states, batch, states) where:
    batch:  TreeBatch          -- padded tensor batch with precomputed mrca_times
    states: Tensor (B, N, d)   -- node embeddings produced by pooling

The figures' aligner is ``aligners.stratified.StratifiedFeatureAligner``; this
module keeps the ``Aligner`` base it shares and ``ObservedMarginalAligner``, the
pairwise marginal aligner of the synthetic context benchmark
(``experiment=synbench_fb_unshared_vpsde_idnn_leaf``).
"""
from __future__ import annotations

import itertools
from typing import Optional

import torch
import torch.nn as nn
from geomloss import SamplesLoss

from datasets.tree_record import TreeBatch


class Aligner(nn.Module):
    """Base class -- subclass and override forward."""

    def forward(
        self,
        batch_1: TreeBatch,
        states_1: torch.Tensor,
        batch_2: TreeBatch,
        states_2: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_many(
        self,
        batches: list[TreeBatch],
        states_list: list[torch.Tensor],
        pair_indices: Optional[list[tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """Apply a pairwise aligner over selected record pairs.

        Stratified aligners can override this to collect feature laws once per
        record and then compare all compatible strata.  The default sums
        forward() over the selected record pairs.
        """
        if len(batches) != len(states_list):
            raise ValueError("batches and states_list must have the same length")
        if len(batches) < 2:
            return states_list[0].new_zeros(()) if states_list else torch.zeros(())
        if pair_indices is None:
            pair_indices = list(itertools.combinations(range(len(batches)), 2))
        total = states_list[0].new_zeros(())
        for i, j in pair_indices:
            total = total + self.forward(
                batches[i], states_list[i], batches[j], states_list[j],
            )
        return total


# ------------------------------------------------------------------ #
#  ObservedMarginalAligner
# ------------------------------------------------------------------ #


class ObservedMarginalAligner(Aligner):
    """
    Align two tree sets at the supplied observation time of the earlier one.

    Given two sets of pooled trees with different leaf (observation) times,
    this aligner:
      1. Determines which set was observed earlier (smaller leaf time).
      2. Collects the supplied observed embeddings from the earlier set -- these are
         the directly observed states at that time point.
      3. Collects *internal-node* embeddings from the later set whose
         node time is close to the earlier leaf time -- these are the
         model's inferred ancestral states at that same time point.
      4. Computes a distributional divergence between the two point
         clouds, encouraging the inferred ancestors to match the
         observed cells.

    If both sets share the same observation time the aligner falls back to
    comparing their supplied observed embeddings directly.

    Args:
        loss:           divergence type passed to ``SamplesLoss``.
        p:              exponent passed to ``SamplesLoss``.
        time_tolerance: maximum absolute difference between a node's time
                        and the target time for the node to be included.
                        Defaults to 0.5.
    """

    def __init__(
        self,
        loss: str = "energy",
        p: int = 2,
        time_tolerance: float = 0.5,
    ):
        super().__init__()
        self.loss_fn = SamplesLoss(loss=loss, p=p, backend="tensorized")
        self.time_tolerance = time_tolerance

    # ---- helpers ---------------------------------------------------- #

    @staticmethod
    def _observed_time(pooled_trees: list[dict]) -> float:
        """Return the max supplied-observation time across the tree set."""
        obs_times: list[float] = []
        for pt in pooled_trees:
            graph = pt["graph"]
            for n in pt["observed_nodes"]:
                t = graph.nodes[n].get("time")
                if t is not None:
                    obs_times.append(t)
        if not obs_times:
            return float("nan")
        return max(obs_times)

    def _collect_observed(
        self, pooled_trees: list[dict]
    ) -> list[torch.Tensor]:
        """Collect supplied-observation embeddings from every tree in the set."""
        embeds: list[torch.Tensor] = []
        for pt in pooled_trees:
            states = pt["states"]
            for n in pt["observed_nodes"]:
                if n in states:
                    embeds.append(states[n])
        return embeds

    def _collect_nodes_at_time(
        self, pooled_trees: list[dict], target_time: float
    ) -> list[torch.Tensor]:
        """Collect embeddings of nodes whose time is within tolerance of *target_time*."""
        embeds: list[torch.Tensor] = []
        for pt in pooled_trees:
            graph, states = pt["graph"], pt["states"]
            for n in graph.nodes:
                if n not in states:
                    continue
                t = graph.nodes[n].get("time")
                if t is not None and abs(t - target_time) <= self.time_tolerance:
                    embeds.append(states[n])
        return embeds

    # ---- forward ---------------------------------------------------- #

    def forward(self, batch_1, states_1, batch_2, states_2):
        pooled_trees_1 = batch_1.to_pooled_tree_list(states_1)
        pooled_trees_2 = batch_2.to_pooled_tree_list(states_2)
        device = states_1.device

        lt1 = self._observed_time(pooled_trees_1)
        lt2 = self._observed_time(pooled_trees_2)

        # order so that 'early' is observed first, 'late' is observed later
        if lt1 <= lt2:
            early, late, target_time = pooled_trees_1, pooled_trees_2, lt1
        else:
            early, late, target_time = pooled_trees_2, pooled_trees_1, lt2

        # observed states: supplied real layer of the earlier tree set
        observed = self._collect_observed(early)

        if lt1 == lt2:
            # same observation time -- compare observed layers directly
            inferred = self._collect_observed(late)
        else:
            # inferred ancestral states: internal nodes of the later
            # tree set at the earlier observation time
            inferred = self._collect_nodes_at_time(late, target_time)

        if not observed or not inferred:
            return torch.tensor(0.0, device=device)

        return self.loss_fn(torch.stack(observed), torch.stack(inferred))
