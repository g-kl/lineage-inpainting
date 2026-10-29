from __future__ import annotations

import torch
import torch.nn as nn

from datasets.tree_record import TreeBatch

class MeanPool(nn.Module):
    """Baseline: plain mean of children states, no learnable parameters."""

    def __init__(self, dim: int, max_children: int | None = None):
        super().__init__()
        self.dim = dim
        self.max_children = max_children
        self._device_param = nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, children_states: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(children_states).mean(dim=0)

    def pool_batch(self, batch: TreeBatch) -> torch.Tensor:
        """Pool a TreeBatch bottom-up via unweighted mean.

        Args:
            batch: TreeBatch with leaf_states filled in (B, N, dim).

        Returns:
            (B, N, dim) embeddings for every node.
        """
        states = batch.leaf_states.detach().clone()
        adj = batch.adj
        n_children = adj.sum(dim=-1, keepdim=True).clamp(min=1)

        for d in range(batch.max_depth, 0, -1):
            targets = (batch.depth == d - 1) & ~batch.leaf_mask & batch.mask
            if not targets.any():
                continue
            children_mean = torch.bmm(adj, states) / n_children
            states = torch.where(targets.unsqueeze(-1), children_mean, states)

        return states