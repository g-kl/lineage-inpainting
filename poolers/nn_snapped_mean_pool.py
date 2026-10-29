from __future__ import annotations

import torch
import torch.nn as nn

from datasets.tree_record import TreeBatch
from pairings.snap import ReferenceBank
from utils.reference import reference_cells


class NNSnappedMeanPool(nn.Module):
    """Parameter-free pooler: mean-pool bottom-up, then snap every internal
    node embedding (and leaves if ``snap_leaves``) to its nearest reference
    cell at the closest observation time.

    Snapping is delegated to a :class:`~pairings.snap.ReferenceBank`, which
    holds one faiss index per observation time and routes each node to the
    bucket nearest its own ``node_time``.

    Usage::

        pooler = NNSnappedMeanPool(dim=64)
        pooler.bind_dataset(dataset)
        states = pooler.pool_batch(batch)

    Args:
        dim:          Embedding dimensionality.
        max_children: Unused; kept to match other poolers.
        snap_leaves:  If ``True``, leaf nodes are also snapped. Default
                      ``False`` -- only pooled internal nodes are snapped.
        gpu_id:       CUDA device ordinal for the reference bank / faiss index.
    """

    def __init__(
        self,
        dim: int,
        max_children: int | None = None,
        snap_leaves: bool = False,
        gpu_id: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.max_children = max_children
        self.snap_leaves = snap_leaves
        self.gpu_id = gpu_id
        self._device_param = nn.Parameter(torch.empty(0), requires_grad=False)
        self.bank: ReferenceBank | None = None

    # ------------------------------------------------------------------ #
    #  Reference cell bank
    # ------------------------------------------------------------------ #

    def bind_dataset(self, dataset) -> None:
        """Pull the real-cell bank off ``dataset`` and bind it (see
        :meth:`bind_cells`)."""
        self.bind_cells(*reference_cells(dataset))

    def bind_cells(self, embeddings: torch.Tensor, times: torch.Tensor) -> None:
        """Build the time-bucketed reference bank used for NN snapping."""
        self.bank = ReferenceBank(
            embeddings, times, device=torch.device(f"cuda:{self.gpu_id}"),
        )

    # ------------------------------------------------------------------ #
    #  Pooler interface
    # ------------------------------------------------------------------ #

    def forward(self, children_states: list[torch.Tensor]) -> torch.Tensor:
        """Mean-pool a list of children embeddings (no snapping here)."""
        return torch.stack(children_states).mean(dim=0)

    def pool_batch(self, batch: TreeBatch) -> torch.Tensor:
        """Pool a TreeBatch bottom-up via mean, then snap to nearest reference.

        Returns (B, N, dim) embeddings with every active node snapped to the
        nearest reference cell at the closest observation time.
        """
        # 1. Bottom-up mean pooling
        states = batch.leaf_states.detach().clone()
        adj = batch.adj
        n_children = adj.sum(dim=-1, keepdim=True).clamp(min=1)

        for d in range(batch.max_depth, 0, -1):
            targets = (batch.depth == d - 1) & ~batch.leaf_mask & batch.mask
            if not targets.any():
                continue
            children_mean = torch.bmm(adj, states) / n_children
            states = torch.where(targets.unsqueeze(-1), children_mean, states)

        # 2. Snap active nodes to their nearest reference cell.
        if self.bank is not None:
            snap_mask = batch.mask if self.snap_leaves else (batch.mask & ~batch.leaf_mask)
            states = self._snap(states, snap_mask, batch.node_times)

        return states

    def _snap(
        self, states: torch.Tensor, mask: torch.Tensor, node_times: torch.Tensor,
    ) -> torch.Tensor:
        flat = states.reshape(-1, self.dim)
        active = mask.reshape(-1)
        if not active.any():
            return states
        snapped, _ = self.bank.snap_nn(
            flat[active], node_times.reshape(-1)[active],
        )
        result = flat.clone()
        result[active] = snapped
        return result.reshape(states.shape)
