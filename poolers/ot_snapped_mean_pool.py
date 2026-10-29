from __future__ import annotations

import torch
import torch.nn as nn

from datasets.tree_record import TreeBatch
from pairings.snap import ReferenceBank
from utils.reference import reference_cells


class OTSnappedMeanPool(nn.Module):
    """Parameter-free pooler: mean-pool bottom-up, then replace every internal
    node embedding (and leaves if ``snap_leaves``) with a *sample from an
    optimal-transport plan* between the batch and a random minibatch of
    reference cells **at the closest observation time**.

    Snapping is delegated to :meth:`pairings.snap.ReferenceBank.snap_ot`:
    nodes are routed to the bucket whose reference time is closest to their
    own ``node_time``, an entropic OT problem is solved per bucket against a
    random ``ref_batch_size`` subsample, and each node is reassigned by
    sampling from its row of the plan. Nothing is back-propagated through;
    the pooler is entirely parameter-free.

    Usage::

        pooler = OTSnappedMeanPool(dim=64, ref_batch_size=2048)
        pooler.bind_dataset(dataset)
        states = pooler.pool_batch(batch)              # stochastic!

    Args:
        dim:              Embedding dimensionality.
        max_children:     Unused; kept for API parity with other poolers.
        snap_leaves:      If ``True``, leaf nodes are also snapped. Default
                          ``False`` -- only pooled internal nodes are snapped.
        ref_batch_size:   Reference cells sampled for OT each forward pass,
                          per time bucket.
        reg:              Entropic regularisation strength for Sinkhorn.
        max_iter:         Maximum Sinkhorn iterations.
        min_snap_warn:    Warn when a per-time Sinkhorn call sees fewer than
                          this many active query nodes.
        warn_small_batch: If ``False``, suppresses the small-batch warning.
    """

    def __init__(
        self,
        dim: int,
        max_children: int | None = None,
        snap_leaves: bool = False,
        ref_batch_size: int = 2048,
        reg: float = 0.05,
        max_iter: int = 100,
        min_snap_warn: int = 1000,
        warn_small_batch: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_children = max_children
        self.snap_leaves = snap_leaves
        self.ref_batch_size = ref_batch_size
        self.reg = reg
        self.max_iter = max_iter
        self.min_snap_warn = min_snap_warn
        self.warn_small_batch = warn_small_batch
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
        """Build the time-bucketed reference bank used for OT snapping."""
        self.bank = ReferenceBank(
            embeddings, times, device=self._device_param.device,
        )

    # ------------------------------------------------------------------ #
    #  Pooler interface
    # ------------------------------------------------------------------ #

    def forward(self, children_states: list[torch.Tensor]) -> torch.Tensor:
        """Mean-pool a list of children embeddings (no OT here)."""
        return torch.stack(children_states).mean(dim=0)

    def pool_batch(self, batch: TreeBatch) -> torch.Tensor:
        """Pool a TreeBatch bottom-up via mean, then snap by sampling an OT plan.

        Returns (B, N, dim) embeddings with every active node stochastically
        mapped to a reference cell via the per-time Sinkhorn transport plan.
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

        # 2. OT-based snapping (per-time)
        if self.bank is None:
            print("warning: no reference bank bound (bind_dataset / bind_cells); returning unsnapped states")
            return states

        snap_mask = batch.mask if self.snap_leaves else (batch.mask & ~batch.leaf_mask)
        return self._snap(states, snap_mask, batch.node_times)

    def _snap(
        self, states: torch.Tensor, mask: torch.Tensor, node_times: torch.Tensor,
    ) -> torch.Tensor:
        flat = states.reshape(-1, self.dim)
        active = mask.reshape(-1)
        if not active.any():
            return states
        snapped = self.bank.snap_ot(
            flat[active], node_times.reshape(-1)[active],
            reg=self.reg,
            max_iter=self.max_iter,
            ref_batch_size=self.ref_batch_size,
            min_snap_warn=self.min_snap_warn,
            warn_small_batch=self.warn_small_batch,
        )
        result = flat.clone()
        result[active] = snapped.to(states.dtype)
        return result.reshape(states.shape)
