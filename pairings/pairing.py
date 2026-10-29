"""Pair synthetic stage-one states to real marginal cells before splitting."""

from __future__ import annotations

import torch
import torch.nn as nn

from datasets.tree_record import TreeBatch
from pairings.snap import ReferenceBank
from utils.reference import reference_cells


class IdentityPairing(nn.Module):
    """No-op pairing; the splitter sees the pooled states unchanged.

    Set ``stop_grad=True`` to detach the states the splitter sees, cutting the
    splitter-loss gradient from reaching the pooler entirely.
    """

    def __init__(self, stop_grad: bool = False):
        super().__init__()
        self.stop_grad = stop_grad
        self.last_n_paired = 0

    def forward(
        self,
        batch: TreeBatch,
        states: torch.Tensor,
        stage2_anchor_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        self.last_n_paired = 0
        return states.detach() if self.stop_grad else states


class NearestNeighborPairing(nn.Module):
    """Hard-snap eligible synthetic nodes to real cells at nearby times.

    Snapping is delegated to a :class:`~pairings.snap.ReferenceBank`: each
    eligible node (active, unobserved, not a stage-two anchor) is matched by
    faiss nearest neighbour against the reference bucket nearest its own time
    plus ``snap_window`` buckets on each side. With ``time_tolerance`` set,
    nodes whose nearest bucket is farther than the tolerance are left unpaired;
    with ``min_snap_time`` set, nodes earlier than the cutoff are left unpaired.
    These mirror the analysis coupling (own bin + adjacent bins, with an early
    cutoff). ``clonal_reference`` restricts the bank to clonally-tracked cells,
    matching the marginal aligner's lineage-only reference.
    """

    def __init__(
        self,
        time_tolerance: float | None = None,
        min_snap_time: float | None = None,
        snap_window: int = 0,
        clonal_reference: bool = False,
        stop_grad: bool = False,
    ):
        super().__init__()
        self.time_tolerance = time_tolerance
        self.min_snap_time = min_snap_time
        self.snap_window = snap_window
        self.clonal_reference = clonal_reference
        # When True, detach the pool-side states before splicing in the
        # snapped real cells -- cuts every gradient path from the splitter
        # loss back to the pooler, so the pooler is trained only by the
        # marginal/path/CA aligner (which already runs on the pool head).
        self.stop_grad = stop_grad
        self.register_buffer("_device", torch.empty(0), persistent=False)
        self.bank: ReferenceBank | None = None
        self.last_n_paired = 0

    def bind_dataset(self, dataset) -> None:
        """Pull the real-cell bank off ``dataset`` and bind it (see
        :meth:`bind_cells`). Restricted to clonal cells when
        ``clonal_reference`` -- the same lineage-only bank the marginal
        aligner matches against."""
        self.bind_cells(*reference_cells(dataset, clonal_only=self.clonal_reference))

    def bind_cells(self, embeddings: torch.Tensor, times: torch.Tensor) -> None:
        """Build the time-bucketed reference bank used for NN pairing."""
        if embeddings.ndim != 2:
            raise ValueError("reference embeddings must have shape (n_cells, dim)")
        if times.ndim != 1 or times.shape[0] != embeddings.shape[0]:
            raise ValueError("reference times must have shape (n_cells,)")
        self.bank = ReferenceBank(embeddings, times, device=self._device.device)

    def forward(
        self,
        batch: TreeBatch,
        states: torch.Tensor,
        stage2_anchor_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        self.last_n_paired = 0
        if self.bank is None:
            raise RuntimeError(
                "NearestNeighborPairing has no reference cells. "
                "Call bind_dataset before training."
            )
        if states.shape[-1] != self.bank.dim:
            raise ValueError(f"expected state dim {self.bank.dim}, got {states.shape[-1]}")

        pair_mask = batch.mask & ~batch.observed_mask
        if stage2_anchor_mask is not None:
            pair_mask = pair_mask & ~stage2_anchor_mask
        if self.min_snap_time is not None:
            pair_mask = pair_mask & (batch.node_times >= self.min_snap_time)

        # If stop_grad, detach the incoming pool states once -- both the
        # passthrough (unpaired) and the snapped (paired) positions then carry
        # no gradient.
        if self.stop_grad:
            states = states.detach()

        flat = states.reshape(-1, states.shape[-1])
        active = pair_mask.reshape(-1)
        if not active.any():
            return states

        active_idx = torch.where(active)[0]
        queries = flat.index_select(0, active_idx)
        active_times = batch.node_times.reshape(-1).index_select(0, active_idx)

        snapped, keep = self.bank.snap_nn(
            queries, active_times,
            time_tolerance=self.time_tolerance,
            window=self.snap_window,
        )
        if not keep.any():
            return states

        result = flat.clone()
        result[active_idx[keep]] = snapped[keep].detach()
        self.last_n_paired = int(keep.sum().item())
        return result.reshape_as(states)


class OTPairing(nn.Module):
    """Snap eligible synthetic nodes to real cells via per-bucket Sinkhorn OT.

    Mirror of :class:`NearestNeighborPairing` but the snap call hits
    :meth:`pairings.snap.ReferenceBank.snap_ot` instead of ``snap_nn``: for
    each time bucket, solve an entropic OT problem between queries and a
    random ``ref_batch_size`` subsample of references, then resample each
    query from its row of the plan. Unlike NN snap, every
    query in an active bucket is snapped (no ``time_tolerance`` carve-out).
    """

    def __init__(
        self,
        min_snap_time: float | None = None,
        ref_batch_size: int = 2048,
        reg: float = 0.05,
        max_iter: int = 100,
        clonal_reference: bool = False,
        stop_grad: bool = False,
        min_snap_warn: int = 1000,
        warn_small_batch: bool = False,
    ):
        super().__init__()
        self.min_snap_time = min_snap_time
        self.ref_batch_size = ref_batch_size
        self.reg = reg
        self.max_iter = max_iter
        self.clonal_reference = clonal_reference
        self.stop_grad = stop_grad
        self.min_snap_warn = min_snap_warn
        self.warn_small_batch = warn_small_batch
        self.register_buffer("_device", torch.empty(0), persistent=False)
        self.bank: ReferenceBank | None = None
        self.last_n_paired = 0

    def bind_dataset(self, dataset) -> None:
        self.bind_cells(*reference_cells(dataset, clonal_only=self.clonal_reference))

    def bind_cells(self, embeddings: torch.Tensor, times: torch.Tensor) -> None:
        if embeddings.ndim != 2:
            raise ValueError("reference embeddings must have shape (n_cells, dim)")
        if times.ndim != 1 or times.shape[0] != embeddings.shape[0]:
            raise ValueError("reference times must have shape (n_cells,)")
        self.bank = ReferenceBank(embeddings, times, device=self._device.device)

    def forward(
        self,
        batch: TreeBatch,
        states: torch.Tensor,
        stage2_anchor_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        self.last_n_paired = 0
        if self.bank is None:
            raise RuntimeError(
                "OTPairing has no reference cells. Call bind_dataset before training."
            )
        if states.shape[-1] != self.bank.dim:
            raise ValueError(f"expected state dim {self.bank.dim}, got {states.shape[-1]}")

        pair_mask = batch.mask & ~batch.observed_mask
        if stage2_anchor_mask is not None:
            pair_mask = pair_mask & ~stage2_anchor_mask
        if self.min_snap_time is not None:
            pair_mask = pair_mask & (batch.node_times >= self.min_snap_time)

        if self.stop_grad:
            states = states.detach()

        flat = states.reshape(-1, states.shape[-1])
        active = pair_mask.reshape(-1)
        if not active.any():
            return states

        active_idx = torch.where(active)[0]
        queries = flat.index_select(0, active_idx)
        active_times = batch.node_times.reshape(-1).index_select(0, active_idx)

        snapped = self.bank.snap_ot(
            queries, active_times,
            reg=self.reg,
            max_iter=self.max_iter,
            ref_batch_size=self.ref_batch_size,
            min_snap_warn=self.min_snap_warn,
            warn_small_batch=self.warn_small_batch,
        )

        result = flat.clone()
        result[active_idx] = snapped.detach()
        self.last_n_paired = int(active.sum().item())
        return result.reshape_as(states)
