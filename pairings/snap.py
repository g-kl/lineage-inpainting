"""Canonical snapping: match query embeddings to a reference cell bank.

Two layers:

* :func:`nearest_neighbor` -- the ground-truth faiss L2 primitive (GPU search
  with a validated CPU fallback). Same routine as ``_nearest_neighbor`` in
  ``analysis/lib/coupling/nearest_neighbor_coupling.py``.
* :class:`ReferenceBank` -- a time-bucketed reference bank for the training
  path (the snap poolers + :class:`~pairings.pairing.NearestNeighborPairing`).
  It buckets reference cells by observation time, routes each query to the
  bucket nearest its own time, and snaps either by nearest neighbour
  (:meth:`ReferenceBank.snap_nn`) or by sampling an optimal-transport plan
  (:meth:`ReferenceBank.snap_ot`).

faiss and POT are imported lazily inside the functions that need them so that
dependency-free consumers (e.g. ``IdentityPairing``) never pull them in.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Ground-truth NN primitive
# --------------------------------------------------------------------------- #

def nearest_neighbor(
    query: np.ndarray,
    targets: np.ndarray,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """For each row in query, return (index, L2 distance) of the nearest row in targets.

    Always validates the returned indices against ``len(targets)``; if the
    GPU path produced garbage (occasional faiss-GPU init issue) we re-run on
    CPU. Default is CPU because at the embedding widths we use here (50-dim
    scVI / ~1.7M cells) brute-force CPU is well under a second.
    """
    import faiss

    query = np.ascontiguousarray(query, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.float32)

    def _cpu_search() -> tuple[np.ndarray, np.ndarray]:
        idx_cpu = faiss.IndexFlatL2(targets.shape[1])
        idx_cpu.add(targets)
        d, i = idx_cpu.search(query, 1)
        return i.ravel(), d.ravel()

    if use_gpu and hasattr(faiss, "index_cpu_to_all_gpus"):
        try:
            index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(targets.shape[1]))
            index.add(targets)
            dists, idx = index.search(query, 1)
            idx = idx.ravel()
            dists = dists.ravel()
            if (idx < 0).any() or (idx >= len(targets)).any():
                logger.warning("faiss GPU returned out-of-range indices; falling back to CPU")
                return _cpu_search()
            return idx, dists
        except Exception as e:
            logger.warning("faiss GPU search failed (%s); falling back to CPU", e)
    return _cpu_search()


# --------------------------------------------------------------------------- #
#  Time-bucketed reference bank (training path)
# --------------------------------------------------------------------------- #

class ReferenceBank:
    """Reference cells bucketed by observation time, for per-query snapping.

    Built once from ``(embeddings, times)`` (typically via
    :func:`utils.reference.reference_cells` behind a ``bind_dataset`` hook).
    Each query is routed to the bucket whose unique reference time is closest
    to the query's own time, then snapped within that bucket.

    For NN snapping the bank builds one persistent GPU faiss index per bucket
    on ``device`` and searches it every step (requires ``faiss-gpu-cu12`` and a
    CUDA ``device``). OT snapping needs no index; it samples a fresh reference
    minibatch per call.

    Parameters
    ----------
    embeddings : Tensor (n_cells, dim)
    times      : Tensor (n_cells,)  per-cell observation time
    device     : where bucket tensors / faiss indices live (default: the
                 embeddings' device); must be CUDA for ``snap_nn``
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        times: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ):
        emb = embeddings.detach().float()
        self.device = torch.device(device) if device is not None else emb.device
        emb = emb.to(self.device)
        ref_times = times.detach().float().cpu()

        self.dim = int(emb.shape[1])
        unique_times = torch.unique(ref_times).sort().values
        self._time_keys: list[float] = [float(t) for t in unique_times.tolist()]
        self._unique_times = unique_times.to(self.device)
        # Per-bucket reference embeddings, on device (NN gathers / OT samples
        # from these directly).
        self._ref_by_time: dict[float, torch.Tensor] = {
            key: emb[(ref_times == key).to(self.device)].contiguous()
            for key in self._time_keys
        }

        # Lazily-built persistent faiss indices (NN only); None until first use.
        self._nn_indices: dict[float, "object"] | None = None
        self._gpu_res = None

    # -- routing ----------------------------------------------------------- #

    def _route(self, query_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (nearest_bin, |time gap to that bin|) for each query."""
        ut = self._unique_times.to(query_times.device)
        diffs = (query_times.unsqueeze(1) - ut.unsqueeze(0)).abs()
        nearest_bin = diffs.argmin(dim=1)
        nearest_diff = diffs.gather(1, nearest_bin.unsqueeze(1)).squeeze(1)
        return nearest_bin, nearest_diff

    # -- NN ---------------------------------------------------------------- #

    def _ensure_nn_indices(self) -> None:
        """Build one persistent GPU faiss index per time bucket (once)."""
        if self._nn_indices is not None:
            return
        import faiss
        import faiss.contrib.torch_utils  # noqa: F401  enables torch Tensor I/O

        self._gpu_res = faiss.StandardGpuResources()
        gpu_id = self.device.index or 0
        self._nn_indices = {}
        for key in self._time_keys:
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = gpu_id
            cfg.useFloat16 = False
            index = faiss.GpuIndexFlatL2(self._gpu_res, self.dim, cfg)
            index.add(self._ref_by_time[key].contiguous())
            self._nn_indices[key] = index

    @torch.no_grad()
    def snap_nn(
        self,
        queries: torch.Tensor,
        query_times: torch.Tensor,
        *,
        time_tolerance: float | None = None,
        window: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Snap each query to its nearest reference cell near its time bucket.

        Each query is routed to the bucket nearest its own time and snapped to
        the nearest reference cell within that bucket plus the ``window`` buckets
        on each side (clamped at the ends); ``window=0`` (default) is the
        single-nearest-bucket behaviour. This mirrors the analysis coupling,
        which snaps within a node's own time bin and the adjacent bins.

        Returns ``(snapped, keep)`` where ``snapped`` has the same shape/dtype
        as ``queries`` (rows not kept are left undefined) and ``keep`` is a bool
        mask. With ``time_tolerance`` set, queries whose nearest bucket is
        farther than the tolerance are dropped (``keep=False``); otherwise every
        query is snapped.
        """
        snapped = torch.empty_like(queries)
        keep = torch.zeros(queries.shape[0], dtype=torch.bool, device=queries.device)
        if queries.shape[0] == 0:
            return snapped, keep

        nearest_bin, nearest_diff = self._route(query_times)
        self._ensure_nn_indices()
        n_keys = len(self._time_keys)
        for bi in range(n_keys):
            sel = nearest_bin == bi
            if time_tolerance is not None:
                sel = sel & (nearest_diff <= float(time_tolerance))
            if not sel.any():
                continue
            q = queries[sel].detach().float().contiguous()
            best_dist: torch.Tensor | None = None
            best_emb: torch.Tensor | None = None
            for ci in range(max(0, bi - window), min(n_keys, bi + window + 1)):
                key = self._time_keys[ci]
                dist, idx = self._nn_indices[key].search(q, 1)
                emb = self._ref_by_time[key][idx[:, 0].long()]
                dist = dist[:, 0]
                if best_dist is None:
                    best_dist, best_emb = dist, emb
                else:
                    better = dist < best_dist
                    best_dist = torch.where(better, dist, best_dist)
                    best_emb = torch.where(better.unsqueeze(1), emb, best_emb)
            snapped[sel] = best_emb.to(snapped.dtype)
            keep[sel] = True
        return snapped, keep

    # -- OT ---------------------------------------------------------------- #

    @torch.no_grad()
    def snap_ot(
        self,
        queries: torch.Tensor,
        query_times: torch.Tensor,
        *,
        reg: float,
        max_iter: int,
        ref_batch_size: int,
        min_snap_warn: int = 1000,
        warn_small_batch: bool = True,
    ) -> torch.Tensor:
        """Snap each query by sampling a per-bucket Sinkhorn transport plan.

        For each time bucket, solve an entropic OT problem between the bucket's
        queries and a random ``ref_batch_size`` subsample of its reference
        cells, then reassign each query by sampling from its row of the plan.
        Returns a tensor shaped like ``queries`` (every query is snapped).
        """
        import ot as pot

        device = queries.device
        snapped = torch.empty_like(queries)
        if queries.shape[0] == 0:
            return snapped
        q_all = queries.detach().float()

        nearest_bin, _ = self._route(query_times)
        for bi, key in enumerate(self._time_keys):
            sel = nearest_bin == bi
            n_active = int(sel.sum())
            if n_active == 0:
                continue
            q = q_all[sel].contiguous()

            ref_bucket = self._ref_by_time[key].to(device)
            n_bucket = ref_bucket.shape[0]
            m = min(ref_batch_size, n_bucket)

            if warn_small_batch and n_active < min_snap_warn:
                warnings.warn(
                    f"ReferenceBank.snap_ot: only {n_active} active queries "
                    f"at ref_time={key} (threshold {min_snap_warn}). Sinkhorn "
                    f"quality may be degraded on tiny query sets. Pass "
                    f"warn_small_batch=False to silence.",
                    stacklevel=2,
                )

            ref_batch = ref_bucket[torch.randperm(n_bucket, device=device)[:m]]

            M = torch.cdist(q, ref_batch, p=2).pow(2)
            M_max = M.max()
            if M_max > 0:
                M = M / M_max
            a = torch.ones(n_active, dtype=torch.float32, device=device) / n_active
            b = torch.ones(m, dtype=torch.float32, device=device) / m

            T = pot.bregman.sinkhorn_log(
                a, b, M, reg=reg, numItermax=max_iter, stopThr=1e-9,
            )
            probs = T / T.sum(dim=1, keepdim=True).clamp(min=1e-12)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
            snapped[sel] = ref_batch[sampled].to(snapped.dtype)
        return snapped
