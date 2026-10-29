"""Slice-based stratified features built on a single shared primitive.

This module implements the recursive-slice construction (see Methods) with
one sampler and one feature class on a uniform, fully-vectorized core.

The primitive
-------------
``SliceSampler.sample(start_nodes, t, q)`` draws ``q`` cells from the **active
cross-section** of the tree at absolute time ``t``, restricted to the subtree
of each ``start`` node. "Cells alive at ``t``" is taken node-uniformly over the
cross-section (a node ``n`` is on it iff ``time(n) <= t`` and ``n`` is a leaf or
has a child after ``t``), i.e. ``S_t(\\mathcal T)`` in the Methods. Draws are
**without replacement** whenever the slice holds at least ``q`` distinct cells,
falling back to **with replacement** only when it does not. This fallback
yields the degenerate ``(X_m, X_m, X_m)`` when a lineage has not yet divided,
so exchangeability needs no separate symmetrization step.

The feature
-----------
``SliceFeature`` is a composition of slice samplers and subtree restrictions
expressed over **absolute grid times**: a list of ``levels = [(time, n_draws),
...]``. Level 0 samples the ancestor from the whole tree at its time; each later
level samples ``n_draws`` cells at its time inside the subtree of the previous
draw. The returned vector concatenates every sampled node's state. Subclasses
turn the batch's active grid (``time_bins`` intersected with the common observed
window) into the compositions to compare:

- **common-ancestor** (:class:`CommonAncestorFeature`): every ordered pair of
  grid times ``(t_i, t_j)``, ``t_i < t_j`` -> ancestor at ``t_i``, two
  descendants at ``t_j``. The leaf layer is only ever an end, never a start, so
  the ancestor is never a leaf. For a 8.5/9.0/9.5 window this is the three pairs
  ``(8.5,9.0), (8.5,9.5), (9.0,9.5)``.
- **path** (:class:`PathFeature`): one lineage from the *earliest* grid time
  through every later grid time (root-to-leaf), a single stratum -- a path never
  starts at an interior slice. For the same window: ``(X@8.5, X@9.0, X@9.5)``.

Strata are keyed by ``(name, *composition, representation)`` so only like-for-
like draws are compared, and ``StratifiedFeatureAligner`` compares them
pairwise across records.

The slice *marginal* remains the reference-bank ``TimeMarginalFeature``
from :mod:`aligners.stratified` (it matches predicted cells against a real-cell
bank, not against another record).
"""
from __future__ import annotations

from typing import Optional

import torch

from datasets.tree_record import TreeBatch, cross_section, max_child_time
from aligners.stratified import (
    FeatureLaws,
    StratifiedFeature,
    _ancestor_table,
    _default_tolerance,
)


__all__ = [
    "SliceSampler",
    "SliceFeature",
    "CommonAncestorFeature",
    "PathFeature",
]

_NEG_INF = float("-inf")


def _descendant_matrix(batch: TreeBatch) -> torch.BoolTensor:
    """``is_desc[b, a, n]``: node ``a`` is an ancestor-or-self of node ``n``.

    Built from the root-to-node ancestor table: ``a`` is an ancestor-or-self of
    ``n`` iff ``depth(a) <= depth(n)`` and the ancestor of ``n`` at ``depth(a)``
    is ``a``.
    """
    B, N = batch.mask.shape
    path_idx, _ = _ancestor_table(batch)          # (B, N, D)
    D = path_idx.shape[2]
    depth = batch.depth                           # (B, N)

    # ancestor of every n at the depth of every a -> (B, N_a, N_n)
    da = depth.clamp(max=D - 1).view(B, N, 1, 1).expand(B, N, N, 1)
    pi = path_idx.view(B, 1, N, D).expand(B, N, N, D)
    anc = pi.gather(3, da).squeeze(-1)            # (B, N_a, N_n)

    a_ids = torch.arange(N, device=batch.mask.device).view(1, N, 1)
    return (
        (anc == a_ids)
        & (depth.view(B, N, 1) <= depth.view(B, 1, N))
        & batch.mask.view(B, N, 1)
        & batch.mask.view(B, 1, N)
    )


class SliceSampler:
    """Draw cells from the active cross-section at a time, within a subtree.

    Construct once per ``(batch, states)`` collect; reuse across all slice
    times and levels. Cost is one ``(B, N, N)`` descendant matrix plus per-call
    ``(B, M, N)`` work.
    """

    def __init__(self, batch: TreeBatch, direction: str = "up"):
        if direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")
        self.batch = batch
        self.direction = direction
        self.B, self.N = batch.mask.shape
        self.device = batch.mask.device
        self.is_desc = _descendant_matrix(batch)                 # (B, N, N)
        # Per-node times the slice needs, precomputed once and reused across
        # every slice time in a collect.
        self._max_child = max_child_time(batch) if direction == "up" else None
        self._parent_time = (
            batch.node_times.gather(1, batch.parent_idx.clamp(min=0))
            if direction == "down" else None
        )

    def cross_section(self, t: float) -> torch.BoolTensor:
        """Nodes representing the branches the slice at ``t`` cuts (see
        :func:`datasets.tree_record.cross_section`): ``direction`` picks the
        node above (``'up'``/floor) or below (``'down'``/ceil) each cut branch."""
        return cross_section(
            self.batch, t, self.direction,
            max_child=self._max_child, parent_time=self._parent_time,
        )

    def sample(
        self, start: torch.Tensor, t: float, q: int,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Sample ``q`` cells at slice ``t`` inside each ``start`` node's subtree.

        ``start`` is ``(B, M)`` node indices. Returns ``(idx, valid)`` with
        ``idx`` of shape ``(B, M, q)`` and ``valid`` ``(B, M)`` false where the
        restricted slice was empty.

        The slice is node-uniform, so a without-replacement draw is just the
        top-``q`` of a uniform random key over the eligible cells (no Gumbel --
        that is only for *weighted* without-replacement sampling). With
        replacement is used only where the slice holds fewer than ``q`` distinct
        cells, which is also what yields the degenerate all-same-cell draw.
        """
        B, N, M = self.B, self.N, start.shape[1]
        cs = self.cross_section(t)                               # (B, N)
        desc = self.is_desc.gather(1, start.unsqueeze(-1).expand(B, M, N))
        eligible = desc & cs.unsqueeze(1)                        # (B, M, N)
        count = eligible.sum(-1)                                 # (B, M)

        # without replacement: top-q of a uniform random key over eligible cells
        key = torch.rand(B, M, N, device=self.device).masked_fill(~eligible, _NEG_INF)
        wo_idx = key.topk(q, dim=-1).indices                    # (B, M, q)

        # with replacement: fills rows whose slice has < q distinct cells
        # (empty rows fall back to a uniform draw and are marked invalid).
        weights = torch.where(
            (count == 0).unsqueeze(-1),
            torch.ones(B, M, N, device=self.device),
            eligible.float(),
        )
        wr_idx = torch.multinomial(
            weights.view(B * M, N), q, replacement=True,
        ).view(B, M, q)

        idx = torch.where((count >= q).unsqueeze(-1), wo_idx, wr_idx)
        return idx, count >= 1


class SliceFeature(StratifiedFeature):
    """Recursive composition of slice samplers over the time grid.

    A composition is a list of ``(absolute_time, n_draws)`` levels: level 0
    samples the ancestor from the whole tree at its time; each later level
    samples ``n_draws`` cells at its time inside the previous draw's subtree.
    The feature vector concatenates every sampled node's state. Subclasses turn
    the batch's active grid (the configured ``time_bins`` intersected with the
    common observed window ``[t_min, t_max]``) into the set of compositions to
    compare, via :meth:`_build_compositions`.

    n_samples : draws per tree per stratum.
    representation : ``"states"`` (raw) or ``"relative"`` (subtract the level-0
        ancestor, making the feature invariant to the ancestor's absolute state).
    """

    def __init__(
        self,
        *,
        time_bins: list[float],
        name: str = "slice",
        n_samples: int = 16,
        representation: str = "states",
        direction: str = "up",
        time_tolerance: Optional[float] = None,
    ):
        super().__init__()
        if representation not in {"states", "relative"}:
            raise ValueError("representation must be 'states' or 'relative'")
        if direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")
        self.name = name
        self.time_bins = [float(t) for t in sorted(time_bins)]
        self.n_samples = int(n_samples)
        self.representation = representation
        self.direction = direction
        # Small shift applied to every slice query so grid-aligned nodes land
        # inside the slice rather than on the knife-edge of the comparison
        # (float jitter on a 9.0 node otherwise flips which layer is grabbed).
        # 'up' takes the floor, so shift forward; 'down' takes the ceil, so
        # shift backward -- each toward the grid-aligned layer's own side.
        diffs = [b - a for a, b in zip(self.time_bins, self.time_bins[1:])]
        eps = 0.1 * (min(diffs) if diffs else 1.0)
        self.slice_eps = eps if direction == "up" else -eps
        self._t_min: Optional[float] = None
        self._t_max: Optional[float] = None

    def set_active_window(
        self, t_min: Optional[float] = None, t_max: Optional[float] = None,
    ) -> None:
        """Record the batch's common observed time window for ``_active_grid``."""
        self._t_min = None if t_min is None else float(t_min)
        self._t_max = None if t_max is None else float(t_max)

    def _active_grid(self) -> list[float]:
        """Configured grid times inside the batch's common window ``[t_min,
        t_max]``. The leaf layer (``t_max``) is included -- it is a valid slice
        *end* (where descendants land on the observed cells), and subclasses
        keep it from being used as an ancestor *start*."""
        tol = _default_tolerance(self.time_bins)
        lo = float("-inf") if self._t_min is None else self._t_min - tol
        hi = float("inf") if self._t_max is None else self._t_max + tol
        return [t for t in self.time_bins if lo <= t <= hi]

    def _build_compositions(
        self, grid: list[float]
    ) -> list[tuple[tuple, list[tuple[float, int]]]]:
        """Return ``[(key_extra, [(abs_time, n_draws), ...]), ...]`` for ``grid``.

        ``key_extra`` is appended to the stratum key so distinct compositions of
        the same feature stay separate. Only the final level may draw more than
        one cell (intermediate levels are a single ancestor chain)."""
        raise NotImplementedError

    def collect(self, batch: TreeBatch, states: torch.Tensor) -> FeatureLaws:
        comps = self._build_compositions(self._active_grid())
        if not comps:
            return {}
        B, _ = batch.mask.shape
        M = self.n_samples
        dim = states.shape[-1]
        sampler = SliceSampler(batch, direction=self.direction)
        b_idx = torch.arange(B, device=states.device).view(B, 1, 1)
        root = batch.root_idx.view(B, 1).expand(B, M)

        laws: FeatureLaws = {}
        for key_extra, levels in comps:
            cur = root
            collected, valid = [], None
            for k, (t_abs, nd) in enumerate(levels):
                # Shift by slice_eps so grid-aligned nodes fall inside the slice.
                idx, vld = sampler.sample(cur, t_abs + self.slice_eps, nd)  # (B, M, nd)
                collected.append(idx)
                valid = vld if valid is None else (valid & vld)
                if k < len(levels) - 1:
                    cur = idx[..., 0]
            node_idx = torch.cat(collected, dim=-1)                 # (B, M, Q)
            Q = node_idx.shape[-1]
            vals = states[b_idx.expand(B, M, Q), node_idx]          # (B, M, Q, d)
            if self.representation == "relative":
                vals = vals - vals[:, :, :1, :]
            feat = vals.reshape(B * M, Q * dim)[valid.reshape(-1)]
            if feat.shape[0] == 0:
                continue
            laws[(self.name, *key_extra, self.representation)] = feat
        return laws


class CommonAncestorFeature(SliceFeature):
    """``(X_m, X_u, ...)`` over every ordered pair of grid times.

    For each pair of active grid times ``(t_i, t_j)`` with ``t_i < t_j``, the
    ancestor is sampled at ``t_i`` and ``n_descendants`` descendants (without
    replacement where possible) at ``t_j`` inside its subtree (one stratum per
    pair). The latest grid time (the leaf layer) only ever appears as an end,
    never a start, so the ancestor is never itself a leaf.
    """

    def __init__(
        self,
        *,
        time_bins: list[float],
        n_descendants: int = 2,
        name: str = "common_ancestor",
        n_samples: int = 16,
        representation: str = "states",
        direction: str = "up",
        time_tolerance: Optional[float] = None,
    ):
        super().__init__(
            time_bins=time_bins, name=name, n_samples=n_samples,
            representation=representation, direction=direction,
            time_tolerance=time_tolerance,
        )
        self.n_descendants = int(n_descendants)

    def _build_compositions(self, grid):
        return [
            ((t_i, t_j), [(t_i, 1), (t_j, self.n_descendants)])
            for i, t_i in enumerate(grid)
            for t_j in grid[i + 1:]
        ]


class PathFeature(SliceFeature):
    """``(X_m, X_{u_1}, ..., X_{u_k})``: one lineage from the earliest grid time
    through every later grid time (root-to-leaf), a single stratum. The start is
    always the earliest time -- a path never begins at an interior slice."""

    def _build_compositions(self, grid):
        if len(grid) < 2:
            return []
        return [((tuple(grid),), [(t, 1) for t in grid])]
