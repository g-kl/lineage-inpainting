"""Topology extenders.

Extend an early-leaf-time subtree forward to a later target time by grafting
donor topology pulled from later records' trees, sliced at the early leaf's
time. The donor contributes topology and times only -- grafted nodes are renamed
to fresh ids so they are not in ``leaf_states`` and collate marks them
unobserved. The original early leaves keep their states and become internal
observed anchors.

Donor candidates are built once at startup into a
:class:`~extenders.reference.SubtreeBank`, stratified by
``(start_leaf_time, end_leaf_time)``: for stratum ``(t0, t1)`` every ``t1``-record
tree is layer-sliced at ``t0``. The extender draws one donor per early leaf and
identifies its layer cell with the leaf, so every grafted edge is strictly
forward in time.

:class:`GraftExtender` is the base of the graft family: it owns the leaf loop,
bank wiring, and depth/size bookkeeping. Subclasses override only how a donor is
chosen per leaf (:meth:`GraftExtender._graft_leaf`) and, optionally, when
extension can run (:meth:`GraftExtender._prepare`). The plain ``GraftExtender``
draws a uniform-random donor (any leaf count); the conditional variants in
``extenders/conditional.py`` draw a count-conditioned donor.

Known bias
----------
The ``min_root_out_degree >= 2`` filter (which forbids 1->1 chains at the
graft junction) excludes layer cells that have exactly one across-cut child,
i.e. lineages that pass through the cut without branching. In the mouse
embryo data, ~18% of natural layer cells fall in this no-branch category and
the empirical 8.5 -> 9.0 cell-count expansion is only ~1.19x. By dropping
the no-branch cells from the candidate pool, the extender produces grafts
that all branch, which over-estimates the leaf count by roughly 30% and
adds ~0.5 to mean tree depth (mean branching factor still matches). This is
accepted in exchange for keeping the junction structurally meaningful (no
trivial 1-child link from the early leaf into the graft).

The ``min_post_cut_gap`` filter further drops candidates whose first
across-cut child sits within ``min_post_cut_gap`` of ``t_cut`` -- it forbids
unrealistically fast splits right under the early leaf (e.g. a junction
edge of length 0.01 when the extension window is 0.5).

These filters are applied at bank-build time (see ``SubtreeBank``).
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

from .reference import SubtreeBank, graft_onto_leaf


class NullExtender:
    """No-op extender. Trainer default; returns subtrees unchanged."""

    def __call__(
        self,
        subtrees: list[dict],
        target_leaf_time: float | None = None,
        rng=None,
        leaf_states: dict | None = None,
    ) -> list[dict]:
        return subtrees


class GraftExtender:
    """Graft a uniform-random real donor clade onto every original leaf.

    Base of the graft family. Override :meth:`_graft_leaf` to change how a donor
    is chosen per leaf and :meth:`_prepare` to change when extension may run; the
    leaf loop and bookkeeping below are shared.
    """

    needs_bank = True

    def __init__(
        self,
        *,
        min_root_out_degree: int = 2,
        min_post_cut_gap: float = 0.05,
        node_id_prefix: str = "ext",
    ):
        if min_root_out_degree < 2:
            raise ValueError(
                "min_root_out_degree must be >= 2 to forbid 1->1 chain grafts"
            )
        self.min_root_out_degree = int(min_root_out_degree)
        # Skip candidates whose first across-cut child is within this much of
        # the cut, so the graft junction edge has a non-trivial time gap.
        self.min_post_cut_gap = float(min_post_cut_gap)
        self.node_id_prefix = node_id_prefix
        self.bank: SubtreeBank | None = None

    def bind_dataset(self, dataset):
        """Build the donor bank from all record topologies (called once)."""
        if not self.needs_bank:
            return
        self.bank = SubtreeBank(
            dataset.records,
            min_root_out_degree=self.min_root_out_degree,
            min_post_cut_gap=self.min_post_cut_gap,
        )

    # ------------------------------------------------------------------ #
    #  Hooks
    # ------------------------------------------------------------------ #

    def _prepare(self, subtrees, target_leaf_time, leaf_states) -> bool:
        """Return True if extension can proceed (default: bank is built)."""
        return not self.needs_bank or self.bank is not None

    def _resolve_end(self, subtrees, target_leaf_time) -> float | None:
        """Absolute time the new leaves land at; None means 'do not extend'.

        Grafting needs a concrete target (the bank strata are keyed by it), so
        the base requires ``target_leaf_time``.
        """
        return None if target_leaf_time is None else float(target_leaf_time)

    def _graft_leaf(self, g, leaf, start, end, rng, leaf_states, name_gen):
        """Attach donor topology beneath ``leaf`` in ``g`` (mutates ``g``)."""
        cand = self.bank.sample(start, end, rng)
        if cand is not None:
            graft_onto_leaf(g, leaf, *cand, name_gen)

    # ------------------------------------------------------------------ #
    #  Driver (shared)
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        subtrees: list[dict],
        target_leaf_time: float | None = None,
        rng=None,
        leaf_states: dict | None = None,
    ) -> list[dict]:
        if not subtrees or not self._prepare(
            subtrees, target_leaf_time, leaf_states
        ):
            return subtrees
        end = self._resolve_end(subtrees, target_leaf_time)
        if end is None:
            return subtrees
        rng = rng or np.random.default_rng()
        # All early subtrees in a batch share the same leaf_time (their record's).
        start = float(subtrees[0]["leaf_time"])
        return [
            self._extend_one(sd, start, end, rng, leaf_states) for sd in subtrees
        ]

    def _extend_one(self, sd, start, end, rng, leaf_states):
        g = sd["graph"].copy()
        early_leaves = [n for n in g if g.out_degree(n) == 0]
        if not early_leaves:
            return sd
        name_gen = (f"{self.node_id_prefix}{i}" for i in itertools.count())
        for leaf in early_leaves:
            self._graft_leaf(g, leaf, start, end, rng, leaf_states, name_gen)
        return self._finalize(sd, g, end)

    @staticmethod
    def _finalize(sd, g, end):
        new_leaves = [n for n in g if g.out_degree(n) == 0]
        depths = {sd["root"]: 0}
        for n in nx.topological_sort(g):
            for c in g.successors(n):
                depths[c] = depths.get(n, 0) + 1
        return {
            **sd,
            "graph": g,
            "leaves": new_leaves,
            "leaf_time": float(end),
            "depth": max(depths[l] for l in new_leaves) - depths[sd["root"]],
            "size": len(new_leaves),
        }
