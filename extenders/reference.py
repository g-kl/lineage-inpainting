"""Shared donor-topology reference for the graft-style extenders.

Both :class:`~extenders.graft.GraftExtender` and
:class:`~extenders.conditional.ConditionalGraftExtender` extend an early subtree
forward to a later target time by grafting *real* donor topology pulled from the
later records' trees. The donors only depend on the pair of observation times
involved -- the early leaf time (where the graft junction sits) and the target
leaf time (where the new leaves land) -- so we build them once at startup into a
:class:`SubtreeBank` stratified by ``(start_leaf_time, end_leaf_time)`` instead
of re-slicing donors every training step.

A donor for stratum ``(t0, t1)`` is produced by layer-slicing a ``t1``-record
tree at ``t0`` (see :func:`layer_candidates`): for every edge ``(u, v)`` with
``u.time <= t0 < v.time`` the layer cell ``u`` is the latest-before-cut node on
that lineage, and the donor is ``u`` plus all of its across-cut descendants.
Identifying ``u`` with an early leaf makes every grafted edge strictly forward in
time, and since donors are real clades the extended trees stay roughly
exchangeable with natural late subtrees.

The same ``min_root_out_degree`` / ``min_post_cut_gap`` filters that
``GraftExtender`` applies (see that module's docstring for the known
leaf-count bias they introduce) are applied here at bank-build time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import networkx as nx
import numpy as np


def layer_candidates(graph, t_cut):
    """Layer-slice ``graph`` at ``t_cut``. Returns ``[(layer_cell, subgraph)]``.

    For every layer cell ``L`` (latest-before-cut node on a lineage that crosses
    ``t_cut``) the subgraph is ``L`` plus all descendants reachable through its
    across-cut children (transitively all ``> t_cut`` by tree monotonicity).
    """
    times = nx.get_node_attributes(graph, "time")
    across = {}   # layer_cell -> set of across-cut children
    for u, v in graph.edges():
        if times[u] <= t_cut < times[v]:
            across.setdefault(u, set()).add(v)
    out = []
    for L, children in across.items():
        keep = {L} | set(children)
        for c in children:
            keep |= nx.descendants(graph, c)
        out.append((L, graph.subgraph(keep).copy()))
    return out


def graft_onto_leaf(g, leaf, root, subgraph, name_gen):
    """Graft ``subgraph`` (rooted at ``root``) onto ``leaf`` of ``g`` in place.

    ``root`` is identified with ``leaf`` (so the junction edge is strictly
    forward in time); every other donor node is added under a fresh
    ``next(name_gen)`` id carrying its ``time`` attribute, so it is absent from
    ``leaf_states`` and collate marks it unobserved. ``leaf`` becomes an internal
    observed anchor.
    """
    rename = {root: leaf}
    for n in subgraph.nodes():
        if n != root:
            rename[n] = next(name_gen)
            g.add_node(rename[n], **subgraph.nodes[n])
    for u, v in subgraph.edges():
        g.add_edge(rename[u], rename[v])


class Donor(NamedTuple):
    """A real donor clade: the subtree of ``source`` on ``nodes`` rooted at ``root``.

    The subgraph copy is made lazily in :meth:`materialize` so the bank can hold
    many donors (across every time stratum) without copying every subgraph.
    """

    source: "nx.DiGraph"
    root: object
    nodes: frozenset
    size: int            # number of leaves

    def materialize(self):
        """Return ``(root, subgraph_copy)`` -- the copy happens only now."""
        return self.root, self.source.subgraph(self.nodes).copy()


class Stratum:
    """The donors for one ``(start, end)`` pair, drawable by leaf count.

    ``draw(rng)`` picks uniformly; ``draw(rng, k)`` picks a donor whose leaf
    count is closest to ``k`` -- exact bucket in O(1), nearest in O(log n).
    """

    def __init__(self, donors):
        self.donors = donors
        by_size = defaultdict(list)
        for d in donors:
            by_size[d.size].append(d)
        self._by_size = dict(by_size)
        self._sizes = np.array(sorted(self._by_size), dtype=np.int64)

    def draw(self, rng, k=None):
        bucket = self.donors if k is None else self._by_size[self._nearest(k)]
        return bucket[int(rng.integers(len(bucket)))]

    def _nearest(self, k):
        i = int(np.searchsorted(self._sizes, k))
        return min(
            (int(self._sizes[j]) for j in (i - 1, i) if 0 <= j < len(self._sizes)),
            key=lambda s: abs(s - k),
        )


class SubtreeBank:
    """Donor clades stratified by ``(start_leaf_time, end_leaf_time)``.

    Built once from a ``records`` mapping (``{name: TreeRecord}``; each record
    exposes ``.trees`` and ``.leaf_time``): for every ordered pair of record
    leaf-times ``t0 < t1`` the ``t1``-records' trees are layer-sliced at ``t0``,
    filtered, and pooled into a :class:`Stratum`.
    """

    def __init__(self, records, *, min_root_out_degree=2, min_post_cut_gap=0.05):
        if min_root_out_degree < 2:
            raise ValueError(
                "min_root_out_degree must be >= 2 to forbid 1->1 chain grafts"
            )
        self.min_root_out_degree = int(min_root_out_degree)
        self.min_post_cut_gap = float(min_post_cut_gap)

        recs = [r for r in records.values() if r.leaf_time is not None]
        start_times = sorted({float(r.leaf_time) for r in recs})

        strata = defaultdict(list)
        for late in recs:
            t1 = float(late.leaf_time)
            for tree in late.trees.values():
                for t0 in start_times:
                    if t0 >= t1:
                        break
                    strata[(t0, t1)].extend(self._donors(tree, t0))
        self.strata = {key: Stratum(d) for key, d in strata.items()}

    def _donors(self, tree, t0):
        """Yield the filtered layer-cell donors of ``tree`` clipped at ``t0``."""
        gap_floor = t0 + self.min_post_cut_gap
        for L, sub in layer_candidates(tree, t0):
            if sub.out_degree(L) < self.min_root_out_degree:
                continue
            if min(sub.nodes[c]["time"] for c in sub.successors(L)) < gap_floor:
                continue
            n_leaves = sum(1 for n in sub if sub.out_degree(n) == 0)
            yield Donor(tree, L, frozenset(sub.nodes()), n_leaves)

    def sample(self, start, end, rng, k=None):
        """A ``(root, subgraph)`` donor for the ``(start, end)`` stratum.

        Uniform when ``k`` is None, else the donor with leaf count nearest ``k``.
        Returns ``None`` if the bank holds no stratum.
        """
        st = self._stratum(start, end)
        return st.draw(rng, k).materialize() if st is not None else None

    def _stratum(self, start, end):
        """Exact ``(start, end)`` stratum, else the nearest by L1 distance."""
        key = (float(start), float(end))
        if key in self.strata:
            return self.strata[key]
        if not self.strata:
            return None
        return self.strata[
            min(self.strata, key=lambda p: abs(p[0] - key[0]) + abs(p[1] - key[1]))
        ]
