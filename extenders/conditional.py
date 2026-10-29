"""Conditional-count extenders: real-donor graft and synthetic skeleton.

Two distinct sibling extenders share a common base that asks an injected count
sampler "how many descendants does this leaf have at the target time?" and then
place exactly that many leaves -- but they place them differently and never fall
back to each other:

* :class:`ConditionalGraftExtender` grafts a *real* donor clade whose leaf count
  is nearest the sampled ``K`` (from the :class:`~extenders.reference.SubtreeBank`).
  If the stratum has no donor, the leaf is left unextended -- no synthetic fill.
* :class:`ConditionalSkeletonExtender` builds a synthetic balanced-binary skeleton
  of ``K`` leaves. No donor bank.

The count comes from ``count_sampler.sample(cond, rng)`` with
``cond = {"state", "time", "delta"}`` where ``delta = target_leaf_time - leaf_time``
-- the extender is the only thing that touches the sampler, and only via
``sample``. The sampler's ``state`` is exactly the observed leaf state passed in
via ``leaf_states``; nothing is imputed -- the original leaf becomes an internal
observed anchor and the added nodes are unobserved.
"""

from __future__ import annotations

import logging

from .graft import GraftExtender
from .reference import graft_onto_leaf

logger = logging.getLogger(__name__)


def attach_skeleton(graph, parent, K, t_parent, t_end, name_gen):
    """Grow a balanced-binary + trifurcation skeleton of ``K`` leaves beneath
    ``parent`` in ``graph`` (mutates ``graph``); returns the new leaf names.

    ``K`` in {2, 3} become a direct bi/tri-furcation at ``t_parent``; ``K >= 4``
    bifurcates at the midpoint and recurses on ``ceil(K/2)`` and ``floor(K/2)``;
    ``K == 1`` becomes a single edge to one leaf at ``t_end``; ``K <= 0`` adds
    nothing. New nodes are drawn from ``name_gen``.
    """
    if K <= 0:
        return []
    if K in (1, 2, 3):
        leaves = []
        for _ in range(K):
            leaf = next(name_gen)
            graph.add_node(leaf, time=t_end)
            graph.add_edge(parent, leaf)
            leaves.append(leaf)
        return leaves
    # K >= 4: bifurcate at midpoint, recurse on ceil/floor halves
    t_mid = 0.5 * (t_parent + t_end)
    leaves = []
    for sub_K in ((K + 1) // 2, K // 2):
        intermediate = next(name_gen)
        graph.add_node(intermediate, time=t_mid)
        graph.add_edge(parent, intermediate)
        leaves += attach_skeleton(
            graph, intermediate, sub_K, t_mid, t_end, name_gen,
        )
    return leaves


class _ConditionalExtender(GraftExtender):
    """Shared base: hold a count sampler and ask it for K per leaf.

    Subclasses implement :meth:`_graft_leaf` to place ``K`` leaves their own way.
    """

    def __init__(
        self,
        *,
        count_sampler,
        min_root_out_degree: int = 2,
        min_post_cut_gap: float = 0.05,
        node_id_prefix: str = "ext",
    ):
        super().__init__(
            min_root_out_degree=min_root_out_degree,
            min_post_cut_gap=min_post_cut_gap,
            node_id_prefix=node_id_prefix,
        )
        self.count_sampler = count_sampler

    def _prepare(self, subtrees, target_leaf_time, leaf_states) -> bool:
        if leaf_states is None:
            logger.warning(
                "%s called without leaf_states; cannot sample counts, "
                "returning subtrees unchanged.", type(self).__name__,
            )
            return False
        return super()._prepare(subtrees, target_leaf_time, leaf_states)

    def _count_for(self, g, leaf, end, leaf_states, rng):
        """Sampled leaf count and time for ``leaf``, or None to skip it."""
        if leaf not in leaf_states:
            return None
        t_leaf = float(g.nodes[leaf]["time"])
        emb = leaf_states[leaf]
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        cond = {"state": emb, "time": t_leaf, "delta": end - t_leaf}
        return self.count_sampler.sample(cond, rng), t_leaf


class ConditionalGraftExtender(_ConditionalExtender):
    """Graft a real donor clade whose leaf count is nearest the sampled ``K``."""

    def _graft_leaf(self, g, leaf, start, end, rng, leaf_states, name_gen):
        ck = self._count_for(g, leaf, end, leaf_states, rng)
        if ck is None:
            return
        K, _ = ck
        cand = self.bank.sample(start, end, rng, k=K)
        if cand is not None:
            graft_onto_leaf(g, leaf, *cand, name_gen)
        # no donor for this stratum -> leaf is left unextended (no synthetic fill)


class ConditionalSkeletonExtender(_ConditionalExtender):
    """Build a synthetic balanced-binary skeleton of the sampled ``K`` leaves."""

    needs_bank = False

    def _graft_leaf(self, g, leaf, start, end, rng, leaf_states, name_gen):
        ck = self._count_for(g, leaf, end, leaf_states, rng)
        if ck is None:
            return
        K, t_leaf = ck
        attach_skeleton(g, leaf, K, t_leaf, end, name_gen)
