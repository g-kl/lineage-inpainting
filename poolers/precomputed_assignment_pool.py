from __future__ import annotations

import pickle

import numpy as np
import torch
import torch.nn as nn

from datasets.tree_record import TreeBatch


class PrecomputedAssignmentPool(nn.Module):
    """Parameter-free pooler that *replays a precomputed node->cell coupling*.

    The snap poolers (:class:`~poolers.nn_snapped_mean_pool.NNSnappedMeanPool`,
    :class:`~poolers.ot_snapped_mean_pool.OTSnappedMeanPool`) pool embeddings and
    snap them to a reference bank live, every forward pass. This pooler instead
    reads a coupling that was already computed once and cached to disk by
    ``analysis/lib/coupling/nearest_neighbor_coupling*`` -- an
    ``assignments_nn_*.pkl`` of the form::

        {tree_name: {internal_node_name: reference_cell_name | "unknown"}}

    (plus a top-level ``"nn_distances"`` book-keeping entry, which is ignored).
    Every internal node that carries an assignment is *filled in* with the
    embedding of its assigned reference cell; leaves keep their observed
    embedding and any node without an assignment (e.g. below the coupling's
    ``min_snap_time``, near the root) falls back to the bottom-up mean.

    Node names are globally unique across trees, so the nested coupling collapses
    to a single flat ``node_name -> embedding`` table built once at
    :meth:`bind_dataset`. Pooling is then a per-batch gather -- no model forward,
    no faiss search -- which makes this an exact, cheap replay of a frozen
    coupling (useful as a fixed teacher for a downstream splitter, or for eval).

    Args:
        dim:              Embedding dimensionality.
        assignments_path: Path to the ``assignments_nn_*.pkl`` to replay.
        max_children:     Unused; kept for API parity with the other poolers.
    """

    def __init__(
        self,
        dim: int,
        assignments_path: str,
        max_children: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.assignments_path = assignments_path
        self.max_children = max_children
        self._device_param = nn.Parameter(torch.empty(0), requires_grad=False)
        # Built by bind_dataset: node_name -> row in ``_assigned_emb``.
        self._node_to_row: dict[str, int] = {}
        self._assigned_emb: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    #  Reference table
    # ------------------------------------------------------------------ #

    def bind_dataset(self, dataset) -> None:
        """Load the cached coupling and build the flat ``node_name -> embedding``
        table from the dataset's reference features.

        ``reference_features`` is the same ``(n_cells, dim)`` matrix the dataset
        builds ``leaf_states`` from -- and the matrix the coupling snapped
        against -- so a filled internal node sits in exactly the leaf embedding
        space.
        """
        with open(self.assignments_path, "rb") as f:
            assignments = pickle.load(f)

        ref = np.asarray(dataset.reference_features, dtype=np.float32)
        obs_to_idx = {n: i for i, n in enumerate(dataset.states.obs_names)}

        # Collapse {tree: {node: cell}} -> flat {node: packed_row}, deduping
        # cells so ``_assigned_emb`` holds each referenced cell once.
        cell_to_row: dict[str, int] = {}
        ref_rows: list[int] = []          # packed row -> reference_features index
        node_to_row: dict[str, int] = {}
        n_assigned = n_unknown = n_missing = 0
        for tree_name, node_map in assignments.items():
            if tree_name == "nn_distances":
                continue
            for node_name, cell_name in node_map.items():
                idx = obs_to_idx.get(cell_name)
                if idx is None:
                    if cell_name == "unknown":
                        n_unknown += 1
                    else:
                        n_missing += 1
                    continue
                row = cell_to_row.get(cell_name)
                if row is None:
                    row = len(ref_rows)
                    cell_to_row[cell_name] = row
                    ref_rows.append(idx)
                node_to_row[node_name] = row
                n_assigned += 1

        if not node_to_row:
            raise ValueError(
                f"PrecomputedAssignmentPool: no node assignments in "
                f"{self.assignments_path} resolved against this dataset's cells "
                f"(unknown={n_unknown}, unmatched={n_missing}). Wrong file?"
            )

        self._node_to_row = node_to_row
        self._assigned_emb = torch.as_tensor(ref[ref_rows], dtype=torch.float32)
        print(
            f"[PrecomputedAssignmentPool] loaded {self.assignments_path}: "
            f"{n_assigned} assigned nodes over {len(cell_to_row)} unique cells "
            f"(unknown={n_unknown}, unmatched={n_missing})"
        )

    # ------------------------------------------------------------------ #
    #  Pooler interface
    # ------------------------------------------------------------------ #

    def forward(self, children_states: list[torch.Tensor]) -> torch.Tensor:
        """Mean-pool a list of children embeddings (no fill here -- the fill
        needs node identity, which only :meth:`pool_batch` has)."""
        return torch.stack(children_states).mean(dim=0)

    def pool_batch(self, batch: TreeBatch) -> torch.Tensor:
        """Mean-pool bottom-up, then overwrite assigned internal nodes with their
        coupled reference-cell embedding.

        Returns (B, N, dim) embeddings: each assigned internal node holds its
        cached reference cell, leaves hold their observed embedding, and every
        other node holds the bottom-up mean fallback.
        """
        # 1. Bottom-up mean pooling (fallback for unassigned nodes).
        states = batch.leaf_states.detach().clone()
        adj = batch.adj
        n_children = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        for d in range(batch.max_depth, 0, -1):
            targets = (batch.depth == d - 1) & ~batch.leaf_mask & batch.mask
            if not targets.any():
                continue
            children_mean = torch.bmm(adj, states) / n_children
            states = torch.where(targets.unsqueeze(-1), children_mean, states)

        # 2. Fill assigned nodes from the cached coupling.
        if self._assigned_emb is None or not self._node_to_row:
            return states

        B, N, _ = states.shape
        get = self._node_to_row.get
        rows = torch.tensor(
            [
                [get(n, -1) for n in names] + [-1] * (N - len(names))
                for names in batch.node_names
            ],
            dtype=torch.long,
            device=states.device,
        )
        fill = rows >= 0
        if fill.any():
            emb = self._assigned_emb.to(states.device)
            states[fill] = emb[rows[fill]].to(states.dtype)
        return states
