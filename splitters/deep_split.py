from __future__ import annotations

import torch
import torch.nn as nn
from geomloss import SamplesLoss

from datasets.tree_record import TreeBatch


class MLPSplit(nn.Module):
    """Time-conditioned additive MLP split with per-child-index heads.

    child_k = parent + rho_k(phi_k(cat(parent, time_embed(t_child))))
    """

    def __init__(self, dim: int, hidden: int | None = 128, max_children: int = 2):
        super().__init__()
        self.dim = dim
        self.max_children = max_children
        h = hidden or dim
        self.time_embed = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.phis = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dim, h), nn.GELU(), nn.Linear(h, h))
            for _ in range(max_children)
        ])
        self.rhos = nn.ModuleList([
            nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, dim))
            for _ in range(max_children)
        ])

    def forward(self, parent_state: torch.Tensor, n_children: int,
                children_times: list[torch.Tensor]) -> list[torch.Tensor]:
        results = []
        for i in range(n_children):
            t = children_times[i]
            if t.dim() == 0:
                t = t.unsqueeze(-1)
            conditioned = torch.cat([parent_state, self.time_embed(t)], dim=-1)
            results.append(parent_state + self.rhos[i](self.phis[i](conditioned)))
        return results

    def split_tree_batch(self, batch: TreeBatch, root_states: torch.Tensor) -> torch.Tensor:
        """Cascaded top-down split. Used at inference. Returns (B, N, dim)."""
        B, N = batch.mask.shape
        device, dtype, dim = root_states.device, root_states.dtype, self.dim

        recon = torch.zeros(B, N, dim, device=device, dtype=dtype)
        recon[torch.arange(B, device=device), batch.root_idx] = root_states

        n_children = batch.adj.sum(dim=-1)
        par = batch.parent_idx.clamp(min=0)
        time_emb = self.time_embed(batch.node_times.unsqueeze(-1))

        for d in range(1, batch.max_depth + 1):
            depth_mask = (batch.depth == d) & batch.mask
            if not depth_mask.any():
                continue
            parent_states = torch.gather(recon, 1, par.unsqueeze(-1).expand(-1, -1, dim))
            parent_nc = torch.gather(n_children, 1, par)
            overflow = parent_nc > self.max_children
            conditioned = torch.cat([parent_states, time_emb], dim=-1)

            new_states = torch.zeros_like(recon)
            for k in range(self.max_children):
                head_mask = (batch.child_index == k) & depth_mask & ~overflow
                if head_mask.any():
                    new_states += (
                        (parent_states + self.rhos[k](self.phis[k](conditioned)))
                        * head_mask.unsqueeze(-1).float()
                    )

            overflow_mask = overflow & depth_mask
            new_states += parent_states * overflow_mask.unsqueeze(-1).float()
            recon = recon + new_states * depth_mask.unsqueeze(-1).float()
        return recon

    def split_teacher_forced(self, batch: TreeBatch, target_states: torch.Tensor) -> torch.Tensor:
        """Predict each non-root node from its TARGET parent state.

        Each parent-to-child split is independent, so errors do not propagate
        down the tree. Returns (B, N, dim) with target values left in place
        at root positions.
        """
        B, N, dim = target_states.shape

        par = batch.parent_idx.clamp(min=0)
        n_children = batch.adj.sum(dim=-1)

        parent_states = torch.gather(target_states, 1, par.unsqueeze(-1).expand(-1, -1, dim))
        parent_nc = torch.gather(n_children, 1, par)
        overflow = parent_nc > self.max_children

        time_emb = self.time_embed(batch.node_times.unsqueeze(-1))
        conditioned = torch.cat([parent_states, time_emb], dim=-1)

        nonroot = batch.mask & (batch.depth > 0)

        new_states = torch.zeros_like(target_states)
        for k in range(self.max_children):
            head_mask = (batch.child_index == k) & nonroot & ~overflow
            if head_mask.any():
                new_states = new_states + (
                    (parent_states + self.rhos[k](self.phis[k](conditioned)))
                    * head_mask.unsqueeze(-1).float()
                )

        overflow_mask = overflow & nonroot
        new_states = new_states + parent_states * overflow_mask.unsqueeze(-1).float()

        return torch.where(nonroot.unsqueeze(-1), new_states, target_states)

    def batch_loss(self, batch: TreeBatch, target_states: torch.Tensor,
                   leaves_only: bool = False) -> torch.Tensor:
        """Energy distance between predicted and target child sets, per parent.

        Predictions are teacher-forced: each child is generated from its
        ground-truth parent state, so each parent-to-child split is scored
        independently.
        """
        B, N, dim = target_states.shape
        device, dtype = target_states.device, target_states.dtype

        par = batch.parent_idx.clamp(min=0)
        n_children = batch.adj.sum(dim=-1).long()
        is_parent = (n_children > 0) & batch.mask

        if leaves_only:
            is_leaf = (n_children == 0) & batch.mask
            nonroot = batch.mask & (batch.depth > 0)
            has_nonleaf_child = torch.zeros(B, N, dtype=torch.long, device=device)
            has_nonleaf_child.scatter_add_(1, par, (~is_leaf & nonroot).long())
            is_parent = is_parent & (has_nonleaf_child == 0)

        if not is_parent.any():
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        recon = self.split_teacher_forced(batch, target_states)

        # Collect (batch, parent, child_rank, node) for every non-root node
        # whose parent is one we want to score.
        child_nodes = batch.mask & (batch.depth > 0)
        b_idx, n_idx = torch.where(child_nodes)
        p_idx = par[b_idx, n_idx]
        c_rank = batch.child_index[b_idx, n_idx].long()

        parent_valid = is_parent[b_idx, p_idx]
        b_idx, n_idx, p_idx, c_rank = (
            b_idx[parent_valid], n_idx[parent_valid],
            p_idx[parent_valid], c_rank[parent_valid],
        )
        parent_nc = n_children[b_idx, p_idx]

        # Group parents by child count so each group is one SamplesLoss call.
        energy = SamplesLoss(loss="energy")
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_parents = 0

        for nc in parent_nc.unique():
            nc = nc.item()
            if nc == 0:
                continue

            sel = parent_nc == nc
            b_sel, n_sel, p_sel, r_sel = b_idx[sel], n_idx[sel], p_idx[sel], c_rank[sel]

            parent_ids = b_sel * N + p_sel
            unique_parents, inv = parent_ids.unique(return_inverse=True)
            G = unique_parents.shape[0]

            pred_children = torch.zeros(G, nc, dim, device=device, dtype=dtype)
            tgt_children = torch.zeros(G, nc, dim, device=device, dtype=dtype)
            pred_children[inv, r_sel] = recon[b_sel, n_sel]
            tgt_children[inv, r_sel] = target_states[b_sel, n_sel]

            total_loss = total_loss + energy(pred_children, tgt_children).sum()
            total_parents += G

        if total_parents == 0:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        return total_loss / total_parents