from __future__ import annotations

import torch
import torch.nn as nn
import networkx as nx
from geomloss import SamplesLoss

from datasets.tree_record import TreeBatch

class FlowMatchingSplit(nn.Module):
    """
    flow-matching split: learns a conditional velocity field that transports
    noise to node states. each non-root node is handled independently,
    conditioned on (root_state, node_time, time_delta).
    """

    def __init__(self, dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        
        self.dim = dim

        # input: x_t (dim) + root_state (dim) + t (1) + node_time (1) + time_delta (1)
        layers: list[nn.Module] = [nn.Linear(2 * dim + 3, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, root_state: torch.Tensor,
                node_time: torch.Tensor, time_delta: torch.Tensor) -> torch.Tensor:
        """predict velocity. all inputs (M, *)."""
        return self.net(torch.cat([x_t, t, root_state, node_time, time_delta], dim=-1))

    @staticmethod
    def _fm_sample(x1: torch.Tensor, device, dtype):
        """sample t, noise, interpolant, and target velocity for fm training."""
        M = x1.shape[0]
        t = torch.rand(M, 1, device=device, dtype=dtype)
        eps = torch.randn_like(x1)
        x_t = (1 - t) * eps + t * x1
        target_vel = x1 - eps
        return t, x_t, target_vel

    def batch_loss(self, batch: TreeBatch, target_states: torch.Tensor,
                   leaves_only: bool = False) -> torch.Tensor:
        """batched conditional flow-matching loss over non-root nodes."""
        B, N, dim = target_states.shape
        device, dtype = target_states.device, target_states.dtype

        root_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        root_mask[torch.arange(B, device=device), batch.root_idx] = True
        eval_mask = batch.mask & ~root_mask
        if leaves_only:
            eval_mask = eval_mask & batch.leaf_mask

        total_M = eval_mask.sum().item()
        if total_M == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        x1 = target_states[eval_mask]
        root_states = target_states[torch.arange(B, device=device), batch.root_idx]
        root_cond = root_states.unsqueeze(1).expand(-1, N, -1)[eval_mask]

        root_times = batch.node_times[torch.arange(B, device=device), batch.root_idx]
        node_t = batch.node_times[eval_mask].unsqueeze(-1)
        time_delta = node_t - root_times.unsqueeze(1).expand(-1, N)[eval_mask].unsqueeze(-1)

        t, x_t, target_vel = self._fm_sample(x1, device, dtype)
        pred_vel = self.forward(x_t, t, root_cond, node_t, time_delta)
        return (pred_vel - target_vel).pow(2).mean()

    @torch.no_grad()
    def split_tree_batch(self, batch: TreeBatch, root_states: torch.Tensor,
                         *, n_steps: int = 10) -> torch.Tensor:
        """generate node states by euler-integrating the learned velocity field. returns (B, N, dim)."""
        B, N = batch.mask.shape
        device, dtype = root_states.device, root_states.dtype
        dim = self.dim

        root_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        root_mask[torch.arange(B, device=device), batch.root_idx] = True
        eval_mask = (batch.mask & ~root_mask).unsqueeze(-1).float()

        root_cond = root_states.unsqueeze(1).expand(-1, N, -1)
        node_times = batch.node_times.reshape(B, N)
        root_times = node_times[torch.arange(B, device=device), batch.root_idx]
        node_t = node_times.unsqueeze(-1)
        time_delta = node_t - root_times.reshape(B, 1, 1)

        x = torch.randn(B, N, dim, device=device, dtype=dtype)
        dt = 1.0 / n_steps

        root_cond_flat = root_cond.reshape(B * N, dim)
        node_t_flat = node_t.reshape(B * N, 1)
        time_delta_flat = time_delta.reshape(B * N, 1)

        for step in range(n_steps):
            t = torch.full((B * N, 1), step * dt, device=device, dtype=dtype)
            vel = self.forward(x.reshape(B * N, dim), t, root_cond_flat, node_t_flat, time_delta_flat)
            x = x + vel.reshape(B, N, dim) * dt

        out = x * eval_mask
        out[torch.arange(B, device=device), batch.root_idx] = root_states
        return out * batch.mask.unsqueeze(-1).float()