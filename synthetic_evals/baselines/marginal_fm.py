"""Marginal flow-matching baseline conditioned on (t_src, t_tgt).

Models the transition kernel between cells at two arbitrary observation
times using independent samples from each marginal. No tree structure --
serving as the "marginal only" counterpart to the pool-split model in the
crossing/non-crossing benchmark.

Training (per step):
  * pick a random adjacent obs-time pair (t_src, t_tgt)
  * draw a minibatch of source and target cells (independently!)
  * sample flow time t in [0, 1]; build x_t = (1 - t) x_src + t x_tgt
  * MSE between predicted velocity and (x_tgt - x_src)

Inference (``predict(x_src, t_src, t_tgt)``):
  * Euler-integrate the learned velocity from x_src with t advancing 0 -> 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VelocityField(nn.Module):
    """MLP velocity field v(x_t, t, t_src, t_tgt) -> R^dim."""

    def __init__(self, dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        # inputs: x_t (dim) | flow_time (1) | t_src (1) | t_tgt (1)
        in_dim = dim + 3
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, t_src, t_tgt):
        return self.net(torch.cat([x_t, t, t_src, t_tgt], dim=-1))


@dataclass
class TimeBank:
    """Pool of cells grouped by observation time.

    ``by_time[t]`` is a (N_t, dim) tensor of all cells observed at time t.
    """
    by_time: dict[float, torch.Tensor]

    @classmethod
    def from_dataset(cls, dataset) -> "TimeBank":
        """Collect a dataset's leaf states into per-time pools."""
        by_time: dict[float, list[torch.Tensor]] = {}
        # One-row-per-(leaf, obs_time) AnnData with obs["time"]. Datasets that
        # rebind ``states`` to something else (FateBiasDataset points it at the
        # reference bank) expose the training leaves as ``train_states``.
        states = getattr(dataset, "train_states", dataset.states)
        X = states.X  # (N, dim) numpy
        times = np.asarray(states.obs["time"], dtype=np.float64)
        for t in np.unique(times):
            mask = times == t
            by_time[float(t)] = torch.from_numpy(X[mask]).float()
        return cls(by_time=by_time)

    def times(self) -> list[float]:
        return sorted(self.by_time.keys())

    def sample(self, t: float, n: int, device) -> torch.Tensor:
        pool = self.by_time[t]
        idx = torch.randint(0, pool.shape[0], (n,))
        return pool[idx].to(device)


def adjacent_pairs(times: list[float]) -> list[tuple[float, float]]:
    """Adjacent (t_src, t_tgt) pairs from a sorted list of obs times."""
    times = sorted(times)
    return [(times[i], times[i + 1]) for i in range(len(times) - 1)]


def train(
    bank: TimeBank,
    *,
    dim: int = 1,
    hidden: int = 256,
    n_layers: int = 4,
    n_steps: int = 4000,
    batch_size: int = 256,
    lr: float = 3e-4,
    pairs: list[tuple[float, float]] | None = None,
    device: str | torch.device = "cuda",
    log_every: int = 200,
    seed: int = 0,
) -> VelocityField:
    """Train the marginal-FM velocity field. Returns the trained network."""
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = VelocityField(dim=dim, hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if pairs is None:
        pairs = adjacent_pairs(bank.times())
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Need at least one (t_src, t_tgt) pair to train.")

    # Time normalisation for the network input: linearly map [t_min, t_max]
    # to [0, 1] so the input is well-scaled regardless of obs-time choice.
    all_ts = [t for pair in pairs for t in pair]
    t_min, t_max = float(min(all_ts)), float(max(all_ts))
    t_span = (t_max - t_min) if t_max != t_min else 1.0
    norm = lambda t: (t - t_min) / t_span  # noqa: E731

    losses: list[float] = []
    rng = np.random.default_rng(seed)

    for step in range(n_steps):
        t_src, t_tgt = pairs[rng.integers(len(pairs))]
        x_src = bank.sample(t_src, batch_size, device)
        x_tgt = bank.sample(t_tgt, batch_size, device)

        t = torch.rand(batch_size, 1, device=device)
        x_t = (1.0 - t) * x_src + t * x_tgt
        target_vel = x_tgt - x_src

        t_src_in = torch.full((batch_size, 1), norm(t_src), device=device)
        t_tgt_in = torch.full((batch_size, 1), norm(t_tgt), device=device)
        pred_vel = model(x_t, t, t_src_in, t_tgt_in)
        loss = (pred_vel - target_vel).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % log_every == 0 or step == n_steps - 1:
            recent = float(np.mean(losses[-log_every:]))
            logger.info("[marginal-fm] step %4d/%d  loss=%.5f", step, n_steps, recent)

    # Stash normalisation on the module so predict() can use it without
    # the caller re-deriving it from the training pairs.
    model.t_min = t_min  # type: ignore[attr-defined]
    model.t_span = t_span  # type: ignore[attr-defined]
    return model


@torch.no_grad()
def predict(
    model: VelocityField,
    x_src: torch.Tensor,
    t_src: float,
    t_tgt: float,
    *,
    n_steps: int = 50,
) -> torch.Tensor:
    """Integrate the learned velocity from x_src (at t_src) to t_tgt.

    ``x_src`` is (M, dim). Returns (M, dim) on the same device.
    """
    model.eval()
    device = x_src.device
    M = x_src.shape[0]
    norm = lambda t: (t - model.t_min) / model.t_span  # type: ignore[attr-defined]
    t_src_in = torch.full((M, 1), norm(t_src), device=device)
    t_tgt_in = torch.full((M, 1), norm(t_tgt), device=device)

    x = x_src.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((M, 1), i * dt, device=device)
        v = model(x, t, t_src_in, t_tgt_in)
        x = x + v * dt
    return x
