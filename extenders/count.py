"""Conditional count sampler for the conditional extenders.

The count model answers one question -- "how many active descendants does this
cell have at ``time + delta``?" -- conditional on the cell's observed state, its
time, and the extension horizon ``delta``. There is a single model: a small MLP
over ``[state, time, delta, noise]`` trained with the **energy score** (a strictly
proper scoring rule for implicit samplers), so repeated draws approximate the
conditional count *distribution*, not just its mean.

The only public entry point an extender uses is ``sample(cond, rng) -> int``,
where ``cond`` is a dict ``{"state", "time", "delta"}``. ``delta`` is required --
it is a model input, which is what lets an extender extend an arbitrary time gap
cleanly (e.g. fill +0.5 or +1.0 forward).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.energy_scores import energy_score as _energy_score


def energy_score(preds, y):
    """Energy score of ``m`` samples per example against the target (lower better).

    ``preds`` is ``(B, m)`` model draws, ``y`` is ``(B,)`` targets. In 1D the
    score is ``mean_i |x_i - y| - 0.5 * mean_{i!=j} |x_i - x_j|`` averaged over the
    batch -- the repulsive second term keeps the samples from collapsing to the
    conditional mean. Thin wrapper over the shared ``beta=1, lam=1`` energy
    score (:func:`utils.energy_scores.energy_score`).
    """
    return _energy_score(preds, y, beta=1.0, lam=1.0)


class _CountMLP(nn.Module):
    """Maps ``[feat, noise]`` to a non-negative count (softplus output)."""

    def __init__(self, feat_dim, noise_dim, hidden=128, n_layers=2):
        super().__init__()
        widths = [feat_dim + noise_dim] + [hidden] * n_layers
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        layers.append(nn.Linear(widths[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, feat, noise):
        return F.softplus(self.net(torch.cat([feat, noise], dim=-1))).squeeze(-1)


def _features(cond):
    """Assemble ``cond`` dict -> feature array ``[state, time, delta]``.

    Works for a single example (state ``(d,)``, scalar time/delta) -> ``(d+2,)``
    or a batch (state ``(N, d)``, time/delta ``(N,)``) -> ``(N, d+2)``.
    """
    state = np.asarray(cond["state"], dtype=np.float32)
    time = np.asarray(cond["time"], dtype=np.float32)
    delta = np.asarray(cond["delta"], dtype=np.float32)
    if state.ndim == 1:
        return np.concatenate([state, [float(time), float(delta)]]).astype(np.float32)
    return np.concatenate(
        [state, time.reshape(-1, 1), delta.reshape(-1, 1)], axis=1
    ).astype(np.float32)


class ConditionalCountSampler:
    """Energy-trained conditional count sampler.

    ``sample(cond, rng) -> int`` and ``sample_batch(cond, rng, n) -> (N, n)`` for
    eval; ``fit(cond, target, ...)`` trains; ``save``/``load`` round-trip weights +
    scaler so a config can point ``model_path`` at the trained file.
    """

    def __init__(
        self,
        *,
        feat_dim: int | None = None,
        model_path: str | None = None,
        noise_dim: int = 16,
        hidden: int = 128,
        n_layers: int = 2,
        min_count: int = 1,
        max_count: int | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.noise_dim = int(noise_dim)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.min_count = int(min_count)
        self.max_count = None if max_count is None else int(max_count)
        if model_path is not None:
            self.load(model_path)
            return
        if feat_dim is None:
            raise ValueError("feat_dim is required when model_path is not given")
        self.feat_dim = int(feat_dim)
        self.net = _CountMLP(
            self.feat_dim, self.noise_dim, self.hidden, self.n_layers
        ).to(self.device)
        self.feat_mean = torch.zeros(self.feat_dim, device=self.device)
        self.feat_std = torch.ones(self.feat_dim, device=self.device)

    def fit(
        self, cond, target, *, n_samples=8, epochs=300, batch_size=4096, lr=1e-3,
        weight_decay=0.0, seed=0, verbose=True,
    ):
        """Train on conditioning ``cond`` (dict of ``(N, ...)`` arrays) and counts
        ``target`` ``(N,)`` by minimizing the energy score."""
        torch.manual_seed(seed)
        X = torch.as_tensor(_features(cond), device=self.device)
        y = torch.as_tensor(np.asarray(target, np.float32), device=self.device)
        self.feat_mean = X.mean(0)
        self.feat_std = X.std(0).clamp_min(1e-6)
        Xn = (X - self.feat_mean) / self.feat_std
        opt = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )
        N = len(Xn)
        self.net.train()
        for epoch in range(epochs):
            perm = torch.randperm(N, device=self.device)
            running = 0.0
            for s in range(0, N, batch_size):
                idx = perm[s:s + batch_size]
                fb, yb = Xn[idx], y[idx]
                B = fb.shape[0]
                noise = torch.randn(B, n_samples, self.noise_dim, device=self.device)
                feat = fb[:, None, :].expand(B, n_samples, self.feat_dim)
                loss = energy_score(self.net(feat, noise), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * B
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d}  energy {running / N:.4f}")
        self.net.eval()
        return self

    def _clamp(self, K):
        K = np.maximum(self.min_count, K)
        if self.max_count is not None:
            K = np.minimum(self.max_count, K)
        return K

    @torch.no_grad()
    def sample(self, cond, rng) -> int:
        feat = (torch.as_tensor(_features(cond), device=self.device)
                - self.feat_mean) / self.feat_std
        noise = torch.as_tensor(
            rng.standard_normal(self.noise_dim).astype(np.float32), device=self.device
        )
        return int(self._clamp(int(round(float(self.net(feat, noise))))))

    @torch.no_grad()
    def sample_batch(self, cond, rng, n_samples=1):
        """Draw ``n_samples`` counts for each example in ``cond`` -> ``(N, n)`` ints."""
        X = torch.as_tensor(_features(cond), device=self.device)
        Xn = (X - self.feat_mean) / self.feat_std
        N = Xn.shape[0]
        noise = torch.as_tensor(
            rng.standard_normal((N, n_samples, self.noise_dim)).astype(np.float32),
            device=self.device,
        )
        feat = Xn[:, None, :].expand(N, n_samples, self.feat_dim)
        x = np.rint(self.net(feat, noise).cpu().numpy()).astype(int)
        return self._clamp(x)

    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "feat_dim": self.feat_dim, "noise_dim": self.noise_dim,
            "hidden": self.hidden, "n_layers": self.n_layers,
            "feat_mean": self.feat_mean.cpu(), "feat_std": self.feat_std.cpu(),
            "min_count": self.min_count, "max_count": self.max_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.feat_dim = int(ckpt["feat_dim"])
        self.noise_dim = int(ckpt["noise_dim"])
        self.hidden = int(ckpt["hidden"])
        self.n_layers = int(ckpt["n_layers"])
        self.min_count = int(ckpt["min_count"])
        self.max_count = ckpt["max_count"]
        self.net = _CountMLP(
            self.feat_dim, self.noise_dim, self.hidden, self.n_layers
        ).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.feat_mean = ckpt["feat_mean"].to(self.device)
        self.feat_std = ckpt["feat_std"].to(self.device)
        return self
