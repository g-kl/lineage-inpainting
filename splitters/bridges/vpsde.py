"""VPSDE diffusion bridge -- a drop-in for the CFM bridge.

Exposes the same interface as :class:`splitters.bridges.flow_matching.CFMBridge`
so it slots into the BridgeTreeFlow / BridgeDeltaTreeFlow models unchanged. The
model still predicts ``x_1`` (the data) directly; only the path geometry differs
-- a variance-preserving SDE instead of the linear CFM interpolant -- so the
multi-step rollout is a *stochastic* reverse-SDE sampler rather than CFM's
deterministic probability-flow ODE.

Convention (identical to CFM, applied end to end):

    x_0   noise / source     at flow time t = 0
    x_1   data / target       at flow time t = 1

Internally the VPSDE schedule lives on its native diffusion clock ``s = 1 - t``
(``s = 0`` is clean data, ``s = 1`` is noise); only the public time axis flips.

Interface used by the models (mirrors CFMBridge):
    prior(shape, device, dtype)        -> x_0 ~ N(0, I)
    sample_t(batch_size, device)       -> per-element flow time ~ U[0, 1)
    interpolate(x_0, x_1, t)           -> x_t for a training step
    sample(model, x_0, n_steps, mask, return_x1, return_trajectory, **z)
                                       -> reverse-SDE rollout, noise -> data
"""

import numpy as np
import torch


def stp(s, ts: torch.Tensor):
    """Scalar-tensor product: broadcast scalar/per-batch ``s`` against ``ts``."""
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


class VPSDE:
    """Variance-preserving SDE ``dx = f(x, t) dt + g(t) dw`` on ``0 <= t <= 1``
    (its native diffusion clock; ``t=0`` clean, ``t=1`` noise)."""

    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def drift(self, x, t):
        """Drift coefficient f(x, t)."""
        return -0.5 * stp(self.squared_diffusion(t), x)

    def diffusion(self, t):
        """Diffusion coefficient g(t)."""
        return self.squared_diffusion(t) ** 0.5

    def squared_diffusion(self, t):
        """Beta(t) -- the noise schedule."""
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def squared_diffusion_integral(self, s, t):
        """Integral of beta(tau) from s to t."""
        return self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5

    def skip_alpha(self, s, t):
        """Alpha for the skip connection from time s to t."""
        x = -self.squared_diffusion_integral(s, t)
        return x.exp() if torch.is_tensor(x) else float(np.exp(x))

    def skip_beta(self, s, t):
        """Beta for the skip connection from time s to t."""
        return 1.0 - self.skip_alpha(s, t)

    def cum_alpha(self, t):
        """Cumulative alpha from 0 to t."""
        return self.skip_alpha(0, t)

    def cum_beta(self, t):
        """Cumulative beta from 0 to t."""
        return self.skip_beta(0, t)

    def marginal_prob(self, x0, t):
        """Mean and std of the forward marginal q(x_t | x0) at diffusion time t."""
        alpha = self.cum_alpha(t)
        beta = self.cum_beta(t)
        mean = stp(alpha ** 0.5, x0)   # E[x_t | x0]
        std = beta ** 0.5              # Cov[x_t | x0] ** 0.5
        return mean, std

    def sample_forward(self, x0, eps, t):
        """Draw x_t ~ q(x_t | x0) at diffusion time t using noise ``eps``."""
        mean, std = self.marginal_prob(x0, t)
        return mean + stp(std, eps)


class DiffusionBridge:
    """VPSDE bridge with the :class:`CFMBridge` interface (x_1-prediction).

    See the module docstring for the convention. The rollout is a stochastic
    reverse-SDE sampler, so repeated ``sample`` calls give different draws --
    the point of using a diffusion bridge over the deterministic CFM ODE.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, sigma: float = 0.0):
        # ``sigma`` is accepted-and-ignored so a config merged over a CFMBridge
        # spec (which carries ``sigma``) still instantiates. The VPSDE has its
        # own noise schedule.
        del sigma
        self.sde = VPSDE(beta_min=beta_min, beta_max=beta_max)

    # -- training (mirrors CFMBridge) ------------------------------------

    def sample_t(self, batch_size, device):
        """Draw the per-element flow time uniformly on [0, 1)."""
        return torch.rand(batch_size, device=device)

    def prior(self, shape, device=None, dtype=None):
        """Sample the source ``x_0 ~ N(0, I)`` the flow starts from at t=0."""
        return torch.randn(shape, device=device, dtype=dtype)

    def interpolate(self, x_0, x_1, t):
        """Build ``x_t`` for a training step: a draw from the VPSDE forward
        marginal ``q(x_t | x_1)`` at diffusion clock ``s = 1 - t``, using the
        supplied ``x_0`` as the Gaussian noise. At ``t=1`` this returns ``x_1``
        (data, alpha=1/beta=0); at ``t=0`` it is ~``x_0`` (pure noise)."""
        return self.sde.sample_forward(x_1, x_0, 1.0 - t)

    # -- one reverse-SDE step (diffusion-clock time) ---------------------

    def sample_step(self, s_curr, s_next, x_t, x_1_pred):
        """Single Euler-Maruyama reverse-SDE step on the diffusion clock.

        ``s_curr``/``s_next`` are diffusion-clock times (``s=0`` is clean data);
        generation walks them downward from 1 (noise) to 0 (data). Takes an
        ``x_1`` (data) prediction, converts it to a score, and steps the reverse
        SDE. The final step (``s_next == 0``) is the deterministic posterior
        mean (no injected noise)."""
        alpha = self.sde.cum_alpha(s_curr)
        beta = self.sde.cum_beta(s_curr)

        # data prediction -> noise prediction -> score = -noise / sqrt(beta)
        noise_pred = (x_t - stp(alpha ** 0.5, x_1_pred)) / stp(beta ** 0.5, torch.ones_like(x_t))
        score = stp(-beta.rsqrt(), noise_pred)

        # reverse drift: f(x,s) - g(s)^2 * score(x,s); step backward in s
        drift = self.sde.drift(x_t, s_curr)
        diffusion = self.sde.diffusion(s_curr)
        reverse_drift = drift - stp(diffusion ** 2, score)

        dt = s_curr - s_next
        mean = x_t - reverse_drift * dt
        if float(s_next) != 0.0:
            sigma = diffusion * dt.sqrt()
            return mean + stp(sigma, torch.randn_like(x_t))
        return mean

    # -- sampling (reverse SDE, flow time t: 0 -> 1) ---------------------

    def sample(self, model, x_0, n_steps=10, mask=None, return_x1=False,
               return_trajectory=False, mode="sde", **z):
        """Integrate noise ``x_0`` -> data over ``n_steps`` with the reverse SDE.

        Same signature/semantics as :meth:`CFMBridge.sample`:
        ``model(x_t=..., t=..., **z) -> x_1`` predicts the data; ``mask``
        (broadcastable to ``x_0``) restricts the flow to a coordinate subset and
        the rest stay clamped at their ``x_0`` value (this pins tree anchors);
        ``return_x1`` returns the model's final data prediction (denoised
        endpoint) instead of the integrated state.

        ``mode`` selects the per-step update:

        - ``"sde"`` (default): one Euler-Maruyama reverse-SDE step via
          :meth:`sample_step` -- the stochastic diffusion sampler.
        - ``"bridge"``: re-noise the model's ``x_1`` prediction to the next time
          level with the *training-time* forward marginal (:meth:`interpolate`),
          reusing the starting ``x_0`` noise. Valid for a posterior-sampling
          model (e.g. an energy-score-trained model): inputs stay on the
          training interpolant distribution, so a one-or-few-step bridge rollout
          is self-consistent rather than a reverse-SDE approximation.

        Not wrapped in ``no_grad`` -- the caller controls gradients, matching the
        CFM bridge.
        """
        if mode not in ("sde", "bridge"):
            raise ValueError(f"mode must be 'sde' or 'bridge', got {mode!r}")
        use_bridge = mode == "bridge"
        b = x_0.shape[0]
        x_t = x_0
        # diffusion clock s: 1 -> 0  (equivalently flow time t: 0 -> 1)
        ss = torch.linspace(1.0, 0.0, n_steps + 1, device=x_0.device, dtype=x_0.dtype)
        traj = [x_t]
        x1_pred = x_0
        for k in range(n_steps):
            s_curr, s_next = ss[k], ss[k + 1]
            t = (1.0 - s_curr).expand(b)            # flow time the model trains on
            x1_pred = model(x_t=x_t, t=t, **z)
            if use_bridge:
                updated = self.interpolate(x_0, x1_pred, (1.0 - s_next).expand(b))
            else:
                updated = self.sample_step(s_curr, s_next, x_t, x1_pred)
            x_t = updated if mask is None else torch.where(mask, updated, x_t)
            if return_trajectory:
                traj.append(x_t)
        out = x1_pred if return_x1 else x_t
        return (out, torch.stack(traj)) if return_trajectory else out

    def __repr__(self):
        return f"DiffusionBridge(beta_min={self.sde.beta_0}, beta_max={self.sde.beta_1})"
