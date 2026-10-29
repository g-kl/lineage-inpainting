"""Conditional flow-matching bridge (x_1-prediction).

Convention (shared by every bridge):

    x_0   noise / source            at t = 0
    x_1   real data / target        at t = 1
    x_t = (1 - t) * x_0 + t * x_1   (+ sigma_t * eps for stochastic interpolants)

The model predicts ``x_1`` (the data) directly, and sampling integrates the
probability-flow ODE *forward* in time, t: 0 -> 1, from noise to data.

A bridge owns only the geometry of the path: how to build ``x_t`` for training
(:meth:`interpolate`) and how to advance one ODE step given an ``x_1`` estimate
(:meth:`step`). It is agnostic to masking and to any projection space;
callers (e.g. BridgeTreeFlow) clamp anchors and project into the model's
working space around these calls.
"""

import torch


def pad_t_like_x(t, x):
    """Reshape a per-element time ``t`` to broadcast against ``x``.

    ``t`` is ``(B,)`` (one time per batch element); ``x`` is ``(B, ..., d)``.
    A python scalar is returned unchanged.
    """
    if isinstance(t, (float, int)):
        return t
    return t.view(-1, *([1] * (x.dim() - 1)))


class CFMBridge:
    """Linear conditional flow matching, ``x_1``-prediction parameterization.

        x_t          = (1 - t) * x_0 + t * x_1   (+ sigma * eps)
        velocity(x_t)= (x_1 - x_t) / (1 - t)

    ``sigma`` adds isotropic noise to the interpolant (the "stochastic
    interpolant" variant). The default ``sigma=0`` is the deterministic linear
    path.
    """

    def __init__(self, sigma: float = 0.0):
        self.sigma = float(sigma)

    # -- path geometry (override for other schedules) --------------------

    def mu_t(self, x_0, x_1, t):
        """Mean of the path, E[x_t | x_0, x_1]."""
        tt = pad_t_like_x(t, x_0)
        return (1.0 - tt) * x_0 + tt * x_1

    def sigma_t(self, t):
        """Std of the path. Constant by default."""
        return self.sigma

    def velocity_from_x1(self, x_t, x1_pred, t):
        """Probability-flow velocity at ``x_t`` given an ``x_1`` estimate.

        For the linear path this is ``(x_1 - x_t) / (1 - t)``.
        """
        one_minus_t = (1.0 - pad_t_like_x(t, x_t)).clamp_min(1e-6)
        return (x1_pred - x_t) / one_minus_t

    # -- training --------------------------------------------------------

    def sample_t(self, batch_size, device):
        """Draw the per-element flow time uniformly on [0, 1)."""
        return torch.rand(batch_size, device=device)

    def prior(self, shape, device=None, dtype=None):
        """Sample the source ``x_0 ~ N(0, I)`` the flow starts from at t=0."""
        return torch.randn(shape, device=device, dtype=dtype)

    def interpolate(self, x_0, x_1, t):
        """Build ``x_t`` for a training step (caller supplies ``t``)."""
        x_t = self.mu_t(x_0, x_1, t)
        s = self.sigma_t(t)
        if isinstance(s, (float, int)):
            if s != 0.0:
                x_t = x_t + s * torch.randn_like(x_t)
        else:
            x_t = x_t + pad_t_like_x(s, x_t) * torch.randn_like(x_t)
        return x_t

    # -- sampling (forward ODE, t: 0 -> 1) -------------------------------

    def step(self, x_t, x1_pred, t, dt):
        """One forward Euler step of size ``dt`` toward the data."""
        return x_t + dt * self.velocity_from_x1(x_t, x1_pred, t)

    def sample(self, model, x_0, n_steps=10, mask=None, return_x1=False,
               return_trajectory=False, mode="ode", **z):
        """Integrate noise ``x_0`` -> data over ``n_steps``.

        ``model(x_t=..., t=..., **z) -> x_1`` predicts the data (same space as
        ``x_t``). ``mask`` (broadcastable to ``x_0``) restricts the flow to a
        subset of coordinates; the rest stay clamped at their ``x_0`` value --
        this is how tree generation pins observed anchors. ``return_x1`` returns
        the model's final data prediction (the denoised endpoint) instead of the
        integrated state.

        ``mode`` selects the per-step update:

        - ``"ode"`` (default): advance the probability-flow ODE with
          :meth:`step` -- the classical CFM sampler.
        - ``"bridge"``: re-noise the model's ``x_1`` prediction to the next time
          level with the *training-time* forward process (:meth:`interpolate`),
          reusing the starting ``x_0`` noise. Valid when the model is trained to
          sample the posterior ``p(x_1 | x_t)`` directly (e.g. a model trained
          with an energy score): every model input then stays exactly on the
          interpolant distribution it was trained on, so a one-or-few-step
          bridge rollout is self-consistent rather than an ODE approximation.

        Not wrapped in ``no_grad``: the caller controls
        gradients (pooling backprops through this; eval wraps it in no_grad).
        """
        if mode not in ("ode", "bridge"):
            raise ValueError(f"mode must be 'ode' or 'bridge', got {mode!r}")
        use_bridge = mode == "bridge"
        b = x_0.shape[0]
        x_t = x_0
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x_0.device, dtype=x_0.dtype)
        traj = [x_t]
        x1_pred = x_0
        for k in range(n_steps):
            t = ts[k].expand(b)
            x1_pred = model(x_t=x_t, t=t, **z)
            if use_bridge:
                updated = self.interpolate(x_0, x1_pred, ts[k + 1].expand(b))
            else:
                updated = self.step(x_t, x1_pred, t, ts[k + 1] - ts[k])
            x_t = updated if mask is None else torch.where(mask, updated, x_t)
            if return_trajectory:
                traj.append(x_t)
        out = x1_pred if return_x1 else x_t
        return (out, torch.stack(traj)) if return_trajectory else out
