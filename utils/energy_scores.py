"""Energy-score scoring rules (torch), in one place.

Two related kernel-energy quantities, both built from the same pairwise-distance
primitive so they stay numerically consistent:

- :func:`energy_score` -- the **conditional energy score**: ``M`` model samples
  scored against a *single* observed target. A strictly proper scoring rule for
  implicit/posterior samplers (lower is better), so minimizing it fits the whole
  conditional distribution, not just its mean. This is the loss the conditional
  count sampler trains on and the FullEnergyDeltaTreeFormer split loss.

- :func:`energy_distance` -- the **two-sample energy distance**: a symmetric
  divergence between two *sets* of samples (Szekely-Rizzo). A from-scratch,
  precision-controlled alternative to ``geomloss.SamplesLoss(loss="energy")``
  for the aligner; used by ``StratifiedFeatureAligner(loss="energy_direct")``.

Conventions and numerics (shared by both):

- Pairwise squared distances are computed by **direct subtraction** then
  square-reduce, never a Gram expansion ``||u||^2 + ||v||^2 - 2<u,v>``. The Gram
  route is fewer FLOPs but routes through a matmul that silently downcasts to
  TF32/bf16 on modern accelerators and leaks ~1e-4 relative noise into
  ``|u-v|^2`` -- catastrophic near coincident samples, where the score's
  gradient lives. Peak memory is the same ``[..., M, M]``.
- ``beta in (0, 2]`` is the distance exponent (the score uses ``||.||^beta``).
  ``beta == 1`` (classical Gneiting-Raftery energy score) and ``beta == 2`` are
  specialised to skip ``pow``/``sqrt`` dispatch. For ``beta < 2`` we eps-clip the
  squared distance *before* the fractional root: fp cancellation can make it
  slightly negative, which would hit ``sqrt``/``pow`` with an undefined
  derivative. ``beta == 2`` has no root, so no clip (it would only add bias).
  Strict properness needs ``beta in (0, 2)``; ``beta == 2`` is included for
  completeness but is *not* strictly proper.
"""
from __future__ import annotations

import torch


def _pow_dist(sq: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    """``sq -> sq ** (beta/2)`` i.e. ``||.||^beta`` from squared distances.

    ``sq`` is ``>= 0`` in exact arithmetic; the eps-clip guards the fractional
    root against fp cancellation (see module docstring).
    """
    if beta == 2.0:
        return sq
    if beta == 1.0:
        return torch.sqrt(sq.clamp_min(eps))
    return torch.pow(sq.clamp_min(eps), beta / 2.0)


def _pairwise_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared L2 distances between sample sets, direct subtraction.

    ``x`` ``[..., N, D]``, ``y`` ``[..., M, D]`` -> ``[..., N, M]``. The
    ``[..., N, M, D]`` intermediate is the precision-safe peak (see module
    docstring); reduction is in the inputs' dtype.
    """
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)            # [..., N, M, D]
    return (diff * diff).sum(-1)                         # [..., N, M]


def energy_score(
    preds: torch.Tensor,                 # [B, M, *data]
    target: torch.Tensor,                # [B, *data]
    *,
    beta: float = 1.0,
    lam: float = 1.0,
    mask: torch.Tensor | None = None,    # broadcastable to [B, *data]
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    """Conditional energy score of ``M`` samples against one target (lower better).

    U-statistic form (Eq. 15-16 of arXiv:2502.02483):

        L_i = mean_j ||x_i - xhat_ij||^beta
              - lam / (2 M (M-1)) * sum_{j != j'} ||xhat_ij - xhat_ij'||^beta

    The data term pulls samples toward the target; the repulsive interaction
    term keeps them from collapsing to the conditional mean. Strict properness
    requires ``lam == 1`` and ``beta in (0, 2)``.

    Args:
      preds: ``[B, M, *data]`` -- ``M >= 2`` model samples per example, e.g. the
        ``M`` noise draws of a distributional splitter stacked along axis 1.
      target: ``[B, *data]`` -- the single observed target per example.
      beta: distance exponent in ``(0, 2]`` (default 1.0).
      lam: interaction weight in ``[0, 1]`` (default 1.0, strictly proper).
      mask: optional per-element weight broadcastable to ``[B, *data]`` (e.g. a
        tree's masked-node mask ``[B, N, 1]``). Zeroed elements drop out of every
        distance, so only the selected coordinates contribute to the score.
      eps: clip added inside fractional roots for stability.
      reduction: ``"mean"`` | ``"sum"`` | ``"none"`` over the batch.

    Returns:
      Scalar (``mean``/``sum``) or ``[B]`` (``none``).
    """
    if preds.ndim < 2:
        raise ValueError(
            f"preds must be [B, M, *data] with a population axis, got {tuple(preds.shape)}."
        )
    B, M = preds.shape[0], preds.shape[1]
    if M < 2:
        raise ValueError(f"energy score needs M >= 2 samples, got M={M}.")
    if target.shape[0] != B:
        raise ValueError(
            f"batch mismatch: preds {B}, target {target.shape[0]}."
        )

    p = preds.reshape(B, M, -1)                          # [B, M, D]
    y = target.reshape(B, -1)                            # [B, D]
    w = (
        torch.broadcast_to(mask, target.shape).reshape(B, -1).to(p.dtype)
        if mask is not None else None
    )

    # Data term: ||target - xhat_j||, optionally masked.
    diff = p - y.unsqueeze(1)                            # [B, M, D]
    if w is not None:
        diff = diff * w.unsqueeze(1)
    data_sq = (diff * diff).sum(-1)                      # [B, M]

    # Interaction term: pairwise ||xhat_j - xhat_j'||, optionally masked.
    pdiff = p.unsqueeze(2) - p.unsqueeze(1)              # [B, M, M, D]
    if w is not None:
        pdiff = pdiff * w[:, None, None, :]
    pair_sq = (pdiff * pdiff).sum(-1)                    # [B, M, M]

    data_term = _pow_dist(data_sq, beta, eps).mean(1)    # [B]
    pair = _pow_dist(pair_sq, beta, eps)                 # [B, M, M]
    off_diag = 1.0 - torch.eye(M, device=pair.device, dtype=pair.dtype)
    interaction = (pair * off_diag).sum((1, 2)) / (2.0 * M * (M - 1))
    per_sample = data_term - lam * interaction           # [B]

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"unknown reduction {reduction!r}.")


def energy_distance(
    a: torch.Tensor,                     # [B, N, *data]
    b: torch.Tensor,                     # [B, M, *data]
    *,
    beta: float = 1.0,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    """Two-sample energy distance between sample sets ``a`` and ``b``.

        ED(a, b) = 2 mean_ij ||a_i - b_j||^beta
                   - mean_ii' ||a_i - a_i'||^beta
                   - mean_jj' ||b_j - b_j'||^beta

    Symmetric, ``>= 0``, zero iff the empirical laws match (for ``beta in
    (0, 2)``). The biased V-statistic estimator (self terms include the zero
    diagonal, divided by ``N^2`` / ``M^2``) -- this matches
    ``geomloss.SamplesLoss(loss="energy")`` up to its ``1/2`` convention
    (geomloss returns ``ED / 2``).

    A from-scratch, precision-controlled alternative to the geomloss aligner
    loss (``loss="energy_direct"`` in :mod:`aligners.stratified`). Both inputs carry an explicit leading batch axis (use
    ``x[None]`` for the unbatched set-vs-set case).

    Args:
      a, b: ``[B, N, *data]`` / ``[B, M, *data]`` sample sets per example.
      beta: distance exponent in ``(0, 2]`` (default 1.0).
      eps: clip added inside fractional roots for stability.
      reduction: ``"mean"`` | ``"sum"`` | ``"none"`` over the batch.

    Returns:
      Scalar (``mean``/``sum``) or ``[B]`` (``none``).
    """
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("a, b must be [B, N, *data] with batch and sample axes.")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"batch mismatch: a {a.shape[0]}, b {b.shape[0]}.")

    B = a.shape[0]
    a2 = a.reshape(B, a.shape[1], -1)                    # [B, N, D]
    b2 = b.reshape(B, b.shape[1], -1)                    # [B, M, D]

    cross = _pow_dist(_pairwise_sq(a2, b2), beta, eps).mean((1, 2))   # [B]
    self_a = _pow_dist(_pairwise_sq(a2, a2), beta, eps).mean((1, 2))  # [B]
    self_b = _pow_dist(_pairwise_sq(b2, b2), beta, eps).mean((1, 2))  # [B]
    per_sample = 2.0 * cross - self_a - self_b           # [B]

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"unknown reduction {reduction!r}.")
