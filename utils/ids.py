from __future__ import annotations

import math
import sys
from typing import Literal, Optional, Tuple, Union

import torch
from tqdm import tqdm

EPSILON = sys.float_info.epsilon
_P_NORM = Literal["max", 1, 2]


class IDS:
    """
    Instantiate with hyperparameters, then call :meth:`estimate` (or the
    instance directly) to compute dependence between tensors.

    Args:
        num_terms:  Number of terms in the Hermite-like feature expansion.
        bandwidth:  Bandwidth parameter for the Gaussian kernel factor.
        p_norm:     Aggregation across expansion terms:
                    ``'max'`` | ``1`` (mean) | ``2`` (root-mean-square).
        seed:       Random seed for reproducibility. ``None`` to skip seeding.

    Example::

        loss = IDS(bandwidth=0.5)
        C = loss.estimate(X, Y)
        C, p = loss.estimate(X, Y, p_val=True, num_tests=500)
    """

    def __init__(
        self,
        num_terms: int = 6,
        bandwidth: float = 0.5,
        p_norm: _P_NORM = "max",
        seed: Optional[int] = 1717,
    ) -> None:
        self.num_terms = num_terms
        self.bandwidth = bandwidth
        self.p_norm = p_norm
        self.seed = seed

    def estimate(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        *,
        p_val: bool = False,
        num_tests: int = 100,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute the IDS between X and Y.

        Args:
            X:          Tensor of shape ``(n, dx)``.
            Y:          Optional tensor of shape ``(n, dy)``.  If ``None``,
                        computes the self-dependence structure of X.
            p_val:      If ``True``, also return per-entry p-values via
                        permutation testing.
            num_tests:  Number of permutations when ``p_val=True``.

        Returns:
            C:          Dependence matrix of shape ``(dx, dy)`` (or ``(dx, dx)``).
            p_vals:     *(only when* ``p_val=True`` *)* Same shape as C.
        """
        self._maybe_seed()
        C = self._core(X, Y)

        if p_val:
            p_vals = self._permutation_p_values(C, X, Y, num_tests=num_tests)
            return C, p_vals
        return C

    def __call__(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Shorthand: ``loss(X, Y)`` is equivalent to ``loss.estimate(X, Y)``."""
        return self.estimate(X, Y, **kwargs)

    def __repr__(self) -> str:
        return (
            f"IDS(num_terms={self.num_terms}, bandwidth={self.bandwidth}, "
            f"p_norm={self.p_norm!r}, seed={self.seed})"
        )

    def _maybe_seed(self) -> None:
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

    def _feature_map(self, Z: torch.Tensor) -> torch.Tensor:
        """Hermite-like random feature expansion."""
        gauss = torch.exp(-self.bandwidth * Z ** 2)
        terms = [
            gauss * Z ** i / math.sqrt(math.factorial(i))
            for i in range(self.num_terms)
        ]
        return torch.cat(terms, dim=-1)

    @staticmethod
    def _center(Z: torch.Tensor) -> torch.Tensor:
        return Z - Z.mean(dim=0, keepdim=True)

    def _core(self, X: torch.Tensor, Y: Optional[torch.Tensor]) -> torch.Tensor:
        n, dx = X.shape
        Xf = self._center(self._feature_map(X))

        if Y is not None:
            dy = Y.shape[1]
            Yf = self._center(self._feature_map(Y))

            cov = Xf.T @ Yf
            x_std = torch.sqrt((Xf ** 2).sum(dim=0))
            y_std = torch.sqrt((Yf ** 2).sum(dim=0))

            C = cov / (x_std[:, None] + EPSILON) / (y_std[None, :] + EPSILON)
            C = C.reshape(self.num_terms, dx, self.num_terms, dy)
        else:
            C = torch.corrcoef(Xf.T)
            C = C.reshape(self.num_terms, dx, self.num_terms, dx)

        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0).abs()
        return self._aggregate(C)

    def _aggregate(self, C: torch.Tensor) -> torch.Tensor:
        """Reduce (terms, dx, terms, dy) -> (dx, dy)."""
        if self.p_norm == "max":
            return C.amax(dim=(0, 2))
        if self.p_norm == 2:
            return C.pow(2).mean(dim=0).mean(dim=1).sqrt()
        if self.p_norm == 1:
            return C.mean(dim=0).mean(dim=1)
        raise ValueError(f"Unsupported p_norm: {self.p_norm!r}")

    @staticmethod
    def _permute_columns(Z: torch.Tensor) -> torch.Tensor:
        n, d = Z.shape
        perm = torch.randn(n, d, device=Z.device).argsort(dim=0)
        return Z[perm, torch.arange(d, device=Z.device).expand(n, d)]

    def _permutation_p_values(
        self,
        observed: torch.Tensor,
        X: torch.Tensor,
        Y: Optional[torch.Tensor],
        num_tests: int,
    ) -> torch.Tensor:
        count = torch.zeros_like(observed)
        for _ in tqdm(range(num_tests), desc="Permutation tests"):
            X_perm = self._permute_columns(X)
            Y_perm = self._permute_columns(Y) if Y is not None else None
            null = self._core(X_perm, Y_perm)
            count += (null > observed).float()
        return count / num_tests