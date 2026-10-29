import numpy as np
from abc import ABC, abstractmethod

class OTSolver(ABC):
    """Strategy interface for an optimal-transport solver.

    Subclass and implement :meth:`solve` to plug in a new algorithm.
    """

    @abstractmethod
    def solve(
        self,
        source: np.ndarray,
        target: np.ndarray,
        a: np.ndarray | None = None,
        b: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute a transport plan between two point clouds.

        Parameters
        ----------
        source : (n, d) source embeddings.
        target : (m, d) target embeddings.
        a : (n,) source marginal weights (uniform when *None*).
        b : (m,) target marginal weights (uniform when *None*).

        Returns
        -------
        coupling : (n, m) transport plan.
        """
        ...


class SinkhornSolver(OTSolver):
    """Entropic-regularised OT"""

    def __init__(
        self,
        epsilon: float = 0.1,
        max_iter: int = 1_000,
        metric: str = "sqeuclidean",
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.metric = metric

    def solve(self, source, target, a=None, b=None):
        import ot as pot

        C = pot.dist(source, target, metric=self.metric)
        C /= C.max() + 1e-8
        n, m = C.shape
        a = np.ones(n, dtype=np.float64) / n if a is None else a
        b = np.ones(m, dtype=np.float64) / m if b is None else b
        return pot.sinkhorn(a, b, C, self.epsilon, numItermax=self.max_iter)


class ExactOTSolver(OTSolver):
    """Exact (LP) optimal transport."""

    def __init__(self, metric: str = "sqeuclidean"):
        self.metric = metric

    def solve(self, source, target, a=None, b=None):
        import ot as pot

        C = pot.dist(source, target, metric=self.metric)
        n, m = C.shape
        a = np.ones(n, dtype=np.float64) / n if a is None else a
        b = np.ones(m, dtype=np.float64) / m if b is None else b
        return pot.emd(a, b, C)
