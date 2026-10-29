"""Waddington-OT and moslin baselines, via moscot.

``wot``    -- moscot.problems.time.TemporalProblem (Schiebinger et al. 2019): entropic OT
              between adjacent time marginals on expression alone. No lineage.
``moslin`` -- moscot.problems.time.LineageProblem (Lange et al. 2023): the fused problem
              whose quadratic term is the within-timepoint patristic distance. The cost is
              built with genot_fgw._minibatch_tree_distances, so moslin and GENOT-FGW see
              the same tree geometry -- GENOT-FGW is the amortized neural version, moslin
              the reference solver.

A plan is only defined on the cells it was solved over, so ``predict`` projects each query
cell onto its nearest solved cell at its own timepoint and samples a partner from that row
of the plan. Plans are keyed by (earlier, later): predicting forward pushes along the plan,
predicting backward pulls along its transpose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from synthetic_evals.baselines.genot_fgw import _minibatch_tree_distances

logger = logging.getLogger(__name__)


@dataclass
class MoscotBank:
    """Per-time expression plus the within-time patristic cost."""
    by_time: dict[float, np.ndarray]
    tree_cost: dict[float, np.ndarray]

    @classmethod
    def from_fgw_bank(cls, bank) -> "MoscotBank":
        by_time, tree_cost = {}, {}
        for t, d in bank.by_time.items():
            by_time[float(t)] = np.asarray(d["expr"], dtype=np.float32)
            tree_cost[float(t)] = _minibatch_tree_distances(
                d["cells"], d["clone_ids"], bank.oracles).astype(np.float64)
        return cls(by_time=by_time, tree_cost=tree_cost)

    def times(self) -> list[float]:
        return sorted(self.by_time)


def adjacent_pairs(times):
    times = sorted(times)
    return [(times[i], times[i + 1]) for i in range(len(times) - 1)]


def _subsample(bank, max_cells, seed):
    """Cap cells per timepoint; the plan is dense, so this bounds it at max_cells^2."""
    rng = np.random.default_rng(seed)
    expr, cost = {}, {}
    for t, X in bank.by_time.items():
        keep = (np.arange(len(X)) if len(X) <= max_cells
                else np.sort(rng.choice(len(X), max_cells, replace=False)))
        expr[t] = X[keep]
        cost[t] = bank.tree_cost[t][np.ix_(keep, keep)]
        if len(keep) < len(X):
            logger.info("[moscot] t=%s: capped %d -> %d cells", t, len(X), len(keep))
    return expr, cost


def _adata(expr, cost, t_early, t_late, *, lineage):
    """Both timepoints in one AnnData. moscot subsets obsp per timepoint, so only the
    two diagonal blocks of the cost matter."""
    import anndata as ad
    import pandas as pd

    x_e, x_l = expr[t_early], expr[t_late]
    n_e, n_l = len(x_e), len(x_l)
    X = np.vstack([x_e, x_l]).astype(np.float32)
    obs = pd.DataFrame(
        {"t": pd.Categorical(np.r_[np.full(n_e, t_early), np.full(n_l, t_late)])},
        index=[str(i) for i in range(n_e + n_l)],
    )
    A = ad.AnnData(X=X, obs=obs)
    A.obsm["X_emb"] = X
    if lineage:
        C = np.zeros((n_e + n_l, n_e + n_l), dtype=np.float64)
        C[:n_e, :n_e] = cost[t_early]
        C[n_e:, n_e:] = cost[t_late]
        A.obsp["cost_matrices"] = C
    return A


def train(bank: MoscotBank, *, dim: int, pairs=None, lineage: bool = False,
          epsilon: float = 1e-2, alpha: float = 0.5, max_cells: int = 2048,
          n_steps=None, seed: int = 0, **_ignored) -> dict:
    """Solve one coupling per time pair. ``n_steps`` is accepted for interface parity and
    unused (nothing is trained); ``seed`` only drives the cell cap."""
    from moscot.problems.time import LineageProblem, TemporalProblem

    if pairs is None:
        pairs = adjacent_pairs(bank.times())
    expr, cost = _subsample(bank, max_cells, seed)

    plans = {}
    for t_a, t_b in pairs:
        key = (t_a, t_b) if t_a <= t_b else (t_b, t_a)
        if key in plans:
            continue
        A = _adata(expr, cost, *key, lineage=lineage)
        if lineage:
            prob = LineageProblem(A).prepare(
                time_key="t", lineage_attr={},
                joint_attr={"attr": "obsm", "key": "X_emb"},
            ).solve(epsilon=epsilon, alpha=alpha)
        else:
            prob = TemporalProblem(A).prepare(
                time_key="t", joint_attr={"attr": "obsm", "key": "X_emb"},
            ).solve(epsilon=epsilon)
        sol = prob[key].solution
        plans[key] = np.asarray(sol.transport_matrix, dtype=np.float64)
        tag = "moslin" if lineage else "wot"
        if not sol.converged:
            logger.warning("[%s] solve %s did not converge (epsilon=%g%s)", tag, key,
                           epsilon, f", alpha={alpha}" if lineage else "")
        logger.info("[%s] solved %s -> plan %s", tag, key, plans[key].shape)
    return {"plans": plans, "by_time": expr, "dim": int(dim)}


def _sample_partner(plan, rows, rng):
    """One partner index per query row, drawn from that row of the plan."""
    out = np.empty(len(rows), dtype=np.int64)
    for k, r in enumerate(rows):
        w = plan[r]
        total = w.sum()
        out[k] = rng.choice(len(w), p=w / total) if total > 0 else rng.integers(len(w))
    return out


@torch.no_grad()
def predict(pkg, x_src, t_src, t_tgt, *, seed: int = 0, n_steps=None) -> torch.Tensor:
    """Transport each query cell across the solved plan; (M, dim) -> (M, dim)."""
    key = (t_src, t_tgt) if t_src <= t_tgt else (t_tgt, t_src)
    plan = pkg["plans"][key]
    if t_src > t_tgt:                       # pulling: rows must index the query's time
        plan = plan.T
    src_cells, tgt_cells = pkg["by_time"][t_src], pkg["by_time"][t_tgt]

    is_tensor = torch.is_tensor(x_src)
    x = np.asarray(x_src.detach().cpu().numpy() if is_tensor else x_src,
                   dtype=np.float32)
    _, rows = cKDTree(src_cells).query(x, k=1)
    out = tgt_cells[_sample_partner(plan, np.atleast_1d(rows), np.random.default_rng(seed))]
    out = torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))
    return out.to(x_src.device) if is_tensor else out
