"""Synthetic branching-SDE data generator.

Simulates branching stochastic differential equations where child dynamics
depend on parent parameters, enabling correlated lineages -- unlike the
standard "clone and independently simulate" approach.

Key abstractions
----------------
- **SDE**: drift + diffusion callables parameterised by a dict.
- **BranchModifier**: at each division, maps (parent_params, state,
  generation, child_index) -> child_params.
- **simulate()**: Euler-Maruyama forward simulation with fixed-interval
  branching, returning a *SimResult* that carries the full decorated tree
  and can produce snapshots / TreeRecords at arbitrary observation times.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# -- SDE -------------------------------------------------------------------


class SDE(ABC):
    """dx = drift(x, p) dt + diffusion(x, p) dW."""

    @abstractmethod
    def drift(self, x: np.ndarray, params: dict) -> np.ndarray: ...

    @abstractmethod
    def diffusion(self, x: np.ndarray, params: dict) -> np.ndarray: ...

    def step(self, x, params, dt, rng):
        """Single Euler-Maruyama step."""
        dW = rng.normal(size=x.shape) * np.sqrt(dt)
        return x + self.drift(x, params) * dt + self.diffusion(x, params) * dW


class OrnsteinUhlenbeck(SDE):
    """dx = theta*(mu - x) dt + sigma dW.  Set theta = 0 for pure Brownian motion."""

    def drift(self, x, params):
        return params.get("theta", 1.0) * (params.get("mu", 0.0) - x)

    def diffusion(self, x, params):
        # Multiply by ones_like(x) so array-valued sigma (one per cell)
        # broadcasts up to (M, dim).
        return params.get("sigma", 0.1) * np.ones_like(x)


# -- Branch modifier -------------------------------------------------------


class BranchModifier(ABC):
    """Maps (parent_params, state, generation, child_index) -> child_params."""

    @abstractmethod
    def modify(self, params: dict, state: np.ndarray,
               generation: int, child_index: int) -> dict: ...


class IdentityModifier(BranchModifier):
    """Children inherit parent params unchanged."""

    def modify(self, params, state, generation, child_index):
        return dict(params)


class HierarchicalOUModifier(BranchModifier):
    """Brownian motion early on, then hierarchical OU attractors.

    Parameters
    ----------
    split_gen : int
        Generation at which the first OU split occurs.
    gap : int
        Generations between first and second split.
    theta, sigma : float
        OU parameters used after the first split.
    attractors_1 : tuple[float, ...]
        mu values assigned to children at the first split (one per child).
    attractors_2 : dict[float, tuple[float, ...]]
        Maps each first-split mu to sub-mu values at the second split.
    """

    def __init__(
        self,
        split_gen: int = 5,
        gap: int = 3,
        theta: float = 2.0,
        sigma: float = 0.1,
        attractors_1: tuple[float, ...] = (-1.0, 1.0),
        attractors_2: dict | None = None,
    ):
        self.split_gen = split_gen
        self.second_split = split_gen + gap
        self.theta = theta
        self.sigma = sigma
        self.attractors_1 = attractors_1
        self.attractors_2 = attractors_2 or {
            -1.0: (-1.5, 0.5),
            1.0: (-0.5, 1.5),
        }

    def modify(self, params, state, generation, child_index):
        if generation == self.split_gen:
            mu = self.attractors_1[child_index]
            return {"theta": self.theta, "mu": mu, "sigma": self.sigma}
        if generation == self.second_split:
            cur_mu = params.get("mu", 0.0)
            key = min(self.attractors_2, key=lambda k: abs(k - cur_mu))
            new_mu = self.attractors_2[key][child_index]
            return {"theta": self.theta, "mu": new_mu, "sigma": self.sigma}
        return dict(params)


class FateBiasModifier(BranchModifier):
    """Two-stage fate bias: primed then committed.

    Before split_gen      : pure Brownian (theta=0, inherited unchanged).
    At split_gen          : children get weak OU attractors mu_1_primed /
                            mu_2_primed (child_index 0 / 1).
    At split_gen + n_primed_generations : cells commit to strong attractors
                            mu_1 / mu_2 based on which primed family they
                            belong to.

    The primed signal is too small to predict single-cell fate from state
    alone, but averaging over sisters reveals it; longer n_primed_generations
    gives more integration time and thus more predictive power.
    """

    def __init__(
        self,
        split_gen: int = 4,
        n_primed_generations: int = 3,
        theta: float = 2.0,
        sigma: float = 0.1,
        mu_1_primed: float = 0.1,
        mu_2_primed: float = -0.1,
        mu_1: float = 1.0,
        mu_2: float = -1.0,
        fate_prob: float = 0.5,
        seed: int = 0,
    ):
        self.split_gen = split_gen
        self.commit_gen = split_gen + n_primed_generations
        self.theta = theta
        self.sigma = sigma
        self.mu_1_primed = mu_1_primed
        self.mu_2_primed = mu_2_primed
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.fate_prob = fate_prob
        self._rng = np.random.default_rng(seed)

    def modify(self, params, state, generation, child_index):
        if generation == self.split_gen:
            goes_fate_1 = self._rng.random() < self.fate_prob
            mu = self.mu_1_primed if goes_fate_1 else self.mu_2_primed
            return {"theta": self.theta, "mu": mu, "sigma": self.sigma}
        if generation == self.commit_gen:
            cur_mu = params.get("mu", 0.0)
            if abs(cur_mu - self.mu_1_primed) <= abs(cur_mu - self.mu_2_primed):
                new_mu = self.mu_1
            else:
                new_mu = self.mu_2
            return {"theta": self.theta, "mu": new_mu, "sigma": self.sigma}
        return dict(params)


# -- Simulation result -----------------------------------------------------


@dataclass
class SimResult:
    """Full simulation output.

    Attributes
    ----------
    tree : nx.DiGraph
        Nodes carry *time*, *state*, *params*, *trajectory* (array of shape
        ``(n_steps+1, 1+dim)`` with columns ``[t, x_0, ..., x_{d-1}]``).
    obs_times : np.ndarray
        Default observation times.
    dim : int
        State dimensionality.
    """

    tree: nx.DiGraph
    obs_times: np.ndarray
    dim: int

    # -- snapshots --------------------------------------------------------

    def _alive_at(self, t_obs: float) -> list[str]:
        """Node ids of cells alive at *t_obs* (**pre-branching** convention).

        At a time that coincides exactly with a branch event the *parent*
        is returned -- its trajectory covers ``t_obs`` as an endpoint --
        not the just-born children.  This means the first observation time
        at which a modifier-induced split is *visible in states* is one
        interval **after** the branch at which the modifier fires.

        Concretely, with ``split_gen=5, branch_interval=1``:

        * ``t_obs = 5.0`` -> gen-4 parents alive; all share Brownian params.
        * ``t_obs = 6.0`` -> gen-5 cells alive; sisters carry different
          ``mu`` and have diverged over one full OU interval.
        """
        eps = 1e-12
        alive = []
        for n, d in self.tree.nodes(data=True):
            # Born at or after t_obs -> not yet alive under pre-branch rule.
            # (Using ``> t_obs - eps`` so that birth == t_obs is excluded.)
            if d["time"] > t_obs - eps:
                continue
            children = list(self.tree.successors(n))
            if not children:
                # Leaf (final-generation cell): alive from birth to end of
                # its trajectory.
                alive.append(n)
            elif all(
                self.tree.nodes[c]["time"] > t_obs - eps for c in children
            ):
                # All children born at or after t_obs -> node has not yet
                # divided (a division at exactly t_obs counts as pre-branch).
                alive.append(n)
        return alive

    def _interp_state(self, node: str, t_obs: float) -> np.ndarray:
        """Linearly interpolate the cell state at *t_obs* from its trajectory."""
        traj = self.tree.nodes[node]["trajectory"]
        ts = traj[:, 0]
        return np.array(
            [np.interp(t_obs, ts, traj[:, 1 + d]) for d in range(self.dim)]
        )

    def snapshot(self, t_obs: float):
        """Return ``(subtree, leaf_states)`` at observation time.

        *subtree* is an ``nx.DiGraph`` whose leaves are cells alive at
        *t_obs* (with ``time = t_obs``) and whose internal nodes are their
        ancestral branch points.  *leaf_states* maps node id -> state vector.
        """
        alive = self._alive_at(t_obs)
        if not alive:
            return None, None

        # collect ancestors
        keep = set()
        for node in alive:
            cur = node
            while cur is not None and cur not in keep:
                keep.add(cur)
                preds = list(self.tree.predecessors(cur))
                cur = preds[0] if preds else None

        sub = self.tree.subgraph(keep).copy()
        # prune non-trajectory attrs to keep snapshots lightweight
        for n in sub:
            sub.nodes[n].pop("trajectory", None)

        # Internal nodes: use *branching* time (= children's birth time)
        # instead of the node's own birth time.  All children are born at
        # the same instant, so any child's original birth time gives the
        # correct branching time.  This aligns internal-node times with
        # the phylogenetic convention (node time == split event time) and
        # ensures consistency with plot_trajectories and leaf_states.
        for n in sub.nodes():
            if sub.out_degree(n) > 0:          # internal node
                child = next(iter(sub.successors(n)))
                sub.nodes[n]["time"] = float(
                    self.tree.nodes[child]["time"]
                )

        leaf_states = {}
        for n in alive:
            sub.nodes[n]["time"] = float(t_obs)
            leaf_states[n] = self._interp_state(n, t_obs)
        return sub, leaf_states

    def to_tree_records(self, obs_times=None, **kw):
        """Build one :class:`TreeRecord` per observation time.

        Returns ``(records, leaf_states, states)`` where:

        * *records*: ``dict[float, TreeRecord]``
        * *leaf_states*: ``dict[str, torch.Tensor]``
          -- flat mapping from node id to its state tensor.  With default
          ``obs_times`` each node is alive at exactly one observation time,
          so there is no ambiguity.  If custom ``obs_times`` cause a node
          to appear at multiple snapshots, the last write wins.
        * *states*: :class:`anndata.AnnData` with one row per
          (leaf, obs-time) pair.  ``obs_names`` are globally unique
          (``"{node_id}@{T}"``).  Columns ``time``, ``mu``, ``theta``,
          ``sigma`` live in ``.obs``.
        """
        from datasets.tree_record import (TreeRecord, _extract_subtree,
                                          _leaf_counts, _node_depths, _all_roots)
        import torch

        obs_times = obs_times if obs_times is not None else self.obs_times
        max_size = kw.get("max_size", 4096)
        min_size = kw.get("min_size", 1)

        all_leaf_tensors: dict[str, "torch.Tensor"] = {}
        records: dict[float, TreeRecord] = {}

        # Accumulators for the combined AnnData
        _ad_keys: list[str] = []     # unique "{node}@{T}" identifiers
        _ad_X: list[np.ndarray] = []
        _ad_time: list[float] = []
        _ad_params: list[dict] = []

        for T in obs_times:
            T_f = float(T)
            snap, ls = self.snapshot(T_f)
            if snap is None:
                continue

            for node_id, state_vec in ls.items():
                all_leaf_tensors[node_id] = torch.tensor(
                    state_vec, dtype=torch.float32
                )

                _ad_keys.append(f"{node_id}@{T_f}")
                _ad_X.append(state_vec)
                _ad_time.append(T_f)
                _ad_params.append(
                    snap.nodes[node_id].get("params", {})
                    if node_id in snap.nodes else {}
                )

            # Bins are every distinct node time in the snapshot (internal
            # AND leaf). Including the leaf_time bin matches the real-data
            # convention used by TreeRecord._build_index, where each leaf
            # is indexed as a singleton subtree at the leaf-time bin --
            # without this, sample_time = leaf_time snaps to the previous
            # bin and the smaller record in a paired-offset sampler
            # contributes a degenerately 1-generation subtree.
            bin_times = sorted({
                round(snap.nodes[n].get("time", 0.0), 10)
                for n in snap.nodes()
            })
            if not bin_times:
                continue

            tr = TreeRecord.__new__(TreeRecord)
            tr.time_bins = bin_times
            tr.max_size = max_size
            tr.min_size = min_size
            tr.time_tolerance = (
                min(b - a for a, b in zip(tr.time_bins, tr.time_bins[1:])) / 2
                if len(tr.time_bins) > 1 else 0.5
            )
            tr.subtree_index = defaultdict(list)
            tr._n_total = 0
            tr.leaf_time = T_f

            sizes = _leaf_counts(snap)
            depths = _node_depths(snap)

            for rnode in _all_roots(snap, sizes, max_size):
                rtime = snap.nodes[rnode].get("time", None)
                if rtime is None:
                    continue
                nearest = min(tr.time_bins,
                              key=lambda b, rt=rtime: abs(b - rt))
                if abs(rtime - nearest) > tr.time_tolerance:
                    continue
                st = _extract_subtree(
                    snap, rnode, sizes, depths, "synthetic"
                )
                if st["size"] >= min_size:
                    tr.subtree_index[nearest].append(st)
                    tr._n_total += 1

            records[T_f] = tr

        if _ad_X:
            X = np.stack(_ad_X, axis=0).astype(np.float32)
            obs = {
                "time":  np.array(_ad_time, dtype=np.float64),
                "mu":    np.array([p.get("mu", np.nan) for p in _ad_params]),
                "theta": np.array([p.get("theta", np.nan) for p in _ad_params]),
                "sigma": np.array([p.get("sigma", np.nan) for p in _ad_params]),
            }
            states = ad.AnnData(X=X, obs=obs)
            states.obs_names = _ad_keys          # unique  "node@T"
        else:
            states = ad.AnnData()

        return records, all_leaf_tensors, states
    
    # -- visualisation -------------------------------------------------------

    def plot_trajectories(self, ax=None, alpha=0.3, lw=0.5, cmap="tab10", dpi=150,
                          all_black=False):
        """Plot every cell trajectory, coloured by terminal family.

        Colour is determined by the ``mu`` parameter of the leaf descendant,
        giving a visual grouping by attractor family.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4), dpi=dpi)

        # map each leaf's mu to a colour
        leaves = [n for n in self.tree if self.tree.out_degree(n) == 0]
        mus = sorted({self.tree.nodes[n]["params"].get("mu", 0.0)
                       for n in leaves})
        cm = plt.get_cmap(cmap)
        mu_color = {m: cm(i / max(len(mus) - 1, 1)) for i, m in enumerate(mus)}

        # walk to a leaf to inherit colour
        _cache: dict[str, tuple] = {}

        def _leaf_color(node):
            if node in _cache:
                return _cache[node]
            cur = node
            while self.tree.out_degree(cur) > 0:
                cur = next(iter(self.tree.successors(cur)))
            c = mu_color[self.tree.nodes[cur]["params"].get("mu", 0.0)]
            _cache[node] = c
            return c

        for n in self.tree.nodes():
            traj = self.tree.nodes[n].get("trajectory")
            if traj is None:
                continue
            c = _leaf_color(n)
            for d in range(self.dim):
                ax.plot(traj[:, 0], traj[:, 1 + d], color=c if not all_black else "black",
                        alpha=alpha, lw=lw)

        for t in self.obs_times:
            ax.axvline(t, color="grey", ls=":", lw=0.5, alpha=0.5)
        ax.set_xlabel("time")
        ax.set_ylabel("state")
        ax.set_title("branch-modifier SDE trajectories")
        return ax


# -- High-dimensional projection ------------------------------------------


class RandomProjector:
    """Embed the low-D latent state into R^D along a fixed random direction.

    ``y = x @ W + eps`` where ``W`` has unit-norm rows (for scalar latents,
    a single unit vector ``w in R^D``) and ``eps ~ N(0, (noise/sqrt(D))^2 I)``.

    Two normalizations keep the embedding's total variance -- and therefore
    the eval metrics -- comparable across ``dim_out``:

    * **unit-norm direction**: the signal energy ``||x @ W||^2 == ||x||^2`` is
      independent of D (a raw ``N(0, I)`` matrix would grow it ~linearly in D).
    * **1/sqrt(D) per-coordinate noise std**: the total noise energy
      ``||eps||^2 ~ noise^2`` is independent of D, i.e. we divide the noise we
      add by ``sqrt(dim_out)`` so the projected variance stays constant.

    A useful consequence for the OT baselines: the squared Euclidean cost
    ``||y_i - y_j||^2 == ||x_i - x_j||^2 + const`` differs from the latent cost
    only by a D-invariant constant offset, which cancels in entropic OT --
    so the baselines' fixed ``epsilon`` transfers across dimension unchanged.

    The direction is seeded by ``seed`` and the noise by ``seed + 1`` so a
    given ``(seed, dim_out)`` reproduces the same ``W`` in both the training
    dataset and the held-out eval simulation (they must share ``w``).
    """

    def __init__(self, dim_in: int, dim_out: int, seed: int, noise: float):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((dim_in, dim_out))
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        self.W = W
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.noise = float(noise)
        self.noise_std = float(noise) / np.sqrt(dim_out)
        self._noise_rng = np.random.default_rng(seed + 1)

    @property
    def w(self) -> np.ndarray:
        """The latent direction as a flat ``(dim_out,)`` vector (scalar latent)."""
        return self.W[0]

    def project(self, x: np.ndarray) -> np.ndarray:
        """Map a state array ``(..., dim_in)`` to ``(..., dim_out)`` (+ noise)."""
        x = np.asarray(x, dtype=np.float64)
        flat = x.reshape(-1, self.dim_in)
        y = flat @ self.W
        if self.noise_std > 0:
            y = y + self._noise_rng.normal(0.0, self.noise_std, size=y.shape)
        return y.reshape(*x.shape[:-1], self.dim_out)


class ProjectedSimResult:
    """View of a 1-D :class:`SimResult` with states embedded into R^D.

    Delegates the tree / lineage interface (``tree``, ``obs_times``,
    ``_alive_at``) unchanged -- only ``_interp_state`` is wrapped to return the
    projected, noised state, and ``dim`` reports the ambient dimension. Because
    the synthetic-eval driver touches the simulation solely through these
    attributes, wrapping the eval sim is enough to make the whole eval path
    high-dimensional. Trajectories stay 1-D (the dynamics plot is intrinsically
    the latent process).
    """

    def __init__(self, base: SimResult, projector: RandomProjector):
        self.base = base
        self.projector = projector

    @property
    def tree(self):
        return self.base.tree

    @property
    def obs_times(self):
        return self.base.obs_times

    @property
    def dim(self) -> int:
        return self.projector.dim_out

    def _alive_at(self, t_obs: float):
        return self.base._alive_at(t_obs)

    def _interp_state(self, node: str, t_obs: float) -> np.ndarray:
        return self.projector.project(self.base._interp_state(node, t_obs))


# -- Simulator -------------------------------------------------------------


def simulate(
    sde: SDE,
    modifier: BranchModifier,
    x0: np.ndarray,
    params0: dict,
    *,
    n_children: int = 2,
    n_generations: int = 10,
    branch_interval: float = 1.0,
    steps_per_interval: int = 100,
    obs_times: np.ndarray | None = None,
    seed: int = 0,
) -> SimResult:
    """Simulate a branching SDE with parameter inheritance.

    Each cell lives for one ``branch_interval``, then divides into
    ``n_children``. The ``modifier`` sets each child's SDE parameters
    based on the parent's parameters, final state, generation, and
    child index. Returns a :class:`SimResult` carrying the full
    decorated tree.
    """
    rng = np.random.default_rng(seed)
    dim = x0.shape[0]
    dt = branch_interval / steps_per_interval

    tree = nx.DiGraph()
    root_id = "0_0"
    tree.add_node(root_id, time=0.0, state=x0.copy(), params=dict(params0))
    active: list[tuple[str, np.ndarray, dict]] = [
        (root_id, x0.copy(), dict(params0)),
    ]

    def _run_interval_vec(active_list, t_start):
        """Vectorized Euler-Maruyama across all active cells in one go.

        Stacks states to (M, dim) and per-cell params to (M, 1) arrays so
        a single rng.normal + drift + diffusion call advances every cell.
        Writes each cell's trajectory back to its tree node.
        """
        M = len(active_list)
        node_ids = [a[0] for a in active_list]
        states = np.stack([a[1] for a in active_list], axis=0).astype(np.float64)

        param_keys = set().union(*(a[2].keys() for a in active_list))
        params_arr = {
            k: np.array(
                [a[2].get(k, 0.0) for a in active_list], dtype=np.float64,
            ).reshape(M, 1)
            for k in param_keys
        }

        # Allocate the full per-cell trajectory tensor up front and fill it
        # in-place; saves O(M * steps) Python-level concatenations.
        trajs = np.empty((M, steps_per_interval + 1, 1 + dim), dtype=np.float64)
        trajs[:, 0, 0] = t_start
        trajs[:, 0, 1:] = states
        sqrt_dt = np.sqrt(dt)

        for i in range(steps_per_interval):
            dW = rng.normal(size=states.shape) * sqrt_dt
            drift = sde.drift(states, params_arr)
            diff = sde.diffusion(states, params_arr)
            states = states + drift * dt + diff * dW
            trajs[:, i + 1, 0] = t_start + (i + 1) * dt
            trajs[:, i + 1, 1:] = states

        for j, nid in enumerate(node_ids):
            tree.nodes[nid]["trajectory"] = trajs[j]
        return states  # (M, dim)

    for gen in range(n_generations):
        t_start = gen * branch_interval
        t_branch = t_start + branch_interval
        finals = _run_interval_vec(active, t_start)
        next_active: list[tuple[str, np.ndarray, dict]] = []
        for (node_id, _x, params), final in zip(active, finals):
            for ci in range(n_children):
                cid = f"{gen + 1}_{len(next_active)}"
                cp = modifier.modify(params, final, gen + 1, ci)
                tree.add_node(cid, time=t_branch,
                              state=final.copy(), params=cp)
                tree.add_edge(node_id, cid)
                next_active.append((cid, final.copy(), cp))
        active = next_active

    # Leaves get a final non-branching segment so SDE state is defined
    # past their birth time (for observations after the last gen).
    t_start = n_generations * branch_interval
    _run_interval_vec(active, t_start)

    if obs_times is None:
        obs_times = np.arange(1, n_generations + 2) * branch_interval

    return SimResult(tree=tree, obs_times=np.asarray(obs_times), dim=dim)

def make_hierarchical_ou(
    n_generations: int = 10,
    branch_interval: float = 1.0,
    steps_per_interval: int = 200,
    sigma: float = 0.1,
    split_gen: int = 5,
    gap: int = 3,
    theta: float = 2.0,
    obs_times: np.ndarray | None = None,
    seed: int = 0,
    attractors_1: tuple[float, ...] | None = None,
    attractors_2: dict | None = None,
) -> SimResult:
    """Create the example hierarchical-OU branching process.

    Generations 0 .. split_gen-1 : pure Brownian motion (theta = 0).
    Generation split_gen         : children get OU(mu = -1) / OU(mu = +1)
                                   (configurable via *attractors_1*).
    Generation split_gen + gap   : each family splits again into the
                                   sub-attractors given by *attractors_2*.
                                   Default ("crossing"): -1 -> {-1.5, 0.5},
                                   +1 -> {-0.5, 1.5} (sisters cross the
                                   midline). A "non-crossing" alternative is
                                   -1 -> {-1.5, -0.5}, +1 -> {0.5, 1.5}.
    """
    mod_kwargs = dict(split_gen=split_gen, gap=gap, theta=theta, sigma=sigma)
    if attractors_1 is not None:
        mod_kwargs["attractors_1"] = tuple(attractors_1)
    if attractors_2 is not None:
        # Hydra may pass attractors_2 as a DictConfig with string keys; coerce
        # back to {float: tuple[float, ...]} which HierarchicalOUModifier expects.
        mod_kwargs["attractors_2"] = {
            float(k): tuple(float(x) for x in v) for k, v in dict(attractors_2).items()
        }
    return simulate(
        sde=OrnsteinUhlenbeck(),
        modifier=HierarchicalOUModifier(**mod_kwargs),
        x0=np.zeros(1),
        params0={"theta": 0.0, "mu": 0.0, "sigma": sigma},
        n_children=2,
        n_generations=n_generations,
        branch_interval=branch_interval,
        steps_per_interval=steps_per_interval,
        obs_times=obs_times,
        seed=seed,
    )

def make_fate_bias(
    n_generations: int = 12,
    branch_interval: float = 1.0,
    steps_per_interval: int = 200,
    sigma: float = 0.1,
    split_gen: int = 4,
    n_primed_generations: int = 3,
    theta: float = 2.0,
    mu_1_primed: float = 0.1,
    mu_2_primed: float = -0.1,
    mu_1: float = 1.0,
    mu_2: float = -1.0,
    fate_prob: float = 0.5,
    obs_times: np.ndarray | None = None,
    seed: int = 0,
) -> SimResult:
    """Create a two-stage fate-bias branching process.

    Generations 0 .. split_gen-1        : pure Brownian motion (theta=0).
    Generation split_gen                : primed split; sisters diverge to
                                          weak attractors mu_1_primed /
                                          mu_2_primed.
    Generation split_gen+n_primed_gens  : commitment; cells jump to strong
                                          attractors mu_1 / mu_2 according
                                          to their primed family.

    fate_prob controls the per-cell probability of being assigned to fate 1
    (mu_1 / mu_1_primed) at the split generation.  Each child draws
    independently, so the population fraction approaches fate_prob in
    expectation.  Default 0.5 gives an unbiased split.
    """
    return simulate(
        sde=OrnsteinUhlenbeck(),
        modifier=FateBiasModifier(
            split_gen=split_gen,
            n_primed_generations=n_primed_generations,
            theta=theta,
            sigma=sigma,
            mu_1_primed=mu_1_primed,
            mu_2_primed=mu_2_primed,
            mu_1=mu_1,
            mu_2=mu_2,
            fate_prob=fate_prob,
            seed=seed,
        ),
        x0=np.zeros(1),
        params0={"theta": 0.0, "mu": 0.0, "sigma": sigma},
        n_children=2,
        n_generations=n_generations,
        branch_interval=branch_interval,
        steps_per_interval=steps_per_interval,
        obs_times=obs_times,
        seed=seed,
    )


def make_multi_seed_records(
    n_trees: int = 40,
    obs_times=(5.0, 6.0, 7.0, 8.0),
    seed_base: int = 0,
    sim_factory=make_hierarchical_ou,
    **sim_kwargs,
) -> tuple[dict, dict, dict]:
    """Generate ``n_trees`` *independent* simulations, one observation time per
    simulation. Mirrors LT data: each "embryo" is its own tree observed at a
    single time point.

    Returns ``(records, leaf_states, gt_node_states)``:

    * ``records`` -- dict keyed by ``(tree_idx, obs_time)``; trainer iterates
      ``records.values()`` so keys are just unique identifiers.
    * ``leaf_states`` -- observed leaf states (input data).
    * ``gt_node_states`` -- true state at every node (leaves + internals)
      from the underlying SDE simulation. Internals are normally hidden but
      we expose them here for *held-out evaluation* -- pool/split predictions
      at internal positions are compared to these ground-truth values.

    Each tree's nodes are prefixed with ``"t{i}_"`` so they don't collide
    across simulations.

    Parameters
    ----------
    n_trees : number of independent simulations to generate.
    obs_times : pool of observation times; each tree is assigned one
        uniformly at random.
    seed_base : starting seed. Tree ``i`` uses ``seed_base + i``; obs_time
        assignment uses an RNG seeded by ``seed_base``.
    sim_factory, **sim_kwargs : passed through to the sim function;
        ``obs_times`` is overridden per-tree.
    """
    import networkx as nx
    sim_kwargs = {k: v for k, v in sim_kwargs.items() if k != "obs_times"}
    rng = np.random.default_rng(seed_base)
    obs_pool = list(obs_times)

    all_records: dict = {}
    all_leaf_states: dict = {}
    all_gt_states: dict = {}

    for i in range(n_trees):
        t = float(rng.choice(obs_pool))
        res = sim_factory(seed=seed_base + i, obs_times=np.array([t]), **sim_kwargs)
        recs, leaves, _ = res.to_tree_records()
        if t not in recs:
            continue
        rec = recs[t]
        prefix = f"t{i}_"

        for name, tensor in leaves.items():
            all_leaf_states[prefix + name] = tensor

        # Ground-truth state for every node in the tree (leaves + internals).
        # Each node's ``state`` attribute is its birth state from the SDE.
        import torch
        for node, attrs in res.tree.nodes(data=True):
            if "state" in attrs:
                all_gt_states[prefix + node] = torch.tensor(
                    attrs["state"], dtype=torch.float32,
                )

        new_subtree_index = defaultdict(list)
        for time_bin, subs in rec.subtree_index.items():
            for st in subs:
                new_st = dict(st)
                new_st["graph"] = nx.relabel_nodes(
                    st["graph"], lambda n: prefix + n, copy=True,
                )
                new_st["root"] = prefix + st["root"]
                new_subtree_index[time_bin].append(new_st)
        rec.subtree_index = new_subtree_index
        rec._n_total = sum(len(v) for v in rec.subtree_index.values())
        if hasattr(rec, "trees") and rec.trees:
            rec.trees = {
                prefix + k: nx.relabel_nodes(v, lambda n: prefix + n, copy=True)
                for k, v in rec.trees.items()
            }
        all_records[(i, t)] = rec

    return all_records, all_leaf_states, all_gt_states

