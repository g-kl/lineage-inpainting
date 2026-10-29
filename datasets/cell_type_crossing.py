"""Real-cell crossing-strength dataset.

A controlled test of lineage-aware coupling recovery, built from real
mouse-embryo scVI cells. It mirrors the synthetic crossing benchmark but swaps
the Gaussian fates for real cell types, and exposes the standard dataset
interface (``records`` / ``leaf_states`` / ``states`` / ``gt_node_states`` /
``x_dim`` / ``reference_features``) so the trainer, aligners and baselines
consume it unchanged.

Each tree is a single progenitor that splits into two lineages, each of which
produces two children -- the crossing is imposed at the mid -> child step::

    root (t_root)                       # hidden common ancestor (binds the mids)
      |-- mid_A (t_src)                 # hidden lineage ancestor A
      |     |-- anchor child  (type A)      observed
      |     +-- inner  child  (A2 or B1)    observed
      +-- mid_B (t_src)                 # hidden lineage ancestor B
            |-- anchor child  (type B)
            +-- inner  child  (B1 or A2)

Only the children are observed; the root and the two mids are hidden -- the
pooler must infer them. Two single-cell *marginal records* (at ``t_root`` and
``t_src``) supply the observed per-time marginals the aligner anchors to, and
the leaf-times the record sampler pairs against.

The imposed coupling
--------------------
Each mid has one *anchor* child of its own type and one *inner* child. A mid
*crosses* when its index ``i >= k``, where

    k = round((1 - alpha) * N)

is the number of mids (per lineage) that keep their own inner type; a crossed
mid takes the *other* lineage's inner type instead. ``k`` is the only
alpha-dependent quantity and the cell pools are fixed, so every time-marginal is
byte-identical across alpha -- only the joint (which mid each child attaches to)
changes. ``alpha=0`` => lineage / OT coupling; ``alpha=1`` => full inner swap.

Picking a working motif (mids ``A, B``; inner children ``A2, B1``) needs two
checks: the lineage grouping is OT-minimal (so OT fails as alpha rises) AND both
alpha=1 crossing pools land on the correct mid (so a hidden-ancestor model can
identify the crossing).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Optional, Sequence

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import torch

from datasets.tree_record import (
    TreeRecord,
    _extract_subtree,
    _leaf_counts,
    _node_depths,
    _snapshot_at,
)

logger = logging.getLogger(__name__)

_EPS = 1e-6


# --------------------------------------------------------------------------
#  Real-cell loading (only the needed (subtype, time) cells; no 52 GB X load)
# --------------------------------------------------------------------------


def _decode_categorical(node):
    cats = np.array([x.decode() if isinstance(x, bytes) else x
                     for x in node["categories"][:]])
    return cats, node["codes"][:]


def load_cell_pools(data_dir, states_filename, embedding_key, subtypes_by_time,
                    *, devmap_only=True):
    """``{(subtype, time): (n, dim) float32}`` via h5py; reads only obsm + obs."""
    import h5py

    needed_times = {float(t) for t in subtypes_by_time}
    with h5py.File(os.path.join(data_dir, states_filename), "r") as f:
        if devmap_only:
            cats, codes = _decode_categorical(f["obs/dataset"])
            keep = codes == int(np.where(cats == "devmap")[0][0])
        else:
            keep = np.ones(f["obs/time"].shape[0], dtype=bool)
        time = f["obs/time"][:].astype(np.float64)
        cats_st, codes_st = _decode_categorical(f["obs/cell_subtype"])
        sub = cats_st[codes_st]
        rows = np.where(keep & np.isin(time, list(needed_times)))[0]
        X = f["obsm/" + embedding_key][:][rows].astype(np.float32)
        sub, time = sub[rows], time[rows]

    return {(name, float(t)): X[(sub == name) & (time == float(t))]
            for t, names in subtypes_by_time.items() for name in names}


# --------------------------------------------------------------------------
#  Records from pre-built graphs (the same indexing as TreeRecord._build_index)
# --------------------------------------------------------------------------


def _record_from_trees(graphs, leaf_time, time_bins, max_size) -> TreeRecord:
    """Index a set of pre-built tree graphs into one :class:`TreeRecord`.

    Reuses the ``tree_record`` snapshot/extract primitives so the subtree index
    is shaped exactly like a record built from real treedata -- the trainer's
    record sampler then treats it like any other record.
    """
    rec = TreeRecord.__new__(TreeRecord)
    rec.time_bins = sorted(float(b) for b in time_bins)
    rec.max_size, rec.min_size = int(max_size), 1
    rec.time_tolerance = (
        min(b - a for a, b in zip(rec.time_bins, rec.time_bins[1:])) / 2.0
        if len(rec.time_bins) > 1 else 0.5
    )
    rec.subtree_index = defaultdict(list)
    rec._n_total = 0
    rec.trees = dict(graphs)
    rec.leaf_time = float(leaf_time)

    for name, tree in graphs.items():
        sizes, depths = _leaf_counts(tree), _node_depths(tree)
        for bin_time in rec.time_bins:
            for root in _snapshot_at(tree, bin_time, eps=_EPS):
                kids = list(tree.successors(root))
                size = 1 if not kids else sum(
                    sizes[c] for c in kids
                    if tree.nodes[c].get("time", float("inf")) > bin_time + _EPS)
                if not (rec.min_size <= size <= rec.max_size):
                    continue
                rec.subtree_index[bin_time].append(_extract_subtree(
                    tree, root, sizes, depths, name, bin_time=bin_time, eps=_EPS))
                rec._n_total += 1
    return rec


# --------------------------------------------------------------------------
#  Dataset
# --------------------------------------------------------------------------


class CellTypeCrossingDataset:
    """Real-marginal, imposed-coupling crossing benchmark.

    Parameters
    ----------
    times : ``(t_root, t_src, t_tgt)``; the crossing is imposed at ``t_src ->
        t_tgt``.
    root_type : subtype of the hidden common ancestor at ``t_root``.
    parent_types : 2 mid subtypes ``(A, B)`` at ``t_src``.
    child_types : ``((A, A2), (B1, B))`` at ``t_tgt``. The anchors ``A``/``B``
        repeat the mid types; ``A2``/``B1`` are the inner (swapped) pair -- pick
        them boundary-facing so the crossing stays recoverable.
    alpha : crossing strength in [0, 1].
    n_trees : number of trees (the coupling signal).
    n_marginal : observed cells per marginal record (marginals + anchors).
    seed : RNG seed; cell draws are alpha-independent, so a sweep over alpha
        shares cells and only the joint moves.
    """

    def __init__(
        self,
        data_dir: str = "data/lt",
        states_filename: str = "qiu24_integrated_hvgs_and_scvi.h5ad",
        embedding_key: str = "X_scvi",
        times: Sequence[float] = (8.5, 9.0, 9.5),
        root_type: str = "cranial neural crest (anterior)",
        parent_types: Sequence[str] = (
            "frontonasal mesenchyme", "hindbrain (r1)"),
        child_types: Sequence[Sequence[str]] = (
            ("frontonasal mesenchyme", "pharyngeal arch mesenchyme (p1-p2)"),
            ("hindbrain (r5-r8)", "hindbrain (r1)")),
        alpha: float = 0.0,
        n_trees: int = 600,
        n_marginal: int = 600,
        max_size: int = 128,
        seed: int = 0,
        devmap_only: bool = True,
        cell_pools: Optional[dict] = None,
    ):
        if len(times) != 3:
            raise ValueError("need exactly 3 timepoints (t_root, t_src, t_tgt)")
        if len(parent_types) != 2 or len(child_types) != 2 or any(len(p) != 2 for p in child_types):
            raise ValueError("need 2 parent types and 2 child-type pairs")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        t_root, t_src, t_tgt = (float(t) for t in times)
        pA, pB = parent_types
        (A, A2), (B1, B) = child_types        # anchors A, B ; inner pair A2, B1
        self.times = (t_root, t_src, t_tgt)
        self.root_type = root_type
        self.parent_types = (pA, pB)
        self.child_types = ((A, A2), (B1, B))
        self.alpha = float(alpha)
        self.t_src, self.t_tgt = t_src, t_tgt

        rng = np.random.default_rng(seed)
        if cell_pools is None:
            cell_pools = load_cell_pools(
                data_dir, states_filename, embedding_key,
                {t_root: [root_type], t_src: [pA, pB], t_tgt: [A, A2, B1, B]},
                devmap_only=devmap_only)
        self.x_dim = int(next(iter(cell_pools.values())).shape[1])

        def sample(subtype, t, n):
            pool = cell_pools[(subtype, float(t))]
            if pool.shape[0] == 0:
                raise ValueError(f"no cells for ({subtype!r}, {t})")
            return pool[rng.choice(pool.shape[0], n, replace=pool.shape[0] < n)]

        # --- observed/hidden node accumulators -> standard interface ---
        leaf_states: dict[str, torch.Tensor] = {}
        gt: dict[str, torch.Tensor] = {}
        obs: list[tuple] = []   # (name, cell, time, subtype) for observed cells

        def add(name, cell, t, subtype, *, observed):
            gt[name] = torch.as_tensor(cell, dtype=torch.float32)
            if observed:
                leaf_states[name] = gt[name]
                obs.append((name, cell, float(t), subtype))

        # --- the fixed (alpha-independent) cell bank, one cell per slot/tree ---
        N = n_trees
        root_cells = sample(root_type, t_root, N)
        mid_cells = {"mA": sample(pA, t_src, N), "mB": sample(pB, t_src, N)}
        anchor = {"mA": sample(A, t_tgt, N), "mB": sample(B, t_tgt, N)}
        innerA2, innerB1 = sample(A2, t_tgt, N), sample(B1, t_tgt, N)

        # mid i crosses iff i >= k. The inner banks are routed mA <-> mB by k, so
        # the same cells are reused for every alpha -- only their parent changes.
        k = round((1.0 - self.alpha) * N)
        mid_spec = {
            "mA": dict(mtype=pA, atype=A, own=(innerA2, A2), cross=(innerB1, B1)),
            "mB": dict(mtype=pB, atype=B, own=(innerB1, B1), cross=(innerA2, A2)),
        }

        # --- build the trees: root -> 2 mids -> (anchor child, inner child) ---
        leaf_graphs: dict[str, nx.DiGraph] = {}
        for i in range(N):
            g = nx.DiGraph()
            r = f"x{i}_r"
            g.add_node(r, time=t_root)
            add(r, root_cells[i], t_root, root_type, observed=False)
            for tag, s in mid_spec.items():
                mid = f"x{i}_{tag}"
                g.add_node(mid, time=t_src)
                g.add_edge(r, mid)
                add(mid, mid_cells[tag][i], t_src, s["mtype"], observed=False)
                inner_bank, inner_type = s["cross"] if i >= k else s["own"]
                children = [(anchor[tag][i], s["atype"]), (inner_bank[i], inner_type)]
                for c, (cell, ctype) in enumerate(children):
                    leaf = f"{mid}_l{c}"
                    g.add_node(leaf, time=t_tgt)
                    g.add_edge(mid, leaf)
                    add(leaf, cell, t_tgt, ctype, observed=True)
            leaf_graphs[f"x{i}"] = g

        # --- observed marginal records (singletons) at t_root and t_src ---
        def marginal_record(prefix, subtypes, t):
            graphs = {}
            for j in range(n_marginal):
                subtype = subtypes[j % len(subtypes)]
                name = f"{prefix}{j}"
                g = nx.DiGraph()
                g.add_node(name, time=t)
                add(name, sample(subtype, t, 1)[0], t, subtype, observed=True)
                graphs[name] = g
            return _record_from_trees(graphs, t, [t], max_size)

        records = {
            "leaf": _record_from_trees(leaf_graphs, t_tgt, self.times, max_size),
            "root": marginal_record("root", [root_type], t_root),
            "mid": marginal_record("mid", [pA, pB], t_src),
        }

        names, cells, time_col, sub_col = zip(*obs)
        states = ad.AnnData(
            X=np.stack(cells).astype(np.float32),
            obs=pd.DataFrame({"time": time_col, "cell_subtype": sub_col}, index=list(names)))

        self.records = records
        self.leaf_states = leaf_states
        self.gt_node_states = gt
        self.states = states
        self.reference_features = states.X
        self.trees = {n: g for rec in records.values() for n, g in rec.trees.items()}
        self.cell_pools = cell_pools

        logger.info(
            "CellTypeCrossingDataset(alpha=%.3f): %d trees, dim=%d, crossing %.2f->%.2f",
            self.alpha, N, self.x_dim, t_src, t_tgt)
