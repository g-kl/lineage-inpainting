"""Real-cell fate-bias dataset: two motifs sharing an ambiguous child type B.

Built from real mouse-embryo scVI cells, reusing the crossing dataset's tree
topology and plumbing (``load_cell_pools`` / ``_record_from_trees``). There are
two independent motifs, one per tree -- a clone of a single progenitor type
whose two sister lineages divide asymmetrically:

    motif A                              motif C
    root (t_root)                        root (t_root)
      |-- prog (t_src)  type A             |-- prog (t_src)  type C
      |     |-- child  A  observed         |     |-- child  C  observed
      |     +-- child  A  observed         |     +-- child  C  observed
      +-- prog (t_src)  type A             +-- prog (t_src)  type C
            |-- child  B  observed               |-- child  B  observed
            +-- child  B  observed               +-- child  B  observed

Both progenitors in a tree are the same type (the tree's motif); exactly one of
the two sisters differentiates into the shared type B while the other keeps its
own type. Since B is produced by both motifs, a {B,B} pair's progenitor is
recoverable only from its cousin pair -- the sister lineage that kept its
identity names the clone.

Exposes the standard dataset interface (``records`` / ``leaf_states`` /
``states`` / ``gt_node_states`` / ``x_dim`` / ``reference_features``); node names
match ``CellTypeCrossingDataset`` (``x{i}_mA`` / ``x{i}_mB`` mids).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import torch

from datasets.cell_type_crossing import _record_from_trees, load_cell_pools

logger = logging.getLogger(__name__)


class CellTypeFateDataset:
    """Real-marginal, shared-child-type fate benchmark.

    Parameters
    ----------
    times : ``(t_root, t_src, t_tgt)``; fate split imposed at ``t_src -> t_tgt``.
    root_type : hidden common-ancestor subtype at ``t_root``.
    parent_types : the two motif progenitor subtypes ``(A, C)`` at ``t_src``. A
        tree uses one of them for *both* of its progenitors.
    shared_type : the ambiguous child subtype ``B`` at ``t_tgt`` (both motifs
        produce it).
    n_trees : number of trees; half take motif A and half motif C, and exactly
        one of a tree's two sister progenitors takes ``shared_type``.
    n_marginal : observed cells per marginal record (root + mid).
    seed : RNG seed for the cell draws.
    """

    def __init__(
        self,
        data_dir: str = "../data/embryos",
        states_filename: str = "qiu24_integrated_hvgs_and_scvi.h5ad",
        embedding_key: str = "X_scvi",
        times: Sequence[float] = (8.5, 9.0, 9.5),
        root_type: str = "cranial neural crest (anterior)",
        parent_types: Sequence[str] = ("frontonasal mesenchyme", "hindbrain (r1)"),
        shared_type: str = "pharyngeal arch mesenchyme (p1-p2)",
        n_trees: int = 600,
        n_marginal: int = 600,
        max_size: int = 128,
        seed: int = 0,
        devmap_only: bool = True,
        cell_pools: Optional[dict] = None,
    ):
        if len(times) != 3:
            raise ValueError("need exactly 3 timepoints (t_root, t_src, t_tgt)")
        if len(parent_types) != 2:
            raise ValueError("need exactly 2 progenitor types (A, C)")

        t_root, t_src, t_tgt = (float(t) for t in times)
        A, C = parent_types
        B = shared_type
        self.times = (t_root, t_src, t_tgt)
        self.root_type = root_type
        self.parent_types = (A, C)
        self.shared_type = B
        self.t_src, self.t_tgt = t_src, t_tgt

        rng = np.random.default_rng(seed)
        if cell_pools is None:
            cell_pools = load_cell_pools(
                data_dir, states_filename, embedding_key,
                {t_root: [root_type], t_src: [A, C], t_tgt: [A, C, B]},
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

        # One motif per tree: the first half are motif A (both progenitors type
        # A), the second half motif C. Within a tree exactly one sister
        # differentiates into B; alternate which one so neither sibling slot is
        # privileged.
        N, k = n_trees, n_trees // 2
        root_cells = sample(root_type, t_root, N)
        motif = [A] * k + [C] * (N - k)                     # progenitor type of tree i
        shared_tag = ["mA" if i % 2 == 0 else "mB" for i in range(N)]
        mid_cells = {
            tag: np.concatenate([sample(A, t_src, k), sample(C, t_src, N - k)])
            for tag in ("mA", "mB")
        }

        def child_type(tag, i):
            return B if tag == shared_tag[i] else motif[i]

        # --- build the trees: root -> 2 same-type progenitors -> 2 children ---
        leaf_graphs: dict[str, nx.DiGraph] = {}
        for i in range(N):
            g = nx.DiGraph()
            r = f"x{i}_r"
            g.add_node(r, time=t_root)
            add(r, root_cells[i], t_root, root_type, observed=False)
            for tag in ("mA", "mB"):
                mid = f"x{i}_{tag}"
                g.add_node(mid, time=t_src)
                g.add_edge(r, mid)
                add(mid, mid_cells[tag][i], t_src, motif[i], observed=False)
                ctype = child_type(tag, i)
                for c in range(2):                          # both children share ctype
                    leaf = f"{mid}_l{c}"
                    g.add_node(leaf, time=t_tgt)
                    g.add_edge(mid, leaf)
                    add(leaf, sample(ctype, t_tgt, 1)[0], t_tgt, ctype, observed=True)
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
            "mid": marginal_record("mid", [A, C], t_src),
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
        # Per-tree ground truth the eval needs and cannot read off the observed
        # cells: the tree's motif type, and which sister took the shared type.
        self.tree_motif = {f"x{i}": motif[i] for i in range(N)}
        self.tree_shared_tag = {f"x{i}": shared_tag[i] for i in range(N)}

        logger.info(
            "CellTypeFateDataset(A=%r, C=%r, B=%r): %d trees (%d motif-A, %d motif-C), "
            "dim=%d, fate %.2f->%.2f", A, C, B, N, k, N - k, self.x_dim, t_src, t_tgt)
