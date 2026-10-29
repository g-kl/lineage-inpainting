"""Build the commitment index (run once, from the repo root).

Does the slow, label-independent work: load the 9.5-leaf-time trees, count
descendant-leaf label histograms per node (all levels at once, one post-order
sweep per tree), and cache the expression of every assigned cell. Writes to
analysis/pipelines/commitment/index/. Query with commitment.py.
"""

import os
import pickle
import time

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.pipelines.commitment import commitment as C

t0 = time.perf_counter()


def tic(msg):
    print(f"[{time.perf_counter() - t0:6.1f}s] {msg}")


# leaf label codes per level: cell -> integer category index (-1 if unknown)
tic("loading labels")
ti = pd.read_csv(C.TYPE_INFO, index_col=0)
subtype = pd.read_csv(C.SUBTYPES, index_col=0)["cell_subtype"]
cols = {lv: sorted(ti[lv].dropna().unique()) for lv in C.LEVELS}
code = {}
for lv in C.LEVELS:
    idx = {name: i for i, name in enumerate(cols[lv])}
    cell_label = subtype.map(dict(zip(ti.index, ti[lv])))
    code[lv] = dict(zip(subtype.index, cell_label.map(idx).fillna(-1).astype(int)))

# column layout of the combined histogram: [total, level0 cats, level1 cats, ...]
widths = [1] + [len(cols[lv]) for lv in C.LEVELS]
offset = np.cumsum([0] + widths)  # offset[0]=0 (total), then one block per level
WIDTH = offset[-1]

tic("loading trees")
import treedata

td = treedata.read_h5td(C.TREES)
tree_names = [n for n in td.obst if n.startswith(f"E{C.LEAF_TIME}")]
tic(f"{len(tree_names)} trees")

tic("loading assignments")
with open(C.ASSIGN_PKL, "rb") as fh:
    assignments = pickle.load(fh)


def sweep(g):
    """Combined descendant histogram per node via one post-order sweep."""
    order = list(nx.topological_sort(g))
    pos = {n: i for i, n in enumerate(order)}
    H = np.zeros((len(order), WIDTH), dtype=np.int32)
    for n in reversed(order):
        i = pos[n]
        succ = list(g.successors(n))
        if not succ:  # leaf
            H[i, 0] = 1  # total
            for k, lv in enumerate(C.LEVELS):
                c = code[lv].get(n, -1)
                if c >= 0:
                    H[i, offset[k + 1] + c] += 1
        else:
            H[i] = H[[pos[s] for s in succ]].sum(0)
    return H, pos, order


def nearest_index_ancestor(g, order, member):
    """For each member node, the nearest strict ancestor that is also a member.

    Top-down pass carrying the last member seen above each node. `member` is a
    set of node names we keep in the index; returns {node: ancestor_name or ''}.
    """
    laa = {}          # last member ancestor propagated down to each node
    anc = {}
    for n in order:   # root first
        cur = laa.get(n, "")
        if n in member:
            anc[n] = cur
            child_val = n
        else:
            child_val = cur
        for c in g.successors(n):
            laa[c] = child_val
    return anc


tic("sweeping trees")
rows = []  # node metadata
hist_blocks = []  # combined histogram rows aligned to `rows`
for tname in tree_names:
    g = td.obst[tname]
    amap = assignments.get(tname, {})
    H, pos, order = sweep(g)
    keep = {}   # node -> (cell, time, bin) for members of the index
    for node, cell in amap.items():
        if cell == C.UNKNOWN or g.out_degree(node) == 0:
            continue
        bi = C.bin_of(float(g.nodes[node].get("time", np.nan)))
        if bi < 0:
            continue
        keep[node] = (cell, float(g.nodes[node]["time"]), bi)
    anc = nearest_index_ancestor(g, order, set(keep))
    for node, (cell, t, bi) in keep.items():
        rows.append((tname, node, cell, t, bi, anc[node]))
        hist_blocks.append(H[pos[node]])
    tic(f"  {tname}: {len(rows)} nodes so far")

H = np.vstack(hist_blocks)
nodes = pd.DataFrame(rows, columns=["tree", "node", "cell", "time", "bin", "anc"])
nodes["n_leaf"] = H[:, 0]

os.makedirs(C.INDEX_DIR, exist_ok=True)
nodes.to_parquet(os.path.join(C.INDEX_DIR, "nodes.parquet"))
for k, lv in enumerate(C.LEVELS):
    block = H[:, offset[k + 1]:offset[k + 2]]
    np.save(os.path.join(C.INDEX_DIR, f"hist_{lv}.npy"), block)
    np.save(os.path.join(C.INDEX_DIR, f"cols_{lv}.npy"),
            np.array(cols[lv], dtype=object))
tic(f"wrote {len(nodes)} nodes")

# cache expression for every unique assigned cell (one CSR row each)
tic("reading expression")
cells = sorted(set(nodes["cell"]))
with h5py.File(C.H5AD, "r") as f:
    obs = np.array([n.decode() for n in f["obs"]["_index"][:]])
    genes = np.array([g.decode() for g in f["var"]["_index"][:]])
    row_of = {n: i for i, n in enumerate(obs)}
    want = [(c, row_of[c]) for c in cells if c in row_of]
    X = f["X"]
    data, indices, indptr = X["data"], X["indices"], X["indptr"][:]
    ncol = int(X.attrs["shape"][1])
    d, ind, ptr = [], [], [0]
    for _, r in want:
        s, e = int(indptr[r]), int(indptr[r + 1])
        d.append(data[s:e]); ind.append(indices[s:e]); ptr.append(ptr[-1] + (e - s))
expr = sp.csr_matrix((np.concatenate(d), np.concatenate(ind), np.array(ptr)),
                     shape=(len(want), ncol))

sp.save_npz(os.path.join(C.INDEX_DIR, "expr.npz"), expr)
np.save(os.path.join(C.INDEX_DIR, "expr_cells.npy"),
        np.array([c for c, _ in want], dtype=object))
np.save(os.path.join(C.INDEX_DIR, "genes.npy"), np.array(genes, dtype=object))
tic(f"wrote expression {expr.shape}")
tic("done")
