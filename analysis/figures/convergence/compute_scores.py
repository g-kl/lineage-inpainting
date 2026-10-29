"""row-1 source data, computed self-contained across ALL timepoints.

light, self-contained implementations (h5py + treedata + faiss) so the scores cover
every tree timepoint, not just the E9.5 leaves.

convergence score (per time-T cell):
  KNN over the time-T cell embeddings, then trace(Cov(.)) of the T_anc = T -
  CONV_T_ANC_OFFSET ancestor embeddings pooled over the cell + its CONV_K
  neighbours. high = nearby present-day cells funnelled in from disparate
  precursors.

OT discrepancy (per time-T leaf):
  the GENOT-W2 assignments share the model's {tree: {node: cell}} node ids, so
  the discrepancy is the L2 (in X_scvi) between the two cells each map assigns
  to the leaf's ancestor node nearest T_anc -- no time discretisation. placed on
  the leaf.

writes numerical_results/convergence_df.pkl (cell, T, cov_trace) and
numerical_results/ot_discrepancy_df.pkl (cell, l2_distance).

    python analysis/figures/convergence/compute_scores.py
"""
import os
import pickle
import sys
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import style as S

UNKNOWN = "unknown"


def _knn_indices(points, k):
    """(n, k) indices of each row's k nearest OTHER rows (self excluded), L2."""
    import faiss
    points = np.ascontiguousarray(points, dtype=np.float32)
    n, d = points.shape
    kk = min(k + 1, n)
    index = faiss.IndexFlatL2(d)
    if hasattr(faiss, "index_cpu_to_all_gpus"):
        try:
            index = faiss.index_cpu_to_all_gpus(index)
        except Exception:
            pass
    index.add(points)
    _, idx = index.search(points, kk)
    out = np.full((n, kk - 1), -1, dtype=np.int64)
    for i in range(n):
        kept = idx[i][idx[i] != i]
        out[i, :min(len(kept), kk - 1)] = kept[:kk - 1]
    return out


def _find_t_ancestor(g, leaf, T):
    """most recent strict ancestor of leaf with node time <= T."""
    cur = leaf
    while True:
        preds = list(g.predecessors(cur))
        if not preds:
            return None
        parent = preds[0]
        pt = float(g.nodes[parent].get("time", np.nan))
        if not np.isnan(pt) and pt <= T:
            return parent
        cur = parent


def _nearest_ancestor(g, leaf, T, tol):
    """the strict ancestor whose node time is closest to T (within tol), or None."""
    chain, cur = [], leaf
    while True:
        preds = list(g.predecessors(cur))
        if not preds:
            break
        cur = preds[0]
        chain.append(cur)
    if not chain:
        return None
    ts = np.array([float(g.nodes[n].get("time", np.nan)) for n in chain])
    j = int(np.nanargmin(np.abs(ts - T)))
    return chain[j] if abs(ts[j] - T) <= tol else None


def _load_embedding():
    with h5py.File(S.H5AD, "r") as f:
        emb = f["obsm"][S.OBSM_KEY][:].astype(np.float32)
        idx_key = f["obs"].attrs.get("_index", "_index")
        raw = f["obs"][idx_key][:]
    names = np.array([n.decode() if isinstance(n, bytes) else n for n in raw])
    return emb, names, {n: i for i, n in enumerate(names)}


def compute_convergence(emb, names, name2idx, trees_by_T, assignments):
    """per-cell convergence, scored at every timepoint (mean over timepoints)."""
    records = []
    for T in sorted(trees_by_T):
        T_anc = T - S.CONV_T_ANC_OFFSET
        leaf_idx, anc_idx = [], []
        for tname, g in trees_by_T[T].items():
            amap = assignments.get(tname, {})
            for leaf in g.nodes():
                if g.out_degree(leaf) > 0:
                    continue
                li = name2idx.get(leaf)
                if li is None:
                    continue
                lt = float(g.nodes[leaf].get("time", np.nan))
                if np.isnan(lt) or abs(lt - T) > S.CONV_LEAF_TIME_TOL:
                    continue
                anc = _find_t_ancestor(g, leaf, T_anc)
                if anc is None:
                    continue
                at = float(g.nodes[anc].get("time", np.nan))
                if np.isnan(at) or at < T_anc - S.CONV_ANC_TIME_TOL:
                    continue
                acell = amap.get(anc)
                ai = name2idx.get(acell) if acell and acell != UNKNOWN else None
                if ai is None:
                    continue
                leaf_idx.append(li)
                anc_idx.append(ai)
        if not leaf_idx:
            continue
        leaf_idx = np.asarray(leaf_idx)
        anc_idx = np.asarray(anc_idx)
        nbrs = _knn_indices(emb[leaf_idx], S.CONV_K)
        for i in range(len(leaf_idx)):
            nb = nbrs[i][nbrs[i] >= 0]
            A = emb[np.concatenate(([anc_idx[i]], anc_idx[nb]))]
            conv = float(np.var(A, axis=0, ddof=1).sum()) if A.shape[0] >= 2 else np.nan
            records.append((names[leaf_idx[i]], T, conv))
        print(f"  T={T}: scored {len(leaf_idx):,} cells", flush=True)
    df = pd.DataFrame(records, columns=["cell", "T", "cov_trace"])
    return df.groupby("cell", as_index=False).agg(
        T=("T", "first"), cov_trace=("cov_trace", "mean"))


def compute_ot_discrepancy(emb, name2idx, trees_by_T, assignments, genot):
    """per time-T leaf, L2 in X_scvi between the model and GENOT cells assigned to
    its ancestor node nearest T_anc = T - CONV_T_ANC_OFFSET, placed on the LEAF (so
    the disagreement about a terminal cell's origin lights up that terminal fate)."""
    cells, mi, gi = [], [], []
    for T in sorted(trees_by_T):
        T_anc = T - S.CONV_T_ANC_OFFSET
        for tname, g in trees_by_T[T].items():
            amap, gmap = assignments.get(tname, {}), genot.get(tname, {})
            if not gmap:
                continue
            for leaf in g.nodes():
                if g.out_degree(leaf) > 0:
                    continue
                lt = float(g.nodes[leaf].get("time", np.nan))
                if np.isnan(lt) or abs(lt - T) > S.CONV_LEAF_TIME_TOL:
                    continue
                anc = _nearest_ancestor(g, leaf, T_anc, S.OT_ANC_TOL)
                if anc is None:
                    continue
                mc, gc = amap.get(anc), gmap.get(anc)
                if not mc or not gc or mc == UNKNOWN or gc == UNKNOWN:
                    continue
                a, b = name2idx.get(mc), name2idx.get(gc)
                if a is None or b is None:
                    continue
                cells.append(leaf); mi.append(a); gi.append(b)
    l2 = np.linalg.norm(emb[np.asarray(mi)] - emb[np.asarray(gi)], axis=1)
    df = pd.DataFrame({"cell": cells, "l2_distance": l2})
    return df.groupby("cell", as_index=False)["l2_distance"].mean()


def main():
    print("loading embedding (X_scvi) ...", flush=True)
    emb, names, name2idx = _load_embedding()
    print(f"  {emb.shape[0]:,} cells x {emb.shape[1]} dims", flush=True)

    print("loading trees + assignments ...", flush=True)
    import treedata
    td = treedata.read_h5td(S.TREES)
    trees_by_T = defaultdict(dict)
    for name, g in td.obst.items():
        T = float(name.split("-")[0].lstrip("E"))
        if T <= S.CONV_T_MAX + 1e-9:
            trees_by_T[T][name] = g
    with open(S.ASSIGN_PKL, "rb") as f:
        assignments = pickle.load(f)
    with open(S.GENOT_ASSIGN_PKL, "rb") as f:
        blob = pickle.load(f)
    # the pkl is either a bare {tree: {node: cell}} map or a model blob carrying it
    # under an "assignments" key.
    genot = blob["assignments"] if isinstance(blob, dict) and "assignments" in blob else blob

    print(f"convergence across {len(trees_by_T)} timepoints (k={S.CONV_K}) ...",
          flush=True)
    conv = compute_convergence(emb, names, name2idx, trees_by_T, assignments)
    conv.to_pickle(S.CONV_DF)
    print(f"  wrote {S.CONV_DF} ({len(conv):,} cells)", flush=True)

    print("OT discrepancy (direct node correspondence, placed on leaf) ...",
          flush=True)
    ot = compute_ot_discrepancy(emb, name2idx, trees_by_T, assignments, genot)
    ot.to_pickle(S.OT_DF)
    print(f"  wrote {S.OT_DF} ({len(ot):,} cells)", flush=True)


if __name__ == "__main__":
    main()
