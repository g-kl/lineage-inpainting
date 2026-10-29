"""geometric projection onto the observed cross-section.

a perturbation is a move in expression space followed by a snap onto the nearest
real cell that already sits at the desired expression level -- no decoder, no
scVI.  three modes, all driven by :func:`project_frontier`:

  control  (knockdown=False)   snap to the nearest real cell; nothing moves.
  knockout (zero_index given)  project every expressing cell onto its nearest
                               gene-off cell.
  dose     (bin_index given)   move each cell to ``old_expression * target_fraction``
                               and project onto the nearest cell in that
                               expression bin (quantile bins; bin 0 = gene-off).
                               ``target_fraction`` spans the whole range: <1
                               knockdown, >1 up-dose, and ``inf`` the
                               overexpression limit -- every cell (including
                               gene-off cells, which no finite multiplier can
                               move) is forced onto the top quantile bin.

projection distance (``proj_dist``) and local spacing (``local_dist``, distance to
the nearest *other* real cell) are recorded per cell so an off-manifold label can
be chosen after the run: a cell is off-manifold iff it changed and projected
farther than ``ratio * local_dist`` (see :func:`perturb_result.off_manifold_descendants`).
"""

import faiss
import numpy as np
import scipy.sparse as sp


def build_faiss_index(embeddings):
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index


def build_time_indices(embeddings, names, times, expression):
    """per-timepoint faiss index + cached metadata.

    ``embeddings`` (n, d), ``names`` (n,), ``times`` (n,) and ``expression``
    (n, n_genes) are all row-aligned to the same set of real cells.  returns a
    dict ``time -> {index, embeddings, names, expression}``.
    """
    times = np.asarray(times, dtype=float)
    out = {}
    for t in np.unique(times):
        m = times == t
        out[float(t)] = {
            "index": build_faiss_index(embeddings[m]),
            "embeddings": np.ascontiguousarray(embeddings[m], dtype=np.float32),
            "names": np.asarray(names)[m],
            "expression": expression[m],
        }
    return out


def snap_to_nearest(state, ti):
    """nearest real cell to a single state -> (idx, dist, name, embedding)."""
    q = np.ascontiguousarray(np.asarray(state).reshape(1, -1), dtype=np.float32)
    sq, idx = ti["index"].search(q, 1)
    i = idx[0, 0]
    return i, float(np.sqrt(max(sq[0, 0], 0.0))), ti["names"][i], ti["embeddings"][i]


def snap_to_nearest_batch(states, ti):
    """nearest real cells to a batch -> (idxs, dists, names, embeddings)."""
    q = np.ascontiguousarray(states, dtype=np.float32)
    sq, idxs = ti["index"].search(q, 1)
    i = idxs[:, 0]
    return i, np.sqrt(np.clip(sq[:, 0], 0.0, None)), ti["names"][i], ti["embeddings"][i]


def get_expression_batch(idxs, ti):
    """dense (n, n_genes) expression for a batch of cell indices."""
    e = ti["expression"][idxs]
    return np.asarray(e.toarray() if sp.issparse(e) else e, dtype=np.float32)


def _gene_column(ti, gene_idx):
    col = ti["expression"][:, gene_idx]
    return col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()


# ---------------------------------------------------------------------------
# expression bins (quantile) for knockout / dose projection
# ---------------------------------------------------------------------------

def assign_bin(values, edges):
    """bin each value: 0 = gene-off, 1..n_bins = the nonzero quantile bins."""
    values = np.asarray(values)
    b = np.searchsorted(edges[1:-1], values, side="right") + 1
    b[values <= 0] = 0
    return b


def build_zero_indices(time_indices, gene_idx):
    """per-timepoint faiss index over the real cells that do not express the gene."""
    out = {}
    for t, ti in time_indices.items():
        m = _gene_column(ti, gene_idx) == 0
        out[t] = {"index": build_faiss_index(ti["embeddings"][m]),
                  "embeddings": np.ascontiguousarray(ti["embeddings"][m], dtype=np.float32),
                  "names": ti["names"][m]}
    return out


def build_bin_indices(time_indices, gene_idx, n_bins=4):
    """per-timepoint expression bins (bin 0 = gene-off, 1..n_bins = nonzero
    quantiles), each with its own faiss index, for dose-response projection."""
    out = {}
    for t, ti in time_indices.items():
        col = _gene_column(ti, gene_idx)
        nz = col[col > 0]
        edges = (np.quantile(nz, np.linspace(0, 1, n_bins + 1))
                 if len(nz) else np.zeros(n_bins + 1))
        labels = assign_bin(col, edges)
        bins = [{"index": build_faiss_index(ti["embeddings"][labels == b]),
                 "embeddings": np.ascontiguousarray(ti["embeddings"][labels == b], dtype=np.float32),
                 "names": ti["names"][labels == b]}
                for b in range(n_bins + 1)]
        out[t] = {"edges": edges, "bins": bins,
                  "nonempty": [b for b in range(n_bins + 1) if bins[b]["names"].size]}
    return out


def ratio_threshold_diagnostics(time_indices, ratio, n_sample=64, rng=None):
    """print, per timepoint, the mean distance between random cells against the
    mean off-manifold threshold (ratio x nearest-other-cell spacing) -- a sanity
    check for choosing ``ratio``."""
    rng = rng or np.random.default_rng()
    for t, ti in time_indices.items():
        emb = ti["embeddings"]
        n = len(emb)
        if n < 2:
            continue
        s = emb[rng.choice(n, min(n_sample, n), replace=False)]
        pair = np.linalg.norm(s[:, None] - s[None], axis=-1)
        rand_d = float(pair[np.triu_indices(len(s), 1)].mean())
        sq, _ = ti["index"].search(s, 2)
        local = float(np.sqrt(np.clip(sq[:, 1], 0.0, None)).mean())
        print(f"  t={t}: mean random-pair dist={rand_d:.3f}  "
              f"mean threshold (x{ratio})={ratio * local:.3f}")


# ---------------------------------------------------------------------------
# the projection step
# ---------------------------------------------------------------------------

def project_frontier(states, ti, gene_idx, knockdown=True, zero_index=None,
                     bin_index=None, target_fraction=0.0):
    """project a batch of frontier states onto the desired expression level.

    returns dict with arrays (length n):
      kd_state   (n, d)  the projected (perturbed) embedding
      changed    (n,)    whether the cell moved expression bin / value
      proj_dist  (n,)    distance from the state to the projected cell
      local_dist (n,)    distance to the nearest *other* real cell (nan for control)
      snapped    (n,)    name of the real cell projected onto
    """
    states = np.ascontiguousarray(states, dtype=np.float32)
    n = len(states)

    if not knockdown:
        _, dist, names, embs = snap_to_nearest_batch(states, ti)
        return {"kd_state": np.array(embs, np.float32, copy=True),
                "changed": np.zeros(n, bool), "proj_dist": dist,
                "local_dist": np.full(n, np.nan), "snapped": names}

    loc = np.sqrt(np.clip(ti["index"].search(states, 2)[0][:, 1], 0.0, None))

    if bin_index is None:               # full knockout: project onto gene-off set
        idxs, _, _, _ = snap_to_nearest_batch(states, ti)
        changed = get_expression_batch(idxs, ti)[:, gene_idx] != 0
        _, proj, names, embs = snap_to_nearest_batch(states, zero_index)
        return {"kd_state": embs, "changed": changed, "proj_dist": proj,
                "local_dist": loc, "snapped": names}

    # dose: move to old_expression * target_fraction, project onto that bin.
    # target_fraction = inf is the overexpression limit: force every cell --
    # including gene-off cells, which 0 * inf can't lift -- onto the top bin.
    idxs, _, _, _ = snap_to_nearest_batch(states, ti)
    e_old = get_expression_batch(idxs, ti)[:, gene_idx]
    cur_bin = assign_bin(e_old, bin_index["edges"])
    if np.isinf(target_fraction):
        new_bin = np.full(n, len(bin_index["bins"]) - 1, dtype=cur_bin.dtype)
    else:
        # counts are integers: round the dosed target back to count space before
        # binning, else any fraction <1 drops an integer below its own (tied)
        # quantile edge and knocks every cell down a bin
        new_bin = assign_bin(np.round(e_old * target_fraction), bin_index["edges"])
    changed = new_bin != cur_bin
    embs = np.empty_like(states)
    names = np.empty(n, dtype=object)
    proj = np.empty(n, dtype=np.float32)
    for b in np.unique(new_bin):
        sel = np.where(new_bin == b)[0]
        tb = b if bin_index["bins"][b]["names"].size else min(
            bin_index["nonempty"], key=lambda x: abs(x - b))
        sub = bin_index["bins"][tb]
        sq, j = sub["index"].search(states[sel], 1)
        embs[sel] = sub["embeddings"][j[:, 0]]
        names[sel] = sub["names"][j[:, 0]]
        proj[sel] = np.sqrt(np.clip(sq[:, 0], 0.0, None))
    return {"kd_state": embs, "changed": changed, "proj_dist": proj,
            "local_dist": loc, "snapped": names}
