"""row-1 cache: per-leaf convergence score + OT-vs-tree ancestor discrepancy,
placed on the UMAP, plus the highest-convergence cell types.

reads the per-leaf convergence + OT discrepancy dataframes compute_scores.py
writes and attaches UMAP coords and labels from the annotation CSVs. both scores
are normalized by the leaf's local spacing (mean X_scvi distance to its SPACING_K
nearest same-stage cells; convergence, a variance, by its square): both grow with
how sparse a region is, and without this the two axes of row 2 mostly track that.

writes:
  numerical_results/umap_scores.pkl  (scored leaves: umap + raw and normalized scores)
  numerical_results/umap_bg.pkl      (subsampled all-cell UMAP backdrop)
  numerical_results/top_conv_types.csv (top-N convergence cell types + colours)
  numerical_results/celltype_conv_vs_ot.csv (per-type mean conv vs OT, dual-scored)

    python analysis/figures/convergence/compute_umap_scores.py
"""
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import style as S

BG_N = 120_000          # all-cell grey backdrop subsample


def _load_embedding():
    with h5py.File(S.H5AD, "r") as f:
        emb = f["obsm"][S.OBSM_KEY][:].astype(np.float32)
        raw = f["obs"][f["obs"].attrs.get("_index", "_index")][:]
        times = f["obs"]["time"][:]
    names = np.array([n.decode() if isinstance(n, bytes) else n for n in raw])
    return emb, times, {n: i for i, n in enumerate(names)}


def _index(points):
    """exact L2 faiss index over `points` (GPU, CPU fallback)."""
    import faiss
    index = faiss.IndexFlatL2(points.shape[1])
    if hasattr(faiss, "index_cpu_to_all_gpus"):
        try:
            index = faiss.index_cpu_to_all_gpus(index)
        except Exception:
            pass
    index.add(np.ascontiguousarray(points, dtype=np.float32))
    return index


def _nn_spacing(emb, times, rows, k):
    """per row, mean X_scvi distance to its k nearest other cells of the same
    stage (obs `time`), over every cell of that stage, scored or not."""
    out = np.full(len(rows), np.nan)
    for t in np.unique(times[rows]):
        q = np.flatnonzero(times[rows] == t)
        index = _index(emb[times == t])
        for s in range(0, len(q), 100_000):
            D, _ = index.search(np.ascontiguousarray(emb[rows[q[s:s + 100_000]]]), k + 1)
            out[q[s:s + 100_000]] = np.sqrt(np.maximum(D[:, 1:], 0)).mean(1)  # drop self
    return out


def _knn_smooth(sub_emb, vals, k):
    """each cell's value averaged over itself + its k nearest neighbours in
    X_scvi. denoises per-cell scores to the cell-type level without moving
    cells around."""
    sub_emb = np.ascontiguousarray(sub_emb, dtype=np.float32)
    index = _index(sub_emb)
    out = np.empty(len(vals), dtype=float)
    for s in range(0, len(vals), 100_000):
        _, I = index.search(sub_emb[s:s + 100_000], min(k + 1, len(vals)))
        out[s:s + 100_000] = vals[I].mean(1)   # self is included (dist 0)
    return out


def _add_smoothed(df, col, emb, name2idx):
    """smoothed copy of `col` (over cells with a value + an embedding)."""
    out = np.full(len(df), np.nan)
    m = df[col].notna() & df["cell"].isin(name2idx)
    rows = df.loc[m, "cell"].map(name2idx).to_numpy(int)
    out[m.to_numpy()] = _knn_smooth(emb[rows], df.loc[m, col].to_numpy(float),
                                    S.KNN_SMOOTH_K)
    return out


def conv_vs_ot(df):
    """per-cell-type mean convergence vs OT discrepancy, averaged over the leaves
    carrying BOTH scores (so a type's x and y summarise the same population).
    keeps every type with >= MIN_CELLS dual-scored leaves; the blood exclusion and
    corner labelling are display choices left to the plot."""
    both = df.dropna(subset=["cov_trace", "l2_distance"])
    g = (both.groupby("cell_type")
              .agg(conv=("conv_norm", "mean"), ot=("ot_norm", "mean"),
                   lineage=("lineage", "first"), n=("cov_trace", "size"))
              .reset_index())
    return g[g["n"] >= S.MIN_CELLS].sort_values("conv", ascending=False)


def main():
    sub2type, type2lineage, type2color = S.load_labels()
    subtype = pd.read_csv(S.SUBTYPES, index_col=0)["cell_subtype"]

    print("loading scores ...", flush=True)
    # convergence lives on time-T leaf cells; OT discrepancy on the model-assigned
    # ancestor cells -- different cell sets, so an outer merge (each panel drops
    # NaN for its own quantity).
    conv = pd.read_pickle(S.CONV_DF)[["cell", "cov_trace"]]
    ot = pd.read_pickle(S.OT_DF)[["cell", "l2_distance"]]
    df = conv.merge(ot, on="cell", how="outer")
    df["cell_type"] = df["cell"].map(subtype).map(sub2type)
    df["lineage"] = df["cell_type"].map(type2lineage)

    umap = pd.read_csv(S.UMAP_CSV, index_col=0)
    umap.columns = ["umap_x", "umap_y"]
    df = df.merge(umap, left_on="cell", right_index=True, how="inner")
    print(f"  {df['cov_trace'].notna().sum():,} convergence-scored + "
          f"{df['l2_distance'].notna().sum():,} OT-scored cells on the UMAP",
          flush=True)

    emb, times, name2idx = _load_embedding()
    print(f"local spacing (k={S.SPACING_K} same-stage cells) ...", flush=True)
    m = df["cell"].isin(name2idx)
    df["nn_spacing"] = np.nan
    df.loc[m, "nn_spacing"] = _nn_spacing(
        emb, times, df.loc[m, "cell"].map(name2idx).to_numpy(int), S.SPACING_K)
    df["ot_norm"] = df["l2_distance"] / df["nn_spacing"]
    df["conv_norm"] = df["cov_trace"] / df["nn_spacing"] ** 2

    # KNN-smooth each score (in X_scvi) so cell-type-level structure reads cleanly.
    print(f"KNN-smoothing (k={S.KNN_SMOOTH_K}) ...", flush=True)
    df["conv_smooth"] = _add_smoothed(df, "conv_norm", emb, name2idx)
    df["ot_smooth"] = _add_smoothed(df, "ot_norm", emb, name2idx)

    # all-cell backdrop (grey), subsampled for a light, fast scatter.
    bg = umap.sample(n=min(BG_N, len(umap)), random_state=int(S.SEED))

    # top convergence cell types: highest mean score among types with enough cells.
    ct = (df.groupby("cell_type")
            .agg(mean_score=("conv_norm", "mean"), n_cells=("conv_norm", "count"))
            .reset_index())
    ct = ct[ct["n_cells"] >= S.MIN_CELLS].sort_values("mean_score", ascending=False)
    top = ct.head(S.N_TOP_CONV_TYPES).copy()
    top["color"] = top["cell_type"].map(type2color).fillna(S.MUTED)

    df.to_pickle(os.path.join(S.NR, "umap_scores.pkl"))
    bg.to_pickle(os.path.join(S.NR, "umap_bg.pkl"))
    top.to_csv(os.path.join(S.NR, "top_conv_types.csv"), index=False)
    conv_vs_ot(df).to_csv(os.path.join(S.NR, "celltype_conv_vs_ot.csv"), index=False)
    print("top convergence types:\n" + top.to_string(index=False), flush=True)
    print(f"wrote umap_scores.pkl ({len(df):,}), umap_bg.pkl ({len(bg):,}), "
          f"top_conv_types.csv", flush=True)


if __name__ == "__main__":
    main()
