#!/usr/bin/env python
#SBATCH --job-name=reannotation
#SBATCH --output=logs/reannotation_%j.log
#SBATCH --error=logs/reannotation_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=bates

"""Reannotate cells by KNN label transfer with per-level uncertainty.

The reference set (default: devmap cells with valid subtype labels) is used
to build a FAISS GPU brute-force L2 index over scVI embeddings. For each
query cell (every cell; reference cells skip their self-match) we find the
k nearest reference cells and take a majority vote at four hierarchy levels:
cell_subtype, cell_type, lineage, germ_layer.

A level is "confident" when its top-vote fraction is at least --threshold.
The consensus label is taken at the finest confident level; if no level
clears the threshold the cell is labelled "ambiguous".

Outputs a CSV indexed by cell name with the per-level winning label,
per-level winning fraction, the consensus label, the consensus level, and
the source dataset. Per-dataset breakdown of consensus levels is logged.

Also writes <out_csv stem>_summary.csv: one row per (true) cell_subtype with
the fraction of its cells called ambiguous (sorted descending) and the cell
type it is most often confused with.

Run from the repo root, e.g.:
    python preprocessing/reannotation.py --out_csv out/reannotation.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(
    0,
    os.environ.get(
        "SLURM_SUBMIT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
    ),
)

import numpy as np
import pandas as pd
from scipy.stats import mode

import anndata as ad  # noqa: E402

logger = logging.getLogger(__name__)

LEVELS = ["cell_subtype", "cell_type", "lineage", "germ_layer"]


def knn_gpu(query: np.ndarray, ref: np.ndarray, k: int, batch_size: int = 200_000):
    """FAISS-GPU brute-force L2 k-NN. Returns (indices, distances) of shape (n_query, k)."""
    import faiss

    query = np.ascontiguousarray(query, dtype=np.float32)
    ref = np.ascontiguousarray(ref, dtype=np.float32)

    index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(ref.shape[1]))
    index.add(ref)

    n = len(query)
    idx_out = np.empty((n, k), dtype=np.int64)
    dist_out = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        d, i = index.search(query[s:e], k)
        idx_out[s:e] = i
        dist_out[s:e] = d
        logger.info("knn batch %d / %d", e, n)

    if (idx_out < 0).any() or (idx_out >= len(ref)).any():
        raise RuntimeError("faiss returned out-of-range indices; check GPU init")
    return idx_out, dist_out


def per_level_vote(nbr_idx: np.ndarray, ref_labels: np.ndarray):
    """For one label level, return (winning_label, winning_count) per query.

    nbr_idx: (n_query, k) int neighbor indices into the reference.
    ref_labels: (n_ref,) array of labels.
    """
    codes, uniques = pd.factorize(ref_labels)
    nbr_codes = codes[nbr_idx]  # (n_query, k)
    result = mode(nbr_codes, axis=1, keepdims=False)
    return uniques[result.mode], result.count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/embryos")
    parser.add_argument("--states_filename", default="qiu24_integrated_hvgs_and_scvi.h5ad")
    parser.add_argument("--annotation_filename", default="final_subtype.csv")
    parser.add_argument("--type_info_filename", default="final_type_info.csv")
    parser.add_argument("--embedding_key", default="X_scvi")
    parser.add_argument("--reference_dataset", default="devmap",
                        help="value of obs['dataset'] used as reference")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="min winning fraction required to call a level confident")
    parser.add_argument("--batch_size", type=int, default=200_000)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_summary_csv", default=None,
                        help="per-subtype ambiguity/confusion summary "
                             "(default: <out_csv stem>_summary.csv)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    logger.info("loading states from %s", args.states_filename)
    states = ad.read_h5ad(os.path.join(args.data_dir, args.states_filename))
    cell_types_df = pd.read_csv(
        os.path.join(args.data_dir, args.annotation_filename), index_col=0,
    )
    type_info_df = pd.read_csv(
        os.path.join(args.data_dir, args.type_info_filename), index_col=0,
    )

    # Per-cell label table at all four levels (NaN where unknown).
    labels = pd.DataFrame(index=states.obs_names)
    labels["cell_subtype"] = cell_types_df["cell_subtype"]
    labels["cell_type"] = labels["cell_subtype"].map(type_info_df["cell_type"])
    labels["lineage"] = labels["cell_subtype"].map(type_info_df["lineage"])
    labels["germ_layer"] = labels["cell_subtype"].map(type_info_df["germ_layer"])

    is_ref_dataset = (states.obs["dataset"] == args.reference_dataset).values
    has_all_labels = labels[LEVELS].notna().all(axis=1).values
    is_ref = is_ref_dataset & has_all_labels
    logger.info("reference: %d cells; total queried: %d cells",
                is_ref.sum(), len(states))

    embeddings = np.asarray(states.obsm[args.embedding_key], dtype=np.float32)
    ref_emb = embeddings[is_ref]
    ref_labels = labels.loc[is_ref].reset_index(drop=True)
    query_names = states.obs_names

    # Every cell is a query. For cells that are themselves in the reference
    # we search k+1 neighbors and drop the self-match, otherwise we drop the
    # last (extra) neighbor. self_pos maps each cell to its row in ref_emb,
    # or -1 if it is not in the reference.
    self_pos = np.full(len(states), -1, dtype=np.int64)
    self_pos[is_ref] = np.arange(int(is_ref.sum()))

    logger.info("running FAISS GPU knn (k=%d, dim=%d)", args.k, ref_emb.shape[1])
    nbr_kp1, _ = knn_gpu(embeddings, ref_emb, args.k + 1, batch_size=args.batch_size)

    # Per-row column to drop: the column matching self_pos if present, else the last.
    drop_col = (nbr_kp1 == self_pos[:, None]).argmax(axis=1)
    drop_col = np.where(self_pos >= 0, drop_col, args.k)
    src_cols = np.arange(args.k)[None, :] + (
        np.arange(args.k)[None, :] >= drop_col[:, None]
    ).astype(np.int64)
    nbr_idx = np.take_along_axis(nbr_kp1, src_cols, axis=1)
    del nbr_kp1, src_cols

    out = pd.DataFrame(index=query_names)
    out.index.name = "cell"
    for level in LEVELS:
        logger.info("voting at level %s", level)
        winner, count = per_level_vote(nbr_idx, ref_labels[level].values)
        out[level] = winner
        out[f"{level}_frac"] = count / args.k

    # Walk finest to coarsest; first level clearing the threshold wins.
    consensus_label = np.full(len(query_names), "ambiguous", dtype=object)
    consensus_level = np.full(len(query_names), "ambiguous", dtype=object)
    for level in LEVELS:
        unset = consensus_level == "ambiguous"
        confident = out[f"{level}_frac"].values >= args.threshold
        pick = unset & confident
        consensus_label[pick] = out[level].values[pick]
        consensus_level[pick] = level
    out["consensus_label"] = consensus_label
    out["consensus_level"] = consensus_level
    out["dataset"] = states.obs["dataset"].values
    out["true_cell_subtype"] = labels["cell_subtype"].values

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    out.to_csv(args.out_csv)
    logger.info("wrote %s (%d rows)", args.out_csv, len(out))

    logger.info("consensus level breakdown:")
    for ds, sub in out.groupby("dataset", observed=True):
        logger.info("  dataset=%s  n=%d", ds, len(sub))
        counts = sub["consensus_level"].value_counts()
        for lvl in LEVELS + ["ambiguous"]:
            n = int(counts.get(lvl, 0))
            logger.info("    %-13s %8d  (%5.1f%%)", lvl, n, 100 * n / len(sub))

    # Per-subtype summary: how often each (true) subtype comes out ambiguous,
    # and the cell type it is most often "confused" with. Confusion is measured
    # over cells whose voted cell_type disagrees with the subtype's true
    # cell_type; the most common such predicted cell_type is reported.
    summary_csv = args.out_summary_csv
    if summary_csv is None:
        stem, ext = os.path.splitext(args.out_csv)
        summary_csv = f"{stem}_summary{ext or '.csv'}"

    rows = []
    for subtype, sub in out.dropna(subset=["true_cell_subtype"]).groupby(
        "true_cell_subtype", observed=True
    ):
        n = len(sub)
        frac_ambiguous = (sub["consensus_level"] == "ambiguous").mean()
        true_cell_type = type_info_df["cell_type"].get(subtype, np.nan)
        confused = sub.loc[sub["cell_type"] != true_cell_type, "cell_type"]
        if len(confused):
            top = confused.value_counts()
            confused_cell_type = top.index[0]
            confused_frac = top.iloc[0] / n
        else:
            confused_cell_type = np.nan
            confused_frac = 0.0
        rows.append({
            "cell_subtype": subtype,
            "true_cell_type": true_cell_type,
            "n_cells": n,
            "frac_ambiguous": frac_ambiguous,
            "top_confused_cell_type": confused_cell_type,
            "top_confused_frac": confused_frac,
        })

    summary = pd.DataFrame(rows).sort_values(
        "frac_ambiguous", ascending=False
    ).reset_index(drop=True)
    summary.to_csv(summary_csv, index=False)
    logger.info("wrote %s (%d subtypes)", summary_csv, len(summary))


if __name__ == "__main__":
    main()
