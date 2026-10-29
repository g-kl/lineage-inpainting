"""panel a cache: the E9.0 perturbation-delta UMAP + the representatives' nn distortion.

for each knockout in the E9.0 HVG screen we keep its target-time leaves under
``prune``, take their mean scVI embedding and subtract the control mean -- one 50-d
perturbation delta per gene -- and UMAP those deltas.  nearby points are knockouts that
push the embryo the same way.  every gene is tagged with the lineage it depletes most
(for hueing).  we then pick the representative TFs that ring the map and score each one's
log nn distortion over the E9.0 reference cells.  writes into numerical_results/:
  perturb_umap.csv          gene, umap1, umap2, depleted_lineage, d_<lineage>..., n_leaves, delta_norm
  perturb_umap.npz          genes, deltas, embed, umap, control_mean, lineages, screen_dir, ...
  perturb_umap_panels.pkl   {panel_val: {gene: (N_REF,)}, ref_xy: (N_REF,2), ref_lineage, draw_n}
the panels are stored as raw per-cell values, like the reference UMAP's coords and
lineage labels, so plot_ring.py rasterises and hues them live -- a palette or colour
limit change needs no recompute.

under ``prune`` a leaf that went off the manifold is dropped, the screen's own
convention.  the delta/leaf helpers come from perturb_umap_lib and the nn
distortion's from the tbx5 support comparison, so neither can drift from its pipeline.
select_representatives() is reused (imported) by plot_ring.py.

    python -m analysis.figures.perturb.compute_ring              # rapids_singlecell env (cuml knn)
    python -m analysis.figures.perturb.compute_ring --umap-only  # re-fit the umap alone
"""

import os
import pickle

import h5py
import numpy as np
import pandas as pd

from analysis.pipelines.perturb.perturb_umap_lib import (
    gene_names, leaf_rows, load_gene, load_control, load_cell_categories)
from analysis.pipelines.tbx5.support_comparison.per_embryo import (
    by_embryo, denominator, over_draws, resident, rotation)
from analysis.figures.perturb import style as S

DEN, PRED = 0, 2_000          # seed offsets, so the two draws of one round are independent


def specific(C):
    """isolate the perturbation-specific signal: unit-normalise each delta, estimate
    the consensus direction (mean unit vector) and project it out, so the embedding
    separates genes by *how* they differ rather than by a shared drift."""
    U = C / np.maximum(np.linalg.norm(C, axis=1), 1e-12)[:, None]
    cons = U.mean(0)
    cons /= np.linalg.norm(cons)
    return U - np.outer(U @ cons, cons)


def select_representatives(df, lineage_order=S.LINEAGE_ORDER, tf_list=S.TF_LIST,
                           n_outer=S.RING_N_OUTER, reps_per_lineage=S.RING_REPS_PER_LINEAGE,
                           min_perts=S.RING_MIN_PERTS, show_lineages=S.RING_SHOW_LINEAGES):
    """indices (into df) of the representative TF knockouts that ring the map.

    specificity of a knockout = the margin between its two largest lineage
    depletions (a large margin = it hits its top lineage selectively).  the ringed
    lineages are show_lineages (in order) when given, else every lineage depleted-most
    by >= min_perts TFs.  for each, take the reps_per_lineage most specific TFs, then
    trim by global specificity to at most n_outer panels.  deterministic from df +
    style params, so compute and plot agree.
    """
    with open(tf_list) as fh:
        tf_set = {ln.strip() for ln in fh if ln.strip()}
    lineage_cols = [f"d_{ln}" for ln in lineage_order]
    depl = -df[lineage_cols].to_numpy()                      # positive = depleted
    top2 = np.sort(depl, axis=1)[:, -2:]
    df = df.copy()
    df["specificity"] = top2[:, 1] - top2[:, 0]

    tf = df[df["gene"].isin(tf_set)]
    counts = tf["depleted_lineage"].value_counts()
    if show_lineages is not None:
        eligible = [ln for ln in show_lineages if counts.get(ln, 0) > 0]
    else:
        eligible = [ln for ln in counts.index if counts[ln] >= min_perts]
    picks = pd.concat([tf[tf["depleted_lineage"] == ln].nlargest(reps_per_lineage, "specificity")
                       for ln in eligible])
    if len(picks) > n_outer:
        picks = picks.nlargest(n_outer, "specificity")
    return picks.index.to_numpy(), eligible


## STATES + THE PRUNE READOUT ##

def load_states():
    """the devmap cells' names, scVI embedding, time and embryo, off the h5ad's obs and
    obsm -- the count matrix is never read."""
    with h5py.File(S.STATES_H5AD) as fh:
        o = fh["obs"]
        cat = lambda k: np.array([c.decode() for c in o[k]["categories"][:]])[o[k]["codes"][:]]
        keep = np.flatnonzero(cat("dataset") == "devmap")
        names = pd.Index([x.decode() for x in o["_index"][:]])[keep]
        time = np.asarray(o["time"][:], dtype=float)[keep]
        embryo = cat("embryo")[keep]
        scvi = fh["obsm"]["X_scvi"][:][keep]
    return names, scvi, time, embryo


def resolve(light_trees, name_to_row):
    """row indices of the readout-time leaves of one gene (or the control), under
    ``prune``: a leaf on the manifold at S.OFF_RATIO contributes the cell it snapped to,
    one off it is dropped.  duplicates are kept, so the row multiset reflects the landed
    population.
    """
    return leaf_rows(light_trees, S.RING_TARGET_TIME, S.RING_TOL, name_to_row, S.OFF_RATIO)


def run_umap(embed):
    import umap
    print(f"running UMAP on {len(embed)} perturbation deltas ...", flush=True)
    return umap.UMAP(n_neighbors=S.RING_UMAP_NEIGHBORS, min_dist=S.RING_UMAP_MIN_DIST,
                     metric=S.RING_UMAP_METRIC, random_state=S.seed).fit_transform(embed)


def umap_only():
    """re-fit the UMAP on the cached deltas and rewrite the coords, nothing else.

    neither the deltas nor the representatives nor their panel values depend on the umap
    params, so a layout change costs one fit rather than the whole gene loop.
    """
    d = dict(np.load(os.path.join(S.NUMERICAL, "perturb_umap.npz"), allow_pickle=True))
    d["umap"] = xy = run_umap(d["embed"])
    np.savez(os.path.join(S.NUMERICAL, "perturb_umap.npz"), **d)
    df = pd.read_csv(os.path.join(S.NUMERICAL, "perturb_umap.csv"))
    df["umap1"], df["umap2"] = xy[:, 0], xy[:, 1]
    df.to_csv(os.path.join(S.NUMERICAL, "perturb_umap.csv"), index=False)
    print("rewrote the umap coords in perturb_umap.csv / .npz", flush=True)


def main(isolate_specific=S.RING_ISOLATE_SPECIFIC):
    from tqdm import tqdm

    genes_dir = os.path.join(S.RING_SCREEN, "genes")
    control_dir = os.path.join(S.RING_SCREEN, "control")

    print("loading devmap states (obs + scVI) ...", flush=True)
    names, scvi, time, embryo = load_states()
    name_to_row = {n: i for i, n in enumerate(names)}
    print(f"  {len(names):,} devmap cells, scVI {scvi.shape}", flush=True)

    _, name_to_lineage, _ = load_cell_categories(S.DATA_DIR)
    row_lineage = name_to_lineage.reindex(names).to_numpy()
    lineages = S.LINEAGE_ORDER

    def lineage_fractions(rows):
        vals = row_lineage[rows]
        n = max(len(rows), 1)
        return np.array([(vals == ln).sum() / n for ln in lineages], dtype=float)

    print("pooling control leaves ...", flush=True)
    control = load_control(control_dir)
    ctrl_rows = resolve(control, name_to_row)
    control_mean = scvi[ctrl_rows].mean(axis=0)
    control_lin_frac = lineage_fractions(ctrl_rows)
    print(f"  control: {len(ctrl_rows)} leaves over {len(control)} trees", flush=True)

    genes_all = gene_names(genes_dir)
    print(f"{len(genes_all)} genes; computing per-gene deltas ...", flush=True)
    records, deltas = [], []
    for gene in tqdm(genes_all, desc="per-gene deltas", unit="gene"):
        rows = resolve(load_gene(genes_dir, gene), name_to_row)
        if len(rows) < S.RING_MIN_LEAVES:
            continue
        delta = scvi[rows].mean(axis=0) - control_mean
        dlin = lineage_fractions(rows) - control_lin_frac        # +enriched / -depleted
        deltas.append(delta)
        rec = {"gene": gene, "n_leaves": len(rows),
               "delta_norm": float(np.linalg.norm(delta)),
               "depleted_lineage": lineages[int(np.argmin(dlin))]}
        rec.update({f"d_{ln}": float(v) for ln, v in zip(lineages, dlin)})
        records.append(rec)
    deltas = np.vstack(deltas)
    df = pd.DataFrame(records)
    print(f"kept {len(df)} genes (>= {S.RING_MIN_LEAVES} leaves); deltas {deltas.shape}",
          flush=True)

    embed = specific(deltas) if isolate_specific else deltas
    print(f"{'consensus projected out' if isolate_specific else 'raw deltas'}", flush=True)
    xy = run_umap(embed)
    df["umap1"], df["umap2"] = xy[:, 0], xy[:, 1]

    cols = ["gene", "umap1", "umap2", "depleted_lineage",
            *(f"d_{ln}" for ln in lineages), "n_leaves", "delta_norm"]
    df[cols].to_csv(os.path.join(S.NUMERICAL, "perturb_umap.csv"), index=False)
    np.savez(os.path.join(S.NUMERICAL, "perturb_umap.npz"),
             genes=df["gene"].to_numpy(), deltas=deltas, embed=embed, umap=xy,
             control_mean=control_mean, lineages=np.array(lineages, dtype=object),
             screen_dir=S.RING_SCREEN, data_dir=S.DATA_DIR,
             target_time=S.RING_TARGET_TIME, tol=S.RING_TOL, mode="prune",
             isolate_specific=isolate_specific)
    print("saved perturb_umap.csv / .npz", flush=True)

    # --- representative TFs + their nn distortion over the reference cells ---
    rep_idx, eligible = select_representatives(df)
    rep_genes = list(df["gene"].iloc[rep_idx])
    print(f"{len(eligible)} eligible lineages; representatives: {rep_genes}", flush=True)

    q = np.flatnonzero(np.isclose(time, S.RING_TARGET_TIME)
                       & ~np.isin(row_lineage, S.RING_SKIP_LINEAGES))
    rng = np.random.default_rng(S.seed)
    sample = rng.choice(len(q), min(S.RING_N_REF, len(q)), replace=False)
    ref_names = names[q[sample]]
    ref_xy = pd.read_csv(S.UMAP_CSV, index_col=0).reindex(ref_names)[
        ["UMAP1", "UMAP2"]].to_numpy(dtype=float)
    ref_lineage = name_to_lineage.reindex(ref_names).to_numpy()

    # per reference cell, the log ratio of its nn distance into a draw of the knockout's
    # leaves over its distance into a draw from another reference embryo, averaged over
    # draws and centred on its median.  the knockout's leaves are reference cells, so the
    # numerator is taken from a third embryo -- neither the query cell's nor the
    # denominator's -- or the same cell wins both searches.
    zq, emb = scvi[q], embryo[q]
    into = np.full(len(names), -1)
    into[q] = np.arange(len(q))
    rot = rotation(emb)
    pool = by_embryo(emb)
    pred = {}
    for gene in rep_genes:
        at = into[resolve(load_gene(genes_dir, gene), name_to_row)]
        pred[gene] = by_embryo(emb, at[at >= 0])
    assert all(set(p) == set(pool) for p in pred.values()), \
        "a representative has no leaf in some reference embryo"
    n = min([S.RING_N_MAX, min(len(v) for v in pool.values()) // 2]
            + [len(v) for p in pred.values() for v in p.values()])
    print(f"nn distortion: {len(q):,} in-scope reference cells over {len(pool)} "
          f"embryos, draw n={n:,}, {S.RING_DRAWS} draws", flush=True)

    den = [denominator(zq, emb, pool, rot, n, DEN + s) for s in range(S.RING_DRAWS)]
    panel_val = {}
    for gene in tqdm(rep_genes, desc="nn distortion", unit="panel"):
        v, _ = over_draws(lambda s: resident(zq, emb, pred[gene], rot, n, PRED + s),
                          den, S.RING_DRAWS)
        panel_val[gene] = v[sample].astype("float32")
        print(f"  {gene}: {np.percentile(v, [2, 50, 98]).round(3)} at q2, q50, q98",
              flush=True)

    # the reference UMAP and the panels are both stored as raw per-cell values, not baked
    # PNGs -- plot_ring.py hues them live, so a palette or colour-limit change needs no
    # recompute.
    with open(os.path.join(S.NUMERICAL, "perturb_umap_panels.pkl"), "wb") as fh:
        pickle.dump({"panel_val": panel_val, "ref_xy": ref_xy,
                     "ref_lineage": ref_lineage, "draw_n": n}, fh)
    print("saved perturb_umap_panels.pkl", flush=True)


if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specific", dest="isolate_specific", action="store_true",
                    default=S.RING_ISOLATE_SPECIFIC,
                    help="project out the shared consensus response before UMAP")
    ap.add_argument("--no-specific", dest="isolate_specific", action="store_false",
                    help="UMAP the raw deltas (default)")
    ap.add_argument("--umap-only", action="store_true",
                    help="re-fit the umap on the cached deltas and rewrite the coords")
    args = ap.parse_args()
    umap_only() if args.umap_only else main(isolate_specific=args.isolate_specific)
