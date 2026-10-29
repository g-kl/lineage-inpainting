#!/usr/bin/env python
#SBATCH --job-name=perturb_umap
#SBATCH --output=logs/perturb_umap_%j.log
#SBATCH --error=logs/perturb_umap_%j.log
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
"""build a UMAP where every point is one perturbation.

for each knockout pkl in a screen directory we pool the target-time leaves it
landed on (across all starting trees), take their mean scVI embedding, and
subtract the control mean -- giving one 50-d "perturbation delta" per gene.  a
UMAP of those deltas is the perturbation map: nearby points are knockouts that
push the embryo in similar directions.

every gene is also tagged with the *lineage* it depletes most (the lineage
whose landed-cell fraction drops furthest below control), for hueing the plot.
the lineage labels come from the small annotation CSVs (final_subtype.csv +
final_type_info.csv) via perturb_umap_lib.

OUTPUT (analysis/results/perturb_umap/):
  perturb_umap.csv    gene, umap1, umap2, depleted_lineage, d_<lineage>..., n_leaves, delta_norm
  perturb_umap.npz    genes, deltas (n,50), embed (n,50), umap (n,2),
                      control_mean (50,), lineages, ...
draw it with analysis/pipelines/perturb/perturb_umap_plot.py.
"""

import os
import sys

base_dir = os.path.abspath(os.environ.get(
    "SLURM_SUBMIT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
))
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from datasets.mouse_embryo import MouseEmbryoDataset
from analysis.pipelines.perturb.perturb_umap_lib import (
    LINEAGE_ORDER, leaf_rows, gene_names, load_gene, load_control,
    load_cell_categories)


## CONFIG ##

experiment_dir = "outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09"
screen_name = "hvg_projection_screen_ko"     # which screen's knockouts to map
data_dir = "../data/embryos"                  # annotation CSVs live here
target_time = 9.0
tol = 0.1                  # |leaf time - target_time| window for terminal leaves
min_leaves = 200           # skip a gene with fewer landed leaves than this
isolate_specific = True    # remove the shared consensus response before embedding
umap_neighbors = 10
umap_min_dist = 0.2
umap_metric = "euclidean"
seed = 0

screen_dir = os.path.join(experiment_dir, screen_name)
genes_dir = os.path.join(screen_dir, "genes")
control_dir = os.path.join(screen_dir, "control")
out_dir = os.path.join(base_dir, "analysis", "results", "perturb_umap")
os.makedirs(out_dir, exist_ok=True)


## DATASET (scVI embeddings) + LINEAGE LABELS ##

print("loading dataset (scVI embeddings) ...", flush=True)
dataset = MouseEmbryoDataset()
states = dataset.states
scvi = np.asarray(states.obsm["X_scvi"], dtype=np.float32)
name_to_row = {n: i for i, n in enumerate(states.obs_names)}
print(f"  states {states.shape}, scVI {scvi.shape}", flush=True)

# per-cell lineage (row-aligned to states) from the small annotation CSVs
_, name_to_lineage, _ = load_cell_categories(data_dir)
row_lineage = name_to_lineage.reindex(states.obs_names).to_numpy()
lineages = LINEAGE_ORDER


def lineage_fractions(rows):
    """fraction of ``rows`` falling in each lineage (fixed LINEAGE_ORDER)."""
    vals = row_lineage[rows]
    n = max(len(rows), 1)
    return np.array([(vals == ln).sum() / n for ln in lineages], dtype=float)


## CONTROL MEAN + COMPOSITION (pooled over all control trees) ##

print("pooling control leaves ...", flush=True)
control = load_control(control_dir)
ctrl_rows = leaf_rows(control, target_time, tol, name_to_row)
control_mean = scvi[ctrl_rows].mean(axis=0)
control_lin_frac = lineage_fractions(ctrl_rows)
print(f"  control: {len(ctrl_rows)} leaves over {len(control)} trees", flush=True)


## PER-GENE DELTA + LINEAGE DEPLETION ##

genes_all = gene_names(genes_dir)
print(f"{len(genes_all)} genes in {genes_dir}", flush=True)

records, deltas = [], []
for gene in tqdm(genes_all, desc="genes"):
    rows = leaf_rows(load_gene(genes_dir, gene), target_time, tol, name_to_row)
    if len(rows) < min_leaves:
        continue
    delta = scvi[rows].mean(axis=0) - control_mean
    dlin = lineage_fractions(rows) - control_lin_frac          # +enriched / -depleted
    deltas.append(delta)
    rec = {"gene": gene, "n_leaves": len(rows),
           "delta_norm": float(np.linalg.norm(delta)),
           "depleted_lineage": lineages[int(np.argmin(dlin))]}
    rec.update({f"d_{ln}": float(v) for ln, v in zip(lineages, dlin)})
    records.append(rec)

deltas = np.vstack(deltas)
df = pd.DataFrame(records)
print(f"kept {len(df)} genes (>= {min_leaves} leaves); delta matrix {deltas.shape}", flush=True)


## EMBED THE DELTAS ##

def specific(C):
    """isolate the perturbation-specific signal in a delta matrix.

    unit-normalise each gene's delta, estimate the consensus direction (the
    response shared by all knockouts) as the mean unit vector, and project it
    out -- what remains is the component specific to each TF, so the embedding
    and clustering separate genes by *how* they differ rather than by a common
    drift.
    """
    U = C / np.maximum(np.linalg.norm(C, axis=1), 1e-12)[:, None]
    cons = U.mean(0)
    cons /= np.linalg.norm(cons)
    return U - np.outer(U @ cons, cons)


embed = specific(deltas) if isolate_specific else deltas

print("running UMAP on perturbation deltas ...", flush=True)
import umap
reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_min_dist,
                    metric=umap_metric, random_state=seed)
xy = reducer.fit_transform(embed)
df["umap1"], df["umap2"] = xy[:, 0], xy[:, 1]


## SAVE ##

cols = ["gene", "umap1", "umap2", "depleted_lineage",
        *(f"d_{ln}" for ln in lineages), "n_leaves", "delta_norm"]
df[cols].to_csv(os.path.join(out_dir, "perturb_umap.csv"), index=False)
np.savez(os.path.join(out_dir, "perturb_umap.npz"),
         genes=df["gene"].to_numpy(), deltas=deltas, embed=embed, umap=xy,
         control_mean=control_mean, lineages=np.array(lineages, dtype=object),
         screen_dir=screen_dir, data_dir=data_dir,
         target_time=target_time, tol=tol)
print(f"saved perturb_umap.csv / .npz to {out_dir}", flush=True)
