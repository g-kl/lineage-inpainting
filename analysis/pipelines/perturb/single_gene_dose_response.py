#!/usr/bin/env python
#SBATCH --job-name=dose_response
#SBATCH --output=logs/dose_response_%j.log
#SBATCH --error=logs/dose_response_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=bates
"""dose response for a single gene by projection.

knocks the gene down / up to a range of target fractions of its own expression
and, per dose, propagates every E<target>-<replicate> tree forward. with
``topology="fixed"`` every lineage keeps the control tree's true topology and only
the perturbed nodes' states are re-predicted -- so cell counts track the observed
tree and a dose shows up as changed states, not a changed cell count. no count
sampler needed. like the projection_screen scripts, each dose's propagated
lineage trees are saved as pickles under analysis/results/dose_response/<gene>_trees/
(dose<NN>.pkl -> dict {tree_name: light-tree}) so summaries can be recomputed
post-hoc. we also keep the target-time leaves to make four
plots in analysis/results/dose_response/:
  1. off-manifold fraction vs dose
  2. perturbation effect norm (cell-type count ratio vs the dose=1 baseline) vs dose
  3. cell-type depletion % vs dose, all types as grey wisps with the biggest drops
     highlighted and labelled
  4. grid of target-time UMAPs, one per dose
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
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from datasets.mouse_embryo import MouseEmbryoDataset
from analysis.pipelines.perturb.perturb import load_model, build_indices, impute_tree_states, run_perturbation
from analysis.pipelines.perturb.perturb_result import off_manifold_descendants


## CONFIG ##

GENE = "Tbx5"
experiment_dir = "outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09"

doses = np.linspace(0, 1, 10)
replicate = "R1"
start_time = 7.5
target_time = 9.0
obs_times = (7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0)
topology = "fixed"
ratio = 1.1
n_bins = 5
n_steps = 32
split_batch_size = 2048
tol = 0.25

out_dir = os.path.join(base_dir, "analysis", "results", "dose_response_v2")
os.makedirs(out_dir, exist_ok=True)
accent = "#c1121f"


def atomic_pickle(obj, path):
    import pickle
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def light(result):
    """drop the latent states; keep only meta (per leaf) + the lineage tree."""
    return {"meta": result.meta, "tree": result.tree}


## SETUP ##

dataset = MouseEmbryoDataset()
print('loaded dataset')

states = dataset.states
umap = dataset.umap_coords
cell_type = states.obs["cell_type"]
cell_types = sorted(cell_type.dropna().unique())

gene = GENE if GENE in states.var_names else GENE[0] + GENE[1:].lower()
gene_idx = list(states.var_names).index(gene)

pooler, splitter, _ = load_model(experiment_dir)
indices = build_indices(dataset, gene_idx, n_bins=n_bins)

print('built indices')

tree_names = [x for x in dataset.trees if f"E{target_time}-{replicate}-" in x]
print(f"{gene}: {len(tree_names)} {replicate} trees, {len(doses)} doses")

pooled = {name: impute_tree_states(splitter, dataset.trees[name], dataset.leaf_states,
                                   batch_size=1024, obs_times=obs_times)
          for name in tree_names}


## RUN: per dose, gather target-time leaves over all trees ##

def target_leaves(tree):
    """on-manifold snapped names at target_time, plus (total, off) counts.

    a cell that projected too far (off-manifold at ``ratio``) and its whole
    subtree are excluded."""
    off = off_manifold_descendants(tree, ratio)
    on, total, n_off = [], 0, 0
    for n, d in tree.nodes(data=True):
        t = d.get("time")
        if t is None or abs(t - target_time) >= tol or not d.get("snapped_name"):
            continue
        total += 1
        if n in off:
            n_off += 1
        else:
            on.append(d["snapped_name"])
    return on, total, n_off


def composition(names):
    """count vector over cell_types for a list of snapped cell names."""
    types = cell_type.reindex(names)
    return np.array([(types == ct).sum() for ct in cell_types], dtype=float)


trees_dir = os.path.join(out_dir, f"{gene}_trees")
os.makedirs(trees_dir, exist_ok=True)

names_by_dose, comp_by_dose, off_frac = {}, {}, []
for dose in doses:
    names, total, n_off = [], 0, 0
    dose_trees = {}
    for name in tree_names:
        result = run_perturbation(
            dataset.trees[name], pooled[name], splitter, None, indices,
            gene_idx, target_time, start_time=start_time, mode="dose",
            topology=topology, target_fraction=dose, ratio=ratio,
            n_steps=n_steps, split_batch_size=split_batch_size, obs_times=obs_times,
            time_tol=tol)
        dose_trees[name] = light(result)
        on, tot, nof = target_leaves(result.tree)
        names += on
        total += tot
        n_off += nof
    names_by_dose[dose] = names
    comp_by_dose[dose] = composition(names)
    off_frac.append(n_off / total if total else 0.0)
    # one pkl per dose: dict {tree_name: light-tree}, mirroring the screen scripts
    dpath = os.path.join(trees_dir, f"dose{dose * 100:.0f}.pkl")
    atomic_pickle(dose_trees, dpath)
    print(f"  dose {dose}: {total} cells, off-manifold {off_frac[-1]:.3f} "
          f"-> {len(dose_trees)} trees saved to {dpath}")


## EFFECT NORM (cell-type count ratio vs the dose=1 baseline) ##

base = comp_by_dose[1.0]
keep = base > 0
effect = [float(np.linalg.norm(comp_by_dose[d][keep] / base[keep] - 1.0)) for d in doses]


## CELL-TYPE DEPLETION (% change in proportion vs the dose=1 baseline) ##
# work in proportions so that differing per-dose cell totals (off-manifold
# exclusion) don't masquerade as depletion. depletion % = 100 * (1 - p/p_base).

props = {d: comp_by_dose[d] / comp_by_dose[d].sum() for d in doses}
base_prop = props[1.0]
ct_keep = base_prop > 0
# depletion[i] is the curve over doses for cell_types[ct_keep][i]
kept_types = [ct for ct, k in zip(cell_types, ct_keep) if k]
depletion = np.array([[100.0 * (1.0 - props[d][j] / base_prop[j]) for d in doses]
                      for j in range(len(cell_types)) if ct_keep[j]])


## PLOTS ##

def line_plot(y, fname):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(doses, y, marker="o", color=accent)
    ax.set_xlabel("target fraction")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


line_plot(off_frac, f"{gene}_off_manifold_fraction.png")
line_plot(effect, f"{gene}_effect_norm.png")


def depletion_plot(fname, n_highlight=5):
    # rank by depletion at the strongest dose (smallest target fraction)
    strongest = int(np.argmin(doses))
    order = np.argsort(depletion[:, strongest])[::-1]   # most depleted first
    top = order[:n_highlight]

    fig, ax = plt.subplots(figsize=(5, 4))
    # every cell type as a faint grey wisp
    for j in range(len(kept_types)):
        ax.plot(doses, depletion[j], color="0.8", lw=0.8, zorder=1)
    # highlight + label the biggest drops
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for rank, j in enumerate(top):
        c = colors[rank % len(colors)]
        ax.plot(doses, depletion[j], color=c, lw=2, zorder=3,
                marker="o", ms=3, label=kept_types[j])
        # label at the strongest dose, where the curves are most separated
        side, ha = (-1, "right") if strongest == 0 else (1, "left")
        ax.annotate(kept_types[j], (doses[strongest], depletion[j, strongest]),
                    color=c, fontsize=8, fontweight="bold",
                    xytext=(side * 4, 0), textcoords="offset points",
                    ha=ha, va="center")
    span = doses[-1] - doses[0]
    ax.set_xlim(doses[0] - 0.18 * span, doses[-1] + 0.05 * span)
    ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=2)
    ax.set_xlabel("target fraction")
    ax.set_ylabel("cell-type depletion (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)


depletion_plot(f"{gene}_celltype_depletion.png")

# UMAP grid: per-dose differential abundance, plus one gene-expression panel.
# differential abundance is a classifier log density ratio: logistic regression
# separates a dose's leaves from the control (dose=1) leaves in scVI latent space,
# and every target-time cell is coloured by logit - log(n_pos/n_neg). this avoids
# the overplotting of duplicated leaves.
name_to_row = {n: i for i, n in enumerate(states.obs_names)}
latent = states.obsm["X_scvi"]
counts = states.layers["counts"]

rng = np.random.default_rng(0)
ref = np.where(np.isclose(states.obs["time"].astype(float), target_time))[0]
if len(ref) > 40000:                       # subsample the reference cells for plotting
    ref = rng.choice(ref, 40000, replace=False)
ref_xy = umap.reindex(states.obs_names[ref])[["UMAP1", "UMAP2"]].to_numpy(float)
ref_latent = latent[ref]
ge = counts[ref, gene_idx]
ref_expr = np.log1p(ge.toarray().ravel() if sp.issparse(ge) else np.asarray(ge).ravel())


def latent_of(names):
    return latent[[name_to_row[n] for n in names]]


control = latent_of(names_by_dose[1.0])


def log_density_ratio(dose):
    pos = latent_of(names_by_dose[dose])
    X = np.vstack([pos, control])
    y = np.r_[np.ones(len(pos)), np.zeros(len(control))]
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    return clf.decision_function(ref_latent) - np.log(len(pos) / len(control))


ratio_by_dose = {d: log_density_ratio(d) for d in doses}
vmax = float(np.abs(np.concatenate(list(ratio_by_dose.values()))).max())

xy_all = umap[["UMAP1", "UMAP2"]].to_numpy(float)
xy_all = xy_all[np.isfinite(xy_all).all(axis=1)]
xlim = (xy_all[:, 0].min(), xy_all[:, 0].max())
ylim = (xy_all[:, 1].min(), xy_all[:, 1].max())


def style(ax, label):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, -0.05, label, transform=ax.transAxes, ha="center", va="top")


n = len(doses) + 1
fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
for ax, dose in zip(axes, doses):
    o = np.argsort(np.abs(ratio_by_dose[dose]))    # strongest enrichment/depletion on top
    sc = ax.scatter(ref_xy[o, 0], ref_xy[o, 1], c=ratio_by_dose[dose][o], cmap="coolwarm",
                    vmin=-vmax, vmax=vmax, s=3, linewidths=0, rasterized=True)
    style(ax, f"{dose:g}")
o = np.argsort(ref_expr)
ex = axes[-1].scatter(ref_xy[o, 0], ref_xy[o, 1], c=ref_expr[o], cmap="viridis",
                      s=3, linewidths=0, rasterized=True)
style(axes[-1], gene)
fig.colorbar(sc, ax=axes[:-1].tolist(), fraction=0.02, pad=0.01)
fig.colorbar(ex, ax=axes[-1], fraction=0.05, pad=0.04)
fig.savefig(os.path.join(out_dir, f"{gene}_umap_grid.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"saved plots to {out_dir}")
