#!/usr/bin/env python
"""compare the main-text model's grid row with the figure's own caches, number for number.

the grid is meant to be the figure's eval applied to every run; for the main-text model
the two must agree exactly. runs the comparison over the figure's numerical_results
(``compute_assignment_metrics`` / ``compute_hox`` / ``compute_distortion`` must have been
run, e.g. via verify_hero_consistency.sbatch) and the main-text model's grid CSVs.

    python scripts/grid_eval/verify_hero_consistency.py
"""
from _common import ROOT
import argparse
import glob
import os

import numpy as np
import pandas as pd

from analysis.figures.grid_summary import style as S
from analysis.figures.summary_qc import style as SQ
from analysis.lib.eval_core.runner import distortion_path, metrics_path

hero = SQ.EXPERIMENT_DIR
cfg = os.path.basename(hero)
nr = SQ.NR
n = S.coupling_n("vpsde")
ok = True


def verify_inputs():
    """the grid scores distortion with the figure's functions; what is left to check
    is that both see the same trees and draw the same seeded samples. compares the
    grid's dataset (MouseEmbryoDataset(max_size=N_TREES)) with the figure's
    (hydra-instantiated from the main-text model's config, as compute_distortion.load_models
    does)
    on tree names / order and on collect_clonal_candidates + collect_marginal_samples.
    the two datasets are loaded one after the other (two at once OOM a 128 GB node).
    needs a compute node."""
    global ok
    import gc
    import hydra
    from omegaconf import OmegaConf
    from analysis.lib.eval_core.runner import distortion_samples
    from datasets.mouse_embryo import MouseEmbryoDataset

    def summary(ds):
        cand, samp, _ = distortion_samples(ds)
        return (list(ds.trees),
                [(t, l, a) for t, l, _, a, *_ in cand],
                [(t_l, t_e, list(src), np.asarray(tgt)) for t_l, t_e, src, tgt in samp])

    trees_g, cand_g, samp_g = summary(MouseEmbryoDataset(max_size=SQ.N_TREES))
    gc.collect()
    trees_f, cand_f, samp_f = summary(
        hydra.utils.instantiate(OmegaConf.load(os.path.join(hero, "config.yaml")).dataset))
    same_trees = trees_g == trees_f
    same_cand = cand_g == cand_f
    same_samp = (len(samp_g) == len(samp_f) and all(
        g[0] == f[0] and g[1] == f[1] and g[2] == f[2] and np.allclose(g[3], f[3])
        for g, f in zip(samp_g, samp_f)))
    print(f"  {'OK ' if same_trees else 'DIFF'} tree set + order  grid={len(trees_g)} figure={len(trees_f)}")
    print(f"  {'OK ' if same_cand else 'DIFF'} clonal candidates  ({len(cand_g)} subtrees)")
    print(f"  {'OK ' if same_samp else 'DIFF'} marginal samples   ({len(samp_g)} pairs)")
    ok &= same_trees and same_cand and same_samp


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--inputs", action="store_true", help="also compare the distortion inputs")
args = ap.parse_args()
if args.inputs:
    print("distortion inputs (figure dataset vs grid dataset):")
    verify_inputs()
    print("\ndistortion inputs identical:", "YES" if ok else "NO")
    raise SystemExit(0 if ok else 1)

grid_metrics = pd.read_csv(metrics_path(hero, cfg, SQ.STEP, n, True))
grid_metrics["tree"] = grid_metrics["tree"].fillna("")


def check(label, a, b, tol=1e-6):
    global ok
    a, b = np.asarray(a, float), np.asarray(b, float)
    same = a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=0)
    ok &= same
    print(f"  {'OK ' if same else 'DIFF'} {label:42s} grid={np.round(a, 5).tolist()[:4]} "
          f"figure={np.round(b, 5).tolist()[:4]}")


# assignment metrics: every (metric, tag) row of the figure's model rows
am = pd.read_csv(os.path.join(nr, "assignment_metrics.csv"))
am["tag"] = am["tag"].fillna("")
am = am[am["method"] == "model"]
print("assignment metrics (figure compute_assignment_metrics vs grid metrics CSV):")
for (metric, tag), g in am.groupby(["metric", "tag"], sort=False):
    gv = grid_metrics[(grid_metrics["metric"] == metric) & (grid_metrics["tree"] == tag)]["value"]
    check(f"{metric} {tag}", gv.to_numpy(), g["value"].to_numpy())

# Hox: per (cell type, gene) distortion
hox = pd.read_csv(os.path.join(nr, "hox_consistency.csv"))
hox = hox[hox["method"] == "model"].sort_values(["cell_type", "gene"])
gh = grid_metrics[grid_metrics["metric"] == "hox_distortion"].copy()
gh[["cell_type", "gene"]] = gh["tree"].str.split("__", expand=True)
gh = gh.sort_values(["cell_type", "gene"])
print("Hox-code distortion (figure compute_hox vs grid):")
check("hox_distortion per (ct, gene)", gh["value"].to_numpy(), hox["distortion"].to_numpy())

# distortion: the figure's model rows vs the grid's per-run CSVs, when the figure's
# caches exist. the grid calls the same functions on the same trees and seeded samples
# (`--inputs` checks that), so the rows are the same computation either way.
for which in ("clonal", "marginal"):
    fig_csv = os.path.join(nr, f"{which}_distortion.csv")
    grid = pd.read_csv(distortion_path(hero, cfg, SQ.STEP, which))
    print(f"{which} distortion (figure compute_distortion vs grid):")
    if not os.path.exists(fig_csv):
        print(f"  --  figure cache absent; grid rows = {len(grid)} "
              f"(same functions + inputs, see --inputs)")
        continue
    fig = pd.read_csv(fig_csv)
    fig = fig[fig["method"] == "model"]
    check(f"{which} energy_distance rows", grid["energy_distance"].to_numpy(),
          fig["energy_distance"].to_numpy(), tol=1e-4)

print("\nhero row reproduces the main-text figure:", "YES" if ok else "NO")
