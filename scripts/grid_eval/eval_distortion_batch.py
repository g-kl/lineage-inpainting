#!/usr/bin/env python
"""the distortion CSVs (eval_core stage 3: clonal + marginal + descendant with the
splitter, marginal_pool with the imputing model) for a batch of (run, step,
coupling n) tuples; only the missing CSVs unless OVERWRITE.

loads the dataset and draws the main-text figure's seeded subtree / cell samples ONCE,
then rolls each run's splitter through compute_distortion.model_clonal /
model_marginal. a clonal CSV without a ``method`` column is treated as stale and
recomputed.

env: BATCH (required; ``<dir>|<step>|<coupling n>;...``, n = the run's sampling-side
coupling: 8 bridges / 1 one-step), N_TREES (512), OVERWRITE (0/1: recompute even when
fresh CSVs exist, e.g. after a change of summary_qc.style.SPLIT_STEPS).
"""
from _common import env_batch, env_flag, env_int
import os
import sys
import time
import traceback

from analysis.lib.eval_core.runner import distortion_path, distortion_samples, run_model_metrics
from datasets.mouse_embryo import MouseEmbryoDataset


def _stale_clonal(model_dir, step):
    p = distortion_path(model_dir, os.path.basename(model_dir), step, "clonal")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        return "method" not in f.readline()


tuples = [(d.rstrip("/"), int(s), int(n)) for d, s, n in env_batch()]
print(f"[eval_distortion_batch] {len(tuples)} (run, step, n) tuples")
dataset = MouseEmbryoDataset(max_size=env_int("N_TREES", 512))
print(f"  {len(dataset.trees)} trees")
samples = distortion_samples(dataset)

n_ok = n_fail = 0
for i, (d, s, n) in enumerate(tuples):
    t0 = time.time()
    print(f"\n--- [{i + 1}/{len(tuples)}] {os.path.basename(d)}  step={s}  n={n} ---", flush=True)
    try:
        out = run_model_metrics(d, dataset, step=s, samples=samples, coupling_n=n,
                                overwrite=env_flag("OVERWRITE") or _stale_clonal(d, s))
        print(f"  ok ({time.time() - t0:.0f}s):", os.path.basename(out["clonal"]))
        n_ok += 1
    except Exception as e:
        n_fail += 1
        print(f"  FAILED: {e}", flush=True)
        traceback.print_exc()
print(f"\n[eval_distortion_batch] done: {n_ok} ok, {n_fail} failed", flush=True)
sys.exit(0 if n_fail == 0 else 1)
