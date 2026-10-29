#!/usr/bin/env python
"""re-score the couplings of a batch of runs (eval_core stage 2, overwrite).

loads the dataset, the annotation maps, the two GENOT couplings (the Hox common
support) and the Hox expression ONCE, then rewrites ``metrics_*.csv`` for every
canonical coupling that exists: the pool side at n=1 and the sampling side at its
canonical n (8 for the bridges, 1 for the one-step splitters).

env: BATCH (required; ``<dir>;...``), STEPS ("5000 20000"), N_TREES (512).
"""
from _common import env_batch, env_int
import os
import pickle
import sys
import traceback

from analysis.lib.eval_core import hox_expression, load_annotations, load_genot_assignments
from analysis.lib.eval_core.runner import assignments_path, run_assignment_metrics
from datasets.mouse_embryo import MouseEmbryoDataset

dirs = [d.rstrip("/") for (d,) in env_batch()]
steps = [int(s) for s in os.environ.get("STEPS", "5000 20000").split()]
print(f"[refresh_metrics_batch] {len(dirs)} runs x steps {steps}")
dataset = MouseEmbryoDataset(max_size=env_int("N_TREES", 512))
ann = load_annotations(dataset)
genot = load_genot_assignments()
expr = hox_expression(dataset)

n_ok = n_fail = 0
for d in dirs:
    cfg = os.path.basename(d)
    n_split = 1 if "_fullen_" in cfg else 8
    for step in steps:
        for n, side in ((1, False), (n_split, True)):
            p = assignments_path(d, cfg, step, n, side)
            if not os.path.exists(p):
                continue
            print(f"\n--- {cfg}  step={step}  n={n}  side={'splitter' if side else 'pool'} ---",
                  flush=True)
            try:
                with open(p, "rb") as f:
                    assignments = pickle.load(f)
                run_assignment_metrics(
                    assignments, d, dataset.trees, ann, step=step, n=n,
                    use_splitter_as_pooler=side, dataset=dataset, genot=genot, expr=expr,
                    overwrite=True)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"  FAILED: {e}", flush=True)
                traceback.print_exc()
print(f"\n[refresh_metrics_batch] done: {n_ok} ok, {n_fail} failed", flush=True)
sys.exit(0 if n_fail == 0 else 1)
