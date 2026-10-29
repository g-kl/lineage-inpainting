#!/usr/bin/env python
"""cell-type transition maps for a batch of coupling pickles, under the main-text
figure's map rule (summary_qc.style: PMI_QUANTILE, MIN_COMPOSITION, COMPOSITION,
RESOLUTION_BY_TIME, every pair), one dataset load.

writes ``cell_type_map_<cfg>_step{S}<suffix>.pkl`` beside each coupling (always
rebuilt, so a changed cutoff propagates) and folds the curated-transition rows into
the matching metrics CSV, as analysis.lib.celltype_map.eval always does.

env: BATCH (required; ``<assignments pkl>;...``), N_TREES (512).
"""
from _common import env_batch, env_int
import os
import sys
import time
import traceback

from analysis.figures.summary_qc import style as SQ
from analysis.lib.celltype_map.eval import build_cell_type_map
from datasets.mouse_embryo import MouseEmbryoDataset

paths = [p for (p,) in env_batch()]
print(f"[eval_cell_type_map_batch] {len(paths)} couplings, pmi_quantile={SQ.PMI_QUANTILE}, "
      f"min_composition={SQ.MIN_COMPOSITION} ({SQ.COMPOSITION})")
dataset = MouseEmbryoDataset(max_size=env_int("N_TREES", 512))

n_ok = n_fail = 0
for i, p in enumerate(paths):
    t0 = time.time()
    print(f"\n--- [{i + 1}/{len(paths)}] {os.path.basename(p)} ---", flush=True)
    try:
        # the main figure's map rule, as compute_transition_map passes it
        build_cell_type_map(p, dataset=dataset, pmi_quantile=SQ.PMI_QUANTILE,
                            n_max=None, resolution_by_time=SQ.RESOLUTION_BY_TIME,
                            min_composition=SQ.MIN_COMPOSITION, composition=SQ.COMPOSITION,
                            palette=SQ.TRANSITION_PALETTE, save_png=False)
        print(f"  ok ({time.time() - t0:.0f}s)")
        n_ok += 1
    except Exception as e:
        n_fail += 1
        print(f"  FAILED: {e}", flush=True)
        traceback.print_exc()
print(f"\n[eval_cell_type_map_batch] done: {n_ok} ok, {n_fail} failed", flush=True)
sys.exit(0 if n_fail == 0 else 1)
