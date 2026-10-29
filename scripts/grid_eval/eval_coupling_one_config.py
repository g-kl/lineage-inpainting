#!/usr/bin/env python
"""couple one grid run and score its coupling (eval_core stages 1 + 2).

env: EXPERIMENT_DIR (required), STEP (default latest), POOL_N_STEPS (default 1),
     USE_SPLITTER_AS_POOLER (0/1), N_TREES (512), OVERWRITE (0/1).
"""
from _common import env_flag, env_int
import os

from analysis.lib.eval_core import hox_expression, load_annotations, load_genot_assignments
from analysis.lib.eval_core.runner import eval_model
from datasets.mouse_embryo import MouseEmbryoDataset

experiment_dir = os.environ["EXPERIMENT_DIR"].rstrip("/")
step = int(os.environ["STEP"]) if os.environ.get("STEP") else None
n = env_int("POOL_N_STEPS", 1)
use_splitter = env_flag("USE_SPLITTER_AS_POOLER")
print(f"[eval_coupling] {os.path.basename(experiment_dir)}  step={step or 'latest'}  "
      f"n={n}  splitter_as_pooler={use_splitter}")

dataset = MouseEmbryoDataset(max_size=env_int("N_TREES", 512))
out = eval_model(
    experiment_dir, dataset, load_annotations(dataset), step=step, n=n,
    use_splitter_as_pooler=use_splitter, overwrite=env_flag("OVERWRITE"),
    genot=load_genot_assignments(), expr=hox_expression(dataset))
print("done:", {k: (os.path.basename(v) if isinstance(v, str) else v) for k, v in out.items()})
