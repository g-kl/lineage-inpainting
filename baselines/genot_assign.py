#!/usr/bin/env python
#SBATCH --job-name=genot_assign
#SBATCH --output=genot_assign_%j.log
#SBATCH --error=genot_assign_%j.log
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=bates
"""
compute the per-node assignments of the two saved GENOT checkpoints with
genot_common.build_assignments, without retraining. only the `assignments` key of
each model pkl changes. it is also written to the assignments_*_unified.pkl
sidecar read by analysis/lib/eval_core. run from the repo root.
"""
import os
import sys
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

sys.path.insert(
    0,
    os.environ.get(
        "SLURM_SUBMIT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
    ),
)
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # leave room for faiss-GPU

import jax.random as jr

from baselines import genot_common as gc
from datasets.mouse_embryo import MouseEmbryoDataset

# (model pkl, assignments sidecar)
PKLS = [
    ("outputs/e_baselines/genot_w2_unified_model.pkl",
     "outputs/e_baselines/assignments_w2_unified.pkl"),
    ("outputs/e_baselines/genot_fgw_unified_model.pkl",
     "outputs/e_baselines/assignments_fgw_unified.pkl"),
]
N_SAMPLES = 5   # backward draws averaged per cell, as N_SAMPLES in the training scripts
SEED = 42       # the training scripts' seed

log.info("loading dataset")
dataset = MouseEmbryoDataset()

for pkl_path, sidecar_path in PKLS:
    log.info("=== %s", pkl_path)
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    model = gc.load_mlp(saved["model_config"], saved["flat_params"])
    parent, m = gc.backward_parent_map(
        model, dataset.states, t_min=saved["t_min"], t_max=saved["t_max"],
        key=jr.PRNGKey(SEED), n_samples=N_SAMPLES,
    )
    saved["assignments"] = gc.build_assignments(
        dataset.trees, dataset.states, parent, m, seed=SEED)

    # write then rename, so a crash can't truncate the checkpoint
    with open(pkl_path + ".tmp", "wb") as f:
        pickle.dump(saved, f)
    os.replace(pkl_path + ".tmp", pkl_path)
    with open(sidecar_path, "wb") as f:
        pickle.dump(saved["assignments"], f)
    log.info("wrote %s and %s", pkl_path, sidecar_path)

log.info("done")
