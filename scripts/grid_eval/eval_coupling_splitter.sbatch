#!/bin/bash
#SBATCH --job-name=eval_coupling_splitter
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --requeue
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split/logs/eval_coupling_splitter_%j.log
set -euo pipefail
cd /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split
source /n/sw/ncf/apps/miniconda3/py39_4.11.0/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/envs/pool-split
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# sampling-side coupling: the splitter used as the pooler, POOL_N_STEPS integration
# steps (8 = the main-text figure setting for the bridges; 1 for one-step splitters).
# env: EXPERIMENT_DIR, STEP, POOL_N_STEPS.
: "${EXPERIMENT_DIR:?}"
export USE_SPLITTER_AS_POOLER=1 POOL_N_STEPS="${POOL_N_STEPS:-8}"
exec python -u scripts/grid_eval/eval_coupling_one_config.py
