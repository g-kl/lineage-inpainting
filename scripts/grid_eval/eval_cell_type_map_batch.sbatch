#!/bin/bash
#SBATCH --job-name=eval_cell_type_map_batch
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --requeue
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split/logs/eval_cell_type_map_batch_%j.log
set -euo pipefail
cd /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split
source /n/sw/ncf/apps/miniconda3/py39_4.11.0/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/envs/pool-split
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
: "${BATCH:?}"
exec python -u scripts/grid_eval/eval_cell_type_map_batch.py
