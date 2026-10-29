#!/bin/bash
# Run an analysis/pipelines/perturb/<SCRIPT>.py screen on the H100 partition.
# Required env: SCRIPT, the screen to run (e.g. heart_projection_screen, tbx5_projection_screen)
#SBATCH --job-name=perturb_h100
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --requeue
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split/logs/perturb_h100_%j.log
set -euo pipefail
cd /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split
source /n/sw/ncf/apps/miniconda3/py39_4.11.0/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/envs/pool-split
: "${SCRIPT:?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec python analysis/pipelines/perturb/${SCRIPT}.py
