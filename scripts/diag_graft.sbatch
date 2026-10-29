#!/bin/bash
#SBATCH --job-name=diag_graft
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split/logs/diag_graft_%x_%j.log
#SBATCH --error=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split/logs/diag_graft_%x_%j.log

set -euo pipefail
cd /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split
source /n/sw/ncf/apps/miniconda3/py39_4.11.0/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/envs/pool-split
exec python scripts/diag_graft_vs_natural.py
