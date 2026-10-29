#!/bin/bash
#SBATCH --job-name=render_grid
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split/logs/render_grid_%j.log
# rebuild the grid-summary caches and figures at every step (the launcher's --then-render)
set -euo pipefail
cd /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split
source /n/sw/ncf/apps/miniconda3/py39_4.11.0/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/envs/pool-split
for s in ${STEPS:-5000 20000}; do
  python -m analysis.figures.grid_summary.compute_grid --step $s
  python -m analysis.figures.grid_summary.assemble --step $s
done
