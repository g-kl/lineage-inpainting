#!/bin/bash
#SBATCH --job-name=verify_hero
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --requeue
#SBATCH --exclude=holygpu8a19102
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split/logs/verify_hero_%j.log
# run the main-text figure's own compute scripts for the main-text model (its
# numerical_results caches), so verify_hero_consistency.py can compare the grid's
# main-text model row against them.
set -uo pipefail
cd /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split
source /n/sw/ncf/apps/miniconda3/py39_4.11.0/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/envs/pool-split
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# the grid's distortion is checked on its inputs: the same functions on the same trees
# and seeded samples give the same rows.
if [[ "${INPUTS_ONLY:-0}" != "1" ]]; then
  python -u -m analysis.figures.summary_qc.compute_assignment_metrics
  python -u -m analysis.figures.summary_qc.compute_hox
fi
python -u scripts/grid_eval/verify_hero_consistency.py --inputs
echo "== figure caches =="; ls -la analysis/figures/summary_qc/numerical_results/
