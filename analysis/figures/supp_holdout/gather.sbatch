#!/bin/bash
#SBATCH --job-name=hold_out_gather
#SBATCH --partition=bates
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/home/gokulg/pool-and-split/logs/hold_out_gather_%j.log
#SBATCH --error=/mnt/home/gokulg/pool-and-split/logs/hold_out_gather_%j.log

# cpu only: the merged caches and hox. PARITY=1 for the parity run.

set -euo pipefail

cd /mnt/home/gokulg/pool-and-split
source /mnt/home/gokulg/miniconda3/etc/profile.d/conda.sh
conda activate lt

export PYTHONPATH=.

python -m analysis.figures.supp_holdout.gather ${PARITY:+--parity}
