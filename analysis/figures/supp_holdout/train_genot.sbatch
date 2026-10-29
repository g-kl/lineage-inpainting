#!/bin/bash
#SBATCH --job-name=hold_out_genot
#SBATCH --partition=bates
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/mnt/home/gokulg/pool-and-split/logs/hold_out_genot_%A_%a.log
#SBATCH --error=/mnt/home/gokulg/pool-and-split/logs/hold_out_genot_%A_%a.log

# one task per genot variant (common.GENOT_VARIANTS order: the ~2 h cbFGW ones first).
# three at a time, so with train_vpsde this fills four gpus.

set -euo pipefail

cd /mnt/home/gokulg/pool-and-split
source /mnt/home/gokulg/miniconda3/etc/profile.d/conda.sh
conda activate lt

export PYTHONPATH=.

NAME=$(python -c "from analysis.figures.supp_holdout.common import GENOT_VARIANTS; print(list(GENOT_VARIANTS)[${SLURM_ARRAY_TASK_ID}])")

python -m analysis.figures.supp_holdout.train_genot "${NAME}"
