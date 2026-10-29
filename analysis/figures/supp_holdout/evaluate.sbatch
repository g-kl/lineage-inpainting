#!/bin/bash
#SBATCH --job-name=hold_out_eval
#SBATCH --partition=bates
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --array=0-9%4
#SBATCH --output=/mnt/home/gokulg/pool-and-split/logs/hold_out_eval_%A_%a.log
#SBATCH --error=/mnt/home/gokulg/pool-and-split/logs/hold_out_eval_%A_%a.log

# one task per method in common.METHODS, four at a time. PARITY=1 runs
# common.PARITY_METHODS with --parity instead (submit with --array=0-2).

set -euo pipefail

cd /mnt/home/gokulg/pool-and-split
source /mnt/home/gokulg/miniconda3/etc/profile.d/conda.sh
conda activate lt

export PYTHONPATH=.

if [[ -n "${PARITY:-}" ]]; then
  LIST=PARITY_METHODS
  FLAG=--parity
else
  LIST=METHODS
  FLAG=
fi
METHOD=$(python -c "from analysis.figures.supp_holdout.common import ${LIST}; print(${LIST}[${SLURM_ARRAY_TASK_ID}])")

python -m analysis.figures.supp_holdout.evaluate "${METHOD}" ${FLAG}
