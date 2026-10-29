#!/bin/bash
#SBATCH --job-name=hold_out_vpsde
#SBATCH --partition=bates
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/gokulg/pool-and-split/logs/hold_out_vpsde_%j.log
#SBATCH --error=/mnt/home/gokulg/pool-and-split/logs/hold_out_vpsde_%j.log

# the paper's model (experiment/mouse_embryo_vpsde.yaml) as is, on the train split.
# the clone list goes into the config, so it is hashed into the run dir and saved
# in its config.yaml.

set -euo pipefail

cd /mnt/home/gokulg/pool-and-split
source /mnt/home/gokulg/miniconda3/etc/profile.d/conda.sh
conda activate lt

export PYTHONPATH=.

CLONES=$(python -c "from analysis.figures.supp_holdout.common import HOLDOUT; print(','.join(HOLDOUT))")

python main.py experiment=mouse_embryo_vpsde experiment_name=hold_out_vpsde \
  dataset._target_=analysis.figures.supp_holdout.common.HeldOutMouseEmbryoDataset \
  "+dataset.holdout_clones=[${CLONES}]"
