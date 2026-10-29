#!/bin/bash
#SBATCH --job-name=scvi_integrate
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --export=ALL
#SBATCH --partition=bates

set -euo pipefail

# Force run from the directory where sbatch was submitted
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

source /mnt/home/gokulg/miniconda3/etc/profile.d/conda.sh
conda activate lt

echo "Running from: $(pwd)"
echo "Using python: $(which python)"

python -u integrate.py
