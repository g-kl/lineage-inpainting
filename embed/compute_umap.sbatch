#!/bin/bash
#SBATCH --job-name=rapids_umap
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --export=ALL
#SBATCH --partition=bates
#SBATCH --gres=gpu:1

set -euo pipefail

# Force run from the directory where sbatch was submitted
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

source /mnt/home/gokulg/miniconda3/etc/profile.d/conda.sh
conda activate rapids_singlecell

echo "Running from: $(pwd)"
echo "Using python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -u embed/compute_umap.py
