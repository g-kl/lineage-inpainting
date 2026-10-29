#!/bin/bash
#SBATCH --job-name=ps_sweep
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00   # kempner_h100's MaxTime; a grid run takes ~1h15
#SBATCH --requeue
#SBATCH --output=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split/logs/ps_sweep_%x_%j.log
#SBATCH --error=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split/logs/ps_sweep_%x_%j.log

set -euo pipefail

cd /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split
source /n/sw/ncf/apps/miniconda3/py39_4.11.0/etc/profile.d/conda.sh
conda activate /n/holylfs06/LABS/mzitnik_lab/Users/nfishman/envs/pool-split

nvidia-smi --query-gpu=name,memory.total --format=csv

: "${EXPERIMENT:?set EXPERIMENT=e_...}"
PAIRING="${PAIRING:-identity}"

overrides=("experiment=${EXPERIMENT}" "pairing=${PAIRING}")
if [[ -n "${ALIGNER:-}" ]]; then
  overrides+=("aligner=${ALIGNER}")
fi
if [[ "$#" -gt 0 ]]; then
  overrides+=("$@")
fi

echo "[sweep] EXPERIMENT=${EXPERIMENT} ALIGNER=${ALIGNER:-<unset>} PAIRING=${PAIRING}"
echo "[sweep] overrides=${overrides[*]}"

exec python main.py "${overrides[@]}"
