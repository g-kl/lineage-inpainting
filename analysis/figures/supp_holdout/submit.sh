#!/bin/bash
# submit the hold-out pipeline with slurm dependencies.
#
#   bash analysis/figures/supp_holdout/submit.sh parity   # paper models on every tree vs fig 3
#   bash analysis/figures/supp_holdout/submit.sh          # train, evaluate, gather

set -euo pipefail

cd /mnt/home/gokulg/pool-and-split
D=analysis/figures/supp_holdout

if [[ "${1:-}" == "parity" ]]; then
  E=$(sbatch --parsable --array=0-2 --export=ALL,PARITY=1 ${D}/evaluate.sbatch)
  G=$(sbatch --parsable --dependency=afterok:${E} --export=ALL,PARITY=1 ${D}/gather.sbatch)
  echo "parity: evaluate ${E}, gather ${G}"
  exit 0
fi

T=$(sbatch --parsable ${D}/train_vpsde.sbatch)
N=$(sbatch --parsable ${D}/train_genot.sbatch)
E=$(sbatch --parsable --dependency=afterok:${T}:${N} ${D}/evaluate.sbatch)
G=$(sbatch --parsable --dependency=afterok:${E} ${D}/gather.sbatch)
echo "train_vpsde ${T}, train_genot ${N}, evaluate ${E}, gather ${G}"
