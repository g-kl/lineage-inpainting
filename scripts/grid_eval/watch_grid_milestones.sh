#!/bin/bash
# Watch the grid for 20k checkpoints and auto-submit the coupling evals.
# Bridge models get n=1,8,32; TreeFormers get n=1.
#
# Markers: each fired (model, milestone) drops .eval_<step>_submitted in the
# model dir so a restart of this watcher doesn't double-fire.
set -uo pipefail
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split
OUT="$ROOT/outputs"
SBATCH_DIR="$ROOT/scripts/grid_eval"
milestones=(20000)

while true; do
  # Bridge models: n=1, 8, 32
  for D in $(ls -d "$OUT"/mouse_embryo_grid_bridge_* 2>/dev/null); do
    # Tag = everything between `mouse_embryo_grid_` and the trailing hex hash.
    # Works for any number of axes (5 for std, 6 with the aligner_set suffix).
    tag=$(basename "$D" | sed -E 's/^mouse_embryo_grid_(.*)_[a-f0-9]+$/\1/')
    # Splitter-as-pooler coupling only makes sense when the pool and split
    # models differ; for shared configs the two paths produce identical
    # results, so skip the extra job.
    # For unshared bridges, the pool model is one-step (DTFM), so the
    # pooler-side coupling is identical at every n -- only fire cpl at n=1
    # there. The splitter is still multi-step, so the splitter-side coupling
    # (cplspl) still varies over n. (The distortion and map batches are the
    # launcher's; see launch_consistency_recompute.py.)
    is_unshared=0
    [[ "$tag" == bridge_unshared_* ]] && is_unshared=1
    # One-step splitters (fullen = FullEnergyDeltaTreeFormer, meanpool =
    # parameter-free MeanPool grid): n>1 is meaningless, only fire n=1.
    one_step=0
    [[ "$tag" == *_fullen_* || "$tag" == *_meanpool ]] && one_step=1
    for step in "${milestones[@]}"; do
      marker="$D/.eval_${step}_submitted"
      if [[ -f "$D/pool_ema_step${step}.pt" && ! -f "$marker" ]]; then
        touch "$marker"
        if [[ $one_step -eq 1 ]]; then ns="1"; else ns="1 8 32"; fi
        for n in $ns; do
          cpl_msg=""
          if [[ $is_unshared -eq 0 || $n -eq 1 ]]; then
            cjid=$(EXPERIMENT_DIR="$D" STEP="$step" N_TREES=512 POOL_N_STEPS=$n \
              sbatch --parsable --job-name="cpl_${tag}_${step}_n${n}" "$SBATCH_DIR/eval_coupling.sbatch" 2>/dev/null)
            cpl_msg="cpl=$cjid "
          fi
          cspl_msg=""
          if [[ $is_unshared -eq 1 ]]; then
            cspljid=$(EXPERIMENT_DIR="$D" STEP="$step" N_TREES=512 POOL_N_STEPS=$n \
              sbatch --parsable --job-name="cplspl_${tag}_${step}_n${n}" "$SBATCH_DIR/eval_coupling_splitter.sbatch" 2>/dev/null)
            cspl_msg=" cplspl=$cspljid"
          fi
          echo "SUBMITTED ${tag} step=${step} n=${n}: ${cpl_msg}${cspl_msg}"
        done
      fi
    done
  done

  # TreeFormer models: n=1 only (one-step inference, no flow integration)
  for D in $(ls -d "$OUT"/mouse_embryo_grid_tf_* 2>/dev/null); do
    # TF tag = everything between `mouse_embryo_grid_` and the trailing hex
    # hash. Handles 3-segment names and suffixes (distnl/full, optional
    # aligner _set).
    tag=$(basename "$D" | sed -E 's/^mouse_embryo_grid_(.*)_[a-f0-9]+$/\1/')
    for step in "${milestones[@]}"; do
      marker="$D/.eval_${step}_submitted"
      if [[ -f "$D/pool_ema_step${step}.pt" && ! -f "$marker" ]]; then
        touch "$marker"
        cjid=$(EXPERIMENT_DIR="$D" STEP="$step" N_TREES=512 POOL_N_STEPS=1 \
          sbatch --parsable --job-name="cpl_${tag}_${step}_n1" "$SBATCH_DIR/eval_coupling.sbatch" 2>/dev/null)
        echo "SUBMITTED ${tag} step=${step} n=1: cpl=$cjid"
      fi
    done
  done

  sleep 300
done
