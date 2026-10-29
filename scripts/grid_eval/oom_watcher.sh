#!/bin/bash
# Watch grid training (and eval) jobs for CUDA OOM failures and emit one line
# per newly-failed job. Dedup by job id via a seen-set file so the same
# failure isn't re-reported on every poll cycle.
set -uo pipefail
ROOT=/n/holylfs06/LABS/mzitnik_lab/Users/nfishman/pool-and-split
SEEN=/tmp/grid_oom_seen.txt
touch "$SEEN"

while true; do
  # Pull recently-failed grid jobs (training: bridge_*/tf_*; evals: cpl_/spl_).
  sacct -u "$USER" --starttime="$(date -d '2 hours ago' +%Y-%m-%dT%H:%M)" \
        --format=JobID,JobName,State -P -n \
    | awk -F'|' '$3=="FAILED" && $2 ~ /^(bridge_|tf_|cpl_|spl_)/ {print $1"|"$2}' \
    | while IFS='|' read jid name; do
        grep -qFx "$jid" "$SEEN" && continue
        # Try training-log naming first, then eval-log naming.
        log=""
        for cand in "$ROOT/logs/ps_sweep_${name}_${jid}.log" \
                    "$ROOT/logs/eval_coupling_${jid}.log" \
                    "$ROOT/logs/eval_tol05_${jid}.log"; do
          [[ -f "$cand" ]] && { log="$cand"; break; }
        done
        if [[ -z "$log" ]]; then
          echo "$jid" >> "$SEEN"
          continue
        fi
        if grep -qE 'OutOfMemoryError|CUDA out of memory' "$log"; then
          # Pull the OOM allocation size and free mem from the message.
          line=$(grep -E 'CUDA out of memory' "$log" | tail -1)
          echo "OOM: ${name} (jid=${jid}) :: ${line:0:160}"
        fi
        echo "$jid" >> "$SEEN"
      done
  sleep 60
done
