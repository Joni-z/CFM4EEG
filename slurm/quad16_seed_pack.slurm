#!/bin/bash
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
case "$SLURM_JOB_PARTITION" in
 mi2104x) cards=4 ;;
 mi2508x) cards=8 ;;
 *) exit 2 ;;
esac
[[ $# -eq $cards ]] || { echo "Expected exactly $cards configurations"; exit 2; }
cpus=$((128/cards))
pids=()
stop() { for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null || true; done; }
trap stop TERM INT
idx=0
for cfg in "$@"; do
 [[ "$cfg" =~ _s([012])\.yaml$ ]] || exit 2
 seed=${BASH_REMATCH[1]}
 srun --exclusive -N1 -n1 -c "$cpus" --export=ALL,QUAD_GPU="$idx"   --output="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu${idx}.out"   bash slurm/quad16_seed_amd.slurm "$cfg" "$seed" &
 pids+=("$!")
 idx=$((idx+1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit "$rc"
