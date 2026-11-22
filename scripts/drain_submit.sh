#!/bin/bash
# Submit the listed configs one at a time whenever the account queue has room (limit 10).
cd "$(dirname "$0")/.."
for c in "$@"; do
  while [ "$(squeue -u yifanwang -h | wc -l)" -ge 10 ]; do sleep 300; done
  n=$(basename "$c" .yaml)
  sbatch --job-name=TwoAxis_$n slurm/twoaxis_single.slurm "$c" >> logs/drain_submit.log 2>&1
  echo "$(date +%m-%d\ %H:%M) submitted $n" >> logs/drain_submit.log
  sleep 60
done
