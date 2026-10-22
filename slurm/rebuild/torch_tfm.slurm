#!/bin/bash
#SBATCH --job-name=cfft-tfm
#SBATCH --account=torch_pr_63_general
#SBATCH --partition=h200_public,l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
RELEASE=${1:?immutable source release}
PYTHON=${2:?absolute TFM environment python}
CONFIG=${3:?config relative to release}
STAGE=${4:?tokenizer, mtp or finetune}
: "${CFM_REBUILD_ROOT:?external output root}"
: "${PACLOCK_PROC:?parent of processed/}"
export CFM_REBUILD_ROOT PACLOCK_PROC PYTHONNOUSERSITE=1
export PACLOCK_CKPT=${PACLOCK_CKPT:-$CFM_REBUILD_ROOT/ckpt}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
cd "$RELEASE"
case "$STAGE" in
  tokenizer|mtp) expected_gpus=2 ;;
  finetune) expected_gpus=1 ;;
  *) echo "Invalid TFM stage: $STAGE" >&2; exit 2 ;;
esac
"$PYTHON" -c \
  'import sys,torch; actual=torch.cuda.device_count(); expected=int(sys.argv[1]); assert actual==expected, f"TFM allocation mismatch: expected {expected} GPUs, visible {actual}"' \
  "$expected_gpus"
set +e
"$PYTHON" -u scripts/rebuild/run_tfm_chain.py "$CONFIG" \
  --stage "$STAGE" --seed 0 --mode auto --resume auto --max-hours 46
status=$?
set -e
if [[ $status == 75 ]]; then
  if [[ ${SLURM_RESTART_COUNT:-0} -ge 5 ]]; then
    echo 'Stage preserved but still incomplete after six allocation windows; requires review.'
    exit 75
  fi
  echo 'Full checkpoint saved; requeueing this same stage before releasing dependents.'
  scontrol requeue "$SLURM_JOB_ID"
  exit 75
fi
exit "$status"
