#!/bin/bash
#SBATCH --account=torch_pr_63_general
#SBATCH --partition=h200_public,l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch/zz5070/miniconda3/etc/profile.d/conda.sh
conda activate py312
export PACLOCK_PROC=/scratch/zz5070/data
export PACLOCK_DATA=/scratch/zz5070/data/raw
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS=${1:?dataset required}
case "$DS" in chbmit|tuev) ;; *) exit 2;; esac
mkdir -p configs/runtime
RUNTIME_CONFIG="configs/runtime/${DS}_${SLURM_JOB_ID}.yaml"
python - "$DS" "$RUNTIME_CONFIG" <<'PYCFG'
import sys,yaml
from pathlib import Path
ds,out=sys.argv[1:];cfg=yaml.safe_load(Path(f'configs/anchored_watch/{ds}_s0.yaml').read_text())
cfg['max_hours']=10.5 if ds=='chbmit' else 6.5
Path(out).write_text(yaml.safe_dump(cfg,sort_keys=False))
PYCFG
python -u smoke/verify_anchored.py --contract-only
python -u smoke/smoke_amd_partition.py --config "$RUNTIME_CONFIG" --output "results/torch-smoke-${DS}.json"
python - "$DS" "$RUNTIME_CONFIG" <<'PY'
import sys,json,yaml,statistics
from pathlib import Path
ds=sys.argv[1];r=json.loads(Path(f'results/torch-smoke-{ds}.json').read_text());c=yaml.safe_load(Path(sys.argv[2]).read_text())
assert r['ok'] and r['warmup_excluded']>=2 and r['augmentation_path_warmups']
epochs=6 if ds=='chbmit' else 20
seconds=statistics.mean(r['steady_step_seconds'])*r['train_batches']*epochs
assert seconds<c['max_hours']*3600*.75,('insufficient runtime margin',seconds)
print('admitted',ds,'reference training-only hours',seconds/3600,flush=True)
PY
python -u -m paclock_bench.training.train --config "$RUNTIME_CONFIG" --seed 0 &
TRAIN_PID=$!
trap 'kill -TERM "$TRAIN_PID" 2>/dev/null || true; wait "$TRAIN_PID" || true' TERM INT
wait "$TRAIN_PID"
