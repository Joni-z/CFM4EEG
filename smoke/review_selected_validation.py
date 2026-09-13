"""Re-evaluate completed, kappa-selected checkpoints on validation only.

Run through slurm/smoke_gpu.slurm; never on a login node.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from torch.utils.data import DataLoader
from paclock_bench.data.datasets import WindowDataset, load_manifest
from paclock_bench.models.build import build_model
from paclock_bench.paths import expand
from paclock_bench.training.losses import build_loss
from paclock_bench.training.train import evaluate, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='run_name:seed')
    args = ap.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    assert job and torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    dest = Path('results') / ('selected-validation-' + job + '.json')
    assert not dest.exists()
    rows = []
    started = time.monotonic()
    for spec in args.runs:
        name, seed = spec.rsplit(':', 1)
        assert Path(name).name == name and seed.isdigit()
        directory = Path('runs') / name / ('seed' + seed)
        metadata = directory / 'result.json'
        checkpoint = directory / 'best.pt'
        original_hash = sha(checkpoint)
        result = json.loads(metadata.read_text())
        assert result['name'] == name and result['seed'] == int(seed)
        assert result['stopped_by'] == 'epochs' and result['epochs_run'] == 20
        cfg = result['config']
        assert result['primary_metric'] == 'cohen_kappa'
        assert cfg.get('select_metric', 'cohen_kappa') == 'cohen_kappa'
        assert not cfg.get('val_subsample')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['config'] == cfg
        assert abs(saved['best_val'] - result['best_val']) < 1e-12
        root = expand(cfg['data_root'])
        assert load_manifest(root)['created_utc'] == result['data_manifest_created']
        data = WindowDataset(root, 'val', preload=True)
        assert data.class_counts().tolist() == result['class_counts']['val']
        loader = DataLoader(data, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=2, pin_memory=True)
        set_seed(int(seed))
        model = build_model(cfg, data.shape).cuda().eval()
        model.load_state_dict(saved['model'], strict=True)
        loss, metrics, logits, labels = evaluate(model, loader, 'cuda',
            build_loss(cfg), cfg['num_classes'], cfg, return_raw=True)
        assert np.isfinite(loss) and np.isfinite(logits).all()
        assert abs(metrics['cohen_kappa'] - result['best_val']) < 1e-6
        recorded = result.get('selected_validation')
        if recorded:
            assert recorded['tag'] == saved['tag']
            for key, value in recorded['metrics'].items():
                assert abs(metrics[key] - value) < 1e-6, (spec, key)
        classes = cfg['num_classes']
        cm = np.bincount(labels.astype(int) * classes + logits.argmax(1),
                         minlength=classes * classes).reshape(classes, classes)
        recall = cm.diagonal() / cm.sum(1)
        assert abs(recall.mean() - metrics['balanced_acc']) < 1e-12
        assert sha(checkpoint) == original_hash
        row = dict(name=name, seed=int(seed), dataset=cfg['dataset'],
                   selected_tag=saved['tag'], checkpoint_sha256=original_hash,
                   metadata_sha256=sha(metadata), metrics=metrics,
                   confusion_matrix=cm.tolist(), class_recall=recall.tolist())
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, saved, data, loader, logits, labels
        gc.collect(); torch.cuda.empty_cache()
    receipt = dict(job=job, rows=rows, training_performed=False,
        test_evaluated=False, elapsed_seconds=time.monotonic() - started,
        host=os.uname().nodename, torch=torch.__version__, rocm=torch.version.hip,
        script_sha256=sha(__file__),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    with dest.open('x') as handle:
        handle.write(json.dumps(receipt, indent=2) + '\n')
    print('SELECTED_VALIDATION_COMPLETE', dest, flush=True)


if __name__ == '__main__':
    main()
