"""Re-evaluate one archived TUEV validation-best checkpoint; never train or overwrite runs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys


def select_validation_record(records):
    """Match the archival callback's strict > rule, preserving the first maximum."""
    best, previous_epoch, previous_step = None, -1, -1
    for row in records:
        epoch, step = row['epoch'], row['step']
        score = float(row['metrics']['val_cohen'])
        if (type(epoch) is not int or type(step) is not int or epoch <= previous_epoch
                or step <= previous_step or not math.isfinite(score) or not -1 <= score <= 1):
            raise ValueError('Invalid or nonchronological validation history')
        if best is None or score > float(best['metrics']['val_cohen']):
            best = row
        previous_epoch, previous_step = epoch, step
    if best is None:
        raise ValueError('No completed validation records')
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vendor', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    run, out, vendor, data = (p.resolve() for p in (args.run, args.out, args.vendor, args.data_root))
    if out.exists() or out == run or run in out.parents or out in run.parents:
        raise ValueError('Output must be a new directory outside the original run')

    import numpy as np
    import pytorch_lightning as pl
    import scipy
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
    import native_biot_entry as bridge_code

    read = lambda p: json.loads(p.read_text())
    provenance = read(run / 'native_provenance.json')
    wrapper_sha = bridge_code.sha(Path(bridge_code.__file__))
    runtime = dict(python=platform.python_version(), torch=torch.__version__,
                   lightning=pl.__version__, numpy=np.__version__, scipy=scipy.__version__)
    if wrapper_sha != provenance['wrapper_sha256']:
        raise ValueError('Evaluation bridge differs from the original run wrapper')
    if any(runtime[k] != provenance[k] for k in ('python', 'torch', 'lightning')):
        raise ValueError('Use the original recorded Python/Torch/Lightning environment')
    original = read(run / 'result.json')
    terminal = read(run / 'checkpoint_audit.json')
    cfg = provenance['config']
    if (cfg.get('dataset') != 'tuev' or cfg.get('model') not in ('biot', 'ffcl')
            or cfg.get('seed') != 0 or original.get('seed') != 0 or original['config'] != cfg
            or original['selected_checkpoint'] != terminal):
        raise ValueError('Expected a consistent completed TUEV scratch seed0 run')
    native_args = bridge_code.native_args(cfg)
    if vars(native_args) != provenance['native_args']:
        raise ValueError('Original native arguments do not match the recorded config')
    bridge_code.verify_vendor(vendor)
    if (provenance['upstream_commit'] != bridge_code.PINNED_COMMIT
            or provenance['upstream_sha256'] != bridge_code.SOURCE_SHA):
        raise ValueError('Original and current upstream source identities differ')
    bridge_code.verify_accepted(data)
    manifest_sha = bridge_code.sha(data / 'manifest.json')
    if manifest_sha != provenance['input_contract']['manifest_sha256']:
        raise ValueError('Current dataset is not the original evaluated cohort')
    records = read(run / 'validation_audit.json')['records']
    if records != original['native_validation']:
        raise ValueError('Completed result and validation audit disagree')
    selected = select_validation_record(records)
    checkpoint = run / ('audit-val-best-epoch%03d.ckpt' % selected['epoch'])
    terminal_path = Path(terminal['path'])
    if not terminal_path.is_absolute():
        terminal_path = run / terminal_path
    if bridge_code.sha(terminal_path) != terminal['sha256']:
        raise ValueError('Original terminal checkpoint changed')
    selected_sha = bridge_code.sha(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu')
    if saved['epoch'] != selected['epoch'] or saved['global_step'] != selected['step']:
        raise ValueError('Archived checkpoint epoch/step differs from validation selection')
    if not torch.cuda.is_available():
        raise RuntimeError('A free CUDA device is required; this tool does not allocate or kill jobs')

    sys.path.insert(0, str(vendor))
    spec = importlib.util.spec_from_file_location('biot_native_eval_source', vendor / 'run_multiclass_supervised.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)  # Definitions only: __name__ is not __main__.
    for name in ('model', 'utils'):
        if vendor not in Path(sys.modules[name].__file__).resolve().parents:
            raise RuntimeError('Imported a non-vendor module: ' + name)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    if native_args.model == 'BIOT':
        model = native.BIOTClassifier(n_classes=6, n_channels=16, n_fft=200, hop_length=100)
    else:
        model = native.FFCL(in_channels=16, n_classes=6, fft=200, steps=20,
                            sample_length=1000, shrink_steps=20)
    module = native.LitModel_finetune(native_args, model)
    module.load_state_dict(saved['state_dict'], strict=True)
    module.float().cuda().eval()
    bridge = bridge_code.NpyPickleBridge(data, 'TUEV')
    native_utils = sys.modules['utils']
    native_utils.open = bridge.open

    def evaluate(split):
        loader = torch.utils.data.DataLoader(bridge.native_dataset(native_utils, split),
                                            batch_size=native_args.batch_size, shuffle=False,
                                            drop_last=False, num_workers=0)
        labels, logits = [], []
        with torch.no_grad():
            for x, y in loader:
                z = module.model(x.cuda().float())
                if z.ndim != 2 or z.shape[1] != 6 or not torch.isfinite(z).all():
                    raise RuntimeError('Invalid evaluation logits')
                labels.append(y.numpy())
                logits.append(z.cpu().numpy())
        y, z = np.concatenate(labels), np.concatenate(logits)
        if not np.array_equal(y, bridge.arrays[split][1]):
            raise RuntimeError('Evaluation row order/coverage mismatch')
        pred = z.argmax(axis=1)
        metrics = dict(accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred),
                       cohen_kappa=cohen_kappa_score(y, pred), weighted_f1=f1_score(y, pred, average='weighted'))
        np.savez_compressed(out / (split + '_scores.npz'), labels=y, logits=z, row_indices=np.arange(len(y)))
        return {k: float(v) for k, v in metrics.items()}

    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='validation_pending', evaluation_kind='posthoc_validation_selected',
                   selection_rule='first_strict_maximum_validation_cohen_kappa', training_performed=False,
                   source_run=str(run), model=cfg['model'], seed=0, config=cfg,
                   selected_validation=selected, selected_checkpoint=str(checkpoint),
                   selected_checkpoint_sha256=selected_sha,
                   original_terminal_checkpoint=dict(terminal, absolute_path=str(terminal_path)),
                   original_terminal_metrics=original['native_metrics'], manifest_sha256=manifest_sha,
                   upstream_commit=provenance['upstream_commit'], upstream_sha256=provenance['upstream_sha256'],
                   script_sha256=bridge_code.sha(Path(__file__)), wrapper_sha256=wrapper_sha, runtime=runtime,
                   dtype='float32', evaluation_workers=0,
                   source_result_sha256=bridge_code.sha(run / 'result.json'),
                   validation_history_sha256=bridge_code.sha(run / 'validation_audit.json'))
    bridge_code.atomic_json(out / 'provenance.json', receipt)
    validation = evaluate('val')
    receipt['validation'] = validation
    if not math.isclose(validation['cohen_kappa'], selected['metrics']['val_cohen'], abs_tol=1e-6, rel_tol=0):
        receipt['status'] = 'validation_mismatch_test_not_evaluated'
        bridge_code.atomic_json(out / 'provenance.json', receipt)
        raise RuntimeError('Restored validation kappa mismatch; test was not evaluated')
    test = evaluate('test')
    receipt.update(status='completed', test=test,
                   prediction_sha256={s: bridge_code.sha(out / (s + '_scores.npz')) for s in ('val', 'test')})
    bridge_code.atomic_json(out / 'result.json', receipt)
    bridge_code.atomic_json(out / 'provenance.json', receipt)
    print(json.dumps(dict(model=cfg['model'], selected_epoch=selected['epoch'], validation=validation, test=test)))


if __name__ == '__main__':
    main()
