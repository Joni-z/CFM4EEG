"""Run pinned BIOT/FFCL entrypoints, adapting only accepted NPY input and audit I/O.

The upstream LightningModule, supervised(), optimizer, Trainer, early stopping,
and test(ckpt_path='best') execute directly. A recorded one-line patch registers
an observational callback. No upstream file is rewritten. Requires the separate
Python3.9/Torch1.13.1/Lightning1.6.4 environment.
"""
from __future__ import annotations

import argparse
import builtins
import difflib
import hashlib
import importlib.util
import io
import json
import os
import pickle
import platform
import sys
import types
from functools import partial
from pathlib import Path

# Torch1.13 rejects the modern allocator option inherited from worker scripts.
# Device visibility and all training settings remain unchanged.
ALLOCATOR_COMPATIBILITY = os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
PINNED_COMMIT = 'd138e32634e52ae9fa6ec98ac9c4087b14ca869a'
SOURCE_SHA = {
    'run_multiclass_supervised.py': '871c8d39dec296d9b12020bf24b3b5ab703a327aaddfc5aa6a78f374f678be01',
    'run_binary_supervised.py': 'a87c0ab90063255e1eb15d159645ee9e2f94a6fde26c8fcd6bc7eba53a3512e0',
    'utils.py': 'db4702ec8db3fa54f29097da39ceffd67c1a21ee605f0716d4bd4c8e172c2d4b',
    'model/__init__.py': 'd0d76cfe8736cdc8c283aaae28b714aab0684cdeafee0844f5ed42395ea0f837',
    'model/biot.py': '2bba53052a93a005033b27ef89f20037fd7894780ce9da130b823c7c73029e01',
    'model/cnn_transformer.py': 'e6d34e0204bebbe5495d2c41ddce1e650991d6bea893a7883db3cc9218d4f4b6',
    'model/contrawr.py': '82eba543ee54cbbc6ac29475e7b8f97d6844c58909013add24f9ff8c73dfa771',
    'model/ffcl.py': '0199d092d6431ebdafebe35f5147e546452502964fefa0d21eace21d9e16bfe3',
    'model/sparcnet.py': '79a422ee04a850131e8bae2b584206b271633c09b6c09207d9a9a2a9835b3e33',
    'model/st_transformer.py': 'fff6d619843e4c33f390102a039c626e02f98b4dd9f571f146ef9e96e34b9d91',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def verify_vendor(vendor: Path) -> None:
    for relative, expected in SOURCE_SHA.items():
        if sha(vendor / relative) != expected:
            raise ValueError('Upstream source differs from pinned commit: ' + relative)


def verify_accepted(root: Path) -> dict:
    manifest = json.loads((root / 'manifest.json').read_text())
    receipt = json.loads((root / 'ACCEPTED.json').read_text())
    if receipt.get('status') != 'accepted' or receipt.get('manifest_sha256') != sha(root / 'manifest.json'):
        raise ValueError('Dataset acceptance / manifest mismatch')
    expected = {f'{s}_{k}.npy' for s in ('train', 'val', 'test') for k in ('signals', 'labels')}
    if not expected.issubset(receipt.get('files', {})):
        raise ValueError('Acceptance does not cover all six NPY files')
    for name, stat in receipt['files'].items():
        actual = (root / name).stat()
        if actual.st_size != stat['size'] or ('mtime_ns' in stat and actual.st_mtime_ns != stat['mtime_ns']):
            raise ValueError('Accepted dataset changed: ' + name)
    p = manifest['protocol']
    expected_norm = 'q95' if manifest['dataset'] == 'chbmit' else 'div100'
    if p.get('sample_rate') != 200 or p.get('normalization') != expected_norm or len(p['channels']) != 16:
        raise ValueError('Expected shared200Hz16-channel arrays with declared normalization')
    return manifest


class NpyPickleBridge:
    """Expose NPY rows as native pickle records; native __getitem__ remains intact.

    MNE Raw[:] / get_data() upstream produce volts. Our accepted arrays contain
    microvolts /100, so multiply by1e-4 before native q95(+1e-8). Already200Hz
    arrays keep their200Hz/5s shape. The native TUEV resample branch is retained:
    it maps1000 samples to1000 samples (FFT roundtrip), without changing duration.
    """
    def __init__(self, root: Path, dataset: str, smoke: bool = False):
        self.dataset = dataset
        self.arrays = {}
        self.scales = {}
        for split in ('train', 'val') if smoke else ('train', 'val', 'test'):
            x = np.load(root / (split + '_signals.npy'), mmap_mode='r')
            y = np.load(root / (split + '_labels.npy'), mmap_mode='r')
            length = 1000 if dataset == 'TUEV' else 2000
            classes = 6 if dataset == 'TUEV' else 2
            if x.ndim != 3 or x.shape[1:] != (16, length) or x.dtype != np.float32 or len(x) != len(y):
                raise ValueError('Invalid native bridge layout: ' + split)
            if y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= classes):
                raise ValueError('Invalid native labels: ' + split)
            self.arrays[split] = (x, y)
            if dataset == 'CHB_MIT':
                scale = np.load(root / (split + '_q95_denominators.npy'), mmap_mode='r')
                if scale.shape != (len(x), 16, 1) or scale.dtype != np.float64:
                    raise ValueError('Invalid CHB inverse-Q95 sidecar')
                self.scales[split] = scale
        self.prefix = str(root / '__native_npy_records__') + os.sep

    def open(self, path, mode='r', *args, **kwargs):
        path = os.fspath(path)
        if not path.startswith(self.prefix):
            return builtins.open(path, mode, *args, **kwargs)
        if mode != 'rb' or args or kwargs:
            raise ValueError('Native row adapter supports read-only binary access')
        split, filename = path[len(self.prefix):].split(os.sep)
        index = int(Path(filename).stem)
        x, y = self.arrays[split]
        signal = np.asarray(x[index], dtype=np.float64)
        signal = signal * self.scales[split][index] if self.dataset == 'CHB_MIT' else signal * 1e-4
        record = ({'signal': signal, 'label': np.array([int(y[index]) + 1])}
                  if self.dataset == 'TUEV' else {'X': signal, 'y': int(y[index])})
        return io.BytesIO(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))

    def native_dataset(self, native_utils, split: str, files=None):
        files = list(map(str, range(len(self.arrays[split][1])))) if files is None else files
        cls = {'TUEV': native_utils.TUEVLoader, 'TUAB': native_utils.TUABLoader, 'CHB_MIT': native_utils.CHBMITLoader}[self.dataset]
        result = cls(self.prefix + split, files, sampling_rate=200)
        return result


def prepare_data(args, *, bridge, native_utils, seed: int, smoke_steps: int):
    # Exact native seeding calls, with the explicitly requested seed0 substitution.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    native_utils.open = bridge.open
    train_files = list(map(str, range(len(bridge.arrays['train'][1]))))
    if bridge.dataset == 'TUAB':
        np.random.shuffle(train_files)  # retained from native TUAB prepare function
    train = bridge.native_dataset(native_utils, 'train', train_files)
    val = bridge.native_dataset(native_utils, 'val')
    if smoke_steps:
        # Distinct smoke output; no test arrays were opened, and no test is executed.
        train = torch.utils.data.Subset(train, range(min(len(train), args.batch_size * smoke_steps)))
        labels = bridge.arrays['val'][1]
        indices = np.concatenate([np.flatnonzero(labels == label)[:args.batch_size] for label in np.unique(labels)])
        val = torch.utils.data.Subset(val, np.sort(indices).tolist())
        test = val
    else:
        test = bridge.native_dataset(native_utils, 'test')
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, drop_last=True, **common)
    test_loader = torch.utils.data.DataLoader(test, shuffle=False, **common)
    val_loader = torch.utils.data.DataLoader(val, shuffle=False, **common)
    if not len(train_loader):
        raise ValueError('No complete native training batch')
    print('Native dataset bridge:', len(train), len(val), len(test), flush=True)
    return train_loader, test_loader, val_loader


def native_args(cfg: dict) -> argparse.Namespace:
    dataset = {'chbmit': 'CHB_MIT'}.get(str(cfg['dataset']), str(cfg['dataset']).upper())
    model = {'biot': 'BIOT', 'ffcl': 'FFCL'}.get(cfg['model'])
    pretrained = cfg.get('pretrained', False)
    if type(pretrained) is not bool:
        raise ValueError('pretrained must be an explicit boolean')
    if dataset not in ('TUEV', 'TUAB', 'CHB_MIT') or model is None:
        raise ValueError('Native wrapper supports reviewed TUEV/TUAB/CHB BIOT/FFCL only')
    checkpoint = cfg.get('checkpoint')
    if bool(checkpoint) != pretrained or (pretrained and model != 'BIOT'):
        raise ValueError('Exactly one released BIOT checkpoint is required for pretrained; scratch forbids it')
    if pretrained:
        checkpoint = Path(os.path.expanduser(os.path.expandvars(checkpoint))).resolve()
        if (not checkpoint.is_file() or not cfg.get('checkpoint_sha256')
                or sha(checkpoint) != cfg['checkpoint_sha256']):
            raise ValueError('Released pretrained checkpoint missing or hash mismatch')
    expected = dict(optimizer='adam', lr=.001, weight_decay=1e-5, epochs=100,
                    patience=5, train_drop_last=True, label_smoothing=0., amp=False)
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError('Config differs from native recipe: ' + key)
    for key in ('scheduler', 'grad_clip'):
        if cfg.get(key) not in (None, 'none'):
            raise ValueError('Native recipe has no ' + key)
    if cfg.get('checkpoint_selection') != 'terminal' or cfg.get('loader_input_multiplier') != (1.0 if dataset == 'CHB_MIT' else 1e-4):
        raise ValueError('Native code uses terminal checkpoint and volt-unit q95; config must declare both')
    classes = 6 if dataset == 'TUEV' else 1
    if cfg['num_classes'] != (6 if dataset == 'TUEV' else 2) or cfg['sample_rate'] != 200:
        raise ValueError('Native task shape mismatch')
    capacity = 18 if pretrained else 16
    if model == 'BIOT' and cfg.get('model_kwargs', {}).get('encoder_channels') != capacity:
        raise ValueError('BIOT channel capacity must match the reviewed scratch/released checkpoint constructor')
    batch = 128 if dataset == 'TUEV' and model == 'BIOT' else 512
    if cfg['batch_size'] != batch or int(cfg.get('num_workers', 32)) < 1:
        raise ValueError('Native batch/workers contract failed')
    return argparse.Namespace(dataset=dataset, model=model, in_channels=capacity,
                              n_classes=classes, sampling_rate=200, token_size=200,
                              hop_length=100, sample_length=5 if dataset == 'TUEV' else 10,
                              batch_size=batch, num_workers=int(cfg['num_workers']),
                              epochs=100, lr=.001, weight_decay=1e-5,
                              pretrain_model_path=str(checkpoint) if pretrained else '')


class SmokeComplete(Exception):
    """Exit only after native fit and checkpoint reload, before any test batch."""


def audit_callback(pl, *, cfg, out, smoke_steps, provenance):
    class NativeAudit(pl.Callback):
        def __init__(self):
            self.validation = []
            self.test_outputs = []
            self.initial = None
            self.backwards = 0
            self.batches = 0
            self.checkpoint = None
            self.best_validation = float('-inf')

        def on_fit_start(self, trainer, module):
            if not cfg.get('pretrained', False):
                return
            path = Path(module.args.pretrain_model_path)
            if sha(path) != cfg['checkpoint_sha256']:
                raise RuntimeError('Released checkpoint changed before training')
            expected = torch.load(path, map_location='cpu')
            actual = module.model.biot.state_dict()
            if (set(actual) != set(expected)
                    or any(not torch.equal(actual[k].detach().cpu(), expected[k]) for k in expected)):
                raise RuntimeError('Native pretrained encoder did not exactly load all released tensors')
            atomic_json(out / 'pretrained_load_audit.json', dict(
                status='passed', checkpoint=str(path), sha256=sha(path),
                encoder_state_tensors=len(actual), encoder_state_values=sum(v.numel() for v in actual.values()),
                channel_capacity=18, actual_input_channels=16,
                constructor_reference='Pinned upstream run_example.py BIOT-pretrain-SHHS+PREST',
                task_classifier='randomly initialized; not present in released encoder checkpoint',
                strict_encoder_load=True, all_loaded_tensors_equal=True))

        def on_train_start(self, trainer, module):
            if smoke_steps:
                self.initial = {n: p.detach().cpu().clone() for n, p in module.named_parameters() if p.requires_grad}

        def on_after_backward(self, trainer, module):
            if smoke_steps:
                gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                    raise FloatingPointError('Missing/nonfinite native gradients')
            self.backwards += 1

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
            self.batches += 1
            if smoke_steps and self.batches >= smoke_steps:
                trainer.should_stop = True  # smoke only; formal training never changes this flag

        def on_validation_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith('val_')}
            self.validation.append(dict(epoch=int(trainer.current_epoch), step=int(trainer.global_step), metrics=metrics))
            atomic_json(out / 'validation_audit.json', {'records': self.validation})
            key = 'val_cohen' if cfg['num_classes'] == 6 else 'val_auroc'
            if not smoke_steps and metrics[key] > self.best_validation:
                self.best_validation = metrics[key]
                # Independent audit copy; does not touch ModelCheckpoint's path,
                # monitor, or native test selection. Every improving epoch retained.
                archive = out / ('audit-val-best-epoch%03d.ckpt' % trainer.current_epoch)
                if archive.exists():
                    raise FileExistsError(archive)
                trainer.save_checkpoint(str(archive))

        def on_test_start(self, trainer, module):
            path = Path(trainer.checkpoint_callback.best_model_path)
            checkpoint = torch.load(path, map_location='cpu')
            actual = module.state_dict()
            saved = checkpoint['state_dict']
            if set(actual) != set(saved) or any(not torch.equal(actual[k].cpu(), saved[k]) for k in saved):
                raise RuntimeError('Native test did not strictly reload the selected checkpoint')
            self.checkpoint = dict(path=str(path), sha256=sha(path), epoch=int(checkpoint['epoch']),
                                   global_step=int(checkpoint['global_step']), monitor=trainer.checkpoint_callback.monitor)
            atomic_json(out / 'checkpoint_audit.json', self.checkpoint)
            if smoke_steps:
                if self.backwards < smoke_steps or int(trainer.global_step) < smoke_steps:
                    raise RuntimeError('Native smoke ran too few optimizer updates')
                if not any(not torch.equal(p.detach().cpu(), self.initial[n]) for n, p in module.named_parameters() if n in self.initial):
                    raise RuntimeError('Native smoke changed no trainable parameter')
                atomic_json(out / 'smoke.json', dict(status='passed', checkpoint_reload='passed',
                            config=cfg, test_evaluated=False, provenance=provenance,
                            optimizer_updates=int(trainer.global_step), native_checkpoint=self.checkpoint,
                            native_validation=self.validation, native_source_sha256=SOURCE_SHA))
                raise SmokeComplete()

        def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx, unused=0):
            self.test_outputs.append(outputs)  # exact native test_step arrays, no extra forward pass

        def on_test_end(self, trainer, module):
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith('test_')}
            scores = np.concatenate([np.asarray(v[0]) for v in self.test_outputs])
            labels = np.concatenate([np.asarray(v[1]).reshape(-1) for v in self.test_outputs])
            np.savez_compressed(out / 'test_scores.npz', scores=scores, labels=labels)
            # Keep native names. Additional reporting derives from the same predictions.
            if cfg['num_classes'] == 6:
                from sklearn.metrics import balanced_accuracy_score
                metrics['test_bacc'] = float(balanced_accuracy_score(labels, scores.argmax(axis=1)))
            atomic_json(out / 'result.json', dict(config=cfg, seed=0, native_metrics=metrics,
                        native_validation=self.validation, selected_checkpoint=self.checkpoint,
                        checkpoint_selection='upstream_default_unmonitored_terminal',
                        native_source_sha256=SOURCE_SHA, provenance=provenance,
                        n_params_M=sum(p.numel() for p in module.model.parameters()) / 1e6))
    return NativeAudit()


def load_native(vendor: Path, dataset: str, callback):
    filename = 'run_multiclass_supervised.py' if dataset == 'TUEV' else 'run_binary_supervised.py'
    original = (vendor / filename).read_text()
    before = 'callbacks=[early_stop_callback],'
    if original.count(before) != 1:
        raise ValueError('Audited callback injection anchor changed')
    patched = original.replace(before, 'callbacks=[early_stop_callback, _cfft_audit_callback],')
    if dataset == 'CHB_MIT':
        anchor = '        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)'
        if original.count(anchor) != 1:
            raise ValueError('Native binary task dispatch anchor changed')
        patched = patched.replace(anchor, anchor + '\n\n    elif args.dataset == \"CHB_MIT\":\n        train_loader, test_loader, val_loader = prepare_CHB_MIT_dataloader(args)')
    diff = ''.join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                      fromfile=filename, tofile=filename + '.audit-only'))
    module = types.ModuleType('_pinned_biot_entrypoint')
    module.__file__ = str(vendor / filename)
    module._cfft_audit_callback = callback
    sys.modules[module.__name__] = module
    exec(compile(patched, str(vendor / filename), 'exec'), module.__dict__)
    return module, diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--vendor', help='Override vendor path for source verification/testing')
    args = parser.parse_args()
    if args.seed != 0 or args.smoke_steps < 0:
        raise ValueError('This run plan is seed0 only')
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    native = native_args(cfg)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing native run: ' + str(out))
    root = Path(os.path.expandvars(cfg['data_root'])).resolve()
    manifest = verify_accepted(root)
    if cfg.get('require_rebuild_ready'):
        ready = json.loads((root/'READY.json').read_text())
        if ready.get('manifest_sha256') != sha(root/'manifest.json'):
            raise ValueError('Changed-data build not READY for this manifest')
    from paclock_bench.paths import vendored
    vendor = Path(args.vendor or vendored('biot')).resolve()
    if vendor != Path(vendored('biot')).resolve():
        raise ValueError('Execution vendor must match the provenance-bound vendored BIOT path')
    verify_vendor(vendor)
    sys.path.insert(0, str(vendor))
    import pytorch_lightning as pl
    if pl.__version__ != '1.6.4' or torch.__version__.split('+')[0] != '1.13.1':
        raise RuntimeError('Use the dedicated native Torch1.13.1/Lightning1.6.4 environment')
    # These Slurm process variables describe allocation, not native one-GPU rank.
    # JOB_ID remains available for logging; no allocation or process is modified.
    for key in ('SLURM_NTASKS', 'SLURM_NTASKS_PER_NODE', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NODEID'):
        os.environ.pop(key, None)
    from paclock_bench.training.recipe_runtime import provenance as make_provenance
    contract = dict(manifest_sha256=sha(root / 'manifest.json'), sample_rate=200,
                    native_input_units=('uV' if native.dataset == 'CHB_MIT' else 'V'),
                    stored_normalization=manifest['protocol']['normalization'],
                    multiplier=(None if native.dataset == 'CHB_MIT' else 1e-4),
                    inverse_q95_sidecar=native.dataset == 'CHB_MIT')
    prov = make_provenance(str(config_path), cfg, {'input_contract': contract})
    callback = audit_callback(pl, cfg=cfg, out=out, smoke_steps=args.smoke_steps, provenance=prov)
    module, diff = load_native(vendor, native.dataset, callback)
    import utils as native_utils
    if Path(native_utils.__file__).resolve() != vendor / 'utils.py':
        raise RuntimeError('Wrong module named utils was imported')
    bridge = NpyPickleBridge(root, native.dataset, bool(args.smoke_steps))
    prepare = partial(prepare_data, bridge=bridge, native_utils=native_utils, seed=0, smoke_steps=args.smoke_steps)
    setattr(module, 'prepare_' + native.dataset + '_dataloader', prepare)
    (out / 'upstream-audit-only.patch').write_text(diff)
    atomic_json(out / 'native_provenance.json', dict(upstream_commit=PINNED_COMMIT,
                upstream_sha256=SOURCE_SHA, wrapper_sha256=sha(Path(__file__)), cli=sys.argv,
                native_args=vars(native), config=cfg, input_contract=contract,
                removed_incompatible_allocator_configuration=ALLOCATOR_COMPATIBILITY,
                python=platform.python_version(), torch=torch.__version__, lightning=pl.__version__,
                adaptation=['accepted shared cohort/preprocessing retained; not native preprocessing reproduction',
                            'native loaders consume virtual pickle records: V for TUH, physical uV for CHB, using exact inverse-Q95 sidecars',
                            'native TUEV resample branch retained: accepted1000-point rows map to1000points',
                            'native hardcoded seed4523(TUEV)/12345(TUAB) explicitly overridden with user seed0',
                            'callback registration patched; CHB_MIT additionally selects existing native CHB loader; model/optimizer/Trainer/fit/test untouched'],
                smoke_steps=args.smoke_steps, provenance=prov))
    original_cwd = Path.cwd()
    try:
        os.chdir(out)  # original TensorBoardLogger paths stay under this run directory
        module.supervised(native)
    except SmokeComplete:
        if not args.smoke_steps:
            raise
    finally:
        os.chdir(original_cwd)
    expected = out / ('smoke.json' if args.smoke_steps else 'result.json')
    if not expected.exists():
        raise RuntimeError('Native entrypoint returned without an audit receipt')


if __name__ == '__main__':
    main()
