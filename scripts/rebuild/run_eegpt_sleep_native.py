"""Pinned EEGPT Sleep-EDF Lightning task on the audited shared patient cohort.

The source class is compiled directly, stopping before its top-level ten-fold
experiment loop. Native model, optimizer subset, OneCycle schedule, interpolation
and objective are unchanged. The shared split is an explicit protocol adaptation.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_native_class(path: Path):
    """Compile only definitions above the upstream experiment, never run folds."""
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = []
    for node in tree.body:
        if not (isinstance(node, ast.ImportFrom) and node.module == 'utils'):
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == 'LitEEGPTCausal':
            break
    else:
        raise RuntimeError('Pinned EEGPT Sleep-EDF class not found')
    import torch
    # Compile the exact interpolation function without utils.py's unrelated
    # BCI dataset imports (which would require MNE/LoadData at runtime).
    util_tree = ast.parse((path.parent/'utils.py').read_text())
    interpolation = next(n for n in util_tree.body if isinstance(n, ast.FunctionDef) and n.name=='temporal_interpolation')
    ns = {'__file__': str(path), '__name__': 'cfft_native_eegpt_sleep', 'torch':torch}
    exec(compile(ast.Module(body=[interpolation], type_ignores=[]), str(path.parent/'utils.py'), 'exec'), ns)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config'); p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int); p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--resume'); p.add_argument('--max-hours', type=float, default=0)
    a = p.parse_args()
    import numpy as np
    import torch, yaml, pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset
    from sklearn.preprocessing import scale
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from paclock_bench.paths import vendored
    from paclock_bench.data.datasets import build_dataloaders
    from paclock_bench.training.recipe_runtime import provenance
    from paclock_bench.training.metrics import compute_metrics
    cfg = yaml.safe_load(Path(a.config).read_text())
    if cfg['dataset'] != 'sleepedf' or not cfg.get('pretrained'):
        raise ValueError('This native task is the released pretrained Sleep-EDF decoder')
    if cfg['batch_size'] != 32 or cfg['epochs'] != 40:
        raise ValueError('Native Sleep-EDF recipe requires batch32 and40 epochs')
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Native Sleep-EDF task requires one allocated CUDA GPU')
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True, exist_ok=True)
    source = Path(vendored('eegpt'))/'downstream/finetune_EEGPT_SleepEDF.py'
    sys.path.insert(0, str(source.parent))
    native = load_native_class(source)
    seed = cfg.get('seed', 0) if a.seed is None else a.seed
    native['seed_torch'](seed)
    torch.set_float32_matmul_precision('medium')
    train0, val0, test0, info = build_dataloaders(cfg)
    if tuple(info['input_shape']) != (2, 3000):
        raise ValueError(f'Native Sleep-EDF expects two30-second100Hz channels, got {info["input_shape"]}')

    class EpochStandardized(Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            # Native prepare_sleep.py uses sklearn.scale channel-wise per epoch.
            # This removes the shared cache's train-only affine normalization.
            x = scale(x.numpy().astype(np.float64), axis=-1).astype(np.float32)
            return torch.from_numpy(x), y

    def loader(base, training):
        return DataLoader(EpochStandardized(base.dataset), batch_size=32,
                          shuffle=training, num_workers=cfg.get('num_workers', 8),
                          drop_last=False, pin_memory=True)
    train, val, test = loader(train0, True), loader(val0, False), loader(test0, False)
    native.update(max_epochs=40, steps_per_epoch=len(train), max_lr=4e-4)
    original_weights = Path(vendored('eegpt'))/'checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt'
    original = torch.load(original_weights, map_location='cpu', weights_only=False)
    state = {k:v for k,v in original['state_dict'].items() if k.startswith('target_encoder.')}
    if not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise RuntimeError('Released EEGPT target_encoder tensor payload missing')
    # Upstream torch.load has no weights_only argument. A tensor-only bridge
    # retains exactly the released target encoder and works with modern Torch.
    bridge = out/'target_encoder_payload.pt'
    if bridge.exists():
        prior=torch.load(bridge,map_location='cpu',weights_only=True)['state_dict']
        if prior.keys()!=state.keys() or any(not torch.equal(prior[k],state[k]) for k in state):
            raise RuntimeError('Existing target-encoder bridge differs from released weights')
    else:
        temporary=bridge.with_suffix('.tmp')
        torch.save({'state_dict': state},temporary);os.replace(temporary,bridge)
    model = native['LitEEGPTCausal'](load_path=str(bridge))
    for parameter in model.target_encoder.parameters(): parameter.requires_grad_(False)
    # Native optimizer already excludes these parameters. Disabling their
    # parameter gradients preserves input gradients to chan_conv and updates.
    callback = ModelCheckpoint(dirpath=str(out), filename='terminal', monitor=None,
                               save_last=True, save_top_k=1, save_weights_only=False)

    class Audit(Callback):
        def __init__(self): self.steps=0; self.initial=None
        def on_train_start(self, trainer, module):
            self.initial = module.linear_probe2.weight.detach().cpu().clone()
        def on_after_backward(self, trainer, module):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise RuntimeError('Missing or nonfinite native Sleep-EDF gradients')
            if module.chan_conv.weight.grad is None:
                raise RuntimeError('Frozen encoder blocked gradients to channel projection')
            self.steps+=1
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss=outputs.get('loss') if isinstance(outputs,dict) else outputs
            if loss is not None and not torch.isfinite(loss).all():
                raise RuntimeError('Nonfinite native Sleep-EDF loss')

    audit=Audit()
    trainer=pl.Trainer(default_root_dir=str(out), accelerator='gpu', devices=1,
        precision='16-mixed', max_epochs=40, max_steps=a.smoke_steps or -1,
        callbacks=[callback,audit], logger=CSVLogger(str(out),name='logs'),
        max_time=None if not a.max_hours else {'seconds':int(a.max_hours*3600)},
        num_sanity_val_steps=0 if a.smoke_steps else 2,
        limit_val_batches=0 if a.smoke_steps else 1.0)
    receipt={'config':cfg,'seed':seed,'provenance':provenance(a.config,cfg,info),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'native_utils_sha256':hashlib.sha256((source.parent/'utils.py').read_bytes()).hexdigest(),
        'weights_sha256':hashlib.sha256(original_weights.read_bytes()).hexdigest(),
        'native_class':'LitEEGPTCausal','selection':'terminal','global_batch':32,
        'input_bridge':'Per-channel per-epoch sklearn.scale, then native interpolation100to256Hz and native common-average reference.',
        'protocol':'Shared fixed patient cohort and0.3–35Hz bandpass; original uses ten-fold training and30Hz lowpass. This is not exact published preprocessing.',
        'encoder_gradient_compatibility':'Native optimizer excludes encoder; requires_gradFalse saves memory and preserves channel-projection input gradients.',
        'test_evaluated':False}
    (out/'provenance.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
    trainer.fit(model,train_dataloaders=train,val_dataloaders=val,ckpt_path=a.resume)
    trainer.save_checkpoint(str(out/'last.ckpt'))
    saved=torch.load(out/'last.ckpt',map_location='cpu',weights_only=False)['state_dict']
    model.to('cuda').eval()
    x=next(iter(val))[0].to('cuda')
    with torch.no_grad(): before=model(x)[1].cpu()
    model.load_state_dict(saved,strict=True)
    with torch.no_grad(): after=model(x)[1].cpu()
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    if a.smoke_steps:
        if trainer.global_step<a.smoke_steps or audit.steps<a.smoke_steps:
            raise RuntimeError('Native smoke did not finish all requested updates')
        if torch.equal(audit.initial,model.linear_probe2.weight.detach().cpu()):
            raise RuntimeError('Native optimizer did not update classifier')
        (out/'smoke.json').write_text(json.dumps(dict(receipt,status='passed',passed=True,
            checkpoint_reload='passed',strict_reload_identical=True,steps=trainer.global_step),indent=2,default=str)+'\n')
        return
    required=40*len(train)
    if trainer.global_step<required:
        (out/'paused.json').write_text(json.dumps({'global_step':trainer.global_step,
            'required_steps':required,'resume':str(out/'last.ckpt')},indent=2)+'\n')
        raise SystemExit(75)
    labels=[];probs=[]
    with torch.no_grad():
        for x,y in test:
            pred=model(x.to('cuda'))[1].float().softmax(-1).cpu().numpy()
            probs.append(pred);labels.append(np.asarray(y))
    y=np.concatenate(labels);prob=np.concatenate(probs)
    metrics=compute_metrics(y,prob,5)
    np.savez_compressed(out/'test_predictions.npz',y_true=y,y_prob=prob)
    result={'cfg':cfg,'seed':seed,'metrics':metrics,'checkpoint_selection':'terminal',
        'stopped_by':'epochs','epochs_run':40,'test_evaluated':True,'provenance':receipt}
    (out/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')


if __name__=='__main__': main()
