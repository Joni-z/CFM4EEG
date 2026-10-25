"""Strict loading of released EEGPT pretraining tensors into native task models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def smoke_identity(config: str, runner: str, accepted: str, cfg: dict) -> dict:
    return {'config_sha256': sha256(config), 'runner_sha256': sha256(runner),
            'loader_sha256': sha256(__file__), 'accepted_sha256': sha256(accepted),
            'pretrained_sha256': cfg['pretrained_sha256']}


def require_smoke(path: str | None, identity: dict) -> None:
    if not path:
        raise ValueError('Pretrained formal run requires --smoke-proof')
    proof = json.loads(Path(path).read_text())
    if proof.get('status') != 'passed' or proof.get('pretrained_identity') != identity:
        raise ValueError('Smoke proof does not match pretrained source/config/data/weights')


def load_released_pretraining(model, cfg: dict) -> dict:
    """Require every active target-encoder tensor; never accept random fallback.

    The native pretraining checkpoint also includes encoder/predictor modules
    absent from the downstream model. Its inactive reconstructor lacks the
    downstream-only cls_token; that single unused tensor is explicitly allowed.
    Native task heads, channel convolution, and fc_norm remain newly initialized.
    """
    import torch

    path = Path(os.path.expanduser(os.path.expandvars(cfg['pretrained_path']))).resolve()
    digest = sha256(path)
    if digest != cfg['pretrained_sha256']:
        raise ValueError('Released EEGPT checkpoint SHA256 mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    encoder = {k.removeprefix('target_encoder.'): v for k, v in state.items()
               if k.startswith('target_encoder.')}
    if set(encoder) != set(model.target_encoder.state_dict()):
        raise ValueError('Released checkpoint does not cover the complete target encoder')
    model.target_encoder.load_state_dict(encoder, strict=True)
    rec = {k.removeprefix('reconstructor.'): v for k, v in state.items()
           if k.startswith('reconstructor.')}
    missing, extra = model.reconstructor.load_state_dict(rec, strict=False)
    if set(missing) != {'cls_token'} or extra:
        raise ValueError(f'Unexpected inactive-reconstructor coverage: {missing}, {extra}')
    loaded = {**{'target_encoder.'+k: v for k, v in encoder.items()},
              **{'reconstructor.'+k: v for k, v in rec.items()}}
    current = model.state_dict()
    if not all(torch.equal(current[k].cpu(), v) for k, v in loaded.items()):
        raise RuntimeError('Loaded tensors differ from released pretraining weights')
    if not all(v.requires_grad for n, v in model.target_encoder.named_parameters()
               if not n.endswith('freqs')):
        raise RuntimeError('Native TUEG full-finetuning encoder unexpectedly frozen')
    return {'path': str(path), 'sha256': digest, 'target_encoder_tensors': len(encoder),
            'loaded_tensors': len(loaded), 'loaded_numel': sum(v.numel() for v in loaded.values()),
            'strict_target_encoder': True, 'loaded_tensor_equality': True,
            'finetuning': 'full_native_TUEG', 'inactive_missing': ['reconstructor.cls_token'],
            'task_initialized': ['head', 'chan_conv', 'fc_norm'],
            'unused_pretraining_modules': ['encoder', 'predictor']}


def configure_native_autocast(precision: str) -> None:
    """Keep upstream autocast(True) decorators compatible with explicit BF16.

    functools.partial(enabled=...) is insufficient: the released TUEV model
    passes enabled positionally in its attention decorators.
    """
    import torch
    if precision == 'fp16':
        return
    original = torch.cuda.amp.autocast
    chosen_dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    chosen_enabled = precision != 'fp32'

    def autocast(enabled=True, dtype=None, cache_enabled=True):
        return original(enabled=bool(enabled) and chosen_enabled,
                        dtype=chosen_dtype, cache_enabled=cache_enabled)

    torch.cuda.amp.autocast = autocast
