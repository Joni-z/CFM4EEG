"""Input contracts shared by rebuild loaders; no silent changes of physical units."""
import hashlib
import json
from pathlib import Path
import numpy as np


def verify_arrays(signals, labels, signal_path, label_path):
    if len(signals) != len(labels):
        raise ValueError(f'signal/label row mismatch: {signal_path}')
    if signals.ndim not in (3,4) or signals.dtype != np.float32:
        raise ValueError(f'unexpected signal layout: {signal_path}: {signals.shape}, {signals.dtype}')
    if labels.dtype.kind not in 'iu':
        raise ValueError(f'noninteger labels: {label_path}')
    for arr,path in ((signals,signal_path),(labels,label_path)):
        # A memmap exposes its actual header offset; separately read headers for
        # preloaded arrays so truncated/extra payloads cannot enter training.
        mm=arr if isinstance(arr,np.memmap) else np.load(path,mmap_mode='r')
        if Path(path).stat().st_size != mm.offset+mm.nbytes:
            raise ValueError(f'NPY payload length mismatch: {path}')


def describe_contract(cfg, manifest, root, loader, **extra):
    protocol=manifest.get('protocol',{})
    actual=protocol.get('sample_rate')
    declared=cfg.get('sample_rate')
    if actual is not None and declared is not None and float(actual)!=float(declared):
        raise ValueError(f'sample_rate mismatch: stored {actual}Hz but config {declared}Hz; resampling must be explicit')
    if cfg.get('require_rebuild_ready'):
        ready=Path(root)/'READY.json';manifest_path=Path(root)/'manifest.json'
        if not ready.exists():raise FileNotFoundError(f'rebuild incomplete: {ready}')
        if json.loads(ready.read_text())['manifest_sha256']!=hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError('READY manifest hash mismatch')
    return {'data_root':str(Path(root).resolve()),
            'manifest_sha256':hashlib.sha256((Path(root)/'manifest.json').read_bytes()).hexdigest(),
            'sample_rate':actual, 'stored_normalization':protocol.get('normalization'),
            'stored_units':protocol.get('units'), 'loader':loader,
            'train_drop_last':bool(cfg.get('train_drop_last',False)),**extra}


def labram_divisor(cfg, manifest):
    protocol=manifest.get('protocol',{})
    normalization=str(protocol.get('normalization','')).lower()
    units=str(protocol.get('units','')).lower()
    expected = 1.0 if normalization in ('div100','q95','train_stats','train_zscore','zscore') else (100.0 if units in ('uv','microvolts') else None)
    explicit=cfg.get('loader_divisor')
    if expected is None and explicit is None:
        raise ValueError('LaBraM loader requires explicit units or loader_divisor; unknown scale must not default to /100')
    divisor=float(explicit if explicit is not None else expected)
    if divisor<=0:raise ValueError('loader_divisor must be positive')
    if expected is not None and divisor!=expected:
        raise ValueError(f'LaBraM divisor {divisor} conflicts with stored normalization={normalization}, units={units}; expected {expected}')
    return divisor


def resolve_signal_transform(root, split, cfg, manifest):
    """Read small normalization sidecar only when a model needs physical scale."""
    mode=cfg.get('loader_signal_mode','stored')
    if mode=='stored':return None,1.0
    if mode!='uv_div100':raise ValueError(f'unknown loader_signal_mode: {mode}')
    protocol=manifest.get('protocol',{})
    norm=protocol.get('normalization')
    if norm=='div100':return None,1.0
    if str(protocol.get('units','')).lower() in ('uv','microvolts'):return None,0.01
    if norm=='q95':
        p=Path(root)/f'{split}_q95_denominators.npy'
        if not p.exists():raise FileNotFoundError(f'cannot recover physical amplitude without {p}')
        denominators=np.load(p,mmap_mode='r')
        signals=np.load(Path(root)/f'{split}_signals.npy',mmap_mode='r')
        if signals.ndim != 3 or denominators.shape != (signals.shape[0],signals.shape[1],1):
            raise ValueError(f'q95 denominator shape mismatch: {p}: {denominators.shape} for {signals.shape}')
        if p.stat().st_size != denominators.offset+denominators.nbytes:
            raise ValueError(f'q95 denominator payload length mismatch: {p}')
        if denominators.dtype != np.float64 or not np.isfinite(denominators).all() or (denominators<=0).any():
            raise ValueError(f'invalid q95 denominator sidecar: {p}')
        return denominators,0.01
    raise ValueError(f'cannot recover physical scale from normalization {norm}')
