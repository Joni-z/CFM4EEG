"""Explicit optimizer-update schedules and provenance for audited runs."""
from __future__ import annotations

import hashlib
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch


def update_count(n_batches: int, accumulation: int, drop_incomplete: bool = False) -> int:
    if n_batches < 1 or accumulation < 1:
        raise ValueError("Nonempty training loader and positive accumulation required")
    count = n_batches // accumulation if drop_incomplete else math.ceil(n_batches / accumulation)
    if count < 1:
        raise ValueError("Training set has no complete effective batch")
    return count


class ArraySchedule:
    """Apply the upstream LaBraM array before each optimizer update.

    Upstream uses np.linspace including both warmup endpoints, followed by a
    cosine whose denominator is the number of remaining updates. Each group's
    depth-dependent multiplier applies to both peak and minimum LR.
    """
    def __init__(self, optimizer, cfg, updates_per_epoch):
        total = int(cfg["epochs"]) * updates_per_epoch
        warm = int(cfg.get("warmup_epochs", 0)) * updates_per_epoch
        if not 0 <= warm < total:
            raise ValueError("warmup must end before the training budget")
        peak, floor = float(cfg["lr"]), float(cfg.get("scheduler_min_lr", 1e-6))
        start = float(cfg.get("warmup_start_lr", 0.0))
        tail = np.arange(total - warm)
        self.values = np.concatenate((
            np.linspace(start, peak, warm),
            floor + .5 * (peak - floor) * (1 + np.cos(np.pi * tail / len(tail))),
        ))
        self.optimizer = optimizer
        self.scales = [g["lr"] / peak for g in optimizer.param_groups]
        self.index = 0
        self._apply()

    def _apply(self):
        for g, scale in zip(self.optimizer.param_groups, self.scales):
            g["lr"] = float(self.values[min(self.index, len(self.values)-1)]) * scale

    def step(self):
        self.index += 1
        self._apply()

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state):
        self.index = state["index"]
        self._apply()


def make_schedule(optimizer, cfg, n_batches):
    updates = update_count(n_batches, int(cfg.get("accum_steps", 1)),
                           bool(cfg.get("drop_incomplete_accumulation", False)))
    name = cfg.get("scheduler")
    total = int(cfg.get("epochs", 20)) * updates
    if name in (None, "none"):
        return None
    if name == "labram_cosine":
        return ArraySchedule(optimizer, cfg, updates)
    if name in ("cosine", "cosine_updates"):
        warm = int(cfg.get("warmup_epochs", 0)) * updates
        floor = float(cfg.get("scheduler_min_lr", 0.0))
        if warm:
            if warm >= total:
                raise ValueError("warmup must end before the training budget")
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3,
                                                    total_iters=warm),
                 torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=total-warm, eta_min=floor)],
                milestones=[warm])
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total, eta_min=floor)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.get("max_lr", cfg["lr"]),
            steps_per_epoch=updates, epochs=cfg["epochs"],
            pct_start=cfg.get("pct_start", .2))
    raise ValueError(f"Unimplemented scheduler: {name}")


def normalize_gradients(parameters, n_examples):
    """Normalize accumulated sum losses, including a short final batch."""
    if n_examples <= 0:
        raise ValueError("No examples in accumulation group")
    for p in parameters:
        if p.grad is not None:
            p.grad.div_(n_examples)


def provenance(cfg_path, cfg, info):
    root = Path(__file__).resolve().parents[2]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*.py"))
              if not p.name.startswith('._') and p.relative_to(root).parts[0] != 'vendor'}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
            text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    runtime = root / 'scripts' / 'rebuild'
    runtime_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(runtime.glob('*.py'))}
    return {"source_commit": commit, "source_sha256": hashes,
            "runtime_source_sha256": runtime_hashes,
            "vendor_source_sha256": vendor_source_hashes(cfg),
            "pretrained_payloads": pretrained_payloads(cfg),
            "config_sha256": hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest(),
            "python": platform.python_version(), "torch": torch.__version__,
            "data_root": str(Path(__import__('os').path.expandvars(cfg['data_root'])).resolve()),
            "data_manifest": info.get("manifest"),
            "input_contract": info.get("input_contract"),
            "torch_cuda": torch.version.cuda}


def vendor_source_hashes(cfg):
    """Bind the upstream implementation even when vendor/ is a shared symlink.

    Only the model used by this configuration is inspected. Model payloads are
    handled separately, so this never walks other models' checkpoints or data.
    """
    from paclock_bench.paths import vendored
    model = cfg.get('model')
    name = {'ffcl': 'biot'}.get(model, model)
    if name not in {'biot', 'cbramod', 'labram', 'eegpt', 'reve', 'tfm'}:
        return {}
    directory = Path(vendored(name)).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing upstream source: {directory}')
    result = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if (not path.is_file() or path.suffix not in {'.py', '.yaml', '.yml', '.toml'}
                or any(part.startswith('.') or part == '__pycache__' for part in relative.parts)):
            continue
        result[f'vendor/{name}/{relative}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise RuntimeError(f'No source files found for used vendor: {directory}')
    if name == 'reve':
        # Electrode coordinates also condition scratch REVE's forward pass.
        positions = directory.parent / 'reve_positions' / 'positions.json'
        result['vendor/reve_positions/positions.json'] = hashlib.sha256(positions.read_bytes()).hexdigest()
    return result


def pretrained_payloads(cfg):
    """Bind loaded initialization payloads, not only their filenames.

    Target-only TFM artifacts are stage dependencies and are recorded when they
    exist; their absence is handled by the stage-specific completion gate.
    """
    import os
    from paclock_bench.paths import vendored
    model = cfg.get('model')
    paths = []
    if model == 'biot' and cfg.get('checkpoint'):
        names = {'shhs_prest18': 'EEG-SHHS+PREST-18-channels.ckpt',
                 'prest16': 'EEG-PREST-16-channels.ckpt',
                 'six_datasets18': 'EEG-six-datasets-18-channels.ckpt'}
        paths = [Path(vendored('biot'))/'pretrained-models'/names[cfg['checkpoint']]]
    elif cfg.get('pretrained'):
        if model == 'cbramod':
            paths = [Path(cfg.get('pretrained_path') or Path(vendored('cbramod'))/'pretrained_weights/pretrained_weights.pth')]
        elif model == 'labram':
            paths = [Path(vendored('labram'))/'checkpoints/labram-base.pth']
        elif model == 'eegpt':
            paths = [Path(vendored('eegpt'))/'checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt']
        elif model == 'reve':
            directory = Path(vendored('reve')).parent/'reve_weights'
            paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in ('.safetensors','.bin','.json','.py'))
        elif model == 'tfm':
            paths = [Path(os.path.expandvars(cfg[k])) for k in ('tokenizer_path','encoder_path') if cfg.get(k)]
            if not paths and cfg.get('tfm_setting') == 'multiple':
                directory = Path(vendored('tfm'))/'pretrained_weigths/multiple_dataset_settings'
                paths = sorted(directory.rglob('*.pth'))
            if cfg.get('tfm_setting') == 'single':
                paths = [p for p in paths if p.exists()]
    result = {}
    for path in paths:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):
                digest.update(block)
        result[str(path.resolve())]={'sha256':digest.hexdigest(),'size':path.stat().st_size}
    return result
