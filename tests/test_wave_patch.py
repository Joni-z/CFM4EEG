"""The waveform control preserves electrodes and reuses the actual encoder."""
from pathlib import Path

import torch
import yaml

from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.triaxial import FreqNone


ROOT = Path(__file__).resolve().parents[1]


def test_wave_control_uses_raw_per_electrode_patches_and_same_encoder():
    full_cfg = yaml.safe_load((ROOT / 'configs/model/tuev-scratch.yaml').read_text())
    wave_cfg = yaml.safe_load((ROOT / 'configs/model/tuev-wave-patch.yaml').read_text())
    permitted = {'name', 'model', 'model_kwargs', 'reproduction'}
    assert {k: v for k, v in full_cfg.items() if k not in permitted} == {
        k: v for k, v in wave_cfg.items() if k not in permitted}
    for cfg in (full_cfg, wave_cfg):
        cfg['model_kwargs'] = {**cfg['model_kwargs'], 'depth': 1}
    full, wave = [build_model(cfg, (16, 1000)) for cfg in (full_cfg, wave_cfg)]
    assert {k: v.shape for k, v in full.encoder.state_dict().items()} == {
        k: v.shape for k, v in wave.encoder.state_dict().items()}
    assert all(isinstance(block.freq, FreqNone) for block in wave.encoder.blocks)
    wave.eval()
    x = torch.randn(2, 16, 1000)
    shapes = []
    hook = wave.encoder.register_forward_pre_hook(lambda _, args: shapes.append(args[0].shape))
    logits = wave(x)
    hook.remove()
    assert shapes == [torch.Size([2, 16, 1, 20, 128])]
    assert logits.shape == (2, 6) and torch.isfinite(logits).all()
    logits.square().sum().backward()
    assert wave.frontend.weight.grad is not None
    assert torch.isfinite(wave.frontend.weight.grad).all()
    # Linear projection shared across electrodes; neither mixes nor filters them.
    patches = x.unfold(-1, 50, 50)
    torch.testing.assert_close(wave.frontend(patches)[:, 4, 3],
                               wave.frontend(x[:, 4, 150:200]))
